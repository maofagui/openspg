# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import tarfile
import time
import threading
from queue import Queue
from alignment.rlhf.hooks.rlhf_train_hook import RLHFTrainHook
from alignment.app.util import logger
import glob
import torch

IN_PROFILING = False
TRACE_PROFILER = None

TMP_PROFILER_DIR = './tmp_rlhf_profiler'
GFS_PROFILER_DIR = 'gfs://gfs1_analyze_sata_em14_online/app/rlhf/{userid}/{jobid}/{experience_step}'
GFS_PROFILER_ROOT = 'gfs://gfs1_analyze_sata_em14_online/app/rlhf/{userid}/{jobid}'


def to_microseconds(s):
    return 1000000 * float(s)

class TraceWriter(threading.Thread):
    def __init__(self, terminator, input_queue, output_stream):
        threading.Thread.__init__(self)
        self.daemon = True
        self.terminator = terminator
        self.input = input_queue
        self.output = output_stream

    def run(self):
        while not (self.terminator.is_set() and self.input.empty()):
            try:
                item = self.input.get(timeout=1)
                self.output.write((json.dumps(item) + ',\n').encode('ascii'))
            except:
                time.sleep(1)


class TraceProfiler(object):
    """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).

     Usage:

        from pytracing import TraceProfiler
        tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
        with tp.traced():
          ...

  """
    TYPES = {'call': 'B', 'return': 'E'}

    def __init__(self, clock=None):
        self.clock = clock or time.time
        self.pid = os.getpid()
        self.queue = Queue()
        self.terminator = threading.Event()

    def install(self, output):
        """Install the trace function and open the JSON output stream."""
        self.writer = TraceWriter(self.terminator, self.queue, output)
        self.writer.start()    # Start the writer thread.

    def shutdown(self):
        self.terminator.set()    # Stop the writer thread.
        self.writer.join()    # Join the writer thread.

    def _add_event_to_queue(self, event_name, event_type):
        ts = time.time() * 1e6
        if isinstance(event_name, str):
            event_name = [event_name]
        for item in event_name:
            event = dict(
                name=item,    # Event Name.
                ph=self.TYPES[event_type],    # Event Type.
                pid=self.pid,    # Process ID.
                ts=ts,    # Timestamp.
            )
            self.queue.put(event)


class ProfileTrainHook(RLHFTrainHook):
    def __init__(self, profile_config, profile_dir=None):
        self._profile_config = profile_config
        self._profile_dir = profile_dir or '/mnt/nas/faster/profile_test_dirs'
        self._skip_steps = profile_config.skip_first_steps
        self._save_steps = profile_config.save_steps
        self._max_profile_steps = self._skip_steps + self._save_steps * profile_config.max_profile_cnt
        self._enable_flops_profile = profile_config.enable_flops_profile
        self._enable_torch_profile = profile_config.enable_torch_profile
        self._last_torch_profile_step = -1
        self._last_train_profile_step = -1
        self._torch_profiler = None

    def on_train_start(self):
        from app.core.global_vars import global_context
        self._global_ctx = global_context()

    def on_experience_learn_start(self, experience_step: int):
        """
        一次experience学习的开始
        一次experience = 一次experience make +  一次experience train
        """
        global IN_PROFILING, TRACE_PROFILER
        if self._skip_steps <= experience_step <= self._max_profile_steps and (
                experience_step - self._skip_steps) % self._save_steps == 0:
            assert not IN_PROFILING
            IN_PROFILING = True

            prefix_dir = self._get_local_profile_prefix_dir(experience_step)
            if not os.path.exists(prefix_dir):
                os.system(f'mkdir -p {prefix_dir}')
            self._fd = open(f'{prefix_dir}/trace_{os.getpid()}', 'wb+')

            TRACE_PROFILER = TraceProfiler()
            TRACE_PROFILER.install(output=self._fd)
            TRACE_PROFILER._add_event_to_queue('learn_experience', 'call')

            # if self._enable_flops_profile:
            #     from app.pytorch.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
            #     [hook.on_profile_begin() for hook in CallModelHooks.MODEL_HOOKS.values()]

    def _get_local_profile_prefix_dir(self, experience_step):
        return f'{self._profile_dir}/{experience_step}'

    def on_experience_learn_end(self, experience_step: int):
        """
           一次experience学习的开始结束
        """
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            TRACE_PROFILER._add_event_to_queue('learn_experience', 'return')
            TRACE_PROFILER.shutdown()

            IN_PROFILING = False
            self._cur_profile_train_batches = 0
            self._cur_profile_exp_batches = 0
            self._fd.close()

            ctx = self._global_ctx

            # /mnt/nas/faster/profile_test_dirs/{experience_step}
            local_profiler_dir = self._get_local_profile_prefix_dir(experience_step)
            # gfs://gfs1_analyze_sata_em14_online/app/rlhf/{userid}/{jobid}/{experience_step}
            gfs_profiler_root = GFS_PROFILER_ROOT.format(userid=ctx.user, jobid=ctx.job_id)
            gfs_profiler_dir = GFS_PROFILER_DIR.format(userid=ctx.user, jobid=ctx.job_id, experience_step=experience_step)

            from app.framework.exporter.location import FileLocation
            local_file_location = FileLocation(local_profiler_dir)
            # local dir 2 gfs dir
            local_file_location.upload_to_gfs(gfs_profiler_root)

            torch.distributed.barrier()
            # 只在rank0上做trace merge的工作
            if torch.distributed.get_rank() == 0:
                cont = b'['
                for item in glob.glob(f'{local_profiler_dir}/trace_*'):
                    with open(item, 'rb') as fd:
                        cont += fd.read()
                cont += b'{}]'
                merged_file_name = f'merged_trace_rank{torch.distributed.get_rank()}'
                merged_file_local_path = f'{local_profiler_dir}/{merged_file_name}'
                with open(merged_file_local_path, 'wb') as fd:
                    fd.write(cont)

                #上传单个文件
                FileLocation(merged_file_local_path).upload_to_gfs(f'{gfs_profiler_dir}/{merged_file_name}')

                self._upload_to_oss_sync_path_modelhub(gfs_profiler_dir)


            # if self._enable_flops_profile:
            #     from app.pytorch.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
            #     for model_name, hook in CallModelHooks.MODEL_HOOKS.items():
            #         hook.on_profile_end(f'{prefix_dir}/flops_profile_{model_name}_rank_{torch.distributed.get_rank()}')

    def on_experience_make_start(self, experience_step: int):
        """
           一次experience产生开始
        """
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            TRACE_PROFILER._add_event_to_queue('make_experience', 'call')

    def on_experience_make_end(self, experience_step: int):
        """
           一次experience产生结束
        """
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            TRACE_PROFILER._add_event_to_queue('make_experience', 'return')

    def on_experience_train_start(self, experience_step: int):
        """
           一次experience的训练开始
        """
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            TRACE_PROFILER._add_event_to_queue('train_experience', 'call')

    def on_experience_train_end(self, experience_step: int, **kwargs):
        """
           一次experience的训练结束
        """
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            TRACE_PROFILER._add_event_to_queue('train_experience', 'return')

    def on_experience_batch_start(self, experience_step: int):
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            if self._enable_torch_profile and self._last_torch_profile_step != experience_step:
                self._torch_profiler = torch.profiler.profile()
                self._torch_profiler.__enter__()
            if self._enable_flops_profile:
                from app.pytorch.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
                [hook.on_profile_begin() for hook in CallModelHooks.MODEL_HOOKS.values()]

    def on_experience_batch_end(self, experience_step: int):
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            if self._enable_torch_profile and self._last_torch_profile_step != experience_step:
                assert self._torch_profiler is not None
                self._torch_profiler.__exit__(None, None, None)
                local_profiler_dir = self._get_local_profile_prefix_dir(experience_step)
                self._torch_profiler.export_chrome_trace(
                    f'{local_profiler_dir}/torch_generate_profile_rank_{torch.distributed.get_rank()}.json'
                )
                if self._torch_profiler is not None:
                    del self._torch_profiler
                self._last_torch_profile_step = experience_step
            if self._enable_flops_profile:
                prefix_dir = self._get_local_profile_prefix_dir(experience_step)
                from app.pytorch.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
                for model_name, hook in CallModelHooks.MODEL_HOOKS.items():
                    hook.on_profile_end(
                        f'{prefix_dir}/make_exp_flops_profile_{model_name}_rank_{torch.distributed.get_rank()}')

    def on_experience_train_batch_start(self, experience_step: int, global_step: int):
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            if self._enable_torch_profile and self._last_train_profile_step != experience_step:
                self._torch_profiler = torch.profiler.profile()
                self._torch_profiler.__enter__()
            if self._enable_flops_profile:
                from alignment.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
                [hook.on_profile_begin() for hook in CallModelHooks.MODEL_HOOKS.values()]

    def on_experience_train_batch_end(self, experience_step: int, global_step: int, metrics: dict):
        global IN_PROFILING, TRACE_PROFILER
        if IN_PROFILING:
            prefix_dir = self._get_local_profile_prefix_dir(experience_step)
            if self._enable_torch_profile and self._last_train_profile_step != experience_step:
                assert self._torch_profiler is not None
                self._torch_profiler.__exit__(None, None, None)
                self._torch_profiler.export_chrome_trace(
                    f'{prefix_dir}/torch_train_profile_rank_{torch.distributed.get_rank()}.json'
                )
                if self._torch_profiler is not None:
                    del self._torch_profiler
                self._last_train_profile_step = experience_step
            if self._enable_flops_profile:
                from app.pytorch.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
                for model_name, hook in CallModelHooks.MODEL_HOOKS.items():
                    hook.on_profile_end(
                        f'{prefix_dir}/train_flops_profile_{model_name}_rank_{torch.distributed.get_rank()}')

    def _upload_to_oss_sync_path_modelhub(self, source_dir: str):
        """
        Download files from gfs dir, package and update to oss
        upload local profiler log to oss, and sync oss path to modelhub
        """
        ctx = self._global_ctx

        # only report profiler_path on chief node
        if not ctx.is_chief or ctx.is_local:
            return

        from app.common.utils.key import app_OSS_HOME
        from app.framework.exporter.location import OssLocation
        from app.common.utils import string_utils

        report_dict = {}
        has_path_to_report = False

        if source_dir is not None:
            from app.common.utils.path_utils import Path
            experience_step = Path.get_base_path(source_dir)

            from app.pytorch.api.utils.save_load_util import write_gfs_dir_to_local
            local_temp_root = f'{TMP_PROFILER_DIR}/{ctx.job_id}'
            local_temp_dir = f'{local_temp_root}/{experience_step}'
            write_gfs_dir_to_local(source_dir, local_temp_root)

            zipped_file_name = f'rlhf_profiler_step_{experience_step}'
            zipped_file_full_name = f'{zipped_file_name}.tar.gz'

            def make_tarfile(output_filename, source):
                with tarfile.open(output_filename, "w:gz") as tar:
                    tar.add(source, arcname=os.path.basename(source_dir))

            zipped_file_local_full_path = f'{TMP_PROFILER_DIR}/{ctx.job_id}/{experience_step}/{zipped_file_full_name}'
            make_tarfile(zipped_file_local_full_path, local_temp_dir)

            oss_file_path = f'{app_OSS_HOME}/profiler/rlhf/{ctx.app_id}/{ctx.job_id}/{zipped_file_full_name}'
            oss_location = OssLocation.from_url(oss_file_path)
            oss_location.copy_from(zipped_file_local_full_path)
            visible_url = oss_location.get_visible_url()

            # 上传成功后删除本地下载的文件（包括打包的文件）
            import shutil
            shutil.rmtree(local_temp_dir)

            if string_utils.is_not_blank(visible_url):
                report_dict["rlhf_profiler"] = [visible_url]
                has_path_to_report = True

        job_id = self._global_ctx.job_id
        if job_id and has_path_to_report:
            try:
                from app.modelhub.api_client import modelhub
                modelhub.update_profiler_path(job_id, report_dict)
                logger.info('-----------------Upload {} to ModelHUB SUCCEED----------------'.format(report_dict))
            except Exception:
                logger.exception("Failed to update job info {}".format(report_dict))


class TraceEventScope(object):
    def __init__(self, event_name):
        self._event_name = event_name

    def __enter__(self):
        global IN_PROFILING, TRACE_PROFILER
        if not IN_PROFILING or not TRACE_PROFILER:
            return
        TRACE_PROFILER._add_event_to_queue(self._event_name, 'call')

    def __exit__(self, exc_type, exc_value, traceback):
        global IN_PROFILING, TRACE_PROFILER
        if not IN_PROFILING or not TRACE_PROFILER:
            return
        TRACE_PROFILER._add_event_to_queue(self._event_name, 'return')

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            global IN_PROFILING, TRACE_PROFILER
            if not IN_PROFILING or not TRACE_PROFILER:
                return func(*args, **kwargs)
            with self:
                return func(*args, **kwargs)

        return wrapped_func