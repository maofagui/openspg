# coding: utf-8

from collections import deque
import functools
import os
import re
from functools import partial
from itertools import cycle

import torch
import torch.nn as nn

from alignment.util.global_vars import global_context
from alignment.rlhf.distributed.model_placement import ModelPlacement
from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedACEngine
from alignment.api.utils import dist_util
from alignment.rlhf.hooks.profile_train_hook import TraceEventScope
from alignment.app.util import logger
from collections import defaultdict

import importlib
import sys
import time

EMPTY_TENSOR_LENGTH = int(os.getenv('EMPTY_TENSOR_LENGTH', 1e5))

TMP_PARAM_DIR = os.environ.get('TMP_PARAM_DIR', '/mnt/nas/faster/llama_new/llama_example_run_single/tmp_param_shape')

PARALLEL_STATE_DICT = defaultdict(lambda: defaultdict())

ALL_MODEL_RANKS = defaultdict(lambda: list)
ALL_MODEL_WORLD = defaultdict(lambda: list)
CUR_RANKS = list()
CUR_WORLD_GROUP = list()

layer_re = re.compile(r'layers\.([0-9]+)')

SCATTER_QUEUE = deque()

__DIST_DATA__ = {}

current_model = None
from collections import OrderedDict
SEND_RECV_DATA_GROUPS = OrderedDict()


def _is_lora_weight(name):
    return 'lora_left_weight' in name or 'lora_right_weight' in name


class DataHelper():
    def __init__(self, input_dtype=torch.int64, data_dtype=torch.float16):
        rlhf_sep_config = global_context().runtime_conf.rlhf_sep_config
        self._batch_size = rlhf_sep_config.rollout_batch_size
        self._seq_length = rlhf_sep_config.seq_length
        self._input_length = rlhf_sep_config.input_length
        self._data_shape = None
        self._input_dtype = input_dtype
        self._data_dtype = data_dtype

    @property
    def data_shape(self):
        if self._data_shape is None:
            data_shape = OrderedDict()
            # 'all_tokens', 'no_padded_query_ids',  'logprobs', 'loss_mask'
            data_shape['all_tokens'] = (self._batch_size * self._seq_length, self._input_dtype,
                                        [self._batch_size, self._seq_length])
            data_shape['all_token_ids_right_padded'] = (self._batch_size * self._seq_length, self._input_dtype,
                                                        [self._batch_size, self._seq_length])
            data_shape['prompt_sizes'] = (self._batch_size, self._input_dtype, [self._batch_size])
            data_shape['logprobs'] = (self._batch_size * (self._seq_length - 1), self._data_dtype,
                                      [self._batch_size, self._seq_length - 1])
            # 修改为0/1？
            data_shape['loss_mask'] = (self._batch_size * self._seq_length, self._input_dtype,
                                       [self._batch_size, self._seq_length])

            data_shape['ref_logprobs'] = (self._batch_size * (self._seq_length - 1), self._data_dtype,
                                          [self._batch_size, self._seq_length - 1])

            data_shape['old_values'] = (self._batch_size * self._seq_length, self._data_dtype,
                                        [self._batch_size, self._seq_length])

            # 这里通过pad，将reward几个元组pad到定长，
            # start表示该元素起始index, end可以通过loss_mask拼出来，取start_end表示有效范围
            data_shape['action_start_indices'] = (self._batch_size, self._input_dtype, [self._batch_size])
            data_shape['action_end_indices'] = (self._batch_size, self._input_dtype, [self._batch_size])

            data_shape['action_logprobs'] = (self._batch_size * (self._seq_length - 1), self._data_dtype,
                                             [self._batch_size, self._seq_length - 1])
            data_shape['action_values'] = (self._batch_size * (self._seq_length - 1), self._data_dtype,
                                           [self._batch_size, self._seq_length - 1])
            data_shape['action_rewards'] = (self._batch_size * (self._seq_length - 1), self._data_dtype,
                                            [self._batch_size, self._seq_length - 1])
            self._data_shape = data_shape
        return self._data_shape

    def _gen_send_op(self,
                     ret_data,
                     ret_data_tensor,
                     ret_input_data_tensor,
                     dst_rank,
                     tag_prefix,
                     keys=None,
                     run_async=False,
                     dst_group_idx=0,
                     dst_groups=1):
        cur_rank = torch.distributed.get_rank()
        # 编写。。
        if keys is None:
            keys = ret_data.keys()
        try:
            send_keys = sorted(keys, key=lambda item: list(self.data_shape.keys()).index(item))
        except:
            print(self.data_shape)
            if cur_rank == 0:
                import pdb
                pdb.set_trace()
            import time
            time.sleep(10000)

        tmp_data_tensor, tmp_input_data_tensor = [], []
        tmp_data_length, tmp_input_data_length = 0, 0

        for item in send_keys:
            cur_length, cur_data_type, cur_shape = self.data_shape[item]
            bs_len = int(cur_shape[0] // dst_groups) if isinstance(cur_shape,
                                                                   (list, tuple)) else int(cur_shape // dst_groups)
            dat = ret_data[item][bs_len * dst_group_idx:bs_len * (dst_group_idx + 1)]
            dat = dat.view(-1) if dat.is_contiguous() else dat.reshape(-1)
            # here
            if cur_data_type == self._data_dtype:
                try:
                    tmp_data_tensor.append(dat)
                except Exception as e:
                    print(f'rank {cur_rank} {e}')
                    if cur_rank == 0:
                        import pdb
                        pdb.set_trace()
                    import time
                    time.sleep(10000)

                tmp_data_length += cur_length
            else:
                assert cur_data_type == self._input_dtype
                try:
                    tmp_input_data_tensor.append(dat)
                except Exception as e:
                    print(f'rank {cur_rank} {e}')
                    if cur_rank == 0:
                        import pdb
                        pdb.set_trace()
                    import time
                    time.sleep(10000)
                tmp_input_data_length += cur_length
        logger.info(f'Begin send data to rank {dst_rank}, tmp_data_length: {tmp_data_length}, bs_len: {bs_len}'
                    f', tmp_input_data_length: {tmp_input_data_length}, {send_keys}, tag_prefix: {tag_prefix}')

        # 后续也可改成, P2Pop，有_start_coalescing
        send_func = torch.distributed.isend if run_async else torch.distributed.send

        ret = []

        if tmp_data_length > 0:
            tag = tag_prefix * 1000
            # concat_data_tensor = ret_data_tensor.narrow(0, 0, tmp_data_length)
            concat_data_tensor = torch.cat([item for item in tmp_data_tensor]).to(torch.bfloat16).cuda()
            ret.append(send_func(concat_data_tensor, dst_rank, group=SEND_RECV_DATA_GROUPS[f'{cur_rank}_{dst_rank}']))

        if tmp_input_data_length > 0:
            tag = tag_prefix * 1000 + 1
            # concat_input_data_tensor = ret_input_data_tensor.narrow(0, 0, tmp_input_data_length)
            try:
                concat_input_data_tensor = torch.cat([item.cuda() for item in tmp_input_data_tensor]).to(torch.int64)
            except:
                if cur_rank == 5:
                    import pdb
                    pdb.set_trace()
                import time
                time.sleep(100000)
            ret.append(
                send_func(concat_input_data_tensor, dst_rank, group=SEND_RECV_DATA_GROUPS[f'{cur_rank}_{dst_rank}']))
        return ret

    def _gen_recv_op(self, keys, ret_data_tensor, ret_input_data_tensor, src_rank, tag_prefix, groups=1):
        # 编写
        # TODO: recv改造下
        cur_rank = torch.distributed.get_rank()
        recv_keys = sorted(keys, key=lambda item: list(self.data_shape.keys()).index(item))

        tmp_data_length, tmp_input_data_length = [(0, None)], [(0, None)]
        for item in recv_keys:
            cur_length, cur_data_type, _ = self.data_shape[item]
            cur_length = int(cur_length // groups)
            if cur_data_type == self._data_dtype:
                tmp_data_length.append((tmp_data_length[-1][0] + cur_length, item))
            else:
                assert cur_data_type == self._input_dtype
                tmp_input_data_length.append((tmp_input_data_length[-1][0] + cur_length, item))
        logger.info(f'Begin recv data from rank {src_rank}, tmp_data_length: {tmp_data_length[-1][0]}'
                    f', tmp_input_data_length: {tmp_input_data_length[-1][0]}, {recv_keys}, tag_prefix: {tag_prefix}')

        data = {}
        if tmp_data_length[-1][0] > 0:
            tag = tag_prefix * 1000
            tmp_data_tensor = ret_data_tensor.narrow(0, 0, tmp_data_length[-1][0]).to(torch.bfloat16)
            torch.distributed.recv(tmp_data_tensor, src_rank, group=SEND_RECV_DATA_GROUPS[f'{src_rank}_{cur_rank}'])
            prev = 0
            for length, key in tmp_data_length[1:]:
                try:
                    data_shape = self.data_shape[key][2].copy()
                    data_shape[0] = int(data_shape[0] // groups)
                    data[key] = ret_data_tensor.narrow(0, prev, length - prev).view(data_shape)
                except:
                    if cur_rank == 7:
                        import pdb
                        pdb.set_trace()
                    import time
                    time.sleep(10000)
                prev = length

        if tmp_input_data_length[-1][0] > 0:
            try:
                tag = tag_prefix * 1000 + 1
                tmp_input_data_tensor = ret_input_data_tensor.narrow(0, 0,
                                                                     tmp_input_data_length[-1][0]).to(torch.int64)
                torch.distributed.recv(tmp_input_data_tensor,
                                       src_rank,
                                       group=SEND_RECV_DATA_GROUPS[f'{src_rank}_{cur_rank}'])
                prev = 0
                for length, key in tmp_input_data_length[1:]:
                    data_shape = self.data_shape[key][2].copy()
                    data_shape[0] = int(data_shape[0] // groups)
                    data[key] = ret_input_data_tensor.narrow(0, prev, length - prev).view(data_shape)
                    prev = length
            except Exception as e:
                print(f'rank: {cur_rank} {tmp_input_data_length}, {e}')
                if cur_rank == 5:
                    import pdb
                    pdb.set_trace()
                import time
                time.sleep(10000)

        return data


class DistModel():
    """各模型步调一致，每运行完一次fowrad后, cur_step+1; 
    暂时只对pred_actor支持多replica，提升generate速度; 
    对于训练模型来说，使用的是数据并行
    """

    def __init__(self, replicas):
        self.replicas = replicas
        self._num_replicas = len(replicas)
        self.register_func()
        self._cur_step = 0
        self._recv_src_models = None
        self._send_dst_models = None
        self._id = 0

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        for idx, item in enumerate(self.replicas):
            item.id = id + idx

    @property
    def recv_src_models(self):
        return self._recv_src_models

    @property
    def send_dst_models(self):
        return self._send_dst_models

    @recv_src_models.setter
    def recv_src_models(self, recv_src_models):
        self._recv_src_models = recv_src_models

    @send_dst_models.setter
    def send_dst_models(self, send_dst_models):
        self._send_dst_models = send_dst_models
        for item in self.replicas:
            item.send_dst_models = send_dst_models

    @property
    def num_replicas(self):
        return self._num_replicas

    def register_func(self):
        for func_name in ["forward_step", "train_step"]:
            dist_call = partial(self.call_replica_func, func_name)
            setattr(self, func_name, dist_call)

    def call_replica_func(self, func, *args, **kwargs):
        replica_id = self._cur_step % self.num_replicas

        recv_src_models = []
        if self.recv_src_models:
            for src_model, keys in self.recv_src_models:
                # src_model is Dist_model
                if isinstance(src_model, DistModel):
                    cur_replica_id = self._cur_step % src_model.num_replicas
                    recv_src_models.append((src_model.models[cur_replica_id], keys))
                else:
                    recv_src_models.append((src_model, keys))

            self.replicas[replica_id].recv_src_models = recv_src_models

        send_dst_models = []
        if self.send_dst_models:
            for dst_model in self.send_dst_models:
                # dst_model is Dist_model
                if isinstance(dst_model, DistModel):
                    cur_replica_id = self._cur_step % dst_model.num_replicas
                    send_dst_models.append(dst_model.models[cur_replica_id])
                else:
                    send_dst_models.append(dst_model)
            self.replicas[replica_id].send_dst_models = send_dst_models

        self._cur_step += 1

        ret = getattr(self.replicas[replica_id], func)(*args, **kwargs)
        return ret

    def sync(self, *args, **kwargs):
        for model in self.replicas:
            model.sync(*args, **kwargs)


def build_send_recv_data_group(origin_src_model):
    if isinstance(origin_src_model, DistModel):
        src_models = origin_src_model.replicas
    else:
        src_models = [origin_src_model]

    for src_model in src_models:
        src_model_parallel_ranks = src_model._place_policy.interleave_model_parallel_ranks[0]
        for src_rank in src_model_parallel_ranks:
            my_rank_idx = src_model_parallel_ranks.index(src_rank)
            for dst_model in src_model._send_dst_models:
                for _, dst_model_parallel_ranks in enumerate(dst_model._place_policy.interleave_model_parallel_ranks):

                    for dst_idx, dst_rank in enumerate(dst_model_parallel_ranks):
                        if dst_idx % len(src_model_parallel_ranks) == my_rank_idx:
                            if dst_rank != src_rank:
                                SEND_RECV_DATA_GROUPS[f"{src_rank}_{dst_rank}"] = torch.distributed.new_group(
                                    ranks=[src_rank, dst_rank])


def haha(src, dst, pipe_stage, tensor_rank, bucket_id, param_id):
    try:
        return int(1e8 * src + 1e6 * dst + 1e4 * pipe_stage + 1e2 * tensor_rank + 10 * bucket_id +  param_id)
    except Exception as e:
        if torch.distributed.get_rank() == 1:
            import pdb
            pdb.set_trace()
        import time
        time.sleep(10000)


def add_debug_trace(rank=0):
    if torch.distributed.get_rank() == rank:
        import pdb
        pdb.set_trace()
    import time
    time.sleep(100000)


MODULE_NAME = 'megatron.core.parallel_state'
ARGS_MODULE_NAME = 'megatron.global_vars'


def update_layer_num(layers_per_part, rank, m):
    # This assumes no interleaved pipeline execution
    layer = int(m.group(1))
    layer += rank * layers_per_part
    return f'layers.{layer}'


def build_pipeline_layer_name_mapping(layers_per_stage, rank, model, requires_grad=False):
    # rank in get_pipeline_model_parallel_rank
    name_mapping = {}
    for src_name, partition_param in model.named_parameters():
        # if requires_grad:
        #     if not partition_param.requires_grad:
        #         # 跳过不需要update的参数同步，避免不必要的传输量
        #         continue

        if src_name.endswith("word_embeddings.weight") and "language_model" not in src_name:
            # See comment in MegatronModule.initialize_word_embeddings()
            tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
        else:
            # Translate destination layer number (0-N for each partition)
            # to source layer number (single-model layer number)
            _update_layer_num = functools.partial(update_layer_num, layers_per_stage, rank)
            tgt_name = re.sub(layer_re, _update_layer_num, src_name)
        name_mapping[tgt_name] = partition_param
        # pipeline参数需要+ layernum
    return name_mapping


def update_param_name(src_name, layers_per_stage, pipe_parallel_rank):
    if src_name.endswith("word_embeddings.weight") and "language_model" not in src_name:
        # See comment in MegatronModule.initialize_word_embeddings()
        tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
    else:
        # Translate destination layer number (0-N for each partition)
        # to source layer number (single-model layer number)
        _update_layer_num = functools.partial(update_layer_num, layers_per_stage, pipe_parallel_rank)
        tgt_name = re.sub(layer_re, _update_layer_num, src_name)
    return tgt_name


def get_recv_params(layers_per_stage,
                    total_layer_stage,
                    pipe_rank,
                    model,
                    src_param_shape,
                    tensor_rank,
                    requires_grad=False):
    # rank in get_pipeline_model_parallel_rank
    ret_params = {}
    src_param_shape = {k.replace('module.', ''):v for k, v in src_param_shape.items()}

    for src_name, partition_param in model.named_parameters():
        # if requires_grad:
        #     if not partition_param.requires_grad:
        #         # 跳过不需要update的参数同步，避免不必要的传输量
        #         continue
        src_name = src_name.replace('module.', '')
        if src_name not in src_param_shape:
            continue

        if 'layers.' in src_name:
            layer_num = int(re.findall(layer_re, src_name)[0])
            if pipe_rank * layers_per_stage <= layer_num < (pipe_rank + 1) * layers_per_stage:
                ret_params[src_name] = partition_param
        elif 'word_embeddings.' in src_name:
            if pipe_rank == 0:
                ret_params[src_name] = partition_param
        else:
            if pipe_rank == total_layer_stage - 1:
                ret_params[src_name] = partition_param

        if src_name in ret_params:
            if partition_param.shape != src_param_shape[src_name]:
                if partition_param.shape[0] != src_param_shape[src_name][0]:
                    ret_params[src_name] = partition_param.narrow(0, tensor_rank * src_param_shape[src_name][0],
                                                                  src_param_shape[src_name][0])
                else:
                    ret_params[src_name] = partition_param.narrow(1, tensor_rank * src_param_shape[src_name][1],
                                                                  src_param_shape[src_name][1])                                                                  
    
    return [ret_params[item] for item in sorted(ret_params.keys())]


def get_send_params(layers_per_stage, total_layer_stage, rank, model, requires_grad=False):
    # rank in get_pipeline_model_parallel_rank
    # 在is_lora_weight加下lora的判断
    ret_params = {}

    for src_name, partition_param in model.named_parameters():
        if not partition_param.requires_grad:
            # 跳过不需要update的参数同步，避免不必要的传输量
            continue
        ret_params[update_param_name(src_name, layers_per_stage, rank)] = partition_param

    return [ret_params[item] for item in sorted(ret_params.keys())
            ], {item: ret_params[item].shape
                for item in sorted(ret_params.keys())}


class PatchSEPDistributedEnv(object):
    def __init__(self, current_model=None):
        self._current_model = current_model

    def __enter__(self):
        global PARALLEL_STATE_DICT, CUR_RANKS, ALL_MODEL_RANKS, ALL_MODEL_WORLD, CUR_WORLD_GROUP
        CUR_RANKS = ALL_MODEL_RANKS[self._current_model]
        CUR_WORLD_GROUP = ALL_MODEL_WORLD[self._current_model]

        self._origin_sep_module = {
            name: module.__dict__
            for name, module in sys.modules.items()
            if name.startswith('megatron.global_vars') or name.startswith('megatron.core.parallel_state')
        # name.startswith('megatron') or name.startswith('alignment.rlhf.models')
        }

        cur_modules = PARALLEL_STATE_DICT[self._current_model] or {}
        from megatron import get_args
        mega_args = get_args()

        # logger.info(
        #     f'Debug before set: {self._current_model}, get pipeline_model_parallel_size: {mega_args.pipeline_model_parallel_size} tensor_model_parallel_size: {mega_args.tensor_model_parallel_size}'
        # )
        for name, module_backup in cur_modules.items():
            # Refer to https://stackoverflow.com/questions/35714374/how-to-reload-a-python-module-clearing-variable-info
            # Not thread-safe
            sys.modules[name].__dict__.clear()
            sys.modules[name].__dict__.update(module_backup)

        from megatron import get_args
        mega_args = get_args()
        # logger.info(
        #     f'Debug after set: {self._current_model}, get pipeline_model_parallel_size: {mega_args.pipeline_model_parallel_size} '
        #     f'tensor_model_parallel_size: {mega_args.tensor_model_parallel_size}, CUR_RANKS: {CUR_RANKS}')

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        # self._origin_sep_module = sys.modules[MODULE_NAME]
        # global MODULE_NAME, ARGS_MODULE_NAME
        # if self._origin_sep_module is not None:
        #     sys.modules[MODULE_NAME] = self._origin_sep_module
        # for name, module in self._origin_sep_module.items():
        #     sys.modules[name].__dict__.clear()
        #     sys.modules[name].__dict__.update(module)
    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            # for装饰器，使用instance.current_model patch当前的current_model
            self._current_model = instance._current_model
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func


class SEPGroupPlacePolicy():
    """ 对 (all_rank - dst_model_ranks) 按照data_parallel_size 划分。跑op前allgather一把
    """

    def __init__(self, my_rank, dst_model_ranks, current_model, model_sep_config, dist_groups=None, group_ranks=None):
        self._has_sep_inited = False
        self._current_model = current_model
        # super().__init__(my_rank, dst_model_ranks, dist_groups, group_ranks)
        global ALL_MODEL_RANKS, CUR_RANKS, ALL_MODEL_WORLD, CUR_WORLD_GROUP
        ALL_MODEL_RANKS[current_model] = dst_model_ranks
        ALL_MODEL_WORLD[current_model] = torch.distributed.new_group(ranks=dst_model_ranks)
        CUR_RANKS = dst_model_ranks
        CUR_WORLD_GROUP = ALL_MODEL_WORLD[current_model]
        self._model_sep_config = model_sep_config

        self._init_sep_module()
        self._data_group_ranks = None
        self._dist_data_groups = None
        self._scatter_idxs = None
        self._interleave_model_parallel_ranks = None
        self._all_data_parallel_group_ranks = None
        self._all_pipeline_global_group_ranks = None
        self._all_pipeline_stage_ranks = None

        assert len(dst_model_ranks) % (self._model_sep_config.TP * self._model_sep_config.PP) == 0
        print(self.interleave_model_parallel_ranks)
        logger.info(f'current_model: {current_model}, {self._model_sep_config.to_config()}'
                    f'self.interleave_model_parallel_ranks: {self.interleave_model_parallel_ranks} '
                    f'self._all_data_parallel_group_ranks: {self._all_data_parallel_group_ranks} '
                    f'self._all_pipeline_global_group_ranks: {self._all_pipeline_global_group_ranks} '
                    f'self._all_pipeline_stage_ranks: {self._all_pipeline_stage_ranks} ')

    def _init_sep_module(self):
        if self._has_sep_inited:
            raise ValueError("SEP module has inited!")
        global PARALLEL_STATE_DICT, MODULE_NAME, ARGS_MODULE_NAME
        assert self._current_model not in PARALLEL_STATE_DICT

        sys_module_name = [
            mod for item, mod in sys.modules.items()
            if item.startswith('megatron.global_vars') or item.startswith('megatron.core.parallel_state')
        # item.startswith('megatron') or item.startswith('alignment.rlhf.models')
        ]
        for item in sys_module_name:
            importlib.reload(item)

        from megatron.initialize import initialize_megatron
        initialize_megatron(args_dict=self._model_sep_config.megatron_args, ignore_unknown_args=True)

        PARALLEL_STATE_DICT[self._current_model] = {
            item: module.__dict__.copy()
            for item, module in sys.modules.items()
            if item.startswith('megatron') or item.startswith('alignment.rlhf.models')
        #  item.startswith('megatron') or item.startswith('alignment.rlhf.models')
        }
        self._has_sep_inited = True

    @property
    def interleave_model_parallel_ranks(self):
        """model_parall_ranks > 1 和 replicas > 1不能同时出现
        """
        if self._interleave_model_parallel_ranks is None:
            # 仅first_stage/last_stage需要传输
            first_last_pipeline_ranks = []
            first_last_pipeline_ranks.extend(self.all_pipeline_stage_ranks[0])
            first_last_pipeline_ranks.extend(self.all_pipeline_stage_ranks[-1])

            data_parallel_size = len(self.all_data_parallel_group_ranks[0])

            self._interleave_model_parallel_ranks = [[
                data_parallel_group_ranks[i] for data_parallel_group_ranks in self.all_data_parallel_group_ranks
                if data_parallel_group_ranks[i] in first_last_pipeline_ranks
            ] for i in range(data_parallel_size)]

            print(f'current_model {self._current_model} interleave_model_parallel_ranks '
                  f'{self._interleave_model_parallel_ranks}, {self.all_data_parallel_group_ranks}')
        # return self._interleave_model_parallel_ranks[0]
        # 数据并行组，len为data_parallel_size，每组仅包含first_stage和last_stage的rank(需要引用原始数据)
        return self._interleave_model_parallel_ranks

    @property
    def all_data_parallel_group_ranks(self):
        """数据并行组，每组的length即数据并行度
        """
        try:
            if self._all_data_parallel_group_ranks is None:
                from megatron.core.parallel_state import _ALL_DATA_PARALLEL_GROUP_RANKS
                self._all_data_parallel_group_ranks = _ALL_DATA_PARALLEL_GROUP_RANKS
            return self._all_data_parallel_group_ranks
        except Exception as e:
            print(f'rank: {torch.distributed.get_rank()} {e}')
            if torch.distributed.get_rank() == 1:
                import pdb
                pdb.set_trace()
            import time
            time.sleep(10000)

    @property
    def all_pipeline_global_group_ranks(self):
        """流水并行组group_ranks，每组的length即流水并行度
        """
        if self._all_pipeline_global_group_ranks is None:
            from megatron.core.parallel_state import _ALL_PIPELINE_GLOBAL_RANKS
            self._all_pipeline_global_group_ranks = _ALL_PIPELINE_GLOBAL_RANKS
        return self._all_pipeline_global_group_ranks

    @property
    def all_pipeline_stage_ranks(self):
        """all_pipeline_stage_ranks，每组为同一个stage
        """
        if self._all_pipeline_stage_ranks is None:
            pipline_stages = len(self.all_pipeline_global_group_ranks[0])
            self._all_pipeline_stage_ranks = [[] for _ in range(pipline_stages)]
            for pipeline_global_group_ranks in self.all_pipeline_global_group_ranks:
                for stage_id, rank in enumerate(pipeline_global_group_ranks):
                    self._all_pipeline_stage_ranks[stage_id].append(rank)
        return self._all_pipeline_stage_ranks


def bucket_tensors(tensors, bucket_size_mb):
    """Group tensors into chunks. We seperate sparse and dense tensor,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Return:
        dense_buckets: Blocks of tensors of same type and within size_limit.
        sparse_bucket: A list of sparse tensors
    """
    size_limit = bucket_size_mb * 1024 * 1024
    buf_dict = defaultdict(lambda: [[], 0])
    dense_buckets = []
    sparse_bucket = []
    for tensor in tensors:
        if tensor.is_sparse:
            sparse_bucket.append(tensor)
            continue
        t = tensor.type()
        size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if size_limit > 0 and buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:    # pylint: disable=chained-comparison
            dense_buckets.append(buf_and_size[0])
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            dense_buckets.append(buf)
    return dense_buckets, sparse_bucket


class DistributedSEPModuleEngine(nn.Module):
    __DATA_HELPER__ = None

    def __init__(self, module, place_policy, current_model, is_pred_model=False):

        # 这里的module，应该是deepspeed的engine。
        super(DistributedSEPModuleEngine, self).__init__()
        if self.__DATA_HELPER__ is None:
            self.__DATA_HELPER__ = DataHelper()
        self.module = module
        self._place_policy = place_policy
        self._current_model = current_model
        self._is_training = False
        self._next_data = None
        self._stream = None
        self._in_generate = False
        self._first_generate = True
        self._async_op_queue = deque()
        self._forward_shape = None
        self._tmp_empty_tensor = None
        self._int64_empty_tensor = None
        self._parameters_to_sync = defaultdict(list)
        self._is_pred_model = is_pred_model

        self._recv_src_models = None
        self._send_dst_models = None

        self._send_param_mapping = None
        self._recv_param_mapping = None

        self._batch_size = self.__DATA_HELPER__._batch_size
        self._seq_len = self.__DATA_HELPER__._seq_length

        self._recv_data_tensor_dict = dict()
        self._async_stream = None

        self._send_next_ops = []
        self._saved_send_param_names = dict()
        self._saved_recv_param_names = dict()

    @property
    def tmp_empty_tensor(self):
        model_dtype = global_context().runtime_conf.dtype
        dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16
        if self._tmp_empty_tensor is None:
            self._tmp_empty_tensor = torch.empty(EMPTY_TENSOR_LENGTH, dtype=dtype).cuda()
        return self._tmp_empty_tensor

    @property
    def int64_empty_tensor(self):
        if self._int64_empty_tensor is None:
            self._int64_empty_tensor = torch.empty(EMPTY_TENSOR_LENGTH, dtype=torch.int64).cuda()
        return self._int64_empty_tensor

    @property
    def async_stream(self):
        if self._async_stream is None:
            self._async_stream = torch.cuda.Stream()
        return self._async_stream

    def _send_data_to_next(self, is_training=False, in_generate=False, **data):
        """
        send: reward_score [bs, 1]
        预测模型目前都先只支持pipeline_stage=1, no_replica。直接根据rank取余，向下游发送数据即可
        """
        # send_data = self._coalesce_send_data(is_training=False, in_generate=False, **data)
        # assert len(self._place_policy.interleave_model_parallel_ranks) == 1
        try:
            # 上游预测模型不开pipeline 且数据并行由replica来做，必然存在该rank_idx
            assert len(self._place_policy.interleave_model_parallel_ranks) == 1
            src_model_parallel_ranks = self._place_policy.interleave_model_parallel_ranks[0]
            my_rank_idx = src_model_parallel_ranks.index(torch.distributed.get_rank())
        except Exception as e:
            if torch.distributed.get_rank() == 7:
                import pdb
                pdb.set_trace()
            import time
            time.sleep(10000)
        # TODO: 默认先set all_keys
        # keys = self._send_keys
        while len(self._send_next_ops) > 0:
            send_op = self._send_next_ops.pop()
            send_op.wait()

        torch.cuda.current_stream().wait_stream(self.async_stream)

        for dst_model in self._send_dst_models:
            print(
                f'rank {torch.distributed.get_rank()} Send data {dst_model._place_policy.interleave_model_parallel_ranks}'
            )
            for dst_group_idx, dst_model_parallel_ranks in enumerate(
                    dst_model._place_policy.interleave_model_parallel_ranks):

                for dst_idx, dst_rank in enumerate(dst_model_parallel_ranks):
                    if dst_idx % len(src_model_parallel_ranks) == my_rank_idx:
                        if dst_rank == torch.distributed.get_rank():
                            # 直接透传
                            # while f'{id(self)}_{id(dst_model)}' in self.__DIST_DATA__:
                            #     import time
                            #     time.sleep(0.1)
                            __DIST_DATA__[f'{id(self)}_{id(dst_model)}_{dst_group_idx}'] = data
                        else:
                            tag_prefix = self.id * 1000 + dst_model.id * 100 + dst_group_idx    # 多pipeline的时候 是否需要包含？
                            with torch.cuda.stream(self.async_stream):
                                ops = self.__DATA_HELPER__._gen_send_op(
                                    data,
                                    self.tmp_empty_tensor,
                                    self.int64_empty_tensor,
                                    dst_rank,
                                    tag_prefix=tag_prefix,
                                    run_async=True,
                                    dst_group_idx=dst_group_idx,
                                    dst_groups=len(dst_model._place_policy.interleave_model_parallel_ranks))
                                self._send_next_ops.extend(ops)

    def _recv_data_from_prev(self, is_training=False, in_generate=False, **data):
        """ TODO: 目前recv多次调用。可能使用同一块tmp，注意加一下计数变量
        seq [bs, seq_length]
        attention_mask [bs, seq_length]
        预测模型目前都先只支持pipeline_stage=1, no_replica。直接根据rank取余，向下游发送数据即可
        """
        my_rank_idx = None
        for group_idx, my_parallel_ranks in enumerate(self._place_policy.interleave_model_parallel_ranks):
            if torch.distributed.get_rank() in my_parallel_ranks:
                my_rank_idx = my_parallel_ranks.index(torch.distributed.get_rank())
                break

        if my_rank_idx is None:
            return

        # if not self._recv_src_models:
        #     return {'all_tokens': torch.randint(100, 50000, [2, 256]).cuda() + torch.distributed.get_rank()}

        self.recv_data_tensor_dict.clear()

        for (src_model, recv_keys) in self.recv_src_models:
            src_model = next(src_model) if isinstance(src_model, cycle) else src_model
            # 多模型注意顺序，防止hang
            assert len(src_model._place_policy.interleave_model_parallel_ranks) == 1
            src_model_parallel_ranks = src_model._place_policy.interleave_model_parallel_ranks[0]

            src_rank_id = my_rank_idx % len(src_model_parallel_ranks)

            src_rank = src_model_parallel_ranks[src_rank_id]

            if src_rank == torch.distributed.get_rank():
                # 理论上上游已经run过，可以直接assert 取一下value
                while f'{id(src_model)}_{id(self)}' not in __DIST_DATA__:
                    time.sleep(1)
                data = __DIST_DATA__.pop(f'{id(src_model)}_{id(self)}_{group_idx}')
            else:
                tag_prefix = src_model.id * 1000 + self.id * 100 + group_idx    # 多pipeline的时候 是否需要包含？
                # 如果rank是自己，就直接从dict取。
                data = self.__DATA_HELPER__._gen_recv_op(recv_keys,
                                                         self.tmp_empty_tensor,
                                                         self.int64_empty_tensor,
                                                         src_rank,
                                                         tag_prefix=tag_prefix,
                                                         groups=len(
                                                             self._place_policy.interleave_model_parallel_ranks))
            self.recv_data_tensor_dict.update(data)
        return self.recv_data_tensor_dict

    @property
    def recv_src_models(self):
        return self._recv_src_models

    @property
    def send_dst_models(self):
        return self._send_dst_models

    @property
    def recv_data_tensor_dict(self):
        return self._recv_data_tensor_dict

    @property
    def send_tmp_data(self):
        pass

    @send_tmp_data.setter
    def send_tmp_data(self, send_tmp_data):
        self._send_tmp_data = send_tmp_data

    @recv_src_models.setter
    def recv_src_models(self, recv_src_models):
        self._recv_src_models = recv_src_models

    @send_dst_models.setter
    def send_dst_models(self, send_dst_models):
        self._send_dst_models = send_dst_models

    def _parse_recv_data(self, is_training=False, in_generate=False):
        pass

    def _coalesce_send_data(self, is_training=False, in_generate=False, **data):
        pass

    def _send_parameters(self, mapping, global_param_dict, group_name, layers_per_stage, total_layer_stage):
        if self.module is not None and getattr(self, 'model', None) is not None:
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
            from megatron.core import sep
            # model_rank = sep.get_pipeline_model_parallel_rank()
            # 目前就是传递了pipeline的全量
            # build_pipeline_layer_name_mapping(layers_per_stage, model_rank, self.model[0])

            for src, dst, pipe_stage, tensor_rank, map_group in mapping:
                if src != torch.distributed.get_rank():
                    continue

                send_mapping_name = f'{src}_{dst}_{pipe_stage}_{tensor_rank}'
                if send_mapping_name not in self._saved_send_param_names:
                    cur_params, cur_param_shape = get_send_params(layers_per_stage, total_layer_stage, pipe_stage,
                                                                  self.model[0])
                    torch.save(cur_param_shape, f'{TMP_PARAM_DIR}/{send_mapping_name}_tmp')
                    os.system(f'mv -f {TMP_PARAM_DIR}/{send_mapping_name}_tmp {TMP_PARAM_DIR}/{send_mapping_name}')
                    self._saved_send_param_names[send_mapping_name] = cur_params

                cur_params = self._saved_send_param_names[send_mapping_name]

                print(f'src {src} dst {dst} pipe_stage {pipe_stage} tensor_rank {tensor_rank}')
                dense_buckets, sparse_bucket = bucket_tensors(cur_params, 100)

                if src == dst:
                    for bucket_id, bucket in enumerate(dense_buckets):
                        global_param_dict[
                            f'{group_name}_pipe_{pipe_stage}_tensor_rank_{tensor_rank}_bucket_{bucket_id}'] = _flatten_dense_tensors(
                                bucket)
                    for param_id, param in enumerate(sparse_bucket):
                        global_param_dict[
                            f'{group_name}_pipe_{pipe_stage}_tensor_rank_{tensor_rank}_param_{param_id}'] = param
                else:
                    for bucket_id, bucket in enumerate(dense_buckets):
                        flat_tensors = _flatten_dense_tensors(bucket)
                        tag = haha(src, dst, pipe_stage, tensor_rank, bucket_id, 0)
                        print(f'send device: {flat_tensors.device},  src {src}, dst {dst} pipe_stage {pipe_stage} '
                              f'tensor_rank {tensor_rank} tag {tag} bucket_id {bucket_id}')
                        torch.distributed.send(flat_tensors, dst, group=map_group)
                    for param_id, param in enumerate(sparse_bucket):
                        tag = haha(src, dst, pipe_stage, tensor_rank, 0, param_id)
                        print(f'send device: {flat_tensors.device},  src {src}, dst {dst} pipe_stage {pipe_stage} '
                              f'tensor_rank {tensor_rank} tag {tag} param_id {param_id}')
                        torch.distributed.send(param, dst, group=map_group)

    def _recv_parameters(self, mapping, global_param_dict, group_name, layers_per_stage, total_layer_stage):
        if self.module is not None and getattr(self, 'model', None) is not None:
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

            cur_model = self.model
            if isinstance(self.model, list):
                cur_model = self.model[0]

            if True:
                # 每次sync recv前，unfuse掉lora的权重
                from alignment.rlhf.models.lora.layers import unfuse_lora_layer
                unfuse_lora_layer(cur_model)

            for src, dst, pipe_stage, tensor_rank, map_group in mapping:
                if dst != torch.distributed.get_rank():
                    continue

                recv_mapping_name = f'{src}_{dst}_{pipe_stage}_{tensor_rank}'
                if recv_mapping_name not in self._saved_recv_param_names:

                    while not os.path.exists(f'{TMP_PARAM_DIR}/{recv_mapping_name}'):
                        time.sleep(1)
                    # 改成原子文件
                    src_param_shape = torch.load(f'{TMP_PARAM_DIR}/{recv_mapping_name}')
                    cur_params = get_recv_params(layers_per_stage, total_layer_stage, pipe_stage, self.model,
                                                 src_param_shape, tensor_rank)
                    self._saved_recv_param_names[recv_mapping_name] = cur_params

                cur_params = self._saved_recv_param_names[recv_mapping_name]

                print(f'src {src} dst {dst} pipe_stage {pipe_stage} tensor_rank {tensor_rank}')
                dense_buckets, sparse_bucket = bucket_tensors(cur_params, 100)
                if src == dst:
                    for bucket_id, bucket in enumerate(dense_buckets):
                        flat_tensors = global_param_dict[
                            f'{group_name}_pipe_{pipe_stage}_tensor_rank_{tensor_rank}_bucket_{bucket_id}']
                        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
                            tensor.copy_(synced)
                    for param_id, param in enumerate(sparse_bucket):
                        param.copy_(global_param_dict[
                            f'{group_name}_pipe_{pipe_stage}_tensor_rank_{tensor_rank}_param_{param_id}'])
                else:
                    for bucket_id, bucket in enumerate(dense_buckets):
                        flat_tensors = _flatten_dense_tensors(bucket)
                        tag = haha(src, dst, pipe_stage, tensor_rank, bucket_id, 0)
                        torch.distributed.recv(flat_tensors, src, group=map_group)
                        print(f'recv device: {flat_tensors.device},  src {src}, dst {dst} pipe_stage {pipe_stage} '
                              f'tensor_rank {tensor_rank} tag {tag} bucket_id {bucket_id}')
                        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
                            tensor.copy_(synced)
                    for param_id, param in enumerate(sparse_bucket):
                        print(f'recv device: {flat_tensors.device},  src {src}, dst {dst} pipe_stage {pipe_stage} '
                              f'tensor_rank {tensor_rank} tag {tag} param_id {param_id}')
                        torch.distributed.recv(param, src, group=map_group)

            if True:
                # TODO: 加下lora的开关，每次sync recv前，unfuse掉lora的权重
                # 只要fuse过，那么fuse_lora这两个小emb就不会进来
                # 目前就是传输前先unfuse_lora;
                from alignment.rlhf.models.lora.layers import fuse_lora_layer
                fuse_lora_layer(cur_model)

    @PatchSEPDistributedEnv()
    def sync(self, send_mapping, recv_mapping, global_param_dict, group_name, layers_per_state, src_pipe_stage):
        with torch.no_grad():
            self._send_parameters(send_mapping, global_param_dict, group_name, layers_per_state, src_pipe_stage)
            self._recv_parameters(recv_mapping, global_param_dict, group_name, layers_per_state, src_pipe_stage)

    @PatchSEPDistributedEnv()
    def _maybe_init_model(self):
        from megatron.training import get_model

        if not getattr(self, '_initialized', None):
            from megatron import get_args
            mega_args = get_args()
            self._initialized = True
            self.model = get_model(self.module, wrap_with_ddp=False)
            if mega_args.load is not None:
                from megatron.checkpointing import load_checkpoint
                from megatron.core.parallel_state import _MODEL_PARALLEL_GROUP
                load_checkpoint(self.model,
                                None,
                                None,
                                adaptive_parallel_strategy=mega_args.adaptive_parallel_strategy_on_checkpoint,
                                group=_MODEL_PARALLEL_GROUP)
            self.model = self.model[0]
            logger.info(
                f'Get args: pipeline_model_parallel_size: {mega_args.pipeline_model_parallel_size}'
                f'tenosr_model_parallel_size: {mega_args.tensor_model_parallel_size}, cur_model: {self._current_model}'
            )
            self.model.eval()

    def setup_model_and_optimizer(self,
                                  model_provider_func,
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0):
        """Setup model and optimizer."""
        from megatron.optimizer import get_megatron_optimizer
        from megatron.training import get_optimizer_param_scheduler, get_model
        from megatron.utils import unwrap_model
        from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
        from megatron.model import DistributedDataParallel as LocalDDP
        from megatron.model import Float16Module
        from megatron.core import sep
        from megatron import get_args

        args = get_args()
        model = get_model(model_provider_func, model_type)
        self.model = model

        if getattr(args, 'enable_lora', False):
            # self.fuse_lora_layer()
            # ignore layer related params
            strict = False
        else:
            strict = True

        if args.load is not None and (args.finetune or args.no_load_optim):    # 是否加载opt
            logger.info(f'Load checkpoint here with no opt')

            from megatron.checkpointing import load_checkpoint
            from megatron.core.parallel_state import _MODEL_PARALLEL_GROUP
            args.iteration = load_checkpoint(self.model, None, None, \
                                             adaptive_parallel_strategy=args.adaptive_parallel_strategy_on_checkpoint, strict=strict,
                                             group=_MODEL_PARALLEL_GROUP)

        unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

        if args.load and args.no_load_optim:
            optimizer = None
            opt_param_scheduler = None
        else:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond, scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        if args.load is not None and not args.finetune:
            logger.info(f'Load checkpoint here with {optimizer}')
            from megatron.checkpointing import load_checkpoint
            from megatron.core.parallel_state import _MODEL_PARALLEL_GROUP
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler, \
                                            adaptive_parallel_strategy=args.adaptive_parallel_strategy_on_checkpoint,
                                            strict=strict,
                                            group=_MODEL_PARALLEL_GROUP)
        else:
            args.iteration = 0

        # We only support local DDP with multiple micro-batches.
        if len(model) > 1 or sep.get_pipeline_model_parallel_world_size() > 1:
            assert args.DDP_impl == 'local'

        # get model without FP16 and/or TorchDDP wrappers
        if len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            # print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if True:
                optimizer.reload_model_params()
        if optimizer is None:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond, scale_lr_cond, lr_mult)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        # if self.enable_lora:
        #     self.unfuse_lora_layer()

        if not isinstance(model, list):
            model = [model]
        [item.train() for item in model]

        return model, optimizer, opt_param_scheduler

    def _forward_step(self, data_loader, model):
        """Forward step."""
        # args = get_args()
        # timers = get_timers()

        # Get the batch.
        # timers('batch-generator').start()
        from functools import partial
        # inputs = self.get_batch(batch_data)
        # timers('batch-generator').stop()

        # TODO: batch_data转成train_batch_size， last pipeline有数据即可
        # inputs = batch_data[1]

        batch_data = self.get_data(next(data_loader))

        losses = model.forward(all_token_ids=batch_data['all_token_ids_right_padded'],
                               all_position_ids=batch_data['all_token_position_ids'],
                               all_token_attention_mask=batch_data['all_token_attention_mask'],
                               training_inputs=batch_data)
        logger.info(f'rank {torch.distributed.get_rank()} Train micro bs ')

        return losses, partial(self.aggregate_loss_func,
                               batch_data)    # will call loss_func(loss_mask, output_tensor) to get loss

    def get_data_iterator(self, epoch, data_loader):
        next_batch = None
        if epoch == 0:
            while next_batch is None:
                data_iterator = self._recv_data_from_prev(is_training=True)

                data_loader.add([data_iterator])
                next_batch = data_loader.next_batch()
        else:
            next_batch = data_loader.next_batch()

        # if torch.distributed.get_rank() == 1:
        #     import pdb
        #     pdb.set_trace()
        # import time
        # time.sleep(100000)

        data_iterator = iter(next_batch)

        return data_iterator

    def get_data(self, batch_data):

        for key in ['action_logprobs', 'action_values', 'action_rewards']:
            # ret = []
            # for idx, dat in enumerate(batch_data[key]):
            #     if batch_data['action_start_indices'][idx] == batch_data['action_end_indices'][idx]:
            #         ret.append(dat[batch_data['action_start_indices'][idx] - 1: batch_data['action_end_indices'][idx] - 1])
            batch_data[key] = batch_data[key][:, batch_data['action_start_indices'][0] -
                                              1:batch_data['action_end_indices'][0] - 1]

        from megatron import get_tokenizer
        from megatron import get_args
        from alignment.rlhf.models.constants_ppo import pad_to_max_len, get_ltor_masks_and_position_ids

        args = get_args()
        # TODO: move to RLHF framework later. add pad to max length config
        all_token_ids_right_padded = pad_to_max_len(batch_data["all_token_ids_right_padded"],
                                                    args.seq_length,
                                                    pad_value=get_tokenizer().eos_token_id)
        # NOTE this pad to max is even better than get_loss_mask again because for the maxedout response cases, get_loss_mask will
        # add a loss mask = 1 to the first eod token which is WRONG because they didn't want to stop and most likely it shouldn't stop. it's just maxed out.
        all_token_loss_mask = pad_to_max_len(batch_data["loss_mask"], args.seq_length, pad_value=0)

        all_token_attention_mask, all_token_position_ids = get_ltor_masks_and_position_ids(all_token_ids_right_padded)
        # print(f"all_token_position_ids: {all_token_position_ids}")
        # response_length = batch_data["action_rewards"].shape[1]

        inputs = {
            "all_token_position_ids": all_token_position_ids,
            "all_token_ids_right_padded": all_token_ids_right_padded,
        # this attention mask is not TRANSFOEMRER attention msak. this actually applies on attention result [b, np, s, s]
            "all_token_attention_mask": all_token_attention_mask.bool(),
            "all_token_loss_mask": all_token_loss_mask.bool(),
            "action_starts": batch_data['action_start_indices'],
            "action_logprobs": batch_data["action_logprobs"],    # response size
            "action_values": batch_data["action_values"],
            "action_rewards": batch_data["action_rewards"],
        }

        return inputs


class RewardDistributedSEPModuleEngine(DistributedSEPModuleEngine):
    def decode(
            self,
            prompt_sizes,
            all_tokens,
    ):
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        from megatron import get_tokenizer
        tokenizer = get_tokenizer().tokenizer
        # Assuming prompts were left-padded
        # 这里本来是传了原始list sample，为了传输方便，和原始不同。假设是右padding，直接传输原始的prompt_sizes。。。
        # prompt_sizes = [len(q) for q in no_padded_query_ids]

        str_samples, str_prompts, str_outputs, response_ids = [], [], [], []
        for sample, prompt_size in zip(all_tokens, prompt_sizes):
            output_start_ix = prompt_size
            str_prompt = tokenizer.decode(sample[:prompt_size].tolist(),
            #  skip_special_tokens=True
                                          )
            str_output = tokenizer.decode(sample[output_start_ix:].tolist(),
            #  skip_special_tokens=True
                                          )
            response_id = sample[output_start_ix:]
            response_ids.append(response_id)

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs, response_ids

    def find_ngrams(self, x, n):
        L = x.size(0)

        if L == 0:
            return 0, None
        # Pad the input tensor with zeros at the end to ensure that we can extract
        # n-grams up to the last element of the tensor.
        padded_x = torch.cat((x, torch.zeros(n - 1, device=x.device, dtype=x.dtype)), dim=0)
        # Use the unfold method to extract all sliding windows of size n from the
        # padded input tensor.
        # The step size is 1, which means we extract n-grams with overlapping
        # elements.
        # The size of the resulting tensor is (L - n + 1, n), which contains all n-grams.
        ngrams = padded_x.unfold(0, n, 1)[:L - n + 1]
        # Count the frequency of each n-gram
        unique_ngrams, counts = torch.unique(ngrams, return_counts=True, dim=0)
        if len(unique_ngrams) == 0:
            return 0, None
        max_count_index = torch.argmax(counts)
        max_count = counts[max_count_index]

        # get the most frequent n-gram
        most_frequent_ngram = unique_ngrams[max_count_index]
        if max_count >= 2:
            indices = torch.nonzero(torch.eq(ngrams, most_frequent_ngram).all(dim=1)).view(-1)

            # if diff is less than ngram size it's overlapping then count as 1
            diff = (torch.diff(indices).float().mean() - n).item()
            if diff < 1.0:
                diff = 1.0
        else:
            diff = None
        return max_count, diff

    def get_n_gram_reward(self, tokens):

        assert len(tokens.size()) == 1, f"must be 1d {tokens}"
        penalty = 0.0
        max_repetition, average_distance = self.find_ngrams(tokens, 2)
        max_repetition_3, average_distance_3 = self.find_ngrams(tokens, 3)

        if average_distance is not None:
            # must have a repetition found
            assert max_repetition >= 2, f"{max_repetition}"
            penalty += max_repetition.item() / (average_distance)

        if average_distance_3 is not None:
            assert max_repetition_3 >= 2, f"{max_repetition_3}"

            penalty += max_repetition_3.item() / (average_distance_3)

        return -penalty

    def forward_step_pipeline(self, list_strs):

        from megatron import get_args
        from megatron.core import sep

        from megatron.utils import get_ltor_masks_and_position_ids
        from alignment.rlhf.models.forward_step import forward_step_helper
        from alignment.rlhf.models.reward_model import batch_padded_tokenize_data
        from megatron.text_generation.communication import broadcast_from_last_to_first_pipeline_stage

        self.model.eval()

        from megatron import get_args, get_tokenizer
        tokenizer = get_tokenizer()
        args = get_args()
        input_ids, pooling_sequence_index = batch_padded_tokenize_data(list_strs, tokenizer, args.seq_length)
        input_ids = input_ids.cuda()
        pooling_sequence_index = pooling_sequence_index.cuda()

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, tokenizer.eod,
                                                                          args.reset_position_ids,
                                                                          args.reset_attention_mask,
                                                                          args.eod_mask_loss)

        batch_size = input_ids.size(0)
        max_length = input_ids.size(1)

        # Log probability of the sequence (prompt + generated tokens).
        output_rewards = None
        output_rewards_size = (batch_size, )

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            # logits will be meanigful only in the last pipeline stage.
            lm_output = forward_step_helper(self.model, input_ids, position_ids, attention_mask,
                                            pooling_sequence_index)

            if sep.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert lm_output is not None
                output_rewards = lm_output

                assert batch_size == 1 or output_rewards.size(0) == batch_size

        # ======================================
        # Broadcast to the first pipeline stage.
        # ======================================
        output_rewards = broadcast_from_last_to_first_pipeline_stage(output_rewards_size, torch.float32,
                                                                     output_rewards)

        return output_rewards

    def normalized_and_clip(self, scores):
        # TODO: 对齐下
        # all_scores_mean, all_scores_std = self.running.update(scores)

        # self.args.__dict__.get('scale_reward', "None") == "running":
        #     if self.running.count >= 2:
        #         scores /= self.running.std

        clip_reward = self.args.__dict__.get('cliprange_reward', 100)
        if clip_reward:
            scores = torch.clip(scores, -clip_reward, clip_reward)
        return scores

    def get_raw_reward(self, list_strs):
        scores = self.forward_step_pipeline(list_strs)
        # minus the sft baseline
        # TODO: 这几个参数不对
        scores -= self.args.__dict__.get('reward_bias', 0)
        scores = self.normalized_and_clip(scores)
        return scores

    def get_math_matching_reward(self, str_prompt, str_response, training_math_golden_reg):
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"

        def extract_answer_qianwen(p):
            p = p.strip()
            p_num = " ".join(p.split('\n'))
            if p_num:
                p_num = re.sub('\([^\(\)]*[^0-9,\(\)./]+[^\(\)]*\)', '', p_num)

            p_num = re.findall('(-?[\d,]+)(\.[\d,]+)?(/-?[\d,]+)?(\.[\d,]+)?', p_num)

            if p_num:
                p_num = ["".join(p) for p in p_num]
                p_num = [''.join(p.split(',')) for p in p_num]
                p_num = [p for p in p_num if p]
                if p_num:
                    try:
                        ret = float(eval(p_num[-1]))
                        return ret
                    except BaseException:
                        return INVALID_ANS
                return INVALID_ANS
            return INVALID_ANS

        def extract_answer(completion):
            match = ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                try:
                    float(match_str)
                except BaseException:
                    return INVALID_ANS
                return match_str
            else:
                return INVALID_ANS

        gold_ans = training_math_golden_reg[str_prompt]
        if "####" in str_response:
            pred_ans = extract_answer(str_response)
        else:
            pred_ans = extract_answer_qianwen(str_response)

        if pred_ans != INVALID_ANS and abs(float(pred_ans) - float(gold_ans)) < 1e-4:
            score = 1.0
        else:
            score = -1.0
        return score

    def get_all_rewards(self, action_starts, action_ends, loss_mask, list_strs, logprobs, ref_logprobs, kl_ctl,
                        action_tokens):
        '''
        # list_strs = [['sada asjdkha dfnkalbkfd', 'sa sajidha ahkjdfa jsdkaj'], ['sada asjdkha dfnsda sdkalbkfd', 'sa sajidha ahkjass s sdfa jsdkaj']]

        :param query_tensors:
        :param loss_mask:
        :param list_strs:
        :param logprobs: logprob of prompt + response
        :param ref_logprobs: logprob of prompt + response
        :param kl_ctl:
        :return:
        '''

        # ngram reward

        n = loss_mask.size(0)
        if self.args.raw_reward_coeff > 0:
            scores = self.args.raw_reward_coeff * self.get_raw_reward(list_strs).view(-1, 1)
        else:
            scores = torch.zeros(n, 1, device=loss_mask.device)
        if self.args.ngram_coef > 0:
            ngram_rewards = []
            for action_token in action_tokens:
                ngram_rewards.append(self.args.ngram_coef * self.get_n_gram_reward(action_token))

            ngram_rewards = torch.tensor(ngram_rewards, device=scores.device)

        if self.args.math_coef > 0:
            # Get the batch.
            math_rewards = []
            for str_prompt, str_response in list_strs:
                math_rewards.append(
                    self.args.math_coef *
                    self.get_math_matching_reward(str_prompt, str_response, self.training_math_golden_reg))

            math_rewards = torch.tensor(math_rewards, device=scores.device)

        rewards = -kl_ctl * (logprobs - ref_logprobs)

        if self.args.lm_coef > 0:
            lm_reward = self.args.lm_coef * ref_logprobs

            rewards += lm_reward

        # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
        # 第一个没有logprobs/ref_logprobs
        rewards = [
            rs[action_starts[ix] - 1:action_ends[ix] - 1] / (action_ends[ix] - action_starts[ix])
            for ix, rs in enumerate(rewards)
        ]

        ret = torch.zeros(logprobs.size(), device=logprobs.device, dtype=rewards[-1].dtype)    # (bs, seq_len - 1)
        # Cosepte rewards
        # all_rewards = [None] * n

        for ix in range(n):
            rs = rewards[ix]
            if len(rs) == 0:
                rs = torch.tensor([0.0])
            rs[-1] += scores[ix].cpu().item()

            if self.args.ngram_coef > 0:
                # TODO: 这里是cpu？
                rs[-1] += ngram_rewards[ix].cpu().item()
            if self.args.math_coef > 0:
                rs[-1] += math_rewards[ix].cpu().item()

            if action_starts[ix] == action_ends[ix]:
                ret[ix][action_starts[ix] - 1] = rs[-1]
            else:
                ret[ix][action_starts[ix] - 1:action_ends[ix] - 1] = rs
            # all_rewards[ix] = rs
        # 这里可能只是为了print
        # all_rw_means = torch.tensor([rr.sum() for rr in all_rewards],
        #                             dtype=torch.float32,
        #                             device=torch.cuda.current_device())
        return ret

    # 类似reward，forward_value直接这样写，暂时还不支持流水
    # TODO: 改成reward_inference.py 中推导的形式，应该ok
    @TraceEventScope("forward_reward")
    @PatchSEPDistributedEnv()
    def forward_step(self, **other_non_tensor_property):
        '''

        RLHF calling
        rlhf framework source:              old_values = self.value.forward_step(policy_output[0])


        :param data_b: micro_batch??
        :return:
            {"old_values": output_values}
        '''
        if self.module is not None:
            input_data = self._recv_data_from_prev()
            seq = input_data['all_tokens']

            self._maybe_init_model()

            # if torch.distributed.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()
            # import time
            # time.sleep(100000)

            loss_mask = input_data["loss_mask"]
            prompt_sizes = input_data["prompt_sizes"]
            all_tokens_right_padded = input_data["all_tokens"]

            # decode完，后续的infer仅取当前bs的max_leng进行padding
            str_samples, str_prompts, str_outputs, _ = self.decode(prompt_sizes, all_tokens_right_padded)

            list_strs = [[str_prompt, str_output] for str_prompt, str_output in zip(str_prompts, str_outputs)]

            old_value = input_data["old_values"]
            ref_logprobs = input_data["ref_logprobs"]
            logprobs = input_data["logprobs"]

            from megatron import get_args
            self.args = get_args()

            if self.args.fix_kl_coef:
                kl_coef = self.args.init_kl_coef
            else:
                kl_coef = self.get("kl_coef")
                if kl_coef is None:
                    kl_coef = self.args.init_kl_coef
            '''
                    "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
                    "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
                    "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
                    "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
                    "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
            '''

            n = all_tokens_right_padded.shape[0]
            # if ends with a eos_token also pad, it doesn't change. if stopped due to len limit, discard last token to align with rewards. because reward is r(s,a) which is a state action pair starts from state, thus the last unstopped token has no reward assigned and thus need to discard
            values = old_value[:, :-1]

            if self.args.loss_on_prompts:
                # because first token has no prob and serve as the first token to attend to so no loss
                starts = torch.tensor([1 for no_padded_query_id in prompt_sizes], dtype=torch.long)
            else:
                starts = torch.tensor([prompt_size for prompt_size in prompt_sizes], dtype=torch.long)
            ends = torch.tensor([start + loss_mask[i, start:].sum() for i, start in enumerate(starts)],
                                dtype=torch.long)
            # -1 because logprobs are logprobs of action_id[1:]!!! so it's already shifted right, to get logprob of first action, we need -1
            # start = query_tensors.shape[1] - 1 is because we need state's value!! so 1 step ahead
            # eg [ pad, q1, q2, q3, a1, a2, a3, pad, pad] -> ends[i] = 4
            # eg [ pad, q1, q2, q3, a1, a2, a3] -> [ pad, q1, q2, q3, a1, a2] ends[i] = 3
            # all values = value(hidden(q3, a1, a2, a3)).
            """TODO: action_values(all_values)/action_logprobs(all_logprobs)挪动到后续train_model里进一步拆分
            """
            # all_values = [values[ix, starts[ix] - 1:ends[ix] - 1] for ix in range(n)]    # we want states
            # # [ pad, q1, q2, q3, a1, a2, a3], logprobs= logprob[ q1, q2, q3, a1, a2, a3]
            # # start = 4 - 1 = 3 ends[i] = 4  logprobs[3: 3 + 4=7] = logprob[a1, a2, a3]]
            # all_logprobs = [logprobs[ix, starts[ix] - 1:ends[ix] - 1] for ix in range(n)]

            action_tokens = [all_tokens_right_padded[ix, starts[ix]:ends[ix]] for ix in range(n)]
            all_rewards = self.get_all_rewards(starts, ends, loss_mask, list_strs, logprobs, ref_logprobs, kl_coef,
                                               action_tokens)

            data = {
                "all_token_ids_right_padded": all_tokens_right_padded,
                "action_start_indices": starts,
                "action_end_indices": ends,
                "action_logprobs": logprobs,    # (bs, seq_len - 1)
                "action_values": values,    # (bs, seq_len - 1)
                "action_rewards": all_rewards,    # (bs, seq_len - 1)
                "loss_mask": loss_mask
            }
            self._send_data_to_next(**data)

            return data

    @property
    def send_tmp_data(self):
        """values, [bs, 1]
        """
        if self._send_tmp_data is None:
            # 目前只传输单值，直接传输即可，不做concat
            # self._send_tmp_data = torch.empty()
            pass
        return self._send_tmp_data


class RefDistributedSEPModuleEngine(DistributedSEPModuleEngine):
    def score_and_return_on_first_stage(self, model, tokens):
        """Function for just scoring.
        Arguments:
            model: no interleaving is supported.
            tokens: prompt tokens extended to be of size [b, max_prompt_length]
            lengths: original prompt length, size: [b]
        Note: Outside of model, other parameters only need to be available on
              rank 0.
        Outputs:
            output_log_probs: log probability of the selected tokens. size: [b, s]
        """
        from megatron import get_args
        from alignment.rlhf.models.constants_ppo import get_ltor_masks_and_position_ids
        from alignment.rlhf.models.forward_step import forward_step_helper
        from megatron.core import sep
        from megatron.text_generation.communication import broadcast_from_last_to_first_pipeline_stage
        from megatron.core.tensor_parallel.utils import VocabUtility
        import torch.nn.functional as F

        from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
        from megatron.model import DistributedDataParallel as LocalDDP
        from megatron.model import Float16Module
        from megatron.utils import unwrap_model

        args = get_args()

        batch_size = tokens.size(0)
        all_tokens_len = tokens.size(1)
        max_sequence_length = min(all_tokens_len, args.max_position_embeddings)

        # Log probability of the sequence (prompt + generated tokens).
        output_log_probs = None
        output_log_probs_size = (batch_size, max_sequence_length - 1)

        unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        parallel_output = getattr(unwrapped_model, 'parallel_output', False)

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            attention_mask, position_ids = get_ltor_masks_and_position_ids(tokens)

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step_helper(self.model, tokens, position_ids, attention_mask)

            if not parallel_output:
                if sep.is_pipeline_last_stage():
                    # Always the last stage should have an output.
                    assert logits is not None
                    assert logits.size(1) == tokens.size(1), "head(hidden(token))"
                    log_probs = F.log_softmax(logits, dim=2)

                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    indices = torch.unsqueeze(tokens[:, 1:], 2)
                    output_log_probs = torch.gather(log_probs, 2, indices).squeeze(2)
            else:
                if sep.is_pipeline_last_stage():
                    vocab_parallel_logits = logits
                    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
                    torch.distributed.all_reduce(logits_max,
                                                 op=torch.distributed.ReduceOp.MAX,
                                                 group=sep.get_tensor_model_parallel_group())
                    logits.sub_(logits_max.unsqueeze(dim=-1))
                    # Get the partition's vocab indecies
                    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
                    partition_vocab_size = vocab_parallel_logits.size()[-1]
                    rank = sep.get_tensor_model_parallel_rank()
                    world_size = sep.get_tensor_model_parallel_world_size()
                    vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

                    indices = torch.unsqueeze(tokens, 2)

                    # Create a mask of valid vocab ids (1 means it needs to be masked).
                    target_mask = (indices < vocab_start_index) | (
                        indices >= vocab_end_index)    # [b,s] 1 for not in range action, 0 for in range

                    masked_actionids = indices - vocab_start_index    # [b,s]
                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    masked_actionids[:, 0, :] = 0
                    masked_actionids[target_mask] = 0    # [b,s]
                    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)    # [n vp]
                    masked_actionids_1d = masked_actionids.view(
                        -1)    # [n] 0 for not in vocab range, target id -start for in range
                    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
                    predicted_logits_1d = logits_2d[
                        arange_1d, masked_actionids_1d]    # [n] in range target logit, not in range logits[0]
                    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
                    action_logits = predicted_logits_1d.view_as(indices)
                    action_logits[target_mask] = 0.0    # [b s] 0 for not in range, logit for in range
                    # All reduce is needed to get the chunks from other GPUs.
                    torch.distributed.all_reduce(action_logits,
                                                 op=torch.distributed.ReduceOp.SUM,
                                                 group=sep.get_tensor_model_parallel_group())
                    # Sum of exponential of logits along vocab dimension across all GPUs.
                    exp_logits = vocab_parallel_logits    # [ b, s, vp ]
                    torch.exp(vocab_parallel_logits, out=exp_logits)
                    sum_exp_logits = exp_logits.sum(dim=-1)
                    torch.distributed.all_reduce(sum_exp_logits,
                                                 op=torch.distributed.ReduceOp.SUM,
                                                 group=sep.get_tensor_model_parallel_group())
                    log_probs = action_logits.squeeze(2) - torch.log(
                        sum_exp_logits + 1e-10)    # log ( exp(l) / sum(exp(li)

                    # shift by 1
                    output_log_probs = log_probs[:, 1:]

        # ======================================
        # Broadcast to the first pipeline stage.
        # ======================================

        if output_log_probs is not None and not output_log_probs.is_contiguous():
            output_log_probs = output_log_probs.contiguous()
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(output_log_probs_size, torch.float32,
                                                                       output_log_probs)
        return output_log_probs

    @TraceEventScope("forward_ref")
    @PatchSEPDistributedEnv()
    def forward_step(self, **other_non_tensor_property):
        '''

        RLHF calling
        rlhf framework source:              old_values = self.value.forward_step(policy_output[0])


        :param data_b: micro_batch??
        :return:
            {"old_values": output_values}
        '''

        if self.module is not None:
            input_data = self._recv_data_from_prev()
            self._maybe_init_model()
            all_tokens = input_data['all_tokens']
            ref_logprobs = self.score_and_return_on_first_stage(self.model, all_tokens)

            assert not torch.isnan(ref_logprobs).any(), f"just out ref_logprobs {ref_logprobs}"

            assert ref_logprobs.size(1) == all_tokens.size(1) - 1, "all token logprob except first one [1:]"

            data = dict(ref_logprobs=ref_logprobs)
            self._send_data_to_next(**data)
            return ref_logprobs

    @property
    def send_tmp_data(self):
        """ref_log_probs, [bs, seq_len, vocab_size]
        """
        if self._send_tmp_data is None:
            # 目前只传输单值，直接传输即可，不做concat
            # self._send_tmp_data = torch.empty()
            pass
        return self._send_tmp_data


class ActorDistributedSEPModuleEngine(DistributedSEPModuleEngine):
    def _tokenize_prompts_and_batch(self, prompts_tokens, tokens_to_generate, add_BOS):
        """Given a set of prompts and number of tokens to generate:
        prompts_tokens: THIS must be left padded to the same length!!!!!
            - tokenize prompts
            - set the sequence length to be the max of length of prompts
              plus the number of tokens we would like to generate
            - pad all the sequences to this length so we can convert them
              into a 2D tensor.
        """
        from megatron import get_args, get_tokenizer
        import torch.nn.functional as F
        from megatron import get_args
        args = get_args()
        # Tokenize all the prompts.
        tokenizer = get_tokenizer()

        # Now we have a list of list of tokens which each list has a different
        # size. We want to extend this list to:
        #   - incorporate the tokens that need to be generated
        #   - make all the sequences equal length.
        # Get the prompts length.
        prompts_length = [len(prompt_token) for prompt_token in prompts_tokens]
        # Get the max prompts length.
        max_prompt_len = max(prompts_length)

        # Number of tokens in the each sample of the batch.
        # samples_length = max_prompt_len + tokens_to_generate
        # 这里写死先，for传输固定长度先。后续可类似broadcast 长度 & tensor
        samples_length = args.seq_length
        # Now update the list of list to be of the same size: samples_length.

        prompts_tokens = [
            F.pad(
                prompt_token,
                (0, samples_length - len(prompt_token)),
                value=tokenizer.eod,    # just pad_token_id
            ) for prompt_token in prompts_tokens
        ]
        prompts_tokens_tensor = torch.vstack(prompts_tokens).to(torch.cuda.current_device())
        assert prompts_tokens_tensor.size(1) == samples_length, "pad to the query_size + max generate size"

        # Now we are in a structured format, we can convert to tensors.
        # prompts_tokens_tensor = torch.cuda.LongTensor(prompts_tokens)

        # print(f"after pad prompts_tokens_tensor size {prompts_tokens_tensor.size()}")
        prompts_length_tensor = torch.cuda.LongTensor(prompts_length)
        # assert torch.all(prompts_length_tensor ==  max_prompt_len), "because left padded"
        return prompts_tokens_tensor, prompts_length_tensor

    def tokenize_prompts(self, prompts_ids=None, tokens_to_generate=None, add_BOS=None, rank=0, group=None):
        """Tokenize prompts and make them avaiable on all ranks."""
        from megatron.text_generation.communication import broadcast_float_list, \
    broadcast_int_list, broadcast_tensor
        # On all ranks set to None so we can pass them to functions
        sizes_list = None
        prompts_tokens_cuda_long_tensor = None
        prompts_length_cuda_long_tensor = None

        # On the specified rank, build the above.
        if torch.distributed.get_rank() == rank:
            assert prompts_ids is not None
            assert tokens_to_generate is not None
            # Tensor of tokens padded and their unpadded length.
            prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
                self._tokenize_prompts_and_batch(prompts_ids, tokens_to_generate, add_BOS)
            # We need the sizes of these tensors for the boradcast
            sizes_list = [
                prompts_tokens_cuda_long_tensor.size(0),    # Batch size
                prompts_tokens_cuda_long_tensor.size(1)
            ]    # Sequence length

        # First, broadcast the sizes.
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank, group=group)

        # Now that we have the sizes, we can boradcast the tokens
        # and length tensors.
        sizes = sizes_tensor.tolist()
        prompts_tokens_cuda_long_tensor = broadcast_tensor(sizes,
                                                           torch.int64,
                                                           tensor=prompts_tokens_cuda_long_tensor,
                                                           rank=rank,
                                                           group=group)
        prompts_length_cuda_long_tensor = broadcast_tensor(sizes[0],
                                                           torch.int64,
                                                           tensor=prompts_length_cuda_long_tensor,
                                                           rank=rank,
                                                           group=group)

        return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor

    def _generate(self,
                  model,
                  prompts_ids=None,
                  tokens_to_generate=0,
                  return_output_log_probs=False,
                  top_k_sampling=0,
                  top_p_sampling=0.0,
                  temperature=1.0,
                  add_BOS=False,
                  use_eod_token_for_early_termination=True,
                  stop_on_double_eol=False,
                  stop_on_eol=False,
                  random_seed=-1):
        """Given prompts and input parameters, run inference and return:
           tokens: prompts plus the generated tokens.
           lengths: length of the prompt + generations. Note that we can
               discard tokens in the tokens tensor that are after the
               corresponding length.
           output_log_probs: log probs of the tokens.
        """
        from megatron.text_generation.communication import broadcast_float_list
        from megatron.text_generation.generation import generate_tokens_probs_and_return_on_first_stage

        from megatron import get_args
        args = get_args()

        # Make sure input params are avaialble to all ranks.
        values = [
            float(item) for item in [
                tokens_to_generate, return_output_log_probs, top_k_sampling, top_p_sampling, temperature, add_BOS,
                use_eod_token_for_early_termination, stop_on_double_eol, stop_on_eol, random_seed
            ]
        ]
        from megatron.core.parallel_state import _MODEL_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP_RANKS

        model_parallel_group = _MODEL_PARALLEL_GROUP

        cur_model_rank = _MODEL_PARALLEL_GROUP_RANKS[0]
        values_float_tensor = broadcast_float_list(10,
                                                   float_list=values,
                                                   group=model_parallel_group,
                                                   rank=cur_model_rank)
        tokens_to_generate = int(values_float_tensor[0].item())
        return_output_log_probs = bool(values_float_tensor[1].item())
        top_k_sampling = int(values_float_tensor[2].item())
        top_p_sampling = values_float_tensor[3].item()
        temperature = values_float_tensor[4].item()
        add_BOS = bool(values_float_tensor[5].item())
        use_eod_token_for_early_termination = bool(values_float_tensor[6].item())
        stop_on_double_eol = bool(values_float_tensor[7].item())
        stop_on_eol = bool(values_float_tensor[8].item())
        random_seed = int(values_float_tensor[9].item())

        if random_seed != -1:
            torch.random.manual_seed(random_seed)

        # Tokenize prompts and get the batch.
        # Note that these tensors are broadcaseted to all ranks.
        if torch.distributed.get_rank(group=model_parallel_group) == 0:
            assert prompts_ids is not None

        prompts_ids, context_length_tensor = self.tokenize_prompts(prompts_ids=prompts_ids,
                                                                   tokens_to_generate=tokens_to_generate,
                                                                   add_BOS=add_BOS,
                                                                   rank=cur_model_rank,
                                                                   group=model_parallel_group)

        # Main inference function.
        # Note that the outputs are available on the first stage.

        # context_length_tensor = torch.ones(prompts_ids.size(0), dtype=torch.long, device=torch.cuda.current_device()) * prompts_ids.size(1)
        if hasattr(args, "use_eod_token_for_early_termination") and not args.use_eod_token_for_early_termination:
            use_eod_token_for_early_termination = False
        logger.info(f"use_eod_token_for_early_termination: {use_eod_token_for_early_termination}")
        # batch_generation_pipeline = rlhf.get_args().active_module_args.batch_generation.batch_generation_pipeline
        if False:
            from rlhf.opt.batch_generation.generation import generate_tokens_probs_and_return_on_first_stage as \
    generate_tokens_probs_and_return_on_first_stage_pipeline
            logger.info(f"Enable batch generation pipeline: num_max_tokens = \
                {rlhf.get_args().active_module_args.batch_generation.batch_generation_pipeline_num_max_tokens}")
            generate_func_internal = generate_tokens_probs_and_return_on_first_stage_pipeline
        else:
            generate_func_internal = generate_tokens_probs_and_return_on_first_stage
        return generate_func_internal(model,
                                      prompts_ids,
                                      context_length_tensor,
                                      return_output_log_probs=return_output_log_probs,
                                      top_k=top_k_sampling,
                                      top_p=top_p_sampling,
                                      temperature=temperature,
                                      use_eod_token_for_early_termination=use_eod_token_for_early_termination,
                                      stop_on_double_eol=stop_on_double_eol,
                                      stop_on_eol=stop_on_eol)

    @property
    def send_tmp_data(self):
        """values, [bs, seq_len, vocab_size]
        """
        if self._send_tmp_data is None:
            # 目前只传输单值，直接传输即可，不做concat
            # self._send_tmp_data = torch.empty()
            pass
        return self._send_tmp_data

    def replace_all_after_first_stop_sequences_by_pad(self, tokens, all_log_probs, stop_token, prompt_sizes):
        '''
        only replace after the stop tokens in the response ignore in the prompt
        :param tokens:
        :param all_log_probs:
        :param stop_token:
        :param prompt_sizes:
        :return:
        '''
        from megatron import get_args, get_tokenizer
        assert len(tokens.size()) == 2
        assert len(all_log_probs.size()) == 2
        occurrences = (tokens == stop_token).float()

        row_indices = torch.arange(tokens.size(1)).unsqueeze(0).expand(tokens.size(0), -1).to(tokens.device)
        response_mask = (row_indices >= prompt_sizes.unsqueeze(1)).int()

        # mask out the stop appear in the prompt:
        occurrences = occurrences * response_mask

        first_stop_sequence_indices = torch.argmax(occurrences, dim=1)

        # if not found a stop sequence, occurrence will be sum 0 dim1
        not_found_mask = torch.sum(occurrences, dim=1) == 0
        # for the not found one. take stop_sequence = tokens.size(1)-1 before everything else replace tokens afterwards
        first_stop_sequence_indices[not_found_mask] = tokens.size(1) - 1
        # print(f"first_stop_sequence_indices {first_stop_sequence_indices}")

        tokenizer = get_tokenizer().tokenizer
        for i in range(tokens.size(0)):
            if first_stop_sequence_indices[i] < tokens.size(1) - 1:
                # if not the last tokne to stop
                tokens[i, first_stop_sequence_indices[i] + 1:] = tokenizer.eos_token_id

        # because all_log_probs is the logprobs of the tokens[:, 1:], thus index 4 at tokens = index 3 at all_log_prob
        all_log_probs_indices = first_stop_sequence_indices - 1

        for i in range(all_log_probs.size(0)):
            if all_log_probs_indices[i] < all_log_probs.size(1) - 1:
                # if not the last log prob to stop
                all_log_probs[i, all_log_probs_indices[i] + 1:] = 0.0

        return tokens, all_log_probs

    def get_loss_mask(self, all_tokens_right_padded, pad_token_id, prompt_sizes):
        '''
        if prompt_sizes is None means it doesn't care about if
        :param all_tokens_right_padded:
        :param pad_token_id:
        :param prompt_sizes:
        :return:
        '''
        loss_mask = (all_tokens_right_padded.not_equal(pad_token_id).to(torch.int64).cuda())
        # we don't just caclulate loss on the action tokens but also the first pad token if present

        #
        # all_tokens_right_padded len = is the max length of the prompts + max generation tokens, so if there is no 0, take it as max len reached.
        # thus just
        occurrences = (all_tokens_right_padded == pad_token_id).float()

        row_indices = torch.arange(all_tokens_right_padded.size(1)).unsqueeze(0).expand(
            all_tokens_right_padded.size(0), -1).to(all_tokens_right_padded.device)
        response_mask = (row_indices >= prompt_sizes.unsqueeze(1)).int()

        # mask out the stop appear in the prompt:
        occurrences = occurrences * response_mask

        first_stop_sequence_indices = torch.argmax(occurrences, dim=1)

        # if not found a stop sequence, occurrence will be sum 0 dim1
        not_found_mask = torch.sum(occurrences, dim=1) == 0
        # for the not found one. take stop_sequence = tokens.size(1)-1 before everything else replace tokens afterwards

        for i in range(loss_mask.size(0)):
            if not_found_mask[i] == 0:
                # if not not found = found a stop sequence.
                loss_mask[i, first_stop_sequence_indices[i]] = 1

        return loss_mask

    @TraceEventScope("forward_actor")
    @PatchSEPDistributedEnv()
    def forward_step(self, data_loader):
        if self.module is not None:
            # input_data = self._recv_data_from_prev()
            from megatron import get_args, get_tokenizer
            args = get_args()

            self._maybe_init_model()
            input_data = next(data_loader)
            no_padded_query_ids = input_data['input_ids']

            tokenizer = get_tokenizer().tokenizer
            # for d in no_padded_query_ids:
            #     d_str = tokenizer.decode(
            #         d.tolist(),
            #         # skip_special_tokens=True
            #     )
            """
                            "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
                            "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
                            "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
                            "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
                            "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
            """
            # tokens: [b, qs + rs],
            tokens, lengths, all_log_probs = self._generate(
                self.model,
                prompts_ids=no_padded_query_ids,
                tokens_to_generate=args.max_new_tokens,
                return_output_log_probs=True,
                top_k_sampling=args.top_k,
                top_p_sampling=args.top_p,
                temperature=args.temperature,
            # top_k_sampling=args.top_k if not eval_mode else args.eval_top_k,
            # top_p_sampling=args.top_p if not eval_mode else args.eval_top_p,
            # temperature=args.temperature if not eval_mode else args.eval_temperature,
                add_BOS=False,
                use_eod_token_for_early_termination=False,
                stop_on_double_eol=False,
                stop_on_eol=False)
            print(f'Cur valid_len: {lengths}')
            # import pdb
            # pdb.set_trace()

            _all_tokens_max_len = tokens.size(1)
            assert not torch.isnan(all_log_probs).any(), f"just out old_logprobs {all_log_probs}"

            assert all_log_probs.size(1) == tokens.size(1) - 1, "because first token hsa no log prob logprob[:, 1:]"

            # everything after stop_token in tokens will be pad_token,
            # everything after stop_token_indx -1 in all_log_probs will be 0.0 (its pad since it's used to minus other logprob)
            prompt_sizes = torch.tensor([len(q) for q in no_padded_query_ids], device=tokens.device)

            # TODO: 参数确认，这里会报错
            tokens, all_log_probs = self.replace_all_after_first_stop_sequences_by_pad(
                tokens, all_log_probs, stop_token=tokenizer.eos_token_id, prompt_sizes=prompt_sizes)
            assert all_log_probs.size(1) == tokens.size(1) - 1, "because first token hsa no log prob logprob[:, 1:]"
            # if not eval_mode and args.log_entropy:
            #     self.log_entropy(iteration)

            assert tokens.size(1) == _all_tokens_max_len, f"tokens size: {tokens.size(1)} " \
                                                        f"_all_tokens_max_len: {_all_tokens_max_len}"

            # TODO: 参数确认，这里会报错
            loss_mask = self.get_loss_mask(tokens, tokenizer.eos_token_id, prompt_sizes)

            data = dict(all_tokens=tokens, prompt_sizes=prompt_sizes, logprobs=all_log_probs, loss_mask=loss_mask)
            self._send_data_to_next(**data)
            # data = {
            # # 这两个传递下去，其他的由reward自行计算？这里可能只用算一次，放reward要算好多次
            #     "all_tokens": tokens,    # 2, 480
            #     "no_padded_query_ids": no_padded_query_ids,    # 2, 256
            #     "logprobs": all_log_probs,    # 2, 479 已经softmax过了
            #     "loss_mask": loss_mask    # 2, 480
            # }

            # if torch.distributed.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()
            # import time
            # time.sleep(1000)
            return data

    def aggregate_loss_func(self, inputs, losses):    # [b, s]
        from alignment.rlhf.models.constants_ppo import select_actions_from_right_padded
        from megatron.utils import average_losses_across_data_parallel_group
        # losses = losses.float()
        # b = losses.size(0)
        # loss = torch.sum(losses.view(-1)) / b

        losses = losses.float()    # [b, response_size]

        old_rewards = inputs['action_rewards']    # [b, responses size]

        response_length = old_rewards.shape[1]
        # Note the tkoken logits to get loss is only the actions. query doesn't have loss.
        action_loss_mask = select_actions_from_right_padded(
            ts=inputs["all_token_loss_mask"],
            action_starts=inputs["action_starts"] - 1,
        # because align iwth logits index
            response_size=response_length,
            pad_value=0,
            dim=-1).contiguous()

        action_loss_mask = action_loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * action_loss_mask) / action_loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        # Reduce loss for logging.
        # self.stats["policy_loss"] = averaged_loss[0]
        return loss, {'policy lm loss': averaged_loss[0]}

    @TraceEventScope("train_actor")
    @PatchSEPDistributedEnv()
    def train_step(self, epoch, data_loader):
        # def forward_value(self, data_list, train_info):
        """Single training step."""
        if self.module is not None:
            data_iterator = self.get_data_iterator(epoch, data_loader)

            from megatron.training import train_step as megatron_train_step
            from megatron.core.enums import ModelType
            if not getattr(self, '_initialized', None):
                from alignment.rlhf.models.policy_trainer import AdaptiveKLController
                from megatron import get_args
                mega_args = get_args()
                self.kl_ctl = AdaptiveKLController(mega_args.init_kl_coef, mega_args.target, mega_args.horizon)
                self._initialized = True
                self.model_type = ModelType.encoder_or_decoder
                self.model, self.optimizer, self.opt_param_scheduler = self.setup_model_and_optimizer(
                    self.module, self.model_type)

            # data_iterator = iter(data_list)
            loss, skipped_iter, grad_norm, num_zeros_in_grad = megatron_train_step(self._forward_step, data_iterator,
                                                                                   self.model, self.optimizer,
                                                                                   self.opt_param_scheduler)
            if loss is not None:            
                print(f'Loss for actor: {loss}, skipped_iter: {skipped_iter}, grad_norm: {grad_norm}, num_zeros_in_grad: {num_zeros_in_grad}')  
            

    @TraceEventScope("backward_actor")
    def backward(self, loss):
        super().backward(loss)

    @TraceEventScope("step_actor")
    def step(self):
        super().step()


class CriticDistributedSEPModuleEngine(DistributedSEPModuleEngine):
    def aggregate_loss_func(self, inputs, losses):    # [b, s]
        from alignment.rlhf.models.constants_ppo import select_actions_from_right_padded
        from megatron.utils import average_losses_across_data_parallel_group

        losses = losses.float()    # [b, response_size]

        # if torch.distributed.get_rank() == 0:
        #     import pdb
        #     pdb.set_trace()
        # import time
        # time.sleep(10000)
        old_rewards = inputs['action_rewards']    # [b, responses size]
        response_length = old_rewards.shape[1]
        # we want to mask logits which is the previous tokens of an action!!! so -1
        action_loss_mask = select_actions_from_right_padded(
            ts=inputs["all_token_loss_mask"],
            action_starts=inputs["action_starts"] - 1,
        # because align iwth logits index
            response_size=response_length,
            pad_value=0,
            dim=-1).contiguous()
        action_loss_mask = action_loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * action_loss_mask) / action_loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        # Reduce loss for logging.
        # stats_update = dict(
        #     value_loss=averaged_loss[0],

        # )
        # self.stats.update(stats_update)
        return loss, {'lm loss': averaged_loss[0]}

    def infer_loss_func(self, inputs, data, **kwargs):    # [b, s]
        from alignment.rlhf.models.constants_ppo import select_actions_from_right_padded

        data = data.float()    # [b, response_size]
        return data, {'data': data}

    @PatchSEPDistributedEnv()
    def _maybe_init_model(self):
        if not getattr(self, '_initialized', None):
            from megatron.core.enums import ModelType
            from megatron import get_args
            mega_args = get_args()

            self._initialized = True
            self.model_type = ModelType.encoder_or_decoder
            self.model, self.optimizer, self.opt_param_scheduler = self.setup_model_and_optimizer(
                self.module, self.model_type)

            logger.info(
                f'Get args: pipeline_model_parallel_size: {mega_args.pipeline_model_parallel_size}'
                f'tenosr_model_parallel_size: {mega_args.tensor_model_parallel_size}, cur_model: {self._current_model}'
            )

    @TraceEventScope("train_critic")
    @PatchSEPDistributedEnv()
    def train_step(self, epoch, data_loader):
        # def forward_value(self, data_list, train_info):
        """Single training step.
        dict_keys(['action_logprobs', 'action_values', 'action_rewards', 'all_token_ids_right_padded', 'loss_mask', 'action_start_indices', 'action_end_indices'])
        """
        if self.module is not None:
            print(f'rank is {torch.distributed.get_rank()}')
            # if torch.distributed.get_rank() == 5:
            #     import pdb
            #     pdb.set_trace()
            data_iterator = self.get_data_iterator(epoch, data_loader)
            with torch.enable_grad():
                self._maybe_init_model()
            from megatron.training import train_step as megatron_train_step

            loss, skipped_iter, grad_norm, num_zeros_in_grad = megatron_train_step(self._forward_step, data_iterator,
                                                                                   self.model, self.optimizer,
                                                                                   self.opt_param_scheduler)
            if loss is not None:            
                print(f'Loss for critic: {loss}, skipped_iter: {skipped_iter}, grad_norm: {grad_norm}, num_zeros_in_grad: {num_zeros_in_grad}')
    def _forward_infer_step(self, batch_data, model):
        """Forward step."""
        # args = get_args()
        # timers = get_timers()

        # Get the batch.
        # timers('batch-generator').start()
        from functools import partial
        from alignment.rlhf.models.constants_ppo import pad_to_max_len, get_ltor_masks_and_position_ids
        # inputs = self.get_batch(batch_data)
        # timers('batch-generator').stop()

        inputs = batch_data
        # if torch.distributed.get_rank() == 0:
        #     import pdb
        #     pdb.set_trace()
        # import time
        # time.sleep(10000)

        all_token_attention_mask, all_token_position_ids = get_ltor_masks_and_position_ids(inputs[0])
        losses = model.forward(all_token_ids=inputs[0],
                               all_position_ids=all_token_position_ids,
                               all_token_attention_mask=all_token_attention_mask,
                               training_inputs=None)

        return losses, partial(self.infer_loss_func,
                               inputs)    # will call loss_func(loss_mask, output_tensor) to get loss

    @property
    def send_tmp_data(self):
        """values, [bs, seq_len, vocab_size]
        """
        if self._send_tmp_data is None:
            # 目前只传输单值，直接传输即可，不做concat
            # self._send_tmp_data = torch.empty()
            pass
        return self._send_tmp_data

    # for TRAIN/PRED共享，跑train的forward
    @TraceEventScope("forward_critic")
    @PatchSEPDistributedEnv()
    def forward_step(self, *other_non_tensor_data):
        if self.module is not None:
            input_data = self._recv_data_from_prev()
            seq = input_data['all_tokens']

            forward_scope = torch.no_grad() if self._is_pred_model else torch.enable_grad()

            if hasattr(self, 'pred_model'):
                with forward_scope:
                    self.pred_model._maybe_init_model()
                self.model = self.pred_model.model

            data_iterator = [seq]
            from megatron.core.pipeline_parallel import get_forward_backward_func
            forward_backward_func = get_forward_backward_func()
            collected_non_loss_data = forward_backward_func(forward_step_func=self._forward_infer_step,
                                                            data_iterator=data_iterator,
                                                            model=self.model,
                                                            timers=None,
                                                            forward_only=True,
                                                            num_microbatches=1,
                                                            collect_non_loss_data=True)
            data = dict(old_values=collected_non_loss_data[0][0])
            return self._send_data_to_next(**data)
            # return collected_non_loss_data[0][0]


class DistributedDeepSpeedSEPACEngine:
    def __init__(self, deep_speed_ac_engine: DeepSpeedACEngine, model_placement: ModelPlacement):
        actor_ranks = model_placement.get_actor_ranks()
        ref_ranks = model_placement.get_init_model_ranks()
        critic_ranks = model_placement.get_critic_ranks()
        reward_ranks = model_placement.get_reward_model_ranks()

        rank = dist_util.get_world_rank()
        print("rank is : {}".format(rank))

        # create_data_model_groups(actor_ranks, critic_ranks, ref_ranks, reward_ranks)

        # patch_deepspeed_groups_clone_world_group()

        rlhf_sep_config = global_context().runtime_conf.rlhf_sep_config

        global current_model
        current_model = "actor"
        actor_policy = SEPGroupPlacePolicy(rank,
                                           actor_ranks,
                                           current_model,
                                           model_sep_config=rlhf_sep_config.actor_sep_config)

        if rank in actor_ranks:
            print("pid : {}, patch actor ranks : {}".format(os.getpid(), actor_ranks))
            with PatchSEPDistributedEnv(current_model):
                module = deep_speed_ac_engine.init_actor()
                self.actor = ActorDistributedSEPModuleEngine(module, actor_policy, current_model)
        else:
            self.actor = ActorDistributedSEPModuleEngine(None, actor_policy, current_model)

        current_model = "critic"
        if hasattr(deep_speed_ac_engine, 'init_critic'):
            # for share_engine, no critic_model
            critic_policy = SEPGroupPlacePolicy(rank,
                                                critic_ranks,
                                                current_model,
                                                model_sep_config=rlhf_sep_config.critic_sep_config)
            if rank in critic_ranks and hasattr(deep_speed_ac_engine, 'init_critic'):
                print("pid : {}, patch critic ranks : {}".format(os.getpid(), critic_ranks))
                with PatchSEPDistributedEnv(current_model):
                    module = deep_speed_ac_engine.init_critic()
                    self.critic = CriticDistributedSEPModuleEngine(module, critic_policy, current_model)
            else:
                self.critic = CriticDistributedSEPModuleEngine(None, critic_policy, current_model)


        from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedACNoneShareSEPEngine
        if isinstance(deep_speed_ac_engine, (DeepSpeedACNoneShareSEPEngine)):
            pred_actor_ranks = model_placement.get_pred_actor_ranks()

            pred_critic_ranks = model_placement.get_pred_critic_ranks()

            pred_actors = []

            pred_actor_replicas = rlhf_sep_config.pred_actor_sep_config.replicas
            for replica_id in range(pred_actor_replicas):
                assert len(pred_actor_ranks) % pred_actor_replicas == 0    # 后续再mod上MP
                num_ranks_per_replicas = int(len(pred_actor_ranks) // pred_actor_replicas)

                cur_ranks = pred_actor_ranks[replica_id * num_ranks_per_replicas:(replica_id + 1) *
                                             num_ranks_per_replicas]
                current_model = f"pred_actor_{replica_id}"

                pred_actor_policy = SEPGroupPlacePolicy(rank,
                                                        cur_ranks,
                                                        current_model,
                                                        model_sep_config=rlhf_sep_config.pred_actor_sep_config)

                if rank in cur_ranks:
                    print("pid : {}, patch pred_actor ranks : {}".format(os.getpid(), cur_ranks))
                    with PatchSEPDistributedEnv(current_model):
                        module = deep_speed_ac_engine.init_pred_actor()
                        pred_actors.append(ActorDistributedSEPModuleEngine(module, pred_actor_policy, current_model))
                else:
                    pred_actors.append(ActorDistributedSEPModuleEngine(None, pred_actor_policy, current_model))
            self.pred_actor = DistModel(pred_actors)

            current_model = "pred_critic"
            if hasattr(deep_speed_ac_engine, 'init_pred_critic'):
                # for share_engine, no pred_critic_model
                pred_critic_policy = SEPGroupPlacePolicy(rank,
                                                         pred_critic_ranks,
                                                         current_model,
                                                         model_sep_config=rlhf_sep_config.pred_critic_sep_config)
                if rank in pred_critic_ranks and hasattr(deep_speed_ac_engine, 'init_pred_critic'):
                    print("pid : {}, patch pred_critic ranks : {}".format(os.getpid(), pred_critic_ranks))
                    with PatchSEPDistributedEnv(current_model):
                        module = deep_speed_ac_engine.init_pred_critic()
                        self.pred_critic = CriticDistributedSEPModuleEngine(module, pred_critic_policy, current_model)


                        # check一下
                        self.pred_critic.pred_model = self.critic
                        self.pred_critic._initialized = True
                else:
                                 self.pred_critic = CriticDistributedSEPModuleEngine(None, pred_critic_policy, current_model)

        current_model = "ref"
        ref_policy = SEPGroupPlacePolicy(rank,
                                         ref_ranks,
                                         current_model,
                                         model_sep_config=rlhf_sep_config.initial_sep_config)
        if rank in ref_ranks:
            print("pid : {}, patch ref ranks : {}".format(os.getpid(), ref_ranks))

            with PatchSEPDistributedEnv(current_model):
                module = deep_speed_ac_engine.init_ref(self.actor)
                self.ref = RefDistributedSEPModuleEngine(module, ref_policy, current_model)
        else:
            self.ref = RefDistributedSEPModuleEngine(None, ref_policy, current_model)

        current_model = "reward"
        reward_policy = SEPGroupPlacePolicy(rank,
                                            reward_ranks,
                                            current_model,
                                            model_sep_config=rlhf_sep_config.reward_sep_config)
        if rank in reward_ranks:
            print("pid : {}, patch reward ranks : {}".format(os.getpid(), reward_ranks))

            with PatchSEPDistributedEnv(current_model):
                module = deep_speed_ac_engine.init_reward()
                self.reward = RewardDistributedSEPModuleEngine(module, reward_policy, current_model)
        else:
            self.reward = RewardDistributedSEPModuleEngine(None, reward_policy, current_model)
        current_model = None

        # actor_ema not supported yet!
        self.actor_ema = None
        self._deep_speed_ac_engine = deep_speed_ac_engine

        assert self.pred_actor is not None
        assert self.pred_critic is not None

        self.pred_actor.id = 1 * 1000
        self.ref.id = 2 * 1000
        self.pred_critic.id = 3 * 1000
        self.reward.id = 4 * 1000
        self.actor.id = 5 * 1000
        self.critic.id = 6 * 1000

        self.pred_actor.send_dst_models = [self.pred_critic, self.ref, self.reward]

        # for item in self.pred_actor.send_dst_models:
        #     __DIST_DATA__[f'{id(self.pred_actor)}_{id(item)}'] = {
        #         'all_tokens': torch.randint(100, 50000, [2, 480]).cuda()
        #     }

        self.ref.recv_src_models = [(cycle(self.pred_actor.replicas),
                                     ['all_tokens', 'prompt_sizes', 'logprobs', 'loss_mask'])]
        self.ref.send_dst_models = [self.reward]

        self.pred_critic.recv_src_models = [(cycle(self.pred_actor.replicas),
                                             ['all_tokens', 'prompt_sizes', 'logprobs', 'loss_mask'])]
        self.pred_critic.send_dst_models = [self.reward]

        self.reward.recv_src_models = [(cycle(self.pred_actor.replicas),
                                        ['all_tokens', 'prompt_sizes', 'logprobs', 'loss_mask']),
                                       (self.ref, ['ref_logprobs']), (self.pred_critic, ['old_values'])]
        self.reward.send_dst_models = [self.actor, self.critic]

        for model in [self.pred_actor, self.ref, self.reward, self.pred_critic]:
            build_send_recv_data_group(model)
        logger.info(f'rank: {torch.distributed.get_rank()} {SEND_RECV_DATA_GROUPS}')
        # import pdb
        # pdb.set_trace()
        self.actor.recv_src_models = [(self.reward, [
            'all_token_ids_right_padded', 'loss_mask', 'action_start_indices', 'action_end_indices', 'action_logprobs',
            'action_values', 'action_rewards'
        ])]

        self.critic.recv_src_models = [(self.reward, [
            'all_token_ids_right_padded', 'loss_mask', 'action_start_indices', 'action_end_indices', 'action_logprobs',
            'action_values', 'action_rewards'
        ])]
