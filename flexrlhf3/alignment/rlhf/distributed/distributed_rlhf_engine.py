# coding: utf-8

from collections import deque
import functools
import os
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed import distributed_c10d
import torch.distributed as dist
from alignment.util.global_vars import global_context
from alignment.rlhf.distributed.model_placement import ModelPlacement
from alignment.rlhf.trainner.app_ds_rlhf_engine import ACEngine
from alignment.api.utils import dist_util
from alignment.app.util import logger

GLOBAL_MP_GROUP = None
EMPTY_TENSOR_LENGTH = int(1e6)
ACTOR_WORLD_GROUP = None
CRITIC_WORLD_GROUP = None
REF_WORLD_GROUP = None
REWARD_WORLD_GROUP = None

SCATTER_QUEUE = deque()

_WORLD_GROUP_DICT = {}

current_model = None


def data_group_ranks_world_info(ranks):
    ranks = sorted(ranks)
    ranks_string = ranks.__str__()
    return ranks_string




class PatchDistData(object):
    def __init__(self):
        self._ori_torch_get_rank = distributed_c10d.get_rank

    def __enter__(self):
        def patch_get_rank(*args, **kwargs):
            return self._ori_torch_get_rank()

        distributed_c10d.get_rank = patch_get_rank

    def __exit__(self, exc_type, exc_value, traceback):
        distributed_c10d.get_rank = self._ori_torch_get_rank

    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func




class GroupPlacePolicy:
    def __init__(self, my_rank, dst_model_ranks, dist_groups=None, group_ranks=None):
        from alignment.api.utils.dist_util import get_total_world_size
        self._total_world_size = get_total_world_size()
        if len(dst_model_ranks) == self._total_world_size:
            logger.info(f'Get dst_model_ranks: {dst_model_ranks} equals to world_size')

        # other policy?
        self._dst_model_ranks = sorted(dst_model_ranks)
        self._other_model_ranks = list(sorted(set(range(self._total_world_size)) - set(self._dst_model_ranks)))

        # 加下dst_model_ranks至少1
        self._group_size = int(len(self._other_model_ranks) // len(self._dst_model_ranks))
        if len(self._other_model_ranks) % len(self._dst_model_ranks):
            logger.warning(f'Other ranks not divisible by dst_model_ranks')
            self._group_size += 1

        self._dist_groups = dist_groups

        self._owner_rank_id = None
        self._is_owner = my_rank in self._dst_model_ranks
        self._my_rank = my_rank
        self._group_ranks = group_ranks

    @property
    def owner_rank_id(self):
        if self._owner_rank_id is None:
            if self.is_owner:
                self._owner_rank_id = self._my_rank
            else:
                assert self._my_rank in self._other_model_ranks
                other_rank_id = self._other_model_ranks.index(self._my_rank)
                self._owner_rank_id = self._dst_model_ranks[int(other_rank_id // self._group_size)]

        return self._owner_rank_id

    @property
    def device_id(self):
        return torch.cuda.current_device()

    @property
    def owner_group_id(self):
        return self.group_ranks.index(self.owner_rank_id)

    @property
    def is_owner(self):
        return self._is_owner

    @property
    def dist_groups(self):
        if self._dist_groups is None:
            print(self.group_ranks)
        return self._dist_groups

    @property
    def group_ranks(self):
        if self._group_ranks is None:
            for i in range(len(self._dst_model_ranks)):
                owner_rank = self._dst_model_ranks[i]

                ranks = [owner_rank]
                ranks.extend([
                    self._other_model_ranks[r]
                    for r in range(i * self._group_size, min((i + 1) * self._group_size, len(self._other_model_ranks)))
                ])
                ranks = list(sorted(ranks))

                # 是否可复用？
                dist_groups = torch.distributed.new_group(ranks=ranks)
                if self.owner_rank_id in ranks:
                    self._group_ranks = ranks
                    self._dist_groups = dist_groups

            print(f'group_ranks {self._group_ranks}'
                  f'owner_rank_id: {self.owner_rank_id}, device_id {self.device_id}')

        return self._group_ranks


class DistributedModuleEngine(nn.Module):
    def __init__(self, module, place_policy, current_model):

        super(DistributedModuleEngine, self).__init__()
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

    def _all_to_all_single(self, input_data, run_async=False, group=None):
        if group is None:
            group = self._place_policy.dist_groups        
        if run_async:
            SCATTER_QUEUE.append(MagicMock())
        logger.info(f'Alltoall data in ranks: {self._place_policy.group_ranks}')
        return input_data        

    @property
    def current_stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def all_gather_device_data_list(self, tensor_list, run_async=False, group=None):
        logger.info(f'Allgather device data in ranks: {self._place_policy.group_ranks}')        
        res_data = [tensor_list for _ in range(len(self._place_policy.group_ranks))]
        if run_async:
            async_op = MagicMock()
            return res_data, async_op
        return res_data
    
    def _scatter_device_data(self, all_data_list, dtype=torch.float16):
        # dtype类型后续需传递
        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            return all_data_list[0]

        data_list = [None]
        # scatter_object_list使用group判断src
        torch.distributed.scatter_object_list(data_list,
                                              all_data_list,
                                              src=self._place_policy.owner_rank_id,
                                              group=self._place_policy.dist_groups)
        # logger.info(f'Get data {data_list}, input is {all_data_list}')
        del all_data_list
        return data_list[0].to(f'cuda:{self._place_policy.device_id}')

    def _forward(self, forward_func, input_data):
        res_data_tensor, async_op = self.all_gather_device_data_list(input_data, run_async=True)

        self._async_op_queue.append((self._in_generate, async_op))

        res_data = []

        def _wait_queued_data():
            for i in range(len(self._async_op_queue) - 1, -1, -1):
                if not self._in_generate and self._async_op_queue[i][0]:
                    continue
                self._async_op_queue[i][1].wait()
                del self._async_op_queue[i]

        if self.module is not None:
            for idx in range(len(self._place_policy.group_ranks)):
                if idx > 0:
                    cur_idx = 0 if idx == self._place_policy.owner_group_id else idx
                    input_data = res_data_tensor[cur_idx]
                    
                res_data.append(forward_func(input_data))


                if idx == 0:
                    _wait_queued_data()
                    if self._in_generate and self._next_data is not None:
                        assert not res_data_tensor
                        res_data_tensor = [self._next_data]
            res_data[0], res_data[self._place_policy.owner_group_id] = res_data[
                self._place_policy.owner_group_id], res_data[0]
            # del self._gathered_kwargs, self._gathered_inputs
            del res_data_tensor
        else:
            _wait_queued_data()

        if self._in_generate:
            pass

        return res_data

    def forward(self, *inputs, **kwargs):
        if self.module is not None:
            if self._is_training:
                return self.module(*inputs, **kwargs)
            else:
                # engine默认调用了sync。 for zero3 是否sync?
                return self.module.module(*inputs, **kwargs)
        else:
            return None

    
    def generate(self, *inputs, **kwargs):
        if self.module is not None:
            return self.module.module.generate(*inputs, **kwargs)
        else:
            return None

    def forward_value(self, *inputs, **kwargs):
        if self.module is not None:
            return self.module.forward_value(*inputs, **kwargs)
        else:
            return None

    
    def backward(self, loss):
        if self.module is not None:
            return self.module.backward(loss)
        else:
            pass

    def step(self):
        if self.module is not None:
            return self.module.step()
        else:
            pass

    def eval(self):
        if self.module is not None:
            self.module.eval()

    def train(self):
        if self.module is not None:
            self.module.train()


class RewardDistributedModuleEngine(DistributedModuleEngine):
    def forward_value(self, input_data, **kwargs):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.compute_log_prob(input_data)
        ret_data = self._forward(self.module.forward_value if self.module else None, input_data)

        if kwargs.get('disable_scatter', False):
            return ret_data
        else:
            return self._scatter_device_data_with_shape(ret_data, run_async=True)


class RefDistributedModuleEngine(DistributedModuleEngine):
    def compute_log_prob(self, input_data, **kwargs):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.compute_log_prob(input_data)

        ret_data = self._forward(self.module.compute_log_prob if self.module else None, input_data)

        if kwargs.get('disable_scatter', False):
            return ret_data
        else:
            return self._scatter_device_data_with_shape(ret_data, run_async=True)   


class ActorDistributedModuleEngine(DistributedModuleEngine):
    def generate(self, input_data):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.generate(input_data)        
        self._in_generate = True
        res_data = self._forward(self.module.generate if self.module else None, input_data)
        self._in_generate = False
        value = self._scatter_device_data(res_data, dtype=torch.int64)
        # print("pid : {}, value {}".format(os.getpid(), value))

        return value

    def compute_log_prob(self, input_data):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.compute_log_prob(input_data)
        last_forward_value = True
        res_data = self._forward(self.module.compute_log_prob if self.module else None, input_data)
        res_data = self._scatter_device_data(res_data)
        return res_data

    def update_actor(self, input_data):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.update_actor(input_data)
        last_forward_value = True
        res_data = self._forward(self.module.update_actor if self.module else None, input_data)
        res_data = self._scatter_device_data(res_data)
        return res_data

    def backward(self, loss):
        super().backward(loss)

    def step(self):
        super().step()


class CriticDistributedModuleEngine(DistributedModuleEngine):
    def forward_value(self, input_data):
        if len(self._place_policy.group_ranks) == 1:
            # 训练module TrainModel patch了下
            return super().forward_value(input_data)
        return_value_only = kwargs.get('return_value_only', True)
        assert return_value_only, "Currently, return_value_only must be True"

        res_data = self._forward(self.module.forward_value if self.module else None, input_data)

        value = self._scatter_device_data(res_data)

        return value

    def update_critic(self, input_data):
        if len(self._place_policy.group_ranks) == 1:
            return self.module.update_critic(input_data)
        last_forward_value = True
        res_data = self._forward(self.module.update_critic if self.module else None, input_data)
        res_data = self._scatter_device_data(res_data)
        return res_data

def create_data_model_groups(actor_ranks, critic_ranks, ref_ranks, reward_ranks):
    global ACTOR_WORLD_GROUP
    global CRITIC_WORLD_GROUP
    global REF_WORLD_GROUP
    global REWARD_WORLD_GROUP
    ACTOR_WORLD_GROUP = create_data_group(actor_ranks)
    CRITIC_WORLD_GROUP = create_data_group(critic_ranks)
    REF_WORLD_GROUP = create_data_group(ref_ranks)
    REWARD_WORLD_GROUP = create_data_group(reward_ranks)


def create_data_group(ranks):
    if len(ranks) == 0:
        return None
    ranks_str = data_group_ranks_world_info(ranks)
    print(f'ranks_str: {ranks_str}')
    global _WORLD_GROUP_DICT
    if _WORLD_GROUP_DICT.get(ranks_str, None) is None:
        # If not cloned already, clone the world group
        _WORLD_GROUP = dist.new_group(ranks=ranks)
        _WORLD_GROUP_DICT[ranks_str] = _WORLD_GROUP
    return _WORLD_GROUP_DICT[ranks_str]


class DistributedACEngine:
    def __init__(self, deep_speed_ac_engine: ACEngine, model_placement: ModelPlacement):
        self.model_placement = model_placement
        actor_ranks = model_placement.get_actor_ranks()
        ref_ranks = model_placement.get_init_model_ranks()
        critic_ranks = model_placement.get_critic_ranks()
        reward_ranks = model_placement.get_reward_model_ranks()

        rank = dist_util.get_world_rank()
        print("rank is : {}".format(rank))

        create_data_model_groups(actor_ranks, critic_ranks, ref_ranks, reward_ranks)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

                

        global current_model
        actor_policy = GroupPlacePolicy(rank, actor_ranks)
        actor_policy.group_ranks

        current_model = "actor"
        if rank in actor_ranks:
            print("pid : {}, patch actor ranks : {}".format(os.getpid(), actor_ranks))
            module = deep_speed_ac_engine.init_actor()
            self.actor = ActorDistributedModuleEngine(module, actor_policy, current_model)
        else:
            self.actor = ActorDistributedModuleEngine(None, actor_policy, current_model)

        current_model = "critic"
        if hasattr(deep_speed_ac_engine, 'init_critic'):
            # for share_engine, no critic_model
            critic_policy = GroupPlacePolicy(rank, critic_ranks)
            critic_policy.group_ranks
            if rank in critic_ranks and hasattr(deep_speed_ac_engine, 'init_critic'):
                print("pid : {}, patch critic ranks : {}".format(os.getpid(), critic_ranks))
                module = deep_speed_ac_engine.init_critic()
                self.critic = CriticDistributedModuleEngine(module, critic_policy, current_model)
            else:
                self.critic = CriticDistributedModuleEngine(None, critic_policy, current_model)

        ref_policy = GroupPlacePolicy(rank, ref_ranks)
        ref_policy.group_ranks
        current_model = "ref"
        if rank in ref_ranks:
            print("pid : {}, patch ref ranks : {}".format(os.getpid(), ref_ranks))

            module = deep_speed_ac_engine.init_ref(self.actor)
            self.ref = RefDistributedModuleEngine(module, ref_policy, current_model)
        else:
            self.ref = RefDistributedModuleEngine(None, ref_policy, current_model)

        current_model = "reward"

        dist_groups, group_ranks = None, None
        from alignment.api.rlhf.config import InitialRewardSeparate
        if isinstance(model_placement.placement, InitialRewardSeparate):
            dist_groups, group_ranks = ref_policy.dist_groups, ref_policy.group_ranks
        reward_policy = GroupPlacePolicy(rank, reward_ranks, dist_groups=dist_groups, group_ranks=group_ranks)
        if rank in reward_ranks:
            print("pid : {}, patch reward ranks : {}".format(os.getpid(), reward_ranks))

            module = deep_speed_ac_engine.init_reward()
            self.reward = RewardDistributedModuleEngine(module, reward_policy, current_model)
        else:
            self.reward = RewardDistributedModuleEngine(None, reward_policy, current_model)
        current_model = None

        # actor_ema not supported yet!
        self.actor_ema = None
        self._deep_speed_ac_engine = deep_speed_ac_engine