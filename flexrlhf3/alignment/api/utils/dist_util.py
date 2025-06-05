# coding: utf-8
import os

# 在指定runtime='pytorch'时aistudio会设置WORLD_SIZE:节点总数，RANK:节点rank
# 而运行lightning任务时会将WORLD_SIZE设置为总gpu的数量，为避免混乱
# 使用Runner提交任务时会将WORLD_SIZE和RANK复制到APP_WORLD_SIZE和APP_RANK
_node_rank = os.environ.get("APP_RANK", None)
_world_size = os.environ.get("APP_WORLD_SIZE", None)


from alignment.app.util import logger
logger.info("APP_RANK is : {}".format(_node_rank))
logger.info("APP_WORLD_SIZE is : {}".format(_world_size))


def is_pytorch_runtime():
    return _node_rank is not None and _world_size is not None


def ps_size() -> int:
    return int(os.environ.get("COWORKER_SIZE", 0))


def worker_size() -> int:
    if ps_size() > 0:
        return get_node_num() - ps_size()
    else:
        return get_total_world_size()


def get_node_rank() -> int:
    """
    获取节点RANK
    获取环境变量中的APP_RANK不为空则返回否则返回0
    """
    return int(_node_rank) if _node_rank is not None else 0


def set_local_rank(local_rank=None):
    """
    设置APP_LOCAL_RANK环境变量用于OdpsIterableDataset数据分片
    local_rank相当于当前进程在本机的rank
    即local_rank为本机中gpu的rank
    例：在2台机器，每台机器2张卡的情况下
    node_0 - gpu_0的进程中需调用set_local_rank(0)
    node_0 - gpu_1的进程中需调用set_local_rank(1)
    node_1 - gpu_0的进程中需调用set_local_rank(0)
    node_1 - gpu_1的进程中需调用set_local_rank(1)
    """
    os.environ['APP_LOCAL_RANK'] = local_rank


def get_local_rank():
    """
    获取当前gpu/进程的rank
    获取local_rank优先级从高到低依次为：
    os.environ.get("APP_LOCAL_RANK", None)：用户调用set_local_rank设置
    os.environ.get("LOCAL_RANK", None)：lightning中多机多卡设置并使用
    torch.cuda.current_device()：torch中torch.cuda.set_device()设置
    """
    import torch
    env_local_rank = os.environ.get("LOCAL_RANK", None)
    env_APP_LOCAL_RANK = os.environ.get("APP_LOCAL_RANK", None)
    if torch.cuda.is_available():
        cuda_device_rank = torch.cuda.current_device()
    else:
        cuda_device_rank = None
    if env_APP_LOCAL_RANK is not None:
        local_rank = env_APP_LOCAL_RANK
    elif env_local_rank is not None:
        local_rank = env_local_rank
    elif cuda_device_rank is not None:
        local_rank = cuda_device_rank
    else:
        local_rank = 0
    local_rank = int(local_rank)
    return local_rank


def get_world_rank(local_rank=None, node_rank=None):
    """
    获取当前进程的rank
    使用gpu时返回node_rank * gpu_count + local_rank
    未使用gpu时返回node_rank + local_rank
    gpu_count：torch.cuda.device_count()获取
    local_rank：get_local_rank()获取，从0开始
    node_rank：get_node_rank()获取，从0开始
    """
    import torch
    if node_rank is None:
        node_rank = get_node_rank()
    pre_rank = 0
    if os.environ.get('NUM_PROCESSES', None) is not None:
        if node_rank >= 1:
            pre_rank = sum(eval(os.environ['NUM_PROCESSES'])[:node_rank])
    else:
        if torch.cuda.is_available():
            pre_rank = node_rank * torch.cuda.device_count()
        else:
            pre_rank = node_rank

    if local_rank is None:
        local_rank = get_local_rank()
    local_rank = int(local_rank)

    return pre_rank + local_rank


def get_node_num() -> int:
    """
    获取节点数量
    获取环境变量中的APP_WORLD_SIZE不为空则返回否则返回1
    """
    return int(_world_size) if _world_size is not None else 1


def get_total_world_size():
    """
    在执行端获取总gpu/进程数。
    使用gpu时返回node_num * gpu_num(torch.cuda.device_count)。
    未使用gpu时返回node_num。
    不要在提交端调用并传到执行端！直接在执行端调用！
    """
    import torch
    if os.environ.get('NUM_PROCESSES', None) is not None:
        return sum(eval(os.environ['NUM_PROCESSES']))

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        return ngpus_per_node * get_node_num()
    else:
        return get_node_num()


def patch_torch_elastic_agent(timeout=30*60):
    from torch.distributed.elastic.agent.server.api import WorkerSpec, _RoleInstanceInfo, SimpleElasticAgent
    import torch.distributed.elastic.utils.store as store_util
    from typing import List

    def patched_share_and_gather(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List:
        agent_role_info = _RoleInstanceInfo(
            spec.role, group_rank, spec.local_world_size
        )
        key_prefix = "torchelastic/role_info"
        agent_config_enc = agent_role_info.serialize()
        role_infos_bytes = store_util.synchronize(
            store, agent_config_enc, group_rank, group_world_size, key_prefix, timeout
        )
        role_infos = [
            _RoleInstanceInfo.deserialize(role_info_bytes)
            for role_info_bytes in role_infos_bytes
        ]
        return role_infos

    SimpleElasticAgent._share_and_gather = patched_share_and_gather
