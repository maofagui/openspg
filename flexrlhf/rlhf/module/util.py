from alignment.rlhf.module.ac_share_module import ACShareDeepSpeedPPO2Module, ACShareDeepSpeedModule
from alignment.rlhf.module.ac_none_share_module import ACNoneShareDeepSpeedModule


def assign_module(rl_algo, ac_share=True):
    rl_algo = rl_algo.lower()
    if rl_algo == 'ppo' or rl_algo == 'ppo2':
        return ACShareDeepSpeedPPO2Module if ac_share else ACNoneShareDeepSpeedModule
    else:
        return ACShareDeepSpeedModule if ac_share else ACNoneShareDeepSpeedModule
