# coding: utf-8

from alignment.api.rlhf.model_provider import ACShareModelProvider, ACNoneShareModelProvider, ModelProvider
from alignment.rlhf.model.model_utils import create_hf_model


class DefaultACNoneShareModelProvider(ACNoneShareModelProvider):

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path=None, **kwargs):
        super(DefaultACNoneShareModelProvider, self).__int__(actor_model_name_or_path, critic_model_name_or_path)
        self.actor_model_name_or_path = actor_model_name_or_path
        self.critic_model_name_or_path = critic_model_name_or_path
        self.kwargs = kwargs

    def get_actor(self):

        actor_model = create_hf_model(role='actor', model_path=self.actor_model_name_or_path)
        return actor_model

    def get_critic(self):
        # Model
        critic_model = create_hf_model(role='critic', model_path=self.critic_model_name_or_path)
        return critic_model

    def get_reward_model(self):
        reward_model = create_hf_model(role='reward', model_path=self.critic_model_name_or_path)
        return reward_model

    def get_initial_model(self):
        ref_model = create_hf_model(role='ref', model_path=self.actor_model_name_or_path)
        return ref_model