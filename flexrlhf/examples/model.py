import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from alignment.api.rlhf.model_provider import ACShareModelProvider
from alignment.rlhf.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from alignment.rlhf.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from alignment.rlhf.model.model_utils import create_hf_model, create_critic_model
from alignment.rlhf.utils.utils import get_optimizer_grouped_parameters


class CustomerChatModelProvider(ACShareModelProvider):

    def __init__(self, initial_model_path,
                       reward_model_path,
                       num_total_iters = 1000):
        print("DefaultChatModelProvider init")

        tokenizer = AutoTokenizer.from_pretrained(initial_model_path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token

        super(CustomerChatModelProvider, self).__int__(initial_model_path, reward_model_path)
        self.initial_model_path = initial_model_path
        self.reward_model_path = reward_model_path
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        # import pdb
        # pdb.set_trace()

    def get_actor(self, ds_config):
        print("CustomerChatModelProvider get_actor")
        
    #     ds_config["flops_profiler"] =  {
    #                                 "enabled": True,
    # "profile_step": 1,
    # "module_depth": -1,
    # "top_modules": 1,
    # "detailed": True,
    # "output_file": None
    # }
        actor_model = create_hf_model(model_class=AutoModelForCausalLM, model_name_or_path=self.initial_model_path, tokenizer=self.tokenizer,ds_config=ds_config)
        # import pdb
        # pdb.set_trace()
        return actor_model


    def get_reward_model(self, ds_config):
        print("CustomerChatModelProvider get_reward_model")
        reward_model = create_critic_model(
            model_name_or_path=self.reward_model_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            num_padding_at_beginning=1,
            rlhf_training=True)
        
        return reward_model

    def get_initial_model(self, ds_config):
        # raise ValueError('fdsfds')
        print("CustomerChatModelProvider get_initial_model")
        ref_model = create_hf_model(AutoModelForCausalLM,
                                    self.initial_model_path, self.tokenizer,
                                    ds_config)
        # if torch.distributed.get_rank() == 0:
        #     import pdb
        #     pdb.set_trace()                      
        # import time
        # time.sleep(1000000)                                                                                                                      
        
        return ref_model
        
    def get_ema(self, ds_config):
        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          self.initial_model_path,
                                          self.tokenizer, ds_config)
        return actor_model_ema

