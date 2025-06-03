from alignment.rlhf.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from alignment.rlhf.model.modeling_ppo import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, \
    AutoModelForCausalLMWithHydraValueHead, AutoModelForSeq2SeqLMWithHydraValueHead
from alignment.rlhf.model.modeling_ppo_glm import AutoModelForGLMWithValueHead, AutoModelForGLMWithHydraValueHead
from alignment.rlhf.model.model_utils import freeze_bottom_seq2seq_layers, freeze_bottom_causal_layers, \
    freeze_head_parameters
from alignment.rlhf.model.modeling_ppo_glm import freeze_bottom_causal_layers as glm_freeze_bottom_causal_layers


class ModelDecoration(object):

    def get_value_head(self, model_arch_type, model):
        if model_arch_type == 'seq2seq':
            model = AutoModelForSeq2SeqLMWithValueHead(model)
        elif model_arch_type == 'glm':
            model = AutoModelForGLMWithValueHead(model)
        else:
            model = AutoModelForCausalLMWithValueHead(model)
        return model

    def get_value_hydra_head(self, model_arch_type, model, num_layers_unfrozen):
        if model_arch_type == 'seq2seq':
            model = AutoModelForSeq2SeqLMWithHydraValueHead(
                model, num_layers_unfrozen=num_layers_unfrozen)
        elif model_arch_type == 'glm':
            model = AutoModelForGLMWithHydraValueHead(
                model, num_layers_unfrozen=num_layers_unfrozen)
        else:
            model = AutoModelForCausalLMWithHydraValueHead(
                model, num_layers_unfrozen=num_layers_unfrozen)
        return model

    def decorate_actor(self, actor_model, model_conf):
        lora_conf = model_conf.lora_conf
        frozen_layer_conf = model_conf.frozen_layer_conf
        if lora_conf and lora_conf.actor_lora_dim > 0:
            # LoRA 实现
            actor_model = convert_linear_layer_to_lora(
                actor_model, lora_conf.actor_lora_module_name,
                lora_conf.actor_lora_dim)
            if lora_conf.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(actor_model)

            if model_conf.need_value_head:
                actor_model = self.get_value_head(model_conf.model_arch_type, actor_model)

        elif frozen_layer_conf:
            # 非LoRA的实现，通过freeze layer来实现部分参数更新

            assert frozen_layer_conf.actor_num_layers_unfrozen > 0, "need to set num_layers_unfrozen"
            base_model = actor_model

            if model_conf.need_value_head:
                from alignment.rlhf.model.modeling_ppo import AutoModelForCausalLMWithHydraValueHead
                actor_model = self.get_value_hydra_head(model_conf.model_arch_type,
                    actor_model, num_layers_unfrozen=frozen_layer_conf.actor_num_layers_unfrozen)

            # actor init ref 共享，freeze掉共用freeze底座
            if model_conf.model_arch_type == "seq2seq":
                freeze_bottom_seq2seq_layers(base_model, frozen_layer_conf.actor_num_layers_unfrozen)
            elif model_conf.model_arch_type == 'glm':
                glm_freeze_bottom_causal_layers(base_model, frozen_layer_conf.actor_num_layers_unfrozen)
            else:
                freeze_bottom_causal_layers(base_model, frozen_layer_conf.actor_num_layers_unfrozen)
        else:
            if model_conf.need_value_head:
                actor_model = self.get_value_head(model_conf.model_arch_type, actor_model)
        # print("actor_model is : {}".format(actor_model))

        return actor_model

    def decorate_ema(self, actor_model_ema, model_conf):
        # LoRA
        lora_conf = model_conf.lora_conf
        if lora_conf and lora_conf.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, lora_conf.actor_lora_module_name,
                lora_conf.actor_lora_dim)
        return actor_model_ema

    def decorate_critic(self, critic_model, model_conf):
        # LoRA
        lora_conf = model_conf.lora_conf
        if lora_conf and lora_conf.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, lora_conf.critic_lora_module_name,
                lora_conf.critic_lora_dim)
            if lora_conf.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)
                critic_model = make_model_gradient_checkpointing_compatible(critic_model)
        return critic_model

    def decorate_ref(self, actor_model, model_conf):
        lora_conf = model_conf.lora_conf
        if lora_conf and lora_conf.critic_lora_dim > 0:
            # 此时ref和actor都是单独的模型，包含v_head
            pass
        else:
            # 此时ref和actor是同一个模型，额外在BranchModel上加一个v_head
            if model_conf.need_value_head:
                actor_model = self.get_value_head(model_conf.model_arch_type, actor_model)
                freeze_head_parameters(actor_model)

        return actor_model