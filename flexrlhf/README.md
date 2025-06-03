# FlexRLHF: A Flexible Placement and Parallelism Framework for Efficient RLHF Training
We present our implementation for the research paper: "FlexRLHF: A Flexible Placement and Parallelism Framework for Efficient RLHF Training." 
At present, we have released the majority of the FlexRLHF framework's source code. Due to the significant effort required to decouple the framework from our proprietary systems, we have not yet released the complete codebase. However, our team is diligently working to prepare the entire FlexRLHF framework for open-source release.

# 🏃 How to train RLHF in FlexRLHF framework
## Run Interleaving Strategy
```shell
cd alignment/examples/interleaving
export PLACEMENT_STRATEGY=interleaving; python main.py
```

## Run Separation Strategy
```shell
cd alignment/examples/sepearation
export PLACEMENT_STRATEGY=separation; python main.py
```

## Run Flattening Strategy
```shell
cd alignment/examples/sepearation
export PLACEMENT_STRATEGY=flattening; python main.py
```
# 🌲Code structure
```
|____app
| |____util
| | |____py_utils.py
| | |____logger.py
| |______init__.py
|____util
| |______init__.py
| |____global_vars.py
|______init__.py
|____README.md
|____rlhf
| |____config.py
| |____distributed
| | |____distributed_rlhf_sep_engine.py # for separation strategy
| | |____distributed_rlhf_engine.py # for interleaving strategy 
| | |____model_placement.py # model placement abstraction
| | |______init__.py
| |____module
| | |____sft_module.py
| | |____rlhf_module.py
| | |____util.py
| | |______init__.py
| | |____ac_none_share_module.py # ac-noshare model structure
| | |____ac_share_module.py # ac-share model structure
| | |____rlhf_sep_module.py
| | |____lora.py
| |____utils
| | |____perf.py
| | |____ds_utils.py
| | |____constants.py
| | |______init__.py
| | |____save_utils.py
| | |____utils.py
| |____hooks
| | |____checkpoint_saver_hook.py
| | |____profile_train_hook.py
| | |____logging_hook.py
| | |______init__.py
| | |____rlhf_train_hook.py
| | |____evaluation_hook.py
| |____model
| | |____model_utils.py
| | |____modeling_ppo.py
| | |____model_decoration.py
| | |______init__.py
| | |____modeling_ppo_glm.py
| | |____reward_model.py
| | |____default_model_impl.py
| |____data
| | |____default_data_impl.py
| | |____raw_datasets.py
| | |______init__.py
| | |____data_utils.py
| |____trainner
| | |____app_sft_trainner.py
| | |______init__.py
| | |____app_ds_rlhf_engine.py
| | |____mixed_model.py
| | |____exec_engine.py
|____examples
| |____requirements.txt
| |____model.py
| |____main.py # user interface
| |____nodes.yaml
|____model
| |____model_utils.py
| |____modeling_ppo.py
| |____model_decoration.py
| |______init__.py
| |____modeling_ppo_glm.py
| |____default_model_impl.py
|____api
| |______init__.py
| |____utils
| | |______init__.py
| | |____dist_util.py
| |____rlhf
| | |____config.py
| | |______init__.py
| | |____rlhf_trainner.py
| | |____model_provider.py
```