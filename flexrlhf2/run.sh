  torchrun  --nnodes 1 --node_rank 0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=23456 \
    alignment/examples/app_main.py \
    --config_path alignment/non_share.yaml