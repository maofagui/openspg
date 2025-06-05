torchrun  --nnodes 1 --node_rank 0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=23456 \
    alignment/examples/app_main.py \
    --config_path interleaving.yaml