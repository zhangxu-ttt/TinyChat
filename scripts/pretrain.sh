export CUDA_VISIBLE_DEVICES=4,5,7
CUDA_VISIBLE_DEVICES=4,5,7 torchrun --nproc_per_node=3 train.py --config_path config/pretrain_config.yaml --task_type pretrain

torchrun train.py --config_path config/pretrain.yaml --task_type pretrain

#python train.py --config_path config/pretrain.yaml --task_type pretrain --local_rank 0


