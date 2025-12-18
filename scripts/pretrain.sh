export CUDA_VISIBLE_DEVICES=4,5,7

torchrun train.py --config_path config/pretrain.yaml --task_type pretrain

#python train.py --config_path config/pretrain.yaml --task_type pretrain --local_rank 0

