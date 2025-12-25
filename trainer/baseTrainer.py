import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import TransformerModel, ModelConfig
from utils import is_main_process, print_rank0


class BaseTrainer(ABC):

    def __init__(self,
                 model,
                 tokenizer,
                 config,
                 local_rank,
                 device
                 ):
        self.local_rank = local_rank
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.config = config
        self.device = device

        self.tokenizer = tokenizer
        self.model = model
        self.dataloader = self.build_dataloader()

        if config['dtype'] == 'float16':
            self.dtype = torch.float16
        elif config['dtype'] == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        lr = config['trainer']['learning_rate']

        self.gradient_accumulation_steps = config['trainer']['gradient_accumulation_steps']
        self.max_grad_norm = config['trainer']['max_grad_norm']

        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == torch.float16))
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(lr),
            weight_decay=config['trainer']['weight_decay'],
            fused=True  # 使用fused kernel加速
        )

        self.logging_steps = config['trainer']['logging_steps']
        self.save_steps = config['trainer'].get('save_steps', None)
        self.output_dir = Path(self.config['output']['output_dir'])

        self.start_epoch = 0
        self.resume_from_checkpoint = self.config['output'].get('resume_from_checkpoint', None)
        self.continue_train = self.config['output'].get('continue_train', False)
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)

        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = None
        if is_main_process() and self.config['wandb']['enabled']:
            self.init_wandb()

        self.global_step = 0
        self.epoch = 0

        self.num_epochs = config['trainer']['num_epochs']
        
        # 性能统计
        self.enable_timing = config['trainer'].get('enable_timing', True)

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    def init_wandb(self):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['run_name'],
                entity=self.config['wandb'].get('entity'),
                config=self.config,
                resume='allow'
            )
            print_rank0("WandB已初始化")
        except ImportError:
            print_rank0("警告: wandb未安装，跳过WandB日志记录")
            self.config['wandb']['enabled'] = False

    @contextmanager
    def create_progress_bar(self, dataloader, epoch: int, total_epochs: int):
        """创建进度条上下文管理器"""
        if is_main_process():
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                total=len(dataloader),
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            try:
                yield pbar
            finally:
                pbar.close()
        else:
            yield dataloader

    def update_progress_bar(self, pbar, metrics: dict):
        """更新进度条显示"""
        if not is_main_process() or not hasattr(pbar, 'set_postfix'):
            return

        postfix = {
            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            'step': self.global_step
        }

        pbar.set_postfix(postfix | metrics)

    def train(self):
        print_rank0("=" * 80)
        print_rank0("开始训练")
        print_rank0("=" * 80)

        self.model.train()
        metrics = {}

        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            self.dataloader.sampler.set_epoch(epoch)

            with self.create_progress_bar(self.dataloader, epoch, self.num_epochs) as pbar:
                t_dataload_start = time.time()

                for idx, batch in enumerate(pbar):
                    # 数据加载计时
                    if self.enable_timing and idx > 0:
                        metrics['time/dataload_ms'] = (time.time() - t_dataload_start) * 1000
                    
                    if self.enable_timing:
                        torch.cuda.synchronize()
                        t_start = time.time()
                    
                    # 前向传播
                    if self.enable_timing:
                        t0 = time.time()
                    
                    with torch.autocast(device_type='cuda', dtype=self.dtype):
                        step_metrics = self.train_step(batch)
                        loss = step_metrics['loss']
                        metrics.update(step_metrics)
                    
                    if self.enable_timing:
                        torch.cuda.synchronize()
                        metrics['time/forward_ms'] = (time.time() - t0) * 1000

                    # 反向传播
                    if self.enable_timing:
                        t0 = time.time()
                    
                    self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
                    
                    if self.enable_timing:
                        torch.cuda.synchronize()
                        metrics['time/backward_ms'] = (time.time() - t0) * 1000

                    # 累计梯度
                    if self.enable_timing:
                        t0 = time.time()
                    
                    if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    
                    if self.enable_timing:
                        torch.cuda.synchronize()
                        metrics['time/optimizer_ms'] = (time.time() - t0) * 1000
                        metrics['time/total_ms'] = (time.time() - t_start) * 1000

                        
                    if (self.global_step + 1) % self.logging_steps == 0:
                        self.log_metrics(metrics)

                    # 定期保存模型
                    if (self.global_step + 1) % self.save_steps == 0:
                        self.save_checkpoint(tag=f"checkpoint-{self.global_step}")

                    self.update_progress_bar(pbar, metrics)
                    self.global_step += 1

                    if self.enable_timing:
                        t_dataload_start = time.time()

                if is_main_process():
                    self.save_checkpoint(tag=f"epoch-{self.epoch + 1}")

        return self.model

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> dict:
        """
        单个训练步骤 - 子类必须实现
        
        Args:
            batch: 批次数据
        
        Returns:
            metrics: 包含损失和其他指标的字典
        """
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, Any]):
        if not is_main_process():
            return

        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.global_step)

    def save_checkpoint(self, tag: str):
        if not is_main_process():
            return

        # 获取实际的模型（处理 DDP 包装的情况）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),  # 必须保存 Scaler 状态
            "epoch": self.epoch
        }, f"{self.output_dir}/{tag}.pth")

    def load_checkpoint(self, tag: str):
        checkpoint = torch.load(f"{self.resume_from_checkpoint}", map_location=f"cuda:{self.local_rank}")
        
        # 获取实际的模型（处理 DDP 包装的情况）
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint["model"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])  # 加载 Scaler 状态

        if self.continue_train:
            # checkpoint["epoch"] 表示已完成的epoch，所以继续训练应该从下一个epoch开始
            self.start_epoch = checkpoint["epoch"] + 1
            print_rank0(f"从 checkpoint 恢复训练，将从 epoch {self.start_epoch} 开始")
        else:
            print_rank0(f"加载 checkpoint 权重，从 epoch 0 开始训练")


