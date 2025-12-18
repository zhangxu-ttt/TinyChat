import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import deepspeed
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

        self.tokenizer = tokenizer
        self.model = model
        self.dataloader = self.build_dataloader()

        dtype = config['dtype']
        lr = config['training']['learning_rate']

        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        self.max_grad_norm = config['training']['max_grad_norm']

        self.scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.logging_steps = config['training']['logging_steps']
        self.save_steps = config['training']['save_steps']
        self.output_dir = Path(self.config['output']['output_dir'])
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = None
        if is_main_process() and self.config['wandb']['enabled']:
            self.init_wandb()

        self.global_step = 0
        self.epoch = 0
        self.num_epochs = config['training']['num_epochs']

    
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

        postfix.update(metrics)

        # 添加额外指标
        for metric_name in self.get_metric_names():
            value = metrics.get_avg_extra_metric(metric_name)
            postfix[metric_name] = f'{value:.4f}'

        pbar.set_postfix(postfix)

    def train(self):
        print_rank0("=" * 80)
        print_rank0("开始训练")
        print_rank0("=" * 80)

        self.model.train()
        metrics = self.build_metrics()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.dataloader.sampler.set_epoch(epoch)

            with self.create_progress_bar(self.dataloader, epoch, self.num_epochs) as pbar:
                for idx, batch in enumerate(pbar):
                    step_metrics = self.train_step(batch)
                    metrics.update(step_metrics)

                    # 累计梯度
                    if (self.global_step+1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    # 定期记录日志到wandb
                    if (self.global_step+1) % self.logging_steps == 0:
                        self.log_metrics(metrics)

                    # 定期保存模型
                    if (self.global_step+1) % self.save_steps == 0:
                        self.save_checkpoint(tag=f"checkpoint-{self.global_step}")

                    self.update_progress_bar(pbar, metrics)
                    self.global_step += 1


                self.save_checkpoint(tag=f"epoch-{self.epoch}")
                self.epoch += 1



    @abstractmethod
    def build_metrics(self) -> Dict[str, Any]:
        """
        构建训练指标字典 - 子类可以重写此方法

        Returns:
            metrics: 指标字典
        """
        raise NotImplementedError


    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        单个训练步骤 - 子类必须实现
        
        Args:
            batch: 批次数据
        
        Returns:
            loss: 损失值
        """
        raise NotImplementedError
    
    
    def log_metrics(self, metrics: Dict[str, Any]):
        if is_main_process():
            if self.wandb_run:
                self.wandb_run.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, tag: str):
        ...
    
    def load_checkpoint(self, checkpoint_path: str):
        ...
    
    def manage_checkpoints(self):
        """管理检查点数量，删除旧的检查点"""
        if not is_main_process():
            return

        save_total_limit = self.config['output'].get('save_total_limit', None)
        if save_total_limit is None or save_total_limit <= 0:
            return
        
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
            key=lambda x: x.stat().st_mtime
        )
        
        while len(checkpoints) > save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            print_rank0(f"删除旧检查点: {old_checkpoint}")
