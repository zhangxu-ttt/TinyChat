from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from baseTrainer import BaseTrainer
from dataset import TextDataset

class PreTrainer(BaseTrainer):
    def build_dataloader(self):
        """
        准备数据集 - 子类必须实现

        Returns:
            train_dataset, eval_dataset
        """
        train_data_path = self.config['dataset']['train_data_path']
        max_length = self.config['dataset']['max_length']
        num_workers = self.config['dataset']['num_workers']
        prefetch_factor = self.config['dataset']['prefetch_factor']
        persistent_workers = self.config['dataset']['persistent_workers']
        pin_memory = self.config['dataset']['pin_memory']
        drop_last = self.config['dataset']['drop_last']

        batch_size = self.config['trainer']['batch_size']

        dataset = TextDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

        return dataloader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...

