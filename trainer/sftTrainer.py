from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .baseTrainer import BaseTrainer
from dataset import ChatMLDataset

class SFTTrainer(BaseTrainer):
    def build_dataloader(self):
        """
        准备数据集 - 子类必须实现

        Returns:
            train_dataset, eval_dataset
        """
        train_data_path = self.config['data']['train_data_path']
        max_length = self.config['data']['max_length']
        num_workers = self.config['data']['num_workers']
        prefetch_factor = self.config['data']['prefetch_factor']
        persistent_workers = self.config['data']['persistent_workers']
        pin_memory = self.config['data']['pin_memory']
        drop_last = self.config['data']['drop_last']

        batch_size = self.config['trainer']['batch_size']

        dataset = ChatMLDataset(
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

    def train_step(self, batch: Dict[str, torch.Tensor]) -> dict:
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        loss_mask = batch['loss_mask'].to(self.device)

        output = self.model(
            input_ids=x,
            labels=y,
            loss_mask=loss_mask,
        )

        loss = output.loss

        accuracy = (output.logits.argmax(dim=-1) == y).float()
        accuracy = (accuracy * loss_mask).mean()

        perplexity = torch.exp(loss).mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': perplexity
        }


