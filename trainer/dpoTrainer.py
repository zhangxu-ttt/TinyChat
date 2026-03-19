from typing import Dict
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .baseTrainer import BaseTrainer
from dataset import ChatMLDataset

class DPOTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ref_model = copy.deepcopy(self.model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        self.beta = self.config['trainer']['beta']

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
        chosen_x = batch['chosen_x'].to(self.device, non_blocking=True)
        chosen_y = batch['chosen_y'].to(self.device, non_blocking=True)
        chosen_mask = batch['chosen_mask'].to(self.device, non_blocking=True)

        rejected_x = batch['rejected_x'].to(self.device, non_blocking=True)
        rejected_y = batch['rejected_y'].to(self.device, non_blocking=True)
        rejected_mask = batch['rejected_mask'].to(self.device, non_blocking=True)

        chosen_output = self.model(
            input_ids=chosen_x,
            labels=chosen_y,
            loss_mask=chosen_mask,
        )

        rejected_output = self.model(
            input_ids=rejected_x,
            labels=rejected_y,
            loss_mask=rejected_mask,
        )

        chosen_ref_output = self.ref_model(
            input_ids=chosen_x,
            labels=chosen_y,
            loss_mask=chosen_mask,
        )

        rejected_ref_output = self.ref_model(
            input_ids=rejected_x,
            labels=rejected_y,
            loss_mask=rejected_mask,
        )

        chosen_policy_logits = chosen_output.logits
        rejected_policy_logits = rejected_output.logits
        chosen_ref_logits = chosen_ref_output.logits
        rejected_ref_logits = rejected_ref_output.logits

        chosen_policy_logps = torch.gather(
            torch.log_softmax(chosen_policy_logits, dim=-1),
            dim=-1,
            index=chosen_y.unsqueeze(-1)
        ).squeeze(-1)
        chosen_policy_logps = torch.sum(chosen_policy_logps * chosen_mask, dim=-1)

        rejected_policy_logps = torch.gather(
            torch.log_softmax(rejected_policy_logits, dim=-1),
            dim=-1,
            index=rejected_y.unsqueeze(-1)
        ).squeeze(-1)
        rejected_policy_logps = torch.sum(rejected_policy_logps * rejected_mask, dim=-1)

        chosen_ref_logps = torch.gather(
            torch.log_softmax(chosen_ref_logits, dim=-1),
            dim=-1,
            index=chosen_y.unsqueeze(-1)
        ).squeeze(-1)
        chosen_ref_logps = torch.sum(chosen_ref_logps * chosen_mask, dim=-1)

        rejected_ref_logps = torch.gather(
            torch.log_softmax(rejected_ref_logits, dim=-1),
            dim=-1,
            index=rejected_y.unsqueeze(-1)
        ).squeeze(-1)
        rejected_ref_logps = torch.sum(rejected_ref_logps * rejected_mask, dim=-1)

        loss, chosen_reward, rejected_reward = self.dpo_loss(
            chosen_policy_logps,
            rejected_policy_logps,
            chosen_ref_logps,
            rejected_ref_logps,
            beta=self.beta
        )

        return {
            'loss': loss,
            'chosen_reward': chosen_reward,
            'rejected_reward': rejected_reward
        }

    def dpo_loss(
        self,
        chosen_policy_logps,
        rejected_policy_logps,
        chosen_ref_logps,
        rejected_ref_logps,
        beta
    ):
        chosen_reward = chosen_policy_logps - chosen_ref_logps
        rejected_reward = rejected_policy_logps - rejected_ref_logps
        loss = - F.logsigmoid(beta * (chosen_reward - rejected_reward)).mean()

        return loss, chosen_reward, rejected_reward



