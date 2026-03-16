"""Training script for anti-spoofing countermeasure with meta-learning and GRL."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from model.lcnn import LCNN
from model.loss import FocalLoss, AutomaticWeightedLoss
from dataset.dataset import MetaDataset, CNSpoofDataset
from pyutils.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class Trainer:
    """Meta-learning trainer with GRL for audio anti-spoofing.

    Combines three regularization strategies:
    1. MAML meta-learning (meta-train → inner update → meta-test)
    2. Gradient Reversal Layer for domain-invariant features
    3. Automatic multi-task loss weighting (uncertainty-based)
    """

    def __init__(self):
        set_seed()
        self.args = self._parse_args()
        self.conf = self._load_config('conf/conf.yaml')

        self.log_dir = Path('exp') / self.args['task']
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = self.conf['log']['log_interval']
        self.device = torch.device(self.conf['train']['device'])

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()

        self.num_epochs = self.conf['train']['epoch']
        self.start_epoch = 1

        logger.info("Initialization done!")

    @staticmethod
    def _parse_args() -> dict:
        parser = argparse.ArgumentParser(description='Anti-spoofing CM training')
        parser.add_argument('--mode', default='train', type=str,
                            choices=['train', 'test'])
        parser.add_argument('--task', default='ordinary', type=str,
                            help='Task name (data subdirectory)')
        parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight-decay', dest='weight_decay',
                            default=0.0001, type=float)
        parser.add_argument('--step-size', dest='step_size', default=1, type=int)
        parser.add_argument('--gamma', default=0.9, type=float)
        parser.add_argument('--ckpt', default='', type=str,
                            help='Checkpoint filename for evaluation')
        return vars(parser.parse_args())

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_data(self):
        trainset = MetaDataset(self.args['task'], split='train')

        # Build DataLoader config, replacing collate_fn name with actual function
        loader_conf = dict(self.conf['dataloader'])
        collate_name = loader_conf.pop('collate_fn')
        self.args['collate_fn'] = collate_name

        if collate_name == 'random':
            loader_conf['collate_fn'] = trainset.random_collate_fn
        elif collate_name == 'balance':
            loader_conf['collate_fn'] = trainset.balance_collate_fn
        else:
            raise ValueError(f"Unknown collate_fn: {collate_name}")

        self.train_loader = DataLoader(trainset, **loader_conf)

        valset = CNSpoofDataset(self.args['task'], split='val')
        evalset = CNSpoofDataset(self.args['task'], split='test')
        self.val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
        self.eval_loader = DataLoader(evalset, batch_size=1, shuffle=False, num_workers=1)

    def _setup_model(self):
        self.model = LCNN(**self.conf['model']['model_args']).to(self.device)
        self.auto_loss_weight = AutomaticWeightedLoss(num=3)

    def _setup_optimizer(self):
        opt_conf = self.conf['train']

        if opt_conf['optimizer'] != 'SGD':
            raise NotImplementedError("Only SGD optimizer is supported")

        self.optimizer = optim.SGD(
            [
                {'params': self.model.parameters()},
                {'params': self.auto_loss_weight.parameters(), 'weight_decay': 0.0},
            ],
            lr=self.args['lr'],
            momentum=self.args['momentum'],
            weight_decay=self.args['weight_decay'],
        )

        if opt_conf['scheduler'] != 'step':
            raise NotImplementedError("Only StepLR scheduler is supported")

        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, **opt_conf['scheduler_params'],
        )

    def _setup_criterion(self):
        self.criterion = nn.BCELoss()
        self.domain_criterion = FocalLoss(gamma=self.conf['train']['gamma'])

    def _get_meta_params(self) -> List[nn.Parameter]:
        """Get model parameters that participate in meta-learning inner loop.

        Excludes frontend (fixed feature extractor) and alignment (domain head)
        parameters, as these should not receive inner-loop updates.
        """
        return [
            p for name, p in self.model.named_parameters()
            if 'frontend' not in name and 'alignment' not in name
        ]

    def _reset_fast_weights(self):
        """Reset fast weights to None before each meta-learning step."""
        for param in self._get_meta_params():
            param.fast = None

    def save_checkpoint(self, epoch: int):
        ckpt_dir = self.log_dir / 'models'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
            },
            ckpt_dir / f'{epoch}.ckpt',
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(self.log_dir / 'models' / path, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch']

    def train(self):
        wandb.init(
            project='interspeech2023',
            group='CM_genre_meta_learning_grl',
            job_type=self.args['task'],
            name=f"{self.args['task']}-{self.args['collate_fn']}",
            config=self.conf,
        )

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._train_epoch(epoch)
            self.lr_scheduler.step()
            if epoch % self.log_interval == 0:
                self.evaluate(epoch, mode='test')
                self.save_checkpoint(epoch)

        wandb.finish()

    def _train_epoch(self, epoch: int):
        self.model.train()
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        stats = {
            'mtr_loss': 0., 'mtr_dloss': 0., 'mtr_samples': 0,
            'mtr_correct': 0, 'mtr_dcorrect': 0,
            'mte_loss': 0., 'mte_samples': 0, 'mte_correct': 0,
        }

        for batch_idx, data in enumerate(progress):
            (mtr_audio, mtr_labels, mtr_lengths, mtr_genres,
             mte_audio, mte_labels, mte_lengths, mte_genres) = data

            stats['mtr_samples'] += len(mtr_audio)
            stats['mte_samples'] += len(mte_audio)

            results = self._train_batch(
                mtr_audio, mtr_labels, mtr_lengths, mtr_genres,
                mte_audio, mte_labels, mte_lengths, mte_genres,
            )

            stats['mtr_loss'] += results['mtr_loss'] * len(mtr_audio)
            stats['mtr_dloss'] += results['mtr_dloss'] * len(mtr_audio)
            stats['mtr_correct'] += results['mtr_correct']
            stats['mtr_dcorrect'] += results['mtr_dcorrect']
            stats['mte_loss'] += results['mte_loss'] * len(mte_audio)
            stats['mte_correct'] += results['mte_correct']

            n_mtr = stats['mtr_samples']
            n_mte = stats['mte_samples']
            progress.set_description(
                f'Epoch {epoch:3d} [{batch_idx+1:4d}/{len(self.train_loader):4d}] '
                f'MTRL: {stats["mtr_loss"]/n_mtr:.4f} '
                f'MTRDL: {stats["mtr_dloss"]/n_mtr:.4f} '
                f'MTRA: {100*stats["mtr_correct"]/n_mtr:.2f}% '
                f'MTRDA: {100*stats["mtr_dcorrect"]/n_mtr:.2f}% '
                f'MTEL: {stats["mte_loss"]/n_mte:.4f} '
                f'MTEA: {100*stats["mte_correct"]/n_mte:.2f}%'
            )

            wandb.log({
                'train/mtr_acc': stats['mtr_correct'] / n_mtr,
                'train/mtr_dacc': stats['mtr_dcorrect'] / n_mtr,
                'train/mte_acc': stats['mte_correct'] / n_mte,
            })

    def _train_batch(self, mtr_audio, mtr_labels, mtr_lengths, mtr_genres,
                     mte_audio, mte_labels, mte_lengths, mte_genres) -> Dict:
        """Execute one meta-learning training step.

        1. Meta-train: forward pass on meta-train data, compute inner gradients
        2. Inner update: create fast weights θ' = θ - α∇L_mtr
        3. Meta-test: forward pass with θ' on meta-test data
        4. Outer update: combined loss backprop through both phases
        """
        self._reset_fast_weights()
        meta_params = self._get_meta_params()

        # === Meta-Train Phase ===
        mtr_audio = mtr_audio.to(self.device)
        mtr_labels = mtr_labels.to(self.device)
        mtr_genres = mtr_genres.to(self.device)

        mtr_scores, mtr_dscores = self.model(mtr_audio, mtr_lengths)
        mtr_loss = self.criterion(mtr_scores, mtr_labels)
        mtr_dloss = self.domain_criterion(mtr_dscores, mtr_genres)

        # Inner loop: compute fast weights
        mtr_grads = torch.autograd.grad(
            mtr_loss, meta_params, create_graph=True, allow_unused=True,
        )
        lr = self.optimizer.param_groups[0]['lr']
        for param, grad in zip(meta_params, mtr_grads):
            param.fast = param - lr * grad

        # === Meta-Test Phase ===
        mte_audio = mte_audio.to(self.device)
        mte_labels = mte_labels.to(self.device)

        mte_scores, _ = self.model(mte_audio, mte_lengths)
        mte_loss = self.criterion(mte_scores, mte_labels)

        # Combined loss with automatic uncertainty weighting
        total_loss = self.auto_loss_weight(mte_loss, mtr_dloss, mtr_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        wandb.log({
            'train/mtr_loss': mtr_loss.item(),
            'train/mtr_dloss': mtr_dloss.item(),
            'train/mte_loss': mte_loss.item(),
            'train/total_loss': total_loss.item(),
            'train/lr': lr,
        })

        return {
            'mtr_correct': ((mtr_scores > 0.5) == mtr_labels).sum().item(),
            'mtr_dcorrect': (mtr_dscores.argmax(dim=1) == mtr_genres).sum().item(),
            'mtr_loss': mtr_loss.item(),
            'mtr_dloss': mtr_dloss.item(),
            'mte_correct': ((mte_scores > 0.5) == mte_labels).sum().item(),
            'mte_loss': mte_loss.item(),
        }

    @torch.no_grad()
    def evaluate(self, epoch: int, mode: str = 'test'):
        """Evaluate model and write scores to file."""
        self._reset_fast_weights()

        loader = self.eval_loader if mode == 'test' else self.val_loader
        score_dir = self.log_dir / 'scores'
        score_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        correct = 0

        with open(score_dir / f'{mode}_{epoch}.score', 'w') as f:
            for audio, label, genre, genreid, lengths in tqdm(loader, desc=f'Eval {mode}'):
                audio = audio.to(self.device)
                label = label.to(self.device)
                score, _ = self.model(audio, lengths)

                if (score > 0.5) == label:
                    correct += 1
                f.write(f'{score.item()} {label.item()} {genre[0]}\n')

        acc = correct / len(loader)
        logger.info(f'{mode} Acc: {100 * acc:.3f}%')

        try:
            wandb.log({f'{mode}/acc': acc})
        except Exception:
            pass


def main():
    trainer = Trainer()
    if trainer.args['mode'] == 'train':
        trainer.train()
    elif trainer.args['mode'] == 'test':
        trainer.load_checkpoint(trainer.args['ckpt'])
        trainer.evaluate(trainer.start_epoch, mode='test')


if __name__ == '__main__':
    main()
