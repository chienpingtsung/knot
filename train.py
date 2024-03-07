import logging
from itertools import count
from pathlib import Path

import torch
import torchvision
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data.processing import Processing
from lib.data.proxies import collate_fn, VideoDatasetProxy
from lib.data.transforms import Compose, ToTensor, Normalize
from lib.dataset.lasot import LaSOT
from lib.model.vit import VisionTransformer
from lib.util.arguments import parse_train_args
from lib.util.config import Config
from lib.util.frameworks import getDevice

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_args, dataset_args, model_args, dataset, model):
        self.train_args = train_args

        # A tensorboard for training process recording.
        self.writer = SummaryWriter(train_args.log_dir)

        # The computing device.
        self.deivce = getDevice()

        # The dataset.
        self.dataset = DataLoader(dataset,
                                  batch_size=dataset_args.batch_size,
                                  shuffle=True,
                                  num_workers=dataset_args.batch_size,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  drop_last=True)

        # The checkpoint.
        self.checkpoint = EasyDict(torch.load(train_args.checkpoint)) if train_args.checkpoint else None

        # The model.
        self.model = model
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint.model)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.deivce)

        # The losses.
        self.criterion_ciou = torchvision.ops.complete_box_iou_loss
        self.criterion_sl1 = nn.SmoothL1Loss()
        self.criterion_fl = torchvision.ops.sigmoid_focal_loss

        # The optimizer.
        self.optimizer = torch.optim.Adam(self.model.parameters())
        if self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint.optimizer)

        # The scout for stop training.
        self.best_loss = self.checkpoint.best_loss if self.checkpoint else float('inf')
        self.best_loss_epoch = self.checkpoint.best_loss_epoch if self.checkpoint else 0

    def train(self):
        for epoch in count(self.checkpoint.epoch + 1 if self.checkpoint else 0):
            self.model.train()
            pbar = tqdm(self.dataset)
            for iput, gt in pbar:
                iput, gt = iput.to(self.deivce), gt.to(self.deivce)

                oput = self.model(iput)

                loss = self.criterion_ciou(oput, gt) + self.criterion_sl1(oput, gt) + self.criterion_fl(oput, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()

                pbar.set_description(f'Training epoch {epoch}, loss{loss}')
                self.writer.add_scalar('train/loss', loss, epoch)

            self.checkpoint = {'model': self.model.module.state_dict() if isinstance(self.model,
                                                                                     nn.DataParallel) else self.model.state_dict(),
                               'optimizer': self.optimizer.state_dict(),
                               'best_loss': loss if loss < self.best_loss else self.best_loss,
                               'best_loss_epoch': epoch if loss < self.best_loss else self.best_loss_epoch,
                               'epoch': epoch}
            torch.save(self.checkpoint, Path(self.writer.log_dir).joinpath(f'epoch{epoch:03}.pth'))

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_loss_epoch = epoch
                torch.save(self.checkpoint, Path(self.writer.log_dir).joinpath(f'best.pth'))

            if epoch - self.best_loss_epoch > self.train_args.scout:
                logger.info(
                    f'No significant decent in loss, stop training and save best checkpoint at epoch {self.best_loss_epoch}')


if __name__ == '__main__':
    train_args = parse_train_args()
    hyper = Config()
    hyper.load(train_args.config)

    transforms = Compose(ToTensor(hyper.pretrained_nl_model, 'max_length', True, hyper.nl_max_length, 'pt'),
                         Normalize(hyper.mean, hyper.std))
    processing = Processing(transforms)
    dataset = VideoDatasetProxy([LaSOT(hyper.dataset_root, hyper.dataset_partition)], processing)

    model = VisionTransformer()
    trainer = Trainer(train_args, hyper, hyper, dataset, model)
    trainer.train()
