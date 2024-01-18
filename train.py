import logging
from itertools import count
from pathlib import Path

import torch
import torchvision.ops
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data import proxies
from lib.data.processing import Processing
from lib.data.proxies import VideoDatasetProxy
from lib.data.transforms import Compose, ToTensor, Normalize
from lib.dataset.lasot import LaSOT
from lib.model.vit import VisionTransformer
from lib.util.arguments import parse_training_args
from lib.util.config import Config
from lib.util.frameworks import get_device

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, dataset, model: nn.Module):
        self.args = args
        self.writer = SummaryWriter(args.log_dir)
        self.device = get_device()

        self.dataset = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.batch_size,
                                  collate_fn=proxies.collate_fn,
                                  pin_memory=True,
                                  drop_last=True)

        self.snapshot = EasyDict(torch.load(args.weights)) if args.weights else None

        self.model = model
        if self.snapshot:
            self.model.load_state_dict(self.snapshot.model)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.criterion_ciou = torchvision.ops.complete_box_iou_loss
        self.criterion_sl1 = nn.SmoothL1Loss()
        self.criterion_fl = torchvision.ops.sigmoid_focal_loss

        self.optimizer = torch.optim.Adam(self.model.parameters())
        if self.snapshot:
            self.optimizer.load_state_dict(self.snapshot.optimizer)

        self.best_loss = self.snapshot.best_loss if self.snapshot else float('inf')
        self.best_loss_epoch = self.snapshot.best_loss_epoch if self.snapshot else 0

    def train(self):
        for epoch in count(self.snapshot.epoch + 1 if self.snapshot else 0):
            self.model.train()
            pbar = tqdm(self.dataset)
            for iput, gt in pbar:
                iput, gt = iput.to(self.device), gt.to(self.device)

                oput = self.model(iput)

                loss = self.criterion_ciou(oput, gt) + self.criterion_sl1(oput, gt) + self.criterion_fl(oput, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()

                pbar.set_description(f'Training epoch {epoch}, loss {loss}')
                self.writer.add_scalar('train/loss', loss, epoch)

            self.snapshot = {'model': self.model.module.state_dict() if isinstance(self.model,
                                                                                   nn.DataParallel) else self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'best_loss': loss if loss < self.best_loss else self.best_loss,
                             'best_loss_epoch': epoch if loss < self.best_loss else self.best_loss_epoch,
                             'epoch': epoch}
            torch.save(self.snapshot, Path(self.writer.log_dir).joinpath(f'epoch{epoch:03}.pth'))

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_loss_epoch = epoch
                torch.save(self.snapshot, Path(self.writer.log_dir).joinpath(f'best.pth'))

            if epoch - self.best_loss_epoch > self.args.scout:
                logger.info(
                    f'No significant decrease in losses, stop training and save best checkpoint at epoch {epoch}.')


if __name__ == '__main__':
    args = parse_training_args()
    hyper = Config()
    hyper.load(args.config)

    transforms = Compose(ToTensor(hyper.pretrained_nl_model, 'max_length', True, hyper.nl_max_length, 'pt'),
                         Normalize(hyper.mean, hyper.std))
    processing = Processing(transforms)
    dataset = VideoDatasetProxy([LaSOT(hyper.dataset_root, hyper.dataset_partition)], processing)

    model = VisionTransformer
    trainer = Trainer(args, dataset,)