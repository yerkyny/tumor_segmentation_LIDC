import random
import numpy as np
import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from utils import get_scores


class Learner(pl.LightningModule):
    def __init__(self, dataloaders, model, optimizer, scheduler, config, loss):
        super(Learner, self).__init__()
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        self.loss = loss
        self.val_length = 0
        self.test_length = 0
        self.best_mean_dice_val = -1

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, pr, gt):
        return self.loss(pr, gt)

    def train_dataloader(self):
        return self.dataloaders["train"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def val_dataloader(self):
        self.val_length = len(self.dataloaders["val"])
        self.include_steps = random.sample(
            range(0, self.val_length), k=self.config["K_SAMPLES"]
        )
        return self.dataloaders["val"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        scores, pred = get_scores(y, pred)
        return {
            "val_loss": loss,
            "dice": scores["dice"],
            "iou": scores["iou"],
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_loss = avg_val_loss.cpu()
        dice_values = list(filter(lambda a: a != -1, [x["dice"] for x in outputs]))
        mean_dice_score = np.mean(dice_values)
        iou_values = list(filter(lambda a: a != -1, [x["iou"] for x in outputs]))
        mean_iou_score = np.mean(iou_values)

        self.log("mean_dice_val", mean_dice_score, on_epoch=True)
        self.log("mean_iou_val", mean_iou_score)
        self.log("val_loss", avg_val_loss)

        if mean_dice_score > self.best_mean_dice_val:
            self.best_mean_dice_val = mean_dice_score

    def test_dataloader(self):
        self.test_length = len(self.dataloaders["test"])
        self.include_steps = random.sample(
            range(0, self.test_length), k=self.config["K_SAMPLES"]
        )
        return self.dataloaders["test"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        scores, pred = get_scores(y, pred)
        return {
            "test_dice": scores["dice"],
            "test_iou": scores["iou"],
        }

    def test_epoch_end(self, outputs):
        dice_values = list(filter(lambda a: a != -1, [x["test_dice"] for x in outputs]))
        mean_dice_score = np.mean(dice_values)
        iou_values = list(filter(lambda a: a != -1, [x["test_iou"] for x in outputs]))
        mean_iou_score = np.mean(iou_values)
        self.log("test_mean_dice", mean_dice_score)
        self.log("test_mean_iou", mean_iou_score)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
