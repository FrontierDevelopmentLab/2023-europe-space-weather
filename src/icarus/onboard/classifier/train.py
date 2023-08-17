import argparse
import logging
import os
import random
import sys
from datetime import datetime
from functools import lru_cache
from os.path import dirname
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.models as models
import yaml
from PIL import Image
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

torch.multiprocessing.set_sharing_strategy("file_system")
import json

from cme_dataloader import CMEDataModule
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import (
    LambdaCallback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner

import wandb

# from ..data_loader import (  # This is just a place holder for the CME data_loader
#    FitsDataModule,
# )


class CME_classifier(LightningModule):
    def __init__(
        self,
        model,
        pretrained=True,
        loss="bce",
        learning_rate=1e-4,
        learning_rate_schedule_patience=10,
        batch_size=4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # define model
        if "resnet" in self.hparams.model:
            if self.hparams.model == "resnet18":
                self.model = models.resnet18(weights=pretrained)
            elif self.hparams.model == "resnet50":
                self.model = models.resnet50(weights=pretrained)

            linear_size = list(self.model.children())[-1].in_features
            self.model.fc = nn.Linear(linear_size, 1)
        elif (
            self.hparams.model == "pixel_counter"
        ):  # simple pixel counter would go here
            print(self.hparams.model, "pixel_counter not implemented")
        else:
            print(self.hparams.model, "model not implemented")

        # define loss as binary cross entropy (CME vs. non CME)
        if self.hparams.loss == "bce":
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            print(self.hparams.loss, "loss not implemented")

        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.confusion = BinaryConfusionMatrix()

        self.batch_size = batch_size

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, y_target = batch
        images = images.repeat(1, 3, 1, 1).float()
        x = images
        y = self.forward(x).flatten()

        y_target = y_target.float()
        loss = self.loss(y, y_target)

        if batch_idx % 10 == 0 and wandb.run:
            self.log(
                "train/loss",
                loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_batch_index = random.randint(0, 4)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        images, y_target = batch
        images = images.repeat(1, 3, 1, 1).float()
        x = images
        y = self.forward(x).flatten()

        y_target = y_target.float()
        loss = self.loss(y, y_target)
        self.f1.update(y, y_target)
        self.accuracy.update(y, y_target)
        self.confusion.update(y, y_target)

        if batch_idx % 10 == 0 and wandb.run:
            self.log(
                "val/loss",
                loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        if batch_idx == self.validation_batch_index and wandb.run:
            x_vis = (x[0, 0] - x[0, 0].min()) / (x[0, 0].max() - x[0, 0].min())
            table = wandb.Table(
                columns=["Observation_image", "Predicted_label", "Ground truth label"]
            )
            img = wandb.Image(x_vis.cpu().numpy())
            table.add_data(img, torch.nn.functional.sigmoid(y[0]), y_target[0])
            wandb.log({"Table": table})

        return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        images, y_target = batch
        images = images.repeat(1, 3, 1, 1).float()

        x = images
        y = self.forward(x).flatten()

        y_target = y_target.float()
        loss = self.loss(y, y_target)

        if batch_idx % 10 == 0 and wandb.run:
            self.log(
                "test/loss",
                loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute()
        f1 = self.f1.compute()
        cm = self.confusion.compute()

        self.log("val/accuracy", accuracy, sync_dist=True, batch_size=self.batch_size)
        self.log("val/f1", f1, sync_dist=True, batch_size=self.batch_size)

        tp = cm[1, 1].float()
        fp = cm[0, 1].float()
        tn = cm[0, 0].float()
        fn = cm[1, 0].float()

        tot = tp+fp+tn+fn

        self.log("val/tp", tp/tot, sync_dist=True, batch_size=self.batch_size)
        self.log("val/fp", fp/tot, sync_dist=True, batch_size=self.batch_size)
        self.log("val/tn", tn/tot, sync_dist=True, batch_size=self.batch_size)
        self.log("val/fn", fn/tot, sync_dist=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    verbose=True,
                    patience=self.hparams.learning_rate_schedule_patience,
                ),
                "monitor": "val/loss",
            },
        }

        return


def get_config(config_path):
    """
    read in yaml config, generate path dirs
    """
    with open(os.path.join(config_path, "onboard_classifier.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # generate unique run_id if not given and use in all output paths
    if "run_id" not in config["train"]:
        config["train"]["run_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    for key in config["train"].keys():
        if type(config["train"][key]) == str:
            config["train"][key] = config["train"][key].replace(
                "<run_id>", config["train"]["run_id"]
            )

    # check if directories exist and generate if not
    for key in config["train"].keys():
        if "dir" in key:
            Path(config["train"][key]).mkdir(parents=True, exist_ok=True)

    return config


# def save_state(ncompressor: NCompressor, data_module: FitsDataModule, save_path):
#     output_path = '/'.join(save_path.split('/')[0:-1])
#     os.makedirs(output_path, exist_ok=True)
#     torch.save({'model': ncompressor.model,
#                 'config': },
#                 save_path)


if __name__ == "__main__":
    seed_everything(42)

    # read in yaml variables from config
    PROJECT_DIR = dirname(dirname(dirname(dirname(dirname(__file__)))))
    config_path = os.path.join(PROJECT_DIR, "config")
    config = get_config(config_path)

    wandb_logger = WandbLogger(
        project="CME_classification",
        entity="ssa_live_twin",
        name=config["train"]["run_id"],
        mode=config["train"]["wandb_mode"],
        config=config,
    )

    # save best and last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.config.train["ckpt_dir"],
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
        verbose=True,
        every_n_epochs=wandb.config.train["log_ckpt_every_n_epochs"],
    )

    # save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    csv_logger = CSVLogger(save_dir=wandb.config.train["log_dir"], name="logs")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model = CME_classifier(
        model=wandb.config.train["model"],
        pretrained=wandb.config.train["pretrained"],
        loss=wandb.config.train["loss"],
        learning_rate=wandb.config.train["learning_rate"],
        learning_rate_schedule_patience=wandb.config.train[
            "learning_rate_schedule_patience"
        ],
        batch_size=wandb.config.data_loader["batch_size"],
    )

    # get checkpoint path if resume training from previous model
    # TODO make into a function
    if wandb.config.train["resume_from_checkpoint"]:
        if wandb.config.train["ckpt_path"] == "last":
            ckpt_path = os.path.join(wandb.config.train["ckpt_dir"], "last.ckpt")
        else:
            ckpt_path = wandb.config.train["ckpt_path"]
        logging.info(f"Load checkpoint: {ckpt_path}")
        # model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    else:
        ckpt_path = None

    # dm = FitsDataModule(wandb.config.data_loader)

    dm = CMEDataModule(wandb.config.data_loader)

    trainer = Trainer(
        max_epochs=wandb.config.train["nepochs"],
        accelerator="auto",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=[csv_logger, wandb_logger],
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=wandb.config.train["log_every_n_steps"],
    )

    logging.info("Start model training")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm)

    trainer.save_checkpoint(os.path.join(wandb.config.train["ckpt_dir"], "final.ckpt"))

    # TODO save best and last to separate dir?
    # out_dir = config["train"]["out_dir"]

    wandb.finish()
