import argparse
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
import yaml
from compressai.zoo import bmshj2018_factorized
from PIL import Image
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy("file_system")
import json

import torchmetrics
import wandb
from data_loader import FitsDataModule
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.tuner.tuning import Tuner


class NCompressor(LightningModule):
    def __init__(
        self, quality=2, learning_rate=1e-4, learning_rate_schedule_patience=10
    ):
        super().__init__()
        self.save_hyperparameters()

        # define model
        self.model = bmshj2018_factorized(quality=quality, pretrained=True)
        self.loss = (
            torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure()
        )  # hala says this was unstable - maybe train with mse
        self.metric = torchmetrics.image.PeakSignalNoiseRatio()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, event_index, fts_file = batch
        x = images / 65535  # TODO: add this to dataloader
        x_hat = self.forward(x)["x_hat"]
        loss = -self.loss(x_hat, x)

        if batch_idx % 10 == 0:
            self.log("train/loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_batch_index = random.randint(0, 4)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        images, event_index, fts_file = batch
        x = images / 65535  # TODO: add this to dataloader
        x_hat = self.model(x)["x_hat"]
        loss = -self.loss(x_hat, x)

        self.metric.update(x_hat, x)

        if batch_idx % 10 == 0:
            self.log("val/loss", loss, prog_bar=True)

        if batch_idx == self.validation_batch_index:
            if wandb.run:
                diff = (x_hat - x).abs()
                wandb.log(
                    {
                        "observations": [
                            wandb.Image(
                                x[0].cpu().permute(1, 2, 0).numpy(),
                                caption="Input",
                            ),
                            wandb.Image(
                                x_hat[0].cpu().permute(1, 2, 0).numpy(),
                                caption="Prediction",
                            ),
                            wandb.Image(
                                diff[0].cpu().permute(1, 2, 0).numpy(),
                                caption="Difference",
                            ),
                        ],
                        "labels": event_index[0],
                    }
                )

        return loss

    def on_validation_epoch_end(self):
        metric = self.metric.compute()
        self.log("PSNR:", metric)

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
    with open(os.path.join(config_path, "onboard.yaml"), "r") as f:
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


if __name__ == "__main__":
    seed_everything(42)

    wandb.init(project="NCompression", entity="ssa_live_twin", mode="offline")

    wandb_logger = WandbLogger()

    # read in yaml variables from config
    PROJECT_DIR = dirname(dirname(dirname(dirname(dirname(__file__)))))
    config_path = os.path.join(PROJECT_DIR, "config")
    config = get_config(config_path)

    ckpt_dir = config["train"]["ckpt_dir"]
    log_dir = config["train"]["log_dir"]
    out_dir = config["train"]["out_dir"]

    print(config)
    sys.exit()

    # torch.set_float32_matmul_precision('medium' | 'high')

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=log_dir,
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    csv_logger = CSVLogger(save_dir=args.log_dir, name="logs")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=[csv_logger, wandb_logger],
    )

    wandb.summary.update(
        {
            "model": args.model,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint": args.checkpoint_dir,
        }
    )

    model = NCompressor(learning_rate=args.learning_rate)

    dm = (
        FitsDataModule()
    )  # TODO: add config parameters in command arguments # just takes everything?

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    wandb.finish()
