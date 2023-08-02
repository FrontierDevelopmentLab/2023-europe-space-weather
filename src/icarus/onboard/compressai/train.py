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
from lightning.pytorch.callbacks import (
    LambdaCallback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.tuner.tuning import Tuner


class NCompressor(LightningModule):
    def __init__(
        self,
        quality=2,
        loss="mse",
        learning_rate=1e-4,
        learning_rate_schedule_patience=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # define model
        self.model = bmshj2018_factorized(quality=quality, pretrained=True)

        # msssim may be unstable
        if self.hparams.loss == "mssim":
            self.loss = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure()
        elif self.hparams.loss == "mse":
            self.loss = (
                torchmetrics.MeanSquaredError()
            )  # TODO maybe this needs to be flattened?
        elif self.hparams.loss == "rmse_sw":
            self.loss = torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow()
        elif self.hparams.loss == "psnr":
            self.loss = torchmetrics.image.PeakSignalNoiseRatio()
        else:
            print(self.hparams.loss, "loss not implemented")

        self.metric = [
            torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(),
            torchmetrics.image.PeakSignalNoiseRatio(),
            torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow(),
        ]

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
        # metric = self.metric.compute()
        # self.log("PSNR:", metric)
        metrics = self.metric.compute()
        self.log({"SSIM": metrics[0], "PSNR": metrics[1], "RMSE_SW": metrics[2]})

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

    wandb.init(
        project="NCompression",
        entity="ssa_live_twin",
        config=config,
        offline=config["train"]["wandb_offline"],
    )  # , mode="offline")
    wandb_logger = WandbLogger()

    # torch.set_float32_matmul_precision('medium' | 'high')

    # save best and last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.config.train.ckpt_dir,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
        verbose=True,
        every_n_train_steps=wandb.config.train.log_every_n_steps,
    )

    # save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    csv_logger = CSVLogger(save_dir=wandb.config.train.log_dir, name="logs")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model = NCompressor(
        quality=wandb.config.train.quality,
        loss=wandb.config.train.loss,
        learning_rate=wandb.config.train.learning_rate,
        learning_rate_schedule_patience=wandb.config.train.learning_rate_schedule_patience,
    )

    # get checkpoint path if resume training from previous model
    # TODO make into a function
    if wandb.config.train.resume_from_checkpoint:
        if wandb.config.train.ckpt_path == "last":
            ckpt_path = os.path.join(wandb.config.train.ckpt_dir, "last.ckpt")
        else:
            ckpt_path = wandb.config.train.ckpt_path
        logging.info(f"Load checkpoint: {ckpt_path}")
        # model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    else:
        ckpt_path = None

    # TODO: just take from config not everything
    dm = FitsDataModule()  # pass andb.config.data_loader?

    trainer = Trainer(
        max_epochs=wandb.config.train.nepochs,
        accelerator="auto",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=[csv_logger, wandb_logger],
    )

    logging.info("Start model training")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm)

    trainer.save_checkpoint(os.path.join(wandb.config.train.ckpt_dir, "final.ckpt"))

    # TODO save best and last to separate dir?
    # out_dir = config["train"]["out_dir"]

    wandb.finish()
