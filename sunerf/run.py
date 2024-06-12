import argparse
import logging
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from sunpy.visualization.colormaps import cm

from sunerf.evaluation.callback import log_overview
from sunerf.model.sunerf import SuNeRFModule, save_state
from sunerf.utilities.data_loader import NeRFDataModule

if __name__ == '__main__':
    N_GPUS = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config = {}
    with open(args.config, 'r') as stream:
        config.update(yaml.load(stream, Loader=yaml.SafeLoader))

    path_to_save = config['path_to_save']
    os.makedirs(path_to_save, exist_ok=True)
    working_dir = config['working_directory'] if 'working_directory' in config else path_to_save
    os.makedirs(working_dir, exist_ok=True)

    data_module = NeRFDataModule(**config['data'], working_dir=working_dir)
    cmap = cm.soholasco2.copy()
    cmap.set_bad(color='green')

    model_config = config['model'] if 'model' in config else {}
    module_config = config['module'] if 'module' in config else {}
    sunerf = SuNeRFModule(sampling_config=config['sampling_config'], **model_config, cmap=cmap,
                          Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                          **module_config)

    # manually laod --> automatic load checkpoint causes error
    chk_path = config['resume_from_checkpoint'] if 'resume_from_checkpoint' in config else None
    if chk_path is not None:
        logging.info(f'Load checkpoint: {chk_path}')
        sunerf.load_state_dict(torch.load(chk_path, map_location=torch.device('cpu'))['state_dict'])

    # save last and best
    train_config = config["train"] if "train" in config else {}
    log_every_n_steps = train_config["log_every_n_steps"] if "log_every_n_steps" in train_config else 1000
    checkpoint_callback = ModelCheckpoint(dirpath=path_to_save,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)

    wandb_config = config['wandb'] if 'wandb' in config else {}
    logger = WandbLogger(**wandb_config, save_dir=working_dir)

    save_path = os.path.join(path_to_save, 'save_state.snf')
    os.makedirs(path_to_save, exist_ok=True)
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path, config))



    epochs = train_config['epochs'] if 'epochs' in train_config else 100
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=0,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      gradient_clip_val=0.5,
                      callbacks=[checkpoint_callback, save_callback],
                      )

    log_overview(data_module.images, data_module.poses, data_module.times, cmap,
                 data_module.seconds_per_dt, data_module.Rs_per_ds, data_module.ref_time)

    logging.info('Start model training')
    trainer.fit(sunerf, data_module, ckpt_path='last')

    trainer.save_checkpoint(os.path.join(path_to_save, 'final.ckpt'))
