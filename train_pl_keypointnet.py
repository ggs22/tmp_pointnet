import os
import faulthandler
import hydra

import pytorch_lightning as pl
import torch.utils.data
from omegaconf import DictConfig

from callbacks.save_predictions import SavePredictions
from data_utils.KeypointsNetDataLoader import KeypointsDataset
from models.pl_pointnet2_kpts_reg import KeypointNet2


@hydra.main(version_base=None, config_path='configs', config_name="config")
def main(cfg: DictConfig):
    root = cfg.training.data_root

    # Setting up data
    train_dataset = KeypointsDataset(
        root=root,
        npoints=cfg.model.npoint,
        split='trainval',
        normal_channel=cfg.model.normal
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    test_dataset = KeypointsDataset(
        root=root,
        npoints=cfg.model.npoint,
        split='test',
        normal_channel=cfg.model.normal
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = KeypointNet2(
        num_classes=cfg.model.num_classes,
        normal_channel=cfg.model.normal,
        channels_offset=cfg.model.channel_offset,
        num_point=cfg.model.npoint,
        batch_size=cfg.training.batch_size
    )

    callbacks = [
        pl.callbacks.EarlyStopping(
            'val_loss',
            patience=5
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss'
        )
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.training.epoch,
        check_val_every_n_epoch=10,
        callbacks=callbacks
    )

    log_dir = trainer.logger.log_dir

    save_prediction_callback = SavePredictions(
        os.path.join(log_dir, 'predictions')
    )

    trainer.callbacks.append(save_prediction_callback)

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    faulthandler.enable()
    main()
