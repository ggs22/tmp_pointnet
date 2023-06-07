import os
import argparse
import faulthandler

import pytorch_lightning as pl
import torch.utils.data

from callbacks.save_predictions import SavePredictions
from data_utils.KeypointsNetDataLoader import KeypointsDataset
from models.pl_pointnet2_kpts_reg import KeypointNet2


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_kpts_reg', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--epoch', default=10, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='pointnet2_kpts_reg', help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=1500, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=40, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts (contained by classes)')
    parser.add_argument('--data_root', type=str, default='data/',
                        help='root directory for the dataset')
    # TODO remove or re-work this argument
    parser.add_argument('--channel_offset', type=int, default=14, help='adjust the input channels for convolution')

    return parser.parse_args()


def main(args):
    root = args.data_root

    # Setting up data
    train_dataset = KeypointsDataset(
        root=root,
        npoints=args.npoint,
        split='trainval',
        normal_channel=args.normal
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1
    )

    test_dataset = KeypointsDataset(
        root=root,
        npoints=args.npoint,
        split='test',
        normal_channel=args.normal
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )

    model = KeypointNet2(
        num_classes=args.num_classes,
        normal_channel=args.normal,
        channels_offset=args.channel_offset,
        num_point=args.npoint
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
        max_epochs=args.epoch,
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
    args = parse_args()
    main(args)
