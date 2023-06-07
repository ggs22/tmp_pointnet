"""
Author: Benny
Date: Nov 2019
Adapted for TMS Systems on april 2023
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import json
import metrics.metrics as m
import re
import faulthandler

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

import utils.paths_utils as pu
from data_utils.KeypointsNetDataLoader import KeypointsDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes = {'beam': [0, 1]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=200, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--num_classes', type=int, default=12, help='number of classes')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts (contained by classes)')
    parser.add_argument('--data_root', type=str, default='data/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
                        help='root directory for the dataset')
    # TODO remove or re-work this argument
    parser.add_argument('--channel_offset', type=int, default=0, help='adjust the input channels for convolution')

    return parser.parse_args()


def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_output_dir = Path('./log/')
    experiment_output_dir.mkdir(exist_ok=True)
    experiment_output_dir = experiment_output_dir.joinpath('part_seg')
    experiment_output_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_output_dir = experiment_output_dir.joinpath(timestr)
    else:
        experiment_output_dir = experiment_output_dir.joinpath(args.log_dir)
    experiment_output_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_output_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_output_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{log_dir}/{args.model}.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(str):
        logger.info(str)
        print(str)

    log_string('PARAMETER ...')
    log_string(args)

    log_string(f"Using CUDA: {torch.cuda.is_available()}")

    # Tensorboard loger
    writer = SummaryWriter(log_dir=str(experiment_output_dir.joinpath("tb_runs")))

    '''DATA'''
    root = args.data_root
    train_dataset = KeypointsDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    drop_last=False)
    test_dataset = KeypointsDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    log_string(f"The number of training data is: {len(train_dataset)}")
    log_string(f"The number of test data is: {len(test_dataset)}")

    num_classes = args.num_classes
    # num_part = args.num_parts

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(experiment_output_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_output_dir))

    classifier = MODEL.GetModel(num_classes,
                                normal_channel=args.normal,
                                channels_offset=args.channel_offset,
                                num_point=args.npoint).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    ''' LOAD MODEL '''
    checkpoint_path = pu.get_current_model_path(str(experiment_output_dir))
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Use pretrain model from: {checkpoint_path}')
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0.5,
                                        weight_decay=args.decay_rate)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-7
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_distance = np.inf
    best_loss = torch.inf
    global_epoch = 0

    '''for each epochs'''
    for epoch in range(start_epoch, args.epoch):
        batchwise_distances = list()
        batchwise_total_losses = list()
        batchwise_mse_losses = list()
        batchwise_bce_losses = list()

        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')

        '''Adjust learning rate and BN momentum'''
        if args.step_size <= 0:
            lr = args.learning_rate
        else:
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)

        log_string(f"Learning rate:{lr}" + f" (sheduled)" * (args.step_size > 0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        else:
            print(f'BN momentum updated to: {momentum}')
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''for each training step'''
        for i, (points, one_hot_target, kpts_target, sample_id) in tqdm(enumerate(train_data_loader),
                                                                        total=len(train_data_loader),
                                                                        smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, one_hot_target, kpts_target = \
                points.float().cuda(), one_hot_target.float().cuda(), kpts_target.float().cuda()
            points = points.transpose(2, 1)

            kpts_pred, class_pred = classifier(points)
            print(f"{torch.squeeze(one_hot_target)}")
            dist = m.euclidian_dist(kpts_pred, torch.squeeze(kpts_target))
            batchwise_distances.append(dist)

            total_loss, mse_loss, bce_loss = criterion(kpts_pred,
                                                       torch.squeeze(kpts_target),
                                                       class_pred,
                                                       torch.squeeze(one_hot_target))
            batchwise_total_losses.append(total_loss.item())
            batchwise_mse_losses.append(mse_loss.item())
            batchwise_bce_losses.append(bce_loss.item())
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            def save_keyppoints_pred_to_json(points: torch.Tensor,
                                             kpts_pred: torch.Tensor,
                                             sample_id: str,
                                             stage: str) -> None:

                tx = points.transpose(1, 2).cpu().numpy().squeeze()
                ty = kpts_pred.cpu().data.numpy()
                output_path = Path(experiment_output_dir).joinpath(f'{sample_id}_{stage}_xyz.txt')
                np.savetxt(fname=str(output_path), X=tx)
                weld_path_dict = dict()
                for weld_path_ix in range(ty.shape[0]):
                    weld_path_dict[f'weld_path{weld_path_ix}'] = list()
                    for keypoint_ix in range(ty.shape[1]):
                        x = float(ty[weld_path_ix, keypoint_ix, :][0])
                        y = float(ty[weld_path_ix, keypoint_ix, :][1])
                        z = float(ty[weld_path_ix, keypoint_ix, :][2])
                        weld_path_dict[f'weld_path{weld_path_ix}'].append([x, y, z])
                output_path = Path(experiment_output_dir).joinpath(f'{sample_id}_{stage}_weld_paths.json')
                with open(file=str(output_path), mode='w') as f:
                    json.dump(obj=weld_path_dict, fp=f)

            # if i % 5 == 0:
            #     save_keyppoints_pred_to_json(points=points, kpts_pred=kpts_pred, sample_id=sample_id[0], stage="train")

        dist_tensor = torch.concatenate(batchwise_distances)
        train_mean_dist = dist_tensor.mean().item()
        quantiles = [.25, .5, .75]
        dist_quantiles = torch.quantile(dist_tensor, q=torch.tensor(data=quantiles, device=dist_tensor.device))
        q_string = str()
        for quantile, distance in zip(quantiles, dist_quantiles):
            q_string += f"{quantile:.0%}: {distance:.2f}, "
        q_string = q_string[:-2:]
        train_mean_loss = np.mean(batchwise_total_losses)
        train_mean_mse_loss = np.mean(batchwise_mse_losses)
        train_mean_bce_loss = np.mean(batchwise_bce_losses)

        writer.add_scalar(tag="lr", scalar_value=lr, global_step=epoch)
        writer.add_scalar(tag="train total loss", scalar_value=train_mean_loss, global_step=epoch)
        writer.add_scalar(tag="train reg. loss", scalar_value=train_mean_mse_loss, global_step=epoch)
        writer.add_scalar(tag="train class. loss", scalar_value=train_mean_bce_loss, global_step=epoch)
        writer.add_scalar(tag="train dist (mm)", scalar_value=train_mean_dist, global_step=epoch)

        # TODO: add quantile histogram to tensorboard

        log_string(f'\nMean train distance: {train_mean_dist:.5f}, '
                   f'quantiles: {q_string}, '
                   f'min: {dist_tensor.min().item():.2f}, '
                   f'max: {dist_tensor.max().item():.2f}, '
                   f'total train loss: {train_mean_loss:.5f}, '
                   f'reg. train loss: {train_mean_mse_loss:.5f}, '
                   f'train clas.. loss: {train_mean_bce_loss:.5f}')

        batchwise_total_losses.clear()
        batchwise_mse_losses.clear()
        batchwise_bce_losses.clear()
        batchwise_distances.clear()

        with torch.no_grad():
            test_metrics = {}

            classifier = classifier.eval()

            for batch_id, (points, one_hot_target, kpts_target, sample_id) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, one_hot_target, kpts_target = points.float().cuda(), one_hot_target.float().cuda(), kpts_target.float().cuda()
                points = points.transpose(2, 1)
                kpts_pred, class_pred = classifier(points)
                kpts_pred_val = kpts_pred.cpu().data.numpy()
                kpts_target = kpts_target.cpu().data.numpy()

                total_loss, mse_loss, bce_loss = criterion(kpts_pred.squeeze().cuda(),
                                                           torch.tensor(kpts_target.squeeze()).cuda(),
                                                           class_pred.cuda(),
                                                           torch.squeeze(one_hot_target).cuda())
                dist = m.euclidian_dist(torch.tensor(kpts_pred_val), torch.squeeze(torch.tensor(kpts_target)))
                batchwise_distances.append(dist)
                batchwise_total_losses.append(total_loss.item())
                batchwise_mse_losses.append(mse_loss.item())
                batchwise_bce_losses.append(bce_loss.item())

                # save_keyppoints_pred_to_json(points=points, kpts_pred=kpts_pred, sample_id=sample_id[0], stage="test")

            test_metrics['distance'] = torch.concat(batchwise_distances).mean().item()
            test_metrics['loss'] = np.mean(batchwise_total_losses)
            test_metrics['mse loss'] = np.mean(batchwise_mse_losses)
            test_metrics['bce loss'] = np.mean(batchwise_bce_losses)
            writer.add_scalar(tag="val total loss", scalar_value=test_metrics['loss'], global_step=epoch)
            writer.add_scalar(tag="val reg. loss", scalar_value=test_metrics['mse loss'], global_step=epoch)
            writer.add_scalar(tag="val class. loss", scalar_value=test_metrics['bce loss'], global_step=epoch)
            writer.add_scalar(tag="val dist (mm)", scalar_value=test_metrics['distance'], global_step=epoch)

            # TODO: add quantile histogram to tensorboard

        log_string(f"Epoch: {epoch+1} "
                   f"test Distance: {test_metrics['distance']}, "
                   f"test total Loss: {test_metrics['loss']}, "
                   f"test regressions Loss: {test_metrics['mse loss']}, "
                   f"test classification Loss: {test_metrics['bce loss']}")

        def save_model(current_model_path: str, epoch: int):
            if Path(current_model_path).exists():
                os.remove(path=current_model_path)
            num_match = re.match(pattern=r'(.*pointnet_model.*epoch_)\d+(\.pth)', string=current_model_path)
            if num_match:
                current_model_path = num_match[1] + str(epoch) + num_match[2]
            logger.info(f"Save model with new best loss {test_metrics['loss']}...")
            log_string(f'Saving at {current_model_path}')
            state = {
                'epoch': epoch,
                'train_dist': train_mean_dist,
                'train_loss': train_mean_loss,
                'test_acc': test_metrics['distance'],
                'test_loss': test_metrics['loss'],
                'test_reg_loss': test_metrics['mse loss'],
                'test_cls_loss': test_metrics['bce loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, current_model_path)

        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            log_string(f'New best loss is: {best_loss:.5f}')
            save_model(pu.get_best_validation_model_path(str(experiment_output_dir)), epoch)
        else:
            save_model(pu.get_current_model_path(str(experiment_output_dir)), epoch)

        ''' Update best metrics in display '''
        if test_metrics['distance'] < best_distance:
            best_distance = test_metrics['distance']
            log_string(f'New best distance is: {best_distance:.5f}')

        global_epoch += 1


if __name__ == '__main__':
    faulthandler.enable()
    args = parse_args()
    main(args)
