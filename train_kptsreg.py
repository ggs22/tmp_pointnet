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

from pathlib import Path
from tqdm import tqdm
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


def get_current_model_path(experiment_output_dir: str, mode: str = 'current') -> str:
    checkpoint_path = Path(experiment_output_dir).joinpath('checkpoints')
    if mode == 'current':
        pattern = r'pointnet_model_epoch_([0-9]+).pth'
    elif mode == 'best':
        pattern = r'pointnet_model_best_valid_epoch_([0-9]+).pth'
    latest_ix = -1
    res = "NONEXISTANT"
    for file_path in checkpoint_path.glob(pattern='*.pth'):
        match = re.match(pattern=pattern, string=str(file_path.name))
        if match is not None:
            ix = int(match[1])
            if ix > latest_ix:
                latest_ix = ix
                res = str(file_path)
    return res


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
    parser.add_argument('--step_size', type=int, default=40, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
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
    num_part = args.num_parts

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(experiment_output_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_output_dir))

    classifier = MODEL.GetModel(num_part,
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
    checkpoint_path = get_current_model_path(str(experiment_output_dir))
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
            optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-4
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_distance = np.inf
    best_loss = torch.inf
    global_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        batchwise_distances = list()
        batchwise_losses = list()

        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string(f'Learning rate:{lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        else:
            print(f'BN momentum updated to: {momentum}')
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target, sample_id) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.float().cuda()
            points = points.transpose(2, 1)

            kpts_pred = classifier(points, to_categorical(label, num_classes))

            dist = m.euclidian_dist(kpts_pred, torch.squeeze(target))
            batchwise_distances.append(dist)

            loss = criterion(kpts_pred, torch.squeeze(target))
            batchwise_losses.append(loss.item())
            loss.backward()
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

            if i % 5 == 0:
                save_keyppoints_pred_to_json(points=points, kpts_pred=kpts_pred, sample_id=sample_id[0], stage="train")

        train_mean_dist = torch.concatenate(batchwise_distances).mean().item()
        train_mean_loss = np.mean(batchwise_losses)
        log_string(f'\nMean train distance: {train_mean_dist:.5f}\n'
                   f'Mean train loss: {train_mean_loss:.5f}')

        batchwise_distances.clear()
        batchwise_losses.clear()

        with torch.no_grad():
            test_metrics = {}

            classifier = classifier.eval()

            for batch_id, (points, label, target, sample_id) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.float().cuda()
                points = points.transpose(2, 1)
                kpts_pred = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = kpts_pred.cpu().data.numpy()
                target = target.cpu().data.numpy()

                loss = criterion(kpts_pred.squeeze().cuda(), torch.tensor(target.squeeze()).cuda())
                batchwise_distances.append(m.euclidian_dist(torch.tensor(cur_pred_val), torch.squeeze(torch.tensor(target))))
                batchwise_losses.append(loss.item())

                save_keyppoints_pred_to_json(points=points, kpts_pred=kpts_pred, sample_id=sample_id[0], stage="test")

            test_metrics['distance'] = torch.concat(batchwise_distances).mean().item()
            test_metrics['loss'] = np.mean(batchwise_losses)

        log_string(f"Epoch: {epoch+1} "
                   f"test Distance: {test_metrics['distance']}, "
                   f"test Loss: {test_metrics['loss']}, ")

        def save_model(tag: str = '', mode: str = 'current'):
            current_model_path = get_current_model_path(str(experiment_output_dir), mode)
            if Path(current_model_path).exists():
                os.remove(path=current_model_path)
            logger.info(f"Save model with new best loss {test_metrics['loss']}...")
            savepath = str(checkpoints_dir) + f'/pointnet_model{tag}.pth'
            log_string(f'Saving at {savepath}')
            state = {
                'epoch': epoch,
                'train_dist': train_mean_dist,
                'train_loss': train_mean_loss,
                'test_acc': test_metrics['distance'],
                'test_loss': test_metrics['loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        save_model(tag=f"_epoch_{epoch}")

        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            log_string(f'New best loss is: {best_loss:.5f}')
            save_model(tag=f"_best_valid_epoch_{epoch}")

        ''' Update best metrics in display '''
        if test_metrics['distance'] < best_distance:
            best_distance = test_metrics['distance']
            log_string(f'New best distance is: {best_distance:.5f}')

        global_epoch += 1


if __name__ == '__main__':
    faulthandler.enable()
    args = parse_args()
    main(args)
