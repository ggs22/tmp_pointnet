"""
Author: Benny
Date: Nov 2019
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

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset
import metrics.metrics as m

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
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA'''
    root = args.data_root

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = args.num_classes
    num_part = args.num_parts

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.GetModel(num_part, normal_channel=args.normal, channels_offset=args.channel_offset).cuda()
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

    try:
        checkpoint_path = str(exp_dir) + '/checkpoints/best_model.pth'
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Use pretrain model from: {checkpoint_path}')
    except:
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

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    best_recall = 0
    best_loss = torch.inf
    global_epoch = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        batchwise_accuracies = list()
        batchwise_recalls = list()
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
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            acc = m.accuracy(pred_choice, target)
            rc = m.recall(pred_choice, target)

            batchwise_accuracies.append(acc)
            batchwise_recalls.append(rc)

            loss = criterion(seg_pred, target, trans_feat)
            batchwise_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            ''' lets output some sample for visialization '''
            if i % 5 == 0:
                tx = points.transpose(1, 2).cpu().numpy().squeeze()
                ty = pred_choice.cpu().numpy().reshape((*pred_choice.shape, 1))
                np.savetxt(fname=f'{exp_dir}{os.path.sep}train_sample_{i}.txt',
                           X=np.concatenate([tx, ty], axis=1))

        train_mean_acc = np.mean(batchwise_accuracies)
        train_mean_recall = np.mean(batchwise_recalls)
        train_mean_loss = np.mean(batchwise_losses)
        log_string(f'\nMean train accuracy: {train_mean_acc:.5f}\n'
                   f'mean train recall: {train_mean_recall:.5f}\n'
                   f'mean train loss: {train_mean_loss:.5f}')

        batchwise_accuracies.clear()
        batchwise_recalls.clear()
        batchwise_losses.clear()

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    ''' lets output some sample for visialization '''
                    if batch_id % 2 == 0:
                        tx = points.transpose(1, 2).cpu().numpy().squeeze()
                        ty = cur_pred_val.reshape((cur_pred_val.shape[1], 1))
                        np.savetxt(fname=f'{exp_dir}{os.path.sep}test_sample_{batch_id}.txt',
                                   X=np.concatenate([tx, ty], axis=1))

                accuracy = np.sum(cur_pred_val == target)
                total_correct += accuracy
                total_seen += (cur_batch_size * NUM_POINT)

                loss = criterion(seg_pred.squeeze().cuda(), torch.tensor(target.squeeze()).cuda(), trans_feat)
                batchwise_accuracies.append(m.accuracy(torch.tensor(cur_pred_val), torch.tensor(target)))
                batchwise_recalls.append(m.recall(torch.tensor(cur_pred_val), torch.tensor(target)))
                batchwise_losses.append(loss.item())

                for label in range(num_part):
                    total_seen_class[label] += np.sum(target == label)
                    total_correct_class[label] += (np.sum((cur_pred_val == label) & (target == label)))

                for i in range(cur_batch_size):
                    seg_pred = cur_pred_val[i, :]
                    seg_lbl = target[i, :]
                    cat = seg_label_to_cat[seg_lbl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for label in seg_classes[cat]:
                        if (np.sum(seg_lbl == label) == 0) and (
                                np.sum(seg_pred == label) == 0):  # part is not present, no prediction as well
                            part_ious[label - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[label - seg_classes[cat][0]] = np.sum((seg_lbl == label) & (seg_pred == label)) / float(
                                np.sum((seg_lbl == label) | (seg_pred == label)))
                            # tst = m.get_iou(pred=seg_pred, target=seg_lbl, label=label)
                    shape_ious[cat].append(np.mean(part_ious[1:]))  # we just want the IoU of the non-background classes

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['recall'] = np.mean(batchwise_recalls)
            test_metrics['loss'] = np.mean(batchwise_losses)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

        log_string(f"Epoch: {epoch+1} "
                   f"test Accuracy: {test_metrics['accuracy']}, "
                   f"test Recall: {test_metrics['recall']}, "
                   f"test Loss: {test_metrics['loss']}, "
                   f"Class avg mIOU: {test_metrics['class_avg_iou']}, "
                   f"Instance avg mIOU: {test_metrics['instance_avg_iou']}")

        if test_metrics['loss'] < best_loss:
            logger.info(f"Save model with new best loss {test_metrics['loss']}...")
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string(f'Saving at {savepath}')
            state = {
                'epoch': epoch,
                'train_acc': train_mean_acc,
                'train_recall': train_mean_recall,
                'train_loss': train_mean_loss,
                'test_acc': test_metrics['accuracy'],
                'test_recall': test_metrics['recall'],
                'test_loss': test_metrics['loss'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        ''' Update best metrics in display '''
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            log_string(f'New best accuracy is: {best_acc:.5f}')
        if test_metrics['recall'] > best_recall:
            best_recall = test_metrics['recall']
            log_string(f'New best recall is: {best_recall:.5f}')
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            log_string(f'New best loss is: {loss:.5f}')
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
            log_string(f'New best class avg mIOU is: {best_class_avg_iou:.5f}')
        if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
            best_instance_avg_iou = test_metrics['instance_avg_iou']
            log_string(f'New best inctance avg mIOU is: %.5f' % best_instance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
