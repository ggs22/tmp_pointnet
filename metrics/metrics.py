import torch
import numpy as np


def accuracy(pred: torch.Tensor, target: torch.Tensor):
    return pred.eq(target.data).cpu().sum()/target.shape[len(target.shape)-1]


def get_true_positives(pred: torch.Tensor, target: torch.Tensor):
    return np.intersect1d(np.where(pred.data.cpu() == 1)[0],
                          np.where(target.data.cpu() == 1)[0]).shape[0]


def get_false_negatives(pred: torch.Tensor, target: torch.Tensor):
    return (pred.data.cpu()[(np.where(target.data.cpu() == 1)[0])] == 0).sum()


def recall(pred: torch.Tensor, target: torch.Tensor):
    tp = get_true_positives(pred, target)
    fn = get_false_negatives(pred, target)
    return tp / (tp + fn)


def euclidian_dist(pred: torch.Tensor, target: torch.Tensor):
    return torch.cdist(x1=pred, x2=target, p=2)


def get_iou(pred: np.ndarray, target: np.ndarray, label: int):
    intersect = np.intersect1d(np.where(pred == label)[0],
                               np.where(target == label)[0]).shape[0]
    union = np.union1d(np.where(pred == label)[0],
                               np.where(target == label)[0]).shape[0]
    iou = intersect/union if union > 0 else np.nan
    return iou
