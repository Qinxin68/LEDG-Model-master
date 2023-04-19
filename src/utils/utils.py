import pickle, json, decimal, math
from config import DEVICE
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F


def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))


def pair_prf_CR(pred_y_list, true_y_list, configs):
    """
    TODO: bayes
    """
    t_p, t_n, f_p, f_n = 0, 0, 0, 0
    for (pred_cou, true_m) in zip(pred_y_list, true_y_list):
        batch, N, _, _ = pred_cou.size()
        pred_cou = pred_cou.reshape(batch, N * N, 2)
        _, pred_cou = torch.max(pred_cou, 2)
        pred_cou = pred_cou.cuda().data.cpu().numpy()
        true_m = torch.stack(true_m, 0).reshape(batch, N*N)
        true_m = true_m.numpy()
        t_p = t_p + np.sum((true_m == 1) & (pred_cou == 1))
        t_n = t_n + np.sum((true_m == 0) & (pred_cou == 0))
        f_p = f_p + np.sum((true_m == 0) & (pred_cou == 1))
        f_n = f_n + np.sum((true_m == 1) & (pred_cou == 0))
    epsilon = 1e-9
    p = t_p / (f_p + t_p + epsilon)
    r = t_p / (f_n + t_p + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    return p, r, f1


def cal_prf(pred_y_list, true_y_list, configs):
    """
        TODO: bayes
    """
    t_p, t_n, f_p, f_n = 0, 0, 0, 0
    for (pred_y, true_y) in zip(pred_y_list, true_y_list):
        true_y = torch.LongTensor(true_y)
        _, pred_y = torch.max(pred_y, 2)
        _, true_y = torch.max(true_y, 2)
        # max_k = max((1,))
        # _, pred_y_0 = pred_y[0, :, :].topk(max_k, 1, True, True)
        # _, pred_y_1 = pred_y[1, :, :].topk(max_k, 1, True, True)
        # pred_y = torch.cat([pred_y_0, pred_y_1], 0).reshape(2, 5, 1).squeeze(-1)
        true_y = true_y.numpy()
        pred_y = pred_y.cuda().data.cpu().numpy()
        t_p = t_p + np.sum((true_y == 1) & (pred_y == 1))
        t_n = t_n + np.sum((true_y == 0) & (pred_y == 0))
        f_p = f_p + np.sum((true_y == 0) & (pred_y == 1))
        f_n = f_n + np.sum((true_y == 1) & (pred_y == 0))
    epsilon = 1e-9
    p = t_p / (f_p + t_p + epsilon)
    r = t_p / (f_n + t_p + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    return p, r, f1


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, pred, labels):
        # assert preds.dim()==2 and labels.dim()==1
        pred = pred.contiguous().view(-1, pred.size(-1))
        self.alpha = self.alpha.to(pred.device)
        pre_softmax = F.softmax(pred, dim=1)
        pre_loft = torch.log(pre_softmax)
        pre_softmax = pre_softmax.gather(1, labels.view(-1, 1))
        pre_loft = pre_loft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - pre_softmax), self.gamma), pre_loft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
