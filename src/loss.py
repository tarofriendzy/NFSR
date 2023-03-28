"""
损失函数
"""
import torch


def euclidean_distance(x, y):
    """
    欧几里德距离的平方
    """
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    """
    近似汉明相似度。
    """
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    """
    计算损失
    :param x: [N, D] float tensor
    :param y: [N, D] float tensor
    :param labels: [N] int tensor，如果 x[i] 和 y[i] 是相似的，则labels[i] = +1, 否则为 -1
    :param loss_type: 损失的类型，可选值{margin, hamming}
    :param margin:float scalar
    :return:
        loss: [N] float tensor
    """
    labels = labels.float()
    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)
