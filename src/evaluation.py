from sklearn import metrics
from loss import *


def exact_hamming_similarity(x, y):
    """
    计算汉明距离
    """
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(config, x, y):
    """
    计算 x 和 y 向量之间的距离。距离将根据训练损失类型进行计算。
    :param config:
    :param x: [n_examples, feature_dim] float tensor
    :param y: [n_examples, feature_dim] float tensor
    :return:
        dist: [n_examples] float tensor
    """
    if config['graph_match']['training']['loss'] == 'margin':
        # 相似度是负距离
        return -euclidean_distance(x, y)
    elif config['graph_match']['training']['loss'] == 'hamming':
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
    """
    计算AUC以进行配对分类。

    Args:
      scores: [n_examples] float.  分数越高，表示被分配+1标签的偏好越高。
      labels: [n_examples] int.  标签为+1或-1。
      **auc_args: tf.metrics.auc可以使用的其他参数。

    Returns:
      auc: ROC曲线下的面积。
    """
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # 将分数归一化为[0，1]，并添加一个小epsilon以确保安全
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)
