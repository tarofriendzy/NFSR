from sklearn import metrics
from loss import *


def exact_hamming_similarity(x, y):
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(config, x, y):

    if config['graph_match']['training']['loss'] == 'margin':

        return -euclidean_distance(x, y)
    elif config['graph_match']['training']['loss'] == 'hamming':
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):

    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)
