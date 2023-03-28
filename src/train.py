"""
模型
"""
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import collections
from config import get_default_config
from dataset import build_label, build_dataset
from loss import pairwise_loss
from evaluation import compute_similarity, auc
from cross_graph import GAT


def build_model(config):
    """
    创建模型
    :param config:
    :return:
    """
    model_conf = config['cross_graph']


    model = GAT(**model_conf['params'])


    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=model_conf['training']['learning_rate'],
                                 weight_decay=model_conf['training']['weight_decay'])

    return model, optimizer


def get_graph(graph, labels):
    """
    获取图的信息
    :param batch:
    :return:
    """
    node_features = torch.from_numpy(graph.node_features)
    edge_features = torch.from_numpy(graph.edge_features)
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels


def reshape_and_split_tensor(tensor, n_splits):
    """
    沿最后一个维度重塑并分割2D张量。
    :param tensor: [num_examples, feature_dim]，num_examples必须是`n_splits`的倍数
    :param n_splits:将张量分割成的分割数， int
    :return:
        splits：一个n_splits张量的列表。第一个分割为[tensor[0], tensor[n_splits], tensor[n_splits * 2], ...]
                                    第二个分割为[tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...]
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def set_environment(config):
    """
    设置环境
    :param config:
    :return:
    """
    # 设置 GPU
    use_cuda = torch.cuda.is_available()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    seed = config['graph_match']['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # Pytorch 提速
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    return device


def record_metrics(accumulated_metrics, sim_pos, sim_neg, loss):
    """
    记录指标
    :return:
    """
    sim_diff = sim_pos - sim_neg
    accumulated_metrics['loss'].append(loss)
    accumulated_metrics['sim_pos'].append(sim_pos)
    accumulated_metrics['sim_neg'].append(sim_neg)
    accumulated_metrics['sim_diff'].append(sim_diff)


def execute_train(config):
    """
    进行训练
    :return:
    """
    graph_match_conf = config['graph_match']

    device = set_environment(config)

    # 构建模型
    model, optimizer = build_model(config)
    model.to(device)

    # 构建数据集
    x_train, y_train, x_test, y_test = build_dataset(config)

    # 记录每一轮每一批次的指标
    accumulated_metrics = collections.defaultdict(list)

    # 进行训练
    t_start = time.time()
    for i_iter in range(graph_match_conf['training']['n_training_steps']):
        model.train(mode=True)
        for i in range(0, len(x_train)):
            batch_graph = x_train[i]
            batch_label = y_train[i]
            # 解包数据
            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch_graph, batch_label)

            labels = labels.to(device)
            graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                  to_idx.to(device), graph_idx.to(device), len(labels) * 2)

            # 计算损失
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            loss = pairwise_loss(x, y, labels, loss_type=graph_match_conf['training']['loss'],
                                 margin=graph_match_conf['training']['margin'])

            # 记录指标
            is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
            is_neg = 1 - is_pos
            n_pos = torch.sum(is_pos)
            n_neg = torch.sum(is_neg)
            sim = compute_similarity(config, x, y)
            sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
            sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)
            record_metrics(accumulated_metrics, sim_pos, sim_neg, loss)

            graph_vec_scale = torch.mean(graph_vectors ** 2)
            if graph_match_conf['training']['graph_vec_regularizer_weight'] > 0:
                loss += (graph_match_conf['training']['graph_vec_regularizer_weight'] * 0.5 * graph_vec_scale)

            # 进行优化
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            nn.utils.clip_grad_value_(model.parameters(), graph_match_conf['training']['clip_value'])
            optimizer.step()

        # 打印训练信息
        if (i_iter + 1) % graph_match_conf['training']['print_after'] == 0:
            metrics_to_print = {k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
            info_str = ', '.join(['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])

            # 重置指标
            accumulated_metrics = collections.defaultdict(list)
            # 进行模型评估
            if ((i_iter + 1) // graph_match_conf['training']['print_after'] %
                    graph_match_conf['training']['eval_after'] == 0):
                model.eval()
                with torch.no_grad():
                    accumulated_pair_auc = []
                    for i in range(0, len(x_test)):
                        batch = x_test[i]
                        label = y_test[i]
                        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch, label)
                        labels = labels.to(device)
                        eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                           to_idx.to(device), graph_idx.to(device), len(labels) * 2)

                        # 评估结果
                        x, y = reshape_and_split_tensor(eval_pairs, 2)
                        similarity = compute_similarity(config, x, y)
                        pair_auc = auc(similarity, labels)
                        accumulated_pair_auc.append(pair_auc)

                    eval_metrics = {'pair_auc': np.mean(accumulated_pair_auc)}
                    info_str += ', ' + ', '.join(['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
                model.train()
            print('iter %d, %s, time %.2fs' % (i_iter + 1, info_str, time.time() - t_start))
            t_start = time.time()


if __name__ == '__main__':
    config = get_default_config()
    build_label(config)
    execute_train(config)
