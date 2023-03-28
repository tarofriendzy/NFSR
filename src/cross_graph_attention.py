import torch

def pairwise_euclidean_similarity(x, y):
    """
    计算 x 和 y 之间的成对欧几里得
    此函数计算每对 x_i 和 y_j 之间的以下相似度值: s(x_i, y_j) = -|x_i - y_j|^2.
    :param x: NxD float tensor.
    :param y: MxD float tensor.
    :return:
        s: NxM float tensor, 成对的欧几里得相似度
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """
    计算 x 和 y 之间的点积
    此函数计算每对 x_i 和 y_j 之间的以下相似度值：s(x_i,y_j)=x_i^T y_j。
    :param x: NxD float tensor.
    :param y: MxD float tensor.
    :return:
        s: NxM float tensor, 成对的点积相似性
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """
    计算 x 和 y 之间的余弦
    此函数计算每对 x_i 和 y_j 之间的以下相似度值: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).
    :param x: NxD float tensor.
    :param y: MxD float tensor.
    :return:
        s: NxM float tensor, 成对的余弦相似度
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {'euclidean': pairwise_euclidean_similarity,
                                'dotproduct': pairwise_dot_product_similarity,
                                'cosine': pairwise_cosine_similarity}


def get_pairwise_similarity(name):
    """
    按名称获取成对相似度指标
    :param name: string, 相似性指标的名称，{dot-product, cosine,  euclidean}.
    :return:
        similarity: a (x, y) -> sim function.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    """
    计算交叉注意力
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))

    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    :param x: NxD float tensor.
    :param y: MxD float tensor.
    :param sim: 相似性指标
    :return:
        attention_x: NxD float tensor.
        attention_y: NxD float tensor.
    """

    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)
    a_y = torch.softmax(a, dim=0)
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_block_pair_attention(data, graph_idx, n_graphs, similarity='dotproduct'):
    """
    计算交叉图注意力
    此函数根据 block_idx 将批处理数据划分为块。
    对于每一对块，x = data[graph_idx == 2i], 并且 y = data[graph_idx == 2i+1],
    计算
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))

    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    并且
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.
    :param data: NxD float tensor.
    :param graph_idx: N-dim int tensor. 每个节点中记录的图的id
    :param n_graphs: integer. 图的数量
    :param similarity: a string, 相似度
    :return:
        attention_output: NxD float tensor, 每个 x_i 都由 tention_x_i 代替。
    """
    if not isinstance(n_graphs, int):
        raise ValueError('n_graphs (%s) has to be an integer.' % str(n_graphs))

    if n_graphs % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_graphs)

    # 相似性指标
    sim = get_pairwise_similarity(similarity)

    # 将节点按图拆分
    partitions = []
    for i in range(n_graphs):
        partitions.append(data[graph_idx == i, :])

    # 计算交叉图注意力
    results = []
    for i in range(0, n_graphs, 2):
        g1 = partitions[i]
        g2 = partitions[i + 1]
        attention_g1, attention_g2 = compute_cross_attention(g1, g2, sim)
        results.append(attention_g1)
        results.append(attention_g2)

    results = torch.cat(results, dim=0)
    return results