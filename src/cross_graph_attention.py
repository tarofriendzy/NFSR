import torch
from cross_graph import GraphEmbeddingNet, GraphPropLayer


def pairwise_euclidean_similarity(x, y):
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):

    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {'euclidean': pairwise_euclidean_similarity,
                                'dotproduct': pairwise_dot_product_similarity,
                                'cosine': pairwise_cosine_similarity}


def get_pairwise_similarity(name):
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)
    a_y = torch.softmax(a, dim=0)
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_block_pair_attention(data, graph_idx, n_graphs, similarity='dotproduct'):

    if not isinstance(n_graphs, int):
        raise ValueError('n_graphs (%s) has to be an integer.' % str(n_graphs))

    if n_graphs % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_graphs)

    sim = get_pairwise_similarity(similarity)

    partitions = []
    for i in range(n_graphs):
        partitions.append(data[graph_idx == i, :])

    results = []
    for i in range(0, n_graphs, 2):
        g1 = partitions[i]
        g2 = partitions[i + 1]
        attention_g1, attention_g2 = compute_cross_attention(g1, g2, sim)
        results.append(attention_g1)
        results.append(attention_g2)

    results = torch.cat(results, dim=0)
    return results

class GraphPropMatchingLayer(GraphPropLayer):

    def forward(self, node_states, from_idx, to_idx, graph_idx, n_graphs, similarity='dotproduct',
                edge_features=None, node_features=None):

        aggregated_messages = self._compute_aggregated_messages(node_states, from_idx, to_idx,
                                                                edge_features=edge_features)

        cross_graph_attention = batch_block_pair_attention(node_states, graph_idx, n_graphs, similarity=similarity)

        attention_input = node_states - cross_graph_attention

        return self._compute_node_update(node_states, [aggregated_messages, attention_input],
                                         node_features=node_features)


class GraphMatchingNet(GraphEmbeddingNet):

    def __init__(self, encoder, aggregator, node_state_dim, edge_hidden_sizes, node_hidden_sizes, n_prop_layers,
                 share_prop_params=False, edge_net_init_scale=0.1, node_update_type='residual',
                 use_reverse_direction=True, reverse_dir_param_different=True,
                 layer_class=GraphPropMatchingLayer, similarity='dotproduct', prop_type='embedding'):
        super(GraphMatchingNet, self).__init__(encoder, aggregator, node_state_dim, edge_hidden_sizes,
                                               node_hidden_sizes, n_prop_layers, share_prop_params=share_prop_params,
                                               edge_net_init_scale=edge_net_init_scale,
                                               node_update_type=node_update_type,
                                               use_reverse_direction=use_reverse_direction,
                                               reverse_dir_param_different=reverse_dir_param_different,
                                               layer_class=layer_class,
                                               prop_type=prop_type)
        self._similarity = similarity

    def _apply_layer(self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features):
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                     similarity=self._similarity, edge_features=edge_features)
