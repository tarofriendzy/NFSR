import torch
import torch.nn as nn
from segment import unsorted_segment_sum


class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, node_hidden_sizes=None,
                 edge_hidden_sizes=None, name='graph-encoder'):
        super(GraphEncoder, self).__init__()

        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes

        self._build_model()

    def _build_model(self):
        if self._node_hidden_sizes is not None:
            layers = [nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0])]
            for i in range(1, len(self._node_hidden_sizes)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self._node_hidden_sizes[i-1], self._node_hidden_sizes[i]))
            self.MLP1 = nn.Sequential(*layers)
        else:
            self.MLP1 = None

        if self._edge_hidden_sizes is not None:
            layers = [nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0])]
            for i in range(1, len(self._edge_hidden_sizes)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self._edge_hidden_sizes[i-1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layers)
        else:
            self.MLP2 = None

    def forward(self, node_feature=None, edge_features=None):
        if self.MLP1 is None:
            node_outputs = node_feature
        else:
            node_outputs = self.MLP1(node_feature)

        if self.MLP2 is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs


def graph_prop_once(node_states, from_idx, to_idx, message_net, edge_features=None):
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)

    message = message_net(edge_inputs)

    tensor = unsorted_segment_sum(message, to_idx, node_states.shape[0])

    return tensor


class GraphPropLayer(nn.Module):
    """
    图传播层（message 传递）
    """
    def __init__(self, node_state_dim, edge_hidden_sizes, node_hidden_sizes,
                 edge_net_init_scale=0.1, node_update_type='residual',
                 use_reverse_direction=True, reverse_dir_param_different=True,
                 prop_type='embedding', name='graph-prop-layer'):
        """
        初始化
        :param node_state_dim: 节点状态的维度
        :param edge_hidden_sizes:
        :param node_hidden_sizes:
        :param edge_net_init_scale: message初始化的规模
        :param node_update_type: 更新节点的神经网络类型，可选值{mlp, gru, residual}
        :param use_reverse_direction: 设置为 True 时，可反向传播 message
        :param reverse_dir_param_different: 设置为 True 时可使用与正向传播不同的参数来计算 message
        :param prop_type:
        :param name:
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]
        # 输出大小为node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._prop_type = prop_type

        self._build_model()

    def _build_model(self):
        """
        构建模型
        _message_net: 每一条连边都通过一个神经网络生成一个message
        _reverse_message_net: 反向传播生成一个message(参数与正向传播共享/不共享)
        GRU/MLP: 更新节点的网络
        :return:
        """
        # 产生message的网络
        layers = [nn.Linear(self._edge_hidden_sizes[0] + 1, self._edge_hidden_sizes[0])]
        for i in range(1, len(self._edge_hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layers)

        # 反向传播 message
        if self._use_reverse_direction:
            # 与正向传播不同的参数
            if self._reverse_dir_param_different:
                layers = [nn.Linear(self._edge_hidden_sizes[0] + 1, self._edge_hidden_sizes[0])]
                for i in range(1, len(self._edge_hidden_sizes)):
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layers)
            else:  # 与正向传播共享参数
                self._reverse_message_net = self._message_net

        # 更新节点的网络
        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layers = []
            if self._prop_type == 'embedding':
                layers.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layers.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layers)

    def _compute_aggregated_messages(self, node_states, from_idx, to_idx, edge_features=None):
        """
        计算每个节点的聚合 message
        :param node_states: 节点状态, [n_nodes, input_node_state_dim] float tensor
        :param from_idx: from节点索引
        :param to_idx: to节点索引
        :param edge_features: 边的特征，[n_edges，edge_embedding_dim] float tensor
        :return:
            aggregated_messages: 每个节点的聚合 message，[n_nodes, aggregated_message_dim] float tensor
        """
        aggregated_messages = graph_prop_once(node_states, from_idx, to_idx, self._message_net,
                                              edge_features=edge_features)

        # 反向计算message
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(node_states, to_idx, from_idx, self._reverse_message_net,
                                                          edge_features=edge_features)
            aggregated_messages += reverse_aggregated_messages

        return aggregated_messages

    def _compute_node_update(self, node_states, node_state_inputs, node_features=None):
        """
        更新节点
        :param node_states: 输入节点状态，[n_nodes, node_state_dim] float tensor
        :param node_state_inputs:用于计算节点更新的张量列表，[n_nodes，feat_dim]
        :param node_features:额外的节点特征，[n_nodes，extra_node_feat_dim] float tensor
        :return:
            new_node_states: 新的节点状态张量, [n_nodes, node_state_dim] float tensor
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)

        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self, node_states, from_idx, to_idx, edge_features=None, node_features=None):
        """

        :param node_states: 节点状态，[n_nodes, input_node_state_dim] float tensor
        :param from_idx: [n_edges] int tensor, from节点索引 node -> ***
        :param to_idx: [n_edges] int tensor, to节点索引 *** -> node
        :param edge_features: 边的特征， [n_edges, edge_embedding_dim] float tensor
        :param node_features: 额外的节点特征，[n_nodes, extra_node_feat_dim] float tensor
        :return:
            node_states: 新的节点状态，[n_nodes, node_state_dim] float tensor
        """

        # 聚合 message
        aggregated_messages = self._compute_aggregated_messages(node_states, from_idx, to_idx,
                                                                edge_features=edge_features)

        return self._compute_node_update(node_states, [aggregated_messages], node_features=node_features)


class GraphAggregator(nn.Module):
    """
    聚合器
    """
    def __init__(self, node_hidden_sizes, graph_transform_sizes=None, input_size=None,
                 gated=True, aggregation_type='sum', name='graph-aggregator'):
        """
        初始化
        :param node_hidden_sizes: 隐藏层节点的数量，最后一个元素表示输出大小
        :param graph_transform_sizes: 转换层的节点数量，最后一个元素表示图表示的最终维度
        :param input_size: # 节点输入的大小
        :param gated: 设置为 True 表示进行门控聚合，否则设置为 False。
        :param aggregation_type: 可选值{sum, max, mean, sqrt_n}
        :param name: 模块名称
        """
        super(GraphAggregator, self).__init__()
        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._input_size = input_size
        self._gated = gated
        self._aggregation_type = aggregation_type

        self._graph_state_dim = node_hidden_sizes[-1]
        self._aggregation_op = None

        self.MLP1, self.MLP2 = self._build_model()

    def _build_model(self):
        """
        构造模型
        MLP1: 节点状态
        MLP2: 转换层
        :return:
        """
        node_hidden_sizes = self._node_hidden_sizes

        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layers = [nn.Linear(self._input_size[0], node_hidden_sizes[0])]
        for i in range(1, len(node_hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layers)

        if self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0:
            layers = [nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0])]
            for i in range(1, len(self._graph_transform_sizes)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layers)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """
        计算图表示
        :param node_states: 沿着第一维连接在一起的一批图的节点状态，[n_nodes, node_state_dim] float tensor
        :param graph_idx: 每个节点的图ID, [n_nodes] int tensor
        :param n_graphs: 批中的图的数量数， integer
        :return:
            graph_states: 图表示形式，[n_graphs, graph_state_dim] float tensor
        """

        # 对节点使用带有门控矢量的加权求和
        node_states_g = self.MLP1(node_states)

        # 门控
        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates
        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == 'max':
            graph_states *= torch.FloatTensor(graph_states > -1e5)

        # 进一步转换简化后的图状态
        if (self._graph_transform_sizes is not None
                and len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states


class GraphEmbeddingNet(nn.Module):
    def __init__(self, encoder, aggregator, node_state_dim, edge_hidden_sizes, node_hidden_sizes, n_prop_layers,
                 share_prop_params=False, edge_net_init_scale=0.1, node_update_type='residual',
                 use_reverse_direction=True, reverse_dir_param_different=True,
                 layer_class=GraphPropLayer, prop_type='embedding', name='graph-embedding-net'):
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self._node_state_dim = node_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_class = layer_class
        self._prop_type = prop_type

        self._prop_layers = nn.ModuleList()  # 存储图传播层

        self._build_model()

    def _build_layer(self, layer_id):
        return self._layer_class(self._node_state_dim, self._edge_hidden_sizes, self._node_hidden_sizes,
                                 edge_net_init_scale=self._edge_net_init_scale,
                                 node_update_type=self._node_update_type,
                                 use_reverse_direction=self._use_reverse_direction,
                                 reverse_dir_param_different=self._reverse_dir_param_different,
                                 prop_type=self._prop_type)

    def _build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]

                self._prop_layers.append(layer)

    def _apply_layer(self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features):
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx, edge_features=edge_features)

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]
        for layer in self._prop_layers:
            node_states = self._apply_layer(layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features)
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs
        return self._aggregator(node_states, graph_idx, n_graphs)


