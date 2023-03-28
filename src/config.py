"""
配置信息
"""
import os


def get_default_config():
    """
    获取默认的配置信息
    Returns:
    """
    name = 'binary diff'
    # binary_dir = 'D://Work//BinaryDiff//ProjectData//ExperimentData//'
    # output_dir = 'D://Work//BinaryDiff//ProjectData//Output//'

    # binary_dir = 'D://Work//BinaryDiff//ProjectData//ExperimentData//findutils//'
    # output_dir = 'D://Work//BinaryDiff//ProjectData//Output//findutils//'

    binary_dir = 'D://Work//BinaryDiff//ProjectData//ExperimentData//test//'
    output_dir = 'D://Work//BinaryDiff//ProjectData//Output//test//'

    if not os.path.exists(binary_dir):
        raise ValueError('请正确配置二进制文件路径')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word2vec_embedding_dim = 128  # 嵌入的维度

    model_type = 'cross_graph'  # 模型的类型, 可选值{matching, embeddin}
    node_state_dim = 32  # 节点状态的维度
    graph_rep_dim = 128  # 图表示的维度
    # 向量空间相似性指标。可选值{dotproduct, euclidean, cosine}
    return dict(
        name=name,
        binary_dir=os.path.normpath(binary_dir),
        output_dir=os.path.normpath(output_dir),
        file_names=dict(
            edge_list='edge_list',  # 边的信息文件
            node_index_to_code='node_index_to_code',  # 节点与tokens信息的文件
            article='article',  # 随机游走形成的文章的文件
            binary_output_path='binary_output_path',  # 保存二进制文件输出路径的目录的文件
            word2vec_train_data='word2vec_train_data',  # 保存Word2Vec的训练数据
            word2vec_vocab_data='word2vec_vocab_data',  # 保存Word2Vec的词汇表的文件
            word2vec_vectors='word2vec_vectors.tsv',  # 嵌入的向量文件
            word2vec_metadata='word2vec_metadata.tsv',  # 嵌入对应的词汇文件
            word2vec_token_embedding='word2vec_token_embedding',  # Token 嵌入
            node_index_to_embedding='node_index_to_embedding',  # Token 嵌入
            graph_match_labels='graph_match_labels',  # 图匹配时的标签
        ),
        walk=dict(
            number_walks=2,  # 每个节点的游走的次数
            walk_length=5,  # 每次随机游走的最大长度
            seed=2,  # 随机种子
            alpha=0,  # 只选择第一个路径的概率
        ),
        word2vec=dict(
            sequence_length=20,  # 句子的长度
            window_size=10,  # 窗口大小
            num_ns=5,  # 每个正采样对应的负采样数量
            seed=42,
            embedding_dim=word2vec_embedding_dim,  # 嵌入的维度
            num_epochs=20,  # 训练的轮数
            batch_size=100,  # 训练的批的大小
            learning_rate=1.0,  # 学习速率
        ),
        cross_graph=dict(
            seed=8,
            dataset_type=['coreutils', 'diffutils', 'findutils', 'test'],
            model_type=model_type,  # 模型的类型, 可选值{matching, embeddin}
            train_proportion=0.8,  # 训练数据所占的比例
            batch_size=2,  # 批处理大小
            encoder=dict(
                # 节点编码器隐藏层的节点数量，最后一个元素是输出的大小,设置为 None 时不对节点特征做任何操作
                node_hidden_sizes=[node_state_dim],
                node_feature_dim=word2vec_embedding_dim,  # 节点特征的维度
                edge_feature_dim=1,  # 边的特征维度
                # 边编码器隐藏层的节点数量，最后一个元素是输出的大小,设置为 None 时不对边特征做任何操作
                edge_hidden_sizes=None,
            ),
            aggregator=dict(
                node_hidden_sizes=[graph_rep_dim],  # 聚合器隐藏层的节点数量，最后一个元素是输出的大小
                graph_transform_sizes=[graph_rep_dim],  # 转换层的节点数量，最后一个元素表示图表示的最终维度
                input_size=[node_state_dim],  # 输入的维度
                gated=True,  # 是否进行门控聚合
                aggregation_type='sum',  # 聚合的类型，可选值{sum, max, mean, sqrt_n}
            ),
            cross_graph = dict(
                node_state_dim=node_state_dim,  # 节点状态的维度
                edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
                node_hidden_sizes=[node_state_dim * 2],
                n_prop_layers=5,  # 图传播层数的数量
                # 设置为 True 时可再所有图传播层之间共享传播参数，设置为 False 将不跨消息传递层共享参数
                share_prop_params=True,
                # 用较小的参数权重初始化 message MLP 以防止聚合的 message 向量爆炸，也可以使用例如 层规范化以控制这些规模
                edge_net_init_scale=0.1,
                node_update_type='gru',  # 更新节点的神经网络类型，，可选值{mlp, gru, residual}
                use_reverse_direction=True,  # 设置为 True 时，可反向传播 message
                reverse_dir_param_different=False, # 设置为 True 时可使用与正向传播不同的参数来计算 message
                prop_type=model_type
            ),
            training=dict(
                learning_rate=1e-4,  # 学习速率
                weight_decay=1e-5,  # 权值衰减系数，即L2正则项的系数
                loss='margin',  # 损失的计算，可选值{margin, hamming}
                margin=1.0,
                graph_vec_regularizer_weight=1e-6,  # 图向量上的一个小正则化器可缩放以避免图向量爆炸。
                clip_value=10.0, # 添加渐变裁剪以避免大的渐变。
                n_training_steps=50,
                print_after=2,  # 打印训练信息之间相隔的训练步骤
                eval_after=2  # 在每个`eval_after * print_after`步骤中评估验证集。
            )
        )
    )
