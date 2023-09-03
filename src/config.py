
import os


def get_default_config():
    name = 'binary diff'

    binary_dir = '/Work/BinaryDiff/ProjectData/ExperimentData/test/'
    output_dir = '/Work/BinaryDiff/ProjectData/Output//test/'

    if not os.path.exists(binary_dir):
        raise ValueError('Config is error')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word2vec_embedding_dim = 128

    model_type = 'cross_graph'
    node_state_dim = 32
    graph_rep_dim = 128
    return dict(
        name=name,
        binary_dir=os.path.normpath(binary_dir),
        output_dir=os.path.normpath(output_dir),
        file_names=dict(
            edge_list='edge_list',
            node_index_to_code='node_index_to_code',
            article='article',
            binary_output_path='binary_output_path',
            word2vec_train_data='word2vec_train_data',
            word2vec_vocab_data='word2vec_vocab_data',
            word2vec_vectors='word2vec_vectors.tsv',
            word2vec_metadata='word2vec_metadata.tsv',
            word2vec_token_embedding='word2vec_token_embedding',
            node_index_to_embedding='node_index_to_embedding',
            graph_match_labels='graph_match_labels',
        ),
        walk=dict(
            number_walks=2,
            walk_length=5,
            seed=2,
            alpha=0,
        ),
        word2vec=dict(
            sequence_length=20,
            window_size=10,
            num_ns=5,
            seed=42,
            embedding_dim=word2vec_embedding_dim,
            num_epochs=20,
            batch_size=100,
            learning_rate=1.0,
        ),
        cross_graph=dict(
            seed=8,
            dataset_type=['coreutils', 'diffutils', 'findutils', 'test'],
            model_type=model_type,
            train_proportion=0.8,
            batch_size=2,
            encoder=dict(

                node_hidden_sizes=[node_state_dim],
                node_feature_dim=word2vec_embedding_dim,
                edge_feature_dim=1,

                edge_hidden_sizes=None,
            ),
            aggregator=dict(
                node_hidden_sizes=[graph_rep_dim],
                graph_transform_sizes=[graph_rep_dim],
                input_size=[node_state_dim],
                gated=True,
                aggregation_type='sum',
            ),
            cross_graph = dict(
                node_state_dim=node_state_dim,
                edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
                node_hidden_sizes=[node_state_dim * 2],
                n_prop_layers=5,

                share_prop_params=True,

                edge_net_init_scale=0.1,
                node_update_type='gru',
                use_reverse_direction=True,
                reverse_dir_param_different=False,
                prop_type=model_type
            ),
            training=dict(
                learning_rate=1e-4,
                weight_decay=1e-5,
                loss='margin',
                margin=1.0,
                graph_vec_regularizer_weight=1e-6,
                clip_value=10.0,
                n_training_steps=50,
                print_after=2,
                eval_after=2
            )
        )
    )
