
import io
import os
import re
import collections
import numpy as np
from config import get_default_config


def build_label(config):

    filename_conf = config['file_names']
    output_dir = config['output_dir']

    binary_output_path_file = os.path.join(output_dir, filename_conf['binary_output_path'])
    if not os.path.exists(binary_output_path_file) or not os.path.isfile(binary_output_path_file):
        raise ValueError('File error')

    binary_output_paths = []
    filenames = []
    binary_dataset_type = []
    dataset_type = config['graph_match']['dataset_type']
    for line in io.open(binary_output_path_file, 'r'):
        path_str = re.findall(r"^(.*)", line)[0]
        if len(path_str) > 0:
            if os.path.exists(path_str) and os.path.isdir(path_str):
                path, filename = os.path.split(path_str)
                binary_output_paths.append(path_str)
                filenames.append(filename)
                for i in range(len(dataset_type)):
                    if os.sep + dataset_type[i] + os.sep in path_str:
                        binary_dataset_type.append(i)

    if len(binary_output_paths) == 0:
        raise ValueError('file error')

    label_writer = io.open(os.path.join(output_dir, filename_conf['graph_match_labels']), 'w')
    for i in range(len(binary_output_paths)):
        for j in range(i+1, len(binary_output_paths)):
            if filenames[i] == filenames[j] and binary_dataset_type[i] == binary_dataset_type[j]:
                label_writer.write(str(1) + '\n')
            else:
                label_writer.write(str(-1) + '\n')
            label_writer.write(binary_output_paths[i] + '\n')
            label_writer.write(binary_output_paths[j] + '\n\n')


def read_label(config):
    filename_conf = config['file_names']
    label_file = os.path.join(config['output_dir'], filename_conf['graph_match_labels'])
    if not os.path.exists(label_file) or not os.path.isfile(label_file):
        raise ValueError('label error')

    edge_filename = filename_conf['edge_list']
    node_embedding = filename_conf['node_index_to_embedding']

    labels = []
    graphs = []
    line_mum = 1
    label = None
    g1_path = None
    g2_path = None
    for line in open(label_file, 'r'):
        if line_mum % 4 == 1:
            label = re.findall(r"^(.*)", line)[0]
        elif line_mum % 4 == 2:
            g1_path = re.findall(r"^(.*)", line)[0]
        elif line_mum % 4 == 3:
            g2_path = re.findall(r"^(.*)", line)[0]
        else:
            g1_edge_file = os.path.join(g1_path, edge_filename)
            g1_node_file = os.path.join(g1_path, node_embedding)
            g2_edge_file = os.path.join(g2_path, edge_filename)
            g2_node_file = os.path.join(g2_path, node_embedding)
            if os.path.exists(g1_edge_file) and os.path.isfile(g1_edge_file) \
                and os.path.exists(g1_node_file) and os.path.isfile(g1_node_file) \
                and os.path.exists(g2_edge_file) and os.path.isfile(g2_edge_file) \
                and os.path.exists(g2_node_file) and os.path.isfile(g2_node_file):
                labels.append(int(label))
                graphs.append((g1_path, g2_path))
        line_mum += 1

    if len(labels) == 0 or len(graphs) == 0:
        raise ValueError('请先构建二进制文件的信息和块嵌入')
    return graphs, labels


def read_node_embedding(g_path, config):
    node_embedding_dict = dict()
    line_mum = 1
    index = None
    embedding = None
    for line in open(os.path.join(g_path, config['file_names']['node_index_to_embedding']), 'r'):
        if line_mum % 3 == 1:
            index = re.findall(r"^(\d+):", line)[0]
            index = int(index)
        elif line_mum % 3 == 2:
            embedding = re.findall(r"^\[(.*)\]", line)[0]
            embedding = [float(x) for x in embedding.split(',')]
        else:
            node_embedding_dict[index] = embedding

        line_mum += 1

    return node_embedding_dict


def read_edge_list(g_path, config):
    from_nodes = []
    to_nodes = []
    nodes = []
    for line in open(os.path.join(g_path, config['file_names']['edge_list']), 'r'):
        from_node, to_node = line.strip().split()[:2]
        from_node = int(from_node)
        to_node = int(to_node)

        from_nodes.append(from_node)
        to_nodes.append(to_node)

        if from_node not in nodes:
            nodes.append(from_node)
        if to_node not in nodes:
            nodes.append(to_node)

    nodes.sort()
    return from_nodes, to_nodes, nodes


def read_node_info(g_path, config):
    from_nodes, to_nodes, nodes = read_edge_list(g_path, config)
    node_feature_dim = config['graph_match']['encoder']['node_feature_dim']

    node_embedding_dict = read_node_embedding(g_path, config)
    node_features = []
    keys = node_embedding_dict.keys()
    for node_index in range(0, nodes[-1]+1):
        if node_index in keys:
            node_features.append(node_embedding_dict[node_index])
        else:
            node_features.append([1.0] * node_feature_dim)

    return np.array(from_nodes), np.array(to_nodes), np.array(node_features, dtype=np.float32)


def pack_graph_data(graph_pair_list, config):

    graphs = []
    for graph in graph_pair_list:
        for inergraph in graph:
            graphs.append(inergraph)

    from_idx = []
    to_idx = []
    graph_idx = []
    all_node_features = []

    n_total_nodes = 0
    n_total_edges = 0
    for i, g_path in enumerate(graphs):
        from_nodes, to_nodes, node_features = read_node_info(g_path, config)

        n_nodes = len(node_features)
        n_edges = len(from_nodes)

        from_idx.append(from_nodes + n_total_nodes)
        to_idx.append(to_nodes + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
        all_node_features.append(node_features)

        n_total_nodes += n_nodes
        n_total_edges += n_edges

    GraphData = collections.namedtuple('GraphData',
                                       ['from_idx', 'to_idx', 'node_features',
                                        'edge_features', 'graph_idx', 'n_graphs'])

    return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        node_features=np.concatenate(all_node_features, axis=0),
        edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
        graph_idx=np.concatenate(graph_idx, axis=0),
        n_graphs=len(graphs),
    )


def build_dataset(config):

    graph_match_conf = config['graph_match']

    graphs, labels = read_label(config)

    batch_graphs = []
    batch_labels = []
    graphs_num = len(graphs)
    batch_size = min(graph_match_conf['batch_size'], len(graphs))
    for i in range(0, graphs_num, batch_size):
        end = min(i + batch_size, graphs_num)

        x_batch = pack_graph_data(graphs[i:end], config)
        y_batch = np.array(labels[i:end], dtype=np.int32)

        batch_graphs.append(x_batch)
        batch_labels.append(y_batch)

    split_index = int(np.rint(graph_match_conf['train_proportion'] * len(batch_graphs)))
    x_train, x_test = batch_graphs[:split_index], batch_graphs[split_index:]
    y_train, y_test = batch_labels[:split_index], batch_labels[split_index:]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    config = get_default_config()
    build_label(config=config)
    build_dataset(config)
