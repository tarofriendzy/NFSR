import os
import numpy as np
import matplotlib.pyplot as plt
from binary_file_analyze import read_cfg

import json


def get_software_dir(graph_dir, architecture, software, optimization):
    software_dir = []
    software_name = []
    for sf in software:
        for op in optimization:
            sf_dir = os.path.join(graph_dir, architecture, sf + op)
            software_dir.append(sf_dir)
            software_name.append(sf + op)

    return software_dir, software_name


def get_graphs(sofware_dir, software_name):
    graphs = []
    group = os.walk(sofware_dir)
    for path, dir_list, file_list in group:
        for i in range(len(file_list)):
            file_name = file_list[i]
            if file_name.endswith('.json'):
                binary_file = os.path.join(path, file_name)
                cfg = read_cfg(binary_file, software=software_name)
                graphs.extend(cfg)

    return graphs


def is_avaliable_cfg(graph):
    """
    判断图是否可用
    :param graph:
    :return:
    """
    # PLT条目
    if graph.is_plt:
        return False
    # Simprocedure
    if graph.is_simprocedure:
        return False
    # 没有节点
    if graph.node_num == 0:
        return False
    # 单节点并且指令数量较少(一条/没有)
    # if graph.node_num == 1 and len(graph.nodes[0]) <= 1:
    #     return False
    if graph.name.startswith('sub_'):
        return False

    return True

def avaliable_graph_num(graphs):
    """
    可用图的数量
    :param graphs:
    :return:
    """
    ava_graph_num = 0
    for graph_item in graphs:
        if is_avaliable_cfg(graph_item):
            ava_graph_num += 1
    return ava_graph_num

def show_node_num(graphs, output_file):
    node_num_info = {'<=5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0,
                     '26-30': 0, '31-35': 0, '36-40': 0, '41-45': 0, '46-50': 0,
                     '>=51': 0}
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for graph in graphs:
        if not is_avaliable_cfg(graph):
            continue
        node_num = graph.node_num
        if 1 <= node_num <= 5:
            node_num_info['<=5'] += 1
            data[0] += 1
        elif 6 <= node_num <= 10:
            node_num_info['6-10'] += 1
            data[1] += 1
        elif 11 <= node_num <= 15:
            node_num_info['11-15'] += 1
            data[2] += 1
        elif 16 <= node_num <= 20:
            node_num_info['16-20'] += 1
            data[3] += 1
        elif 21 <= node_num <= 25:
            node_num_info['21-25'] += 1
            data[4] += 1
        elif 26 <= node_num <= 30:
            node_num_info['26-30'] += 1
            data[5] += 1
        elif 31 <= node_num <= 35:
            node_num_info['31-35'] += 1
            data[6] += 1
        elif 36 <= node_num <= 40:
            node_num_info['36-40'] += 1
            data[7] += 1
        elif 41 <= node_num <= 45:
            node_num_info['41-45'] += 1
            data[8] += 1
        elif 46 <= node_num <= 50:
            node_num_info['46-50'] += 1
            data[9] += 1
        else:
            node_num_info['>=51'] += 1
            data[10] += 1

    print(node_num_info)
    labels = ['<=5', '6-10', '11-15', '16-20', '21-25',
              '26-30', '31-35', '36-40', '41-45', '46-50',
              '>=51']
    plt.bar(range(len(data)), data, tick_label=labels)

    # plt.show()
    plt.savefig(output_file, format='svg')

def show_node_num2(graphs, output_file):
    node_num_info = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    data = [0, 0, 0, 0, 0]
    for graph in graphs:
        if not is_avaliable_cfg(graph):
            continue
        node_num = graph.node_num
        if node_num == 1:
            node_num_info['1'] += 1
            data[0] += 1
        elif node_num == 2:
            node_num_info['2'] += 1
            data[1] += 1
        elif node_num == 3:
            node_num_info['3'] += 1
            data[2] += 1
        elif node_num == 4:
            node_num_info['4'] += 1
            data[3] += 1
        elif node_num == 5:
            node_num_info['5'] += 1
            data[4] += 1

    print(node_num_info)
    labels = ['1', '2', '3', '4', '5']
    plt.bar(range(len(data)), data, tick_label=labels)

    # plt.show()
    plt.savefig(output_file, format='svg')


if __name__ == '__main__':
    graph_dir = 'E:\\Binary Source\\trex\\analysis'
    architecture = ['arm-32', 'mips-32', 'x86-32']
    software_arm32 = ['binutils-2.30-', 'binutils-2.31-', 'binutils-2.32-', 'binutils-2.34-',
                      'busybox-1.32.0-',
                      'coreutils-8.15-', 'coreutils-8.25-', 'coreutils-8.29-', 'coreutils-8.30-', 'coreutils-8.31-',
                      'coreutils-8.32-',
                      'curl-7.71.1-',
                      'diffutils-3.1-', 'diffutils-3.2-', 'diffutils-3.3-', 'diffutils-3.7-',
                      'findutils-4.4.2-', 'findutils-4.6.0-', 'findutils-4.7.0-',
                      'gmp-6.2.0-',
                      'ImageMagick-7.0.10-27-',
                      'libmicrohttpd-0.9.71-',
                      'libtomcrypt-1.18.2-',
                      'putty-0.74-',
                      'sqlite-3.34.0-',
                      'zlib-1.2.11-']
    software_arm32_o0 = ['coreutils-8.26-', 'coreutils-8.27-', 'coreutils-8.28-']
    software_mip32 = ['binutils-2.30-', 'binutils-2.31-', 'binutils-2.32-', 'binutils-2.34-',
                      'busybox-1.32.0-',
                      'coreutils-8.29-', 'coreutils-8.30-', 'coreutils-8.31-', 'coreutils-8.32-',
                      'curl-7.71.1-',
                      'diffutils-3.1-', 'diffutils-3.2-', 'diffutils-3.3-', 'diffutils-3.7-',
                      'findutils-4.2.30-', 'findutils-4.2.31-', 'findutils-4.2.32-', 'findutils-4.2.33-',
                      'findutils-4.4.0-', 'findutils-4.4.1-', 'findutils-4.4.2-', 'findutils-4.6.0-', 'findutils-4.7.0-'
                      'gmp-6.2.0-',
                      'ImageMagick-7.0.10-27-',
                      'libmicrohttpd-0.9.71-',
                      'libtomcrypt-1.18.2-',
                      'putty-0.74-',
                      'sqlite-3.34.0-',
                      'zlib-1.2.11-']
    software_x8632 = ['binutils-2.30-', 'binutils-2.34-',
                      'busybox-1.32.0-',
                      'coreutils-6.10-', 'coreutils-7.5-', 'coreutils-8.15-', 'coreutils-8.32-',
                      'curl-7.71.1-',
                      'diffutils-3.6-', 'diffutils-3.7-',
                      'findutils-4.4.0-', 'findutils-4.4.2-', 'findutils-4.6.0-', 'findutils-4.7.0-'
                                                                                  'gmp-6.2.0-',
                      'ImageMagick-7.0.10-27-',
                      'libmicrohttpd-0.9.71-',
                      'libtomcrypt-1.18.2-',
                      'putty-0.74-',
                      'sqlite-3.34.0-',
                      'zlib-1.2.11-']
    optimization = ['O0', 'O1', 'O2', 'O3']

    # database 1
    # 生成软件的路径
    # software_dir_dict = {}
    # for arch in architecture:
    #     if arch == 'arm-32':
    #         software = software_arm32
    #     elif arch == 'mips-32':
    #         software = software_mip32
    #     elif arch == 'x86-32':
    #         software = software_x8632
    #
    #     software_dir, software_name = get_software_dir(graph_dir, arch, software, optimization)
    #     software_dir_dict[arch] = {'software_dirs': software_dir,
    #                                'software_names': software_name}
    #
    # # 获取函数的CFG
    # graphs = []
    # for arch, info in software_dir_dict.items():
    #     software_dirs = info['software_dirs']
    #     software_names = info['software_names']
    #     for i in range(len(software_dirs)):
    #         sf_dir = software_dirs[i]
    #         sf_name = software_names[i]
    #         sf_graph = get_graphs(sf_dir, sf_name)
    #         graphs.extend(sf_graph)
    # print('database 1')
    # # print(avaliable_graph_num(graphs))
    # # show_node_num(graphs, 'node_num_info_database1.svg')
    # show_node_num2(graphs, 'node_num_info2_database1.svg')
    # print()

    # database 2
    software_openssl = ['openssl-1.0.1f-', 'openssl-1.0.1u-']
    openssl_dir_dict = {}
    for arch in architecture:
        software_dir, software_name = get_software_dir(graph_dir, arch, software_openssl, optimization)
        openssl_dir_dict[arch] = {'software_dirs': software_dir,
                                  'software_names': software_name}

    openssl_graph = []
    for arch, info in openssl_dir_dict.items():
        software_dirs = info['software_dirs']
        software_names = info['software_names']
        for i in range(len(software_dirs)):
            sf_dir = software_dirs[i]
            sf_name = software_names[i]
            sf_graph = get_graphs(sf_dir, sf_name)
            openssl_graph.extend(sf_graph)
    print('database 2')
    # print(avaliable_graph_num(openssl_graph))
    # show_node_num(openssl_graph, 'node_num_info_database2.svg')
    # show_node_num2(openssl_graph, 'node_num_info2_database2.svg')
    #
    node_num = 0
    for item in openssl_graph:
        if is_avaliable_cfg(item):
            node_num += item.node_num

    print(node_num)
    graph_num = avaliable_graph_num(openssl_graph)
    print()

