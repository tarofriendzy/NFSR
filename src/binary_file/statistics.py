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
    if graph.is_plt:
        return False
    # Simprocedure
    if graph.is_simprocedure:
        return False
    if graph.node_num == 0:
        return False
    # if graph.node_num == 1 and len(graph.nodes[0]) <= 1:
    #     return False
    if graph.name.startswith('sub_'):
        return False

    return True

def avaliable_graph_num(graphs):
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



