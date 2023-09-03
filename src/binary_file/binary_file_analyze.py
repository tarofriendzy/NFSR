import os
import stat
import angr
import arpy
import json
import numpy as np
from angrutils import *
import matplotlib.pyplot as plt
import networkx as nx
from angrutils import *


class graph(object):
    """
    CFG
    """
    def __init__(self, name, node_num, binary_name, has_return, is_plt, is_syscall, normalized, is_simprocedure,
                 software=None):
        self.name = name
        self.node_num = node_num
        self.binary_name = binary_name
        self.has_return = has_return
        self.is_plt = is_plt
        self.is_syscall = is_syscall
        self.normalized = normalized
        self.is_simprocedure = is_simprocedure
        self.softwar = software

        self.nodes = []
        self.succs = []
        self.preds = []
        if (node_num > 0):
            for i in range(node_num):
                self.nodes.append([])
                self.succs.append([])
                self.preds.append([])

    def add_node(self, feature=[]):
        self.node_num += 1
        self.nodes.append(feature)
        self.succs.append([])
        self.preds.append([])

    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.nodes[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


def get_cfg(file_name, output_file):
    """
    Obtain the CFG (Control Flow Graph) of each function in the binary file
    :param file_name:
    :param output_file:
    :return:
    """
    if not os.path.exists(file_name):
        raise Exception('filename error')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    project = angr.Project(file_name, load_options={'auto_load_libs': False})

    cfg = project.analyses.CFGFast()
    funcs_addr_set = cfg.kb.functions.function_addrs_set
    writer = open(output_file, 'w')
    for func_addr in iter(funcs_addr_set):
        func = cfg.kb.functions[func_addr]

        # function info
        binary_name = func.binary_name  # str
        has_return = func.has_return  # bool
        is_plt = func.is_plt  # bool
        is_syscall = func.is_syscall  # bool
        name = func.name  # str, function name
        normalized = func.normalized
        is_simprocedure = func.is_simprocedure  # bool

        if name == 'UnresolvableJumpTarget' or name == 'UnresolvableCallTarget':
            continue

        # if name != 'set_hex':
        #     continue
        if name != 'close_stdout':
            continue

        func_graph = func.graph

        graph_num = len(list(func_graph.nodes()))

        if len(func.block_addrs) != graph_num:
            raise Exception('error')

        nodes = {}
        nodes_des = {}
        node_id = 0
        for node in func_graph.nodes:
            nodes[node] = node_id
            nodes_des[node_id] = node
            node_id += 1

        edges_info = []
        nodes_info = []
        for node_id in range(len(nodes.keys())):
            node = nodes_des[node_id]

            # nodes_info.append([])
            block = project.factory.block(node.addr)
            content = block.pp(show_addresses=False, show_lable=False, show_edges=False)
            if len(content) == 0:
                nodes_info.append([])
            else:
                nodes_info.append(content.split('\n'))

            # insn_list = []
            # for insn in block.capstone.insns:
            #     token = str(insn.mnemonic)
            #     opstrs = insn.op_str
            #     insn_list.append(token + ' ' + opstrs)
            # nodes_info.append(insn_list)

            successors = func_graph.successors(node)
            edges = []
            for succ in successors:
                edges.append(nodes[succ])
            edges_info.append(edges)

        func_json = {'name': name,
                     'node_num': graph_num,
                     'binary_name': binary_name,
                     'has_return': has_return,
                     'is_plt': is_plt,
                     'is_syscall': is_syscall,
                     'is_simprocedure': is_simprocedure,
                     'normalized': normalized,
                     'edges': edges_info,
                     'nodes': nodes_info}
        writer.write(json.dumps(func_json) + '\n')

    writer.close()


def read_cfg(file_name, software=None):
    """
    :param file_name:
    :return:
    """
    graphs = []
    with open(file_name) as inf:
        for line in inf:
            if len(line.strip()) == 0:
                continue

            g_info = json.loads(line.strip())
            cur_graph = graph(name=g_info['name'], node_num=g_info['node_num'], binary_name=g_info['binary_name'],
                              has_return=g_info['has_return'], is_plt=g_info['is_plt'], is_syscall=g_info['is_syscall'],
                              normalized=g_info['normalized'], is_simprocedure=g_info['is_simprocedure'],
                              software=software)
            for u in range(g_info['node_num']):
                cur_graph.nodes[u] = np.array(g_info['nodes'][u])
                for v in g_info['edges'][u]:
                    cur_graph.add_edge(u, v)
            graphs.append(cur_graph)

    return graphs


def is_elf_file(filepath):
    """
    :param filepath:
    :return:
    """
    if not os.path.exists(filepath):
        return False
    try:
        file_states = os.stat(filepath)
        file_mode = file_states[stat.ST_MODE]
        if not stat.S_ISREG(file_mode) or stat.S_ISLNK(file_mode):
            return False
        with open(filepath, 'rb') as f:
            header = (bytearray(f.read(4))[1:4]).decode(encoding="utf-8")
            if header in ["ELF"]:
                return True
    except UnicodeDecodeError as e:
        pass

    return False


def get_all_file(binary_dir, output_dir, judge_elf=True):
    binary_files = []
    output_files = []

    group = os.walk(binary_dir)

    for path, dir_list, file_list in group:
        for i in range(len(file_list)):
            file_name = file_list[i]
            file_path = os.path.join(path, file_name)
            if judge_elf:
                is_write = is_elf_file(file_path)
            else:
                is_write = True

            if is_write:
                binary_files.append(file_path)
                output_path = file_path.replace(binary_dir, output_dir)
                output_files.append(output_path)

    return binary_files, output_files


def uzip_ar(file_name):
    if not os.path.exists(file_name):
        raise Exception('file is not exists')
    file_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name).split('.')[0]

    output_dir = os.path.join(file_dir, base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ar = arpy.Archive(file_name)
    ar.read_all_headers()
    archived_files = ar.archived_files

    for k, v in archived_files.items():
        data = v.read()
        output_file = os.path.join(output_dir, k.decode("utf-8"))
        writer = open(output_file, 'wb')
        writer.write(data)
        writer.close()


def save_graph_object(file_name, output_file):
    if not os.path.exists(file_name):
        raise Exception('filename error')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    project = angr.Project(file_name, load_options={'auto_load_libs': False})

    cfg = project.analyses.CFGFast()
    funcs_addr_set = cfg.kb.functions.function_addrs_set

    for func_addr in iter(funcs_addr_set):
        func = cfg.kb.functions[func_addr]

        name = func.name  # str, function name

        if name == 'UnresolvableJumpTarget' or name == 'UnresolvableCallTarget':
            continue

        if name != 'set_hex':
            continue
        # if name != 'close_stdout':
        #     continue

        func_graph = func.graph
        nx.write_gpickle(func_graph, output_file + '.gpickle')
        nx.draw(func_graph,
                pos=nx.kamada_kawai_layout(func_graph),
                node_color='#1f78b4',
                alpha=0.9,
                width=0.9)
        plt.draw()
        # plt.show()
        plt.savefig(output_file + '.svg', format='svg')


def save_graph_svg(graph_file_name, output_file):
    func_graph = nx.read_gpickle(graph_file_name)
    nx.draw(func_graph,
            pos=nx.kamada_kawai_layout(func_graph),
            node_color='#1f78b4',
            alpha=0.9,
            width=0.9)
    plt.draw()
    # plt.show()
    plt.savefig(output_file + '.svg', format='svg')
