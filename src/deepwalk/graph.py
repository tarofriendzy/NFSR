import random

from six import iterkeys
from collections import defaultdict


class Graph(defaultdict):
    """
    记录图信息（字典，k=node, v=[sub1, sub2...]）
    """
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        """
        图中所有的节点
        :return:
        """
        return self.keys()

    def make_consistent(self):
        """
        使记录一致（对value进行排序）
        :return:
        """
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        """
        移除自循环（value中的key）
        :return:
        """
        for x in self:
            if x in self[x]:
                self[x].remove(x)

        return self

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        被截断的随机游走
        :param path_length: 随机游走的长度
        :param alpha: 重新开始的概率
        :param rand:
        :param start: 随机游走的起始节点。
        :return:
        """
        graph_dict = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(graph_dict.keys()))]

        while len(path) < path_length:
            current_node = path[-1]
            if len(graph_dict[current_node]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(graph_dict[current_node]))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]


def load_edgelist(file):
    """
    加载CFG
    :param file:
    :return:
    """
    graph_dict = Graph()
    with open(file) as edglist_file:
        for line in edglist_file:
            x, y = line.strip().split()[:2]
            x = int(x)
            y = int(y)
            graph_dict[x].append(y)

    graph_dict.make_consistent()
    return graph_dict


def build_deepwalk_corpus(graph_dict, num_paths, path_length, alpha, rand=random.Random(0)):
    """
    进行随机游走
    :param graph_dict:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :return:
    """
    walks = []
    nodes = list(graph_dict.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(graph_dict.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks