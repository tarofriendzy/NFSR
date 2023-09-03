
import random
from deepwalk import graph


def random_walks_generator(edge_list_file, number_walks, walk_length, alpha, seed):

    graph_dict = graph.load_edgelist(edge_list_file)
    walks = graph.build_deepwalk_corpus(graph_dict, num_paths=number_walks, path_length=walk_length, alpha=alpha,
                                        rand=random.Random(seed))

    return walks
