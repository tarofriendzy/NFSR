import preprocessing
import collections

opcode_idx_list = []


def vocabulary_build(block_idx_to_tokens):
    global opcode_idx_list

    vocabulary = []

    reversed_dictionary = dict()

    index = 0
    for idx in block_idx_to_tokens:
        for token in block_idx_to_tokens[idx]:
            vocabulary.append(token)
            if token not in reversed_dictionary:
                reversed_dictionary[token] = index
                if token in preprocessing.opcode_list and index not in opcode_idx_list:
                    opcode_idx_list.append(index)
                index += 1

    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))

    return dictionary, reversed_dictionary


def articles_generator(walks, block_idx_to_tokens, reversed_dictionary):
   
    article = []

    block_boundary_idx = []

    for walk in walks:
        for idx in walk:
            if idx in block_idx_to_tokens:
                tokens = block_idx_to_tokens[idx]
                for token in tokens:
                    article.append(reversed_dictionary[token])
            block_boundary_idx.append(len(article) - 1)

    insn_starting_indices = []
    index_to_current_insns_start = {}

    for i in range(0, len(article)):
        if article[i] in opcode_idx_list:
            insn_starting_indices.append(i)
        index_to_current_insns_start[i] = len(insn_starting_indices) - 1

    return article, block_boundary_idx, insn_starting_indices, index_to_current_insns_start
