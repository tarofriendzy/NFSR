import preprocessing
import collections

# 操作码的id
opcode_idx_list = []


def vocabulary_build(block_idx_to_tokens):
    """
    构建单词的id与单词的映射
    :param block_idx_to_tokens: 块的索引和序列化后的Token的字典
        block_idx_to_tokens[block id] = token list
    :return: dictionary: index to token, reversed_dictionary: token to index
    """
    global opcode_idx_list

    # 记录所有的单词（重复记录）
    vocabulary = []

    # 单词与单词id的映射
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
    """
    为 word2vec 生成文章
    在块之间放置一个标签
    :param walks:
    :param block_idx_to_tokens:
    :param reversed_dictionary:
    :return:
    """
    # 记录所有文章，每个文章本身是一个列表
    article = []

    # 存储所有块边界索引。 blockBoundaryIndices[i]是用于存储articles[i]索引的列表。
    # 每一项都存储块中最后一个令牌的索引
    block_boundary_idx = []

    for walk in walks:
        # 一个随机游走序列作为一篇文章
        for idx in walk:
            if idx in block_idx_to_tokens:
                tokens = block_idx_to_tokens[idx]
                for token in tokens:
                    article.append(reversed_dictionary[token])
            block_boundary_idx.append(len(article) - 1)

    insn_starting_indices = []
    index_to_current_insns_start = {}

    # 遍历当前块以检索指令起始索引
    for i in range(0, len(article)):
        if article[i] in opcode_idx_list:
            insn_starting_indices.append(i)
        index_to_current_insns_start[i] = len(insn_starting_indices) - 1

    return article, block_boundary_idx, insn_starting_indices, index_to_current_insns_start
