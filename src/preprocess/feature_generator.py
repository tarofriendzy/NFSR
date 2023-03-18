"""
特征生成器
构建二进制文件的信息, Word2Vec的数据集, Token 嵌入, 块嵌入
"""
import re
import io
import os
import tqdm
import numpy as np
import tensorflow as tf
from config import get_default_config
from word2vec.skip_gram import train_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from utils import is_elf_file
from deepwalk.walk import random_walks_generator
from preprocessing import preprocessing


def articles_generator(walks, block_idx_to_tokens, output_dir, filename_config):
    """
    根据随机游走生成文章
    :param filename_config:
    :param walks:
    :param block_idx_to_tokens:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename_config['article']), 'w') as article_writer:
        for walk in walks:
            sentence = []
            for idx in walk:
                if idx in block_idx_to_tokens:
                    tokens = block_idx_to_tokens[idx]
                    sub_sentence = ' '.join(tokens)
                    sentence.append(sub_sentence)
            article_writer.write(' '.join(sentence) + '. \n')


def build_info(config):
    """
    创建二进制文件的信息
    :param binary_files:
    :param output_dir:
    :return:
    """
    binary_dir = config['binary_dir']
    output_dir = config['output_dir']

    binary_files = []
    output_dirs = []

    group = os.walk(binary_dir)
    path_level = 0
    for path, dir_list, file_list in group:
        path_level += 1
        for i in range(len(file_list)):
            file_name = file_list[i]
            file_path = os.path.join(path, file_name)
            if is_elf_file(file_path):
                binary_files.append(file_path)

                split_path = os.path.normpath(file_path).split(os.path.sep)
                output_path = os.path.join(output_dir, os.path.sep.join(split_path[-path_level:]))
                output_dirs.append(output_path)
            if i == len(file_list) - 1:
                path_level -= 1

    filename_config = config['file_names']
    binary_output_path_writer = io.open(os.path.join(output_dir, filename_config['binary_output_path']), 'w')

    for i in range(len(binary_files)):
        binary_file = binary_files[i]
        output_path = output_dirs[i]

        block_idx_to_tokens, per_block_neighbors_bids, node_dict = preprocessing(
            binary_file, output_path, filename_config)
        walks = random_walks_generator(edge_list_file=os.path.join(output_path, 'edge_list'), **config['walk'])
        articles_generator(walks, block_idx_to_tokens, output_path, filename_config)
        binary_output_path_writer.write(output_path + '\n')


def generate_training_data(sequences, vocab_size, config):
    """
    根据窗口大小，负采样的数量和词汇量，
    为序列（int编码的句子）列表生成带有负样本的skip-gram语法对
    :param sequences: 序列
    :param vocab_size:词汇表大小
    :return:
    """
    output_dir = config['output_dir']
    filename_word2vec = config['file_names']['word2vec_train_data']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_writer = io.open(os.path.join(output_dir, filename_word2vec), 'w')

    targets, contexts, labels = [], [], []

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=config['word2vec']['window_size'],
            negative_samples=0
        )

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=config['word2vec']['num_ns'],
                unique=True,
                range_max=vocab_size,
                seed=config['word2vec']['seed'],
                name='negative_sampling'
            )

            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * config['word2vec']['num_ns'], dtype='int64')

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
            data_writer.write(str(target_word) + '\n')
            data_writer.write('[' + ','.join([str(x) for x in context.numpy()]) + ']\n')
            data_writer.write('[' + ','.join([str(x) for x in label.numpy()]) + ']\n\n')


def read_training_data(config):
    """
    读取训练数据
    :param config:
    :return:
    """
    word2vec_train_data_file = os.path.join(config['output_dir'], config['file_names']['word2vec_train_data'])
    if not os.path.exists(word2vec_train_data_file) or not os.path.isfile(word2vec_train_data_file):
        raise ValueError('构建Word2Vec的数据集')

    targets, contexts, labels = [], [], []
    line_mum = 1
    target_word = None
    context_arr = None
    label_array = None
    for line in io.open(word2vec_train_data_file, 'r'):
        if line_mum % 4 == 1:
            target_word = re.findall(r"^\d+", line)[0]
        elif line_mum % 4 == 2:
            context_str = re.findall(r"^\[(.*)\]", line)[0]
            context_arr = []
            for item in context_str.split(','):
                vals = re.findall(r"^\[(.*)\]", item)[0]
                context_arr.append([np.int64(x) for x in vals.split(',')])
        elif line_mum % 4 == 3:
            label = re.findall(r"^\[(.*)\]", line)[0]
            label_array = [np.int64(x) for x in label.split(',')]
        else:
            targets.append(np.int64(target_word))
            contexts.append(tf.convert_to_tensor(context_arr))
            labels.append(tf.convert_to_tensor(label_array))
        line_mum += 1

    return targets, contexts, labels


def record_vocab(config, vocab):
    """
    记录词汇表
    :param config:
    :param vocab:
    :return:
    """
    output_dir = config['output_dir']
    filename_word2vec = config['file_names']['word2vec_vocab_data']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_writer = io.open(os.path.join(output_dir, filename_word2vec), 'w')
    for i in range(len(vocab)):
        data_writer.write(str(i) + ':\n')
        data_writer.write(vocab[i] + '\n\n')


def read_vocab(config):
    """
    读取词汇表
    :param config:
    :return:
    """
    word2vec_vocab_data_file = os.path.join(config['output_dir'], config['file_names']['word2vec_vocab_data'])
    if not os.path.exists(word2vec_vocab_data_file) or not os.path.isfile(word2vec_vocab_data_file):
        raise ValueError('构建Word2Vec的数据集')

    vocab = []

    line_mum = 1
    index = None
    word = None
    for line in io.open(word2vec_vocab_data_file, 'r'):
        if line_mum % 3 == 1:
            index = re.findall(r"^\d+", line)[0]
        elif line_mum % 3 == 2:
            word = re.findall(r"^(.*)", line)[0]
        else:
            vocab.append(word)
        line_mum += 1

    return np.array(vocab)


def build_data(config):
    """
    构建训练时所需要的数据
    :param config:
    :return:
    """
    filename_conf = config['file_names']
    binary_output_path_file = os.path.join(config['output_dir'], filename_conf['binary_output_path'])
    if not os.path.exists(binary_output_path_file) or not os.path.isfile(binary_output_path_file):
        raise ValueError('请先构建二进制文件的信息')

    article_files = []
    for line in io.open(binary_output_path_file, 'r'):
        filename = re.findall(r"^(.*)", line)[0]
        if len(filename) > 0:
            article_file = os.path.join(filename, filename_conf['article'])
            if os.path.exists(article_file) and os.path.isfile(article_file):
                article_files.append(article_file)

    if len(article_files) == 0:
        raise ValueError('请先构建二进制文件的信息')

    text_dateset = tf.data.TextLineDataset(article_files).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    def custom_standardization(input_data):
        return tf.strings.regex_replace(input_data, '[.]', '')

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        output_sequence_length=config['word2vec']['sequence_length']
    )
    vectorize_layer.adapt(text_dateset.batch(1024))
    inverse_vocab = vectorize_layer.get_vocabulary()

    text_vector_ds = text_dateset.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())

    generate_training_data(sequences=sequences, vocab_size=len(inverse_vocab), config=config)

    record_vocab(config, inverse_vocab)


def token_embedding_generation(config):
    """
    生成Token嵌入
    :param article_file:
    :param output_dir
    :return:
    """
    word2vec_conf = config['word2vec']
    filename_conf = config['file_names']
    targets, contexts, labels = read_training_data(config)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(10000).batch(min(word2vec_conf['batch_size'], len(targets)), drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    vocab = read_vocab(config)

    weights = train_model(dataset=dataset, vocab_size=len(vocab), num_ns=word2vec_conf['num_ns'],
                          num_epochs=word2vec_conf['num_epochs'], embedding_dim=word2vec_conf['embedding_dim'],
                          learning_rate=word2vec_conf['learning_rate'])

    output_dir = config['output_dir']
    out_v = io.open(os.path.join(output_dir, filename_conf['word2vec_vectors']), 'w')
    out_m = io.open(os.path.join(output_dir, filename_conf['word2vec_metadata']), 'w')
    result_writer = io.open(os.path.join(output_dir, filename_conf['word2vec_token_embedding']), 'w')

    for index, word in enumerate(vocab):
        if index == 0:
            continue
        vec = weights[index]

        out_v.write('\t'.join([str(x) for x in vec]) + '\n')
        out_m.write(word + '\n')

        result_writer.write(str(index) + ':' + word + '\n')
        result_writer.write('[' + ','.join([str(x) for x in vec]) + ']\n\n')

    out_v.close()
    out_m.close()


def read_token_embedding(token_embedding_file):
    """
    读取Token嵌入
    :param token_embedding_file:
    :return:
    """
    token_embedding_dict = dict()
    inverse_vocab = []
    line_mum = 1
    word = None
    embedding = None
    for line in open(token_embedding_file, 'r'):
        if line_mum % 3 == 1:
            index, word = re.findall(r"^(\d+):(.*)", line)[0]
        elif line_mum % 3 == 2:
            embedding = re.findall(r"^\[(.*)\]", line)[0]
            embedding = [float(x) for x in embedding.split(',')]
        else:
            token_embedding_dict[word] = embedding
            inverse_vocab.append(word)
        line_mum += 1

    return inverse_vocab, token_embedding_dict


def cal_block_embedding(inverse_vocab, token_embedding, tokens):
    """
    计算块嵌入
    :param token_embedding:
    :param tokens:
    :return:
    """
    block_embs = []
    for token in tokens:
        if token in inverse_vocab:
            token_emb = token_embedding[token]
            block_embs.append(token_emb)
    if len(block_embs) > 0:
        block_embs = np.array(block_embs).sum(0)
        return block_embs
    return None


def block_embedding_generation(config):
    """
    生成块的嵌入
    :param node_index_to_code_file:
    :return:
    """
    filename_conf = config['file_names']
    output_dir = config['output_dir']

    binary_output_path_file = os.path.join(output_dir, filename_conf['binary_output_path'])
    if not os.path.exists(binary_output_path_file) or not os.path.isfile(binary_output_path_file):
        raise ValueError('请先构建二进制文件的信息')

    binary_output_path_files = []
    for line in io.open(binary_output_path_file, 'r'):
        filename = re.findall(r"^(.*)", line)[0]
        if len(filename) > 0:
            if os.path.exists(filename) and os.path.isdir(filename):
                binary_output_path_files.append(filename)

    if len(binary_output_path_files) == 0:
        raise ValueError('请先构建二进制文件的信息')

    token_embedding_file = os.path.join(output_dir, filename_conf['word2vec_token_embedding'])
    if not os.path.exists(token_embedding_file) or not os.path.isfile(token_embedding_file):
        raise ValueError('请先构建Word2Vec的数据集')

    inverse_vocab, token_embedding = read_token_embedding(token_embedding_file)

    for binary_output_path in binary_output_path_files:
        block_embedding_writer = io.open(os.path.join(binary_output_path, filename_conf['node_index_to_embedding']), 'w')

        line_mum = 1
        index = None
        tokens = None
        for line in io.open(os.path.join(binary_output_path, filename_conf['node_index_to_code']), 'r'):
            if line_mum % 3 == 1:
                index = re.findall(r"^\d+", line)[0]
            elif line_mum % 3 == 2:
                tokens = re.findall(r"^(.*)", line)[0]
                tokens = [x for x in tokens.split(', ')]
            else:
                block_embedding = cal_block_embedding(inverse_vocab, token_embedding, tokens)
                if not block_embedding is None:
                    block_embedding_writer.write(index + ':\n')
                    block_embedding_writer.write('[' + ','.join([str(x) for x in block_embedding]) + ']\n\n')
            line_mum += 1


if __name__ == '__main__':
    config = get_default_config()
    build_info(config)
    build_data(config)
    token_embedding_generation(config)
    block_embedding_generation(config)