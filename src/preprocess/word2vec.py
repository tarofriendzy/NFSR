"""
Word2vec模型
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten


class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        """
        初始化
        :param vocab_size: 词汇量,即：最大整数索引 + 1
        :param embedding_dim: 密集嵌入的维度
        :param num_ns: 每个正上下文词的负采样数量
        """
        # 当单词作为目标词出现时，查找单词的嵌入
        self.target_embedding = Embedding(input_dim=vocab_size,
                                          output_dim=embedding_dim,
                                          input_length=1,
                                          name='target_embedding')
        # 当单词作为上下文出现时，查找单词的向量
        self.context_embedding = Embedding(input_dim=vocab_size,
                                           output_dim=embedding_dim,
                                           input_length=num_ns+1,
                                           name='context_embedding')
        # 根据训练对，计算目标词嵌入和上下文嵌入的点积
        self.dots = Dot(axes=(3, 2))

        # 展平
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        target_embedding = self.target_embedding(target)
        content_embedding = self.context_embedding(context)
        dots = self.dots([content_embedding, target_embedding])
        return self.flatten(dots)


