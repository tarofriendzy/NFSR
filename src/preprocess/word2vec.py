
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten


class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):

        self.target_embedding = Embedding(input_dim=vocab_size,
                                          output_dim=embedding_dim,
                                          input_length=1,
                                          name='target_embedding')

        self.context_embedding = Embedding(input_dim=vocab_size,
                                           output_dim=embedding_dim,
                                           input_length=num_ns+1,
                                           name='context_embedding')
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        target_embedding = self.target_embedding(target)
        content_embedding = self.context_embedding(context)
        dots = self.dots([content_embedding, target_embedding])
        return self.flatten(dots)


