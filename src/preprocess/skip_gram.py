
import io
import tensorflow as tf
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Dot, Embedding, Flatten


class Word2Vec(Model):

    def __init__(self, vocab_size, embedding_dim, num_ns):

        super(Word2Vec, self).__init__()
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
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


def train_model(dataset, vocab_size, num_ns, embedding_dim, num_epochs, learning_rate):

    word2vec = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, num_ns=num_ns)
    word2vec.compile(optimizer=optimizers.SGD(learning_rate=learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    word2vec.fit(dataset, epochs=num_epochs, callbacks=[tensorboard_callback])

    return word2vec.get_layer('target_embedding').get_weights()[0]





