import torch
from torch.autograd import Variable
import torch.nn.functional as F


class BlockEmbedding(torch.nn.Module):

    def __init__(self, batch_size, lstm_hid_dim, d_a, r, max_len, emb_dim=100, vocab_size=None,
                 use_pretrained_embeddings=False, embeddings=None, type=0, n_classes=1):
        super(BlockEmbedding, self).__init__()

        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim
        self.r = r
        self.max_len = max_len
        self.type = type
        self.n_classes = n_classes

        self.embeddings, emb_dim = self._load_embeddings(use_pretrained_embeddings, embeddings, vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=lstm_hid_dim, num_layers=1, batch_first=True)

        self.linear_first = torch.nn.Linear(lstm_hid_dim, d_a)
        self.linear_first.bias.data.fill_(0)

        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)

        self.linear_final = torch.nn.Linear(lstm_hid_dim, self.n_classes)

        self.hidden_state = self.init_hidden()

    def _load_embeddings(self, use_pretrained_embeddings, embeddings, vocab_size, emb_dim):

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")


        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)

        return word_embeddings, emb_dim

    def init_hidden(self):

        return (Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)))

    def softmax(self, input, axis=1):

        input_size = input.size()

        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()

        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):

        embeddings = self.embeddings(x)

        outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size, self.max_len, -1), self.hidden_state)

        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x, 1)

        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ outputs

        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r

        if not bool(self.type):
            output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
            return output, attention
        else:
            return F.log_softmax(self.linear_final(avg_sentence_embeddings)), attention
