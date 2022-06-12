from utils.np import *
import torchtext

import settings
import nn.models


class Random(nn.models.Model):
    def __init__(self, vocab, embed_dim):
        super(Random, self).__init__()
        self.embed_dim = embed_dim
        vocab_size = len(vocab)
        self.W = np.random.randn(vocab_size, embed_dim).astype(np.float16)

        self.params = []
        self.grads = []

    def get_dim(self):
        return self.embed_dim

    def forward(self, indices):
        """convert token index to embedded vector
        Args:
            indices np.array: (batch_size, sentence_length)
        Returns:
            embedded_vec np.array: (batch_size, sentence_length, embed_dim)
        """
        batch_size, sentence_length = indices.shape

        embedded_vec = np.zeros(
            (batch_size, sentence_length, self.embed_dim), dtype=np.float16
        )
        un_mask = indices != settings.PAD_ID  # embedded_vec for <pad> token is 0 vector
        embedded_vec[un_mask] = self.W[indices[un_mask]]
        return embedded_vec

    def backward(self, dout):
        return dout
