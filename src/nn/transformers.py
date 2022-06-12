import math

from utils.np import *

import nn.models
import nn.layers
import nn.embeddings


class FeedForward(nn.models.Model):
    def __init__(self, model_dim, feed_forward_dim):
        super(FeedForward, self).__init__()
        self.layer1 = nn.layers.Linear(model_dim, feed_forward_dim)
        self.layer2 = nn.layers.Linear(feed_forward_dim, model_dim)

        self.layers = [self.layer1, self.layer2]
        self.params = sum([layer.params for layer in self.layers], start=[])
        self.grads = sum([layer.grads for layer in self.layers], start=[])

    def forward(self, x):
        """
        Args:
          x: (batch_size, sentence_length, embed_dim)
        Returns:
          (batch_size, sentence_length, embed_dim)
        """
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x

    def backward(self, dout):
        dout = self.layer2.backward(dout)
        dout = self.layer1.backward(dout)
        return dout


class Encoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, feed_forward_dim, num_heads, sentence_length):
        super(Encoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.attention = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, feed_forward_dim)

        self.layers = [self.embedding, self.attention, self.feed_forward]
        self.params = sum([layer.params for layer in self.layers], start=[])
        self.grads = sum([layer.grads for layer in self.layers], start=[])

    def forward(self, indices):
        """
        Args:
            indices: (batch_size, sentence_length)
        Returns:
            x: (batch_size, sentence_length, embed_dim)
        """
        # embedding tokens
        x = self.embedding.forward(indices)

        # attention
        x = self.attention.forward(x, x, x)

        # feed forward
        x = self.feed_forward.forward(x)
        return x

    def backward(self, dout):
        dout = self.feed_forward.backward(dout)
        dxq, dxk, dxv = self.attention.backward(dout)
        dout = dxq + dxk + dxv
        return dout


class Decoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, feed_forward_dim, num_heads, sentence_length):
        super(Decoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.attention1 = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.attention2 = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, feed_forward_dim)
        self.affine = nn.layers.Linear(self.embedding.get_dim(), len(vocab))

        self.layers = [
            self.embedding,
            self.attention1,
            self.attention2,
            self.feed_forward,
            self.affine,
        ]
        self.params = sum([layer.params for layer in self.layers], start=[])
        self.grads = sum([layer.grads for layer in self.layers], start=[])

    def forward(self, indices, encoder_output):
        """
        Args:
            indices: (batch_size, sentence_length)
            encoder_output: (batch_size, sentence_length, embed_dim)
        Returns:
            x: (batch_size, sentence_length, vocab_size)
        """
        # embedding tokens
        x = self.embedding.forward(indices)

        # generate mask
        _, sentence_length, _ = x.shape
        mask = np.triu(np.ones((sentence_length, sentence_length)), k=1)

        # first attention
        x = self.attention1.forward(x, x, x, mask)

        # second attention
        x = self.attention2.forward(x, encoder_output, encoder_output)

        # feed forward
        x = self.feed_forward.forward(x)

        # affine conversion
        x = self.affine.forward(x)
        return x

    def backward(self, dout):
        dout = self.affine.backward(dout)

        dout = self.feed_forward.backward(dout)

        dxq, dxk, dxv = self.attention2.backward(dout)
        dout = dxq
        dencoder_output = dxk + dxv

        dxq, dxk, dxv = self.attention1.backward(dout)
        dout = dxq + dxk + dxv

        dout = self.embedding.backward(dout)
        return dout, dencoder_output


class Transformer(nn.models.Model):
    def __init__(
        self,
        vocab_ja,
        vocab_en,
        embed_dim,
        feed_forward_dim,
        num_heads,
        sentence_length,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_en,
            embed_dim,
            feed_forward_dim,
            num_heads,
            sentence_length,
        )
        self.decoder = Decoder(
            vocab_ja,
            embed_dim,
            feed_forward_dim,
            num_heads,
            sentence_length,
        )

        self.layers = [self.encoder, self.decoder]
        self.params = sum([layer.params for layer in self.layers], start=[])
        self.grads = sum([layer.grads for layer in self.layers], start=[])

    def forward(self, encoder_input, decoder_input):
        """
        Args:
            encoder_input np.array: sentence_en_array # (batch_size, sentence_length, embed_dim)
            decoder_input np.array: sentence_ja_array # (batch_size, sentence_length, embed_dim)
        """
        encoder_output = self.encoder.forward(
            encoder_input
        )  # (batch_size, sentence_length, embed_dim)
        decoder_output = self.decoder.forward(
            decoder_input, encoder_output
        )  # (batch_size, sentence_length, vocab_size)
        return decoder_output

    def backward(self, dout):
        ddecoder_out, dencoder_out = self.decoder.backward(dout)
        dencoder_out = self.encoder.backward(dencoder_out)
        return dencoder_out, ddecoder_out
