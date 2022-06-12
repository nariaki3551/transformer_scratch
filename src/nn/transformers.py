import math

from utils.np import *

import nn.models
import nn.layers
import nn.embeddings


class Encoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, sentence_length):
        super(Encoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.attention = nn.layers.ScaledDotProductAttention(embed_dim)

        self.layers = [self.embedding, self.attention]
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
        return x

    def backward(self, dout):
        dxq, dxk, dxv = self.attention.backward(dout)
        dout = dxq + dxk + dxv
        return dout


class Decoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, sentence_length):
        super(Decoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.attention1 = nn.layers.ScaledDotProductAttention(embed_dim)
        self.attention2 = nn.layers.ScaledDotProductAttention(embed_dim)
        self.affine = nn.layers.Linear(self.embedding.get_dim(), len(vocab))

        self.layers = [
            self.embedding,
            self.attention1,
            self.attention2,
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

        # first attention
        x = self.attention1.forward(x, x, x)

        # second attention
        x = self.attention2.forward(x, encoder_output, encoder_output)

        # affine conversion
        x = self.affine.forward(x)
        return x

    def backward(self, dout):
        dout = self.affine.backward(dout)

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
        sentence_length,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_en,
            embed_dim,
            sentence_length,
        )
        self.decoder = Decoder(
            vocab_ja,
            embed_dim,
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
