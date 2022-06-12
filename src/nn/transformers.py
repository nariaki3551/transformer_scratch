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


class PositionalEncoder(nn.models.Model):
    def __init__(self, embed_dim, sentence_length, dtype=np.float16):
        super(PositionalEncoder, self).__init__()
        self.pe = np.zeros((sentence_length, embed_dim), dtype=dtype)
        for pos in range(sentence_length):
            for i in range(0, embed_dim, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                self.pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / embed_dim))
                )

        self.params = []
        self.grads = []

    def forward(self, x):
        """
        Args:
          x: (batch_size, sentence_length, embed_dim)
        Returns:
          (batch_size, sentence_length, embed_dim)
        """
        sentence_length = x.shape[1]
        return x + self.pe[:sentence_length, :]

    def backward(self, dout):
        return dout


class Encoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, feed_forward_dim, num_heads, sentence_length):
        super(Encoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.positional_encoder = PositionalEncoder(embed_dim, sentence_length)
        self.attention = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, feed_forward_dim)
        self.norm1 = nn.layers.LayerNorm()
        self.norm2 = nn.layers.LayerNorm()

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
        x = self.positional_encoder.forward(x)

        # attention
        x = self.norm1.forward(self.attention.forward(x, x, x) + x)

        # feed forward
        x = self.norm2.forward(self.feed_forward.forward(x) + x)
        return x

    def backward(self, dout):
        dout = self.norm2.backward(dout)
        dout = self.feed_forward.backward(dout) + dout

        dout = self.norm1.backward(dout)
        dxq, dxk, dxv = self.attention.backward(dout)
        dout = dxq + dxk + dxv + dout

        dout = self.positional_encoder.backward(dout)
        dout = self.embedding.backward(dout)
        return dout


class Decoder(nn.models.Model):
    def __init__(self, vocab, embed_dim, feed_forward_dim, num_heads, sentence_length):
        super(Decoder, self).__init__()
        self.embedding = nn.embeddings.Random(vocab, embed_dim)
        self.positional_encoder = PositionalEncoder(embed_dim, sentence_length)
        self.attention1 = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.attention2 = nn.layers.MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, feed_forward_dim)
        self.affine = nn.layers.Linear(self.embedding.get_dim(), len(vocab))
        self.norm1 = nn.layers.LayerNorm()
        self.norm2 = nn.layers.LayerNorm()
        self.norm3 = nn.layers.LayerNorm()

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
        x = self.positional_encoder.forward(x)

        # generate mask
        _, sentence_length, _ = x.shape
        mask = np.triu(np.ones((sentence_length, sentence_length)), k=1)

        # first attention
        x = self.norm1.forward(self.attention1.forward(x, x, x, mask) + x)

        # second attention
        x = self.norm2.forward(
            self.attention2.forward(x, encoder_output, encoder_output) + x
        )

        # feed forward
        x = self.norm3.forward(self.feed_forward.forward(x) + x)

        # affine conversion
        x = self.affine.forward(x)
        return x

    def backward(self, dout):
        dout = self.affine.backward(dout)

        dout = self.norm3.backward(dout)
        dout = self.feed_forward.backward(dout) + dout

        dout = self.norm2.backward(dout)
        dxq, dxk, dxv = self.attention2.backward(dout)
        dout = dxq + dout
        dencoder_output = dxk + dxv

        dout = self.norm1.backward(dout)
        dxq, dxk, dxv = self.attention1.backward(dout)
        dout = dxq + dxk + dxv + dout

        dout = self.positional_encoder.backward(dout)
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
