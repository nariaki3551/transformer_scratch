from utils.np import *

from nn.functions import softmax, cross_entropy_error


class Linear:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dtype=np.float32,
        grad_dtype=np.float64,
    ):
        self.W = np.random.randn(input_dim, output_dim).astype(dtype)
        self.b = np.random.randn(output_dim).astype(dtype)
        self.params = [self.W, self.b]
        self.grads = [
            np.zeros_like(self.W, dtype=grad_dtype),
            np.zeros_like(self.b, dtype=grad_dtype),
        ]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params

        if dout.ndim == 2:
            dx = np.dot(dout, W.T)
            dW = np.dot(self.x.T, dout)
            db = np.sum(dout, axis=0)
        elif dout.ndim == 3:
            dx = np.dot(dout, W.T)
            dW = np.einsum("ijk,ijz->kz", self.x, dout)
            db = np.einsum("ijk->k", dout)
        else:
            raise NotImplementedError

        self.grads[0][...] += dW
        self.grads[1][...] += db

        self.x = None

        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        dx -= self.out * np.sum(dx, axis=1, keepdims=True)
        self.out = None
        return dx


class CrossEntropyLoss:
    def __init__(self):
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル
        self.mask = None

    def forward(self, x, t, ignore_index=None):
        y = softmax(x)

        self.t = t
        self.y = y

        if ignore_index is None:
            loss = cross_entropy_error(y, t)
        else:
            self.mask = (t != ignore_index).squeeze()
            loss = cross_entropy_error(y[self.mask], t[self.mask])
        return loss

    def backward(self, dout=1.0):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        if self.mask is None:
            dx[np.arange(batch_size), self.t] -= 1
        else:
            dx[self.mask, self.t[self.mask]] -= 1
        dx *= dout
        dx /= batch_size

        self.y = None
        self.t = None
        self.mask = None

        return dx


class ScaledDotProductAttention:
    def __init__(
        self, input_dim, output_dim=None, dtype=np.float32, grad_dtype=np.float64
    ):
        self.input_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.Wq = np.random.randn(input_dim, output_dim).astype(dtype)
        self.Wk = np.random.randn(input_dim, output_dim).astype(dtype)
        self.Wv = np.random.randn(input_dim, output_dim).astype(dtype)
        self.softmax = Softmax()

        self.q = None
        self.k = None
        self.v = None
        self.f = None
        self.xq = None
        self.xk = None
        self.xv = None

        self.params = [
            self.Wq,
            self.Wk,
            self.Wv,
        ]
        self.grads = [
            np.zeros_like(self.Wq, dtype=grad_dtype),
            np.zeros_like(self.Wk, dtype=grad_dtype),
            np.zeros_like(self.Wv, dtype=grad_dtype),
        ]

    def forward(self, xq, xk, xv, mask=None):
        """
        Args:
            xq (batch_size, sentence_length, embed_dim), bse
            xk (batch_size, sentence_length, embed_dim), bse
            xv (batch_size, sentence_length, embed_dim), bse
            mask (sentence_length, sentence_length), ss

        Returns:
            out (batch_size, sentence_length, embed_dim)
        """
        batch_size, sentence_length, embed_dim = xq.shape
        assert xq.shape == xk.shape == xv.shape

        self.xq = xq
        self.xk = xk
        self.xv = xv
        self.q = np.dot(xq, self.Wq)  # np.einsum("bjk,kl->bjl", x, self.Wq) bse
        self.k = np.dot(xk, self.Wk)  # np.einsum("bjk,kl->bjl", x, self.Wq) bse
        self.v = np.dot(xv, self.Wv)  # np.einsum("bjk,kl->bjl", x, self.Wq) bse

        s = np.einsum("bie,bje->bij", self.q, self.k) / np.sqrt(
            embed_dim
        )  # bse,bse->bss

        # mask
        self.mask = mask
        if mask is not None:
            for i in range(batch_size):
                s[i][mask == 0] = 0.0

        self.f = self.softmax.forward(s)  # bss

        out = np.einsum("bij,bjk->bik", self.f, self.v)  # bss,bse->bse
        return out

    def backward(self, dout):
        """
        Args:
            dout (batch_size, sentence_length, embed_dim)

        Returns:
            dxq (batch_size, sentence_length, embed_dim)
            dxk (batch_size, sentence_length, embed_dim)
            dxv (batch_size, sentence_length, embed_dim)
        """
        batch_size, _, _ = dout.shape
        Wq, Wk, Wv = self.params

        # df = np.dot(dout, self.v.T)
        df = np.einsum("bij,bkj->bik", dout, self.v)  # bse,bse->bss
        ds = self.softmax.backward(df) / np.sqrt(self.input_dim)  # bss
        if self.mask is not None:
            for i in range(batch_size):
                ds[i][self.mask == 0] = 0.0

        # dq = np.dot(ds, self.k)
        dq = np.einsum("bij,bjk->bik", ds, self.k)  # bss,bse->bse
        # dWq = np.dot(self.x.T, dq)
        dWq = np.einsum("bij,bik->jk", self.xq, dq)  # bse,bse->ee
        # dxq = np.dot(dq, Wq.T)
        dxq = np.einsum("bsj,kj->bsk", dq, Wq)  # bse,ee->bse

        # dk = np.dot(ds, self.q)
        dk = np.einsum("bij,bjk->bik", ds, self.q)  # bss,bse->bse
        # dWk = np.dot(self.x.T, dk)
        dWk = np.einsum("bij,bik->jk", self.xk, dk)  # bse,bse->ee
        # dxk = np.dot(dk, Wk.T)
        dxk = np.einsum("bsj,kj->bsk", dk, Wk)  # bse,ee->bse

        # dv = np.dot(self.f, dout)
        dv = np.einsum("bij,bjk->bik", self.f, dout)  # bss,bse->bse
        # dWv = np.dot(self.x.T, dv)
        dWv = np.einsum("bij,bik->jk", self.xv, dv)  # bse,bse->ee
        # dxv = np.dot(dv, Wv.T)
        dxv = np.einsum("bsj,kj->bsk", dv, Wv)  # bse,ee->bse

        self.grads[0][...] += dWq
        self.grads[1][...] += dWk
        self.grads[2][...] += dWv

        self.q = None
        self.k = None
        self.v = None
        self.f = None
        self.xq = None
        self.xk = None
        self.xv = None
        self.mask = None

        return dxq, dxk, dxv


class MultiHeadAttention:
    def __init__(self, input_dim, num_heads, dtype=np.float32, grad_dtype=np.float64):
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim

        self.attentions = [
            ScaledDotProductAttention(input_dim, self.head_dim, dtype, grad_dtype)
            for _ in range(num_heads)
        ]
        self.linear = Linear(input_dim, input_dim)
        self.mask = None

        self.layers = self.attentions + [self.linear]
        self.params = sum([layer.params for layer in self.layers], start=[])
        self.grads = sum([layer.grads for layer in self.layers], start=[])

    def forward(self, xq, xk, xv, mask=None):
        """
        Args:
            xq (batch_size, sentence_length, embed_dim), bse
            xk (batch_size, sentence_length, embed_dim), bse
            xv (batch_size, sentence_length, embed_dim), bse
            mask (sentence_length, sentence_length), ss

        Returns:
            out (batch_size, sentence_length, embed_dim)
        """
        outputs = [
            attention.forward(xq, xk, xv, mask) for attention in self.attentions
        ]  # each (batch_size, sentence_length, head_dim)
        out = np.concatenate(
            outputs, axis=-1
        )  # (batch_size, sentence_length, embed_dim)
        out = self.linear.forward(out)  # (batch_size, sentence_length, embed_dim)
        return out

    def backward(self, dout):
        """
        Args: dout (batch_size, sentence_length, embed_dim)

        Returns:
            dxq (batch_size, sentence_length, embed_dim)
            dxk (batch_size, sentence_length, embed_dim)
            dxv (batch_size, sentence_length, embed_dim)
        """
        batch_size, sentence_length, embed_dim = dout.shape
        assert self.num_heads * self.head_dim == embed_dim

        dout = self.linear.backward(
            dout
        )  # (batch_size, sentence_length, num_heads * head_dim)
        dout = dout.reshape(
            (batch_size, sentence_length, self.num_heads, self.head_dim)
        )
        dout = np.einsum(
            "bshe->hbse", dout
        )  # (num_heads, batch_size, sentence_length, head_dim)
        douts = [
            self.attentions[i].backward(dout[i]) for i in range(self.num_heads)
        ]  # each tuple of triple of (batch_size, sentence_length, embed_dim)
        dxq = sum([dout[0] for dout in douts])
        dxk = sum([dout[1] for dout in douts])
        dxv = sum([dout[2] for dout in douts])
        return dxq, dxk, dxv
