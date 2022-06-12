from utils.np import *


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def cross_entropy_error(y, t):
    """
    Args:
        y np.array: softmax値 (batch_size, num_of_classes) or (num_of_classes, )
        t array like object: 正解ラベル (batch_size, ) or int
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    out = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return out
