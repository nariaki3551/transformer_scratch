from utils.np import *


class Adam:
    def __init__(
        self, params, grads, lr=0.001, beta1=0.9, beta2=0.999, dtype=np.float64
    ):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.dtype = dtype
        self.iter = 0
        self.m = []
        self.v = []
        for param in params:
            self.m.append(np.zeros_like(param, dtype=dtype))
            self.v.append(np.zeros_like(param, dtype=dtype))

    def update(self):
        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for i in range(len(self.params)):
            self.m[i] += (1.0 - self.beta1) * (
                self.grads[i].astype(self.dtype) - self.m[i]
            )
            self.v[i] += (1.0 - self.beta2) * (
                self.grads[i].astype(self.dtype) ** 2 - self.v[i]
            )
            self.params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

    def zero_grad(self):
        for grad in self.grads:
            grad[...] = np.zeros_like(grad)
