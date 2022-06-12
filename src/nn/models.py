import os
import pickle

from utils.np import *
import nn.layers

import settings


class Model:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pickle"

        params = [p.astype(np.float16) for p in self.params]
        if settings.GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pickle"

        if "/" in file_name:
            file_name = file_name.replace("/", os.sep)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [p.astype("f") for p in params]
        if settings.GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]



