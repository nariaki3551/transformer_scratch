import settings
from _logging import getLogger, setLevel

logger = getLogger(__name__)


if settings.GPU:
    import cupy as np

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    logger.info("-" * 60)
    logger.info(" " * 23 + "GPU Mode (cupy)")
    logger.info("-" * 60)
else:
    import numpy as np


def to_cpu(x):
    import numpy

    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy

    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)
