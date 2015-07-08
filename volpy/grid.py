import numpy as np

from ._grid import grid_scalar_eval, grid_vector_eval

MIN_COORDINATE = np.array([-0.5, -0.5, -0.5, 1])
MAX_COORDINATE = np.array([0.5, 0.5, 0.5, 1])


class Grid(object):

    def __init__(self, array, transform=None, default=0):
        self.array = np.asarray(array, dtype=np.float64)
        self.default = default
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        self.transform = np.asarray(transform, dtype=np.float32)

    def __call__(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float32)
        count = xyz.shape[0]
        ndim = self.array.ndim
        if ndim == 3:
            result = np.ndarray((count,), dtype=np.float32)
            grid_scalar_eval(self.array, self.transform, xyz, self.default,
                             result)
            return result
        elif ndim == 4:
            dim = self.array.shape[3]
            result = np.ndarray((count, dim), dtype=np.float32)
            grid_vector_eval(self.array, self.transform, xyz, self.default,
                             result)
            return result
        raise ValueError('Unsupported grid ndim: %d' % ndim)
