'''
Fast grid evaluation for NumPy arrays.

'''
import numpy as np

from ._grid import grid_scalar_eval, grid_vector_eval

MIN_COORDINATE = np.array([-0.5, -0.5, -0.5, 1])
MAX_COORDINATE = np.array([0.5, 0.5, 0.5, 1])


class Grid(object):
    '''
    A wrapper object for convenient evaluation and interpolation of NumPy
    arrays. The evaluation happens inside of a C extension so it is much
    faster, and releases the global interpreter lock so multiple threads can
    execute in the meantime.
    '''

    def __init__(self, array, transform=None, default=0):
        '''
        Grid constructor.

        Parameters
        ----------
        array : array-like
            Any scalar or vector array.
        transform : array-like
            A 4x4 transformation matrix which maps world coordinates to the
            default normalized grid coordinates.
        default : float
            A default value for points outside of the grid. This value will be
            broadcasted into the result for vector arrays.
        '''
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
