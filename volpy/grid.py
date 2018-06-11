'''
Fast grid evaluation for NumPy arrays.

'''
import numpy as np

from ._grid import grid_scalar_eval, grid_vector_eval
from .peval import peval

MIN_COORDINATE = np.array([-0.5, -0.5, -0.5, 1])
MAX_COORDINATE = np.array([0.5, 0.5, 0.5, 1])


class Grid(object):
    '''
    A wrapper object for convenient evaluation and interpolation of NumPy
    arrays. The evaluation happens inside of a C extension so it is much
    faster, and releases the global interpreter lock so multiple threads can
    execute in the meantime.

    The normalized grid coordinate space is the centered unit cube:

        [-0.5, -0.5, -0,5] x [0.5, 0.5, 0.5].
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
        self.itransform = np.linalg.inv(self.transform)

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

    @property
    def nelements(self):
        '''
        Returns the number of elements in the grid.

        '''
        n = 1
        for k in self.array.shape[:3]:
            n *= k
        return n

    def indices(self):
        '''
        Returns the indices of the grid in order of increasing dimension.

        Returns
        -------
        indices : array
            An ``(nelements, 3)`` array listing the grid indices, suitable for
            use in ``igspace()``.

        '''
        indices = np.indices(self.array.shape).transpose()
        return indices.reshape(self.nelements, 3)

    def igspace(self, indices):
        '''
        Compute the grid-space coordinates of given indices.

        Parameters
        ----------
        indices : array
            An ``(n, 3)`` array listing the grid indices, in the format
            returned by ``indices()``.

        Returns
        -------
        gspace : array
            An ``(nelements, 4)`` array containing the homogenous grid-space
            coordinates.

        '''
        shape = np.array(self.array.shape)
        ijk = np.ndarray((self.nelements, 4))
        ijk[:, :3] = np.divide(indices, shape - 1) - 0.5
        ijk[:, 3] = 1
        return ijk

    def gwspace(self, gspace):
        '''
        Compute the world-space coordinates of grid-space coordinates.

        Parameters
        ----------
        gspace : array
            An ``(n, 4)`` array listing the grid-space coordinates, in the
            format returned by ``igspace()``.

        Returns
        -------
        wspace : array
            An ``(nelements, 4)`` array containing the world-space coordinates.

        '''
        return np.dot(gspace, self.itransform)

    def stamp(self, field):
        '''
        Overwrite the values of the grid with the equivalent world-space values
        of the given field.

        Parameters
        ----------
        field : callable
            A field function. Will be called with the world-space coordinates
            of the grid. The returned values will overwrite the grid values at
            the corresponding indices.

        '''
        return self._stamp(field)

    def pstamp(self, field, method='thread', workers=None):
        '''
        Like ``stamp()`` but use ``volpy.peval()`` to do the evaluation.

        Parameters
        ----------
        field : callable
            A field function. Will be called with the world-space coordinates
            of the grid. The returned values will overwrite the grid values at
            the corresponding indices.
        method : str
            Either 'thread' or 'fork'. Determines the concurrency method used
            for evaluating. With 'thread' multiple threads are launched. With
            'fork' multiple processes are launched. Note that threads are more
            likely to have less CPU utilization due to the GIL. Processes are
            not as likely, but their memory will be duplicated.
        workers : int or None
            Number of worker threads/processes. If None, use the number of CPUs
            returned by ``multiprocessing.cpu_count()``.

        '''
        return self._stamp(field, parallel=True, method=method,
                           workers=workers)

    def _stamp(self, field, parallel=False, method='thread', workers=None):
        '''
        Common internal implementation for stamp() and pstamp().

        '''
        indices = self.indices()
        gspace = self.igspace(indices)
        wspace = self.gwspace(gspace)
        if parallel:
            result = peval(field, wspace, method=method, workers=workers)
        else:
            result = field(wspace)
        i, j, k = indices.transpose()
        self.array[i, j, k] = result
