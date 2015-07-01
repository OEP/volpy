import numpy as np


class Grid(object):

    def __init__(self, array, transform=None, default=0):
        array = np.asarray(array)
        self.array = np.ndarray((array.shape[0] + 2,
                                 array.shape[1] + 2,
                                 array.shape[2] + 2))
        self.array.fill(default)
        self.array[1:-1, 1:-1, 1:-1] = array
        self.transform = transform or np.eye(3)
        self._original_shape = array.shape

        shape = self.array.shape
        self._shape = np.asarray(shape).reshape(len(shape), 1)

    def __call__(self, xyz):
        ijk = 1 + np.dot(self.transform, xyz.T) * self._shape
        ijk0 = ijk.astype(np.integer)
        ijk1 = ijk0 + 1

        self._force_bounds(ijk0)
        self._force_bounds(ijk1)

        q = ijk - ijk0
        p = 1 - q

        i0, j0, k0 = ijk0
        i1, j1, k1 = ijk1
        q0, q1, q2 = q
        p0, p1, p2 = p
        return (
            self.array[i0, j0, k0] * p0 * p1 * p2 +
            self.array[i1, j0, k0] * q0 * p1 * p2 +
            self.array[i0, j1, k0] * p0 * q1 * p2 +
            self.array[i1, j1, k0] * q0 * q1 * p2 +
            self.array[i0, j0, k1] * p0 * p1 * q2 +
            self.array[i1, j0, k1] * q0 * p1 * q2 +
            self.array[i0, j1, k1] * p0 * q1 * q2 +
            self.array[i1, j1, k1] * q0 * q1 * q2
        )

    def _force_bounds(self, ijk):
        ijk[ijk < 0] = 0
        for i in range(3):
            ijk[i, ijk[i, :] >= self.array.shape[i]] = self.array.shape[i] - 1
