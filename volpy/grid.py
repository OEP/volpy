import numpy as np


class Grid(object):

    def __init__(self, array, transform=None, default=0):
        self.array = np.asarray(array)
        self.default = default
        self.transform = transform if transform is not None else np.eye(4)
        shape = self.array.shape
        self._shape = np.asarray(shape).reshape(len(shape), 1)

    def __call__(self, xyz):
        xyz = np.asarray(xyz)
        ijk_normalized = np.dot(self.transform, xyz.T)[:3]
        ijk = ijk_normalized * self._shape[:3]
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
        result = (
            self.array[i0, j0, k0] * p0 * p1 * p2 +
            self.array[i1, j0, k0] * q0 * p1 * p2 +
            self.array[i0, j1, k0] * p0 * q1 * p2 +
            self.array[i1, j1, k0] * q0 * q1 * p2 +
            self.array[i0, j0, k1] * p0 * p1 * q2 +
            self.array[i1, j0, k1] * q0 * p1 * q2 +
            self.array[i0, j1, k1] * p0 * q1 * q2 +
            self.array[i1, j1, k1] * q0 * q1 * q2
        )

        oob = np.any((ijk_normalized < 0) |
                     (ijk_normalized > 1), axis=0)
        return np.where(oob, self.default, result)

    def _force_bounds(self, ijk):
        ijk[ijk < 0] = 0
        for i in range(3):
            ijk[i, ijk[i, :] >= self.array.shape[i]] = self.array.shape[i] - 1
