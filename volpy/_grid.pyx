import numpy as np
cimport numpy as np

ctypedef np.float64_t GRID_t
ctypedef np.float32_t GEOM_t
ctypedef np.float32_t RESULT_t

cpdef void grid_scalar_eval(
    GRID_t [:, :, :] array,
    GEOM_t [:, :] transform,
    GEOM_t [:, :] xyz,
    RESULT_t default,
    RESULT_t [:] result,
) nogil:
    cdef int count = xyz.shape[0], idx
    with nogil:
        for idx in range(count):
            result[idx] = grid_scalar_eval_at(array, transform, xyz, default,
                                              idx)


cpdef void grid_vector_eval(
    GRID_t [:, :, :, :] array,
    GEOM_t [:, :] transform,
    GEOM_t [:, :] xyz,
    RESULT_t default,
    RESULT_t [:, :] result,
) nogil:
    cdef int count = xyz.shape[0], dim = array.shape[3], i, j
    with nogil:
        for i in range(count):
            for j in range(dim):
                result[i, j] = grid_scalar_eval_at(array[:, :, :, j],
                                                   transform, xyz, default, i)


cdef RESULT_t grid_scalar_eval_at(
    GRID_t [:, :, :] array,
    GEOM_t [:, :] transform,
    GEOM_t [:, :] xyz,
    RESULT_t default,
    int idx,
) nogil:
    cdef float i, j, k, q0, q1, q2, p0, p1, p2
    cdef int i0, j0, k0, i1, j1, k1

    grid_transform(xyz, transform, idx, &i, &j, &k)
    if (
        i < -0.5 or i > 0.5
        or j < -0.5 or j > 0.5
        or k < -0.5 or k > 0.5
    ):
        return default
    i += 0.5
    j += 0.5
    k += 0.5

    i0 = int(i)
    j0 = int(j)
    k0 = int(k)
    i1 = i0 + 1 if i0 < array.shape[0] - 1 else i0
    j1 = j0 + 1 if j0 < array.shape[1] - 1 else j0
    k1 = k0 + 1 if k0 < array.shape[2] - 1 else k0

    q0 = i - i0
    q1 = j - j0
    q2 = k - k0
    p0 = 1 - q0
    p1 = 1 - q1
    p2 = 1 - q2

    return (
        array[i0, j0, k0] * p0 * p1 * p2 +
        array[i1, j0, k0] * q0 * p1 * p2 +
        array[i0, j1, k0] * p0 * q1 * p2 +
        array[i1, j1, k0] * q0 * q1 * p2 +
        array[i0, j0, k1] * p0 * p1 * q2 +
        array[i1, j0, k1] * q0 * p1 * q2 +
        array[i0, j1, k1] * p0 * q1 * q2 +
        array[i1, j1, k1] * q0 * q1 * q2
    )


cdef void grid_transform(
    GEOM_t [:, :] xyz,
    GEOM_t [:, :] transform,
    int idx,
    float *i,
    float *j,
    float *k,
) nogil:
    i[0] = (transform[0, 0] * xyz[idx, 0] +
            transform[0, 1] * xyz[idx, 1] +
            transform[0, 2] * xyz[idx, 2] +
            transform[0, 3] * xyz[idx, 3])
    j[0] = (transform[1, 0] * xyz[idx, 0] +
            transform[1, 1] * xyz[idx, 1] +
            transform[1, 2] * xyz[idx, 2] +
            transform[1, 3] * xyz[idx, 3])
    k[0] = (transform[2, 0] * xyz[idx, 0] +
            transform[2, 1] * xyz[idx, 1] +
            transform[2, 2] * xyz[idx, 2] +
            transform[2, 3] * xyz[idx, 3])
