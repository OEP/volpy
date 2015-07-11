import numpy as np
cimport numpy as np
cimport libc.math as math

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def cast_rays(
    scene,
    np.ndarray[DTYPE_t, ndim=2] positions,
    np.ndarray[DTYPE_t, ndim=2] directions,
    float step,
    float tol,
):
    cdef float distance = scene.camera.near, far = scene.camera.far
    cdef int ray_count = positions.shape[0]
    cdef float optical_length = scene.scatter * step

    cdef np.ndarray[DTYPE_t, ndim=2] light = np.zeros((ray_count, 4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] transmissivity = np.ones((ray_count,), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] emit_density = np.zeros(ray_count, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] emit_color = np.ones((ray_count, 3),
                                                          dtype=DTYPE)

    while (
        distance < far
        and np.any(transmissivity > tol)
    ):
        # XXX Cull rays which have no transmissivity

        _handle_element(scene.emit, positions, emit_density, emit_color)

        _march(positions, directions, transmissivity,
               emit_density, emit_color,
               light, step, optical_length)
        distance += step
    light[:, 3] = np.reshape(1 - transmissivity, ray_count)
    return light


def _handle_element(
    element,
    np.ndarray[DTYPE_t, ndim=2] positions,
    np.ndarray[DTYPE_t, ndim=1] density,
    np.ndarray[DTYPE_t, ndim=2] color
):
    if element is None:
        return
    density[:] = element.density(positions)
    if element.color is not None:
        color[:] = element.color(positions)


cdef _march(
    DTYPE_t [:, :] positions,
    DTYPE_t [:, :] directions,
    DTYPE_t [:] transmissivity,
    DTYPE_t [:] emit_density,
    DTYPE_t [:, :] emit_color,
    DTYPE_t [:, :] light,
    float step,
    float optical_length,
):
    with nogil:
        for idx in range(positions.shape[0]):
            emit_density[idx] = math.exp(-optical_length * emit_density[idx])

            emit_color[idx, 0] *= (1 - emit_density[idx]) * transmissivity[idx]
            emit_color[idx, 1] *= (1 - emit_density[idx]) * transmissivity[idx]
            emit_color[idx, 2] *= (1 - emit_density[idx]) * transmissivity[idx]

            light[idx, 0] += emit_color[idx, 0]
            light[idx, 1] += emit_color[idx, 1]
            light[idx, 2] += emit_color[idx, 2]

            transmissivity[idx] *= emit_density[idx]

            # Cast the rays forward one step.
            positions[idx, 0] += step * directions[idx, 0]
            positions[idx, 1] += step * directions[idx, 1]
            positions[idx, 2] += step * directions[idx, 2]
