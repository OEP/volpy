import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def cast_rays(
    scene,
    np.ndarray[DTYPE_t, ndim=2] positions,
    np.ndarray[DTYPE_t, ndim=2] directions,
    float step
):
    cdef float distance = scene.camera.near
    cdef int ray_count = positions.shape[0]
    cdef float optical_length = scene.scatter * step

    cdef np.ndarray[DTYPE_t, ndim=2] deltas = np.ndarray((ray_count, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] transmissivity = np.ones((ray_count, 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] light = np.zeros((ray_count, 4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] delta_transmissivity = np.ndarray((ray_count, 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] emit_buffer = np.ndarray((ray_count,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] color = np.ndarray((ray_count, 3), dtype=DTYPE)

    while (
        distance < scene.camera.far
        and np.any(transmissivity > scene.tol)
    ):
        # XXX Cull rays which have no transmissivity

        # The client will put their results in the emit_buffer variable which
        # has ndim=1. This is a bit more natural that needing to reshape into
        # an ndim=2 array every time.
        scene.emit(positions, emit_buffer)
        delta_transmissivity = emit_buffer.reshape(ray_count, 1)
        delta_transmissivity *= -optical_length
        np.exp(delta_transmissivity, out=delta_transmissivity)

        # Compute the light color.
        if scene.emit_color is None:
            color.fill(1)
        else:
            scene.emit_color(positions, color)
        color *= transmissivity
        color *= (1 - delta_transmissivity)
        light[:, 0:3] += color
        transmissivity *= delta_transmissivity

        # Cast the rays forward one step.
        np.multiply(directions, step, out=deltas)
        positions += deltas
        distance += step

    light[:, 3] = np.reshape(1 - transmissivity, ray_count)
    return light
