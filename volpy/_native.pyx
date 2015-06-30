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
    cdef int idx

    cdef np.ndarray[DTYPE_t, ndim=2] color = np.ndarray((ray_count, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] light = np.zeros((ray_count, 4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] transmissivity = np.ones((ray_count,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] density = np.ndarray((ray_count,), dtype=DTYPE)

    cdef DTYPE_t [:] density_view = density
    cdef DTYPE_t [:] transmissivity_view =  transmissivity
    cdef DTYPE_t [:, :] color_view = color
    cdef DTYPE_t [:, :] light_view = light
    cdef DTYPE_t [:, :] positions_view = positions
    cdef DTYPE_t [:, :] directions_view = directions

    while (
        distance < far
        and np.any(transmissivity > tol)
    ):
        # XXX Cull rays which have no transmissivity

        # The client will put their results in the emit_buffer variable which
        # has ndim=1. This is a bit more natural that needing to reshape into
        # an ndim=2 array every time.
        scene.emit(positions, density)

        # Compute the light color.
        if scene.emit_color is None:
            color.fill(1)
        else:
            scene.emit_color(positions, color)

        with nogil:
            for idx in range(ray_count):
                density_view[idx] = math.exp(-optical_length * density_view[idx])

                color_view[idx, 0] *= (1 - density_view[idx]) * transmissivity_view[idx]
                color_view[idx, 1] *= (1 - density_view[idx]) * transmissivity_view[idx]
                color_view[idx, 2] *= (1 - density_view[idx]) * transmissivity_view[idx]

                light_view[idx, 0] += color_view[idx, 0]
                light_view[idx, 1] += color_view[idx, 1]
                light_view[idx, 2] += color_view[idx, 2]

                transmissivity_view[idx] *= density_view[idx]

                # Cast the rays forward one step.
                positions_view[idx, 0] += step * directions_view[idx, 0]
                positions_view[idx, 1] += step * directions_view[idx, 1]
                positions_view[idx, 2] += step * directions_view[idx, 2]
            distance += step
    light[:, 3] = np.reshape(1 - transmissivity, ray_count)
    return light
