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

    cdef np.ndarray[DTYPE_t, ndim=1] ambient_density = np.zeros(ray_count, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] ambient_color = np.ones((ray_count, 3),
                                                          dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] diffuse_density = np.zeros(ray_count, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] diffuse_color = np.ones((ray_count, 3),
                                                          dtype=DTYPE)

    while (
        distance < far
        and np.any(transmissivity > tol)
    ):
        # XXX Cull rays which have no transmissivity

        _handle_element(scene.ambient, positions, ambient_density, ambient_color)
        _handle_element(scene.diffuse, positions, diffuse_density, diffuse_color)

        _march(positions, directions, transmissivity,
               ambient_density, ambient_color,
               diffuse_density, diffuse_color,
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
    DTYPE_t [:] ambient_density,
    DTYPE_t [:, :] ambient_color,
    DTYPE_t [:] diffuse_density,
    DTYPE_t [:, :] diffuse_color,
    DTYPE_t [:, :] light,
    float step,
    float optical_length,
):
    cdef float weight, total_density, dt
    cdef float[3] total_color
    with nogil:
        for idx in range(positions.shape[0]):
            total_density = ambient_density[idx] + diffuse_density[idx]
            dt = math.exp(-optical_length * total_density)

            if total_density > 0:

                # Add ambient component
                total_color[0] = ambient_color[idx, 0] * ambient_density[idx]
                total_color[1] = ambient_color[idx, 1] * ambient_density[idx]
                total_color[2] = ambient_color[idx, 2] * ambient_density[idx]

                # XXX Add diffuse component

                # Weight all color contributions with respect to the total
                # density.
                total_color[0] /= total_density
                total_color[1] /= total_density
                total_color[2] /= total_density

                weight = (1 - dt) * transmissivity[idx]
                total_color[0] *= weight
                total_color[1] *= weight
                total_color[2] *= weight

                light[idx, 0] += total_color[0]
                light[idx, 1] += total_color[1]
                light[idx, 2] += total_color[2]

            transmissivity[idx] *= dt

            # Cast the rays forward one step.
            positions[idx, 0] += step * directions[idx, 0]
            positions[idx, 1] += step * directions[idx, 1]
            positions[idx, 2] += step * directions[idx, 2]
