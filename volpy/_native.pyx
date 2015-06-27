import numpy as np


def cast_rays(scene, positions, directions, image, step):
    distance = scene.camera.near
    ray_count = positions.shape[0]
    deltas = np.ndarray(directions.shape)
    transmissivity = np.ones((ray_count, 1))
    optical_length = scene.scatter * step
    light = np.zeros((ray_count, 4))

    delta_transmissivity = np.ndarray((ray_count, 1))
    color = np.ndarray((ray_count, 3))

    while (
        distance < scene.camera.far
        and np.any(transmissivity > scene.tol)
    ):
        # XXX Cull rays which have no transmissivity

        # Calculate the change in transmissivity. Here, we are reshaping
        # the delta_transmisivity vector to pass to the client functions,
        # which expect ndim=1. We reshape later for broadcasting purposes.
        delta_transmissivity = np.reshape(delta_transmissivity,
                                            (ray_count,))
        scene.emit(positions, delta_transmissivity)
        delta_transmissivity = delta_transmissivity.reshape(ray_count, 1)
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
