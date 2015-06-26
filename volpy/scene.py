import numpy as np

from .camera import Camera
from ._util import cartesian, ascolumn


class Scene(object):

    def __init__(self, emit=None, emit_color=None, camera=None, scatter=1.,
                 tol=1e-6):
        self.emit = emit
        self.emit_color = emit_color
        self.camera = camera or _default_camera()
        self.scatter = scatter
        self.tol = tol

    def render(self, shape, step=None):
        if not len(shape) == 2:
            raise ValueError('Shape must have length 2')
        if step is None:
            step = (self.camera.far - self.camera.near) / 100

        origins, directions = self._linspace_rays(shape)
        pixels = shape[0] * shape[1]
        image = np.zeros((pixels, 3))

        light = self._cast_rays(origins, directions, image, step)
        image += light
        return image.reshape((shape[1], shape[0], 3))

    def _cast_rays(self, positions, directions, image, step):
        distance = self.camera.near
        ray_count = positions.shape[0]
        deltas = np.ndarray(directions.shape)
        transmissivity = np.ones((ray_count, 1))
        optical_length = self.scatter * step
        light = np.zeros((ray_count, 3))

        delta_transmissivity = np.ndarray((ray_count, 1))
        color = np.ndarray((ray_count, 3))

        while (
            distance < self.camera.far
            and np.any(transmissivity > self.tol)
        ):
            # XXX Cull rays which have no transmissivity

            # Calculate the change in transmissivity. Here, we are reshaping
            # the delta_transmisivity vector to pass to the client functions,
            # which expect ndim=1. We reshape later for broadcasting purposes.
            delta_transmissivity = np.reshape(delta_transmissivity,
                                              (ray_count,))
            self.emit(positions, delta_transmissivity)
            delta_transmissivity = ascolumn(delta_transmissivity)
            delta_transmissivity *= -optical_length
            np.exp(delta_transmissivity, out=delta_transmissivity)

            # XXX Evaluate the color function instead of this next line.
            color.fill(1)
            color *= transmissivity
            color *= (1 - delta_transmissivity)
            light += color
            transmissivity *= delta_transmissivity

            # Cast the rays forward one step.
            np.multiply(directions, step, out=deltas)
            positions += deltas
            distance += step

        return light

    def _linspace_rays(self, shape):
        imy, imx = cartesian([np.linspace(0, 1, shape[1]),
                              np.linspace(0, 1, shape[0])]).transpose()
        return self.camera.cast(imx, imy)


def _default_camera():
    return Camera(eye=(0., 0., 0.), view=(0., 0., 1.))
