import numpy as np

from .camera import Camera
from ._util import cartesian


class Scene(object):

    def __init__(self, emit=None, emit_color=None, camera=None, scatter=1.):
        self.emit = emit
        self.emit_color = emit_color
        self.camera = camera or _default_camera()
        self.scatter = scatter

    def render(self, shape, step=None):
        if not len(shape) == 2:
            raise ValueError('Shape must have length 2')
        if step is None:
            step = (self.camera.far - self.camera.near) / 100

        origins, directions = self._linspace_rays(shape)
        image = np.zeros(shape)

        self._cast_rays(origins, directions, image, step)
        return image

    def _cast_rays(self, positions, directions, image, step):
        distance = self.camera.near
        ray_count = positions.shape[0]
        deltas = np.ndarray(directions.shape)
        transmissivity = np.ones((ray_count,))
        delta_transmissivity = np.ndarray((ray_count,))
        optical_length = self.scatter * step

        while distance < self.camera.far:
            # XXX Cull the rays which have no transmissivity.

            # Calculate the change in transmissivity.
            self.emit(positions, delta_transmissivity)
            delta_transmissivity *= -optical_length
            transmissivity *= delta_transmissivity

            # Cast the rays forward one step.
            np.multiply(directions, step, out=deltas)
            np.add(positions, deltas, out=positions)
            distance += step

        return image

    def _linspace_rays(self, shape):
        imx, imy = cartesian([np.linspace(0, 1, shape[0]),
                              np.linspace(0, 1, shape[1])]).transpose()
        return self.camera.cast(imx, imy)


def _default_camera():
    return Camera(eye=(0, 0, 0), view=(0, 0, 1))
