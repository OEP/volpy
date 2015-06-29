import numpy as np

from .camera import Camera
from ._util import cartesian
from ._native import cast_rays


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
        image = np.zeros((pixels, 4))

        light = cast_rays(self, origins, directions, step)
        image += light
        return image.reshape((shape[1], shape[0], 4))

    def _linspace_rays(self, shape):
        imy, imx = cartesian([np.linspace(0, 1, shape[1]),
                              np.linspace(0, 1, shape[0])]).transpose()
        return self.camera.cast(imx, imy)


def _default_camera():
    return Camera(eye=(0., 0., 0.), view=(0., 0., 1.))
