import numpy as np

import threading
import multiprocessing

from .camera import Camera
from ._util import cartesian
from ._native import cast_rays


class Scene(object):

    def __init__(self, emit=None, emit_color=None, camera=None, scatter=1.):
        self.emit = emit
        self.emit_color = emit_color
        self.camera = camera or _default_camera()
        self.scatter = scatter

    def render(self, shape, step=None, threads=None, tol=1e-6):
        if threads is None:
            threads = multiprocessing.cpu_count()
        elif threads < 1:
            raise ValueError('Must have at least 1 thread.')
        if tol <= 0:
            raise ValueError('Tolerance must be >0.')
        if not len(shape) == 2:
            raise ValueError('Shape must have length 2')
        if step is None:
            step = (self.camera.far - self.camera.near) / 100

        origins, directions = self._linspace_rays(shape)
        pixels = shape[0] * shape[1]
        image = np.zeros((pixels, 4))

        light = self._cast_rays(origins, directions, step, threads, tol)
        image += light
        return image.reshape((shape[1], shape[0], 4))

    def _cast_rays(self, positions, directions, step, threads, tol):
        wait = []
        chunk_size = max(1, int(len(positions) / threads))
        for i in range(0, len(positions), chunk_size):
            thread = TraceRay(
                scene=self,
                positions=positions[i:i + chunk_size],
                directions=directions[i:i + chunk_size],
                step=step,
                tol=tol,
            )
            wait.append(thread)
            thread.start()

        wait[0].join()
        light = wait[0].light
        for thread in wait[1:]:
            thread.join()
            light = np.append(light, thread.light, axis=0)
        return light

    def _linspace_rays(self, shape):
        imy, imx = cartesian([np.linspace(0, 1, shape[1]),
                              np.linspace(0, 1, shape[0])]).transpose()
        return self.camera.cast(imx, imy)


class TraceRay(threading.Thread):

    def __init__(self, scene, positions, directions, step, tol):
        super().__init__()
        self.positions = positions
        self.directions = directions
        self.scene = scene
        self.light = None
        self.step = step
        self.tol = tol

    def run(self):
        self.light = cast_rays(self.scene, self.positions,
                               self.directions, self.step,
                               self.tol)


def _default_camera():
    return Camera(eye=(0., 0., 0.), view=(0., 0., 1.))
