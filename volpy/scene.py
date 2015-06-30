import numpy as np

import threading
import multiprocessing

from .camera import Camera
from ._util import cartesian, ascolumn


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
        distance = self.scene.camera.near
        far = self.scene.camera.far
        ray_count = self.positions.shape[0]
        deltas = np.ndarray(self.directions.shape)
        transmissivity = np.ones((ray_count, 1))
        optical_length = self.scene.scatter * self.step
        self.light = np.zeros((ray_count, 4))

        delta_transmissivity = np.ndarray((ray_count, 1))
        color = np.ndarray((ray_count, 3))

        while (
            distance < far
            and np.any(transmissivity > self.tol)
        ):
            # XXX Cull rays which have no transmissivity

            # Calculate the change in transmissivity. Here, we are reshaping
            # the delta_transmisivity vector to pass to the client functions,
            # which expect ndim=1. We reshape later for broadcasting purposes.
            delta_transmissivity = np.reshape(delta_transmissivity,
                                              (ray_count,))
            self.scene.emit(self.positions, delta_transmissivity)
            delta_transmissivity = ascolumn(delta_transmissivity)
            delta_transmissivity *= -optical_length
            np.exp(delta_transmissivity, out=delta_transmissivity)

            # Compute the light color.
            if self.scene.emit_color is None:
                color.fill(1)
            else:
                self.emit_color(self.positions, color)
            color *= transmissivity
            color *= (1 - delta_transmissivity)
            self.light[:, 0:3] += color
            transmissivity *= delta_transmissivity

            # Cast the rays forward one step.
            np.multiply(self.directions, self.step, out=deltas)
            self.positions += deltas
            distance += self.step

        self.light[:, 3] = np.reshape(1 - transmissivity, ray_count)


def _default_camera():
    return Camera(eye=(0., 0., 0.), view=(0., 0., 1.))
