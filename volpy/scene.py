import numpy as np

import threading
import multiprocessing

from .camera import Camera
from ._util import cartesian
from ._native import cast_rays


class Scene(object):

    def __init__(self, emit=None, emit_color=None, camera=None, scatter=1.):
        '''
        Scene constructor.

        Parameters
        ----------
        emit : callable
            A scalar field callable representing the emission of the object.
        emit_color : callable
            A color field callable respresenting the color of the emission.
        camera : Camera
            The camera for the rendered image.
        scatter : float
            The global scattering constant. Higher values increase how "solid"
            a density looks in the resulting image.

        '''
        self.emit = emit
        self.emit_color = emit_color
        self.camera = camera or _default_camera()
        self.scatter = scatter

    def render(self, shape, step=None, workers=None, tol=1e-6,
               method='thread'):
        '''
        Render an image.

        Parameters
        ----------
        shape : tuple of int of length 2
            The desired shape of the output image.
        step : float or None
            The ray step size. If None, step exactly 100 times.
        workers : int or None
            Number of worker threads/processes. If None, use the number of CPUs
            returned by ``multiprocessing.cpu_count()``.
        tol : float
            Minimum transmissivity value for a ray to be considered alive. If
            no ray's transmissivity is above this value, the trace is stopped
            early.
        method : str
            Either 'thread' or 'fork'. Determines the concurrency method used
            for rendering. With 'thread' multiple threads are launched. With
            'fork' multiple processes are launched. Note that threads are more
            likely to have less CPU utilization due to the GIL. Processes are
            not as likely, but their memory will be duplicated.

        Returns
        -------
        image : numpy.ndarray
            A 2D image of shape ``(shape[1], shape[0], 4)`` containing the
            results. The color channel is RGBA format with floating point
            depth.

        '''
        if workers is None:
            workers = multiprocessing.cpu_count()
        elif workers < 1:
            raise ValueError('Must have at least 1 worker.')
        if tol <= 0:
            raise ValueError('Tolerance must be >0.')
        if not len(shape) == 2:
            raise ValueError('Shape must have length 2')
        if step is None:
            step = (self.camera.far - self.camera.near) / 100

        origins, directions = self._linspace_rays(shape)
        pixels = shape[0] * shape[1]
        image = np.zeros((pixels, 4))

        light = _cast_rays(self, origins, directions, step, workers, tol,
                           method)
        image += light
        return image.reshape((shape[1], shape[0], 4))

    def _linspace_rays(self, shape):
        imy, imx = cartesian([np.linspace(0, 1, shape[1]),
                              np.linspace(0, 1, shape[0])]).transpose()
        return self.camera.cast(imx, imy)


class Job(object):

    def __init__(self, scene, positions, directions, step, tol):
        self.positions = positions
        self.directions = directions
        self.scene = scene
        self.step = step
        self.tol = tol


class TraceRay(threading.Thread):

    def __init__(self, job):
        super().__init__()
        self.job = job
        self.light = None

    def run(self):
        self.light = _run_job(self.job)


def _default_camera():
    return Camera(eye=(0., 0., 0.), view=(0., 0., 1.))


def _cast_rays(scene, positions, directions, step, workers, tol, method):
    jobs = []
    chunk_size = max(1, int(len(positions) / workers))
    for i in range(0, len(positions), chunk_size):
        job = Job(
            scene=scene,
            positions=positions[i:i + chunk_size],
            directions=directions[i:i + chunk_size],
            step=step,
            tol=tol,
        )
        jobs.append(job)

    if method == 'thread':
        return _cast_rays_thread(jobs)
    elif method == 'fork':
        return _cast_rays_fork(jobs)
    else:
        raise ValueError('Invalid method: %s' % method)


def _cast_rays_thread(jobs):
    wait = []
    for job in jobs:
        thread = TraceRay(job)
        wait.append(thread)
        thread.start()

    wait[0].join()
    light = wait[0].light
    for thread in wait[1:]:
        thread.join()
        light = np.append(light, thread.light, axis=0)
    return light


def _cast_rays_fork(jobs):
    pool = multiprocessing.Pool(len(jobs))
    results = pool.map(_run_job, jobs)
    light = results[0]
    for result in results[1:]:
        light = np.append(light, result, axis=0)
    return light


def _run_job(job):
    return cast_rays(job.scene, job.positions, job.directions, job.step,
                     job.tol)
