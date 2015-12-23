'''
Parallel evalulation function.
'''

import multiprocessing
import threading
import numpy as np


def peval(func, xyz, method='thread', workers=None):
    '''
    Evaluate a function in parallel using multiple threads or processes.

    Parameters
    ----------
    func : a callable
        The function to evaluate.
    xyz : array
        A ``(n, 4)`` array of homogenous XYZ coordinates to evaluate the
        function at. This array will be divided into chunks and passed into
        ``func`` by the parallel worker units.
    method : str
        Either 'thread' or 'fork'. Determines the concurrency method used for
        evaluating. With 'thread' multiple threads are launched. With 'fork'
        multiple processes are launched. Note that threads are more likely to
        have less CPU utilization due to the GIL. Processes are not as likely,
        but their memory will be duplicated.
    workers : int or None
        Number of worker threads/processes. If None, use the number of CPUs
        returned by ``multiprocessing.cpu_count()``.

    Returns
    -------
    result : array
        Exactly the same value as ``func(array)``.

    '''
    if workers is None:
        workers = multiprocessing.cpu_count()
    elif workers < 1:
        raise ValueError('Must have at least 1 worker.')
    jobs = []
    chunksize = max(1, int(len(xyz) / workers))
    for i in range(0, len(xyz), chunksize):
        chunk = xyz[i:i + chunksize]
        jobs.append(_Job(func, chunk))
    if method == 'thread':
        return _peval_thread(jobs)
    elif method == 'fork':
        return _peval_fork(jobs)
    else:
        raise ValueError('Invalid method: %s' % method)


def _peval_thread(jobs):
    wait = []
    for job in jobs:
        thread = _EvalThread(job)
        thread.start()
        wait.append(thread)
    wait[0].join()
    result = wait[0].job.result
    for thread in wait[1:]:
        thread.join()
        result = np.append(result, thread.job.result, axis=0)
    return result


def _peval_fork(jobs):
    pool = multiprocessing.Pool(len(jobs))
    results = pool.map(_run_job, jobs)
    result = results[0]
    for r in results[1:]:
        result = np.append(result, r, axis=0)
    return result


def _run_job(job):
    job.run()
    return job.result


class _Job(object):

    def __init__(self, func, xyz):
        self.func = func
        self.xyz = xyz
        self.result = None

    def run(self):
        self.result = self.func(self.xyz)


class _EvalThread(threading.Thread):

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        self.job.run()
