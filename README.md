volpy
=====

A volume renderer for Python.

Volpy is a dual multi-process, multithread implementation of the ray casting
method of volume rendering for Python supporting ambient and diffuse lighting.
It supports visualizing scalar fields (Python functions) and scalar grids
(NumPy arrays). Each of these may be assigned color also using Python functions
or NumPy arrays.

Unit Tests
----------

Unit tests are in the `tests` subdirectory. It is recommended you install volpy
in a virtual environment before testing:

    virtualenv .
    . bin/activate
    pip install cython numpy
    python3 setup.py install
    ./runtests.sh

Benchmarks
----------

Benchmarking scripts are available in the `benchmarks` subfolder available as
executable Python scripts. They require PIL for image output.

Examples
--------

    import numpy as np
    import volpy
    center = np.array([0, 0, 2, 1])
    def sphere(x):
        norms = np.linalg.norm(x - 1, axis=1)
        return np.where(norms < 1, 1, 0)
    scene = volpy.Scene(ambient=sphere)
    image = scene.render((100, 100))
