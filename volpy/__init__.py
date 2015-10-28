'''
Volpy
=====

A fast volume rendering implementation for Python. Volpy has support for:

  1. Multithreading or multiprocessing at the rendering step
  2. Native implementation of ray casting
  3. Native access to NumPy arrays during rendering
  4. Support for ambient and diffuse lighting terms

How to use this package
-----------------------

Volpy is organized into several different modules but the API is imported into
the root of the package. Therefore, you should write your code like this:

    >>> import volpy
    >>> scene = volpy.Scene(ambient=my_func)

'''
from .camera import Camera
from .scene import Scene, Element, Light
from .version import __version__
from .grid import Grid
from .homogeneous import (translate, scale, rotatex, rotatey, rotatez, rotatexyz,
                          rotate_axis, cross)
from .geometry import Geometry, BBox
