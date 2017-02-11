from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

try:
    import numpy
except ImportError:
    numpy = None

import os

os.chdir(os.path.normpath(os.path.dirname(__file__)))

__version__ = None
exec(open('volpy/version.py').read())

compiler_directives = {
    'boundscheck': False,
}

# Try to specify extension modules
ext_modules = []
if cythonize:
    ext_modules = cythonize('volpy/*.pyx',
                            compiler_directives=compiler_directives)

# Try to get numpy's include directories
include_dirs = []
if numpy:
    include_dirs = numpy.get_include()

setup(
    name='volpy',
    version=__version__,
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    packages=['volpy'],
    include_package_data=True,
    description='A volume renderer for python',
    author='Paul Kilgo',
    author_email='paulkilgo@gmail.com',
    classifiers=[
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
    ],
)
