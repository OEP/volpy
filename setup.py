from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

__version__ = None
exec(open('volpy/version.py').read())


setup(
    name='volpy',
    version=__version__,
    ext_modules=cythonize('volpy/*.pyx'),
    include_dirs=[np.get_include()],
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
