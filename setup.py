from setuptools import setup
from Cython.Build import cythonize

__version__ = None
exec(open('volpy/version.py').read())


setup(
    name='volpy',
    version=__version__,
    ext_modules=cythonize('volpy/_native.pyx'),
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
