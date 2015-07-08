'''
Grid geometries.
'''
import numpy as np

from abc import ABCMeta, abstractmethod

from .homogeneous import translate, scale
from .grid import MIN_COORDINATE, MAX_COORDINATE

class Geometry(object, metaclass=ABCMeta):

    @abstractmethod
    def transform(self):
        '''
        Create a transform matrix for world space to grid space conversion.

        Returns
        -------
        T : array
            A 4x4 transformation matrix which transforms world space to grid
            space.
        '''
        raise NotImplementedError

    def inverse_transform(self):
        '''
        Create a transform matrix for grid space to world space conversion.

        Returns
        -------
        T : array
            A 4x4 transformation matrix which transforms grid space to world
            space.
        '''
        return np.linalg.inv(self.transform())


class BBox(Geometry):

    def __init__(self, corners):
        self.corners = np.asarray(corners)

    def transform(self):
        s_model = (MAX_COORDINATE - MIN_COORDINATE)[:3]
        s_actual = (self.corners[1] - self.corners[0])[:3]
        sxyz = s_model / s_actual
        txyz = -(self.corners[1] + self.corners[0])[:3] / 2
        S = scale(*sxyz)
        T = translate(*txyz)
        return S.dot(T)
