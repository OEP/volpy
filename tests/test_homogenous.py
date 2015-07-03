import numpy as np
import numpy.testing as npt

import math
import unittest

import volpy


ZEROS = np.array([[0, 0, 0, 1]])
ONES = np.array([[1, 1, 1, 1]])


class HomogenousTestCase(unittest.TestCase):

    def test_translate1(self):
        result = volpy.translate(0, 0, 0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_translate2(self):
        result = volpy.translate(1, 2, 3)
        expected = [[1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1]]
        npt.assert_almost_equal(expected, result)

    def test_translate3(self):
        result = volpy.translate(1, 2, 3).dot(ZEROS.T).T
        expected = [[1, 2, 3, 1]]
        npt.assert_almost_equal(expected, result)

    def test_scale1(self):
        result = volpy.scale(1, 1, 1)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_scale2(self):
        result = volpy.scale(1, 2, 3)
        expected = [[1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 3, 0],
                    [0, 0, 0, 1]]
        npt.assert_almost_equal(expected, result)

    def test_scale3(self):
        result = volpy.scale(1, 2, 3).dot(ONES.T).T
        expected = [[1, 2, 3, 1]]
        npt.assert_almost_equal(expected, result)

    def test_rotatex1(self):
        result = volpy.rotatex(0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_rotatey1(self):
        result = volpy.rotatey(0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_rotatez1(self):
        result = volpy.rotatez(0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_rotatexyz1(self):
        result = volpy.rotatexyz(0, 0, 0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_rotate_axis1(self):
        result = volpy.rotate_axis([1, 0, 0], 0)
        expected = np.eye(4)
        npt.assert_almost_equal(expected, result)

    def test_rotate_axis2(self):
        result = volpy.rotate_axis([1, 0, 0], math.pi)
        expected = [[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]]
        npt.assert_almost_equal(expected, result)

    def test_rotate_axis3(self):
        result = volpy.rotate_axis([1, 0, 0], math.pi).dot(ONES.T).T
        expected = [[1, -1, -1, 1]]
        npt.assert_almost_equal(expected, result)
