import numpy.testing as npt

import volpy

import unittest


class BBoxTestCase(unittest.TestCase):

    def setUp(self):
        self.bbox = volpy.BBox([[1, 1, 1, 1],
                                [5, 5, 5, 1]])

    def test_transform1(self):
        result = self.bbox.transform().dot(self.bbox.corners[0])
        expected = [-0.5, -0.5, -0.5, 1]
        npt.assert_almost_equal(expected, result)

    def test_transform2(self):
        result = self.bbox.transform().dot(self.bbox.corners[1])
        expected = [0.5, 0.5, 0.5, 1]
        npt.assert_almost_equal(expected, result)

    def test_transform3(self):
        result = self.bbox.transform().dot((self.bbox.corners[0] +
                                            self.bbox.corners[1]) / 2)
        expected = [0, 0, 0, 1]
        npt.assert_almost_equal(expected, result)

    def test_inverse_transform1(self):
        result = self.bbox.inverse_transform().dot([-0.5, -0.5, -0.5, 1])
        expected = self.bbox.corners[0]
        npt.assert_almost_equal(expected, result)

    def test_inverse_transform2(self):
        result = self.bbox.inverse_transform().dot([0.5, 0.5, 0.5, 1])
        expected = self.bbox.corners[1]
        npt.assert_almost_equal(expected, result)

    def test_inverse_transform3(self):
        result = self.bbox.inverse_transform().dot([0, 0, 0, 1])
        expected = (self.bbox.corners[0] + self.bbox.corners[1]) / 2
        npt.assert_almost_equal(expected, result)
