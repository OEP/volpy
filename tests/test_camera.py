from volpy import Camera

import numpy as np
import numpy.testing as npt

import unittest

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])


class CameraTest(unittest.TestCase):

    def setUp(self):
        self.camera = Camera([0, 0, 0], [0, 0, 1])

    def test_near1(self):
        self.assertEqual(0.1, self.camera.near)

    def test_far1(self):
        self.assertEqual(2., self.camera.far)

    def test_eye1(self):
        self.assertIsInstance(self.camera.eye, np.ndarray)
        self.assertEqual(self.camera.eye.dtype, np.dtype(np.float32))
        npt.assert_almost_equal([0, 0, 0], self.camera.eye)

    def test_eye2(self):
        self.camera.eye = (0, 1, 1)
        self.assertIsInstance(self.camera.eye, np.ndarray)
        npt.assert_almost_equal([0, 1, 1], self.camera.eye)

    def test_view1(self):
        self.assertIsInstance(self.camera.view, np.ndarray)
        self.assertEqual(self.camera.view.dtype, np.dtype(np.float32))
        npt.assert_almost_equal([0, 0, 1], self.camera.view)
        npt.assert_almost_equal(0, self.camera.up.dot(self.camera.view))

    def test_view2(self):
        self.camera.view = [2, 0, 0]
        self.assertIsInstance(self.camera.view, np.ndarray)
        npt.assert_almost_equal([1, 0, 0], self.camera.view)
        npt.assert_almost_equal(0, self.camera.up.dot(self.camera.view))

    def test_up1(self):
        self.assertIsInstance(self.camera.up, np.ndarray)
        self.assertEqual(self.camera.up.dtype, np.dtype(np.float32))
        npt.assert_almost_equal([0, 1, 0], self.camera.up)
        npt.assert_almost_equal(0, self.camera.up.dot(self.camera.view))

    def test_up2(self):
        self.camera.up = [2, 0, 0]
        self.assertIsInstance(self.camera.up, np.ndarray)
        npt.assert_almost_equal([1, 0, 0], self.camera.up)
        npt.assert_almost_equal(0, self.camera.up.dot(self.camera.view))

    def test_right1(self):
        self.assertIsInstance(self.camera.right, np.ndarray)
        self.assertEqual(self.camera.right.dtype, np.dtype(np.float32))
        npt.assert_almost_equal(0, self.camera.up.dot(self.camera.right))
        npt.assert_almost_equal(0, self.camera.view.dot(self.camera.right))

    def test_right2(self):
        with self.assertRaises(AttributeError):
            self.camera.right = (0, 1, 0)

    def test_cast1(self):
        origin, direction = self.camera.cast([0.5], [0.5])
        self.assertIsInstance(origin, np.ndarray)
        self.assertIsInstance(direction, np.ndarray)
        self.assertEqual(origin.dtype, np.dtype(np.float32))
        self.assertEqual(direction.dtype, np.dtype(np.float32))
        npt.assert_almost_equal(origin, [[0, 0, 0.1]])
        npt.assert_almost_equal(direction, [[0, 0, 1]])

    def test_cast2(self):
        '''Cast should not write to arguments'''
        coord = np.linspace(0, 1, 10)
        coord.flags.writeable = False
        self.camera.cast(coord, coord)

    def test_cast3(self):
        '''Checks for out of bounds'''
        coord = np.linspace(0, 1.1, 10)
        with self.assertRaises(ValueError):
            self.camera.cast(coord, coord)

    def test_cast4(self):
        '''Checks for same size'''
        coord1 = np.linspace(0, 1.0, 10)
        coord2 = np.linspace(0, 1.0, 11)
        with self.assertRaises(ValueError):
            self.camera.cast(coord1, coord2)
