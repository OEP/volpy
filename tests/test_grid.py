import volpy

import numpy as np
import numpy.testing as npt

import unittest


class ScalarGridTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100)))

    def test_call1(self):
        result = self.grid([[-0.1, -0.1, -0.1, 1]])
        expected = [0]
        npt.assert_almost_equal(expected, result)

    def test_call2(self):
        result = self.grid([[0, 0, 0, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call3(self):
        result = self.grid([[0.5, 0.5, 0.5, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call4(self):
        result = self.grid([[0.98, 0.98, 0.98, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call5(self):
        result = self.grid([[1, 1, 1, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call6(self):
        result = self.grid([[1.1, 1.1, 1.1, 1]])
        expected = [0]
        npt.assert_almost_equal(expected, result)


class DefaultValueTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100)), default=-1)

    def test_call1(self):
        '''Out of bounds returns default value'''
        result = self.grid([[-0.1, -0.1, -0.1]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)

    def test_call2(self):
        '''In bounds returns grid value'''
        result = self.grid([[0.5, 0.5, 0.5]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call3(self):
        '''One out of bounds coordinate returns default value (positive)'''
        result = self.grid([[0.5, 100, 0.5]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)

    def test_call4(self):
        '''One out of bounds coordinate returns default value (negative)'''
        result = self.grid([[-0.1, 0.5, 0.5]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)


class VectorGridTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100, 3)))

    def test_call1(self):
        result = self.grid([[-0.1, -0.1, -0.1, 1]])
        expected = [[0, 0, 0]]
        npt.assert_almost_equal(expected, result)

    def test_call2(self):
        result = self.grid([[0, 0, 0, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call3(self):
        result = self.grid([[0.5, 0.5, 0.5, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call4(self):
        result = self.grid([[0.98, 0.98, 0.98, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call5(self):
        result = self.grid([[1, 1, 1, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call6(self):
        result = self.grid([[1.1, 1.1, 1.1, 1]])
        expected = [[0, 0, 0]]
        npt.assert_almost_equal(expected, result)
