import volpy

import numpy as np
import numpy.testing as npt

import unittest


def _scalar_stamp(xyz):
    return xyz[:, 0]


class ScalarGridTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100)))

    def test_call1(self):
        result = self.grid([[-0.6, -0.6, -0.6, 1]])
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
        result = self.grid([[0.4, 0.4, 0.4, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call5(self):
        result = self.grid([[0.5, 0.5, 0.5, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call6(self):
        result = self.grid([[0.6, 0.6, 0.6, 1]])
        expected = [0]
        npt.assert_almost_equal(expected, result)

    def test_indices1(self):
        result = self.grid.indices()
        self.assertEqual((100**3, 3), result.shape)
        npt.assert_almost_equal((0, 0, 0), result[0])
        npt.assert_almost_equal((99, 99, 99), result[100**3 - 1])

    def test_igspace1(self):
        indices = self.grid.indices()
        result = self.grid.igspace(indices)
        self.assertEqual((100**3, 4), result.shape)
        npt.assert_almost_equal((-0.5, -0.5, -0.5, 1), result[0])
        npt.assert_almost_equal((0.5, 0.5, 0.5, 1), result[100**3 - 1])

    def test_gwspace1(self):
        indices = self.grid.indices()
        gspace = self.grid.igspace(indices)
        result = self.grid.gwspace(gspace)
        self.assertEqual((100**3, 4), result.shape)
        npt.assert_almost_equal((-0.5, -0.5, -0.5, 1), result[0])
        npt.assert_almost_equal((0.5, 0.5, 0.5, 1), result[100**3 - 1])

    def test_nelements1(self):
        self.assertEqual(self.grid.nelements, 100**3)

    def test_stamp1(self):
        self.grid.stamp(_scalar_stamp)
        npt.assert_almost_equal(-0.5, self.grid.array[0, :, :])
        npt.assert_almost_equal(0.5, self.grid.array[99, :, :])


class DefaultValueTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100)), default=-1)

    def test_call1(self):
        '''Out of bounds returns default value'''
        result = self.grid([[-0.6, -0.6, -0.6, 1]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)

    def test_call2(self):
        '''In bounds returns grid value'''
        result = self.grid([[0.5, 0.5, 0.5, 1]])
        expected = [1]
        npt.assert_almost_equal(expected, result)

    def test_call3(self):
        '''One out of bounds coordinate returns default value (positive)'''
        result = self.grid([[0.5, 100, 0.5, 1]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)

    def test_call4(self):
        '''One out of bounds coordinate returns default value (negative)'''
        result = self.grid([[-0.6, 0.5, 0.5, 1]])
        expected = [-1]
        npt.assert_almost_equal(expected, result)


class VectorGridTestCase(unittest.TestCase):

    def setUp(self):
        self.grid = volpy.Grid(np.ones((100, 100, 100, 3)))

    def test_call1(self):
        result = self.grid([[-0.6, -0.6, -0.6, 1]])
        expected = [[0, 0, 0]]
        npt.assert_almost_equal(expected, result)

    def test_call2(self):
        result = self.grid([[-0.5, -0.5, -0.5, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call3(self):
        result = self.grid([[0, 0, 0, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call4(self):
        result = self.grid([[0.5, 0.5, 0.5, 1]])
        expected = [[1, 1, 1]]
        npt.assert_almost_equal(expected, result)

    def test_call5(self):
        result = self.grid([[0.6, 0.6, 0.6, 1]])
        expected = [[0, 0, 0]]
        npt.assert_almost_equal(expected, result)

    # XXX: Tests for stamp() with vector grids.
