import numpy.testing as npt
import numpy as np

import unittest

import volpy


def _func(xyz):
    return np.linalg.norm(xyz, axis=1)


class PEvalTestCase(unittest.TestCase):

    def setUp(self):
        self.xyz = np.ndarray((100, 4))
        self.xyz[:, 0] = np.linspace(0.1, 0.5, 100)
        self.xyz[:, 1] = np.linspace(-3, 0.2, 100)
        self.xyz[:, 2] = np.linspace(-3, 20, 100)
        self.xyz[:, 3] = 1
        self.expected = _func(self.xyz)

    def test_peval(self):
        result = volpy.peval(_func, self.xyz)
        npt.assert_almost_equal(result, self.expected)

    def test_peval_thread(self):
        result = volpy.peval(_func, self.xyz, method='thread')
        npt.assert_almost_equal(result, self.expected)

    def test_peval_fork(self):
        result = volpy.peval(_func, self.xyz, method='fork')
        npt.assert_almost_equal(result, self.expected)

    def test_peval_workers(self):
        result = volpy.peval(_func, self.xyz, workers=1)
        npt.assert_almost_equal(result, self.expected)

    def test_peval_workers_error(self):
        with self.assertRaises(ValueError) as cm:
            result = volpy.peval(_func, self.xyz, workers=0)
        result, = cm.exception.args
        expected = 'Must have at least 1 worker.'
        self.assertEqual(expected, result)

    def test_peval_method_error(self):
        with self.assertRaises(ValueError) as cm:
            result = volpy.peval(_func, self.xyz, method='spoon')
        result, = cm.exception.args
        expected = 'Invalid method: spoon'
        self.assertEqual(expected, result)
