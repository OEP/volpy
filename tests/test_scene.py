import numpy.testing as npt

import unittest

import volpy


class SceneTestCase(unittest.TestCase):

    def setUp(self):
        self.scene = volpy.Scene()
        self.shape = (100, 100)

    def test_camera1(self):
        '''Test default camera'''
        self.assertIsInstance(self.scene.camera, volpy.Camera)
        npt.assert_almost_equal((0, 0, 0), self.scene.camera.eye)
        npt.assert_almost_equal((0, 0, 1), self.scene.camera.view)

    # XXX no-args render() test case

    def test_render1(self):
        '''Workers must be >=1'''
        with self.assertRaises(ValueError) as cm:
            self.scene.render(self.shape, workers=0)
        result, = cm.exception.args
        expected = 'Must have at least 1 worker.'
        self.assertEqual(expected, result)

    def test_render2(self):
        '''Tolerance must be >=1'''
        with self.assertRaises(ValueError) as cm:
            self.scene.render(self.shape, tol=0)
        result, = cm.exception.args
        expected = 'Tolerance must be >0.'
        self.assertEqual(expected, result)

    def test_render3(self):
        '''Shape must have length 2.'''
        with self.assertRaises(ValueError) as cm:
            self.scene.render((100,))
        result, = cm.exception.args
        expected = 'Shape must have length 2'
        self.assertEqual(expected, result)
