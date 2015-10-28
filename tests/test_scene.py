import numpy.testing as npt
import numpy as np

import unittest

import volpy


class SceneTestCase(unittest.TestCase):

    def setUp(self):
        self.scene = volpy.Scene()
        self.shape = (100, 100)

    def test_camera1(self):
        '''Test default camera'''
        self.assertIsInstance(self.scene.camera, volpy.Camera)
        npt.assert_almost_equal((0, 0, 0, 1), self.scene.camera.eye)
        npt.assert_almost_equal((0, 0, 1, 0), self.scene.camera.view)

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

    def test_render4(self):
        '''At least one density element must be set.'''
        with self.assertRaises(ValueError) as cm:
            self.scene.render(self.shape)
        result, = cm.exception.args
        expected = 'At least one scene element is required.'
        self.assertEqual(expected, result)

    def test_render5(self):
        '''Invalid render method specified'''
        self.scene.ambient = volpy.Element(1)
        with self.assertRaises(ValueError) as cm:
            self.scene.render(self.shape, method='spoon')
        result, = cm.exception.args
        expected = 'Invalid method: spoon'
        self.assertEqual(expected, result)

    def test_render6(self):
        '''Scene.render() produces an image'''
        self.scene.ambient = volpy.Element(1)
        image = self.scene.render(self.shape)
        self.assertEqual((100, 100, 4), image.shape)
        pixel = image[0, 0]

        # Pixel RGBA values should all be the same positive value.
        self.assertTrue((pixel > 0).all())
        self.assertTrue(np.allclose(pixel[0], pixel))
        self.assertTrue(np.allclose(pixel[0], image))

    def test_render7(self):
        '''Scene.render() with a color argument'''
        self.scene.ambient = volpy.Element(1, (1, 0, 0))
        image = self.scene.render(self.shape)
        self.assertEqual((100, 100, 4), image.shape)
        pixel = image[0, 0]
        r, g, b, a = pixel

        # Red and alpha channels should be positive and the same. Green and
        # blue channels should be zero.
        self.assertTrue(r > 0 and a > 0)
        self.assertTrue(np.isclose(g, 0) and np.isclose(b, 0))
        self.assertTrue(np.allclose(r, image[:, :, 0]))
        self.assertTrue(np.allclose(g, image[:, :, 1]))
        self.assertTrue(np.allclose(b, image[:, :, 2]))
        self.assertTrue(np.allclose(a, image[:, :, 3]))

    def test_render8(self):
        '''Scene.render() with fork method'''
        self.scene.ambient = volpy.Element(1)
        image = self.scene.render(self.shape, workers=2, method='fork')
        self.assertEqual((100, 100, 4), image.shape)
