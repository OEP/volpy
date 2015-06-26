import numpy as np

import math

from ._util import unit, normalize, ascolumn

ASPECT_16_9 = 16. / 9.
ASPECT_16_10 = 16. / 10.
ASPECT_4_3 = 4. / 3.


class Camera(object):

    def __init__(self, eye, view, up=(0, 1, 0), fov=60.,
                 aspect_ratio=ASPECT_16_9, near=0.1, far=2.0):
        self.eye = eye
        self.view = view
        self.up = up
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

    @property
    def eye(self):
        return self._eye

    @eye.setter
    def eye(self, eye):
        self._eye = np.asarray(eye)

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, view):
        self._view = np.asarray(view)
        self._update()

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, up):
        self._up = np.asarray(up)
        self._update()

    @property
    def right(self):
        return self._right

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov
        self._update_fov()

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio):
        self._aspect_ratio = aspect_ratio
        self._update_fov()

    def cast(self, imx, imy):
        """
        Cast a ray through the image plane

        Parameters
        ----------
        imx : array-like
            The normalized X image component.
        imy : array-like
            The normalized Y image component.

        Returns
        -------
        origins : ndarray
            A collection of ray origins.
        directions : ndarray
            A collection of ray directions.
        """
        if not (
            np.all((0 <= imx) & (imx <= 1))
            and np.all((0 <= imy) & (imy <= 1))
        ):
            raise ValueError('imx and imy must be in range [0, 1]')
        if not len(imx) == len(imy):
            raise ValueError('imx and imy must have same length')
        x = ascolumn((2. * imx - 1.) * self._tan_hfov)
        y = ascolumn((2. * imy - 1.) * self._tan_vfov)
        up = [self.up for _ in range(len(y))]
        right = [self.right for _ in range(len(x))]
        view = [self.view for _ in range(len(x))]

        directions = normalize(y * up + x * right + view)
        origins = (np.array([self.eye for _ in range(len(directions))]) +
                   directions * self.near)
        return origins, directions

    def _update(self):
        try:
            up = self.up
            view = self.view
        except AttributeError:
            return
        normalize(view)
        up = unit(up - up.dot(view) * view)
        right = unit(np.cross(view, up))
        self._up = up
        self._view = view
        self._right = right

    def _update_fov(self):
        try:
            fov = self.fov
            aspect_ratio = self.aspect_ratio
        except AttributeError:
            return
        self._tan_hfov = math.tan(fov * math.pi / 180.0 / 2)
        self._tan_vfov = self._tan_hfov / aspect_ratio
