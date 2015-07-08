'''
Functions for homogeneous coordinate systems.

'''

import numpy as np

import math


def translate(vx, vy, vz):
    '''
    Create a translation transformation matrix.

    Parameters
    ----------
    vx : scalar
        Translation amount in the X direction.
    vy : scalar
        Translation amount in the Y direction.
    vz : scalar
        Translation amount in the Z direction.

    Returns
    -------
    T : array
        A 4x4 matrix for homogeneous coordinate translation.
    '''
    return np.array([[1, 0, 0, vx],
                     [0, 1, 0, vy],
                     [0, 0, 1, vz],
                     [0, 0, 0, 1]])


def scale(sx, sy, sz):
    '''
    Create a scaling transformation matrix.

    Parameters
    ----------
    sx : scalar
        Scaling amount in the X direction.
    sy : scalar
        Scaling amount in the Y direction.
    sz : scalar
        Scaling amount in the Z direction.

    Returns
    -------
    S : array
        A 4x4 matrix for homogeneous coordinate scaling.
    '''
    return np.array([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0, 1]])


def rotate_axis(axis, theta):
    '''
    Rotation matrix for arbitrary axis-angle rotations.

    Parameters
    ----------
    axis : array-like
        A length 3 array representing the axis to rotate across. This should be
        normalized by the caller.
    theta : scalar
        Angle in radians.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    axis = np.asarray(axis)
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    return np.array([
        [
            a * a + b * b - c * c - d * d,
            2 * (b * c + a * d),
            2 * (b * d - a * c),
            0,
        ],
        [
            2 * (b * c - a * d),
            a * a + c * c - b * b - d * d,
            2 * (c * d + a * b),
            0,
        ],
        [
            2 * (b * d + a * c),
            2 * (c * d - a * b),
            a * a + d * d - b * b - c * c,
            0,
        ],
        [0, 0, 0, 1],
    ])


def rotatex(theta):
    '''
    Rotation matrix for rotating across the X axis. This is a convenience for
    ``rotate_axis``.

    Parameters
    ----------
    theta : scalar
        Angle in radians.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    return rotate_axis([1, 0, 0], theta)


def rotatey(theta):
    '''
    Rotation matrix for rotating across the Y axis. This is a convenience for
    ``rotate_axis``.

    Parameters
    ----------
    theta : scalar
        Angle in radians.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    return rotate_axis([0, 1, 0], theta)


def rotatez(theta):
    '''
    Rotation matrix for rotating across the Z axis. This is a convenience for
    ``rotate_axis``.

    Parameters
    ----------
    theta : scalar
        Angle in radians.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    return rotate_axis([0, 0, 1], theta)


def rotatexyz(alpha, beta, gamma):
    '''
    Rotation matrix for rotating across the X, Y, and Z axes, in that order.
    This is a convenience for ``rotate_axis``.

    Parameters
    ----------
    alpha : scalar
        Angle in radians to rotate across the X axis.
    beta : scalar
        Angle in radians to rotate across the Y axis.
    gamma : scalar
        Angle in radians to rotate across the Z axis.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    return rotatez(alpha).dot(rotatey(beta).dot(rotatex(gamma)))


def cross(u, v, dtype=float):
    '''
    Cross product for homogeneous vectors.

    Parameters
    ----------
    u : array-like
        First homogeneous vector.
    v : array-like
        Second homogeneous vector.
    dtype : data-type, optional
        A numpy data type.

    Returns
    -------
    R : array
        A 4x4 matrix for homogeneous coordinate rotations.
    '''
    u = np.asarray(u)
    v = np.asarray(v)
    if not u.shape == (4,) or not v.shape == (4,):
        raise ValueError('Incompatible dimension for homogenous vectors')
    x = np.ndarray((4,), dtype=dtype)
    x[:3] = np.cross(u[:3], v[:3])
    x[3] = 0
    return x
