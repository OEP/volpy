import numpy as np

import math


def translate(vx, vy, vz):
    return np.array([[1, 0, 0, vx],
                     [0, 1, 0, vy],
                     [0, 0, 1, vz],
                     [0, 0, 0, 1]])


def scale(sx, sy, sz):
    return np.array([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0, 1]])


def rotate_axis(axis, theta):
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
    return rotate_axis([1, 0, 0], theta)


def rotatey(theta):
    return rotate_axis([0, 1, 0], theta)


def rotatez(theta):
    return rotate_axis([0, 0, 1], theta)


def rotatexyz(alpha, beta, gamma):
    return rotatez(alpha).dot(rotatey(beta).dot(rotatex(gamma)))


def cross(u, v, dtype=float):
    '''Cross product for homogenous vectors'''
    u = np.asarray(u)
    v = np.asarray(v)
    if not u.shape == (4,) or not v.shape == (4,):
        raise ValueError('Incompatible dimension for homogenous vectors')
    x = np.ndarray((4,), dtype=dtype)
    x[:3] = np.cross(u[:3], v[:3])
    x[3] = 0
    return x
