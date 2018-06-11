import numpy as np
import volpy
from libbenchmark import render, get_parser

import math
import sys


RED = (1, 0, 0)
BLUE = (0, 0, 1)


def grid_color(x, transform=None):
    siny = np.sin(45 * x[:, 1])
    mask = np.sin(siny + 45 * x[:, 0]) > 0
    mask = mask.reshape(len(mask), 1)
    return mask * RED + (1 - mask) * BLUE


class Texture(object):

    def __init__(self, color, transform):
        self.color = color
        self.transform = transform

    def __call__(self, x):
        x = self.transform.dot(x.T).T
        return self.color(x)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    transform = np.eye(4)
    for rst in args.order:
        if rst not in 'RST':
            print('Invalid transform type: ' + rst)
            return 1
        if rst == 'R' and args.rotate:
            args.rotate = np.array(args.rotate) * math.pi / 180
            transform = volpy.rotatexyz(*args.rotate).dot(transform)
        elif rst == 'S' and args.scale:
            transform = volpy.scale(*args.scale).dot(transform)
        elif rst == 'T' and args.translate:
            transform = volpy.translate(-args.translate[0],
                                        -args.translate[1],
                                        -args.translate[2]).dot(transform)

    grid = volpy.Grid(np.ones(args.grid_shape),
                      transform=transform,
                      default=args.default)
    element = volpy.Element(grid)
    if args.color:
        texture = Texture(grid_color, transform)
        element.color = texture

    scene = volpy.Scene(ambient=element, scatter=args.scatter)
    image = render(scene, args)
    image.save(args.output)


def _get_parser():
    parser = get_parser()
    parser.add_argument('-D', '--default', type=float, default=0)
    parser.add_argument('-g', '--grid-shape', type=int, nargs=3,
                        default=(100, 100, 100))
    parser.add_argument('-o', '--output', default='out.png')
    parser.add_argument('-c', '--color', action='store_true')
    parser.add_argument('-k', '--scatter', type=float, default=10)
    parser.add_argument('-T', '--translate', type=float, default=[0, 0, 2],
                        nargs=3)
    parser.add_argument('-R', '--rotate', type=float, nargs=3)
    parser.add_argument('-S', '--scale', type=float, nargs=3)
    parser.add_argument('-O', '--order', type=str, default='RST')
    return parser

if __name__ == '__main__':
    sys.exit(main())
