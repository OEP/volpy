import numpy as np
import volpy
from libbenchmark import render, get_parser

import math


def main():
    parser = _get_parser()
    args = parser.parse_args()

    transform = np.eye(4)
    if args.rotate:
        args.rotate = np.array(args.rotate) * math.pi / 180
        transform = volpy.rotatexyz(*args.rotate).dot(transform)
    if args.scale:
        transform = volpy.scale(*args.scale).dot(transform)
    if args.translate:
        transform = volpy.translate(-args.translate[0],
                                    -args.translate[1],
                                    -args.translate[2]).dot(transform)

    grid = volpy.Grid(np.ones(args.grid_shape),
                      transform=transform,
                      default=args.default)
    scene = volpy.Scene(ambient=grid, scatter=args.scatter)
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
    return parser

if __name__ == '__main__':
    main()
