'''
Benchmark script for volpy.Grid.stamp().

'''
import numpy as np
import volpy
from libbenchmark import get_parser

CENTER = np.array([0, 0, 2.5, 1])
RADIUS = 1.0


def sphere(x):
    norms = np.linalg.norm(x - CENTER, axis=1)
    return np.where(norms < RADIUS, 1, 0)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    transform = volpy.translate(*CENTER[:3])

    grid = volpy.Grid(np.ones(args.grid_shape),
                      transform=transform,
                      default=args.default)
    grid.stamp(sphere)
    if args.output:
        np.save(args.output, grid.array)


def _get_parser():
    parser = get_parser()
    parser.add_argument('-D', '--default', type=float, default=0)
    parser.add_argument('-g', '--grid-shape', type=int, nargs=3,
                        default=(100, 100, 100))
    parser.add_argument('-o', '--output')
    return parser

if __name__ == '__main__':
    main()
