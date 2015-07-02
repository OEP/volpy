import numpy as np
import volpy
from libbenchmark import render, get_parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    grid = volpy.Grid(np.ones(args.grid_shape), default=args.default)
    scene = volpy.Scene(emit=grid, scatter=args.scatter)
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
    return parser

if __name__ == '__main__':
    main()
