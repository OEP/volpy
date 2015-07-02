import numpy as np
import volpy
import scipy.misc
from libbenchmark import render, get_parser

CENTER = np.array([0, 0, 2.5])
RADIUS = 1.0
RED = (1, 0, 0)
BLUE = (0, 0, 1)


def sphere(x):
    norms = np.linalg.norm(x - CENTER, axis=1)
    return np.where(norms < RADIUS, 1, 0)


def sphere_color(x):
    siny = np.sin(45 * x[:, 1])
    mask = np.sin(siny + 45 * x[:, 0]) > 0
    mask = mask.reshape(len(mask), 1)
    return mask * RED + (1 - mask) * BLUE


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if args.color:
        emit_color = sphere_color
    else:
        emit_color = None

    scene = volpy.Scene(emit=sphere, emit_color=emit_color,
                        scatter=args.scatter)
    image = render(scene, args)
    scipy.misc.imsave(args.output, image)


def _get_parser():
    parser = get_parser()
    parser.add_argument('-o', '--output', default='out.png')
    parser.add_argument('-c', '--color', action='store_true')
    parser.add_argument('-k', '--scatter', type=float, default=10)
    return parser

if __name__ == '__main__':
    main()
