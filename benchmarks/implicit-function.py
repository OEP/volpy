import numpy as np
import volpy
from libbenchmark import render, get_parser

CENTER = np.array([0, 0, 2.5, 1])
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


def plane_light(x):
    f = x[:, 1] / 2
    return np.where(f > 0, f, 0)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    element = volpy.Element(sphere)
    if args.color:
        element.color = sphere_color

    scene = volpy.Scene(scatter=args.scatter)
    if args.diffuse:
        scene.diffuse = element
        scene.lights.append(volpy.Light(plane_light))
    else:
        scene.ambient = element

    image = render(scene, args)
    image.save(args.output)


def _get_parser():
    parser = get_parser()
    parser.add_argument('-o', '--output', default='out.png')
    parser.add_argument('-c', '--color', action='store_true')
    parser.add_argument('-k', '--scatter', type=float, default=10)
    parser.add_argument('-D', '--diffuse', action='store_true')
    return parser

if __name__ == '__main__':
    main()
