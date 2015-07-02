from PIL import Image
import numpy as np

import argparse


def render(scene, args):
    array = scene.render(
        args.dimensions, step=args.step, workers=args.workers,
        method=args.method
    )
    array = (255 * array).astype(np.uint8)
    return Image.fromarray(array)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimensions', nargs=2, type=int,
                        default=(1920, 1080))
    parser.add_argument('-s', '--step', type=float, default=0.001)
    parser.add_argument('-m', '--method', default='thread')
    parser.add_argument('-w', '--workers', type=int, default=None)
    return parser
