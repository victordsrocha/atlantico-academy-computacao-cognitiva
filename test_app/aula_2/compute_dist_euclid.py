from src.gen_functions import dist_euclid
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--point1',
                    default=np.array([0.0, -2.5]),
                    help='first point to calculate the Euclidean distance')
    ap.add_argument('-p2', '--point2',
                    default=np.array([0.0, 2.5]),
                    help='second point to calculate the Euclidean distance')

    args = vars(ap.parse_args())

    point1 = args['point1']
    point2 = args['point2']
    result = dist_euclid(point1, point2)
    print('The Euclidean distance of the points {} and {} is: {}'.format(point1, point2, result))


if __name__ == '__main__':
    main()
