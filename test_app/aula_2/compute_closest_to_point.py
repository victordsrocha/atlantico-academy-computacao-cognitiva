from src.gen_functions import closest_to_point
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-pl', '--points_list',
                    default=np.array([[1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5], [2.5, 3.5]]),
                    help='first point to calculate the Euclidean distance')
    ap.add_argument('-fp', '--focus_point',
                    default=np.array([2.75, 2.25]),
                    help='second point to calculate the Euclidean distance')

    args = vars(ap.parse_args())

    points_list = args['points_list']
    focus_point = args['focus_point']
    result = closest_to_point(points_list, focus_point)
    print('points_list: \n{}'.format(points_list))
    print('The closest point to {} is: {}'.format(focus_point, result))


if __name__ == '__main__':
    main()
