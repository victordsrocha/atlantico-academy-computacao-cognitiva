from src.gen_functions import region_of_interest_list
import argparse
import numpy as np


def main():
    default_matrix = np.array([[1, 2, 3, 4, 5],
                               [6, 7, 8, 9, 10],
                               [11, 12, 13, 14, 15],
                               [16, 17, 18, 19, 20],
                               [21, 22, 23, 24, 25]])

    ap = argparse.ArgumentParser()
    ap.add_argument('-ma', '--matrix',
                    default=default_matrix,
                    help='A numpy.ndarray with two dimensions.')
    ap.add_argument('-rc', '--roi_center_list',
                    default=[(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)],
                    help='A list containing tuples. Each tuple is composed of integers containing the coordinates '
                         'of the center of each region of interest')
    ap.add_argument('-rs', '--roi_shape_list',
                    default=[(3, 3), (3, 3), (3, 5), (5, 3), (5, 5)],
                    help='A list containing tuples. Each tuple is composed of integers containing the dimensions '
                         'of each region of interest.')

    args = vars(ap.parse_args())

    matrix = args['matrix']
    roi_center = args['roi_center_list']
    roi_shape = args['roi_shape_list']
    result = region_of_interest_list(matrix, roi_center, roi_shape)
    print('Original matrix:\n{}'.format(matrix))
    print('\nROIs:')
    for i in range(len(result)):
        print('center = {}, shape = {}'.format(matrix[roi_center[i][0]][roi_center[i][1]], roi_shape[i]))
        print('\n{}\n'.format(result[i]))


if __name__ == '__main__':
    main()
