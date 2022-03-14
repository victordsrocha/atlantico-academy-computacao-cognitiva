from src.gen_functions import matrix_thresholding
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
                    help='a numpy array with two dimensions')
    ap.add_argument('-th', '--threshold',
                    default=12,
                    help='the threshold')

    args = vars(ap.parse_args())

    matrix = args['matrix']
    threshold = args['threshold']
    result = matrix_thresholding(matrix, threshold)
    print('Original matrix:\n{}'.format(matrix))
    print('\nThreshold = {}\nThresholded matrix:\n{}'.format(threshold, result))


if __name__ == '__main__':
    main()
