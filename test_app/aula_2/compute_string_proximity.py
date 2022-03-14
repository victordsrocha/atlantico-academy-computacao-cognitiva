from src.gen_functions import string_proximity
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['abacate', 'arroz', 'pera', 'uva', 'banana', 'maça', 'repolho', 'uva', 'feijão', 'arroz'],
                    help='A list of strings.')
    ap.add_argument('-p2', '--analyzed_string',
                    default='arroz',
                    help='Analyzed string.')
    ap.add_argument('-p3', '--threshold',
                    default=3,
                    help='An integer indicating the analysis distance threshold.')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    analyzed_string = args['analyzed_string']
    threshold = args['threshold']

    result = string_proximity(string_list, analyzed_string, threshold)
    print('String list: {}'.format(string_list))
    print('analyzed_string = \'{}\'; threshold = {}'.format(analyzed_string, threshold))
    print('Result of the proximity analysis: ', result)


if __name__ == '__main__':
    main()
