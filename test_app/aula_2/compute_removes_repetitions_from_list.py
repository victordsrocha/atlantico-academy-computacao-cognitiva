from src.gen_functions import removes_repetitions_from_list
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-il', '--input_list',
                    default=[True, 'uva', 'uva', 2, 'uva', 'uva', 'uva', 2, 'feijão', 'arroz', 2, True],
                    help='A list with elements of any type.')

    args = vars(ap.parse_args())

    input_list = args['input_list']

    result = removes_repetitions_from_list(input_list)
    print('The input list: {}'.format(input_list))
    print('The input list without repetitions: {}'.format(result))


if __name__ == '__main__':
    main()
