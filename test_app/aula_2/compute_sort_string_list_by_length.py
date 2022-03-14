from src.gen_functions import sort_string_list_by_length
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-sl', '--string_list',
                    default=['4444', '1', '333', '666666', '55555', '22'],
                    help='list of strings')
    ap.add_argument('-dc', '--descending',
                    default=False,
                    help='true for descending order and false otherwise')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    descending = args['descending']
    result = sort_string_list_by_length(string_list, descending)

    print('Original list: {}'.format(string_list))
    print('descending: {}'.format(descending))
    print('Sorted list by length of elements: {}'.format(result))


if __name__ == '__main__':
    main()
