from src.gen_functions import occurrences
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['maria', 'vania', 'vagner', 'vagner', 'vagner', 'maria', 'dirceu', 'ana', 'maria'],
                    help='list of strings')
    ap.add_argument('-p2', '--string_key',
                    default='maria',
                    help='key string')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    string_key = args['string_key']
    result = occurrences(string_list, string_key)
    print('string_list: {}'.format(string_list))
    print('string_key: {}'.format(string_key))
    print('result: {}'.format(result))


if __name__ == '__main__':
    main()
