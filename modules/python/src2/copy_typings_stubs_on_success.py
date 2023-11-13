import argparse
import warnings
import os
import sys

if sys.version_info >= (3, 8, ):
    # shutil.copytree received the `dirs_exist_ok` parameter
    from functools import partial
    import shutil

    copy_tree = partial(shutil.copytree, dirs_exist_ok=True)
else:
    from distutils.dir_util import copy_tree


def main():
    args = parse_arguments()
    py_typed_path = os.path.join(args.stubs_dir, 'py.typed')
    if not os.path.isfile(py_typed_path):
        warnings.warn(
            '{} is missing, it means that typings stubs generation is either '
            'failed or has been skipped. Ensure that Python 3.6+ is used for '
            'build and there is no warnings during Python source code '
            'generation phase.'.format(py_typed_path)
        )
        return
    copy_tree(args.stubs_dir, args.output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Copies generated typing stubs only when generation '
        'succeeded. This is identified by presence of the `py.typed` file '
        'inside typing stubs directory.'
    )
    parser.add_argument('--stubs_dir', type=str,
                        help='Path to directory containing generated typing '
                        'stubs file')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
