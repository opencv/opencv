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


def _remove_stale_pyi_files(directory):
    """Remove .pyi files and py.typed markers from the directory tree.

    During incremental builds, disabling a previously enabled module leaves
    stale typing stubs in the loader directory from a previous copy.  Since
    copy_tree merges rather than replaces, those stale files persist.
    Removing all stub files before copying ensures only stubs for currently
    enabled modules are present.  Runtime .py files are not affected.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        for fname in filenames:
            if fname.endswith('.pyi') or fname == 'py.typed':
                os.remove(os.path.join(dirpath, fname))


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
    if os.path.isdir(args.output_dir):
        _remove_stale_pyi_files(args.output_dir)
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
