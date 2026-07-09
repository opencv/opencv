#!/usr/bin/env python3
"""
Build OpenCV into an xcframework for iOS device and simulator targets.
"""

from __future__ import print_function

import argparse
import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
import traceback


def execute(cmd, cwd=None):
    print("Executing: %s" % " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, cwd=cwd)


def print_header(msg):
    print("")
    print("=" * 60)
    print(msg)
    print("=" * 60)
    print("")


def print_error(msg):
    print("ERROR: %s" % msg, file=sys.stderr)


if __name__ == "__main__":
    description = """
This script builds OpenCV into an xcframework supporting iOS device and simulator.
"""
    epilog = """
Any arguments not recognized by this script are passed through to build_framework.py.
"""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-o', '--out', metavar='OUTDIR', required=True,
                        help='The directory where the xcframework will be created')
    parser.add_argument('--framework_name', default='opencv2',
                        help='Name of OpenCV xcframework (default: opencv2)')
    parser.add_argument('--iphoneos_archs', default=None,
                        help='select iPhoneOS target ARCHS. Default is "armv7,armv7s,arm64"')
    parser.add_argument('--iphonesimulator_archs', default=None,
                        help='select iPhoneSimulator target ARCHS. Default is "i386,x86_64"')
    parser.add_argument('--build_only_specified_archs', default=False, action='store_true',
                        help='if enabled, only directly specified archs are built and defaults are ignored')

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("Passing through to build_framework.py: %s" % unknown_args)

    iphoneos_archs = args.iphoneos_archs
    if not iphoneos_archs and not args.build_only_specified_archs:
        iphoneos_archs = "armv7,armv7s,arm64"
        print('Using iPhoneOS ARCHS=%s' % iphoneos_archs)

    iphonesimulator_archs = args.iphonesimulator_archs
    if not iphonesimulator_archs and not args.build_only_specified_archs:
        iphonesimulator_archs = "i386,x86_64"
        print('Using iPhoneSimulator ARCHS=%s' % iphonesimulator_archs)

    if not iphoneos_archs and not iphonesimulator_archs:
        print_error("At least one of --iphoneos_archs or --iphonesimulator_archs must be specified")
        sys.exit(1)

    try:
        ios_script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'ios', 'build_framework.py'))

        build_folders = []

        def get_or_create_build_folder(base_dir, platform):
            build_folder = os.path.join(base_dir, platform)
            pathlib.Path(build_folder).mkdir(parents=True, exist_ok=True)
            return build_folder

        if iphoneos_archs:
            build_folder = get_or_create_build_folder(args.out, "iphoneos")
            build_folders.append(build_folder)
            command = [
                sys.executable, ios_script_path, build_folder,
                "--iphoneos_archs", iphoneos_archs,
                "--build_only_specified_archs",
            ] + unknown_args
            print_header("Building iPhoneOS framework")
            execute(command)

        if iphonesimulator_archs:
            build_folder = get_or_create_build_folder(args.out, "iphonesimulator")
            build_folders.append(build_folder)
            command = [
                sys.executable, ios_script_path, build_folder,
                "--iphonesimulator_archs", iphonesimulator_archs,
                "--build_only_specified_archs",
            ] + unknown_args
            print_header("Building iPhoneSimulator framework")
            execute(command)

        xcframework_path = os.path.join(args.out, "%s.xcframework" % args.framework_name)
        print_header("Building %s" % xcframework_path)

        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(xcframework_path)
            print("Removed existing xcframework at %s" % xcframework_path)

        xcframework_build_command = [
            "xcodebuild",
            "-create-xcframework",
            "-output", xcframework_path,
        ]
        for folder in build_folders:
            xcframework_build_command += [
                "-framework",
                os.path.join(folder, "%s.framework" % args.framework_name),
            ]
        execute(xcframework_build_command)

        print_header("Finished building %s" % xcframework_path)

    except Exception as e:
        print_error(e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
