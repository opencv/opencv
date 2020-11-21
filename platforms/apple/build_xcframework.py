#!/usr/bin/env python3
"""
This script builds OpenCV into an xcframework compatible with the platforms
of your choice. Just run it and grab a snack; you'll be waiting a while.
"""

import sys, os, argparse, pathlib, traceback
from cv_build_utils import execute, print_error, print_header, get_xcode_version, get_cmake_version

if __name__ == "__main__":

    # Check for dependencies
    assert sys.version_info >= (3, 6), f"Python 3.6 or later is required! Current version is {sys.version_info}"
    # Need CMake 3.18.5/3.19 or later for a Silicon-related fix to building for the iOS Simulator.
    # See https://gitlab.kitware.com/cmake/cmake/-/issues/21425 for context.
    assert get_cmake_version() >= (3, 18, 5), f"CMake 3.18.5 or later is required. Current version is {get_cmake_version()}"
    # Need Xcode 12.2 for Apple Silicon support
    assert get_xcode_version() >= (12, 2), f"Xcode 12.2 command line tools or later are required! Current version is {get_xcode_version()}. \
    Run xcode-select to switch if you have multiple Xcode installs."

    # Parse arguments
    parser = argparse.ArgumentParser(description='This script builds OpenCV into an xcframework supporting the Apple platforms of your choice.')
    parser.add_argument('out', metavar='OUTDIR', help='The directory where the xcframework will be created')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV xcframework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--iphoneos_archs', default=None, help='select iPhoneOS target ARCHS. Default is "armv7,arm64"')
    parser.add_argument('--iphonesimulator_archs', default=None, help='select iPhoneSimulator target ARCHS. Default is "x86_64,arm64"')
    parser.add_argument('--macos_archs', default=None, help='Select MacOS ARCHS. Default is "x86_64,arm64"')
    parser.add_argument('--catalyst_archs', default=None, help='Select Catalyst ARCHS. Default is "x86_64,arm64"')
    parser.add_argument('--build_only_specified_archs', default=False, action='store_true', help='if enabled, only directly specified archs are built and defaults are ignored')
    parser.add_argument('passthrough_args', nargs=argparse.REMAINDER, help='Any flags not captured by this script will be passed through to the ios/osx build_framework.py scripts')

    args, unknown_args = parser.parse_known_args()
    if args.passthrough_args:
        print(f"The following args are not recognized by this script and will be passed through to the ios/osx build_framework.py scripts: {args.passthrough_args}")

    # Parse architectures from args

    iphoneos_archs = args.iphoneos_archs
    if not iphoneos_archs and not args.build_only_specified_archs:
        # Supply defaults
        iphoneos_archs = "armv7,arm64"
    print(f'Using iPhoneOS ARCHS={iphoneos_archs}')

    iphonesimulator_archs = args.iphonesimulator_archs
    if not iphonesimulator_archs and not args.build_only_specified_archs:
        # Supply defaults
        iphonesimulator_archs = "x86_64,arm64"
    print(f'Using iPhoneSimulator ARCHS={iphonesimulator_archs}')

    macos_archs = args.macos_archs
    if not macos_archs and not args.build_only_specified_archs:
        # Supply defaults
        macos_archs = "x86_64,arm64"
    print(f'Using MacOS ARCHS={macos_archs}')

    catalyst_archs = args.macos_archs
    if not catalyst_archs and not args.build_only_specified_archs:
        # Supply defaults
        catalyst_archs = "x86_64,arm64"
    print(f'Using Catalyst ARCHS={catalyst_archs}')

    # Build phase

    try:
        # Build .frameworks for each platform
        osx_script_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../osx/build_framework.py')
        ios_script_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios/build_framework.py')

        build_folders = []

        def get_or_create_build_folder(base_dir, platform):
            build_folder = f"./{base_dir}/{platform}".replace(" ", "\\ ")  # Escape spaces in output path
            pathlib.Path(build_folder).mkdir(parents=True, exist_ok=True)
            return build_folder

        if iphoneos_archs:
            build_folder = get_or_create_build_folder(args.out, "iphoneos")
            build_folders.append(build_folder)
            command = ["python3", ios_script_path, "--iphoneos_archs", iphoneos_archs, "--framework_name", args.framework_name, "--build_only_specified_archs", build_folder] + args.passthrough_args
            print_header("Building iPhoneOS frameworks")
            print(command)
            execute(command, cwd=os.getcwd())
        if iphonesimulator_archs:
            build_folder = get_or_create_build_folder(args.out, "iphonesimulator")
            build_folders.append(build_folder)
            command = ["python3", ios_script_path, "--iphonesimulator_archs", iphonesimulator_archs, "--framework_name", args.framework_name, "--build_only_specified_archs", build_folder] + args.passthrough_args
            print_header("Building iPhoneSimulator frameworks")
            execute(command, cwd=os.getcwd())
        if macos_archs:
            build_folder = get_or_create_build_folder(args.out, "macos")
            build_folders.append(build_folder)
            command = ["python3", osx_script_path, "--macos_archs", macos_archs, "--framework_name", args.framework_name, "--build_only_specified_archs", build_folder] + args.passthrough_args
            print_header("Building MacOS frameworks")
            execute(command, cwd=os.getcwd())
        if catalyst_archs:
            build_folder = get_or_create_build_folder(args.out, "catalyst")
            build_folders.append(build_folder)
            command = ["python3", osx_script_path, "--catalyst_archs", catalyst_archs, "--framework_name", args.framework_name, "--build_only_specified_archs", build_folder] + args.passthrough_args
            print_header("Building Catalyst frameworks")
            execute(command, cwd=os.getcwd())

        # Put all the built .frameworks together into a .xcframework
        print_header("Building xcframework")
        xcframework_build_command = [
            "xcodebuild",
            "-create-xcframework",
            "-output",
            f"{args.out}/{args.framework_name}.xcframework",
        ]
        for folder in build_folders:
            xcframework_build_command += ["-framework", f"{folder}/{args.framework_name}.framework"]
        execute(xcframework_build_command, cwd=os.getcwd())

        print("")
        print_header(f"Finished building {args.out}/{args.framework_name}.xcframework")
    except Exception as e:
        print_error(e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)