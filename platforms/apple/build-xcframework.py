#!/usr/bin/env python3
"""
The script builds OpenCV.framework for OSX.
This script builds OpenCV into an xcframework compatible the platforms
of your choice. Just run it and grab a snack; you'll be waiting a while.
"""

import sys, os, argparse, pathlib, traceback
from cv_build_utils import execute, print_error, print_header

assert sys.version_info >= (3, 6), "Python 3.6 or newer is required!"

def get_framework_build_command_for_platform(platform, destination, framework_name, only_64_bit=False):
    """
    Generates the build command that creates a framework supporting the given platform.
    This command can be handed off to the command line for execution.

    Parameters
    ----------
    platform : str
        The name of the platform you want to build for.
    destination : str
        The directory you want to build the framework into.
    framework_name : str
        The name of the generated framework.
    only_64_bit : bool, optional
        Build only 64-bit archs, by default False
    """
    destination = destination.replace(" ", "\\ ")  # Escape spaces in destination path
    if platform == 'macos':
        return ["python3", "../osx/build_framework.py", "--archs", "x86_64,arm64", "--framework_name", framework_name, "--build_only_specified_archs", destination]
    elif platform == 'ios-maccatalyst':
        # This is not a mistake. For Catalyst, we use the osx toolchain.
        # TODO: This is building with objc turned off due to an issue with CMake. See here for discussion: https://gitlab.kitware.com/cmake/cmake/-/issues/21436
        return ["python3", "../osx/build_framework.py", "--catalyst_archs", "x86_64,arm64", "--framework_name", framework_name, "--without=objc", "--build_only_specified_archs", destination]
    elif platform == 'ios':
        archs = "arm64" if only_64_bit else "arm64,armv7,armv7s"
        return ["python3", "../ios/build_framework.py", "--iphoneos_archs", archs, "--framework_name", framework_name, "--build_only_specified_archs", destination]
    elif platform == 'ios-simulator':
        archs = "x86_64,arm64" if only_64_bit else "x86_64,arm64,i386"
        return ["python3", "../ios/build_framework.py", "--iphonesimulator_archs", archs, "--framework_name", framework_name, "--build_only_specified_archs", destination]
    else:
        raise Exception(f"Platform {platform} has no associated build commands.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script builds OpenCV into an xcframework supporting the Apple platforms of your choice.')
    parser.add_argument('out', metavar='OUTDIR', help='The directory where the xcframework will be created')
    parser.add_argument('--platform', default='ios,ios-simulator,ios-maccatalyst,macos', help='Platforms to build for (default: ios,ios-simulator,ios-maccatalyst,macos)')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV xcframework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--only_64_bit', default=False, action='store_true', help='Build for 64-bit archs only')

    args = parser.parse_args()

    platforms = args.platform.split(',')
    print(f'Building for platforms: {platforms}')

    try:
        # Build .frameworks for each platform
        build_folders = []
        for platform in platforms:
            folder = f"./xcframework-build/{platform}"
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            build_folders.append(folder)
            framework_build_command = get_framework_build_command_for_platform(platform, folder, args.framework_name, args.only_64_bit)

            print("")
            print_header(f"Building frameworks for {platform}")
            execute(framework_build_command, cwd=os.getcwd())

        # Put all the built .frameworks together into a .xcframework
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