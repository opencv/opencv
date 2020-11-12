#!/usr/bin/env python3
"""
The script builds OpenCV.framework for OSX.
This script builds OpenCV into an xcframework compatible the platforms
of your choice. Just run it and grab a snack; you'll be waiting a while.
"""

import sys, os, os.path, argparse, pathlib, traceback
from cv_build_utils import execute, print_error, print_header

assert sys.version_info >= (3, 6), "Python 3.6 or newer is required!"

def get_or_create_folder_for_platform(platform):
    folder_name = f"./xcframework-build/{platform}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name

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
    if platform == 'macos':
        return f"python3 ../osx/build_framework.py --archs x86_64,arm64 --framework_name {framework_name} --without=objc --build-only-specified-archs {destination}"
    elif platform == 'ios-maccatalyst':
        # This is not a mistake. For Catalyst, we use the osx toolchain.
        return f"python3 ../osx/build_framework.py --catalyst_archs x86_64,arm64 --framework_name {framework_name} --without=objc --build-only-specified-archs {destination}"
    elif platform == 'ios':
        archs = "arm64" if only_64_bit else "arm64,armv7,armv7s"
        return f"python3 ../ios/build_framework.py --iphoneos_archs {archs} --framework_name {framework_name} --without=objc --build-only-specified-archs {destination}"
    elif platform == 'ios-simulator':
        return f"python3 ../ios/build_framework.py --iphonesimulator_archs x86_64,arm64 --framework_name {framework_name} --without=objc --build-only-specified-archs {destination}"
    else:
        raise Exception(f"Platform {platform} has no associated build commands.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script builds an OpenCV .xcframework supporting the Apple platforms of your choice.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built xcframework into')
    parser.add_argument('--platform', default='ios,ios-simulator,ios-maccatalyst,macos', help='platforms to build for')
    parser.add_argument('--framework-name', default='opencv2', dest='framework_name', action='store_true', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--only-64-bit', default=False, dest='only_64_bit', action='store_true', help='only build for 64-bit archs')

    args = parser.parse_args()

    platforms = args.platform.split(',')
    print(f'Building for platforms: {platforms}')

    try:
        # Build .frameworks for each platform
        build_folders = []
        for platform in platforms:
            folder = get_or_create_folder_for_platform(platform)
            build_folders.append(folder)
            framework_build_command = get_framework_build_command_for_platform(platform, folder, args.framework_name, args.only_64_bit)

            print("")
            print_header(f"Building frameworks for {platform}")
            execute(framework_build_command)

        # Put all the built .frameworks together into a .xcframework
        framework_args = " ".join([f"-framework {folder}/{args.framework_name}.framework" for folder in build_folders])
        xcframework_build_command = f"xcodebuild -create-xcframework {framework_args} -output {args.out}/{args.framework_name}.xcframework"
        execute(xcframework_build_command)

        print("")
        print_header(f"Finished building {args.out}/{args.framework_name}.xcframework")
    except Exception as e:
        print_error(e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)