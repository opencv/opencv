#!/usr/bin/env python3
"""
The script builds OpenCV.framework for OSX.
This script builds OpenCV into an xcframework compatible the platforms
of your choice. Just run it and grab a snack; you'll be waiting a while.
"""

import sys, os, os.path, argparse, pathlib, traceback
from cv_build_utils import execute, print_error

def get_or_create_folder_for_platform(platform):
    folder_name = "./xcframework-build/%s" % platform
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The script builds OpenCV.xcframework for OSX.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built xcframework into')
    parser.add_argument('--platform', default='ios,ios-simulator,ios-maccatalyst,macos', help='Platforms to build for')

    args = parser.parse_args()

    platforms = args.platform.split(',')
    print('Building for platforms: ' + str(platforms))

    build_folders = []
    try:
        if 'macos' in platforms:
            folder = get_or_create_folder_for_platform('macos')
            execute("python3 ../osx/build_framework.py --archs x86_64,arm64 --without=objc --build-only-specified-archs %s" % folder)
            build_folders.append(folder)
        if 'ios-maccatalyst' in platforms:
            folder = get_or_create_folder_for_platform('ios-maccatalyst')
            execute("python3 ../osx/build_framework.py --catalyst_archs x86_64,arm64 --without=objc --build-only-specified-archs %s" % folder)
            build_folders.append(folder)
        if 'ios' in platforms:
            folder = get_or_create_folder_for_platform('ios')
            execute("python3 ../ios/build_framework.py --iphoneos_archs arm64,armv7 --without=objc --build-only-specified-archs %s" % folder)
            build_folders.append(folder)
        if 'ios-simulator' in platforms:
            folder = get_or_create_folder_for_platform('ios-simulator')
            execute("python3 ../ios/build_framework.py --iphonesimulator_archs x86_64,arm64 --without=objc --build-only-specified-archs %s" % folder)
            build_folders.append(folder)

        framework_args = " ".join(["-framework %s/opencv2.framework" % folder for folder in build_folders])
        xcframework_build_command = "xcodebuild -create-xcframework %s -output %s/opencv2.xcframework" % (framework_args, args.out)
        execute(xcframework_build_command)
        print("Finished building %s/opencv2.xcframework" % args.out)
    except Exception as e:
        print_error(e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)