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

def get_build_command_for_platform(platform, only_64_bit=False):
    if platform == 'macos':
        return "python3 ../osx/build_framework.py --archs %s --without=objc --build-only-specified-archs" % "x86_64,arm64"
    elif platform == 'ios-maccatalyst':
        return "python3 ../osx/build_framework.py --catalyst_archs %s --without=objc --build-only-specified-archs" % "x86_64,arm64"
    elif platform == 'ios':
        return "python3 ../ios/build_framework.py --iphoneos_archs %s --without=objc --build-only-specified-archs" % "arm64" if only_64_bit else "arm64,armv7,armv7s"
    elif platform == 'ios-simulator':
        return "python3 ../ios/build_framework.py --iphonesimulator_archs %s --without=objc --build-only-specified-archs" % "x86_64,arm64"
    else:
        raise Exception("Platform %s has no associated build commands." % platform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script builds an OpenCV .xcframework supporting the Apple platforms of your choice.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built xcframework into')
    parser.add_argument('--platform', default='ios,ios-simulator,ios-maccatalyst,macos', help='platforms to build for')
    parser.add_argument('--only-64-bit', default=False, dest='only_64_bit', action='store_true', help='only build for 64-bit archs')

    args = parser.parse_args()

    platforms = args.platform.split(',')
    print('Building for platforms: ' + str(platforms))

    build_folders = []
    try:
        for platform in platforms:
            folder = get_or_create_folder_for_platform('macos')
            build_folders.append(folder)
            execute("%s %s" % (get_build_command_for_platform(platform, args.only_64_bit), folder))

        framework_args = " ".join(["-framework %s/opencv2.framework" % folder for folder in build_folders])
        xcframework_build_command = "xcodebuild -create-xcframework %s -output %s/opencv2.xcframework" % (framework_args, args.out)
        execute(xcframework_build_command)
        print("Finished building %s/opencv2.xcframework" % args.out)
    except Exception as e:
        print_error(e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)