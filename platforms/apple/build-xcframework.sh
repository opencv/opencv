#!/bin/bash

# This script builds OpenCV into an xcframework compatible with all Apple platforms.
# Just run it and grab a snack -- you'll be waiting a while.

set -e

cd "`dirname \"$0\"`"

mkdir -p ./xcframework-build/catalyst
mkdir -p ./xcframework-build/iphoneos
mkdir -p ./xcframework-build/iphonesimulator
mkdir -p ./xcframework-build/osx

# Build OSX slices
python3 ../osx/build_framework.py --catalyst_archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/catalyst
python3 ../osx/build_framework.py --archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/osx
# Build iOS slices
python3 ../ios/build_framework.py --iphoneos_archs arm64 --without=objc --build-only-specified-archs ./xcframework-build/iphoneos
python3 ../ios/build_framework.py --iphonesimulator_archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/iphonesimulator
# Stitch them all together into an xcframework
xcodebuild -create-xcframework -framework ./xcframework-build/catalyst/opencv2.framework -framework ./xcframework-build/osx/opencv2.framework -framework ./xcframework-build/iphonesimulator/opencv2.framework -framework ./xcframework-build/iphoneos/opencv2.framework -output ./xcframework-build/opencv2.xcframework