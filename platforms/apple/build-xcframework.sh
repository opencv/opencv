#!/bin/bash

# This script builds OpenCV into an xcframework compatible with all Apple platforms.
# Just run it and grab a snack -- you'll be waiting a while.

set -e

cd "`dirname \"$0\"`"

mkdir -p ./xcframework-build/ios
mkdir -p ./xcframework-build/ios-simulator
mkdir -p ./xcframework-build/ios-maccatalyst
mkdir -p ./xcframework-build/macos

# Build OSX slices
python3 ../osx/build_framework.py --archs x86_64,arm64 --without=objc --build-only-specified-archs ./xcframework-build/macos
python3 ../osx/build_framework.py --catalyst_archs x86_64,arm64 --without=objc --build-only-specified-archs ./xcframework-build/ios-maccatalyst
# Build iOS slices
python3 ../ios/build_framework.py --iphoneos_archs arm64,armv7 --without=objc --build-only-specified-archs ./xcframework-build/ios
python3 ../ios/build_framework.py --iphonesimulator_archs x86_64,arm64 --without=objc --build-only-specified-archs ./xcframework-build/ios-simulator
# Stitch them all together into an xcframework
xcodebuild -create-xcframework \
    -framework ./xcframework-build/ios/opencv2.framework \
    -framework ./xcframework-build/ios-simulator/opencv2.framework \
    -framework ./xcframework-build/macos/opencv2.framework \
    -framework ./xcframework-build/ios-maccatalyst/opencv2.framework \
    -output ./xcframework-build/opencv2.xcframework
