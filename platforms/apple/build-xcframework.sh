#!/bin/bash

# This script builds OpenCV into an xcframework compatible with all Apple platforms.
# Just run it and grab a snack -- you'll be waiting a while.

set -e

cd "`dirname \"$0\"`"

mkdir -p ./xcframework-build/catalyst-x86_64
mkdir -p ./xcframework-build/osx-x86_64
mkdir -p ./xcframework-build/catalyst-arm64
mkdir -p ./xcframework-build/osx-arm64
mkdir -p ./xcframework-build/iphoneos-arm64
mkdir -p ./xcframework-build/iphonesimulator-x86_64
mkdir -p ./xcframework-build/iphonesimulator-arm64

# Build OSX slices
python3 ../osx/build_framework.py --archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/osx-x86_64
python3 ../osx/build_framework.py --archs arm64 --without=objc --build-only-specified-archs ./xcframework-build/osx-arm64
python3 ../osx/build_framework.py --catalyst_archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/catalyst-x86_64
python3 ../osx/build_framework.py --catalyst_archs arm64 --without=objc --build-only-specified-archs ./xcframework-build/catalyst-arm64
# Build iOS slices
python3 ../ios/build_framework.py --iphoneos_archs arm64 --without=objc --build-only-specified-archs ./xcframework-build/iphoneos-arm64
python3 ../ios/build_framework.py --iphonesimulator_archs x86_64 --without=objc --build-only-specified-archs ./xcframework-build/iphonesimulator-x86_64
python3 ../ios/build_framework.py --iphonesimulator_archs arm64 --without=objc --build-only-specified-archs ./xcframework-build/iphonesimulator-arm64
# Stitch them all together into an xcframework
xcodebuild -create-xcframework \
    -framework ./xcframework-build/catalyst-x86_64/opencv2.framework \
    -framework ./xcframework-build/catalyst-arm64/opencv2.framework \
    -framework ./xcframework-build/osx-x86_64/opencv2.framework \
    -framework ./xcframework-build/osx-arm64/opencv2.framework \
    -framework ./xcframework-build/iphonesimulator-arm64/opencv2.framework \
    -framework ./xcframework-build/iphonesimulator-x86_64/opencv2.framework \
    -framework ./xcframework-build/iphoneos-arm64/opencv2.framework \
    -output ./xcframework-build/opencv2.xcframework
