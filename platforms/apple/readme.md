# Building OpenCV xcframework for iOS

`build_xcframework.py` creates an xcframework with separate iOS device and simulator slices.

## Requirements

- macOS 10.15 or later
- Python 3.6 or later
- CMake 3.18.5 or later (`cmake` on your PATH)
- Xcode 12.2 or later (command line tools)

## Quick start (Apple Silicon)

```bash
cd ~/<my_working_directory>
python3 opencv/platforms/apple/build_xcframework.py --out ios \
  --iphoneos_archs arm64 \
  --iphonesimulator_archs arm64 \
  --build_only_specified_archs \
  --disable-bitcode
```

Result: `ios/opencv2.xcframework` with `ios-arm64` and `ios-arm64-simulator` slices.

## Intel Mac simulator (x86_64)

```bash
python3 opencv/platforms/apple/build_xcframework.py --out ios \
  --iphoneos_archs arm64 \
  --iphonesimulator_archs x86_64 \
  --build_only_specified_archs \
  --disable-bitcode
```

## Default build (all default archs)

Without `--build_only_specified_archs`, the script builds:

- iPhoneOS: `armv7`, `arm64`
- iPhoneSimulator: `x86_64`, `arm64`

```bash
python3 opencv/platforms/apple/build_xcframework.py --out ios --disable-bitcode
```

## Passthrough arguments

Unrecognized flags are passed to `platforms/ios/build_framework.py`, for example:

```bash
python3 opencv/platforms/apple/build_xcframework.py --out ios \
  --iphoneos_archs arm64 --iphonesimulator_archs arm64 \
  --build_only_specified_archs --contrib opencv_contrib --without video
```

## How it works

The script builds one `.framework` per platform (device and/or simulator), then runs `xcodebuild -create-xcframework` to combine them. This is required when the same architecture (e.g. `arm64`) appears on both device and simulator.

See also [CMake issue #21425](https://gitlab.kitware.com/cmake/cmake/-/issues/21425) and [CMake issue #20989](https://gitlab.kitware.com/cmake/cmake/-/issues/20989) for Apple Silicon cross-compile context.

## Legacy fat framework

`platforms/ios/build_framework.py` still builds a single universal fat `.framework` for one platform at a time. Do not combine device and simulator `arm64` in one fat framework; use `build_xcframework.py` instead.
