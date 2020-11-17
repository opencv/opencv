# Building for Apple Platforms

build_xcframework.py creates an xcframework supporting a variety of Apple platforms.

You'll need the following to run these steps:
- MacOS 10.15 or later
- Python 3.6 or later
- CMake 10.19 or later (make sure the `cmake` command is available on your PATH)
- Xcode 12.2 or later (and its command line tools)

You can then run build_xcframework.py, as below:
```
cd ~/<my_working_directory>
python opencv/platforms/apple/build_xcframework.py
```

Grab a coffee, because you'll be here for a while. By default this builds OpenCV for 10 architectures across 4 platforms:

- iOS: arm64, armv7, armv7s
- iOS Simulator: x86_64, arm64, i386
- MacOS: x86_64, arm64 (and Catalyst versions of each)

If everything's fine, you will eventually get `~/<my_working_directory>/apple/opencv2.xcframework`.

The script has some configuration options to exclude platforms and architectures you don't want to build for. Use the `--help` flag for more information.
