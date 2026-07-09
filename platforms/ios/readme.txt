Building OpenCV from Source, using CMake and Command Line
=========================================================

Fat framework (single platform):

cd ~/<my_working_directory>
python3 opencv/platforms/ios/build_framework.py ios

xcframework (iOS device + simulator, recommended for Apple Silicon):

cd ~/<my_working_directory>
python3 opencv/platforms/apple/build_xcframework.py --out ios \
  --iphoneos_archs arm64 --iphonesimulator_archs arm64 \
  --build_only_specified_archs --disable-bitcode

See platforms/apple/readme.md for more options.

If everything's fine, you will get opencv2.framework or opencv2.xcframework under your output directory.
