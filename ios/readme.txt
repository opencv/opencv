Assuming that your build directory is on the same level that opencv source,
From the build directory run
  ../opencv/ios/configure-device_xcode.sh
or
  ../opencv/ios/configure-simulator_xcode.sh

Then from the same folder invoke

xcodebuild -sdk iphoneos -configuration Release -target ALL_BUILD
xcodebuild -sdk iphoneos -configuration Release -target install install

or

xcodebuild -sdk iphonesimulator -configuration Release -target ALL_BUILD
xcodebuild -sdk iphonesimulator -configuration Release -target install install