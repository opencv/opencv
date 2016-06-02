set(OPENCV_APPLE_BUNDLE_NAME "OpenCV")
set(OPENCV_APPLE_BUNDLE_ID "org.opencv")

if(IOS)
  configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.plist.in"
                 "${CMAKE_BINARY_DIR}/ios/Info.plist")
elseif(APPLE)
  configure_file("${OpenCV_SOURCE_DIR}/platforms/osx/Info.plist.in"
                 "${CMAKE_BINARY_DIR}/osx/Info.plist")
endif()
