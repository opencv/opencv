if(OPENCV_EXTRA_WORLD)
  set(OPENCV_APPLE_BUNDLE_NAME "OpenCV_contrib")
  set(OPENCV_APPLE_BUNDLE_ID "org.opencv_contrib")
else()
  set(OPENCV_APPLE_BUNDLE_NAME "OpenCV")
  set(OPENCV_APPLE_BUNDLE_ID "org.opencv")
endif()

if(IOS)
  configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.plist.in"
                 "${CMAKE_BINARY_DIR}/ios/Info.plist")
elseif(APPLE)
  configure_file("${OpenCV_SOURCE_DIR}/platforms/osx/Info.plist.in"
                 "${CMAKE_BINARY_DIR}/osx/Info.plist")
endif()
