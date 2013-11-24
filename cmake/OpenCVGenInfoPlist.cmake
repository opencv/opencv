if(IOS)
  configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.plist.in"
                 "${CMAKE_BINARY_DIR}/ios/Info.plist")
endif()
