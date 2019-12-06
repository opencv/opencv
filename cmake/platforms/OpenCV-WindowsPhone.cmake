include("${CMAKE_CURRENT_LIST_DIR}/OpenCV-WinRT.cmake")

# Adding additional using directory for WindowsPhone 8.0 to get Windows.winmd properly
if(WINRT_8_0)
  set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} /AI\$(WindowsSDK_MetadataPath)")
endif()
