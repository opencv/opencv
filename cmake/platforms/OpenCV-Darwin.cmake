if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64.*|ARM64.*")
  set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} -march=armv8.4-a+dotprod+fp16fml")
endif()
