if(WITH_ITT)
  if(BUILD_ITT)
    add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/ittnotify")
    set(ITT_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/ittnotify/include")
    set(ITT_INCLUDE_DIRS "${ITT_INCLUDE_DIR}")
    set(ITT_LIBRARIES "ittnotify")
    set(HAVE_ITT 1)
  else()
    #TODO
  endif()
endif()

set(OPENCV_TRACE 1)
