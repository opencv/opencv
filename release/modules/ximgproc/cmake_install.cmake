# Install script for directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_ximgproc.3.2.0.dylib"
    "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_ximgproc.3.2.dylib"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.3.2.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.3.2.dylib"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/chihiro/Programs/opencv/opencv/release/lib"
        "${file}")
      execute_process(COMMAND /usr/bin/install_name_tool
        -add_rpath "/usr/local/lib"
        "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_ximgproc.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/chihiro/Programs/opencv/opencv/release/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/usr/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_ximgproc.dylib")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/deriche_filter.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/edge_filter.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/estimated_covariance.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/fast_hough_transform.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/fast_line_detector.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/lsc.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/paillou_filter.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/seeds.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/segmentation.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/slic.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/sparse_match_interpolator.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/structured_edge_detection.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/ximgproc" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/weighted_median_filter.hpp")
endif()

