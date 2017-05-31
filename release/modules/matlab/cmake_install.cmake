# Install script for directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/matlab

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

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/chihiro/Programs/opencv/opencv_contrib/modules/matlab/include/")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/matlab/+cv" TYPE DIRECTORY FILES "/Users/chihiro/Programs/opencv/opencv/release/modules/matlab/+cv/")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/matlab" TYPE FILE FILES "/Users/chihiro/Programs/opencv/opencv/release/modules/matlab/cv.m")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(
    COMMAND /Users/chihiro/.pyenv/shims/python2.7
            /Users/chihiro/Programs/opencv/opencv_contrib/modules/matlab/generator/cvmex.py
            --jinja2 /Users/chihiro/Programs/opencv/opencv/3rdparty
            --opts=-largeArrayDims
            --include_dirs=-I/usr/local/include
            --lib_dir=-L/usr/local/lib
            --libs=-lopencv_dnn\ -lopencv_core\ -lopencv_imgproc\ -lopencv_ml\ -lopencv_imgcodecs\ -lopencv_videoio\ -lopencv_highgui\ -lopencv_objdetect\ -lopencv_flann\ -lopencv_features2d\ -lopencv_photo\ -lopencv_video\ -lopencv_videostab\ -lopencv_calib3d\ -lopencv_stitching\ -lopencv_superres\ -lopencv_xfeatures2d
            --flags=\ \ -fsigned-char\ -W\ -Wall\ -Werror=return-type\ -Werror=non-virtual-dtor\ -Werror=address\ -Werror=sequence-point\ -Wformat\ -Werror=format-security\ -Wmissing-declarations\ -Wmissing-prototypes\ -Wstrict-prototypes\ -Wundef\ -Winit-self\ -Wpointer-arith\ -Wshadow\ -Wsign-promo\ -Wno-narrowing\ -Wno-delete-non-virtual-dtor\ -Wno-unnamed-type-template-args\ -Wno-comment\ -fdiagnostics-show-option\ -Wno-long-long\ -Qunused-arguments\ -Wno-semicolon-before-method-body\ -fno-omit-frame-pointer\ -msse\ -msse2\ -mno-avx\ -msse3\ -mno-ssse3\ -mno-sse4.1\ -mno-sse4.2\ \ 
            --outdir /usr/local/matlab
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/chihiro/Programs/opencv/opencv/release/modules/matlab/test/cmake_install.cmake")

endif()

