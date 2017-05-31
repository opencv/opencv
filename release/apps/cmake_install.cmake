# Install script for directory: /Users/chihiro/Programs/opencv/opencv/apps

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/traincascade/cmake_install.cmake")
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/createsamples/cmake_install.cmake")
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/annotation/cmake_install.cmake")
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/visualisation/cmake_install.cmake")
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/interactive-calibration/cmake_install.cmake")
  include("/Users/chihiro/Programs/opencv/opencv/release/apps/version/cmake_install.cmake")

endif()

