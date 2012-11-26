# Install script for directory: /home/rodrygojose/opencv/doc

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES
    "/home/rodrygojose/opencv/doc/haartraining.htm"
    "/home/rodrygojose/opencv/doc/check_docs_whitelist.txt"
    "/home/rodrygojose/opencv/doc/packaging.txt"
    "/home/rodrygojose/opencv/doc/license.txt"
    "/home/rodrygojose/opencv/doc/CMakeLists.txt"
    "/home/rodrygojose/opencv/doc/opencv.jpg"
    "/home/rodrygojose/opencv/doc/opencv-logo-white.png"
    "/home/rodrygojose/opencv/doc/acircles_pattern.png"
    "/home/rodrygojose/opencv/doc/opencv-logo.png"
    "/home/rodrygojose/opencv/doc/opencv-logo2.png"
    "/home/rodrygojose/opencv/doc/pattern.png"
    "/home/rodrygojose/opencv/doc/opencv2manager.pdf"
    "/home/rodrygojose/opencv/doc/opencv_user.pdf"
    "/home/rodrygojose/opencv/doc/opencv2refman.pdf"
    "/home/rodrygojose/opencv/doc/opencv_tutorials.pdf"
    "/home/rodrygojose/opencv/doc/opencv_cheatsheet.pdf"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc/vidsurv" TYPE FILE FILES
    "/home/rodrygojose/opencv/doc/vidsurv/TestSeq.doc"
    "/home/rodrygojose/opencv/doc/vidsurv/Blob_Tracking_Modules.doc"
    "/home/rodrygojose/opencv/doc/vidsurv/Blob_Tracking_Tests.doc"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")

