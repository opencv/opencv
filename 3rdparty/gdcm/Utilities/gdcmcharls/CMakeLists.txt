cmake_minimum_required(VERSION 2.8.9)

if(NOT CHARLS_NAMESPACE)
  set(CHARLS_NAMESPACE "CHARLS")
  set(CHARLS_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${CHARLS_NAMESPACE} CHARLS_LIBRARY_NAME)

project(${CHARLS_NAMESPACE} CXX)

#-----------------------------------------------------------------------------
# CHARLS version number
set(CHARLS_MAJOR_VERSION 1)
set(CHARLS_MINOR_VERSION 0)
set(CHARLS_BUILD_VERSION 0)
set(CHARLS_VERSION
      "${CHARLS_MAJOR_VERSION}.${CHARLS_MINOR_VERSION}.${CHARLS_BUILD_VERSION}")

#-----------------------------------------------------------------------------
#set (EXECUTABLE_OUTPUT_PATH ${CHARLS_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
#set (LIBRARY_OUTPUT_PATH ${CHARLS_BINARY_DIR}/bin CACHE PATH "Single output directory for building all libraries.")
#mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

if(NOT CHARLS_INSTALL_BIN_DIR)
  set(CHARLS_INSTALL_BIN_DIR "bin")
endif()
if(NOT CHARLS_INSTALL_LIB_DIR)
  set(CHARLS_INSTALL_LIB_DIR "lib")
endif()
if(NOT CHARLS_INSTALL_INCLUDE_DIR)
  set(CHARLS_INSTALL_INCLUDE_DIR "include")
endif()

#set(MANGLE_PREFIX ${JPEGLS_LIBRARY_NAME})
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jpegls_mangle.h.in
#               ${CMAKE_CURRENT_BINARY_DIR}/jpegls_mangle.h
#               @ONLY )



# When user specify build type do not override settings:
if(NOT CMAKE_BUILD_TYPE)
  # The following compiler option are only meant for GCC:
  if(CMAKE_COMPILER_IS_GNUCC)
    #set(CMAKE_CXX_FLAGS "-Wall -O3 -g" )
  endif()
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

#option(charls_BUILD_SHARED_LIBS "Build CharLS with shared libraries." OFF)
#set(BUILD_SHARED_LIBS ${charls_BUILD_SHARED_LIBS})

if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DCHARLS_SHARED)
  endif()
endif()

add_library(${CHARLS_LIBRARY_NAME} header.cpp  interface.cpp  jpegls.cpp
  stdafx.cpp
  )
set_target_properties(${CHARLS_LIBRARY_NAME} PROPERTIES ${GDCM_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${CHARLS_LIBRARY_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${CHARLS_LIBRARY_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

# Install library
if(NOT CHARLS_INSTALL_NO_LIBRARIES)
  # Runtime
  install(TARGETS ${CHARLS_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${CHARLS_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${CHARLS_INSTALL_LIB_DIR} COMPONENT Libraries ${NAMELINK_SKIP}
    ARCHIVE DESTINATION ${CHARLS_INSTALL_LIB_DIR} COMPONENT DebugDevel
  )
  #Development
  if(NAMELINK_ONLY)
    install(TARGETS ${CHARLS_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
      LIBRARY DESTINATION ${CHARLS_INSTALL_LIB_DIR} COMPONENT DebugDevel ${NAMELINK_ONLY}
    )
  endif()
endif()

if(NOT CHARLS_INSTALL_NO_DEVELOPMENT)
  file(GLOB header_files "*.h")
  install(FILES ${header_files}
    DESTINATION ${CHARLS_INSTALL_INCLUDE_DIR} COMPONENT Headers
  )
endif()
#add_executable(charlstest test/test.cpp test/time.cpp)
#target_link_libraries (charlstest CharLS)

#include(CTest)
#add_test(charlstest charlstest)
