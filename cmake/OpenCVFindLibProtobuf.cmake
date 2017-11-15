# By default, we use protobuf sources from 3rdparty subdirectory and pre-generated .proto files
# Note: In case of .proto model updates these variables should be used:
# - Protobuf_PROTOC_EXECUTABLE (required)
# - Protobuf_INCLUDE_DIRS
# - Protobuf_LIBRARIES or Protobuf_LIBRARY / Protobuf_LIBRARY_DEBUG for find_package()
OCV_OPTION(BUILD_PROTOBUF "Force to build libprotobuf from sources" ON)
OCV_OPTION(PROTOBUF_UPDATE_FILES "Force to rebuild .proto files" OFF)

if(PROTOBUF_UPDATE_FILES)
  if(NOT COMMAND PROTOBUF_GENERATE_CPP)
    find_package(Protobuf QUIET)
  endif()
  if(DEFINED Protobuf_PROTOC_EXECUTABLE AND EXISTS ${Protobuf_PROTOC_EXECUTABLE})
    message(STATUS "The protocol buffer compiler is found (${Protobuf_PROTOC_EXECUTABLE})")
  else()
    message(FATAL_ERROR "The protocol buffer compiler is not found (Protobuf_PROTOC_EXECUTABLE='${Protobuf_PROTOC_EXECUTABLE}')")
  endif()
endif()

if(NOT BUILD_PROTOBUF AND NOT (DEFINED Protobuf_INCLUDE_DIRS AND DEFINED Protobuf_LIBRARIES))
  find_package(Protobuf QUIET)
endif()

if(Protobuf_FOUND AND NOT BUILD_PROTOBUF)
  # nothing
else()
  set(Protobuf_LIBRARIES libprotobuf)
  set(Protobuf_INCLUDE_DIRS "${OpenCV_SOURCE_DIR}/3rdparty/protobuf/src")
  if(NOT TARGET ${Protobuf_LIBRARIES})
    add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/protobuf" "${OpenCV_BINARY_DIR}/3rdparty/protobuf")
  endif()
  set(Protobuf_FOUND 1)
endif()
