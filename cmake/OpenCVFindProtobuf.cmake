# If protobuf is found - libprotobuf target is available

ocv_option(WITH_PROTOBUF "Enable libprotobuf" ON)
ocv_option(BUILD_PROTOBUF "Force to build libprotobuf from sources" ON)
ocv_option(PROTOBUF_UPDATE_FILES "Force rebuilding .proto files (protoc should be available)" OFF)

set(HAVE_PROTOBUF FALSE)

if(NOT WITH_PROTOBUF)
  return()
endif()

function(get_protobuf_version version include)
  file(STRINGS "${include}/google/protobuf/stubs/common.h" ver REGEX "#define GOOGLE_PROTOBUF_VERSION [0-9]+")
  string(REGEX MATCHALL "[0-9]+" ver ${ver})
  math(EXPR major "${ver} / 1000000")
  math(EXPR minor "${ver} / 1000 % 1000")
  math(EXPR patch "${ver} % 1000")
  set(${version} "${major}.${minor}.${patch}" PARENT_SCOPE)
endfunction()

if(BUILD_PROTOBUF)
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/protobuf")
  set(HAVE_PROTOBUF TRUE)
else()
  unset(Protobuf_VERSION CACHE)
  find_package(Protobuf QUIET)

  # Backwards compatibility
  # Define camel case versions of input variables
  foreach(UPPER
      PROTOBUF_FOUND
      PROTOBUF_LIBRARY
      PROTOBUF_INCLUDE_DIR
      PROTOBUF_VERSION
      )
      if (DEFINED ${UPPER})
          string(REPLACE "PROTOBUF_" "Protobuf_" Camel ${UPPER})
          if (NOT DEFINED ${Camel})
              set(${Camel} ${${UPPER}})
          endif()
      endif()
  endforeach()
  # end of compatibility block

  if(Protobuf_FOUND)
    if(TARGET protobuf::libprotobuf)
      add_library(libprotobuf INTERFACE)
      target_link_libraries(libprotobuf INTERFACE protobuf::libprotobuf)
    else()
      add_library(libprotobuf UNKNOWN IMPORTED)
      set_target_properties(libprotobuf PROPERTIES
        IMPORTED_LOCATION "${Protobuf_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}"
      )
      get_protobuf_version(Protobuf_VERSION "${Protobuf_INCLUDE_DIR}")
    endif()
    set(HAVE_PROTOBUF TRUE)
  endif()
endif()

if(HAVE_PROTOBUF AND PROTOBUF_UPDATE_FILES AND NOT COMMAND PROTOBUF_GENERATE_CPP)
  find_package(Protobuf QUIET)
  if(NOT COMMAND PROTOBUF_GENERATE_CPP)
    message(FATAL_ERROR "PROTOBUF_GENERATE_CPP command is not available")
  endif()
endif()

if(HAVE_PROTOBUF)
  list(APPEND CUSTOM_STATUS protobuf)
  list(APPEND CUSTOM_STATUS_protobuf "    Protobuf:"
    BUILD_PROTOBUF THEN "build (${Protobuf_VERSION})"
    ELSE "${Protobuf_LIBRARY} (${Protobuf_VERSION})")
endif()
