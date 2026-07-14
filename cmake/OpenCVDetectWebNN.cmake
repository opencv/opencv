if(NOT EMSCRIPTEN)
  if(WITH_WEBNN)
    ocv_check_environment_variables(WEBNN_HEADER_DIRS)
    ocv_check_environment_variables(WEBNN_INCLUDE_DIRS)
    ocv_check_environment_variables(WEBNN_LIBRARIES)
    if(NOT DEFINED WEBNN_HEADER_DIRS)
      set(WEBNN_HEADER_DIRS "$ENV{WEBNN_NATIVE_DIR}/gen/src/include")
    endif()
    if(NOT DEFINED WEBNN_INCLUDE_DIRS)
      set(WEBNN_INCLUDE_DIRS "$ENV{WEBNN_NATIVE_DIR}/../../src/include")
    endif()
    if(NOT DEFINED WEBNN_LIBRARIES)
      set(WEBNN_LIBRARIES "$ENV{WEBNN_NATIVE_DIR}/libwebnn_native.so;$ENV{WEBNN_NATIVE_DIR}/libwebnn_proc.so")
    endif()
  endif()
  try_compile(VALID_WEBNN
        "${OpenCV_BINARY_DIR}"
        SOURCES "${OpenCV_SOURCE_DIR}/cmake/checks/webnn.cpp"
                "$ENV{WEBNN_NATIVE_DIR}/gen/src/webnn/webnn_cpp.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${WEBNN_INCLUDE_DIRS}\;${WEBNN_HEADER_DIRS}"
                    "-DLINK_LIBRARIES:STRING=${WEBNN_LIBRARIES}"
        OUTPUT_VARIABLE TRY_OUT
        )
else()
  try_compile(VALID_WEBNN
    "${OpenCV_BINARY_DIR}"
    SOURCES "${OpenCV_SOURCE_DIR}/cmake/checks/webnn.cpp"
    OUTPUT_VARIABLE TRY_OUT
    )
endif()

if(NOT VALID_WEBNN)
  if(NOT EMSCRIPTEN)
    message(WARNING "Can't use WebNN-native")
    return()
  else()
    message(WARNING "Can't use WebNN")
    return()
  endif()
else()
  set(HAVE_WEBNN ON)
  message(STATUS "Set HAVE_WEBNN = ${HAVE_WEBNN}")
endif()

if(NOT EMSCRIPTEN)
  message(AUTHOR_WARNING "Use WebNN-native")
else()
  message(AUTHOR_WARNING "Use WebNN")
endif()