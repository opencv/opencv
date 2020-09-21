ocv_clear_vars(HAVE_WEBGPU)
ocv_clear_vars(DAWN_EMSDK)
ocv_clear_vars(DAWN_METAL)
if(WITH_WEBGPU)
  set(DAWN_EMSDK 1)
  set(WEBGPU_HEADER_DIRS "$ENV{WEBGPU_ROOT_DIR}/out/Release/gen/src/include")
  set(WEBGPU_INCLUDE_DIRS "$ENV{WEBGPU_ROOT_DIR}/src/include")
  set(WEBGPU_LIBRARIES "$ENV{WEBGPU_ROOT_DIR}/out/Release/gen/src/dawn")
endif()

try_compile(VALID_WEBGPU
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/webgpu.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${WEBGPU_INCLUDE_DIRS}\;${WEBGPU_HEADER_DIRS}"
                  "-DLINK_LIBRARIES:STRING=${WEBGPU_LIBRARIES}"
      OUTPUT_VARIABLE TRY_OUT
      )
if(NOT ${VALID_WEBGPU})
  message(WARNING "Can't use WebGPU-Dawn")
  return()
endif()
message(AUTHOR_WARNING "Use Dawn for native WebGPU")

set(HAVE_WEBGPU 1)
if(APPLE)
  set(DAWN_METAL 1)
endif()

if(NOT EMSCRIPTEN AND HAVE_WEBGPU)
  include_directories(${WEBGPU_INCLUDE_DIRS} ${WEBGPU_HEADER_DIRS})
  link_directories(${WEBGPU_LIBRARIES})
endif()