if(NOT APPLE)
  set(HAVE_METAL 0)
  return()
endif()

if(IOS OR XROS)
  set(HAVE_METAL 0)
  message(WARNING "The initial OpenCV Metal runtime supports macOS builds only")
  return()
endif()

find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)

if(NOT METAL_LIBRARY OR NOT FOUNDATION_LIBRARY)
  set(HAVE_METAL 0)
  message(WARNING "Apple Metal frameworks were not found")
  return()
endif()

set(METAL_LIBRARIES ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})

find_program(XCRUN_EXECUTABLE xcrun)
if(NOT XCRUN_EXECUTABLE)
  set(HAVE_METAL 0)
  message(WARNING "xcrun was not found; Apple Metal shaders cannot be compiled")
  return()
endif()

execute_process(
  COMMAND "${XCRUN_EXECUTABLE}" -f metal
  RESULT_VARIABLE METAL_COMPILER_RESULT
  OUTPUT_VARIABLE METAL_COMPILER
  ERROR_VARIABLE METAL_COMPILER_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${XCRUN_EXECUTABLE}" -f metallib
  RESULT_VARIABLE METALLIB_COMPILER_RESULT
  OUTPUT_VARIABLE METALLIB_COMPILER
  ERROR_VARIABLE METALLIB_COMPILER_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT METAL_COMPILER_RESULT EQUAL 0 OR NOT METALLIB_COMPILER_RESULT EQUAL 0)
  set(HAVE_METAL 0)
  message(WARNING "Apple Metal Toolchain is unavailable. Install it with: xcodebuild -downloadComponent MetalToolchain\n${METAL_COMPILER_ERROR}${METALLIB_COMPILER_ERROR}")
  return()
endif()

set(METAL_COMPILER "${METAL_COMPILER}" CACHE INTERNAL "Apple Metal compiler")
set(METALLIB_COMPILER "${METALLIB_COMPILER}" CACHE INTERNAL "Apple metallib linker")
set(XCRUN_EXECUTABLE "${XCRUN_EXECUTABLE}" CACHE INTERNAL "Apple xcrun tool")
set(METAL_SDK "macosx" CACHE INTERNAL "Apple SDK used for Metal shaders")

try_compile(VALID_METAL
  "${OpenCV_BINARY_DIR}"
  SOURCES "${OpenCV_SOURCE_DIR}/cmake/checks/metal.mm"
  CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=${METAL_LIBRARIES}"
  OUTPUT_VARIABLE METAL_TRY_COMPILE_OUTPUT
)

if(VALID_METAL)
  set(HAVE_METAL 1)
else()
  set(HAVE_METAL 0)
  message(WARNING "Apple Metal is not available:\n${METAL_TRY_COMPILE_OUTPUT}")
endif()
