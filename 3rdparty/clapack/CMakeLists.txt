# ----------------------------------------------------------------------------
#  CMake file for opencv_lapack. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(clapack)

# TODO: extract it from sources somehow
set(CLAPACK_VERSION "3.9.0" PARENT_SCOPE)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# The .cpp files:
file(GLOB lapack_srcs src/*.c)
file(GLOB runtime_srcs runtime/*.c)
file(GLOB lib_hdrs include/*.h)

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------

set(the_target "libclapack")

add_library(${the_target} STATIC ${lapack_srcs} ${runtime_srcs} ${lib_hdrs})

ocv_warnings_disable(CMAKE_C_FLAGS -Wno-parentheses -Wno-uninitialized -Wno-array-bounds
    -Wno-implicit-function-declaration -Wno-unused -Wunused-parameter -Wstringop-truncation
    -Wtautological-negation-compare) # gcc/clang warnings
ocv_warnings_disable(CMAKE_C_FLAGS /wd4244 /wd4554 /wd4723 /wd4819) # visual studio warnings

set_target_properties(${the_target}
  PROPERTIES OUTPUT_NAME ${the_target}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${the_target}
  COMPILE_PDB_NAME_DEBUG "${the_target}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

set(CLAPACK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
set(CLAPACK_LIBRARIES ${the_target} PARENT_SCOPE)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${the_target} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(clapack lapack_LICENSE)
