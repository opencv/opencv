# ----------------------------------------------------------------------------
#  CMake file for libpng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if(ARM AND ENABLE_NEON AND NOT AARCH64 AND NOT IOS)
  project(${PNG_LIBRARY} ASM)
else()
  project(${PNG_LIBRARY})
endif()

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" ${ZLIB_INCLUDE_DIR})

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)

if(ARM AND ENABLE_NEON AND NOT AARCH64 AND NOT IOS)
  list(APPEND lib_srcs arm/filter_neon.S arm/arm_init.c)
  add_definitions(-DPNG_ARM_NEON_OPT=2)
else()
  add_definitions(-DPNG_ARM_NEON_OPT=0)
endif()

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_definitions(-DPNG_CONFIGURE_LIBPNG)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

if (HAVE_WINRT)
  add_definitions(-DHAVE_WINRT)
endif()

add_library(${PNG_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})
target_link_libraries(${PNG_LIBRARY} ${ZLIB_LIBRARY})

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wcast-align)

set_target_properties(${PNG_LIBRARY}
  PROPERTIES OUTPUT_NAME ${PNG_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${PNG_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${PNG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PNG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${PNG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
