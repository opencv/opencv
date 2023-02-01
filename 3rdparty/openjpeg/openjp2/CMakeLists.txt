# Defines the source code for the library
set(OPENJPEG_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/thread.c
  ${CMAKE_CURRENT_SOURCE_DIR}/bio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/cio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/dwt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/event.c
  ${CMAKE_CURRENT_SOURCE_DIR}/image.c
  ${CMAKE_CURRENT_SOURCE_DIR}/invert.c
  ${CMAKE_CURRENT_SOURCE_DIR}/j2k.c
  ${CMAKE_CURRENT_SOURCE_DIR}/jp2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mct.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mqc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg.c
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_clock.c
  ${CMAKE_CURRENT_SOURCE_DIR}/pi.c
  ${CMAKE_CURRENT_SOURCE_DIR}/t1.c
  ${CMAKE_CURRENT_SOURCE_DIR}/t2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tcd.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tgt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/function_list.c
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_malloc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_array.c
)

option(OPJ_DISABLE_TPSOT_FIX "Disable TPsot==TNsot fix. See https://github.com/uclouvain/openjpeg/issues/254." OFF)
if(OPJ_DISABLE_TPSOT_FIX)
  add_definitions(-DOPJ_DISABLE_TPSOT_FIX)
endif()

# Special case for old i586-mingw32msvc-gcc cross compiler
# if(NOT WIN32 AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_C_COMPILER MATCHES ".*mingw32msvc.*" )
#   set(WIN32 YES)
# endif()

ocv_warnings_disable(CMAKE_C_FLAGS
    -Wundef -Wstrict-prototypes -Wcast-function-type
    -Wshadow   # v2.4.0: GCC
    -Wunused-function   # v2.4.0: Clang
)

ocv_warnings_disable(CMAKE_C_FLAGS /wd4819) # vs2019 Win64

add_library(${OPENJPEG_LIBRARY_NAME} STATIC ${OPENJPEG_SRCS})

target_compile_definitions(${OPENJPEG_LIBRARY_NAME} PUBLIC OPJ_STATIC)

ocv_include_directories("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

if(UNIX)
  target_link_libraries(${OPENJPEG_LIBRARY_NAME} PRIVATE m)
endif()

set_target_properties(${OPENJPEG_LIBRARY_NAME}
  PROPERTIES
    ${OPENJPEG_LIBRARY_PROPERTIES}
)

#################################################################################
# threading configuration
#################################################################################

option(OPJ_USE_THREAD "Build with thread/mutex support " ON)
if(NOT OPJ_USE_THREAD)
  add_definitions(-DMUTEX_stub)
endif()

find_package(Threads QUIET)

if(OPJ_USE_THREAD AND WIN32 AND NOT Threads_FOUND )
  add_definitions(-DMUTEX_win32)
  set(Threads_FOUND YES)
endif()

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_WIN32_THREADS_INIT)
  add_definitions(-DMUTEX_win32)
endif()

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT )
  add_definitions(-DMUTEX_pthread)
endif()

if(OPJ_USE_THREAD AND NOT Threads_FOUND)
  message(STATUS "No thread library found and thread/mutex support is required by OPJ_USE_THREAD option")
  set(OCV_CAN_BUILD_OPENJPEG FALSE PARENT_SCOPE)
endif()

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
  target_link_libraries(${OPENJPEG_LIBRARY_NAME} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${OPENJPEG_LIBRARY_NAME}
    EXPORT OpenCVModules
    ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev
  )
endif()
