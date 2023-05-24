include_regular_expression("^.*$")

#
install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/opj_config.h
 DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers)

include_directories(
  ${${OPENJPEG_NAMESPACE}_BINARY_DIR}/src/lib/openjp2 # opj_config.h and opj_config_private.h
)
# Defines the source code for the library
set(OPENJPEG_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/thread.c
  ${CMAKE_CURRENT_SOURCE_DIR}/thread.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/bio.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/cio.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dwt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/dwt.h
  ${CMAKE_CURRENT_SOURCE_DIR}/event.c
  ${CMAKE_CURRENT_SOURCE_DIR}/event.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ht_dec.c
  ${CMAKE_CURRENT_SOURCE_DIR}/image.c
  ${CMAKE_CURRENT_SOURCE_DIR}/image.h
  ${CMAKE_CURRENT_SOURCE_DIR}/invert.c
  ${CMAKE_CURRENT_SOURCE_DIR}/invert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/j2k.c
  ${CMAKE_CURRENT_SOURCE_DIR}/j2k.h
  ${CMAKE_CURRENT_SOURCE_DIR}/jp2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/jp2.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mct.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mct.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mqc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mqc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mqc_inl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg.c
  ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_clock.c
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_clock.h
  ${CMAKE_CURRENT_SOURCE_DIR}/pi.c
  ${CMAKE_CURRENT_SOURCE_DIR}/pi.h
  ${CMAKE_CURRENT_SOURCE_DIR}/t1.c
  ${CMAKE_CURRENT_SOURCE_DIR}/t1.h
  ${CMAKE_CURRENT_SOURCE_DIR}/t2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/t2.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tcd.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tcd.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tgt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tgt.h
  ${CMAKE_CURRENT_SOURCE_DIR}/function_list.c
  ${CMAKE_CURRENT_SOURCE_DIR}/function_list.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_codec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_includes.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_intmath.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_malloc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_malloc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opj_stdint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_array.c
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_array.h
)
if(BUILD_JPIP)
  add_definitions(-DUSE_JPIP)
  set(OPENJPEG_SRCS
    ${OPENJPEG_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cidx_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cidx_manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/phix_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/ppix_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/thix_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tpix_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/indexbox_manager.h
  )
endif()

option(OPJ_DISABLE_TPSOT_FIX "Disable TPsot==TNsot fix. See https://github.com/uclouvain/openjpeg/issues/254." OFF)
if(OPJ_DISABLE_TPSOT_FIX)
  add_definitions(-DOPJ_DISABLE_TPSOT_FIX)
endif()

# Special case for old i586-mingw32msvc-gcc cross compiler
if(NOT WIN32 AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_C_COMPILER MATCHES ".*mingw32msvc.*" )
  set(WIN32 YES)
endif()

# Build the library
if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
  add_library(${OPENJPEG_LIBRARY_NAME} ${OPENJPEG_SRCS})
  set(INSTALL_LIBS ${OPENJPEG_LIBRARY_NAME})
else()
  if(BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
    # Builds both static and dynamic libs
    add_library(${OPENJPEG_LIBRARY_NAME} SHARED ${OPENJPEG_SRCS})
    add_library(openjp2_static STATIC ${OPENJPEG_SRCS})
    set_target_properties(openjp2_static PROPERTIES OUTPUT_NAME ${OPENJPEG_LIBRARY_NAME})
    set(INSTALL_LIBS ${OPENJPEG_LIBRARY_NAME} openjp2_static)
    target_include_directories(openjp2_static PUBLIC $<INSTALL_INTERFACE:${OPENJPEG_INSTALL_INCLUDE_DIR}>)
  else()
    add_library(${OPENJPEG_LIBRARY_NAME} ${OPENJPEG_SRCS})
    set(INSTALL_LIBS ${OPENJPEG_LIBRARY_NAME})
  endif()
endif()

target_include_directories(${OPENJPEG_LIBRARY_NAME} PUBLIC $<INSTALL_INTERFACE:${OPENJPEG_INSTALL_INCLUDE_DIR}>)
if(UNIX)
  target_link_libraries(${OPENJPEG_LIBRARY_NAME} m)
endif()
set_target_properties(${OPENJPEG_LIBRARY_NAME} PROPERTIES ${OPENJPEG_LIBRARY_PROPERTIES})
if(NOT ${CMAKE_VERSION} VERSION_LESS "2.8.12")
  target_compile_options(${OPENJPEG_LIBRARY_NAME} PRIVATE ${OPENJP2_COMPILE_OPTIONS})
endif()

# Install library
install(TARGETS ${INSTALL_LIBS}
  EXPORT OpenJPEGTargets
  RUNTIME DESTINATION ${OPENJPEG_INSTALL_BIN_DIR} COMPONENT Applications
  LIBRARY DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT Libraries
  ARCHIVE DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT Libraries
)

# Install includes files
install(FILES openjpeg.h opj_stdint.h
  DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
)

if(BUILD_DOC)
# install man page of the library
install(
  FILES       ${OPENJPEG_SOURCE_DIR}/doc/man/man3/libopenjp2.3
  DESTINATION ${OPENJPEG_INSTALL_MAN_DIR}/man3)
endif()

if(BUILD_LUTS_GENERATOR)
# internal utility to generate t1_luts.h and t1_ht_luts.h (part of the jp2 lib)
# no need to install:
  add_executable(t1_generate_luts t1_generate_luts.c t1_ht_generate_luts.c)
  if(UNIX)
    target_link_libraries(t1_generate_luts m)
  endif()
endif()

# Experimental option; let's how cppcheck performs
# Implementation details:
# I could not figure out how to easily upload a file to CDash. Instead simply
# pretend cppcheck is part of the Build step. Technically cppcheck can even
# output gcc formatted error/warning report
# Another implementation detail: I could not redirect error to the error
# catching mechanism something is busted in cmake 2.8.5, I had to use the
# warning regex to catch them.
if(OPENJPEG_CPPCHECK)
  find_package(CPPCHECK REQUIRED)
  foreach(f ${OPENJPEG_SRCS})
    # cppcheck complains about too many configuration, pretend to be WIN32:
    add_custom_command(TARGET ${OPENJPEG_LIBRARY_NAME}
      COMMAND ${CPPCHECK_EXECUTABLE} -DWIN32 ${f})
  endforeach()
endif()

if(OPJ_USE_DSYMUTIL)
  if(BUILD_SHARED_LIBS)
    add_custom_command(TARGET ${OPENJPEG_LIBRARY_NAME} POST_BUILD
    COMMAND "dsymutil" "$<TARGET_SONAME_FILE:${OPENJPEG_LIBRARY_NAME}>"
    COMMENT "dsymutil $<TARGET_SONAME_FILE:${OPENJPEG_LIBRARY_NAME}>"
    DEPENDS ${OPENJPEG_LIBRARY_NAME})
  endif()
endif()

#################################################################################
# threading configuration
#################################################################################
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)

option(OPJ_USE_THREAD "Build with thread/mutex support " ON)
if(NOT OPJ_USE_THREAD)
   add_definitions( -DMUTEX_stub)
endif(NOT OPJ_USE_THREAD)

find_package(Threads QUIET)

if(OPJ_USE_THREAD AND WIN32 AND NOT Threads_FOUND )
    add_definitions( -DMUTEX_win32)
    set(Threads_FOUND YES)
endif()

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_WIN32_THREADS_INIT )
   add_definitions( -DMUTEX_win32)
endif(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_WIN32_THREADS_INIT )

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT )
   add_definitions( -DMUTEX_pthread)
endif(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT )

if(OPJ_USE_THREAD AND NOT Threads_FOUND)
  message(FATAL_ERROR "No thread library found and thread/mutex support is required by OPJ_USE_THREAD option")
endif(OPJ_USE_THREAD AND NOT Threads_FOUND)

if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
   TARGET_LINK_LIBRARIES(${OPENJPEG_LIBRARY_NAME} ${CMAKE_THREAD_LIBS_INIT})
endif(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)

if(BUILD_UNIT_TESTS AND UNIX)
    add_executable(bench_dwt bench_dwt.c)
    if(UNIX)
        target_link_libraries(bench_dwt m ${OPENJPEG_LIBRARY_NAME})
    endif()
    if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
        target_link_libraries(bench_dwt ${CMAKE_THREAD_LIBS_INIT})
    endif(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)

    add_executable(test_sparse_array test_sparse_array.c)
    if(UNIX)
        target_link_libraries(test_sparse_array m ${OPENJPEG_LIBRARY_NAME})
    endif()
    if(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
        target_link_libraries(test_sparse_array ${CMAKE_THREAD_LIBS_INIT})
    endif(OPJ_USE_THREAD AND Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
endif(BUILD_UNIT_TESTS AND UNIX)
