# ----------------------------------------------------------------------------
#  CMake file for Intel ITT API. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if(NOT ITT_LIBRARY)
  set(ITT_LIBRARY "ittnotify")
endif()
project(${ITT_LIBRARY} C)

if(NOT WIN32)
  include(CheckLibraryExists)
  if(COMMAND CHECK_LIBRARY_EXISTS)
    CHECK_LIBRARY_EXISTS(dl dlerror "" HAVE_DL_LIBRARY)
  endif()
endif()

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ITT_INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(ITT_PUBLIC_HDRS
    include/ittnotify.h
    include/jitprofiling.h
    include/libittnotify.h
    include/llvm_jit_event_listener.hpp
)
set(ITT_PRIVATE_HDRS
    src/ittnotify/disable_warnings.h
    src/ittnotify/ittnotify_config.h
    src/ittnotify/ittnotify_static.h
    src/ittnotify/ittnotify_types.h
)
set(ITT_SRCS
    src/ittnotify/ittnotify_static.c
    src/ittnotify/jitprofiling.c
)

add_library(${ITT_LIBRARY} STATIC ${ITT_SRCS} ${ITT_PUBLIC_HDRS} ${ITT_PRIVATE_HDRS})

if(NOT WIN32)
  if(HAVE_DL_LIBRARY)
    target_link_libraries(${ITT_LIBRARY} dl)
  endif()
endif()

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

set_target_properties(${ITT_LIBRARY} PROPERTIES
        OUTPUT_NAME ${ITT_LIBRARY}
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        COMPILE_PDB_NAME ${ITT_LIBRARY}
        COMPILE_PDB_NAME_DEBUG "${ITT_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    )

ocv_warnings_disable(CMAKE_C_FLAGS -Wundef -Wsign-compare)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${ITT_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${ITT_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
