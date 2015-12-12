# ----------------------------------------------------------------------------
#  CMake file for libwebp. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${WEBP_LIBRARY})

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cpu-features")

file(GLOB lib_srcs dec/*.c dsp/*.c enc/*.c mux/*.c utils/*.c webp/*.c)
file(GLOB lib_hdrs dec/*.h dsp/*.h enc/*.h mux/*.h utils/*.h webp/*.h)

# FIXIT
if(ANDROID AND ARMEABI_V7A AND NOT NEON)
  foreach(file ${lib_srcs})
    if("${file}" MATCHES "_neon.c")
      set_source_files_properties("${file}" COMPILE_FLAGS "-mfpu=neon")
    endif()
  endforeach()
endif()

file(GLOB cpuf_s cpu-features/*.c)
file(GLOB cpuf_h cpu-features/*.h)

if(ANDROID)
  set(lib_srcs ${lib_srcs} ${cpuf_s})
  set(lib_hdrs ${lib_hdrs} ${cpuf_h})
endif()

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_definitions(-DWEBP_USE_THREAD)

add_library(${WEBP_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable -Wunused-function -Wshadow -Wmaybe-uninitialized)
ocv_warnings_disable(CMAKE_C_FLAGS /wd4244 /wd4267) # vs2005

set_target_properties(${WEBP_LIBRARY}
  PROPERTIES OUTPUT_NAME ${WEBP_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${WEBP_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${WEBP_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${WEBP_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${WEBP_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

