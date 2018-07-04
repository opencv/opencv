# ----------------------------------------------------------------------------
#  CMake file for libwebp. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${WEBP_LIBRARY})

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(ANDROID)
  ocv_include_directories(${CPUFEATURES_INCLUDE_DIRS})
endif()

file(GLOB lib_srcs src/dec/*.c src/demux/*.c src/dsp/*.c src/enc/*.c src/mux/*.c src/utils/*.c src/webp/*.c)
file(GLOB lib_hdrs src/dec/*.h src/demux/*.h src/dsp/*.h src/enc/*.h src/mux/*.h src/utils/*.h src/webp/*.h)

# FIXIT
if(ANDROID AND ARMEABI_V7A AND NOT NEON)
  foreach(file ${lib_srcs})
    if("${file}" MATCHES "_neon.c")
      set_source_files_properties("${file}" COMPILE_FLAGS "-mfpu=neon")
    endif()
  endforeach()
endif()



# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_definitions(-DWEBP_USE_THREAD)

add_library(${WEBP_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})
if(ANDROID)
  target_link_libraries(${WEBP_LIBRARY} ${CPUFEATURES_LIBRARIES})
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable -Wunused-function -Wshadow -Wmaybe-uninitialized
    -Wmissing-prototypes  # clang
    -Wmissing-declarations # gcc
    -Wimplicit-fallthrough
)
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

