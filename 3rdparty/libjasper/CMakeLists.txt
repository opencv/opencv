# ----------------------------------------------------------------------------
#  CMake file for libjasper. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${JASPER_LIBRARY})


add_definitions(-DEXCLUDE_MIF_SUPPORT -DEXCLUDE_PNM_SUPPORT -DEXCLUDE_BMP_SUPPORT -DEXCLUDE_RAS_SUPPORT  -DEXCLUDE_JPG_SUPPORT -DEXCLUDE_PGX_SUPPORT)

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)
file(GLOB lib_ext_hdrs jasper/*.h)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(${JASPER_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

if(MSVC)
  add_definitions(-DJAS_WIN_MSVC_BUILD)
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wno-implicit-function-declaration -Wno-uninitialized -Wmissing-prototypes -Wmissing-declarations -Wunused -Wshadow
                                   /wd4013 /wd4018 /wd4715 /wd4244 /wd4101 /wd4267)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

set_target_properties(${JASPER_LIBRARY}
  PROPERTIES
  OUTPUT_NAME ${JASPER_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}"
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${JASPER_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS ${JASPER_LIBRARY} ARCHIVE DESTINATION share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
endif()

