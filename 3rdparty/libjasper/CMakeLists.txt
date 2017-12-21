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

if(WIN32 AND NOT MINGW)
  add_definitions(-DJAS_WIN_MSVC_BUILD)
endif(WIN32 AND NOT MINGW)

ocv_warnings_disable(CMAKE_C_FLAGS -Wno-implicit-function-declaration -Wno-uninitialized -Wmissing-prototypes
                                   -Wno-unused-but-set-parameter -Wmissing-declarations -Wunused -Wshadow
                                   -Wsign-compare -Wstrict-overflow -Wpointer-compare
                                   -Wabsolute-value  # clang on Linux
)
ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wstrict-prototypes) # clang
ocv_warnings_disable(CMAKE_C_FLAGS /wd4013 /wd4018 /wd4101 /wd4244 /wd4267 /wd4715) # vs2005

set_target_properties(${JASPER_LIBRARY}
  PROPERTIES
  OUTPUT_NAME ${JASPER_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${JASPER_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${JASPER_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${JASPER_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${JASPER_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
