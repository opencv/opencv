# ----------------------------------------------------------------------------
#  CMake file for libraw. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${RAW_LIBRARY})

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB lib_srcs src/*.cpp src/*/*.cpp )
file(GLOB lib_hdrs *.h)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(${RAW_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})
#ocv_warnings_disable(CMAKE_CPP_FLAGS /wd4146 /wd4703 /wd4244 /wd4244 /wd4018 /wd4101 /wd4267) 
ocv_warnings_disable(CMAKE_CXX_FLAGS -Warray-bounds -Wimplicit-fallthrough
                                     -Wmisleading-indentation -Wmissing-field-initializers
									                   -Wformat-overflow -Wformat-truncation  -Wmissing-declarations
                                     -Wrestrict
									                   -Wsign-compare -Wshift-negative-value -Wstringop-truncation
                                     -Wsuggest-override  -Wtype-limits
									                   -Wunused-parameter -Wunused-result -Wunused-variable 
                                     /wd4146 /wd4703 /wd4244 /wd4244 /wd4018 /wd4101 /wd4267 /wd4800
) 

add_definitions(-DLIBRAW_NOTHREADS -DLIBRAW_BUILDLIB -DLIBRAW_NODLL)

set_target_properties(${RAW_LIBRARY}
  PROPERTIES OUTPUT_NAME ${RAW_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${RAW_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${RAW_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${RAW_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${RAW_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(libraw COPYRIGHT)
