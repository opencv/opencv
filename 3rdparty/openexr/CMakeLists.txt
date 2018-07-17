# ----------------------------------------------------------------------------
#  CMake file for openexr
#
# ----------------------------------------------------------------------------

project(openexr CXX)

if(UNIX)
  if(APPLE)
    set(HAVE_POSIX_SEMAPHORES 0)  # Unnamed semaphores are not supported: https://github.com/opencv/opencv/issues/9361
  else()
    include(CheckIncludeFile)
    check_include_file(semaphore.h HAVE_POSIX_SEMAPHORES)
  endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/IlmBaseConfig.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/IlmBaseConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenEXRConfig.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h" @ONLY)

set(OPENEXR_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/Half"
                          "${CMAKE_CURRENT_SOURCE_DIR}/Iex"
                          "${CMAKE_CURRENT_SOURCE_DIR}/IlmThread"
                          "${CMAKE_CURRENT_SOURCE_DIR}/Imath"
                          "${CMAKE_CURRENT_SOURCE_DIR}/IlmImf")

ocv_include_directories("${CMAKE_CURRENT_BINARY_DIR}" ${ZLIB_INCLUDE_DIRS} ${OPENEXR_INCLUDE_PATHS})

file(GLOB lib_srcs Half/half.cpp Iex/*.cpp IlmThread/*.cpp Imath/*.cpp IlmImf/*.cpp)
file(GLOB lib_hdrs Half/*.h Iex/Iex*.h IlmThread/IlmThread*.h Imath/Imath*.h IlmImf/*.h)
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/IlmBaseConfig.h" "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h")

ocv_list_filterout(lib_srcs IlmImf/b44ExpLogTable.cpp)

if(WIN32)
  ocv_list_filterout(lib_srcs Posix.*cpp)
else()
  ocv_list_filterout(lib_srcs Win32.cpp)
endif()

source_group("Include" FILES ${lib_hdrs} )
source_group("Src" FILES ${lib_srcs})

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Wunused -Wsign-compare -Wundef -Wmissing-declarations -Wuninitialized -Wswitch -Wparentheses -Warray-bounds -Wextra
                                     -Wdeprecated-declarations -Wmisleading-indentation -Wdeprecated
                                     -Wsuggest-override -Winconsistent-missing-override
                                     -Wimplicit-fallthrough
)
if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wclass-memaccess)
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4018 /wd4099 /wd4100 /wd4101 /wd4127 /wd4189 /wd4245 /wd4305 /wd4389 /wd4512 /wd4701 /wd4702 /wd4706 /wd4800) # vs2005
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4334) # vs2005 Win64
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244) # vs2008
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267) # vs2008 Win64
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4456) # vs2015

if(MSVC AND CV_ICC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qrestrict")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qrestrict")
endif()

add_library(IlmImf STATIC ${lib_hdrs} ${lib_srcs})
target_link_libraries(IlmImf ${ZLIB_LIBRARIES})

set_target_properties(IlmImf
    PROPERTIES
    OUTPUT_NAME "IlmImf"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME "IlmImf"
    COMPILE_PDB_NAME_DEBUG "IlmImf${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(IlmImf PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(IlmImf EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(openexr LICENSE AUTHORS.ilmbase AUTHORS.openexr fix_msvc2013_errors.patch)

set(OPENEXR_INCLUDE_PATHS ${OPENEXR_INCLUDE_PATHS} PARENT_SCOPE)
set(OPENEXR_VERSION "1.7.1" PARENT_SCOPE)
