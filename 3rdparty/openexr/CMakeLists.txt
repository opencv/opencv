# ----------------------------------------------------------------------------
#  CMake file for openexr
#
# ----------------------------------------------------------------------------

project(openexr CXX)


if(NOT HAVE_CXX11)
  ocv_check_compiler_flag(CXX "-std=c++11" HAVE_STD_CXX11 "${OpenCV_SOURCE_DIR}/cmake/checks/cxx11.cpp")
  if(HAVE_STD_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    if(BUILD_OPENEXR)
      message(WARNING "OpenCV: builtin OpenEXR requires C++11 support. OpenEXR is disabled.")
    endif()
    return()
  endif()
endif()



set(ILMBASE_VERSION_MAJOR "2")
set(ILMBASE_VERSION_MINOR "3")
set(ILMBASE_VERSION_PATCH "0")

set(ILMBASE_VERSION "${ILMBASE_VERSION_MAJOR}.${ILMBASE_VERSION_MINOR}.${ILMBASE_VERSION_PATCH}")
set(ILMBASE_VERSION_API ${ILMBASE_VERSION_MAJOR}_${ILMBASE_VERSION_MINOR})

set(OPENEXR_VERSION_MAJOR "2")
set(OPENEXR_VERSION_MINOR "3")
set(OPENEXR_VERSION_PATCH "0")

set(OPENEXR_VERSION "${OPENEXR_VERSION_MAJOR}.${OPENEXR_VERSION_MINOR}.${OPENEXR_VERSION_PATCH}")
set(OPENEXR_VERSION_API ${OPENEXR_VERSION_MAJOR}_${OPENEXR_VERSION_MINOR})

set(OPENEXR_VERSION "${OPENEXR_VERSION}" PARENT_SCOPE)

if(WIN32)
  set(HAVE_COMPLETE_IOMANIP 1)
  set(OPENEXR_IMF_HAVE_COMPLETE_IOMANIP 1)
  set(PLATFORM_WINDOWS 1)
elseif(APPLE)
  set(HAVE_POSIX_SEMAPHORES 0)  # Unnamed semaphores are not supported: https://github.com/opencv/opencv/issues/9361
  if(DARWIN)
    set(OPENEXR_IMF_HAVE_DARWIN 1)
  endif()
elseif(UNIX)
  include(CheckIncludeFile)
  check_include_file(semaphore.h HAVE_POSIX_SEMAPHORES)
endif()

set(ILMBASE_VERSION_API "opencv")
set(ILMBASE_INTERNAL_NAMESPACE_CUSTOM 1)
set(IMATH_INTERNAL_NAMESPACE "Imath_${ILMBASE_VERSION_API}")
set(IEX_INTERNAL_NAMESPACE "Iex_${ILMBASE_VERSION_API}")
set(ILMTHREAD_INTERNAL_NAMESPACE "IlmThread_${ILMBASE_VERSION_API}")

set(ILMBASE_NAMESPACE_CUSTOM 0)
set(IMATH_NAMESPACE "Imath")
set(IEX_NAMESPACE "Iex")
set(ILMTHREAD_NAMESPACE "IlmThread")
set(ILMBASE_VERSION_STRING "\"${ILMBASE_VERSION}\"" )
set(ILMBASE_PACKAGE_STRING "\"IlmBase ${ILMBASE_VERSION}\"" )


set(OPENEXR_VERSION_API "opencv")
set(OPENEXR_IMF_INTERNAL_NAMESPACE_CUSTOM 1)
set(OPENEXR_IMF_INTERNAL_NAMESPACE "Imf_${ILMBASE_VERSION_API}")
set(OPENEXR_IMF_NAMESPACE_CUSTOM 0)
set(OPENEXR_IMF_NAMESPACE "Imf")

set(OPENEXR_VERSION_STRING "\"${OPENEXR_VERSION}\"" )
set(OPENEXR_PACKAGE_STRING "\"OpenEXR ${OPENEXR_VERSION}\"" )


configure_file("${CMAKE_CURRENT_SOURCE_DIR}/IlmBaseConfig.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/IlmBaseConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenEXRConfig.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h" @ONLY)

set(OPENEXR_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/Half"
                          "${CMAKE_CURRENT_SOURCE_DIR}/Iex"
                          "${CMAKE_CURRENT_SOURCE_DIR}/IlmThread"
                          "${CMAKE_CURRENT_SOURCE_DIR}/Imath"
                          "${CMAKE_CURRENT_SOURCE_DIR}/IlmImf"
                          "${CMAKE_CURRENT_BINARY_DIR}")

ocv_include_directories("${CMAKE_CURRENT_BINARY_DIR}" ${ZLIB_INCLUDE_DIRS} ${OPENEXR_INCLUDE_PATHS})

file(GLOB lib_srcs Half/half.cpp Iex/*.cpp IlmThread/*.cpp Imath/*.cpp IlmImf/*.cpp)
file(GLOB lib_hdrs Half/*.h Iex/Iex*.h IlmThread/IlmThread*.h Imath/Imath*.h IlmImf/*.h)
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/IlmBaseConfig.h" "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h")

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
                                     -Wtautological-compare  # clang
                                     -Wmissing-prototypes  # gcc/clang
                                     -Wreorder
                                     -Wunused-result
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

ocv_install_3rdparty_licenses(openexr LICENSE AUTHORS.ilmbase AUTHORS.openexr)

set(OPENEXR_INCLUDE_PATHS ${OPENEXR_INCLUDE_PATHS} PARENT_SCOPE)
