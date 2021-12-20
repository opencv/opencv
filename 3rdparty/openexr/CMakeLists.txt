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



set(Imath_VERSION_MAJOR "3")
set(Imath_VERSION_MINOR "1")
set(Imath_VERSION_PATCH "3")

set(IMATH_VERSION "${Imath_VERSION_MAJOR}.${Imath_VERSION_MINOR}.${Imath_VERSION_PATCH}")
set(IMATH_VERSION_API ${Imath_VERSION_MAJOR}_${Imath_VERSION_MINOR})

set(OpenEXR_VERSION_MAJOR "3")
set(OpenEXR_VERSION_MINOR "1")
set(OpenEXR_VERSION_PATCH "3")

set(OPENEXR_VERSION "${OpenEXR_VERSION_MAJOR}.${OpenEXR_VERSION_MINOR}.${OpenEXR_VERSION_PATCH}")
set(OPENEXR_VERSION_API ${OpenEXR_VERSION_MAJOR}_${OpenEXR_VERSION_MINOR})

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

set(ILMTHREAD_THREADING_ENABLED 1)

set(IMATH_VERSION_API "opencv")
set(IMATH_INTERNAL_NAMESPACE_CUSTOM 1)
set(IMATH_INTERNAL_NAMESPACE "Imath_${IMATH_VERSION_API}")
set(IMATH_NAMESPACE_CUSTOM 0)
set(IMATH_NAMESPACE "Imath")

set(OPENEXR_VERSION_API "opencv")
set(OPENEXR_INTERNAL_IMF_NAMESPACE_CUSTOM 1)
set(OPENEXR_INTERNAL_IMF_NAMESPACE "Imf_${OPENEXR_VERSION_API}")
set(OPENEXR_IMF_NAMESPACE_CUSTOM 0)
set(OPENEXR_IMF_NAMESPACE "Imf")

set(IEX_INTERNAL_NAMESPACE "Iex_${OPENEXR_VERSION_API}")
set(ILMTHREAD_INTERNAL_NAMESPACE "IlmThread_${OPENEXR_VERSION_API}")

set(IEX_NAMESPACE "Iex")
set(ILMTHREAD_NAMESPACE "IlmThread")

set(IMATH_VERSION_STRING "\"${IMATH_VERSION}\"" )
set(IMATH_PACKAGE_NAME "Imath ${IMATH_VERSION}" )

set(OPENEXR_VERSION_STRING "\"${OPENEXR_VERSION}\"" )
set(OPENEXR_PACKAGE_NAME "OpenEXR ${OPENEXR_VERSION}" )


configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ImathConfig.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/ImathConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/IexConfig.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/IexConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/IexConfigInternal.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/IexConfigInternal.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/IlmThreadConfig.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/IlmThreadConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenEXRConfig.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenEXRConfigInternal.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfigInternal.h" @ONLY)

set(OPENEXR_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/Iex"
                          "${CMAKE_CURRENT_SOURCE_DIR}/IlmThread"
                          "${CMAKE_CURRENT_SOURCE_DIR}/Imath"
                          "${CMAKE_CURRENT_SOURCE_DIR}/OpenEXR"
                          "${CMAKE_CURRENT_BINARY_DIR}")

ocv_include_directories("${CMAKE_CURRENT_BINARY_DIR}" ${ZLIB_INCLUDE_DIRS} ${OPENEXR_INCLUDE_PATHS})

file(GLOB lib_srcs Iex/*.cpp IlmThread/*.cpp Imath/*.cpp OpenEXR/*.cpp)
list(REMOVE_ITEM lib_srcs "${CMAKE_CURRENT_SOURCE_DIR}/OpenEXR/b44ExpLogTable.cpp")
list(REMOVE_ITEM lib_srcs "${CMAKE_CURRENT_SOURCE_DIR}/OpenEXR/dwaLookups.cpp")

file(GLOB lib_hdrs Iex/Iex*.h IlmThread/IlmThread*.h Imath/Imath*.h OpenEXR/*.h)
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/IexConfig.h")
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/IexConfigInternal.h")
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/IlmThreadConfig.h")
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ImathConfig.h")
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfig.h")
list(APPEND lib_hdrs "${CMAKE_CURRENT_BINARY_DIR}/OpenEXRConfigInternal.h")

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
                                     -Wimplicit-const-int-float-conversion  # clang
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

add_library(IlmImf STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_hdrs} ${lib_srcs})
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
  ocv_install_target(IlmImf EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(openexr LICENSE.imath LICENSE.openexr AUTHORS.imath AUTHORS.openexr)

set(OPENEXR_INCLUDE_PATHS ${OPENEXR_INCLUDE_PATHS} PARENT_SCOPE)
