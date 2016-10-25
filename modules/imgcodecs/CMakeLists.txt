set(the_description "Image codecs")
ocv_add_module(imgcodecs opencv_imgproc WRAP java python)

# ----------------------------------------------------------------------------
#  CMake file for imgcodecs. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

ocv_clear_vars(GRFMT_LIBS)

if(HAVE_WINRT_CX AND NOT WINRT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(HAVE_PNG OR HAVE_TIFF OR HAVE_OPENEXR)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${ZLIB_LIBRARIES})
endif()

if(HAVE_JPEG)
  ocv_include_directories(${JPEG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JPEG_LIBRARIES})
endif()

if(WITH_WEBP)
  add_definitions(-DHAVE_WEBP)
  ocv_include_directories(${WEBP_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${WEBP_LIBRARIES})
endif()

if(HAVE_PNG)
  add_definitions(${PNG_DEFINITIONS})
  ocv_include_directories(${PNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${PNG_LIBRARIES})
endif()

if(HAVE_GDCM)
  ocv_include_directories(${GDCM_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${GDCM_LIBRARIES})
endif()

if(HAVE_TIFF)
  ocv_include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${TIFF_LIBRARIES})
endif()

if(HAVE_JASPER)
  ocv_include_directories(${JASPER_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JASPER_LIBRARIES})
endif()

if(HAVE_OPENEXR)
  include_directories(SYSTEM ${OPENEXR_INCLUDE_PATHS})
  list(APPEND GRFMT_LIBS ${OPENEXR_LIBRARIES})
endif()

if(HAVE_GDAL)
  include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${GDAL_LIBRARY})
endif()

file(GLOB grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/grfmt*.hpp)
file(GLOB grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/grfmt*.cpp)

list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/bitstrm.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/bitstrm.cpp)
list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/rgbe.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/rgbe.cpp)
list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/jpeg_exif.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/jpeg_exif.cpp)

source_group("Src\\grfmts" FILES ${grfmt_hdrs} ${grfmt_srcs})

set(imgcodecs_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.hpp
    )

set(imgcodecs_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/loadsave.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    )

file(GLOB imgcodecs_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
     )

if(IOS)
  list(APPEND imgcodecs_srcs ${CMAKE_CURRENT_LIST_DIR}/src/ios_conversions.mm)
  list(APPEND IMGCODECS_LIBRARIES "-framework Accelerate" "-framework CoreGraphics" "-framework CoreImage" "-framework QuartzCore" "-framework AssetsLibrary")
endif()

if(UNIX)
  #these variables are set by CHECK_MODULE macro
  foreach(P ${IMGCODECS_INCLUDE_DIRS})
    ocv_include_directories(${P})
  endforeach()

  foreach(P ${IMGCODECS_LIBRARY_DIRS})
    link_directories(${P})
  endforeach()
endif()

source_group("Src" FILES ${imgcodecs_srcs} ${imgcodecs_hdrs})
source_group("Include" FILES ${imgcodecs_ext_hdrs})
ocv_set_module_sources(HEADERS ${imgcodecs_ext_hdrs} SOURCES ${imgcodecs_srcs} ${imgcodecs_hdrs} ${grfmt_srcs} ${grfmt_hdrs})
ocv_module_include_directories()

ocv_create_module(${GRFMT_LIBS} ${IMGCODECS_LIBRARIES})

macro(ocv_imgcodecs_configure_target)
if(APPLE)
  add_apple_compiler_options(the_module)
endif()

if(BUILD_SHARED_LIBS)
  add_definitions(-DIMGCODECS_EXPORTS)
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

#stop automatic dependencies propagation for this module
set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)
endmacro()

if(NOT BUILD_opencv_world)
  ocv_imgcodecs_configure_target()
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
