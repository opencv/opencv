# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------

if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  ocv_module_disable(python)
endif()

if(ANDROID OR IOS OR NOT PYTHONLIBS_FOUND OR NOT PYTHON_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python)
endif()

set(the_description "The python bindings")
ocv_add_module(python BINDINGS opencv_core opencv_flann opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_imgcodecs opencv_videoio opencv_highgui opencv_calib3d opencv_photo opencv_objdetect OPTIONAL opencv_nonfree)

ocv_module_include_directories(
    "${PYTHON_INCLUDE_PATH}"
    ${PYTHON_NUMPY_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src2"
    )

set(opencv_hdrs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/types.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/persistence.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/utility.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/ocl.hpp"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/opencv2/flann/miniflann.hpp"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/opencv2/imgproc.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/background_segm.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/opencv2/photo.hpp"
    "${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include/opencv2/imgcodecs.hpp"
    "${OPENCV_MODULE_opencv_videoio_LOCATION}/include/opencv2/videoio.hpp"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/opencv2/highgui.hpp"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/opencv2/ml.hpp"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/opencv2/features2d.hpp"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/opencv2/calib3d.hpp"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/opencv2/objdetect.hpp"
    )

if(HAVE_opencv_nonfree)
  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp")
endif()

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_include.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_func_tab.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_const_reg.h")

add_custom_command(
   OUTPUT ${cv2_generated_hdrs}
   COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/hdr_parser.py
   DEPENDS ${opencv_hdrs})

add_library(${the_module} SHARED src2/cv2.cpp ${cv2_generated_hdrs})
set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS OPENCV_NOSTL)

if(PYTHON_DEBUG_LIBRARIES AND NOT PYTHON_LIBRARIES MATCHES "optimized.*debug")
  target_link_libraries(${the_module} debug ${PYTHON_DEBUG_LIBRARIES} optimized ${PYTHON_LIBRARIES})
else()
  target_link_libraries(${the_module} ${PYTHON_LIBRARIES})
endif()
target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${the_module} PROPERTIES
                      PREFIX ""
                      OUTPUT_NAME cv2
                      SUFFIX ${CVPY_SUFFIX})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(MSVC)
    add_definitions(-DCVAPI_EXPORTS)
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

if(MSVC AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100") #unreferenced formal parameter
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4127") #conditional expression is constant
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4505") #unreferenced local function has been removed
  string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

if(MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
  set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else()
  set(PYTHON_INSTALL_CONFIGURATIONS "")
endif()

if(WIN32)
  set(PYTHON_INSTALL_ARCHIVE "")
else()
  set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif()

if(NOT INSTALL_CREATE_DISTRIB)
  install(TARGETS ${the_module}
          ${PYTHON_INSTALL_CONFIGURATIONS}
          RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
          LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
          ${PYTHON_INSTALL_ARCHIVE}
          )
else()
  if(DEFINED PYTHON_VERSION_MAJOR)
    set(__ver "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  else()
    set(__ver "unknown")
  endif()
  install(TARGETS ${the_module}
          CONFIGURATIONS Release
          RUNTIME DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
          LIBRARY DESTINATION python/${__ver}/${OpenCV_ARCH} COMPONENT python
          )
endif()
