# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------

if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  ocv_module_disable(python)
endif()
if(ANDROID OR IOS OR NOT PYTHONLIBS_FOUND OR NOT PYTHON_USE_NUMPY)
  ocv_module_disable(python)
endif()

set(the_description "The python bindings")
ocv_add_module(python BINDINGS opencv_core opencv_flann opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_photo opencv_nonfree opencv_objdetect opencv_legacy opencv_contrib opencv_stitching)

ocv_include_directories(${PYTHON_INCLUDE_PATH})
ocv_include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src2"
    "${OpenCV_SOURCE_DIR}/modules/core/include"
    "${OpenCV_SOURCE_DIR}/modules/flann/include"
    "${OpenCV_SOURCE_DIR}/modules/imgproc/include"
    "${OpenCV_SOURCE_DIR}/modules/video/include"
    "${OpenCV_SOURCE_DIR}/modules/photo/include"
    "${OpenCV_SOURCE_DIR}/modules/highgui/include"
    "${OpenCV_SOURCE_DIR}/modules/ml/include"
    "${OpenCV_SOURCE_DIR}/modules/features2d/include"
    "${OpenCV_SOURCE_DIR}/modules/flann/include"
    "${OpenCV_SOURCE_DIR}/modules/calib3d/include"
    "${OpenCV_SOURCE_DIR}/modules/objdetect/include"
    "${OpenCV_SOURCE_DIR}/modules/nonfree/include"
    "${OpenCV_SOURCE_DIR}/modules/legacy/include"
    "${OpenCV_SOURCE_DIR}/modules/contrib/include"
    "${OpenCV_SOURCE_DIR}/modules/stitching/include"
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(opencv_hdrs "${OpenCV_SOURCE_DIR}/modules/core/include/opencv2/core/core.hpp"
    "${OpenCV_SOURCE_DIR}/modules/flann/include/opencv2/flann/miniflann.hpp"
    "${OpenCV_SOURCE_DIR}/modules/imgproc/include/opencv2/imgproc/imgproc.hpp"
    "${OpenCV_SOURCE_DIR}/modules/video/include/opencv2/video/background_segm.hpp"
    "${OpenCV_SOURCE_DIR}/modules/video/include/opencv2/video/tracking.hpp"
    "${OpenCV_SOURCE_DIR}/modules/photo/include/opencv2/photo/photo.hpp"
    "${OpenCV_SOURCE_DIR}/modules/highgui/include/opencv2/highgui/highgui.hpp"
    "${OpenCV_SOURCE_DIR}/modules/ml/include/opencv2/ml/ml.hpp"
    "${OpenCV_SOURCE_DIR}/modules/features2d/include/opencv2/features2d/features2d.hpp"
    "${OpenCV_SOURCE_DIR}/modules/nonfree/include/opencv2/nonfree/features2d.hpp"
    "${OpenCV_SOURCE_DIR}/modules/calib3d/include/opencv2/calib3d/calib3d.hpp"
    "${OpenCV_SOURCE_DIR}/modules/objdetect/include/opencv2/objdetect/objdetect.hpp"
    "${OpenCV_SOURCE_DIR}/modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_func_tab.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_const_reg.h")


add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated0.i
   COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src2/gen.py" "${CMAKE_CURRENT_SOURCE_DIR}/src2"
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/api
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/defs
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/gen.py
   )

add_custom_command(
     OUTPUT ${cv2_generated_hdrs}
     COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/gen2.py
     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src2/hdr_parser.py
     DEPENDS ${opencv_hdrs})

set(cv2_target "opencv_python")
add_library(${cv2_target} SHARED src2/cv2.cpp ${CMAKE_CURRENT_BINARY_DIR}/generated0.i ${cv2_generated_hdrs} src2/cv2.cv.hpp)
target_link_libraries(${cv2_target} ${PYTHON_LIBRARIES} opencv_core opencv_flann opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib opencv_photo opencv_stitching)
    
set_target_properties(${cv2_target} PROPERTIES PREFIX "")
set_target_properties(${cv2_target} PROPERTIES OUTPUT_NAME "cv2")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('SO')"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${cv2_target} PROPERTIES SUFFIX ${CVPY_SUFFIX})

if (MSVC AND NOT BUILD_SHARED_LIBS)
    set_target_properties(${cv2_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

set(cvpymodules ${cv2_target})

if(WIN32)
  set(CVPY_PATH "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages")
else()
  set(CVPY_PATH ${PYTHON_PACKAGES_PATH})
endif()

install(TARGETS ${cvpymodules}
      RUNTIME DESTINATION ${CVPY_PATH} COMPONENT main
      LIBRARY DESTINATION ${CVPY_PATH} COMPONENT main
      ARCHIVE DESTINATION ${CVPY_PATH} COMPONENT main
      )
install(FILES src2/cv.py DESTINATION ${CVPY_PATH} COMPONENT main)
