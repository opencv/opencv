# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------
project(opencv_python)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src1"
    "${CMAKE_CURRENT_SOURCE_DIR}/src2"
    "${CMAKE_SOURCE_DIR}/modules/core/include"
    "${CMAKE_SOURCE_DIR}/modules/imgproc/include"
    "${CMAKE_SOURCE_DIR}/modules/video/include"
    "${CMAKE_SOURCE_DIR}/modules/highgui/include"
    "${CMAKE_SOURCE_DIR}/modules/ml/include"
    "${CMAKE_SOURCE_DIR}/modules/features2d/include"
    "${CMAKE_SOURCE_DIR}/modules/flann/include"
    "${CMAKE_SOURCE_DIR}/modules/calib3d/include"
    "${CMAKE_SOURCE_DIR}/modules/objdetect/include"
    "${CMAKE_SOURCE_DIR}/modules/legacy/include"
    "${CMAKE_SOURCE_DIR}/modules/contrib/include"
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(opencv_hdrs "${CMAKE_SOURCE_DIR}/modules/core/include/opencv2/core/core.hpp"
    "${CMAKE_SOURCE_DIR}/modules/imgproc/include/opencv2/imgproc/imgproc.hpp"
    "${CMAKE_SOURCE_DIR}/modules/video/include/opencv2/video/background_segm.hpp"
    "${CMAKE_SOURCE_DIR}/modules/video/include/opencv2/video/tracking.hpp"
    "${CMAKE_SOURCE_DIR}/modules/highgui/include/opencv2/highgui/highgui.hpp"
    "${CMAKE_SOURCE_DIR}/modules/ml/include/opencv2/ml/ml.hpp"
    "${CMAKE_SOURCE_DIR}/modules/features2d/include/opencv2/features2d/features2d.hpp"
    "${CMAKE_SOURCE_DIR}/modules/calib3d/include/opencv2/calib3d/calib3d.hpp"
    "${CMAKE_SOURCE_DIR}/modules/objdetect/include/opencv2/objdetect/objdetect.hpp"
    "${CMAKE_SOURCE_DIR}/modules/python/src2/opencv_extra_api.hpp")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated0.i
   COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src1/gen.py" "${CMAKE_CURRENT_SOURCE_DIR}/src1"
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src1/api
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src1/defs
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src1/gen.py
   )


set(cv_target "opencv_python")
add_library(${cv_target} SHARED src1/cv.cpp ${CMAKE_CURRENT_BINARY_DIR}/generated0.i)
target_link_libraries(${cv_target} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib)
    
set_target_properties(${cv_target} PROPERTIES PREFIX "")
set_target_properties(${cv_target} PROPERTIES OUTPUT_NAME "cv")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('SO')"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${cv_target} PROPERTIES SUFFIX ${CVPY_SUFFIX})

if (MSVC AND NOT BUILD_SHARED_LIBS)
    set_target_properties(${cv_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

set(cvpymodules ${cv_target})

if(PYTHON_USE_NUMPY)

set(cv2_generated_hdrs
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

set(cv2_target "opencv2_python")
add_library(${cv2_target} SHARED src2/cv2.cpp src2/opencv_extra_api.hpp ${cv2_generated_hdrs})
target_link_libraries(${cv2_target} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib)

set_target_properties(${cv2_target} PROPERTIES PREFIX "")
set_target_properties(${cv2_target} PROPERTIES OUTPUT_NAME "cv2")
set_target_properties(${cv2_target} PROPERTIES SUFFIX ${CVPY_SUFFIX})

if (MSVC AND NOT BUILD_SHARED_LIBS)
    set_target_properties(${cv2_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

set(cvpymodules ${cvpymodules} ${cv2_target})

endif()

if(WIN32)
  install(TARGETS ${cvpymodules}
          RUNTIME DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          LIBRARY DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          ARCHIVE DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          )
else()
  #install(FILES ${LIBRARY_OUTPUT_PATH}/cv${CVPY_SUFFIX} DESTINATION ${PYTHON_PACKAGES_PATH})
  install(TARGETS ${cvpymodules} 
          RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
          LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
          ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main)
endif()
