# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------
project(opencv_python)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
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
    "${CMAKE_SOURCE_DIR}/modules/python/src/opencv_extra_api.hpp")

set(generated_hdrs
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_func_tab.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
        "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_const_reg.h")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_hdrs "src/*.h")

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated0.i
   COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/gen.py" "${CMAKE_CURRENT_SOURCE_DIR}/src"
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/api
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/defs
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gen.py
   )

add_custom_command(
  OUTPUT ${generated_hdrs}
  COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gen2.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/hdr_parser.py
  DEPENDS ${opencv_hdrs}
  )

set(the_target "opencv_python")
add_library(${the_target} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs} ${CMAKE_CURRENT_BINARY_DIR}/generated0.i src/opencv2x.h src/opencv_extra_api.hpp ${generated_hdrs})
target_link_libraries(${the_target} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib)
    
set_target_properties(${the_target} PROPERTIES PREFIX "")
set_target_properties(${the_target} PROPERTIES OUTPUT_NAME "cv")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('SO')"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${the_target} PROPERTIES SUFFIX ${CVPY_SUFFIX})

set(cvpy_files cv${CVPY_SUFFIX})

if(WIN32)
    install(TARGETS ${the_target}
            RUNTIME DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            LIBRARY DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            ARCHIVE DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            )
else()
    #install(FILES ${LIBRARY_OUTPUT_PATH}/cv${CVPY_SUFFIX} DESTINATION ${PYTHON_PACKAGES_PATH})
    install(TARGETS ${the_target} 
            RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
            LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
            ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main)
endif()
