set(the_description "C Wrappers")
ocv_define_module(c BINDINGS opencv_core opencv_contrib opencv_flann opencv_ml opencv_imgproc opencv_calib3d opencv_features2d opencv_video opencv_objdetect opencv_highgui opencv_photo opencv_bioinspired opencv_legacy opencv_stitching opencv_superres opencv_ts OPTIONAL opencv_nonfree)

set(opencv_hdrs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/types.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/persistence.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/utility.hpp"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/opencv2/flann/miniflann.hpp"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/opencv2/imgproc.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/background_segm.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/opencv2/photo.hpp"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/opencv2/highgui.hpp"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/opencv2/ml.hpp"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/opencv2/features2d.hpp"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/opencv2/calib3d.hpp"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/opencv2/objdetect.hpp"
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/opencv2/contrib.hpp"
    "${OPENCV_MODULE_opencv_bioinspired_LOCATION}/include/opencv2/bioinspired.hpp"
    "${OPENCV_MODULE_opencv_stitching_LOCATION}/include/opencv2/stitching.hpp"
    "${OPENCV_MODULE_opencv_superres_LOCATION}/include/opencv2/superres.hpp"
    "${OPENCV_MODULE_opencv_ts_LOCATION}/include/opencv2/ts.hpp"
    )

set(opencv_hdr_dirs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_bioinspired_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_stitching_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_ts_LOCATION}/include/"
    )
    

if(HAVE_opencv_nonfree)
  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp")
endif()

set(c_generated_sources
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.cpp")

set(c_wrapper_headers
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mat.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/point.hpp"
    )

add_custom_command(
    OUTPUT ${c_generated_sources}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/genc.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/genc.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/hdr_parser.py
    DEPENDS ${opencv_hdrs}
)

include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR} 
                        "${CMAKE_SOURCE_DIR}/include"
                        ${opencv_hdr_dirs})
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/mat.cpp" APPEND PROPERTY OBJECT_DEPENDS ${c_generated_sources})
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/point.cpp" APPEND PROPERTY OBJECT_DEPENDS ${c_generated_sources})

add_dependencies(${the_module} ${c_generated_sources})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

install(FILES ${c_wrapper_headers} DESTINATION include/opencv2/c/)
install(FILES "src/opencv.h" DESTINATION include/opencv2)
