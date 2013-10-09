set(the_description "C Wrappers")
ocv_define_module(c BINDINGS opencv_core opencv_contrib opencv_flann opencv_ml opencv_imgproc opencv_calib3d opencv_features2d opencv_video opencv_objdetect opencv_highgui opencv_photo opencv_bioinspired opencv_legacy opencv_stitching opencv_superres opencv_ts OPTIONAL opencv_nonfree)

set(opencv_hdrs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/types.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/persistence.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/utility.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/mat.hpp"
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
    "${OPENCV_MODULE_opencv_superres_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_ts_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/"
    )
    

if(HAVE_opencv_nonfree)
  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp"
  )
endif()

set(c_generated_sources
    "${CMAKE_CURRENT_BINARY_DIR}/include/opencv2/c/opencv_generated.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/src/opencv_generated.cpp"
)

set(c_wrapper_header
    "${CMAKE_CURRENT_BINARY_DIR}/include/opencv2/c/opencv_generated.hpp"
)

set(headers
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/mat.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/point.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/rect.hpp"
)

set(sources
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/mat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/point.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/c/rect.cpp"
)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/opencv2/c/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/opencv2/c/ )
add_custom_command(
    OUTPUT ${c_generated_sources}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/genc.py" ${CMAKE_CURRENT_BINARY_DIR} ${opencv_hdrs}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.hpp ${CMAKE_CURRENT_BINARY_DIR}/include/opencv2/c/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.cpp ${CMAKE_CURRENT_BINARY_DIR}/src
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${CMAKE_BINARY_DIR}/include/
    COMMAND ${CMAKE_COMMAND} -E copy ${c_wrapper_header} ${CMAKE_BINARY_DIR}/include/opencv2/c/
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.hpp ${CMAKE_CURRENT_BINARY_DIR}/opencv_generated.cpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/genc.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/hdr_parser.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/opencv.h
    DEPENDS ${headers}
    DEPENDS ${opencv_hdrs}
)

include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR}/include 
                        "${CMAKE_SOURCE_DIR}/include"
                        ${opencv_hdr_dirs})
set_property(SOURCE ${sources} APPEND PROPERTY OBJECT_DEPENDS ${c_generated_sources})

add_dependencies(${the_module} ${c_generated_sources} ${sources} ${headers})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

install(FILES ${c_wrapper_header} DESTINATION include/opencv2/c/)
install(FILES ${headers} DESTINATION include/opencv2/c/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/opencv.h DESTINATION include/opencv2/)
