set(the_description "Haskell Bindings")

if(ANDROID OR IOS OR NOT GHC_EXECUTABLE OR NOT CABAL_EXECUTABLE)
  ocv_module_disable(haskell)
endif()

ocv_define_module(haskell BINDINGS opencv_c)

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

if(HAVE_opencv_nonfree)
  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp")
endif()

set(haskell_generated_sources
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Types.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Funcs.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Consts.hsc"
)

set(haskell_sources
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Mat.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Point.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/Setup.hs"
    "${CMAKE_CURRENT_BINARY_DIR}/opencv.cabal"
)

# Copy necessary files over to the build directory.
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Mat.hsc"
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy "${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/Mat.hsc" ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Point.hsc"
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy "${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/Point.hsc" ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Setup.hs"
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Setup.hs" ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/opencv.cabal"
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Setup.hs" ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/
)

# generate ffi wrappers
add_custom_command(
    OUTPUT ${haskell_generated_sources}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/genhsc.py" "${CMAKE_CURRENT_BINARY_DIR}/OpenCV" ${opencv_hdrs}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/genhsc.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/hdr_parser.py
    DEPENDS ${opencv_hdrs}
)

add_dependencies(${the_module} ${haskell_generated_sources} ${haskell_sources})

install(CODE "execute_process(COMMAND \"${CABAL_EXECUTABLE} install --global --only-dependencies\")")
install(CODE "execute_process(COMMAND \"${CABAL_EXECUTABLE} install --global --with-gcc=${CMAKE_CXX_COMPILER} --extra-include-dirs=${OPENCV_MODULE_opencv_c_LOCATION} \")")
