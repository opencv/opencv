# ----------------------------------------------------------------------------
# CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
 
SET(opencv "../")
include_directories(
${opencv}include
${opencv}modules/objdetect/include
${opencv}modules/imgproc/include
${opencv}modules/highgui/include
${opencv}modules/core/include
${opencv}modules/ml/include/
${opencv}modules/objdetect/
${opencv} ${opencv} )
 
 
 
add_definitions(-D__OPENCV_BUILD)
# modules used : core highgui objdetect ml imgproc
 
set_source_files_properties(face.c PROPERTIES LANGUAGE CXX)
 
ADD_LIBRARY( opencv-face STATIC
${opencv}/modules/core/opencl_kernels.cpp
${opencv}/modules/highgui/src/bitstrm.cpp
${opencv}/modules/highgui/src/grfmt_base.cpp
${opencv}/modules/highgui/src/grfmt_bmp.cpp
${opencv}/modules/highgui/src/grfmt_exr.cpp
${opencv}/modules/highgui/src/grfmt_hdr.cpp
${opencv}/modules/highgui/src/grfmt_jpeg.cpp
${opencv}/modules/highgui/src/grfmt_jpeg2000.cpp
${opencv}/modules/highgui/src/grfmt_png.cpp
${opencv}/modules/highgui/src/grfmt_pxm.cpp
${opencv}/modules/highgui/src/grfmt_sunras.cpp
${opencv}/modules/highgui/src/grfmt_tiff.cpp
${opencv}/modules/highgui/src/grfmt_webp.cpp
${opencv}/modules/highgui/src/loadsave.cpp
${opencv}/modules/highgui/src/rgbe.cpp
${opencv}/modules/highgui/src/utils.cpp
${opencv}/modules/imgproc/src/accum.cpp
${opencv}/modules/imgproc/src/approx.cpp
${opencv}/modules/imgproc/src/blend.cpp
${opencv}/modules/imgproc/src/canny.cpp
${opencv}/modules/imgproc/src/clahe.cpp
${opencv}/modules/imgproc/src/color.cpp
${opencv}/modules/imgproc/src/connectedcomponents.cpp
${opencv}/modules/imgproc/src/contours.cpp
${opencv}/modules/imgproc/src/convhull.cpp
${opencv}/modules/imgproc/src/corner.cpp
${opencv}/modules/imgproc/src/cornersubpix.cpp
${opencv}/modules/imgproc/src/demosaicing.cpp
${opencv}/modules/imgproc/src/deriv.cpp
${opencv}/modules/imgproc/src/distransform.cpp
${opencv}/modules/imgproc/src/emd.cpp
${opencv}/modules/imgproc/src/featureselect.cpp
${opencv}/modules/imgproc/src/filter.cpp
${opencv}/modules/imgproc/src/floodfill.cpp
${opencv}/modules/imgproc/src/gabor.cpp
${opencv}/modules/imgproc/src/generalized_hough.cpp
${opencv}/modules/imgproc/src/geometry.cpp
${opencv}/modules/imgproc/src/grabcut.cpp
${opencv}/modules/imgproc/src/histogram.cpp
${opencv}/modules/imgproc/src/hough.cpp
${opencv}/modules/imgproc/src/imgwarp.cpp
${opencv}/modules/imgproc/src/intersection.cpp
${opencv}/modules/imgproc/src/linefit.cpp
${opencv}/modules/imgproc/src/lsd.cpp
${opencv}/modules/imgproc/src/matchcontours.cpp
${opencv}/modules/imgproc/src/min_enclosing_triangle.cpp
${opencv}/modules/imgproc/src/moments.cpp
${opencv}/modules/imgproc/src/morph.cpp
${opencv}/modules/imgproc/src/phasecorr.cpp
${opencv}/modules/imgproc/src/pyramids.cpp
${opencv}/modules/imgproc/src/rotcalipers.cpp
${opencv}/modules/imgproc/src/samplers.cpp
${opencv}/modules/imgproc/src/segmentation.cpp
${opencv}/modules/imgproc/src/shapedescr.cpp
${opencv}/modules/imgproc/src/smooth.cpp
${opencv}/modules/imgproc/src/subdivision2d.cpp
${opencv}/modules/imgproc/src/sumpixels.cpp
${opencv}/modules/imgproc/src/tables.cpp
${opencv}/modules/imgproc/src/templmatch.cpp
${opencv}/modules/imgproc/src/thresh.cpp
${opencv}/modules/imgproc/src/undistort.cpp
${opencv}/modules/imgproc/src/utils.cpp
${opencv}/modules/ml/src/ann_mlp.cpp
${opencv}/modules/ml/src/boost.cpp
${opencv}/modules/ml/src/cnn.cpp
${opencv}/modules/ml/src/data.cpp
${opencv}/modules/ml/src/em.cpp
${opencv}/modules/ml/src/ertrees.cpp
${opencv}/modules/ml/src/estimate.cpp
${opencv}/modules/ml/src/gbt.cpp
${opencv}/modules/ml/src/inner_functions.cpp
${opencv}/modules/ml/src/knearest.cpp
${opencv}/modules/ml/src/ml_init.cpp
${opencv}/modules/ml/src/nbayes.cpp
${opencv}/modules/ml/src/rtrees.cpp
${opencv}/modules/ml/src/svm.cpp
${opencv}/modules/ml/src/testset.cpp
${opencv}/modules/ml/src/tree.cpp
${opencv}/modules/objdetect/opencl_kernels.cpp
${opencv}/modules/objdetect/src/cascadedetect.cpp
${opencv}/modules/objdetect/src/cascadedetect_convert.cpp
${opencv}/modules/objdetect/src/datamatrix.cpp
${opencv}/modules/objdetect/src/distancetransform.cpp
${opencv}/modules/objdetect/src/erfilter.cpp
${opencv}/modules/objdetect/src/featurepyramid.cpp
${opencv}/modules/objdetect/src/fft.cpp
${opencv}/modules/objdetect/src/haar.cpp
${opencv}/modules/objdetect/src/hog.cpp
${opencv}/modules/objdetect/src/latentsvm.cpp
${opencv}/modules/objdetect/src/latentsvmdetector.cpp
${opencv}/modules/objdetect/src/linemod.cpp
${opencv}/modules/objdetect/src/lsvmparser.cpp
${opencv}/modules/objdetect/src/lsvmtbbversion.cpp
${opencv}/modules/objdetect/src/matching.cpp
${opencv}/modules/objdetect/src/resizeimg.cpp
${opencv}/modules/objdetect/src/routine.cpp
${opencv}modules/core/src/algorithm.cpp
${opencv}modules/core/src/alloc.cpp
${opencv}modules/core/src/arithm.cpp
${opencv}modules/core/src/array.cpp
${opencv}modules/core/src/command_line_parser.cpp
${opencv}modules/core/src/convert.cpp
${opencv}modules/core/src/copy.cpp
${opencv}modules/core/src/cuda_buffer_pool.cpp
${opencv}modules/core/src/cuda_gpu_mat.cpp
${opencv}modules/core/src/cuda_host_mem.cpp
${opencv}modules/core/src/cuda_info.cpp
${opencv}modules/core/src/cuda_stream.cpp
${opencv}modules/core/src/datastructs.cpp
${opencv}modules/core/src/directx.cpp
${opencv}modules/core/src/drawing.cpp
${opencv}modules/core/src/dxt.cpp
${opencv}modules/core/src/glob.cpp
${opencv}modules/core/src/lapack.cpp
${opencv}modules/core/src/mathfuncs.cpp
${opencv}modules/core/src/matmul.cpp
${opencv}modules/core/src/matop.cpp
${opencv}modules/core/src/matop.cpp
${opencv}modules/core/src/matrix.cpp
${opencv}modules/core/src/ocl.cpp
${opencv}modules/core/src/out.cpp
${opencv}modules/core/src/parallel.cpp
${opencv}modules/core/src/persistence.cpp
${opencv}modules/core/src/rand.cpp
${opencv}modules/core/src/stat.cpp
${opencv}modules/core/src/stl.cpp
${opencv}modules/core/src/system.cpp
${opencv}modules/core/src/tables.cpp
${opencv}modules/core/src/types.cpp
${opencv}modules/core/src/umatrix.cpp
${opencv}modules/core/src/opengl.cpp)
 
add_executable(face face.c )
find_package( ZLIB REQUIRED )
find_package (Threads)
FIND_LIBRARY(LIBDL_LIBRARIES NAMES dl libdl )
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
 
include_directories(${PTHREAD_INCLUDE_DIR})
 
target_link_libraries(face ${OPENCV_LINKER_LIBS} ${OPENCV_C_SAMPLES_REQUIRED_DEPS})
target_link_libraries( face ${ZLIB_LIBRARIES} )
target_link_libraries (face ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(face ${PTHREAD_LIBRARY})
target_link_libraries(face ${PNG_LIBRARY})
target_link_libraries(face ${JPEG_LIBRARY})
target_link_libraries(face opencv-face)
