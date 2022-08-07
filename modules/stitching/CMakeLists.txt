set(the_description "Images stitching")

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wstrict-aliasing)
endif()

set(STITCHING_CONTRIB_DEPS "opencv_xfeatures2d")
if(BUILD_SHARED_LIBS AND BUILD_opencv_world AND OPENCV_WORLD_EXCLUDE_EXTRA_MODULES)
  set(STITCHING_CONTRIB_DEPS "")
endif()
ocv_define_module(stitching opencv_imgproc opencv_features2d opencv_calib3d opencv_flann
                  OPTIONAL opencv_cudaarithm opencv_cudawarping opencv_cudafeatures2d opencv_cudalegacy opencv_cudaimgproc ${STITCHING_CONTRIB_DEPS}
                  WRAP python)
