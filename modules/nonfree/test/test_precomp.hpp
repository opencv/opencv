#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <iostream>
#include "opencv2/ts.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/nonfree.hpp"

#include "opencv2/ts/cuda_test.hpp"

#include "opencv2/opencv_modules.hpp"
#include "cvconfig.h"

#ifdef HAVE_OPENCV_OCL
#  include "opencv2/nonfree/ocl.hpp"
#endif

#ifdef HAVE_CUDA
#  include "opencv2/nonfree/cuda.hpp"
#endif

#endif
