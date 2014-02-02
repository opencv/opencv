#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include "cvconfig.h"

#include "opencv2/ts.hpp"
#include "opencv2/nonfree.hpp"
#include "opencv2/imgcodecs.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_OCL
#  include "opencv2/nonfree/ocl.hpp"
#endif

#ifdef HAVE_CUDA
#  include "opencv2/nonfree/cuda.hpp"
#endif

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
