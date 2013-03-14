#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include <cstdio>
#include <iostream>

#include "cvconfig.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "opencv2/ts.hpp"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/gpu.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/nonfree.hpp"
#include "opencv2/legacy.hpp"
#include "opencv2/photo.hpp"

#include "utility.hpp"

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
