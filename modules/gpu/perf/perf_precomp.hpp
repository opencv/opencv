#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  ifdef __clang__
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

#include "opencv2/ts/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/photo/photo.hpp"

#include "utility.hpp"

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
