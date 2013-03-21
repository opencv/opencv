#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "opencv2/ts/gpu_perf.hpp"
#include "opencv2/superres/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
