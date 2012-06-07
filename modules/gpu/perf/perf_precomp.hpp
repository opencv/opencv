#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include <cstdio>
#include <iostream>

#include "cvconfig.h"

#include "opencv2/ts/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "perf_utility.hpp"

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
