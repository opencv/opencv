// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_CORE_SRC_PARALLEL_PARALLEL_HPP
#define OPENCV_CORE_SRC_PARALLEL_PARALLEL_HPP

#include "opencv2/core/parallel/parallel_backend.hpp"

namespace cv { namespace parallel {

extern int numThreads;

std::shared_ptr<ParallelForAPI>& getCurrentParallelForAPI();

#ifndef BUILD_PLUGIN

#ifdef HAVE_TBB
std::shared_ptr<cv::parallel::ParallelForAPI> createParallelBackendTBB();
#endif

#ifdef HAVE_OPENMP
std::shared_ptr<cv::parallel::ParallelForAPI> createParallelBackendOpenMP();
#endif

#endif  // BUILD_PLUGIN

}}  // namespace

#endif // OPENCV_CORE_SRC_PARALLEL_PARALLEL_HPP
