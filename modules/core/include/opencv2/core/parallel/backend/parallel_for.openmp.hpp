// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_PARALLEL_FOR_OPENMP_HPP
#define OPENCV_CORE_PARALLEL_FOR_OPENMP_HPP

#include "opencv2/core/parallel/parallel_backend.hpp"

#if !defined(_OPENMP) && !defined(OPENCV_SKIP_OPENMP_PRESENSE_CHECK)
#error "This file must be compiled with enabled OpenMP"
#endif

#include <omp.h>

namespace cv { namespace parallel { namespace openmp {

/** OpenMP parallel_for API implementation
 *
 * @sa setParallelForBackend
 * @ingroup core_parallel_backend
 */
class ParallelForBackend : public ParallelForAPI
{
protected:
    int numThreads;
    int numThreadsMax;
public:
    ParallelForBackend()
    {
        numThreads = 0;
        numThreadsMax = omp_get_max_threads();
    }

    virtual ~ParallelForBackend() {}

    virtual void parallel_for(int tasks, FN_parallel_for_body_cb_t body_callback, void* callback_data) CV_OVERRIDE
    {
#pragma omp parallel for schedule(dynamic) num_threads(numThreads > 0 ? numThreads : numThreadsMax)
        for (int i = 0; i < tasks; ++i)
            body_callback(i, i + 1, callback_data);
    }

    virtual int getThreadNum() const CV_OVERRIDE
    {
        return omp_get_thread_num();
    }

    virtual int getNumThreads() const CV_OVERRIDE
    {
        return numThreads > 0
               ? numThreads
               : numThreadsMax;
    }

    virtual int setNumThreads(int nThreads) CV_OVERRIDE
    {
        int oldNumThreads = numThreads;
        numThreads = nThreads;
        // nothing needed as numThreads is used in #pragma omp parallel for directly
        return oldNumThreads;
    }

    const char* getName() const CV_OVERRIDE
    {
        return "openmp";
    }
};

}}}  // namespace

#endif  // OPENCV_CORE_PARALLEL_FOR_OPENMP_HPP
