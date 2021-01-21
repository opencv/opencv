// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_PARALLEL_BACKEND_HPP
#define OPENCV_CORE_PARALLEL_BACKEND_HPP

#include <memory>

namespace cv { namespace parallel {
#ifndef CV_API_CALL
#define CV_API_CALL
#endif

/** @addtogroup core_parallel_backend
 * @{
 * API below is provided to resolve problem of CPU resource over-subscription by multiple thread pools from different multi-threading frameworks.
 * This is common problem for cases when OpenCV compiled threading framework is different from the Users Applications framework.
 *
 * Applications can replace OpenCV `parallel_for()` backend with own implementation (to reuse Application's thread pool).
 *
 * @note This call is not thread-safe. Consider calling this function from the `main()` before any other OpenCV processing functions (and without any other created threads).
 *
 * #### Intel TBB usage example:
 *
 * - include header with simple implementation of TBB backend:
 *   @snippet parallel_backend/example-tbb.cpp tbb_include
 * - execute backend replacement code:
 *   @snippet parallel_backend/example-tbb.cpp tbb_backend
 * - configuration of compiler/linker options is responsibility of Application's scripts
 *
 * #### OpenMP usage example:
 *
 * - include header with simple implementation of OpenMP backend:
 *   @snippet parallel_backend/example-openmp.cpp openmp_include
 * - execute backend replacement code:
 *   @snippet parallel_backend/example-openmp.cpp openmp_backend
 * - Configuration of compiler/linker options is responsibility of Application's scripts
 */

/** Interface for parallel_for backends implementations
 *
 * @sa setParallelForBackend
 */
class CV_EXPORTS ParallelForAPI
{
public:
    virtual ~ParallelForAPI();

    typedef void (CV_API_CALL *FN_parallel_for_body_cb_t)(int start, int end, void* data);

    virtual void parallel_for(int tasks, FN_parallel_for_body_cb_t body_callback, void* callback_data) = 0;

    virtual int getThreadNum() const = 0;

    virtual int getNumThreads() const = 0;

    virtual int setNumThreads(int nThreads) = 0;

    virtual const char* getName() const = 0;
};

/** @brief Replace OpenCV parallel_for backend
 *
 * Application can replace OpenCV `parallel_for()` backend with own implementation.
 *
 * @note This call is not thread-safe. Consider calling this function from the `main()` before any other OpenCV processing functions (and without any other created threads).
 */
CV_EXPORTS void setParallelForBackend(const std::shared_ptr<ParallelForAPI>& api, bool propagateNumThreads = true);

//! @}
}}  // namespace
#endif  // OPENCV_CORE_PARALLEL_BACKEND_HPP
