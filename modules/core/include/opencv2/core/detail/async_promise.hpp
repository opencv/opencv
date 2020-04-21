// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_ASYNC_PROMISE_HPP
#define OPENCV_CORE_ASYNC_PROMISE_HPP

#include "../async.hpp"

#include "exception_ptr.hpp"

namespace cv {

/** @addtogroup core_async
@{
*/


/** @brief Provides result of asynchronous operations

*/
class CV_EXPORTS AsyncPromise
{
public:
    ~AsyncPromise() CV_NOEXCEPT;
    AsyncPromise() CV_NOEXCEPT;
    explicit AsyncPromise(const AsyncPromise& o) CV_NOEXCEPT;
    AsyncPromise& operator=(const AsyncPromise& o) CV_NOEXCEPT;
    void release() CV_NOEXCEPT;

    /** Returns associated AsyncArray
    @note Can be called once
    */
    AsyncArray getArrayResult();

    /** Stores asynchronous result.
    @param[in] value result
    */
    void setValue(InputArray value);

    // TODO "move" setters

#if CV__EXCEPTION_PTR
    /** Stores exception.
    @param[in] exception exception to be raised in AsyncArray
    */
    void setException(std::exception_ptr exception);
#endif

    /** Stores exception.
    @param[in] exception exception to be raised in AsyncArray
    */
    void setException(const cv::Exception& exception);

#ifdef CV_CXX11
    explicit AsyncPromise(AsyncPromise&& o) { p = o.p; o.p = NULL; }
    AsyncPromise& operator=(AsyncPromise&& o) CV_NOEXCEPT { std::swap(p, o.p); return *this; }
#endif


    // PImpl
    typedef struct AsyncArray::Impl Impl; friend struct AsyncArray::Impl;
    inline void* _getImpl() const CV_NOEXCEPT { return p; }
protected:
    Impl* p;
};


//! @}
} // namespace
#endif // OPENCV_CORE_ASYNC_PROMISE_HPP
