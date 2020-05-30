// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
//#undef CV_CXX11  // debug non C++11 mode
#include "opencv2/core/async.hpp"
#include "opencv2/core/detail/async_promise.hpp"

#include "opencv2/core/cvstd.hpp"

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#include <opencv2/core/utils/logger.hpp>


#ifdef CV_CXX11
#include <mutex>
#include <condition_variable>
#include <chrono>
#endif

namespace cv {

/**
Manages shared state of asynchronous result
*/
struct AsyncArray::Impl
{
    int refcount;
    void addrefFuture() CV_NOEXCEPT { CV_XADD(&refcount_future, 1); CV_XADD(&refcount, 1); } \
    void releaseFuture() CV_NOEXCEPT { CV_XADD(&refcount_future, -1); if(1 == CV_XADD(&refcount, -1)) delete this; } \
    int refcount_future;
    void addrefPromise() CV_NOEXCEPT { CV_XADD(&refcount_promise, 1); CV_XADD(&refcount, 1); } \
    void releasePromise() CV_NOEXCEPT { CV_XADD(&refcount_promise, -1); if(1 == CV_XADD(&refcount, -1)) delete this; } \
    int refcount_promise;

#ifdef CV_CXX11
    mutable std::mutex mtx;
    mutable std::condition_variable cond_var;
#else
    mutable cv::Mutex mtx;
#endif

    mutable bool has_result; // Mat, UMat or exception

    mutable cv::Ptr<Mat> result_mat;
    mutable cv::Ptr<UMat> result_umat;


    bool has_exception;
#if CV__EXCEPTION_PTR
    std::exception_ptr exception;
#endif
    cv::Exception cv_exception;

    mutable bool result_is_fetched;

    bool future_is_returned;

    Impl()
        : refcount(1), refcount_future(0), refcount_promise(1)
        , has_result(false)
        , has_exception(false)
        , result_is_fetched(false)
        , future_is_returned(false)
    {
        // nothing
    }

    ~Impl()
    {
        if (has_result && !result_is_fetched)
        {
            CV_LOG_INFO(NULL, "Asynchronous result has not been fetched");
        }
    }

    bool get(OutputArray dst, int64 timeoutNs) const
    {
        CV_Assert(!result_is_fetched);
        if (!has_result)
        {
            if(refcount_promise == 0)
                CV_Error(Error::StsInternal, "Asynchronous result producer has been destroyed");
            if (!wait_for(timeoutNs))
                return false;
        }
#ifdef CV_CXX11
        std::unique_lock<std::mutex> lock(mtx);
#else
        cv::AutoLock lock(mtx);
#endif
        if (has_result)
        {
            if (!result_mat.empty())
            {
                dst.move(*result_mat.get());
                result_mat.release();
                result_is_fetched = true;
                return true;
            }
            if (!result_umat.empty())
            {
                dst.move(*result_umat.get());
                result_umat.release();
                result_is_fetched = true;
                return true;
            }
#if CV__EXCEPTION_PTR
            if (has_exception && exception)
            {
                result_is_fetched = true;
                std::rethrow_exception(exception);
            }
#endif
            if (has_exception)
            {
                result_is_fetched = true;
                throw cv_exception;
            }
            CV_Error(Error::StsInternal, "AsyncArray: invalid state of 'has_result = true'");
        }
        CV_Assert(!has_result);
        CV_Assert(timeoutNs < 0);
        return false;
    }

    bool valid() const CV_NOEXCEPT
    {
        if (result_is_fetched)
            return false;
        if (refcount_promise == 0 && !has_result)
            return false;
        return true;
    }

    bool wait_for(int64 timeoutNs) const
    {
        CV_Assert(valid());
        if (has_result)
            return has_result;
        if (timeoutNs == 0)
            return has_result;
        CV_LOG_INFO(NULL, "Waiting for async result ...");
#ifdef CV_CXX11
        std::unique_lock<std::mutex> lock(mtx);
        const auto cond_pred = [&]{ return has_result == true; };
        if (timeoutNs > 0)
            return cond_var.wait_for(lock, std::chrono::nanoseconds(timeoutNs), cond_pred);
        else
        {
            cond_var.wait(lock, cond_pred);
            CV_Assert(has_result);
            return true;
        }
#else
        CV_Error(Error::StsNotImplemented, "OpenCV has been built without async waiting support (C++11 is required)");
#endif
    }

    AsyncArray getArrayResult()
    {
        CV_Assert(refcount_future == 0);
        AsyncArray result;
        addrefFuture();
        result.p = this;
        future_is_returned = true;
        return result;
    }

    void setValue(InputArray value)
    {
        if (future_is_returned && refcount_future == 0)
            CV_Error(Error::StsError, "Associated AsyncArray has been destroyed");
#ifdef CV_CXX11
        std::unique_lock<std::mutex> lock(mtx);
#else
        cv::AutoLock lock(mtx);
#endif
        CV_Assert(!has_result);
        int k = value.kind();
        if (k == _InputArray::UMAT)
        {
            result_umat = makePtr<UMat>();
            value.copyTo(*result_umat.get());
        }
        else
        {
            result_mat = makePtr<Mat>();
            value.copyTo(*result_mat.get());
        }
        has_result = true;
#ifdef CV_CXX11
        cond_var.notify_all();
#endif
    }

#if CV__EXCEPTION_PTR
    void setException(std::exception_ptr e)
    {
        if (future_is_returned && refcount_future == 0)
            CV_Error(Error::StsError, "Associated AsyncArray has been destroyed");
#ifdef CV_CXX11
        std::unique_lock<std::mutex> lock(mtx);
#else
        cv::AutoLock lock(mtx);
#endif
        CV_Assert(!has_result);
        has_exception = true;
        exception = e;
        has_result = true;
#ifdef CV_CXX11
        cond_var.notify_all();
#endif
    }
#endif

    void setException(const cv::Exception e)
    {
        if (future_is_returned && refcount_future == 0)
            CV_Error(Error::StsError, "Associated AsyncArray has been destroyed");
#ifdef CV_CXX11
        std::unique_lock<std::mutex> lock(mtx);
#else
        cv::AutoLock lock(mtx);
#endif
        CV_Assert(!has_result);
        has_exception = true;
        cv_exception = e;
        has_result = true;
#ifdef CV_CXX11
        cond_var.notify_all();
#endif
    }
};


AsyncArray::AsyncArray() CV_NOEXCEPT
    : p(NULL)
{
}

AsyncArray::~AsyncArray() CV_NOEXCEPT
{
    release();
}

AsyncArray::AsyncArray(const AsyncArray& o) CV_NOEXCEPT
    : p(o.p)
{
    if (p)
        p->addrefFuture();
}

AsyncArray& AsyncArray::operator=(const AsyncArray& o) CV_NOEXCEPT
{
    Impl* newp = o.p;
    if (newp)
        newp->addrefFuture();
    release();
    p = newp;
    return *this;
}

void AsyncArray::release() CV_NOEXCEPT
{
    Impl* impl = p;
    p = NULL;
    if (impl)
        impl->releaseFuture();
}

bool AsyncArray::get(OutputArray dst, int64 timeoutNs) const
{
    CV_Assert(p);
    return p->get(dst, timeoutNs);
}

void AsyncArray::get(OutputArray dst) const
{
    CV_Assert(p);
    bool res = p->get(dst, -1);
    CV_Assert(res);
}

bool AsyncArray::wait_for(int64 timeoutNs) const
{
    CV_Assert(p);
    return p->wait_for(timeoutNs);
}

bool AsyncArray::valid() const CV_NOEXCEPT
{
    if (!p) return false;
    return p->valid();
}


//
// AsyncPromise
//

AsyncPromise::AsyncPromise() CV_NOEXCEPT
    : p(new AsyncArray::Impl())
{
}

AsyncPromise::~AsyncPromise() CV_NOEXCEPT
{
    release();
}

AsyncPromise::AsyncPromise(const AsyncPromise& o) CV_NOEXCEPT
    : p(o.p)
{
    if (p)
        p->addrefPromise();
}

AsyncPromise& AsyncPromise::operator=(const AsyncPromise& o) CV_NOEXCEPT
{
    Impl* newp = o.p;
    if (newp)
        newp->addrefPromise();
    release();
    p = newp;
    return *this;
}

void AsyncPromise::release() CV_NOEXCEPT
{
    Impl* impl = p;
    p = NULL;
    if (impl)
        impl->releasePromise();
}

AsyncArray AsyncPromise::getArrayResult()
{
    CV_Assert(p);
    return p->getArrayResult();
}

void AsyncPromise::setValue(InputArray value)
{
    CV_Assert(p);
    return p->setValue(value);
}

void AsyncPromise::setException(const cv::Exception& exception)
{
    CV_Assert(p);
    return p->setException(exception);
}

#if CV__EXCEPTION_PTR
void AsyncPromise::setException(std::exception_ptr exception)
{
    CV_Assert(p);
    return p->setException(exception);
}
#endif

} // namespace
