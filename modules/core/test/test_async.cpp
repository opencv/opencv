// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include <opencv2/core/async.hpp>
#include <opencv2/core/detail/async_promise.hpp>

#include <opencv2/core/bindings_utils.hpp>

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
#include <thread>
#include <chrono>
#endif

namespace opencv_test { namespace {

TEST(Core_Async, BasicCheck)
{
    Mat m(3, 3, CV_32FC1, Scalar::all(5.0f));
    AsyncPromise p;
    AsyncArray r = p.getArrayResult();
    EXPECT_TRUE(r.valid());

    // Follow the limitations of std::promise::get_future
    // https://en.cppreference.com/w/cpp/thread/promise/get_future
    EXPECT_THROW(AsyncArray r2 = p.getArrayResult(), cv::Exception);

    p.setValue(m);

    Mat m2;
    r.get(m2);
    EXPECT_EQ(0, cvtest::norm(m, m2, NORM_INF));

    // Follow the limitations of std::future::get
    // https://en.cppreference.com/w/cpp/thread/future/get
    EXPECT_FALSE(r.valid());
    Mat m3;
    EXPECT_THROW(r.get(m3), cv::Exception);
}

TEST(Core_Async, ExceptionCheck)
{
    Mat m(3, 3, CV_32FC1, Scalar::all(5.0f));
    AsyncPromise p;
    AsyncArray r = p.getArrayResult();
    EXPECT_TRUE(r.valid());

    try
    {
        CV_Error(Error::StsOk, "Test: Generated async error");
    }
    catch (const cv::Exception& e)
    {
        p.setException(e);
    }

    try {
        Mat m2;
        r.get(m2);
        FAIL() << "Exception is expected";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_EQ(Error::StsOk, e.code) << e.what();
    }

    // Follow the limitations of std::future::get
    // https://en.cppreference.com/w/cpp/thread/future/get
    EXPECT_FALSE(r.valid());
}


TEST(Core_Async, LikePythonTest)
{
    Mat m(3, 3, CV_32FC1, Scalar::all(5.0f));
    AsyncArray r = cv::utils::testAsyncArray(m);
    EXPECT_TRUE(r.valid());
    Mat m2;
    r.get(m2);
    EXPECT_EQ(0, cvtest::norm(m, m2, NORM_INF));

    // Follow the limitations of std::future::get
    // https://en.cppreference.com/w/cpp/thread/future/get
    EXPECT_FALSE(r.valid());
}


#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)

TEST(Core_Async, AsyncThread_Simple)
{
    Mat m(3, 3, CV_32FC1, Scalar::all(5.0f));
    AsyncPromise p;
    AsyncArray r = p.getArrayResult();

    std::thread t([&]{
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        try {
            p.setValue(m);
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        } catch (...) {
            std::cout << "Unknown C++ exception" << std::endl;
        }
    });

    try
    {
        Mat m2;
        r.get(m2);
        EXPECT_EQ(0, cvtest::norm(m, m2, NORM_INF));

        t.join();
    }
    catch (...)
    {
        t.join();
        throw;
    }
}

TEST(Core_Async, AsyncThread_DetachedResult)
{
    Mat m(3, 3, CV_32FC1, Scalar::all(5.0f));
    AsyncPromise p;
    {
        AsyncArray r = p.getArrayResult();
        r.release();
    }

    bool exception_ok = false;

    std::thread t([&]{
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        try {
            p.setValue(m);
        } catch (const cv::Exception& e) {
            if (e.code == Error::StsError)
                exception_ok = true;
            else
                std::cout << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        } catch (...) {
            std::cout << "Unknown C++ exception" << std::endl;
        }
    });
    t.join();

    EXPECT_TRUE(exception_ok);
}

#endif

}} // namespace
