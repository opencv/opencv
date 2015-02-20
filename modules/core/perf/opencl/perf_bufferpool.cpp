// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

struct BufferPoolState
{
    BufferPoolController* controller_;
    size_t oldMaxReservedSize_;

    BufferPoolState(BufferPoolController* c, bool enable)
        : controller_(c)
    {
        if (!cv::ocl::useOpenCL())
        {
            throw ::perf::TestBase::PerfSkipTestException();
        }
        oldMaxReservedSize_ = c->getMaxReservedSize();
        if (oldMaxReservedSize_ == (size_t)-1)
        {
            throw ::perf::TestBase::PerfSkipTestException();
        }
        if (!enable)
        {
            c->setMaxReservedSize(0);
        }
        else
        {
            c->freeAllReservedBuffers();
        }
    }

    ~BufferPoolState()
    {
        controller_->setMaxReservedSize(oldMaxReservedSize_);
    }
};

typedef TestBaseWithParam<bool> BufferPoolFixture;

OCL_PERF_TEST_P(BufferPoolFixture, BufferPool_UMatCreation100, Bool())
{
    BufferPoolState s(cv::ocl::getOpenCLAllocator()->getBufferPoolController(), GetParam());

    Size sz(1920, 1080);

    OCL_TEST_CYCLE()
    {
        for (int i = 0; i < 100; i++)
        {
            UMat u(sz, CV_8UC1);
        }
    }

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(BufferPoolFixture, BufferPool_UMatCountNonZero100, Bool())
{
    BufferPoolState s(cv::ocl::getOpenCLAllocator()->getBufferPoolController(), GetParam());

    Size sz(1920, 1080);

    OCL_TEST_CYCLE()
    {
        for (int i = 0; i < 100; i++)
        {
            UMat u(sz, CV_8UC1);
            countNonZero(u);
        }
    }

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(BufferPoolFixture, BufferPool_UMatCanny10, Bool())
{
    BufferPoolState s(cv::ocl::getOpenCLAllocator()->getBufferPoolController(), GetParam());

    Size sz(1920, 1080);

    int aperture = 3;
    bool useL2 = false;
    double thresh_low = 100;
    double thresh_high = 120;

    OCL_TEST_CYCLE()
    {
        for (int i = 0; i < 10; i++)
        {
            UMat src(sz, CV_8UC1);
            UMat dst;
            Canny(src, dst, thresh_low, thresh_high, aperture, useL2);
            dst.getMat(ACCESS_READ); // complete async operations
        }
    }

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(BufferPoolFixture, BufferPool_UMatIntegral10, Bool())
{
    BufferPoolState s(cv::ocl::getOpenCLAllocator()->getBufferPoolController(), GetParam());

    Size sz(1920, 1080);

    OCL_TEST_CYCLE()
    {
        for (int i = 0; i < 10; i++)
        {
            UMat src(sz, CV_32FC1);
            UMat dst;
            integral(src, dst);
            dst.getMat(ACCESS_READ); // complete async operations
        }
    }

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
