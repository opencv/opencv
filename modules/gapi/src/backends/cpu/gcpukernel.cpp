// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <cassert>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

const cv::Mat& cv::GCPUContext::inMat(int input)
{
    return inArg<cv::Mat>(input);
}

cv::Mat&  cv::GCPUContext::outMatR(int output)
{
    return *util::get<cv::Mat*>(m_results.at(output));
}

const cv::Scalar& cv::GCPUContext::inVal(int input)
{
    return inArg<cv::Scalar>(input);
}

cv::Scalar& cv::GCPUContext::outValR(int output)
{
    return *util::get<cv::Scalar*>(m_results.at(output));
}

cv::detail::VectorRef& cv::GCPUContext::outVecRef(int output)
{
    return util::get<cv::detail::VectorRef>(m_results.at(output));
}

cv::detail::OpaqueRef& cv::GCPUContext::outOpaqueRef(int output)
{
    return util::get<cv::detail::OpaqueRef>(m_results.at(output));
}

cv::GCPUKernel::GCPUKernel()
{
}

cv::GCPUKernel::GCPUKernel(const GCPUKernel::F &f)
    : m_f(f)
{
}

void cv::GCPUKernel::apply(GCPUContext &ctx)
{
    GAPI_Assert(m_f);
    m_f(ctx);
}
