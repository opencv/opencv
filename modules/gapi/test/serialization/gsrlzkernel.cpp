// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <cassert>

#include "gsrlzkernel.hpp"

const cv::Mat& opencv_test::GSRLZContext::inMat(int input)
{
    return inArg<cv::Mat>(input);
}

cv::Mat&  opencv_test::GSRLZContext::outMatR(int output)
{
    return *cv::util::get<cv::Mat*>(m_results.at(output));
}

const cv::Scalar& opencv_test::GSRLZContext::inVal(int input)
{
    return inArg<cv::Scalar>(input);
}

cv::Scalar& opencv_test::GSRLZContext::outValR(int output)
{
    return *cv::util::get<cv::Scalar*>(m_results.at(output));
}

cv::detail::VectorRef& opencv_test::GSRLZContext::outVecRef(int output)
{
    return cv::util::get<cv::detail::VectorRef>(m_results.at(output));
}

cv::detail::OpaqueRef& opencv_test::GSRLZContext::outOpaqueRef(int output)
{
    return cv::util::get<cv::detail::OpaqueRef>(m_results.at(output));
}

opencv_test::GSRLZKernel::GSRLZKernel()
{
}

opencv_test::GSRLZKernel::GSRLZKernel(const GSRLZKernel::F &f)
    : m_f(f)
{
}

void opencv_test::GSRLZKernel::apply(GSRLZContext &ctx)
{
    GAPI_Assert(m_f);
    m_f(ctx);
}
