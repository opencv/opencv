// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <cassert>

#include "gs11nkernel.hpp"

const cv::Mat& opencv_test::GS11NContext::inMat(int input)
{
    return inArg<cv::Mat>(input);
}

cv::Mat&  opencv_test::GS11NContext::outMatR(int output)
{
    return *cv::util::get<cv::Mat*>(m_results.at(output));
}

const cv::Scalar& opencv_test::GS11NContext::inVal(int input)
{
    return inArg<cv::Scalar>(input);
}

cv::Scalar& opencv_test::GS11NContext::outValR(int output)
{
    return *cv::util::get<cv::Scalar*>(m_results.at(output));
}

cv::detail::VectorRef& opencv_test::GS11NContext::outVecRef(int output)
{
    return cv::util::get<cv::detail::VectorRef>(m_results.at(output));
}

cv::detail::OpaqueRef& opencv_test::GS11NContext::outOpaqueRef(int output)
{
    return cv::util::get<cv::detail::OpaqueRef>(m_results.at(output));
}

opencv_test::GS11NKernel::GS11NKernel()
{
}

opencv_test::GS11NKernel::GS11NKernel(const GS11NKernel::F &f)
    : m_f(f)
{
}

void opencv_test::GS11NKernel::apply(GS11NContext &ctx)
{
    GAPI_Assert(m_f);
    m_f(ctx);
}
