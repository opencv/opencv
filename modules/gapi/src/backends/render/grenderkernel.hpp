// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GRENDERKERNEL_HPP
#define OPENCV_GAPI_GRENDERKERNEL_HPP

#include <unordered_map>
#include <vector>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/own/mat.hpp>
#include <opencv2/gapi/render/render.hpp>

namespace cv
{
namespace gapi
{
namespace render
{

struct GRenderContext
{
    // Generic accessor API
    template<typename T>
    const T& inArg(std::size_t input) { return args.at(input).get<T>(); }

    // Syntax sugar
    const cv::gapi::own::Mat& inMat(std::size_t input) {
        return inArg<cv::gapi::own::Mat>(input);
    }

    const cv::gapi::wip::draw::Prims& inPrims(std::size_t input) {
        return inArg<cv::detail::VectorRef>(input).rref<cv::gapi::wip::draw::Prim>();
    }

    cv::gapi::own::Mat& outMatR(std::size_t output) {
        return *cv::util::get<cv::gapi::own::Mat*>(results.at(output));
    }

    std::vector<GArg> args;
    std::unordered_map<std::size_t, GRunArgP> results;
};

} // namespace render
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GRENDERKERNEL_HPP
