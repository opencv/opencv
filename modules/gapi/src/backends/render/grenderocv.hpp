// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GRENDEROCV_HPP
#define OPENCV_GAPI_GRENDEROCV_HPP

#include "backends/render/grenderkernel.hpp"

namespace cv
{
namespace gapi
{
namespace render
{
namespace ocv
{

GAPI_EXPORTS cv::gapi::GBackend backend();

struct KImpl {
    using Run = std::function<void(GRenderContext& ctx)>;
    Run run;
};

template <typename Name, typename K>
struct GRenderKernelImpl
{
    using API = K;

    static cv::gapi::GBackend backend() { return cv::gapi::render::ocv::backend(); }
    static KImpl kernel()               { return KImpl{Name::run};                 }
};

} // namespace ocv
} // namespace render
} // namespace gapi
} // namespace cv

#define GAPI_RENDER_OCV_KERNEL(Name, API) struct Name: public cv::gapi::render::ocv::GRenderKernelImpl<Name, API>, \
                                                       public cv::detail::KernelTag

#endif // OPENCV_GAPI_GRENDEROCV_HPP
