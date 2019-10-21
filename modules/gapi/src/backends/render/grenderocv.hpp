// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GRENDEROCV_HPP
#define OPENCV_GAPI_GRENDEROCV_HPP

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace cv
{
namespace gapi
{
namespace render
{
namespace ocv
{

GAPI_EXPORTS cv::gapi::GBackend backend();

template<typename, typename>
struct add_type_to_tuple;

template<typename P, typename ...Ts>
struct add_type_to_tuple<P, std::tuple<Ts...>>
{
    using type = std::tuple<Ts..., P>;
};

template<class Impl, class K>
class GRenderKernelImpl: public cv::detail::OCVCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
                         public cv::detail::KernelTag
{
    // TODO Use this mechanism for adding new parameter to run method
    // using InArgs = typename add_type_to_tuple<IBitMaskCreator, typename K::InArgs>::type;
    using InArgs = typename K::InArgs;
    using P = detail::OCVCallHelper<Impl, InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend()  { return cv::gapi::render::ocv::backend(); }
    static cv::GCPUKernel     kernel()   { return GCPUKernel(&P::call);             }
};

#define GAPI_RENDER_OCV_KERNEL(Name, API) struct Name: public cv::gapi::render::ocv::GRenderKernelImpl<Name, API>

} // namespace ocv
} // namespace render
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GRENDEROCV_HPP
