// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#ifndef OPENCV_GAPI_PYTHON_API_HPP
#define OPENCV_GAPI_PYTHON_API_HPP

#include <opencv2/gapi/gkernel.hpp>     // GKernelPackage
#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace python {

GAPI_EXPORTS cv::gapi::GBackend backend();

struct GPythonContext
{
    const cv::GArgs      &ins;
    const cv::GMetaArgs  &in_metas;
    const cv::GTypesInfo &out_info;
};

using Impl = std::function<cv::GRunArgs(const GPythonContext&)>;

class GAPI_EXPORTS GPythonKernel
{
public:
    GPythonKernel() = default;
    GPythonKernel(Impl run);

    cv::GRunArgs operator()(const GPythonContext& ctx);
private:
    Impl m_run;
};

class GAPI_EXPORTS GPythonFunctor : public cv::gapi::GFunctor
{
public:
    using Meta = cv::GKernel::M;

    GPythonFunctor(const char* id, const Meta &meta, const Impl& impl);

    GKernelImpl    impl()    const override;
    gapi::GBackend backend() const override;

private:
    GKernelImpl impl_;
};

} // namespace python
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_PYTHON_API_HPP
