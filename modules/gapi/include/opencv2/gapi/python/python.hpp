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

/**
 * @brief This namespace contains G-API Python backend functions,
 * structures, and symbols.
 *
 * This functionality is required to enable G-API custom operations
 * and kernels when using G-API from Python, no need to use it in the
 * C++ form.
 */
namespace python {

GAPI_EXPORTS cv::gapi::GBackend backend();

struct GPythonContext
{
    const cv::GArgs      &ins;
    const cv::GMetaArgs  &in_metas;
    const cv::GTypesInfo &out_info;

    cv::optional<cv::GArg> m_state;
};

using Impl = std::function<cv::GRunArgs(const GPythonContext&)>;
using Setup = std::function<cv::GArg(const GMetaArgs&, const GArgs&)>;

class GAPI_EXPORTS GPythonKernel
{
public:
    GPythonKernel() = default;
    GPythonKernel(Impl run, Setup setup);

    Impl  run;
    Setup setup       = nullptr;
    bool  is_stateful = false;
};

class GAPI_EXPORTS GPythonFunctor : public cv::gapi::GFunctor
{
public:
    using Meta = cv::GKernel::M;

    GPythonFunctor(const char* id, const Meta& meta, const Impl& impl,
                   const Setup& setup = nullptr);

    GKernelImpl    impl()    const override;
    gapi::GBackend backend() const override;

private:
    GKernelImpl impl_;
};

} // namespace python
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_PYTHON_API_HPP
