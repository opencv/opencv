// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_INF_ENGINE
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

static Backend parseInferenceEngineBackendType(const cv::String& backend)
{
    CV_Assert(!backend.empty());
    if (backend == CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        return DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    if (backend == CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API)
        return DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019;
    CV_Error(Error::StsBadArg, cv::format("Unknown IE backend: %s", backend.c_str()));
}
static const char* dumpInferenceEngineBackendType(Backend backend)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        return CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        return CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API;
    CV_Error(Error::StsBadArg, cv::format("Invalid backend ID for IE: %d", backend));
}
Backend& getInferenceEngineBackendTypeParam()
{
    static Backend param = parseInferenceEngineBackendType(
        utils::getConfigurationParameterString("OPENCV_DNN_BACKEND_INFERENCE_ENGINE_TYPE",
#ifdef HAVE_DNN_NGRAPH
            CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
#elif defined(HAVE_DNN_IE_NN_BUILDER_2019)
            CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API
#else
#error "Build configuration error: nGraph or NN Builder API backend should be enabled"
#endif
        )
    );
    return param;
}

CV__DNN_EXPERIMENTAL_NS_BEGIN

cv::String getInferenceEngineBackendType()
{
    return dumpInferenceEngineBackendType(getInferenceEngineBackendTypeParam());
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
    Backend newBackend = parseInferenceEngineBackendType(newBackendType);
    Backend& param = getInferenceEngineBackendTypeParam();
    Backend old = param;
    param = newBackend;
    return dumpInferenceEngineBackendType(old);
}

CV__DNN_EXPERIMENTAL_NS_END

#if !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
static bool detectMyriadX_()
{
    AutoLock lock(getInitializationMutex());
    // Lightweight detection
    InferenceEngine::Core& ie = getCore("MYRIAD");
    const std::vector<std::string> devices = ie.GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (i->find("MYRIAD") != std::string::npos)
        {
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            CV_LOG_INFO(NULL, "Myriad device: " << name);
            return name.find("MyriadX") != std::string::npos  || name.find("Myriad X") != std::string::npos;
        }
    }
    return false;
}
#endif  // !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
#endif  // HAVE_INF_ENGINE

CV__DNN_EXPERIMENTAL_NS_BEGIN

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    AutoLock lock(getInitializationMutex());
    // Unregister both "MYRIAD" and "HETERO:MYRIAD,CPU" plugins
    InferenceEngine::Core& ie = getCore("MYRIAD");
    try
    {
        ie.UnregisterPlugin("MYRIAD");
        ie.UnregisterPlugin("HETERO");
    }
    catch (...) {}
#endif  // HAVE_INF_ENGINE
}

#ifdef HAVE_INF_ENGINE
bool isMyriadX()
{
     static bool myriadX = getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
     return myriadX;
}

static std::string getInferenceEngineVPUType_()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_DNN_IE_VPU_TYPE", "");
    if (param_vpu_type == "")
    {
#if defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
        param_vpu_type = OPENCV_DNN_IE_VPU_TYPE_DEFAULT;
#else
        CV_LOG_INFO(NULL, "OpenCV-DNN: running Inference Engine VPU autodetection: Myriad2/X. In case of other accelerator types specify 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        try {
            bool isMyriadX_ = detectMyriadX_();
            if (isMyriadX_)
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
            }
            else
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2;
            }
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "OpenCV-DNN: Failed Inference Engine VPU autodetection. Specify 'OPENCV_DNN_IE_VPU_TYPE' parameter.");
            param_vpu_type.clear();
        }
#endif
    }
    CV_LOG_INFO(NULL, "OpenCV-DNN: Inference Engine VPU type='" << param_vpu_type << "'");
    return param_vpu_type;
}

cv::String getInferenceEngineVPUType()
{
    static cv::String vpu_type = getInferenceEngineVPUType_();
    return vpu_type;
}

#else  // HAVE_INF_ENGINE

cv::String getInferenceEngineBackendType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
    CV_UNUSED(newBackendType);
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String getInferenceEngineVPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
#endif  // HAVE_INF_ENGINE


CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, namespace cv
