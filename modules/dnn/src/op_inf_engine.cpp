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
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

CV__DNN_INLINE_NS_BEGIN

cv::String getInferenceEngineBackendType()
{
    return "NGRAPH";
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
    if (newBackendType != "NGRAPH")
        CV_Error(Error::StsNotImplemented, cv::format("DNN/IE: only NGRAPH backend is supported: %s", newBackendType.c_str()));
    return newBackendType;
}

CV__DNN_INLINE_NS_END


Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();

    int type = -1;
    switch (precision)
    {
        case InferenceEngine::Precision::FP32: type = CV_32F; break;
        case InferenceEngine::Precision::U8: type = CV_8U; break;
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    }
    return Mat(size, type, (void*)blob->buffer());
}

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats)
{
    mats.resize(blobs.size());
    for (int i = 0; i < blobs.size(); ++i)
        mats[i] = infEngineBlobToMat(blobs[i]);
}


static bool init_IE_plugins()
{
    // load and hold IE plugins
    static InferenceEngine::Core* init_core = new InferenceEngine::Core();  // 'delete' is never called
    (void)init_core->GetAvailableDevices();
    return true;
}
static InferenceEngine::Core& retrieveIECore(const std::string& id, std::map<std::string, std::shared_ptr<InferenceEngine::Core> >& cores)
{
    AutoLock lock(getInitializationMutex());
    std::map<std::string, std::shared_ptr<InferenceEngine::Core> >::iterator i = cores.find(id);
    if (i == cores.end())
    {
        std::shared_ptr<InferenceEngine::Core> core = std::make_shared<InferenceEngine::Core>();
        cores[id] = core;
        return *core.get();
    }
    return *(i->second).get();
}
static InferenceEngine::Core& create_IE_Core_instance(const std::string& id)
{
    static std::map<std::string, std::shared_ptr<InferenceEngine::Core> > cores;
    return retrieveIECore(id, cores);
}
static InferenceEngine::Core& create_IE_Core_pointer(const std::string& id)
{
    // load and hold IE plugins
    static std::map<std::string, std::shared_ptr<InferenceEngine::Core> >* cores =
            new std::map<std::string, std::shared_ptr<InferenceEngine::Core> >();
    return retrieveIECore(id, *cores);
}
InferenceEngine::Core& getCore(const std::string& id)
{
    // to make happy memory leak tools use:
    // - OPENCV_DNN_INFERENCE_ENGINE_HOLD_PLUGINS=0
    // - OPENCV_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND=0
    static bool param_DNN_INFERENCE_ENGINE_HOLD_PLUGINS = utils::getConfigurationParameterBool("OPENCV_DNN_INFERENCE_ENGINE_HOLD_PLUGINS", true);
    static bool init_IE_plugins_ = param_DNN_INFERENCE_ENGINE_HOLD_PLUGINS && init_IE_plugins(); CV_UNUSED(init_IE_plugins_);

    static bool param_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND =
            utils::getConfigurationParameterBool("OPENCV_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND",
#ifdef _WIN32
                true
#else
                false
#endif
            );

    InferenceEngine::Core& core = param_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND
            ? create_IE_Core_pointer(id)
            : create_IE_Core_instance(id);
    return core;
}


static bool detectArmPlugin_()
{
    InferenceEngine::Core& ie = getCore("CPU");
    const std::vector<std::string> devices = ie.GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (i->find("CPU") != std::string::npos)
        {
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            CV_LOG_INFO(NULL, "CPU plugin: " << name);
            return name.find("arm_compute::NEON") != std::string::npos;
        }
    }
    return false;
}

#if !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
static bool detectMyriadX_(const std::string& device)
{
    AutoLock lock(getInitializationMutex());

    // Lightweight detection
    InferenceEngine::Core& ie = getCore(device);
    const std::vector<std::string> devices = ie.GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (i->find(device) != std::string::npos)
        {
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            CV_LOG_INFO(NULL, "Myriad device: " << name);
            return name.find("MyriadX") != std::string::npos || name.find("Myriad X") != std::string::npos || name.find("HDDL") != std::string::npos;
        }
    }
    return false;
}
#endif  // !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)


#endif  // HAVE_INF_ENGINE


CV__DNN_INLINE_NS_BEGIN

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    CV_LOG_INFO(NULL, "DNN: Unregistering both 'MYRIAD' and 'HETERO:MYRIAD,CPU' plugins");

    AutoLock lock(getInitializationMutex());

    InferenceEngine::Core& ie = getCore("MYRIAD");
    try
    {
        ie.UnregisterPlugin("MYRIAD");
        ie.UnregisterPlugin("HETERO");
    }
    catch (...) {}
#endif  // HAVE_INF_ENGINE
}

void releaseHDDLPlugin()
{
#ifdef HAVE_INF_ENGINE
    CV_LOG_INFO(NULL, "DNN: Unregistering both 'HDDL' and 'HETERO:HDDL,CPU' plugins");

    AutoLock lock(getInitializationMutex());

    InferenceEngine::Core& ie = getCore("HDDL");
    try
    {
        ie.UnregisterPlugin("HDDL");
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

bool isArmComputePlugin()
{
    static bool armPlugin = getInferenceEngineCPUType() == CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE;
    return armPlugin;
}

static std::string getInferenceEngineVPUType_()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_DNN_IE_VPU_TYPE", "");
    if (param_vpu_type == "")
    {
#if defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
        param_vpu_type = OPENCV_DNN_IE_VPU_TYPE_DEFAULT;
#else
        CV_LOG_INFO(NULL, "OpenCV-DNN: running Inference Engine VPU autodetection: Myriad2/X or HDDL. In case of other accelerator types specify 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        try {
            bool isMyriadX_ = detectMyriadX_("MYRIAD");
            bool isHDDL_ = detectMyriadX_("HDDL");
            if (isMyriadX_ || isHDDL_)
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

cv::String getInferenceEngineCPUType()
{
    static cv::String cpu_type = detectArmPlugin_() ?
                                 CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE :
                                 CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86;
    return cpu_type;
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

cv::String getInferenceEngineCPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
#endif  // HAVE_INF_ENGINE


CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv
