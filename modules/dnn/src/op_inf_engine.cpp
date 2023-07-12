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
#elif defined(ENABLE_PLUGINS)
// using plugin API
#include "backend.hpp"
#include "factory.hpp"
#endif

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

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
namespace InferenceEngine {

CNNNetwork::CNNNetwork() {}

CNNNetwork::CNNNetwork(std::shared_ptr<ov::Model> model) : model(model) {}

std::shared_ptr<ov::Model> CNNNetwork::getFunction() const {
    return model;
}

void CNNNetwork::serialize(const std::string& xmlPath, const std::string& binPath) {
    ov::pass::Serialize(xmlPath, binPath).run_on_model(model);
}

void CNNNetwork::reshape(const std::map<std::string, std::vector<size_t> >& shapes) {
    std::map<std::string, ov::PartialShape> partialShapes;
    for (const auto& it : shapes) {
        ov::PartialShape shape;
        shape.insert(shape.begin(), it.second.begin(), it.second.end());
        partialShapes.insert({it.first, shape});
    }
    model->reshape(partialShapes);
}

std::vector<std::string> Core::GetAvailableDevices() {
    return get_available_devices();
}

void Core::UnregisterPlugin(const std::string& id) {
    unload_plugin(id);
}

CNNNetwork Core::ReadNetwork(const std::string& xmlPath, const std::string& binPath) {
    return read_model(xmlPath, binPath);
}

ExecutableNetwork Core::LoadNetwork(CNNNetwork net, const std::string& device,
                                    const std::map<std::string, std::string>& config) {
    ov::AnyMap props;
    for (const auto& it : config) {
        props.insert(it);
    }
    return compile_model(net.getFunction(), device, props);
}

ExecutableNetwork::ExecutableNetwork() {}

ExecutableNetwork::ExecutableNetwork(const ov::CompiledModel& copy) : CompiledModel(copy) {}

ov::InferRequest ExecutableNetwork::CreateInferRequest() { return create_infer_request(); }

}  // namespace InferenceEngine

Mat infEngineBlobToMat(const ov::Tensor& blob)
{
    std::vector<size_t> dims = blob.get_shape();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob.get_element_type();

    int type = -1;
    switch (precision)
    {
        case ov::element::f32: type = CV_32F; break;
        case ov::element::u8: type = CV_8U; break;
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    }
    return Mat(size, type, blob.data());
}

void infEngineBlobsToMats(const ov::TensorVector& blobs,
                          std::vector<Mat>& mats)
{
    mats.resize(blobs.size());
    for (int i = 0; i < blobs.size(); ++i)
        mats[i] = infEngineBlobToMat(blobs[i]);
}

#else

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
#endif // OpenVINO >= 2022.1

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
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
            const std::string name = ie.get_property(*i, ov::device::full_name);
#else
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
#endif
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
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
            const std::string name = ie.get_property(*i, ov::device::full_name);
#else
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
#endif
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


namespace openvino {

bool checkTarget(Target target)
{
    // Lightweight detection
    const std::vector<std::string> devices = getCore("").GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (std::string::npos != i->find("MYRIAD") && target == DNN_TARGET_MYRIAD)
            return true;
        if (std::string::npos != i->find("HDDL") && target == DNN_TARGET_HDDL)
            return true;
        else if (std::string::npos != i->find("FPGA") && target == DNN_TARGET_FPGA)
            return true;
        else if (std::string::npos != i->find("CPU") && target == DNN_TARGET_CPU)
            return true;
        else if (std::string::npos != i->find("GPU") && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
            return true;
    }
    return false;
}

}  // namespace openvino

#else  // HAVE_INF_ENGINE


namespace openvino {

bool checkTarget(Target target)
{
#if defined(ENABLE_PLUGINS)
    try
    {
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        return networkBackend.checkTarget(target);
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: checkTarget failed: " << e.what())
    }
#endif
    return false;
}

}  // namespace openvino


cv::String getInferenceEngineBackendType()
{
#if defined(ENABLE_PLUGINS)
    try
    {
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        CV_UNUSED(networkBackend);
        return CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: plugin is not available: " << e.what())
    }
#endif
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
#if defined(ENABLE_PLUGINS)
    try
    {
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        CV_UNUSED(networkBackend);
        CV_Assert(newBackendType == CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: plugin is not available: " << e.what())
    }
#endif
    CV_UNUSED(newBackendType);
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String getInferenceEngineVPUType()
{
#if defined(ENABLE_PLUGINS)
    try
    {
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        if (networkBackend.checkTarget(DNN_TARGET_MYRIAD))
            return CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;  // 2021.4 supports NCS2 only
        CV_Error(Error::StsError, "DNN/OpenVINO: DNN_TARGET_MYRIAD is not available");
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: plugin is not available: " << e.what())
    }
#endif
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}

cv::String getInferenceEngineCPUType()
{
#if defined(ENABLE_PLUGINS)
    try
    {
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        CV_UNUSED(networkBackend);
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
        return CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE;
#else
        return CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86;
#endif
    }
    catch (const std::exception& e)
    {
        CV_LOG_INFO(NULL, "DNN/OpenVINO: plugin is not available: " << e.what())
    }
#endif
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}

#endif  // HAVE_INF_ENGINE


CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv
