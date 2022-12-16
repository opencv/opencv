// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "op_cann.hpp"

#include <mutex>
#include <map>
#include <cstring> // memcpy

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_CANN

std::shared_ptr<AclEnvGuard> AclEnvGuard::global_acl_env_ = nullptr;
std::mutex AclEnvGuard::global_acl_env_mutex_;

AclEnvGuard::AclEnvGuard()
{
    CV_LOG_INFO(NULL, "Start to initialize CANN");
    ACL_CHECK_RET(aclInit(NULL));
    CV_LOG_INFO(NULL, "[Success] initialized CANN");
}

AclEnvGuard::~AclEnvGuard()
{
    CV_LOG_INFO(NULL, "Start to finalize CANN");
    ACL_CHECK_RET(aclFinalize());
    CV_LOG_INFO(NULL, "[Success] finalized CANN");
}

std::shared_ptr<AclEnvGuard> AclEnvGuard::GetAclEnv()
{
    std::shared_ptr<AclEnvGuard> acl_env;

    std::lock_guard<std::mutex> lock(global_acl_env_mutex_);
    acl_env = global_acl_env_;
    if (acl_env != nullptr)
    {
        CV_LOG_INFO(NULL, "CANN has been initialized. Skipping...");
    }
    else
    {
        acl_env = std::make_shared<AclEnvGuard>();
        global_acl_env_ = acl_env;
    }
    return acl_env;
}

CannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)
{
    std::vector<int64_t> shape_{shape.begin(), shape.end()};

    auto ge_shape = ge::Shape(shape_);
    auto ge_dtype = ge::DT_FLOAT;
    switch (dtype)
    {
        case CV_32F: break;
        case CV_32S: ge_dtype = ge::DT_INT32; break;
        default: CV_Error(Error::StsNotImplemented, "Unsupported data type");
    }
    auto size_of_type = sizeof(float);
    switch (dtype)
    {
        case CV_32F: break;
        case CV_32S: size_of_type = sizeof(int); break;
        default: CV_Error(Error::StsNotImplemented, "Unsupported data type");
    }
    desc_ = std::make_shared<ge::TensorDesc>(ge_shape, ge::FORMAT_NCHW, ge_dtype);
    auto ge_tensor = std::make_shared<ge::Tensor>();
    ge_tensor->SetTensorDesc(*desc_);
    ge_tensor->SetData(data, ge_shape.GetShapeSize() * size_of_type);
    op_ = std::make_shared<ge::op::Const>(name);
    op_->set_attr_value(*ge_tensor);
}

CannBackendNode::CannBackendNode(const std::shared_ptr<ge::Operator>& op)
    : BackendNode(DNN_BACKEND_CANN), op_(op) { }

std::shared_ptr<ge::Operator> CannBackendNode::getOp() { return op_; }

CannBackendWrapper::CannBackendWrapper(const Mat& m)
    : BackendWrapper(DNN_BACKEND_CANN, DNN_TARGET_NPU),  host((Mat*)&m)
{
    auto mat_shape = shape(*host);
    std::vector<int64_t> shape_{mat_shape.begin(), mat_shape.end()};

    auto ge_shape = ge::Shape(shape_);
    desc_ = std::make_shared<ge::TensorDesc>(ge_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
}

void CannBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "Not implemented");
}

void CannBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "Not implemented");
}

CannNet::~CannNet()
{
    CV_LOG_INFO(NULL, "In ~CannNet, inputs = " << inputs << ", outputs = " << outputs);
    if (!model_desc)
    {
        CV_LOG_INFO(NULL, "[Failed] Tried to deconstruct CannNet but model is not loaded");
        return;
    }
    // free datasets: inputs, outputs
    if (inputs)
    {
        CV_LOG_INFO(NULL, "In ~CannNet: destroy inputs");
        destroyDataset(&inputs);
    }
    if (outputs)
    {
        CV_LOG_INFO(NULL, "In ~CannNet: destroy outputs");
        destroyDataset(&outputs);
    }
    // unload model
    ACL_CHECK_RET(aclmdlUnload(model_id));
    // destroy model_desc
    ACL_CHECK_RET(aclmdlDestroyDesc(model_desc));
    model_desc = nullptr;
    CV_LOG_INFO(NULL, "[Success] Unloaded model (id=" << model_id << ")");

    // destroy context
    if (context != nullptr)
    {
        ACL_CHECK_RET(aclrtDestroyContext(context));
        context = nullptr;
    }
    // reset device
    if (context == nullptr)
    {
        ACL_CHECK_RET(aclrtResetDevice(device_id));
    }
}

bool CannNet::empty() const
{
    return (model_desc == nullptr);
}

void CannNet::loadModelBuffer(std::shared_ptr<ge::ModelBufferData> modelBuffer)
{
    model.clear();
    model.resize(modelBuffer->length);
    std::memcpy(reinterpret_cast<void*>(model.data()),
                reinterpret_cast<void*>(modelBuffer->data.get()),
                modelBuffer->length);
    loadToDevice();
}

void CannNet::bindInputWrappers(const std::vector<Ptr<BackendWrapper>>& inputWrappers)
{
    CV_Assert(inputWrappers.size() == getInputNum());
    for (size_t i = 0; i < inputWrappers.size(); ++i)
    {
        auto wrapper = inputWrappers[i].dynamicCast<CannBackendWrapper>();

        // verify size
        aclmdlIODims model_dims;
        ACL_CHECK_RET(aclmdlGetInputDims(model_desc, i, &model_dims));
        CV_CheckEQ((int)model_dims.dimCount, wrapper->host->dims, "Dimension of input does not match with model's requirement");
        for (size_t j = 0; j < model_dims.dimCount; ++j)
            CV_CheckEQ((int)model_dims.dims[j], wrapper->host->size[j], "Size of input does not match with model's requirement");

        input_wrappers.push_back(wrapper);
    }
}

void CannNet::bindOutputWrappers(const std::vector<Ptr<BackendWrapper>>& outputWrappers)
{
    CV_Assert(outputWrappers.size() == getOutputNum());
    for (int i = 0; i < outputWrappers.size(); ++i)
    {
        auto wrapper = outputWrappers[i].dynamicCast<CannBackendWrapper>();

        // verify size
        aclmdlIODims model_dims;
        ACL_CHECK_RET(aclmdlGetOutputDims(model_desc, i, &model_dims));
        CV_CheckEQ((int)model_dims.dimCount, wrapper->host->dims, "Dimension of input does not match with model's requirement");
        for (size_t j = 0; j < model_dims.dimCount; ++j)
            CV_CheckEQ((int)model_dims.dims[j], wrapper->host->size[j], "Size of input does not match with model's requirement");

        output_wrappers.push_back(wrapper);
    }
}

void CannNet::forward()
{
    // send inputs from host to device
    CV_LOG_DEBUG(NULL, "DNN/CANN: start sending inputs to device");
    for (size_t i = 0; i < input_wrappers.size(); ++i)
    {
        const void* p_host = (const void*)input_wrappers[i]->host->data;

        auto db = aclmdlGetDatasetBuffer(inputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);

        ACL_CHECK_RET(aclrtMemcpy(p_device, db_size, p_host, db_size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CV_LOG_DEBUG(NULL, "DNN/CANN: finished sending inputs to device");

    // forward
    CV_LOG_DEBUG(NULL, "DNN/CANN: start network forward");
    ACL_CHECK_RET(aclrtSetCurrentContext(context));
    ACL_CHECK_RET(aclmdlExecute(model_id, inputs, outputs));
    CV_LOG_DEBUG(NULL, "DNN/CANN: finished network forward");

    // fetch ouputs from device to host
    CV_LOG_DEBUG(NULL, "DNN/CANN: start fetching outputs to host");
    for (size_t i = 0; i < output_wrappers.size(); ++i)
    {
        void* p_host = (void*)output_wrappers[i]->host->data;

        auto db = aclmdlGetDatasetBuffer(outputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);

        ACL_CHECK_RET(aclrtMemcpy(p_host, db_size, p_device, db_size, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    CV_LOG_DEBUG(NULL, "DNN/CANN: finish fetching outputs to host");
}

size_t CannNet::getInputNum() const
{
    return aclmdlGetNumInputs(model_desc);
}

size_t CannNet::getOutputNum() const
{
    return aclmdlGetNumOutputs(model_desc);
}

void CannNet::init()
{
    ACL_CHECK_RET(aclrtSetDevice(device_id));
    ACL_CHECK_RET(aclrtCreateContext(&context, device_id));
}

void CannNet::loadToDevice()
{
    if (model_desc != nullptr)
    {
        CV_LOG_INFO(NULL, "Model has been loaded to device. Skipping ...");
        return;
    }

    CV_LOG_INFO(NULL, "Load model to NPU memory");
    ACL_CHECK_RET(aclmdlLoadFromMem(reinterpret_cast<const void*>(model.data()), model.size(), &model_id));

    CV_LOG_INFO(NULL, "Create model description");
    model_desc = aclmdlCreateDesc();
    ACL_CHECK_RET(aclmdlGetDesc(model_desc, model_id));

    createInputDataset();
    createOutputDataset();
}

void CannNet::createInputDataset()
{
    inputs = aclmdlCreateDataset();
    size_t n_inputs = aclmdlGetNumInputs(model_desc);
    size_t length;
    for (size_t i = 0; i < n_inputs; i++)
    {
        length = aclmdlGetInputSizeByIndex(model_desc, i);
        CV_LOG_INFO(NULL, "length = " << length);
        void* p_device = nullptr;
        ACL_CHECK_RET(aclrtMalloc(&p_device, length, ACL_MEM_MALLOC_NORMAL_ONLY));
        auto p_data_buffer = aclCreateDataBuffer(p_device, length);
        ACL_CHECK_RET(aclmdlAddDatasetBuffer(inputs, p_data_buffer));
    }
}

void CannNet::createOutputDataset()
{
    outputs = aclmdlCreateDataset();
    size_t n_outputs = aclmdlGetNumOutputs(model_desc);
    size_t length;
    for (size_t i = 0; i < n_outputs; i++)
    {
        length = aclmdlGetOutputSizeByIndex(model_desc, i);
        void* p_device = nullptr;
        ACL_CHECK_RET(aclrtMalloc(&p_device, length, ACL_MEM_MALLOC_NORMAL_ONLY));
        auto p_data_buffer = aclCreateDataBuffer(p_device, length);
        ACL_CHECK_RET(aclmdlAddDatasetBuffer(outputs, p_data_buffer));
    }
}

void CannNet::destroyDataset(aclmdlDataset** dataset)
{
    if (!dataset)
    {
        CV_LOG_INFO(NULL, "CANN dataset is not initialized");
        return;
    }
    auto buffer_count = aclmdlGetDatasetNumBuffers(*dataset);
    CV_LOG_INFO(NULL, "buffer_count = " << buffer_count);
    for (auto i = 0; i < buffer_count; i++)
    {
        auto data_buffer = aclmdlGetDatasetBuffer(*dataset, i);
        auto p_device = aclGetDataBufferAddr(data_buffer);
        if (p_device)
        {
            ACL_CHECK_RET(aclrtFree(p_device)); // 107000?
        }
        else
        {
            CV_LOG_INFO(NULL, "Data buffer (i=" << i << ") from ACL dataset is invalid");
        }
        ACL_CHECK_RET(aclDestroyDataBuffer(data_buffer));
    }
    ACL_CHECK_RET(aclmdlDestroyDataset(*dataset));
    *dataset = nullptr;
    CV_LOG_INFO(NULL, "[Success] Destroyed dataset");
}

#endif // HAVE_CANN

}} // namespace cv::dnn
