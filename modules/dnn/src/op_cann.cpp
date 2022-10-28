// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "op_cann.hpp"

#include <mutex>
#include <map>
#include <cstring> // memcpy
#include <cstdlib> // atexit

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

static void finalizeAclGraphBuilder()
{
    CV_LOG_INFO(NULL, "Finalizing CANN Graph builder");
    ge::aclgrphBuildFinalize();
}

void initAclGraphBuilder()
{
    using namespace ge;

    static std::mutex mtx;
    static bool aclGraphInitialized = false;
    mtx.lock();
    if (!aclGraphInitialized)
    {
        CV_LOG_INFO(NULL, "Initialize CANN Graph builder");
        std::map<AscendString, AscendString> global_options = {
            {AscendString(ir_option::SOC_VERSION), AscendString("Ascend310")},
        }; // TODO: support other chips
        ACL_CHECK_GRAPH_RET(aclgrphBuildInitialize(global_options));
        aclGraphInitialized = true;
    }
    mtx.unlock();
}

CannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)
{
    std::vector<int64_t> shape_{shape.begin(), shape.end()};
    // for (int s : shape)
    //     shape_.push_back((int64_t)s);

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

// void CannBackendNode::createCannNet(ge::Graph& graph)
// {
//     cannNet = std::make_shared<CannNet>();
//     cannNet->buildFromGraph(graph);
//     cannNet->loadToDevice();
// }

CannBackendWrapper::CannBackendWrapper(const Mat& m)
    : BackendWrapper(DNN_BACKEND_CANN, DNN_TARGET_NPU),  host((Mat*)&m)
{
    auto mat_shape = shape(*host);
    std::vector<int64_t> shape_;
    for (int s : mat_shape)
        shape_.push_back((int64_t)s);

    auto ge_shape = ge::Shape(shape_);
    desc_ = std::make_shared<ge::TensorDesc>(ge_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
}

void CannBackendWrapper::copyToHost()
{
    CV_LOG_INFO(NULL, "Not implemented");
}

void CannBackendWrapper::setHostDirty()
{
    CV_LOG_INFO(NULL, "Not implemented");
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

    // destroy stream
    // if (stream != nullptr)
    // {
    //     ACL_CHECK_RET(aclrtDestroyStream(stream));
    //     stream = nullptr;
    // }
    // destroy context
    if (context != nullptr)
    {
        ACL_CHECK_RET(aclrtDestroyContext(context));
        context = nullptr;
    }
    // reset device
    // if (context == nullptr && stream == nullptr)
    if (context == nullptr)
    {
        ACL_CHECK_RET(aclrtResetDevice(device_id));
    }
}

void CannNet::finalize()
{
    finalizeAclGraphBuilder();
}

bool CannNet::empty() const
{
    return (model_desc == nullptr);
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

void CannNet::setInput(const Mat& input, const String& name)
{
    size_t idx;
    if (!name.empty())
    {
        ACL_CHECK_RET(aclmdlGetInputIndexByName(model_desc, name.c_str(), &idx));
    }
    else
    {
        idx = 0;
    }

    // verify size
    aclmdlIODims model_dims;
    ACL_CHECK_RET(aclmdlGetInputDims(model_desc, idx, &model_dims));
    CV_CheckEQ((int)model_dims.dimCount, input.dims, "Dimension of input does not match with model's requirement");
    for (int i = 0; i < input.dims; i++)
        // CV_Assert(model_dims.dims[i] == input.size[i]);
        CV_CheckEQ((int)model_dims.dims[i], input.size[i], "Size of input does not match with model's requirement");

    auto data_buffer = aclmdlGetDatasetBuffer(inputs, idx);
    auto data_size = aclGetDataBufferSizeV2(data_buffer);
    auto p_device = aclGetDataBufferAddr(data_buffer);
    // TODO: verify input size
    CV_LOG_INFO(NULL, "Data size = " << data_size << ", Mat total = " << input.total() << ", Mat shape = " << input.size);

    const void* p_host = (const void*)input.data;
    ACL_CHECK_RET(aclrtMemcpy(p_device, data_size, p_host, data_size, ACL_MEMCPY_HOST_TO_DEVICE));
}

void CannNet::forward()
{
    ACL_CHECK_RET(aclrtSetCurrentContext(context));
    ACL_CHECK_RET(aclmdlExecute(model_id, inputs, outputs));
    CV_LOG_INFO(NULL, "[Success] Finished forward");
}

void CannNet::fetchOutput(Mat& output, const String& name)
{
    size_t idx;
    if (name.empty())
        idx = 0;
    else
        idx = getOutputIndexByName(name);

    fetchOutput(output, idx);
}

void CannNet::fetchOutput(Mat& output, const size_t idx)
{
    // TODO: check idx in range [0, net_output_number)

    auto data_buffer = aclmdlGetDatasetBuffer(outputs, idx);
    auto data_size = aclGetDataBufferSizeV2(data_buffer);
    auto p_device = aclGetDataBufferAddr(data_buffer);

    // get output dimensions
    aclmdlIODims dimensions;
    ACL_CHECK_RET(aclmdlGetCurOutputDims(model_desc, idx, &dimensions));
    std::vector<int> dims;
    for (int i = 0; i < dimensions.dimCount; i++)
        dims.push_back((int)dimensions.dims[i]);

    if (!output.empty())
        output.release();

    output = Mat(dims.size(), dims.data(), CV_32FC1); // TODO: consider other types?
    void* p_host = (void*)output.data;

    ACL_CHECK_RET(aclrtMemcpy(p_host, data_size, p_device, data_size, ACL_MEMCPY_DEVICE_TO_HOST));
}

size_t CannNet::getOutputNum() const
{
    return aclmdlGetNumOutputs(model_desc);
}

void CannNet::setOutputNames(const std::vector<std::string>& names)
{
    CV_Assert(names.size() == getOutputNum());

    output_names.clear();
    output_names.assign(names.begin(), names.end());
}

void CannNet::buildFromGraph(ge::Graph& graph)
{
    using namespace ge;

    ModelBufferData om_model;
    std::map<AscendString, AscendString> build_options;
    CV_LOG_INFO(NULL, "Build OM model from graph");
    ACL_CHECK_GRAPH_RET(aclgrphBuildModel(graph, build_options, om_model));


#if 0
    // (optional). Dump model
    aclgrphDumpGraph(graph, "test", 4);
    // (optional). Save model
    aclgrphSaveModel("test", om_model);
#endif

    model.clear();
    model.resize(om_model.length);
    std::memcpy(reinterpret_cast<void*>(model.data()),
                reinterpret_cast<void*>(om_model.data.get()),
                om_model.length);
}

void CannNet::init()
{
    ACL_CHECK_RET(aclrtSetDevice(device_id));
    ACL_CHECK_RET(aclrtCreateContext(&context, device_id));
    // ACL_CHECK_RET(aclrtCreateStream(&stream));

    initAclGraphBuilder();
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

int CannNet::getOutputIndexByName(const std::string& name)
{
    int idx;
    for (idx = 0; idx < output_names.size(); idx++)
        if (output_names[idx] == name)
            return idx;

    return -1;
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
