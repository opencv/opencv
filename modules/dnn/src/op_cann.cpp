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

bool CannGraph::loadFromONNX(const String& modelFile)
{
    using namespace ge;

    std::map<AscendString, AscendString> parse_options; // TODO: support custom parse_options from file
    CV_LOG_INFO(NULL, "Load graph from " << modelFile << " with ge::aclgrphParseONNX");
    ACL_CHECK_GRAPH_RET(aclgrphParseONNX(modelFile.c_str(), parse_options, graph));

    return true;
}

bool CannGraph::loadFromONNXFromMem(const char* buffer, size_t length)
{
    using namespace ge;

    std::map<AscendString, AscendString> parse_options; // // TODO: support custom parse_options from file
    CV_LOG_INFO(NULL, "Load graph from buffer with ge::aclgrphParseONNXFromMem");
    ACL_CHECK_GRAPH_RET(aclgrphParseONNXFromMem(buffer, length, parse_options, graph));

    return true;
}

bool CannGraph::loadFromTensorFlow(const String& modelFile)
{
    using namespace ge;

    std::map<AscendString, AscendString> parse_options; // TODO: support custom parse_options from file
    CV_LOG_INFO(NULL, "Load graph from " << modelFile << " with ge::aclgrphParseTensorFlow");
    ACL_CHECK_GRAPH_RET(aclgrphParseTensorFlow(modelFile.c_str(), parse_options, graph));

    return true;
}

bool CannGraph::loadFromCaffe(const String& modelFile, const String& weightFile)
{
    using namespace ge;

    std::map<AscendString, AscendString> parse_options; // TODO: support custom parse_options from file
    CV_LOG_INFO(NULL, "Load graph from " << modelFile << " with ge::aclgrphParseTensorFlow");
    ACL_CHECK_GRAPH_RET(aclgrphParseCaffe(modelFile.c_str(), weightFile.c_str(), parse_options, graph));

    return true;
}

void CannClient::init(int deviceId)
{
    device_id = deviceId;

    ACL_CHECK_RET(aclrtSetDevice(device_id));
    ACL_CHECK_RET(aclrtCreateContext(&context, device_id));
    // ACL_CHECK_RET(aclrtCreateStream(&stream));

    initAclGraphBuilder();
}

CannClient::~CannClient()
{
    CV_LOG_INFO(NULL, "In ~CannClient, inputs = " << inputs << ", outputs = " << outputs);
    if (!model_desc)
    {
        CV_LOG_INFO(NULL, "[Failed] Tried to deconstruct CannClient but model is not loaded");
        return;
    }
    // free datasets: inputs, outputs
    if (inputs)
    {
        CV_LOG_INFO(NULL, "In ~CannClient: destroy inputs");
        destroyDataset(&inputs);
    }
    if (outputs)
    {
        CV_LOG_INFO(NULL, "In ~CannClient: destroy outputs");
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

void CannClient::finalize()
{
    finalizeAclGraphBuilder();
}

bool CannClient::empty() const
{
    return (model_desc == nullptr);
}

void CannClient::loadModel()
{
    CV_LOG_INFO(NULL, "Load model to NPU memory");
    ACL_CHECK_RET(aclmdlLoadFromMem(reinterpret_cast<const void*>(model.data()), model.size(), &model_id));

    CV_LOG_INFO(NULL, "Create model description");
    model_desc = aclmdlCreateDesc();
    ACL_CHECK_RET(aclmdlGetDesc(model_desc, model_id));

    createInputDataset();
    createOutputDataset();
}

void CannClient::setInput(const Mat& input, const String& name)
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

void CannClient::forward()
{
    ACL_CHECK_RET(aclrtSetCurrentContext(context));
    ACL_CHECK_RET(aclmdlExecute(model_id, inputs, outputs));
    CV_LOG_INFO(NULL, "[Success] Finished forward");
}

void CannClient::fetchOutput(Mat& output, const String& name)
{
    size_t idx;
    if (name.empty())
        idx = 0;
    else
        ACL_CHECK_RET(aclmdlGetOutputIndexByName(model_desc, name.c_str(), &idx));

    fetchOutput(output, idx);
}

void CannClient::fetchOutput(Mat& output, const size_t idx)
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

size_t CannClient::getOutputNum() const
{
    return aclmdlGetNumOutputs(model_desc);
}

void CannClient::buildModelFromGraph(ge::Graph& graph)
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

void CannClient::createInputDataset()
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

void CannClient::createOutputDataset()
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

void CannClient::destroyDataset(aclmdlDataset** dataset)
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
