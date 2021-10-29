// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "test_precomp.hpp"

#ifdef HAVE_INF_ENGINE
#include <opencv2/core/utils/filesystem.hpp>


//
// Synchronize headers include statements with src/op_inf_engine.hpp
//
//#define INFERENCE_ENGINE_DEPRECATED  // turn off deprecation warnings from IE
//there is no way to suppress warnings from IE only at this moment, so we are forced to suppress warnings globally
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#include <inference_engine.hpp>
#include <ie_icnn_network.hpp>
#include <ie_extension.h>

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif


namespace opencv_test { namespace {

static void initDLDTDataPath()
{
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
#if INF_ENGINE_RELEASE <= 2018050000
        const char* dldtTestDataPath = getenv("INTEL_CVSDK_DIR");
        if (dldtTestDataPath)
            cvtest::addDataSearchPath(dldtTestDataPath);
#else
        const char* omzDataPath = getenv("OPENCV_OPEN_MODEL_ZOO_DATA_PATH");
        if (omzDataPath)
            cvtest::addDataSearchPath(omzDataPath);
        const char* dnnDataPath = getenv("OPENCV_DNN_TEST_DATA_PATH");
        if (dnnDataPath)
            cvtest::addDataSearchPath(std::string(dnnDataPath) + "/omz_intel_models");
#endif
        initialized = true;
    }
#endif
}

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

struct OpenVINOModelTestCaseInfo
{
    const char* modelPathFP32;
    const char* modelPathFP16;
};

static const std::map<std::string, OpenVINOModelTestCaseInfo>& getOpenVINOTestModels()
{
    static std::map<std::string, OpenVINOModelTestCaseInfo> g_models {
#if INF_ENGINE_RELEASE >= 2018050000 && \
    INF_ENGINE_RELEASE <= 2020999999  // don't use IRv5 models with 2020.1+
        // layout is defined by open_model_zoo/model_downloader
        // Downloaded using these parameters for Open Model Zoo downloader (2019R1):
        // ./downloader.py -o ${OPENCV_DNN_TEST_DATA_PATH}/omz_intel_models --cache_dir ${OPENCV_DNN_TEST_DATA_PATH}/.omz_cache/ \
        //     --name face-person-detection-retail-0002,face-person-detection-retail-0002-fp16,age-gender-recognition-retail-0013,age-gender-recognition-retail-0013-fp16,head-pose-estimation-adas-0001,head-pose-estimation-adas-0001-fp16,person-detection-retail-0002,person-detection-retail-0002-fp16,vehicle-detection-adas-0002,vehicle-detection-adas-0002-fp16
        { "age-gender-recognition-retail-0013", {
            "Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013",
            "Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013-fp16"
        }},
        { "face-person-detection-retail-0002", {
            "Retail/object_detection/face_pedestrian/rmnet-ssssd-2heads/0002/dldt/face-person-detection-retail-0002",
            "Retail/object_detection/face_pedestrian/rmnet-ssssd-2heads/0002/dldt/face-person-detection-retail-0002-fp16"
        }},
        { "head-pose-estimation-adas-0001", {
            "Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001",
            "Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16"
        }},
        { "person-detection-retail-0002", {
            "Retail/object_detection/pedestrian/hypernet-rfcn/0026/dldt/person-detection-retail-0002",
            "Retail/object_detection/pedestrian/hypernet-rfcn/0026/dldt/person-detection-retail-0002-fp16"
        }},
        { "vehicle-detection-adas-0002", {
            "Transportation/object_detection/vehicle/mobilenet-reduced-ssd/dldt/vehicle-detection-adas-0002",
            "Transportation/object_detection/vehicle/mobilenet-reduced-ssd/dldt/vehicle-detection-adas-0002-fp16"
        }},
#endif
#if INF_ENGINE_RELEASE >= 2020010000
        // Downloaded using these parameters for Open Model Zoo downloader (2020.1):
        // ./downloader.py -o ${OPENCV_DNN_TEST_DATA_PATH}/omz_intel_models --cache_dir ${OPENCV_DNN_TEST_DATA_PATH}/.omz_cache/ \
        //     --name person-detection-retail-0013,age-gender-recognition-retail-0013
        { "person-detection-retail-0013", {  // IRv10
            "intel/person-detection-retail-0013/FP32/person-detection-retail-0013",
            "intel/person-detection-retail-0013/FP16/person-detection-retail-0013"
        }},
        { "age-gender-recognition-retail-0013", {
            "intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013",
            "intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013"
        }},
#endif
#if INF_ENGINE_RELEASE >= 2021020000
        // OMZ: 2020.2
        { "face-detection-0105", {
            "intel/face-detection-0105/FP32/face-detection-0105",
            "intel/face-detection-0105/FP16/face-detection-0105"
        }},
        { "face-detection-0106", {
            "intel/face-detection-0106/FP32/face-detection-0106",
            "intel/face-detection-0106/FP16/face-detection-0106"
        }},
#endif
#if INF_ENGINE_RELEASE >= 2021040000
        // OMZ: 2021.4
        { "person-vehicle-bike-detection-2004", {
            "intel/person-vehicle-bike-detection-2004/FP32/person-vehicle-bike-detection-2004",
            "intel/person-vehicle-bike-detection-2004/FP16/person-vehicle-bike-detection-2004"
            //"intel/person-vehicle-bike-detection-2004/FP16-INT8/person-vehicle-bike-detection-2004"
        }},
#endif
    };

    return g_models;
}

static const std::vector<std::string> getOpenVINOTestModelsList()
{
    std::vector<std::string> result;
    const std::map<std::string, OpenVINOModelTestCaseInfo>& models = getOpenVINOTestModels();
    for (const auto& it : models)
        result.push_back(it.first);
    return result;
}

inline static std::string getOpenVINOModel(const std::string &modelName, bool isFP16)
{
    const std::map<std::string, OpenVINOModelTestCaseInfo>& models = getOpenVINOTestModels();
    const auto it = models.find(modelName);
    if (it != models.end())
    {
        OpenVINOModelTestCaseInfo modelInfo = it->second;
        if (isFP16 && modelInfo.modelPathFP16)
            return std::string(modelInfo.modelPathFP16);
        else if (!isFP16 && modelInfo.modelPathFP32)
            return std::string(modelInfo.modelPathFP32);
    }
    return std::string();
}

static inline void genData(const InferenceEngine::TensorDesc& desc, Mat& m, Blob::Ptr& dataPtr)
{
    const std::vector<size_t>& dims = desc.getDims();
    if (desc.getPrecision() == InferenceEngine::Precision::FP32)
    {
        m.create(std::vector<int>(dims.begin(), dims.end()), CV_32F);
        randu(m, -1, 1);
        dataPtr = make_shared_blob<float>(desc, (float*)m.data);
    }
    else if (desc.getPrecision() == InferenceEngine::Precision::I32)
    {
        m.create(std::vector<int>(dims.begin(), dims.end()), CV_32S);
        randu(m, -100, 100);
        dataPtr = make_shared_blob<int>(desc, (int*)m.data);
    }
    else
    {
        FAIL() << "Unsupported precision: " << desc.getPrecision();
    }
}

void runIE(Target target, const std::string& xmlPath, const std::string& binPath,
           std::map<std::string, cv::Mat>& inputsMap, std::map<std::string, cv::Mat>& outputsMap)
{
    SCOPED_TRACE("runIE");

    std::string device_name;

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GT(2019010000)
    Core ie;
#else
    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;
#endif

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GT(2019030000)
    CNNNetwork net = ie.ReadNetwork(xmlPath, binPath);
#else
    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);

    CNNNetwork net = reader.getNetwork();
#endif

    ExecutableNetwork netExec;
    InferRequest infRequest;

    try
    {
        switch (target)
        {
            case DNN_TARGET_CPU:
                device_name = "CPU";
                break;
            case DNN_TARGET_OPENCL:
            case DNN_TARGET_OPENCL_FP16:
                device_name = "GPU";
                break;
            case DNN_TARGET_MYRIAD:
                device_name = "MYRIAD";
                break;
            case DNN_TARGET_FPGA:
                device_name = "FPGA";
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Unknown target");
        };

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2019010000)
        auto dispatcher = InferenceEngine::PluginDispatcher({""});
        enginePtr = dispatcher.getPluginByDevice(device_name);
#endif
        if (target == DNN_TARGET_CPU || target == DNN_TARGET_FPGA)
        {
            std::string suffixes[] = {"_avx2", "_sse4", ""};
            bool haveFeature[] = {
                checkHardwareSupport(CPU_AVX2),
                checkHardwareSupport(CPU_SSE4_2),
                true
            };
            for (int i = 0; i < 3; ++i)
            {
                if (!haveFeature[i])
                    continue;
#ifdef _WIN32
                std::string libName = "cpu_extension" + suffixes[i] + ".dll";
#elif defined(__APPLE__)
                std::string libName = "libcpu_extension" + suffixes[i] + ".dylib";
#else
                std::string libName = "libcpu_extension" + suffixes[i] + ".so";
#endif  // _WIN32
                try
                {
                    IExtensionPtr extension = make_so_pointer<IExtension>(libName);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GT(2019010000)
                    ie.AddExtension(extension, device_name);
#else
                    enginePtr->AddExtension(extension, 0);
#endif
                    break;
                }
                catch(...) {}
            }
            // Some of networks can work without a library of extra layers.
        }
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GT(2019010000)
        netExec = ie.LoadNetwork(net, device_name);
#else
        plugin = InferencePlugin(enginePtr);
        netExec = plugin.LoadNetwork(net, {});
#endif
        infRequest = netExec.CreateInferRequest();
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }

    // Fill input blobs.
    inputsMap.clear();
    BlobMap inputBlobs;
    for (auto& it : net.getInputsInfo())
    {
        const InferenceEngine::TensorDesc& desc = it.second->getTensorDesc();
        genData(desc, inputsMap[it.first], inputBlobs[it.first]);
        if (cvtest::debugLevel > 0)
        {
            const std::vector<size_t>& dims = desc.getDims();
            std::cout << "Input: '" << it.first << "' precison=" << desc.getPrecision() << " dims=" << dims.size() << " [";
            for (auto d : dims)
                std::cout << " " << d;
            std::cout << "]  ocv_mat=" << inputsMap[it.first].size << " of " << typeToString(inputsMap[it.first].type()) << std::endl;
        }
    }
    infRequest.SetInput(inputBlobs);

    // Fill output blobs.
    outputsMap.clear();
    BlobMap outputBlobs;
    for (auto& it : net.getOutputsInfo())
    {
        const InferenceEngine::TensorDesc& desc = it.second->getTensorDesc();
        genData(desc, outputsMap[it.first], outputBlobs[it.first]);
        if (cvtest::debugLevel > 0)
        {
            const std::vector<size_t>& dims = desc.getDims();
            std::cout << "Output: '" << it.first << "' precison=" << desc.getPrecision() << " dims=" << dims.size() << " [";
            for (auto d : dims)
                std::cout << " " << d;
            std::cout << "]  ocv_mat=" << outputsMap[it.first].size << " of " << typeToString(outputsMap[it.first].type()) << std::endl;
        }
    }
    infRequest.SetOutput(outputBlobs);

    infRequest.Infer();
}

void runCV(Backend backendId, Target targetId, const std::string& xmlPath, const std::string& binPath,
           const std::map<std::string, cv::Mat>& inputsMap,
           std::map<std::string, cv::Mat>& outputsMap)
{
    SCOPED_TRACE("runOCV");

    Net net = readNet(xmlPath, binPath);
    for (auto& it : inputsMap)
        net.setInput(it.second, it.first);

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    if (cvtest::debugLevel > 0)
    {
        std::cout << "OpenCV output names: " << outNames.size() << std::endl;
        for (auto name : outNames)
            std::cout << "- " << name << std::endl;
    }
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    outputsMap.clear();
    EXPECT_EQ(outs.size(), outNames.size());
    for (int i = 0; i < outs.size(); ++i)
    {
        EXPECT_TRUE(outputsMap.insert({outNames[i], outs[i]}).second);
    }
}

typedef TestWithParam<tuple< tuple<Backend, Target>, std::string> > DNNTestOpenVINO;
TEST_P(DNNTestOpenVINO, models)
{
    initDLDTDataPath();

    const Backend backendId = get<0>(get<0>(GetParam()));
    const Target targetId = get<1>(get<0>(GetParam()));
    std::string modelName = get<1>(GetParam());

    ASSERT_FALSE(backendId != DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) <<
        "Inference Engine backend is required";

#if INF_ENGINE_VER_MAJOR_GE(2021030000)
    if (targetId == DNN_TARGET_MYRIAD && (false
            || modelName == "person-detection-retail-0013"  // ncDeviceOpen:1013 Failed to find booted device after boot
            || modelName == "age-gender-recognition-retail-0013"  // ncDeviceOpen:1013 Failed to find booted device after boot
            || modelName == "face-detection-0105"  // get_element_type() must be called on a node with exactly one output
            || modelName == "face-detection-0106"  // get_element_type() must be called on a node with exactly one output
            || modelName == "person-vehicle-bike-detection-2004"  // 2021.4+: ncDeviceOpen:1013 Failed to find booted device after boot
        )
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (targetId == DNN_TARGET_OPENCL && (false
            || modelName == "face-detection-0106"  // Operation: 2278 of type ExperimentalDetectronPriorGridGenerator(op::v6) is not supported
        )
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (targetId == DNN_TARGET_OPENCL_FP16 && (false
            || modelName == "face-detection-0106"  // Operation: 2278 of type ExperimentalDetectronPriorGridGenerator(op::v6) is not supported
        )
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

#if INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (targetId == DNN_TARGET_MYRIAD && backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (modelName == "person-detection-retail-0013")  // IRv10
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

#if INF_ENGINE_VER_MAJOR_EQ(2020040000)
    if (targetId == DNN_TARGET_MYRIAD && modelName == "person-detection-retail-0002")  // IRv5, OpenVINO 2020.4 regression
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        setInferenceEngineBackendType(CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API);
    else if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        setInferenceEngineBackendType(CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
    else
        FAIL() << "Unknown backendId";

    bool isFP16 = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD);

    const std::string modelPath = getOpenVINOModel(modelName, isFP16);
    ASSERT_FALSE(modelPath.empty()) << modelName;

    std::string xmlPath = findDataFile(modelPath + ".xml", false);
    std::string binPath = findDataFile(modelPath + ".bin", false);

    std::map<std::string, cv::Mat> inputsMap;
    std::map<std::string, cv::Mat> ieOutputsMap, cvOutputsMap;
    // Single Myriad device cannot be shared across multiple processes.
    if (targetId == DNN_TARGET_MYRIAD)
        resetMyriadDevice();
    if (targetId == DNN_TARGET_HDDL)
        releaseHDDLPlugin();
    EXPECT_NO_THROW(runIE(targetId, xmlPath, binPath, inputsMap, ieOutputsMap)) << "runIE";
    if (targetId == DNN_TARGET_MYRIAD)
        resetMyriadDevice();
    EXPECT_NO_THROW(runCV(backendId, targetId, xmlPath, binPath, inputsMap, cvOutputsMap)) << "runCV";

    double eps = 0;
#if INF_ENGINE_VER_MAJOR_GE(2020010000)
    if (targetId == DNN_TARGET_CPU && checkHardwareSupport(CV_CPU_AVX_512F))
        eps = 1e-5;
#endif
#if INF_ENGINE_VER_MAJOR_GE(2021030000)
    if (targetId == DNN_TARGET_CPU && modelName == "face-detection-0105")
        eps = 2e-4;
#endif
#if INF_ENGINE_VER_MAJOR_GE(2021040000)
    if (targetId == DNN_TARGET_CPU && modelName == "person-vehicle-bike-detection-2004")
        eps = 1e-6;
#endif

    EXPECT_EQ(ieOutputsMap.size(), cvOutputsMap.size());
    for (auto& srcIt : ieOutputsMap)
    {
        auto dstIt = cvOutputsMap.find(srcIt.first);
        CV_Assert(dstIt != cvOutputsMap.end());
        double normInf = cvtest::norm(srcIt.second, dstIt->second, cv::NORM_INF);
        EXPECT_LE(normInf, eps) << "output=" << srcIt.first;
    }
}


INSTANTIATE_TEST_CASE_P(/**/,
    DNNTestOpenVINO,
    Combine(dnnBackendsAndTargetsIE(),
            testing::ValuesIn(getOpenVINOTestModelsList())
    )
);

typedef TestWithParam<Target> DNNTestHighLevelAPI;
TEST_P(DNNTestHighLevelAPI, predict)
{
    initDLDTDataPath();

    Target target = (dnn::Target)(int)GetParam();
    bool isFP16 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD);
    const std::string modelName = "age-gender-recognition-retail-0013";
    const std::string modelPath = getOpenVINOModel(modelName, isFP16);
    ASSERT_FALSE(modelPath.empty()) << modelName;

    std::string xmlPath = findDataFile(modelPath + ".xml");
    std::string binPath = findDataFile(modelPath + ".bin");

    Model model(xmlPath, binPath);
    Mat frame = imread(findDataFile("dnn/googlenet_1.png"));
    std::vector<Mat> outs;
    model.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    model.setPreferableTarget(target);
    model.predict(frame, outs);

    Net net = readNet(xmlPath, binPath);
    Mat input = blobFromImage(frame, 1.0, Size(62, 62));
    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(target);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> refs;
    net.forward(refs, outNames);

    CV_Assert(refs.size() == outs.size());
    for (int i = 0; i < refs.size(); ++i)
        normAssert(outs[i], refs[i]);
}

INSTANTIATE_TEST_CASE_P(/**/,
    DNNTestHighLevelAPI, testing::ValuesIn(getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE))
);

}}
#endif  // HAVE_INF_ENGINE
