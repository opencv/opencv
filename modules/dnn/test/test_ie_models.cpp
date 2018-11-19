// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "test_precomp.hpp"

#ifdef HAVE_INF_ENGINE
#include <opencv2/core/utils/filesystem.hpp>

#include <inference_engine.hpp>
#include <ie_icnn_network.hpp>
#include <ie_extension.h>

namespace opencv_test { namespace {

static void initDLDTDataPath()
{
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
        const char* dldtTestDataPath = getenv("INTEL_CVSDK_DIR");
        if (dldtTestDataPath)
            cvtest::addDataSearchPath(cv::utils::fs::join(dldtTestDataPath, "deployment_tools"));
        initialized = true;
    }
#endif
}

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

static inline void genData(const std::vector<size_t>& dims, Mat& m, Blob::Ptr& dataPtr)
{
    std::vector<int> reversedDims(dims.begin(), dims.end());
    std::reverse(reversedDims.begin(), reversedDims.end());

    m.create(reversedDims, CV_32F);
    randu(m, -1, 1);

    dataPtr = make_shared_blob<float>(Precision::FP32, dims, (float*)m.data);
}

void runIE(Target target, const std::string& xmlPath, const std::string& binPath,
           std::map<std::string, cv::Mat>& inputsMap, std::map<std::string, cv::Mat>& outputsMap)
{
    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);

    CNNNetwork net = reader.getNetwork();

    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;
    ExecutableNetwork netExec;
    InferRequest infRequest;
    TargetDevice targetDevice;
    switch (target)
    {
        case DNN_TARGET_CPU:
            targetDevice = TargetDevice::eCPU;
            break;
        case DNN_TARGET_OPENCL:
        case DNN_TARGET_OPENCL_FP16:
            targetDevice = TargetDevice::eGPU;
            break;
        case DNN_TARGET_MYRIAD:
            targetDevice = TargetDevice::eMYRIAD;
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };

    try
    {
        enginePtr = PluginDispatcher({""}).getSuitablePlugin(targetDevice);

        if (targetDevice == TargetDevice::eCPU)
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
#else
                std::string libName = "libcpu_extension" + suffixes[i] + ".so";
#endif  // _WIN32
                try
                {
                    IExtensionPtr extension = make_so_pointer<IExtension>(libName);
                    enginePtr->AddExtension(extension, 0);
                    break;
                }
                catch(...) {}
            }
            // Some of networks can work without a library of extra layers.
        }
        plugin = InferencePlugin(enginePtr);

        netExec = plugin.LoadNetwork(net, {});
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
        genData(it.second->getDims(), inputsMap[it.first], inputBlobs[it.first]);
    }
    infRequest.SetInput(inputBlobs);

    // Fill output blobs.
    outputsMap.clear();
    BlobMap outputBlobs;
    for (auto& it : net.getOutputsInfo())
    {
        genData(it.second->dims, outputsMap[it.first], outputBlobs[it.first]);
    }
    infRequest.SetOutput(outputBlobs);

    infRequest.Infer();
}

std::vector<String> getOutputsNames(const Net& net)
{
    std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void runCV(Target target, const std::string& xmlPath, const std::string& binPath,
           const std::map<std::string, cv::Mat>& inputsMap,
           std::map<std::string, cv::Mat>& outputsMap)
{
    Net net = readNet(xmlPath, binPath);
    for (auto& it : inputsMap)
        net.setInput(it.second, it.first);
    net.setPreferableTarget(target);

    std::vector<String> outNames = getOutputsNames(net);
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    outputsMap.clear();
    EXPECT_EQ(outs.size(), outNames.size());
    for (int i = 0; i < outs.size(); ++i)
    {
        EXPECT_TRUE(outputsMap.insert({outNames[i], outs[i]}).second);
    }
}

typedef TestWithParam<tuple<Target, String> > DNNTestOpenVINO;
TEST_P(DNNTestOpenVINO, models)
{
    Target target = (dnn::Target)(int)get<0>(GetParam());
    std::string modelName = get<1>(GetParam());

#ifdef INF_ENGINE_RELEASE
#if INF_ENGINE_RELEASE <= 2018030000
    if (target == DNN_TARGET_MYRIAD && (modelName == "landmarks-regression-retail-0001" ||
                                        modelName == "semantic-segmentation-adas-0001" ||
                                        modelName == "face-reidentification-retail-0001"))
        throw SkipTestException("");
#elif INF_ENGINE_RELEASE == 2018040000
    if (modelName == "single-image-super-resolution-0034" ||
        (target == DNN_TARGET_MYRIAD && (modelName == "license-plate-recognition-barrier-0001" ||
                                         modelName == "landmarks-regression-retail-0009" ||
                                          modelName == "semantic-segmentation-adas-0001")))
        throw SkipTestException("");
#endif
#endif

    std::string precision = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? "FP16" : "FP32";
    std::string prefix = utils::fs::join("intel_models",
                         utils::fs::join(modelName,
                         utils::fs::join(precision, modelName)));
    std::string xmlPath = findDataFile(prefix + ".xml");
    std::string binPath = findDataFile(prefix + ".bin");

    std::map<std::string, cv::Mat> inputsMap;
    std::map<std::string, cv::Mat> ieOutputsMap, cvOutputsMap;
    // Single Myriad device cannot be shared across multiple processes.
    resetMyriadDevice();
    runIE(target, xmlPath, binPath, inputsMap, ieOutputsMap);
    runCV(target, xmlPath, binPath, inputsMap, cvOutputsMap);

    EXPECT_EQ(ieOutputsMap.size(), cvOutputsMap.size());
    for (auto& srcIt : ieOutputsMap)
    {
        auto dstIt = cvOutputsMap.find(srcIt.first);
        CV_Assert(dstIt != cvOutputsMap.end());
        double normInf = cvtest::norm(srcIt.second, dstIt->second, cv::NORM_INF);
        EXPECT_EQ(normInf, 0);
    }
}

static testing::internal::ParamGenerator<String> intelModels()
{
    initDLDTDataPath();
    std::vector<String> modelsNames;

    std::string path;
    try
    {
        path = findDataDirectory("intel_models", false);
    }
    catch (...)
    {
        std::cerr << "ERROR: Can't find OpenVINO models. Check INTEL_CVSDK_DIR environment variable (run setup.sh)" << std::endl;
        return ValuesIn(modelsNames);  // empty list
    }

    cv::utils::fs::glob_relative(path, "", modelsNames, false, true);

    modelsNames.erase(
        std::remove_if(modelsNames.begin(), modelsNames.end(),
                       [&](const String& dir){ return !utils::fs::isDirectory(utils::fs::join(path, dir)); }),
        modelsNames.end()
    );
    CV_Assert(!modelsNames.empty());

    return ValuesIn(modelsNames);
}

static testing::internal::ParamGenerator<Target> dnnDLIETargets()
{
    std::vector<Target> targets;
    targets.push_back(DNN_TARGET_CPU);
#ifdef HAVE_OPENCL
    if (cv::ocl::useOpenCL() && ocl::Device::getDefault().isIntel())
    {
        targets.push_back(DNN_TARGET_OPENCL);
        targets.push_back(DNN_TARGET_OPENCL_FP16);
    }
#endif
    if (checkMyriadTarget())
        targets.push_back(DNN_TARGET_MYRIAD);
    return testing::ValuesIn(targets);
}

INSTANTIATE_TEST_CASE_P(/**/, DNNTestOpenVINO, Combine(
    dnnDLIETargets(), intelModels()
));

}}
#endif  // HAVE_INF_ENGINE
