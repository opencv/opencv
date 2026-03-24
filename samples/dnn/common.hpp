#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/dnn.hpp>
#include<iostream>
using namespace cv;
using namespace cv::dnn;

std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key = ' ', std::string defaultVal = "");

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile, const std::string& prefix);

std::string findFile(const std::string& filename);

std::string findModel(const std::string& filename, const std::string& sha1);

std::vector<std::string> findAliases(std::string& zooFile, const std::string& sampleType);

inline int getBackendID(const String& backend) {
    std::map<String, int> backendIDs = {
        {"default", cv::dnn::DNN_BACKEND_DEFAULT},
        {"openvino", cv::dnn::DNN_BACKEND_INFERENCE_ENGINE},
        {"opencv", cv::dnn::DNN_BACKEND_OPENCV},
        {"vkcom", cv::dnn::DNN_BACKEND_VKCOM},
        {"cuda", cv::dnn::DNN_BACKEND_CUDA},
        {"webnn", cv::dnn::DNN_BACKEND_WEBNN}
    };
    if(backendIDs.find(backend) != backendIDs.end()){
        return backendIDs[backend];
    }else {
        throw std::invalid_argument("Invalid backend name: " + backend);
    }
}

inline int getTargetID(const String& target) {
    std::map<String, int> targetIDs = {
        {"cpu", cv::dnn::DNN_TARGET_CPU},
        {"opencl", cv::dnn::DNN_TARGET_OPENCL},
        {"opencl_fp16", cv::dnn::DNN_TARGET_OPENCL_FP16},
        {"vpu", cv::dnn::DNN_TARGET_MYRIAD},
        {"vulkan", cv::dnn::DNN_TARGET_VULKAN},
        {"cuda", cv::dnn::DNN_TARGET_CUDA},
        {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
    };
    if(targetIDs.find(target) != targetIDs.end()){
        return targetIDs[target];
    }else {
        throw std::invalid_argument("Invalid target name: " + target);
    }
}

inline int getEngineID(const String& engine) {
    std::map<String, int> engineIDs = {
        {"auto", cv::dnn::ENGINE_AUTO},
        {"classic", cv::dnn::ENGINE_CLASSIC},
        {"new", cv::dnn::ENGINE_NEW},
        {"ort", cv::dnn::ENGINE_ORT}
    };
    if(engineIDs.find(engine) != engineIDs.end()){
        return engineIDs[engine];
    }else {
        throw std::invalid_argument("Invalid engine name: " + engine);
    }
}

inline String getBackendName(int backendId) {
    switch (backendId) {
        case cv::dnn::DNN_BACKEND_DEFAULT: return "default";
        case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE: return "openvino";
        case cv::dnn::DNN_BACKEND_OPENCV: return "opencv";
        case cv::dnn::DNN_BACKEND_VKCOM: return "vkcom";
        case cv::dnn::DNN_BACKEND_CUDA: return "cuda";
        case cv::dnn::DNN_BACKEND_WEBNN: return "webnn";
        default: return "unknown";
    }
}

inline String getTargetName(int targetId) {
    switch (targetId) {
        case cv::dnn::DNN_TARGET_CPU: return "cpu";
        case cv::dnn::DNN_TARGET_OPENCL: return "opencl";
        case cv::dnn::DNN_TARGET_OPENCL_FP16: return "opencl_fp16";
        case cv::dnn::DNN_TARGET_MYRIAD: return "vpu";
        case cv::dnn::DNN_TARGET_VULKAN: return "vulkan";
        case cv::dnn::DNN_TARGET_CUDA: return "cuda";
        case cv::dnn::DNN_TARGET_CUDA_FP16: return "cuda_fp16";
        default: return "unknown";
    }
}

inline String getEngineName(int engineId) {
    switch (engineId) {
        case cv::dnn::ENGINE_CLASSIC: return "classic";
        case cv::dnn::ENGINE_NEW: return "new";
        case cv::dnn::ENGINE_AUTO: return "auto";
        case cv::dnn::ENGINE_ORT: return "ort";
        default: return "unknown";
    }
}

const std::string engine_keys = cv::format(
    "{ engine | auto | Choose one of DNN engines: "
                    "auto: automatically (by default), "
                    "classic: classic DNN engine, "
                    "new: new graph-based DNN engine, "
                    "ort: ONNX Runtime }");

std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key, std::string defaultVal)
{
    if (!modelName.empty())
    {
        FileStorage fs(zooFile, FileStorage::READ);
        if (fs.isOpened())
        {
            FileNode node = fs[modelName];
            if (!node.empty())
            {
                FileNode value = node[argName];
                if (argName.find("sha1") != std::string::npos) {
                    std::string prefix = argName.substr(0, argName.find("sha1"));
                    if (prefix == "config_"){
                        value = node[prefix+"load_info"]["sha1"];
                    }
                    else{
                        value = node[prefix+"load_info"][argName];
                    }
                }
                if (argName.find("download_sha") != std::string::npos) {
                    std::string prefix = argName.substr(0, argName.find("download_sha"));
                    value = node[prefix+"load_info"][argName];
                }
                if (!value.empty())
                {
                    if (value.isReal())
                        defaultVal = format("%f", (float)value);
                    else if (value.isString())
                        defaultVal = (std::string)value;
                    else if (value.isInt())
                        defaultVal = format("%d", (int)value);
                    else if (value.isSeq())
                    {
                        for (size_t i = 0; i < value.size(); ++i)
                        {
                            FileNode v = value[(int)i];
                            if (v.isInt())
                                defaultVal += format("%d ", (int)v);
                            else if (v.isReal())
                                defaultVal += format("%f ", (float)v);
                            else
                              CV_Error(Error::StsNotImplemented, "Unexpected value format");
                        }
                    }
                    else
                        CV_Error(Error::StsNotImplemented, "Unexpected field format");
                }
            }
        }
    }
    return "{ " + argName + " " + key + " | " + defaultVal + " | " + help + " }";
}

std::string findModel(const std::string& filename, const std::string& sha1)
{
    if (filename.empty() || utils::fs::exists(filename))
        return filename;

    if(!getenv("OPENCV_DOWNLOAD_CACHE_DIR")){
        std::cout<< "[WARN] Please specify a path to model download directory in OPENCV_DOWNLOAD_CACHE_DIR environment variable"<<std::endl;
        return findFile(filename);
    }
    else{
        std::string modelPath = utils::fs::join(getenv("OPENCV_DOWNLOAD_CACHE_DIR"), utils::fs::join(sha1, filename));
        if (utils::fs::exists(modelPath))
            return modelPath;
        modelPath = utils::fs::join(getenv("OPENCV_DOWNLOAD_CACHE_DIR"),filename);
        if (utils::fs::exists(modelPath))
            return modelPath;
    }

    std::cout << "File " + filename + " not found! "
              << "Please specify a path to model download directory in OPENCV_DOWNLOAD_CACHE_DIR "
              << "environment variable or pass a full path to " + filename
              << std::endl;
    std::exit(1);
}

std::string findFile(const std::string& filename)
{
    if (filename.empty() || utils::fs::exists(filename))
        return filename;

    if(!getenv("OPENCV_SAMPLES_DATA_PATH")){
        std::cout<< "[WARN] Please specify a path to opencv/samples/data in OPENCV_SAMPLES_DATA_PATH environment variable"<<std::endl;
    }
    else{
        std::string samplePath = utils::fs::join(getenv("OPENCV_SAMPLES_DATA_PATH"), filename);
        if (utils::fs::exists(samplePath))
            return samplePath;
    }
    const char* extraPaths[] = {getenv("OPENCV_SAMPLES_DATA_PATH"),
                                getenv("OPENCV_DNN_TEST_DATA_PATH"),
                                getenv("OPENCV_TEST_DATA_PATH")};
    for (int i = 0; i < 3; ++i)
    {
        if (extraPaths[i] == NULL)
            continue;
        std::string absPath = utils::fs::join(extraPaths[i], utils::fs::join("dnn", filename));
        if (utils::fs::exists(absPath))
            return absPath;
    }
    std::cout << "File " + filename + " not found! "
              << "Please specify the path to /opencv/samples/data in the OPENCV_SAMPLES_DATA_PATH environment variable, "
              << "or specify the path to opencv_extra/testdata in the OPENCV_DNN_TEST_DATA_PATH environment variable, "
              << "or specify the path to the model download cache directory in the OPENCV_DOWNLOAD_CACHE_DIR environment variable, "
              << "or pass the full path to " + filename + "."
              << std::endl;
    std::exit(1);
}

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile, const std::string& prefix="")
{
    return genArgument(prefix + "model", "Path to a binary file of model contains trained weights. "
                                "It could be a file with extensions .caffemodel (Caffe), "
                                ".pb (TensorFlow), .weights (Darknet), .bin (OpenVINO).",
                       modelName, zooFile, 'm') +
           genArgument(prefix + "config", "Path to a text file of model contains network configuration. "
                                 "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO).",
                       modelName, zooFile, 'c') +
           genArgument(prefix + "mean", "Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.",
                       modelName, zooFile) +
           genArgument(prefix + "std", "Preprocess input image by dividing on a standard deviation.",
                       modelName, zooFile) +
           genArgument(prefix + "scale", "Preprocess input image by multiplying on a scale factor.",
                       modelName, zooFile, ' ', "1.0") +
           genArgument(prefix + "width", "Preprocess input image by resizing to a specific width.",
                       modelName, zooFile, ' ', "-1") +
           genArgument(prefix + "height", "Preprocess input image by resizing to a specific height.",
                       modelName, zooFile, ' ', "-1") +
           genArgument(prefix + "rgb", "Indicate that model works with RGB input images instead BGR ones.",
                       modelName, zooFile)+
           genArgument(prefix + "labels", "Path to a text file with names of classes to label detected objects.",
                       modelName, zooFile)+
           genArgument(prefix + "postprocessing", "Indicate the postprocessing type of model i.e. yolov8, yolonas, etc.",
                       modelName, zooFile)+
           genArgument(prefix + "sha1", "Optional path to hashsum of downloaded model to be loaded from models.yml",
                       modelName, zooFile)+
           genArgument(prefix + "config_sha1", "Optional path to hashsum of downloaded config to be loaded from models.yml",
                       modelName, zooFile)+
           genArgument(prefix + "download_sha", "Optional path to hashsum of downloaded model to be loaded from models.yml",
                       modelName, zooFile);
}

std::vector<std::string> findAliases(std::string& zooFile, const std::string& sampleType) {
    std::vector<std::string> aliases;

    zooFile = findFile(zooFile);

    cv::FileStorage fs(zooFile, cv::FileStorage::READ);

    cv::FileNode root = fs.root();
    for (const auto& node : root) {
        std::string alias = node.name();
        cv::FileNode sampleNode = node["sample"];

        if (!sampleNode.empty() && sampleNode.isString()) {
            std::string sampleValue = (std::string)sampleNode;
            if (sampleValue == sampleType) {
                aliases.push_back(alias);
            }
        }
    }

    return aliases;
}