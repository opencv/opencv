#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;

std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key = ' ', std::string defaultVal = "");

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile);

std::string findFile(const std::string& filename);

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

std::string findFile(const std::string& filename)
{
    if (filename.empty() || utils::fs::exists(filename))
        return filename;

    const char* extraPaths[] = {getenv("OPENCV_DNN_TEST_DATA_PATH"),
                                getenv("OPENCV_TEST_DATA_PATH")};
    for (int i = 0; i < 2; ++i)
    {
        if (extraPaths[i] == NULL)
            continue;
        std::string absPath = utils::fs::join(extraPaths[i], utils::fs::join("dnn", filename));
        if (utils::fs::exists(absPath))
            return absPath;
    }
    CV_Error(Error::StsObjectNotFound, "File " + filename + " not found! "
             "Please specify a path to /opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH "
             "environment variable or pass a full path to model.");
}

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile)
{
    return genArgument("model", "Path to a binary file of model contains trained weights. "
                                "It could be a file with extensions .caffemodel (Caffe), "
                                ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO).",
                       modelName, zooFile, 'm') +
           genArgument("config", "Path to a text file of model contains network configuration. "
                                 "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO).",
                       modelName, zooFile, 'c') +
           genArgument("mean", "Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.",
                       modelName, zooFile) +
           genArgument("scale", "Preprocess input image by multiplying on a scale factor.",
                       modelName, zooFile, ' ', "1.0") +
           genArgument("width", "Preprocess input image by resizing to a specific width.",
                       modelName, zooFile, ' ', "-1") +
           genArgument("height", "Preprocess input image by resizing to a specific height.",
                       modelName, zooFile, ' ', "-1") +
           genArgument("rgb", "Indicate that model works with RGB input images instead BGR ones.",
                       modelName, zooFile);
}
