/*************************************************
USAGE:
./model_diagnostics -m <model file location>
**************************************************/
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/dnn/utils/debug_utils.hpp>

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/shape_utils.hpp>

using namespace cv;
using namespace dnn;


static
int diagnosticsErrorCallback(int /*status*/, const char* /*func_name*/,
                             const char* /*err_msg*/, const char* /*file_name*/,
                             int /*line*/, void* /*userdata*/)
{
    fflush(stdout);
    fflush(stderr);
    return 0;
}

static std::string checkFileExists(const std::string& fileName)
{
    if (fileName.empty() || utils::fs::exists(fileName))
        return fileName;

    CV_Error(Error::StsObjectNotFound, "File " + fileName + " was not found! "
         "Please, specify a full path to the file.");
}

std::string diagnosticKeys =
        "{ model m     | | Path to the model file. }"
        "{ config c    | | Path to the model configuration file. }"
        "{ framework f | | [Optional] Name of the model framework. }";

int main( int argc, const char** argv )
{
    LayerParams layerParams;
    layerParams.name = "MyName";
    layerParams.type = "Broadcast";
    Ptr<Layer> layer = cv::dnn::BroadcastLayer::create(layerParams);
    std::cout << "Created" << std::endl;

    int requiredOutputs = 3;
    std::vector<MatShape> input_shapes{{238, 1, 23}, {1, 1}, {10, 1, 38, 23}};
    std::vector<MatShape> output_shapes;
    std::vector<MatShape> internal_shapes;

    layer->getMemoryShapes(input_shapes, requiredOutputs, output_shapes, internal_shapes);
    std::cout << "Got output shapes" << std::endl;

    std::vector<Mat> inputs;
    std::vector<Mat> outputs;
    std::vector<Mat> internals;

    for (int i = 0; i < requiredOutputs; ++i)
    {
        inputs.emplace_back(input_shapes[i], CV_32FC1, 42.);
        outputs.emplace_back(output_shapes[i], CV_32FC1);
    }

    layer->finalize(inputs, outputs);
    std::cout << "Finalized" << std::endl;

    layer->forward(inputs, outputs, internals);
    std::cout << "Forwarded" << std::endl;

    layer->forward(inputs, outputs, internals);
    std::cout << "Forwarded" << std::endl;

    return 0;
    CommandLineParser argParser(argc, argv, diagnosticKeys);
    argParser.about("Use this tool to run the diagnostics of provided ONNX/TF model"
                    "to obtain the information about its support (supported layers).");

    if (argc == 1)
    {
        argParser.printMessage();
        return 0;
    }

    std::string model = checkFileExists(argParser.get<std::string>("model"));
    std::string config = checkFileExists(argParser.get<std::string>("config"));
    std::string frameworkId = argParser.get<std::string>("framework");

    CV_Assert(!model.empty());

    enableModelDiagnostics(true);
    skipModelImport(true);
    redirectError(diagnosticsErrorCallback, NULL);

    Net ocvNet = readNet(model, config, frameworkId);

    return 0;
}
