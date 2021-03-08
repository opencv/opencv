/*************************************************
USAGE:
./model_diagnostics -m <onnx file location>
**************************************************/
#include <opencv2/dnn.hpp>
#include "../../samples/dnn/common.hpp"

#include <fstream>
#include <iostream>


using namespace cv;
using namespace dnn;


static void diagnosticsErrorCallback(const Exception& exc)
{
    fflush(stdout); fflush(stderr);
}

std::string diagnosticKeys =
        "{ model m     | | Path to the model .onnx file. }"
        "{ config c    | | Path to the model configuration file. }"
        "{ framework f | | [Optional] Name of the model framework. }";



int main( int argc, const char** argv )
{
    CommandLineParser argParser(argc, argv, diagnosticKeys);
    argParser.about("Use this tool to run the diagnostics of provided ONNX model"
                    "to obtain the information about its support (supported layers).");

    if (argc == 1)
    {
        argParser.printMessage();
        return 0;
    }

    String model = findFile(argParser.get<String>("model"));
    String config = findFile(argParser.get<String>("config"));
    String frameworkId = argParser.get<String>("framework");

    CV_Assert(!model.empty());

    enableModelDiagnostics(true);
    cv::redirectError((cv::ErrorCallback)diagnosticsErrorCallback, NULL);

    Net ocvNet = readNet(model, config, frameworkId);

    return 0;
}
