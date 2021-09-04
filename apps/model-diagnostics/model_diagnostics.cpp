/*************************************************
USAGE:
./model_diagnostics -m <model file location>
**************************************************/
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/dnn/utils/debug_utils.hpp>

#include <iostream>


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
