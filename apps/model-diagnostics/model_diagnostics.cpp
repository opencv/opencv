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

static std::vector<int> parseShape(const std::string &shape_str) {
    std::stringstream ss(shape_str);
    std::string item;
    std::vector<std::string> items;

    while (std::getline(ss, item, ',')) {
        items.push_back(item);
    }

    std::vector<int> shape;
    for (size_t i = 0; i < items.size(); i++) {
        shape.push_back(std::stoi(items[i]));
    }
    return shape;
}

std::string diagnosticKeys =
        "{ model m     | | Path to the model file. }"
        "{ config c    | | Path to the model configuration file. }"
        "{ framework f | | [Optional] Name of the model framework. }"
        "{ input0_name | | [Optional] Name of input0. Use with input0_shape}"
        "{ input0_shape | | [Optional] Shape of input0. Use with input0_name}"
        "{ input1_name | | [Optional] Name of input1. Use with input1_shape}"
        "{ input1_shape | | [Optional] Shape of input1. Use with input1_name}"
        "{ input2_name | | [Optional] Name of input2. Use with input2_shape}"
        "{ input2_shape | | [Optional] Shape of input2. Use with input2_name}"
        "{ input3_name | | [Optional] Name of input3. Use with input3_shape}"
        "{ input3_shape | | [Optional] Shape of input3. Use with input3_name}"
        "{ input4_name | | [Optional] Name of input4. Use with input4_shape}"
        "{ input4_shape | | [Optional] Shape of input4. Use with input4_name}";

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

    std::string input0_name = argParser.get<std::string>("input0_name");
    std::string input0_shape = argParser.get<std::string>("input0_shape");
    std::string input1_name = argParser.get<std::string>("input1_name");
    std::string input1_shape = argParser.get<std::string>("input1_shape");
    std::string input2_name = argParser.get<std::string>("input2_name");
    std::string input2_shape = argParser.get<std::string>("input2_shape");
    std::string input3_name = argParser.get<std::string>("input3_name");
    std::string input3_shape = argParser.get<std::string>("input3_shape");
    std::string input4_name = argParser.get<std::string>("input4_name");
    std::string input4_shape = argParser.get<std::string>("input4_shape");

    CV_Assert(!model.empty());

    enableModelDiagnostics(true);
    skipModelImport(true);
    redirectError(diagnosticsErrorCallback, NULL);

    Net ocvNet = readNet(model, config, frameworkId);

    std::vector<std::string> input_names;
    std::vector<std::vector<int>> input_shapes;
    if (!input0_name.empty() || !input0_shape.empty()) {
        CV_CheckFalse(input0_name.empty(), "input0_name cannot be empty");
        CV_CheckFalse(input0_shape.empty(), "input0_shape cannot be empty");
        input_names.push_back(input0_name);
        input_shapes.push_back(parseShape(input0_shape));
    }
    if (!input1_name.empty() || !input1_shape.empty()) {
        CV_CheckFalse(input1_name.empty(), "input1_name cannot be empty");
        CV_CheckFalse(input1_shape.empty(), "input1_shape cannot be empty");
        input_names.push_back(input1_name);
        input_shapes.push_back(parseShape(input1_shape));
    }
    if (!input2_name.empty() || !input2_shape.empty()) {
        CV_CheckFalse(input2_name.empty(), "input2_name cannot be empty");
        CV_CheckFalse(input2_shape.empty(), "input2_shape cannot be empty");
        input_names.push_back(input2_name);
        input_shapes.push_back(parseShape(input2_shape));
    }
    if (!input3_name.empty() || !input3_shape.empty()) {
        CV_CheckFalse(input3_name.empty(), "input3_name cannot be empty");
        CV_CheckFalse(input3_shape.empty(), "input3_shape cannot be empty");
        input_names.push_back(input3_name);
        input_shapes.push_back(parseShape(input3_shape));
    }
    if (!input4_name.empty() || !input4_shape.empty()) {
        CV_CheckFalse(input4_name.empty(), "input4_name cannot be empty");
        CV_CheckFalse(input4_shape.empty(), "input4_shape cannot be empty");
        input_names.push_back(input4_name);
        input_shapes.push_back(parseShape(input4_shape));
    }

    if (!input_names.empty() && !input_shapes.empty() && input_names.size() == input_shapes.size()) {
        ocvNet.setInputsNames(input_names);
        for (size_t i = 0; i < input_names.size(); i++) {
            Mat input(input_shapes[i], CV_32F);
            ocvNet.setInput(input, input_names[i]);
        }

        size_t dot_index = model.rfind('.');
        std::string graph_filename = model.substr(0, dot_index) + ".pbtxt";
        ocvNet.dumpToPbtxt(graph_filename);
    }

    return 0;
}
