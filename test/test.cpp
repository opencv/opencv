#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

void printNetInfo(dnn::Net net) {
    vector<String> layerNames = net.getLayerNames();
    cout << "Input: " << net.getLayer(0)->name << endl;
    for (auto i : layerNames) {
        int id = net.getLayerId(i);
        cout << "layer name: " << i << ", id=" << id << endl;
        auto v = net.getLayerInputs(id);
        cout << "  input layer: " << endl;
        for (auto j : v) {
            cout << "    " << j->name << endl;
        }
    }
}

void printMatRec(const cv::Mat& mat, int depth, int *mat_dims, int *dims) {
    for (int i = 0; i < mat.dims - depth; i++) {
        cout << " ";
    }
    cout << "[";
    if (depth == 1) {
        for (int i = 0; i < mat_dims[0]; i++) {
            dims[0] = i;
            cout << mat.at<float>(dims) << ",";
        }
        dims[0] = 0;
    }
    else {
        cout << endl;
        for (int i = 0; i < mat_dims[depth - 1]; i++) {
            dims[depth - 1] = i;
            printMatRec(mat, depth - 1, mat_dims, dims);
        }
        for (int i = 0; i < mat.dims - depth; i++) {
            cout << " ";
        }
    }
    cout << "]" << endl;

}

void printMat(const cv::Mat& mat) {
    int ndims = mat.dims;
    vector<int> mat_dims;
    vector<int> dims;
    for (int i = 0; i < ndims; i++) {
        mat_dims.push_back(mat.size[i]);
        dims.push_back(0);
    }
    printMatRec(mat, ndims, mat_dims.data(), dims.data());
}


void cal(Mat &input1, Mat &input2, Mat &output, dnn::Net& net) {
    net.setInput(input1, "input1");
    net.setInput(input2, "input2");

    output = net.forward();

    auto begin = std::chrono::high_resolution_clock::now();
    CV_LOG_DEBUG(NULL, "start forwarding");
    output = net.forward();
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";
}

void testSingle(Mat &input1, Mat &input2, dnn::Net& net, bool print = false)
{
    Mat output;
    cal(input1, input2, output, net);
    if (print)
    {
        cout << "input1: " << endl;
        printMat(input1);
        cout << "input2: " << endl;
        printMat(input2);
        cout << "output: " << endl;
            printMat(output);
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
    dnn::Net net = dnn::Net();
    net.setPreferableBackend(dnn::DNN_BACKEND_VKCOM);
    net.setPreferableTarget(dnn::DNN_TARGET_VULKAN);
    dnn::LayerParams params = dnn::LayerParams();
    params.name = "NaryEltwise";
    params.type = "NaryEltwise";
    params.set("operation", "add");
    net.addLayer(params.name, params.type, params);
    net.setInputsNames({"input1", "input2"});
    net.connect(0, 0, 1, 0);
    net.connect(0, 1, 1, 1);

    Mat input1, input2, output;

    int matDimH = 16392, matDimW = 16392;
    input1 = Mat::ones(matDimH, matDimW, CV_32F);
    input2 = Mat::ones(matDimH, matDimW, CV_32F);

    net.setPreferableBackend(dnn::DNN_BACKEND_VKCOM);
    net.setPreferableTarget(dnn::DNN_TARGET_VULKAN);
    cout << "Using Vulkan.\n";
    testSingle(input1, input2, net);

    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    cout << "Using CPU.\n";
    testSingle(input1, input2, net);

    // input1 = Mat::ones(3, 2, CV_32F);
    // input2 = Mat::ones(3, 1, CV_32F);
    // input2.at<float>(0, 0) = 2;
    // cal(input1, input2, output, net);
    // printMat(output);

    // input1 = Mat::ones(3, 1, CV_32F);
    // input2 = Mat::ones(1, 3, CV_32F);
    // input2.at<float>(0, 0) = 2;
    // cal(input1, input2, output, net);

    // generate a mat with 4 dims
    // input1 = Mat::ones(4, vector<int>{3, 2, 2, 2}.data(), CV_32F);
    // input2 = Mat::ones(2, 2, CV_32F);
    // input2.at<float>(0, 0) = 1;
    // input2.at<float>(0, 1) = 2;
    // input2.at<float>(1, 0) = 3;
    // input2.at<float>(1, 1) = 4;
    // cal(input1, input2, output, net);

    return 0;
}
