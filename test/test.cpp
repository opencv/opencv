#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <chrono>
#include <random>

using std::vector, std::cout, std::endl, std::getchar;
using cv::dnn::Net, cv::Mat, cv::String;

void printNetInfo(Net net) {
    vector<String> layerNames = net.getLayerNames();
    cout << "Input: " << net.getLayer(0)->name << endl;
    for (auto& i : layerNames) {
        int id = net.getLayerId(i);
        cout << "layer name: " << i << ", id=" << id << endl;
        auto v = net.getLayerInputs(id);
        cout << "  input layer: " << endl;
        for (auto j : v) {
            cout << "    " << j->name << endl;
        }
    }
}

void printMatRec(const cv::Mat& mat, int depth, int* mat_dims, int* dims) {
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


void cal(Mat& input1, Mat& input2, Mat& output, Net& net) {
    net.setInput(input1, "input1");
    net.setInput(input2, "input2");

    cout << "\033[93m" << "First forwarding." << "\033[0m\n";
    output = net.forward();
    cout << "\033[93m" << "Formal forwarding." << "\033[0m\n";

    auto begin = std::chrono::high_resolution_clock::now();
    output = net.forward();
    auto end = std::chrono::high_resolution_clock::now();
    cout << "\033[93mTime elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\033[0m\n";
}

Mat testSingle(Mat& input1, Mat& input2, Net& net, bool useVK, bool print = false, bool useCuda = false, bool useInternelPrint = true)
{
    Mat output;
    if (useVK)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
        cout << "\033[95mUsing Vulkan.\033[0m\n";
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cout << "\033[95mUsing CPU.\033[0m\n";
    }
    if (useCuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        cout << "\033[95mUsing CUDA.\033[0m\n";
    }


    cal(input1, input2, output, net);
    if (print)
    {
        cout << "input1: " << endl;
        if (useInternelPrint)
            cout << input1 << endl;
        else
            printMat(input1);
        cout << "input2: " << endl;
        if (useInternelPrint)
            cout << input2 << endl;
        else
            printMat(input2);

        cout << "output: " << endl;
        if (useInternelPrint)
            cout << output << endl;
        else
            printMat(output);
    }
    return output;
}

void verifyResult(Mat& mat1, Mat& mat2)
{
    size_t sz = 0;
    for (auto it1 = mat1.begin<float>(), it2 = mat2.begin<float>(); it1 != mat1.end<float>(); ++it1, ++it2, ++sz)
    {
        if (std::fabs(*it1 - *it2) > 1e-9)
        {
            cout << "\033[91mElement unmatch: " << *it1 << " != " << *it2 << ", at " << sz << "\033[0m\n";
            //abort();
            return;
        }
    }
    cout << "\033[92mResults passed verification.\033[0m\n";
}

void validityTest(Net& net)
{
    Mat input1, input2;

    input1 = Mat::ones(8, 8, CV_32F);
    input2 = Mat::ones(8, 1, CV_32F);
    input1.at<float>(3, 2) = 25;
    input2.at<float>(3, 0) = 17;

    Mat output1 = testSingle(input1, input2, net, true, true);
    Mat output2 = testSingle(input1, input2, net, false, true);
    verifyResult(output1, output2);
}

void speedTest(Net& net)
{
    using std::mt19937, std::uniform_real_distribution, std::uniform_int_distribution;
    Mat input1, input2;

    int matDimH = 16384, matDimW = 16384;
    input1 = Mat::ones(matDimH, matDimW, CV_32F);
    input2 = Mat::ones(matDimH, matDimW, CV_32F);
    mt19937 rng;
    uniform_real_distribution<float> dist3;
    uniform_int_distribution<int> distidxX(0, matDimH - 1);
    uniform_int_distribution<int> distidxY(0, matDimW - 1);
    uniform_int_distribution<int> distsel(0, 1);

    int maxDisturbanceNum = 262144;
    for (int i = 0; i < maxDisturbanceNum; ++i)
    {
        if (distsel(rng) == 0)
        {
            input1.at<float>(distidxX(rng), distidxY(rng)) = dist3(rng) * 1e5;
        }
        else
        {
            input2.at<float>(distidxX(rng), distidxY(rng)) = dist3(rng) * 1e5;
        }
    }

    Mat output1 = testSingle(input1, input2, net, true, false);
    Mat output2 = testSingle(input1, input2, net, false, false);
    //verifyResult(output1, output2);
}


#ifdef WITH_CUDA
void testCuda() {
    Net net = Net();
    cv::dnn::LayerParams params = cv::dnn::LayerParams();
    params.name = "Eltwise";
    params.type = "Eltwise";
    params.set("operation", "sum");
    params.set("output_channels_mode", "input_0_truncate");
    net.addLayer(params.name, params.type, params);
    net.setInputsNames({ "input1", "input2" });
    net.connect(0, 0, 1, 0);
    net.connect(0, 1, 1, 1);
    Mat input1, input2;

    input1 = Mat::ones(8, 8, CV_32F);
    input2 = Mat::ones(8, 1, CV_32F);
    input1.at<float>(3, 2) = 25;
    input2.at<float>(3, 0) = 17;
    Mat output = testSingle(input1, input2, net, true, true, true);
    return;



    int matDimH = 16384, matDimW = 16384;
    input1 = Mat::ones(matDimH, matDimW, CV_32F);
    input2 = Mat::ones(matDimH, matDimW, CV_32F);
    using std::mt19937, std::uniform_real_distribution, std::uniform_int_distribution;
    mt19937 rng;
    uniform_real_distribution<float> dist3;
    uniform_int_distribution<int> distidxX(0, matDimH - 1);
    uniform_int_distribution<int> distidxY(0, matDimW - 1);
    uniform_int_distribution<int> distsel(0, 1);

    int maxDisturbanceNum = 262144;
    for (int i = 0; i < maxDisturbanceNum; ++i)
        int maxDisturbanceNum = 262144;
    for (int i = 0; i < maxDisturbanceNum; ++i)
    {
        if (distsel(rng) == 0)
        {
            input1.at<float>(distidxX(rng), distidxY(rng)) = dist3(rng) * 1e5;
        }
        else
        {
            input2.at<float>(distidxX(rng), distidxY(rng)) = dist3(rng) * 1e5;
        }
    }
    testSingle(input1, input2, net, false, false, true);
    testSingle(input1, input2, net, false, false, false);
}
#endif

// Set up capturing API
#if __has_include("renderdoc_app.h")

#define RENDERDOC_ENABLED 1
#include <renderdoc_app.h>

#if defined(WIN32)
#include <Windows.h>
#include <windef.h>
#include <libloaderapi.h>

#elif defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
#include <dlfcn.h>
#endif  /* PLATFORM */

RENDERDOC_API_1_6_0* rdoc_api = NULL;
#endif /* RENDERDOC_ENABLED */

int main() {

#ifdef WITH_CUDA
   testCuda();
   return 0;
#endif


#ifdef RENDERDOC_ENABLED
#if defined(WIN32)
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        assert(ret == 1);
    }
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        assert(ret == 1);
    }
#else
#error "Unknown platform"
#endif /* PLATFORM */
#endif /* RENDERDOC_ENABLED */

#ifdef RENDERDOC_ENABLED
    cout << "\033[92m" << "RenderDoc library is present." << "\033[0m\n";
#else
    cout << "\033[91m" << "RenderDoc library is not present. Capturing shader execution is not possible." << "\033[0m\n";
#endif

#ifdef RENDERDOC_ENABLED
    cout << "\033[93m" << "RenderDoc API loaded: " << "\033[0m";
    if (rdoc_api)
        cout << "\033[92m" << "TRUE" << "\033[0m\n";
    else
        cout << "\033[91m" << "FALSE" << "\033[0m\n";
<<<<<<< HEAD
#endif
=======
#endif /* RENDERDOC_ENABLED */

>>>>>>> d379785317 (Checkpoint)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);

    // Create the actual net object
    Net net = Net();
    cv::dnn::LayerParams params = cv::dnn::LayerParams();
    params.name = "NaryEltwise";
    params.type = "NaryEltwise";
    params.set("operation", "add");
    net.addLayer(params.name, params.type, params);
    net.setInputsNames({ "input1", "input2" });
    net.connect(0, 0, 1, 0);
    net.connect(0, 1, 1, 1);

#ifdef RENDERDOC_ENABLED
    if (rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);
#endif /* RENDERDOC_ENABLED */

    // ============= BEGIN OF EXECUTION ===============
    
    //validityTest(net);
    speedTest(net);

    // ============= END OF EXECUTION ===============

#ifdef RENDERDOC_ENABLED
    if (rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);
#endif /* RENDERDOC_ENABLED */

    //cout << "Press any key to continue...";
    //getchar();

    return 0;
}
