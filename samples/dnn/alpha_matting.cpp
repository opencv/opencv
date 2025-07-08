/**
 * @file alpha_matting.cpp
 * @brief MODNet Alpha Matting using OpenCV DNN
 *
 * This sample demonstrates human portrait alpha matting using MODNet model.
 * MODNet is a trimap-free portrait matting method that can produce high-quality
 * alpha mattes for portrait images in real-time.
 *
 * Usage:
 *   ./alpha_matting --input=image.jpg                      # Process image
 *   ./alpha_matting --input=video.mp4                      # Process video
 *   ./alpha_matting                                        # Use webcam
 *   ./alpha_matting modnet                                 # Use config alias
 *
 * Requirements:
 *   - OpenCV >= 4.5.0 with DNN module
 *   - MODNet ONNX model
 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

static void loadModel(const string modelPath, String backend, String target, Net &net, EngineType engine)
{
    net = readNetFromONNX(modelPath, engine);
    net.setPreferableBackend(getBackendID(backend));
    net.setPreferableTarget(getTargetID(target));
}

static void postprocess(const Mat &image, const Mat &alpha_output, Mat &alpha_mask, Mat &composite)
{
    int h = image.rows;
    int w = image.cols;

    Mat alpha;
    if (alpha_output.dims == 4 && alpha_output.size[0] == 1 && alpha_output.size[1] == 1)
    {
        alpha = alpha_output.reshape(0, {alpha_output.size[2], alpha_output.size[3]});
    }
    else
    {
        alpha = alpha_output.clone();
    }

    resize(alpha, alpha, Size(w, h));

    alpha = max(0.0, min(1.0, alpha));
    alpha.convertTo(alpha_mask, CV_8U, 255.0);

    Mat alpha_3ch;
    cvtColor(alpha_mask, alpha_3ch, COLOR_GRAY2BGR);
    alpha_3ch.convertTo(alpha_3ch, CV_32F, 1.0 / 255.0);

    Mat image_f;
    image.convertTo(image_f, CV_32F);
    multiply(image_f, alpha_3ch, composite);
    composite.convertTo(composite, CV_8U);
}

static void processFrame(const Mat &frame, Mat &alpha_mask, Mat &composite, Net &net,
                         float scale, int width, int height, const Scalar &mean, bool swapRB)
{
    if (frame.empty())
        return;

    Mat blob = blobFromImage(frame, scale, Size(width, height), mean, swapRB, false, CV_32F);

    if (abs(scale - 1.0f) < 1e-6)
    {
        blob = (blob / 127.5f) - 1.0f;
    }

    net.setInput(blob);
    Mat output = net.forward();
    postprocess(frame, output, alpha_mask, composite);
}

static void setupWindows()
{
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Alpha Mask", WINDOW_AUTOSIZE);
    namedWindow("Composite", WINDOW_AUTOSIZE);
    moveWindow("Alpha Mask", 200, 0);
    moveWindow("Composite", 400, 0);
}

int main(int argc, char **argv)
{
    const string about =
        "This sample demonstrates human portrait alpha matting using MODNet model.\n"
        "MODNet is a trimap-free portrait matting method that can produce high-quality\n"
        "alpha mattes for portrait images in real-time.\n\n"
        "Usage examples:\n"
        "\t./alpha_matting --input=image.jpg\n"
        "\t./alpha_matting --input=video.mp4\n"
        "\t./alpha_matting (for webcam)\n"
        "\t./alpha_matting modnet (using config alias)\n\n"
        "Download MODNet model from: https://github.com/ZHKKKe/MODNet/releases\n"
        "Press 'q' or ESC to quit \n";

    const string param_keys =
        "{ help h          |                   | Print help message }"
        "{ @alias          |                   | An alias name of model to extract preprocessing parameters from models.yml file }"
        "{ zoo             | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
        "{ input i         |                   | Path to input image or video file. Skip this argument to capture frames from a camera }"
        "{ model           |                   | Path to MODNet ONNX model file }";

    const string backend_keys = format(
        "{ backend         | default | Choose one of computation backends: "
        "default: automatically (by default), "
        "openvino: Intel's Deep Learning Inference Engine, "
        "opencv: OpenCV implementation, "
        "vkcom: VKCOM, "
        "cuda: CUDA, "
        "webnn: WebNN }");

    const string target_keys = format(
        "{ target          | cpu | Choose one of target computation devices: "
        "cpu: CPU target (by default), "
        "opencl: OpenCL, "
        "opencl_fp16: OpenCL fp16 (half-float precision), "
        "vpu: VPU, "
        "vulkan: Vulkan, "
        "cuda: CUDA, "
        "cuda_fp16: CUDA fp16 (half-float precision) }");

    string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    string zooFile = parser.get<String>("zoo");

    const char *path = getenv("OPENCV_SAMPLES_DATA_PATH");
    if ((path != NULL) || parser.has("@alias") || (parser.get<String>("model") != ""))
    {
        if (modelName.empty())
        {
            modelName = "modnet"; // Default alias
        }
        zooFile = findFile(zooFile);
    }

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);

    int input_width = parser.has("width") ? parser.get<int>("width") : 512;
    int input_height = parser.has("height") ? parser.get<int>("height") : 512;
    float scale_factor = parser.has("scale") ? parser.get<float>("scale") : 1.0f;
    Scalar mean_values = parser.has("mean") ? parser.get<Scalar>("mean") : Scalar(0.0, 0.0, 0.0);
    bool swapRB = parser.has("rgb") ? parser.get<bool>("rgb") : true;
    String backend = parser.get<String>("backend");
    String target = parser.get<String>("target");
    String sha1 = parser.has("sha1") ? parser.get<String>("sha1") : "";

    string model = findModel(parser.get<String>("model"), sha1);

    if (model.empty())
    {
        cout << "[ERROR] MODNet model not found!" << endl;
        cout << "Please download MODNet ONNX model from: https://github.com/ZHKKKe/MODNet/releases" << endl;
        cout << "Or specify model path using --model argument" << endl;
        return -1;
    }

    parser.about(about);

    EngineType engine = ENGINE_AUTO;
    if (backend != "default" || target != "cpu")
    {
        engine = ENGINE_CLASSIC;
    }

    Net net;
    loadModel(model, backend, target, net, engine);
    if (net.empty())
    {
        cout << "[ERROR] Failed to load model: " << model << endl;
        return -1;
    }

    VideoCapture cap;
    if (parser.has("input"))
    {
        string input_path = samples::findFile(parser.get<String>("input"));
        cap.open(input_path);
    }
    else
    {
        cap.open(0);
    }

    if (!cap.isOpened())
    {
        cerr << "Error: Cannot open input source" << endl;
        return -1;
    }

    setupWindows();

    cout << "Press 'q' or ESC to quit, 's' to save current frame results" << endl;

    Mat frame, alpha_mask, composite;

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "End of video or cannot read frame" << endl;
            break;
        }

        processFrame(frame, alpha_mask, composite, net, scale_factor, input_width, input_height, mean_values, swapRB);

        imshow("Input", frame);
        imshow("Alpha Mask", alpha_mask);
        imshow("Composite", composite);

        int key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}