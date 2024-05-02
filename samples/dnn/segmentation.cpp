#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string param_keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ colors      | | Optional path to a text file with colors for an every class. "
    "An every color is represented with three values from 0 to 255 in BGR channels order. }";
std::string backend_keys = cv::format(
    "{ backend   | 0 | Choose one of computation backends: "
    "%d: automatically (by default), "
    "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
    "%d: OpenCV implementation, "
    "%d: VKCOM, "
    "%d: CUDA }",
    cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_BACKEND_VKCOM, cv::dnn::DNN_BACKEND_CUDA);
std::string target_keys = cv::format(
    "{ target    | 0 | Choose one of target computation devices: "
    "%d: CPU target (by default), "
    "%d: OpenCL, "
    "%d: OpenCL fp16 (half-float precision), "
    "%d: VPU, "
    "%d: Vulkan, "
    "%d: CUDA, "
    "%d: CUDA fp16 (half-float preprocess) }",
    cv::dnn::DNN_TARGET_CPU, cv::dnn::DNN_TARGET_OPENCL, cv::dnn::DNN_TARGET_OPENCL_FP16, cv::dnn::DNN_TARGET_MYRIAD, cv::dnn::DNN_TARGET_VULKAN, cv::dnn::DNN_TARGET_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);
std::string keys = param_keys + backend_keys + target_keys;

using namespace cv;
using namespace std;
using namespace dnn;

std::vector<std::string> classes;
std::vector<Vec3b> colors;

void showLegend();

void colorizeSegmentation(const Mat &score, Mat &segm);

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run semantic segmentation deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findFile(parser.get<String>("model"));
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Open file with colors.
    if (parser.has("colors"))
    {
        std::string file = parser.get<String>("colors");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            std::istringstream colorStr(line.c_str());

            Vec3b color;
            for (int i = 0; i < 3 && !colorStr.eof(); ++i)
                colorStr >> color[i];
            colors.push_back(color);
        }
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    CV_Assert(!model.empty());
    //! [Read and initialize network]
    Net net = readNetFromONNX(model);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    // Create a window
    static const std::string kWinName = "Deep learning semantic segmentation in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));
    //! [Open a video file or an image file or a camera stream]

    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        cv::imshow("Original Image", frame);

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
        //! [Create a 4D blob from a frame]
        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
        Mat score = net.forward();
        //! [Make forward pass]

        cv::Mat mask, saliency_map, foreground_overlay, background_overlay, foreground_segmented;
        if (modelName == "u2netp")
        {

            saliency_map = cv::Mat(score.size[2], score.size[3], CV_32F, score.ptr<float>(0, 0));
            cv::threshold(saliency_map, mask, 0.5, 255, cv::THRESH_BINARY);
            cv::resize(mask, mask, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_NEAREST);
            mask.convertTo(mask, CV_8U);

            // Create overlays for foreground and background
            foreground_overlay = cv::Mat::zeros(frame.size(), frame.type());
            background_overlay = cv::Mat::zeros(frame.size(), frame.type());

            // Set foreground (object) to red and background to blue
            for (int i = 0; i < mask.rows; i++)
            {
                for (int j = 0; j < mask.cols; j++)
                {
                    if (mask.at<uchar>(i, j) == 255)
                    {
                        foreground_overlay.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255); // Red foreground
                    }
                    else
                    {
                        background_overlay.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0); // Blue background
                    }
                }
            }
            // Blend the overlays with the original frame
            cv::addWeighted(frame, 1, foreground_overlay, 0.5, 0, foreground_segmented);
            cv::addWeighted(foreground_segmented, 1, background_overlay, 0.5, 0, frame);
        }
        else
        {
            Mat segm;
            colorizeSegmentation(score, segm);

            resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
            addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);
        }

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
        if (!classes.empty())
            showLegend();
    }
    return 0;
}

void colorizeSegmentation(const Mat &score, Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    if (colors.empty())
    {
        // Generate colors.
        colors.push_back(Vec3b());
        for (int i = 1; i < chns; ++i)
        {
            Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    }
    else if (chns != (int)colors.size())
    {
        CV_Error(Error::StsError, format("Number of output classes does not match "
                                         "number of colors (%d != %zu)",
                                         chns, colors.size()));
    }

    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
}

void showLegend()
{
    static const int kBlockHeight = 30;
    static Mat legend;
    if (legend.empty())
    {
        const int numClasses = (int)classes.size();
        if ((int)colors.size() != numClasses)
        {
            CV_Error(Error::StsError, format("Number of output classes does not match "
                                             "number of labels (%zu != %zu)",
                                             colors.size(), classes.size()));
        }
        legend.create(kBlockHeight * numClasses, 200, CV_8UC3);
        for (int i = 0; i < numClasses; i++)
        {
            Mat block = legend.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
            block.setTo(colors[i]);
            putText(block, classes[i], Point(0, kBlockHeight / 2), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255, 255, 255));
        }
        namedWindow("Legend", WINDOW_NORMAL);
        imshow("Legend", legend);
    }
}
