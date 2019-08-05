#include <fstream>
#include <sstream>
#include <iostream>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ num_classes | | number of classes with which the model was trained. If a classes file is provided this parameter will be infered.}"
    "{ colors      | | Optional path to a text file with colors for an every class. "
                      "An every color is represented with three values from 0 to 255 in BGR channels order. }"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;
std::vector<Vec3b> colors;

void showLegend();

void colorizeSegmentation(const Mat &score, Mat &segm, int num_classes);

int main(int argc, char** argv)
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
    String config = findFile(parser.get<String>("config"));
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    int num_classes;
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
        num_classes = classes.size();
    }
    // If no classes file was given num_classes must be provided
    else
    {
        CV_Assert(parser.has("num_classes"));
        num_classes = parser.get<int>("num_classes");

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
    // FCN-8 Trained on Pascal
    // http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
    SegmentationModel net(config, model);
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

    //! [Set Input Parameters]
    net.setInputParams(scale, Size(inpWidth, inpHeight), mean, swapRB, false);
    //! [Set Input Parameters]

    // Process frames.
    Mat frame;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        //! [Network Forward pass]
        Mat mask;
        net.segment(frame, mask);
        //! [Network Forward pass]

        Mat segm;
        colorizeSegmentation(mask, segm, num_classes);

        resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
        addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);

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

void colorizeSegmentation(const Mat &mask, Mat &segm, int num_classes)
{
    const int rows = mask.size[0];
    const int cols = mask.size[1];

    if (colors.empty())
    {
        // Generate colors.
        colors.push_back(Vec3b());
        for (int i = 1; i < num_classes; ++i)
        {
            Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    }
    else if (num_classes != (int)colors.size())
    {
        CV_Error(Error::StsError, format("Number of output classes does not match "
                                         "number of colors (%d != %zu)", num_classes, colors.size()));
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = mask.ptr<uchar>(row);
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
                                             "number of labels (%zu != %zu)", colors.size(), classes.size()));
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
