/*
Sample of using OpenCV dnn module with Torch ENet model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
using namespace std;

const String keys =
        "{help h    || Sample app for loading ENet Torch model. "
                       "The model and class names list can be downloaded here: "
                       "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AAD5YkUa8XgMpHs2gCRgmCVCa }"
        "{model m   || path to Torch .net model file (model_best.net) }"
        "{image i   || path to image file }"
        "{result r  || path to save output blob (optional, binary format, NCHW order) }"
        "{show s    || whether to show all output channels or not}"
        "{o_blob    || output blob's name. If empty, last blob's name in net is used}";

static const int kNumClasses = 20;

static const String classes[] = {
    "Background", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
    "TrafficLight", "TrafficSign", "Vegetation", "Terrain", "Sky", "Person",
    "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
};

static const Vec3b colors[] = {
    Vec3b(0, 0, 0), Vec3b(244, 126, 205), Vec3b(254, 83, 132), Vec3b(192, 200, 189),
    Vec3b(50, 56, 251), Vec3b(65, 199, 228), Vec3b(240, 178, 193), Vec3b(201, 67, 188),
    Vec3b(85, 32, 33), Vec3b(116, 25, 18), Vec3b(162, 33, 72), Vec3b(101, 150, 210),
    Vec3b(237, 19, 16), Vec3b(149, 197, 72), Vec3b(80, 182, 21), Vec3b(141, 5, 207),
    Vec3b(189, 156, 39), Vec3b(235, 170, 186), Vec3b(133, 109, 144), Vec3b(231, 160, 96)
};

static void showLegend();

static void colorizeSegmentation(const Mat &score, Mat &segm);

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }

    String modelFile = parser.get<String>("model");
    String imageFile = parser.get<String>("image");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    String resultFile = parser.get<String>("result");

    //! [Read model and initialize network]
    dnn::Net net = dnn::readNetFromTorch(modelFile);

    //! [Prepare blob]
    Mat img = imread(imageFile), input;
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    Mat inputBlob = blobFromImage(img, 1./255, Size(1024, 512), Scalar(), true, false);   //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob);        //set the network input
    //! [Set input blob]

    TickMeter tm;

    String oBlob = net.getLayerNames().back();
    if (!parser.get<String>("o_blob").empty())
    {
        oBlob = parser.get<String>("o_blob");
    }

    //! [Make forward pass]
    tm.start();
    Mat result = net.forward(oBlob);
    tm.stop();

    if (!resultFile.empty()) {
        CV_Assert(result.isContinuous());

        ofstream fout(resultFile.c_str(), ios::out | ios::binary);
        fout.write((char*)result.data, result.total() * sizeof(float));
        fout.close();
    }

    std::cout << "Output blob: " << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << "\n";
    std::cout << "Inference time, ms: " << tm.getTimeMilli()  << std::endl;

    if (parser.has("show"))
    {
        Mat segm, show;
        colorizeSegmentation(result, segm);
        showLegend();

        cv::resize(segm, segm, img.size(), 0, 0, cv::INTER_NEAREST);
        addWeighted(img, 0.1, segm, 0.9, 0.0, show);

        imshow("Result", show);
        waitKey();
    }
    return 0;
} //main

static void showLegend()
{
    static const int kBlockHeight = 30;

    cv::Mat legend(kBlockHeight * kNumClasses, 200, CV_8UC3);
    for(int i = 0; i < kNumClasses; i++)
    {
        cv::Mat block = legend.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
        block.setTo(colors[i]);
        putText(block, classes[i], Point(0, kBlockHeight / 2), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255, 255, 255));
    }
    imshow("Legend", legend);
}

static void colorizeSegmentation(const Mat &score, Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

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
