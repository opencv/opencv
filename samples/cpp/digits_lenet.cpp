//  This example provides a digital recognition based on LeNet-5 and connected component analysis.
//  It makes it possible for OpenCV beginner to run dnn models in real time using only CPU.
//  It can read pictures from the camera in real time to make predictions, and display the recognized digits as overlays on top of the original digits.
//
//  In order to achieve a better display effect, please write the number on white paper and occupy the entire camera.
//
//  You can follow the following guide to train LeNet-5 by yourself using the MNIST dataset.
//  https://github.com/intel/caffe/blob/a3d5b022fe026e9092fc7abc7654b1162ab9940d/examples/mnist/readme.md
//
//  You can also download already trained model directly.
//  https://github.com/zihaomu/opencv_digit_text_recognition_demo/tree/master/src


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;

const char *keys =
    "{ help     h  | | Print help message. }"
    "{ input    i  | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ device      |  0  | camera device number. }"
    "{ modelBin    |     | Path to a binary .caffemodel file contains trained network.}"
    "{ modelTxt    |     | Path to a .prototxt file contains the model definition of trained network.}"
    "{ width       | 640 | Set the width of the camera }"
    "{ height      | 480 | Set the height of the camera }"
    "{ thr         | 0.7 | Confidence threshold. }";

// Find best class for the blob (i.e. class with maximal probability)
static void getMaxClass(const Mat &probBlob, int &classId, double &classProb);

void predictor(Net net, const Mat &roi, int &class_id, double &probability);

int main(int argc, char **argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int vWidth = parser.get<int>("width");
    int vHeight = parser.get<int>("height");
    float confThreshold = parser.get<float>("thr");
    std::string modelTxt = parser.get<String>("modelTxt");
    std::string modelBin = parser.get<String>("modelBin");

    Net net;
    try
    {
        net = readNet(modelTxt, modelBin);
    }
    catch (cv::Exception &ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        std::cout << "Can't load the network by using the flowing files:" << std::endl;
        std::cout << "modelTxt: " << modelTxt << std::endl;
        std::cout << "modelBin: " << modelBin << std::endl;
        return 1;
    }

    const std::string resultWinName = "Please write the number on white paper and occupy the entire camera.";
    const std::string preWinName = "Preprocessing";

    namedWindow(preWinName, WINDOW_AUTOSIZE);
    namedWindow(resultWinName, WINDOW_AUTOSIZE);

    Mat labels, stats, centroids;
    Point position;

    Rect getRectangle;
    bool ifDrawingBox = false;

    int classId = 0;
    double probability = 0;

    Rect basicRect = Rect(0, 0, vWidth, vHeight);
    Mat rawImage;

    double fps = 0;

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));

    TickMeter tm;

    while (waitKey(1) < 0)
    {
        cap >> rawImage;
        if (rawImage.empty())
        {
            waitKey();
            break;
        }

        tm.reset();
        tm.start();

        Mat image = rawImage.clone();
        // Image preprocessing
        cvtColor(image, image, COLOR_BGR2GRAY);
        GaussianBlur(image, image, Size(3, 3), 2, 2);
        adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 10);
        bitwise_not(image, image);

        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1,-1));
        dilate(image, image, element, Point(-1,-1), 1);
        // Find connected component
        int nccomps = cv::connectedComponentsWithStats(image, labels, stats, centroids);

        for (int i = 1; i < nccomps; i++)
        {
            ifDrawingBox = false;

            // Extend the bounding box of connected component for easier recognition
            if (stats.at<int>(i - 1, CC_STAT_AREA) > 80 && stats.at<int>(i - 1, CC_STAT_AREA) < 3000)
            {
                ifDrawingBox = true;
                int left = stats.at<int>(i - 1, CC_STAT_HEIGHT) / 4;
                getRectangle = Rect(stats.at<int>(i - 1, CC_STAT_LEFT) - left, stats.at<int>(i - 1, CC_STAT_TOP) - left, stats.at<int>(i - 1, CC_STAT_WIDTH) + 2 * left, stats.at<int>(i - 1, CC_STAT_HEIGHT) + 2 * left);
                getRectangle &= basicRect;
            }

            if (ifDrawingBox && !getRectangle.empty())
            {
                Mat roi = image(getRectangle);
                predictor(net, roi, classId, probability);

                if (probability < confThreshold)
                    continue;

                rectangle(rawImage, getRectangle, Scalar(128, 255, 128), 2);

                position = Point(getRectangle.br().x - 7, getRectangle.br().y + 25);
                putText(rawImage, std::to_string(classId), position, 3, 1.0, Scalar(128, 128, 255), 2);
            }
        }

        tm.stop();
        fps = 1 / tm.getTimeSec();
        std::string fpsString = format("Inference FPS: %.2f.", fps);
        putText(rawImage, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 255, 128));

        imshow(resultWinName, rawImage);
        imshow(preWinName, image);

    }

    return 0;
}

static void getMaxClass(const Mat &probBlob, int &classId, double &classProb)
{
    Mat probMat = probBlob.reshape(1, 1);
    Point classNumber;
    minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
    classId = classNumber.x;
}

void predictor(Net net, const Mat &roi, int &classId, double &probability)
{
    Mat pred;
    // Convert Mat to batch of images
    Mat inputBlob = dnn::blobFromImage(roi, 1.0, Size(28, 28));
    // Set the network input
    net.setInput(inputBlob);
    // Compute output
    pred = net.forward();
    getMaxClass(pred, classId, probability);
}
