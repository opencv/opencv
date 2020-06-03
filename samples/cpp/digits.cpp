//  This example provides a digital recognition based on LeNet-5 and connected component analysis.
//  It makes it possible for OpenCV beginner to run dnn models in real time using only CPU.
//  It can read pictures from the camera in real time to make predictions, and display the recognized digits as overlays on top of the original digits.
//
//  In order to achieve a better display effect, please write the number on white paper and occupy the entire camera.
//
//  You can follow the following guide to train LeNet-5 by yourself using the minist dataset.
//  https://github.com/intel/caffe/blob/a3d5b022fe026e9092fc7abc7654b1162ab9940d/examples/mnist/readme.md
//
//  You can also download and train the model directly.
//  https://github.com/zihaomu/opencv_lenet_demo/blob/master/src/lenet.caffemodel
//  https://github.com/zihaomu/opencv_lenet_demo/blob/master/src/lenet.prototxt

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char *keys =
    "{ help     h  | | Print help message. }"
    "{ modelBin    | | Path to a binary .caffemodel file contains trained network.}"
    "{ modelTxt    | | Path to a .prototxt file contains the model definition of trained network.}"
    "{ width       | 640 | Set the width of the camera }"
    "{ height      | 480 | Set the height of the camera }"
    "{ thr         | 0.8 | Confidence threshold. }";

// Find best class for the blob (i.e. class with maximal probability)
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb);

void predictor(dnn::Net net, Mat &roi, int &class_id, double &probability);

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
    String modelTxt = parser.get<String>("modelTxt");
    String modelBin = parser.get<String>("modelBin");

    dnn::Net net;
    try
    {
        net = dnn::readNetFromCaffe(modelTxt, modelBin);
    }
    catch (cv::Exception &ee)
    {
        cerr << "Exception: " << ee.what() << endl;
        if (net.empty())
        {
            cout << "Can't load the network by using the flowing files:" << endl;
            cout << "modelTxt: " << modelTxt << endl;
            cout << "modelBin: " << modelBin << endl;
            exit(-1);
        }
    }

    static const string resultWinName = "LeNet Result";
    static const string preWinName = "Preprocessing";

    namedWindow(resultWinName, WINDOW_AUTOSIZE);
    namedWindow(preWinName, WINDOW_AUTOSIZE);

    Mat labels, img_color, stats, centroids;
    Point positiosn;

    Rect getRectangle;
    bool ifDrawingBox = false;

    int classId = 0;
    double probability = 0;

    VideoCapture cap(0);

    // Set camera resolution
    cap.set(CAP_PROP_FRAME_WIDTH, vWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, vHeight);

    Rect basicRact = Rect(0, 0, 640, 480);
    Mat rawImage;

    double fps = 0;

    // Open a video file or an image file or a camera stream.
    if (cap.isOpened())
    {
        TickMeter cvtm;

        while (true)
        {
            cvtm.reset();
            cvtm.start();
            cap >> rawImage;

            Mat image = rawImage.clone();

            // Image preprocessing
            cvtColor(image, image, COLOR_BGR2GRAY);
            GaussianBlur(image, image, Size(3, 3), 2, 2);
            adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 10);
            bitwise_not(image, image);

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
                    getRectangle &= basicRact;
                }

                if (ifDrawingBox)
                {
                    Mat roi = image(getRectangle);
                    predictor(net, roi, classId, probability);

                    if (probability < confThreshold)
                        continue;

                    // cout << "probability : "<<probability << endl;

                    rectangle(rawImage, getRectangle, Scalar(128, 255, 128), 2);

                    positiosn = Point(getRectangle.br().x - 7, getRectangle.br().y + 25);
                    putText(rawImage, to_string(classId), positiosn, 3, 1.0, Scalar(128, 128, 255), 2);
                }
            }

            cvtm.stop();
            fps = 1 / cvtm.getTimeSec();
            string fpsString = format("Inference FPS: %.2f ms", fps);
            putText(image, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(128, 255, 128));

            // printf("time = %gms\n", cvtm.getTimeMilli());
            imshow(resultWinName, image);
            imshow(preWinName, rawImage);
            waitKey(30);
        }
    }

    return 0;
}

static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1, 1);
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

void predictor(dnn::Net net, Mat &roi, int &classId, double &probability)
{
    Mat pred;
    //Convert Mat to batch of images
    Mat inputBlob = dnn::blobFromImage(roi, 1, Size(28, 28), Scalar(), false);
    //set the network input, "data" is the name of the input layer
    net.setInput(inputBlob, "data");

    //compute output, "prob" is the name of the output layer
    pred = net.forward("prob");
    //cout << pred << endl;
    getMaxClass(pred, &classId, &probability);
}
