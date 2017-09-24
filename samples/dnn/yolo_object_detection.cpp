#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

const size_t network_width = 416;
const size_t network_height = 416;

const char* about = "This sample uses You only look once (YOLO)-Detector "
                    "(https://arxiv.org/abs/1612.08242)"
                    "to detect objects on image\n"; // TODO: link

const char* params
    = "{ help           | false | print usage         }"
      "{ cfg            |       | model configuration }"
      "{ model          |       | model weights       }"
      "{ image          |       | image for detection }"
      "{ min_confidence | 0.24  | min confidence      }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        std::cout << about << std::endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("cfg");
    String modelBinary = parser.get<string>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }

    cv::Mat frame = cv::imread(parser.get<string>("image"));

    //! [Resizing without keeping aspect ratio]
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(network_width, network_height));
    //! [Resizing without keeping aspect ratio]

    //! [Prepare blob]
    Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob, "data");                //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    cv::Mat detectionMat = net.forward("detection_out");	//compute output
   //! [Make forward pass]


    float confidenceThreshold = parser.get<float>("min_confidence");
    for (int i = 0; i < detectionMat.rows; i++)
    {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

        size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        if (confidence > confidenceThreshold)
        {
            float x = detectionMat.at<float>(i, 0);
            float y = detectionMat.at<float>(i, 1);
            float width = detectionMat.at<float>(i, 2);
            float height = detectionMat.at<float>(i, 3);
            float xLeftBottom = (x - width / 2) * frame.cols;
            float yLeftBottom = (y - height / 2) * frame.rows;
            float xRightTop = (x + width / 2) * frame.cols;
            float yRightTop = (y + height / 2) * frame.rows;

            std::cout << "Class: " << objectClass << std::endl;
            std::cout << "Confidence: " << confidence << std::endl;

            std::cout << " " << xLeftBottom
                << " " << yLeftBottom
                << " " << xRightTop
                << " " << yRightTop << std::endl;

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

            rectangle(frame, object, Scalar(0, 255, 0));
        }
    }

    imshow("detections", frame);
    waitKey();

    return 0;
} // main