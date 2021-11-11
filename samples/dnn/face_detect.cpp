#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static Mat visualize(Mat input, Mat faces, int thickness=2)
{
    Mat output = input.clone();
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << faces.at<float>(i, 14) << "\n";

        // Draw bounding box
        rectangle(output, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(output, Point2i(int(faces.at<float>(i, 4)),  int(faces.at<float>(i, 5))),  2, Scalar(255,   0,   0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 6)),  int(faces.at<float>(i, 7))),  2, Scalar(  0,   0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 8)),  int(faces.at<float>(i, 9))),  2, Scalar(  0, 255,   0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255,   0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(  0, 255, 255), thickness);
    }
    return output;
}

int main(int argc, char ** argv)
{
    CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message.}"
        "{input i           |            | Path to the input image. Omit for detecting on default camera.}"
        "{model m           | yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold.}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold.}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS.}"
        "{save  s           | false      | Set true to save results. This flag is invalid when using camera.}"
        "{vis   v           | true       | Set true to open a window for result visualization. This flag is invalid when using camera.}"
    );
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }

    String modelPath = parser.get<String>("model");

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    bool save = parser.get<bool>("save");
    bool vis = parser.get<bool>("vis");

    // Initialize FaceDetectorYN
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);

    // If input is an image
    if (parser.has("input"))
    {
        String input = parser.get<String>("input");
        Mat image = imread(input);

        // Set input size before inference
        detector->setInputSize(image.size());

        // Inference
        Mat faces;
        detector->detect(image, faces);

        // Draw results on the input image
        Mat result = visualize(image, faces);

        // Save results if save is true
        if(save)
        {
            cout << "Results saved to result.jpg\n";
            imwrite("result.jpg", result);
        }

        // Visualize results
        if (vis)
        {
            namedWindow(input, WINDOW_AUTOSIZE);
            imshow(input, result);
            waitKey(0);
        }
    }
    else
    {
        int deviceId = 0;
        VideoCapture cap;
        cap.open(deviceId, CAP_ANY);
        int frameWidth = int(cap.get(CAP_PROP_FRAME_WIDTH));
        int frameHeight = int(cap.get(CAP_PROP_FRAME_HEIGHT));
        detector->setInputSize(Size(frameWidth, frameHeight));

        Mat frame;
        TickMeter tm;
        String msg = "FPS: ";
        while(waitKey(1) < 0) // Press any key to exit
        {
            // Get frame
            if (!cap.read(frame))
            {
                cerr << "No frames grabbed!\n";
                break;
            }

            // Inference
            Mat faces;
            tm.start();
            detector->detect(frame, faces);
            tm.stop();

            // Draw results on the input image
            Mat result = visualize(frame, faces);
            putText(result, msg + to_string(tm.getFPS()), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            // Visualize results
            imshow("Live", result);

            tm.reset();
        }
    }
}