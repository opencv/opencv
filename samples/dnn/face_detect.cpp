#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        cerr << "Usage " << argv[0] << ": "
             << "<onnx_path> "
             << "<image>\n";
        cerr << "Download the face detection model at https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx\n";
        return -1;
    }

    String onnx_path = argv[1];
    String image_path = argv[2];
    Mat image = imread(image_path);

    float score_thresh = 0.9;
    float nms_thresh = 0.3;
    int top_k = 5000;

    // Initialize FaceDetector
    Ptr<FaceDetector> faceDetector = FaceDetector::create(onnx_path, "", image.size(), score_thresh, nms_thresh, top_k);

    // Forward
    Mat faces;
    faceDetector->detect(image, faces);

    // Visualize results
    const int thickness = 2;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << faces.at<float>(i, 14) << "\n";

        // Draw bounding box
        rectangle(image, Rect2i(faces.at<float>(i, 0), faces.at<float>(i, 1), faces.at<float>(i, 2), faces.at<float>(i, 3)), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(image, Point2i(faces.at<float>(i, 4),  faces.at<float>(i, 5)),  2, Scalar(255,   0,   0), thickness);
        circle(image, Point2i(faces.at<float>(i, 6),  faces.at<float>(i, 7)),  2, Scalar(  0,   0, 255), thickness);
        circle(image, Point2i(faces.at<float>(i, 8),  faces.at<float>(i, 9)),  2, Scalar(  0, 255,   0), thickness);
        circle(image, Point2i(faces.at<float>(i, 10), faces.at<float>(i, 11)), 2, Scalar(255,   0, 255), thickness);
        circle(image, Point2i(faces.at<float>(i, 12), faces.at<float>(i, 13)), 2, Scalar(  0, 255, 255), thickness);
    }

    try
    {
        // Save result image
        std::cout << "Saved to result.jpg\n";
        imwrite("result.jpg", image);
        // Display result image
        namedWindow(image_path, WINDOW_AUTOSIZE);
        imshow(image_path, image);
        waitKey(0);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

