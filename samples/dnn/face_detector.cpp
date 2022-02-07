#include <iostream>
#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void visualize(
    cv::Mat& image,
    const std::vector<float> confidences,
    const std::vector<cv::Rect> boxes,
    const std::vector<std::vector<cv::Point>> landmarks,
    const double fps,
    const int thickness = 2
)
{
    cv::String fps_string = cv::format("fps : %.2lf", fps);
    std::cout << fps_string << std::endl;
    cv::putText(image, fps_string, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    const int num_faces = boxes.size();
    for (int i = 0; i < num_faces; i++)
    {
        const float confidence = confidences[i];
        const cv::Rect& box = boxes[i];
        const std::vector<cv::Point>& landmark = landmarks[i];

        const cv::String result = cv::format("face%d : ( %d, %d ), %d x %d, %.2f", i, box.x, box.y, box.width, box.height, confidence);
        std::cout << result << std::endl;

        // Draw Bounding Box
        const cv::Scalar color = cv::Scalar(0, 255, 0);
        cv::rectangle(image, box, color, thickness);

        // Draw Landmarks
        const int radius = 2;
        for (const cv::Point& point : landmark){
            cv::circle(image, point, radius, color, thickness);
        }
    }
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h                    |                                     | Print this message                                                                                                     }"
        "{image i                   |                                     | Path to the input image. Omit for detecting through VideoCapture                                                       }"
        "{video v                   | 0                                   | Path to the input video or camera index                                                                                }"
        "{scale s                   | 1.0                                 | Scale factor used to resize input video frames (0.0-1.0]                                                               }"
        "{face_detection_model fd   | face_detection_yunet_2021dec.onnx   | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet }"
        "{score_threshold st        | 0.9                                 | Filter out faces of score < score_threshold                                                                            }"
        "{nms_threshold nt          | 0.3                                 | Suppress bounding boxes of iou >= nms_threshold                                                                        }"
    );
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        std::cout << "e.g. face_detector.exe --image=./image.jpg --face_detection_model=./face_detection_yunet_2021dec.onnx" << std::endl;
        return 0;
    }

    const float scale = parser.get<float>("scale");
    const float score_threshold = parser.get<float>("score_threshold");
    const float nms_threshold = parser.get<float>("nms_threshold");

    //! [Create_FaceDetectionModel_YN]
    // Create FaceDetectionModel_YN
    const cv::String face_detection_model_path = parser.get<cv::String>("face_detection_model");
    cv::dnn::FaceDetectionModel_YN face_detector = cv::dnn::FaceDetectionModel_YN(face_detection_model_path);
    //! [Create_FaceDetectionModel_YN]

    cv::TickMeter tm;

    if (parser.has("image"))
    {
        const cv::String image_path = parser.get<cv::String>("image");
        cv::Mat image = cv::imread(cv::samples::findFile(image_path));
        if (image.empty())
        {
            std::cerr << "can't read image: " << image_path << std::endl;
            return -1;
        }

        cv::resize(image, image, cv::Size(), scale, scale);

        tm.reset();
        tm.start();

        //! [Face_Detection]
        // Detect faces from image.
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        face_detector.detect(image, confidences, boxes, score_threshold, nms_threshold);
        //! [Face_Detection]

        //! [Face_Landmarks]
        // Get face landmarks of each faces.
        // If the face detection model have the feature to detect face landmarks, you can get face landmarks using getLandmarks().
        // The number and order of face landmarks varies by model.
        // In the case of YuNet, you can get positions of Right-Eye, Left-Eye, Nose, Right-Mouth Corner, and Right-Mouth Corner.
        std::vector<std::vector<cv::Point>> landmarks;
        face_detector.getLandmarks( landmarks );
        //! [Face_Landmarks]

        tm.stop();

        const double fps = tm.getFPS();
        visualize(image, confidences, boxes, landmarks, fps);

        cv::imshow("face detector", image);
        cv::waitKey(0);
    }
    else if (parser.has("video"))
    {
        cv::VideoCapture capture;
        const cv::String video_path = parser.get<cv::String>("video");
        if (video_path.size() == 1 && std::all_of(video_path.cbegin(), video_path.cend(), std::isdigit))
        {
            capture.open(std::stoi(video_path));
        }
        else
        {
            capture.open(cv::samples::findFileOrKeep(video_path));
        }

        if (!capture.isOpened())
        {
            std::cerr << "can't open video: " << video_path << std::endl;
            return -1;
        }

        for (;;)
        {
            cv::Mat frame;
            const bool result = capture.read(frame);
            if (!result)
            {
                std::cout << "can't grab frame!" << std::endl;
                break;
            }

            cv::resize(frame, frame, cv::Size(), scale, scale);

            tm.start();

            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            face_detector.detect(frame, confidences, boxes, score_threshold, nms_threshold);

            std::vector<std::vector<cv::Point>> landmarks;
            face_detector.getLandmarks( landmarks );

            tm.stop();

            const double fps = tm.getFPS();
            visualize( frame, confidences, boxes, landmarks, fps);

            cv::imshow("face detector", frame );
            const int key = cv::waitKey(1);
            if (key != -1)
            {
                break;
            }
        }
    }
    return 0;
}
