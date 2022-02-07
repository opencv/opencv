#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr double COSINE_THRESHOLD = 0.363;
constexpr double L2NORM_THRESHOLD = 1.128;

void visualize(
    cv::Mat& image1,
    const cv::Rect box1,
    cv::Mat& image2,
    const cv::Rect box2,
    const double score,
    const cv::dnn::FaceRecognitionModel::DisType distance_type = cv::dnn::FaceRecognitionModel::DisType::FR_COSINE,
    const int thickness = 2
)
{
    // Green is same identity, Red is different identities.
    bool same_identity = false;
    if (distance_type == cv::dnn::FaceRecognitionModel::DisType::FR_COSINE)
    {
        same_identity = (score >= L2NORM_THRESHOLD);
    }
    else{
        same_identity = (score <= L2NORM_THRESHOLD);
    }
    color = same_identity ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);

    // Draw Bounding Box
    cv::rectangle(image1, box1, color, thickness);
    cv::rectangle(image2, box2, color, thickness);
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h                    |                                     | Print this message                                                                                                                                       }"
        "{image1 i1                 |                                     | Path to the input image1. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.}"
        "{image2 i2                 |                                     | Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.}"
        "{scale sc                  | 1.0                                 | Scale factor used to resize input video frames (0.0-1.0]                                                                                                 }"
        "{face_detection_model fd   | face_detection_yunet_2021dec.onnx   | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet                                   }"
        "{face_recognition_model fr | face_recognition_sface_2021dec.onnx | Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface                 }"
        "{score_threshold st        | 0.9                                 | Filter out faces of score < score_threshold                                                                                                              }"
        "{nms_threshold nt          | 0.3                                 | Suppress bounding boxes of iou >= nms_threshold                                                                                                          }"
    );
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        std::cout << "e.g. face_recognizer.exe --image1=./image1.jpg --image2=./image2.jpg --face_detection_model=./face_detection_yunet_2021dec.onnx -face_recognition_model=./face_recognition_sface_2021dec.onnx" << std::endl;
        return 0;
    }
    if (!parser.has("image1") || !parser.has("image2"))
    {
        std::cerr << "can't specify image paths" << std::endl;
        return -1;
    }

    const float scale = parser.get<float>("scale");
    const float score_threshold = parser.get<float>("score_threshold");
    const float nms_threshold = parser.get<float>("nms_threshold");

    const cv::String face_detection_model_path = parser.get<cv::String>("face_detection_model");
    cv::dnn::FaceDetectionModel_YN face_detector = cv::dnn::FaceDetectionModel_YN(face_detection_model_path);

    //! [Create_FaceRecognitionModel_SF]
    // Create FaceRecognitionModel_SF
    const cv::String face_recognition_model_path = parser.get<cv::String>("face_recognition_model");
    cv::dnn::FaceRecognitionModel_SF face_recognizer = cv::dnn::FaceRecognitionModel_SF(face_recognition_model_path);
    //! [Create_FaceRecognitionModel_SF]

    const cv::String image1_path = parser.get<cv::String>("image1");
    cv::Mat image1 = cv::imread(cv::samples::findFile(image1_path));
    if (image1.empty())
    {
        std::cerr << "can't read image1: " << image1_path << std::endl;
        return -1;
    }

    cv::resize(image1, image1, cv::Size(), scale, scale);

    std::vector<float> confidences1;
    std::vector<cv::Rect> boxes1;
    face_detector.detect(image1, confidences1, boxes1, score_threshold, nms_threshold);

    std::vector<std::vector<cv::Point>> landmarks1;
    face_detector.getLandmarks( landmarks1 );

    if (landmarks1.size() == 0)
    {
        std::cerr << "can't detect face from image1" << std::endl;
        return -1;
    }

    const cv::String image2_path = parser.get<cv::String>("image2");
    cv::Mat image2 = cv::imread(cv::samples::findFile(image2_path));
    if (image2.empty())
    {
        std::cerr << "can't read image2: " << image2_path << std::endl;
        return -1;
    }

    cv::resize(image2, image2, cv::Size(), scale, scale);

    std::vector<float> confidences2;
    std::vector<cv::Rect> boxes2;
    face_detector.detect(image2, confidences2, boxes2, score_threshold, nms_threshold);

    std::vector<std::vector<cv::Point>> landmarks2;
    face_detector.getLandmarks( landmarks2 );

    if (landmarks2.size() == 0)
    {
        std::cerr << "can't detect face from image2" << std::endl;
        return -1;
    }

    //! [Align_Crop]
    // Aligning and Cropping facial image through the first face of faces detected.
    // In the case of SFace, It use five landmarks that lined up in a specific order.
    // (Right-Eye, Left-Eye, Nose, Right-Mouth Corner, Right-Mouth Corner)
    cv::Mat aligned_face1, aligned_face2;
    face_recognizer.alignCrop(image1, aligned_face1, landmarks1[0]);
    face_recognizer.alignCrop(image2, aligned_face2, landmarks2[0]);
    //! [Align_Crop]

    //! [Extract_Feature]
    // Run feature extraction with given aligned_face.
    cv::Mat face_feature1, face_feature2;
    face_recognizer.feature(aligned_face1, face_feature1);
    face_recognizer.feature(aligned_face2, face_feature2);
    //! [Extract_Feature]

    //! [Match_Features]
    // Match two features using each distance types.
    // * DisType::FR_COSINE : Cosine similarity. Higher value means higher similarity. (max 1.0)
    // * DisType::FR_NORM_L2 : L2-Norm distance. Lower value means higher similarity. (min 0.0)
    const double cos_score = face_recognizer.match(face_feature1, face_feature2, cv::dnn::FaceRecognitionModel::DisType::FR_COSINE);
    const double l2norm_score = face_recognizer.match(face_feature1, face_feature2, cv::dnn::FaceRecognitionModel::DisType::FR_NORM_L2);
    //! [Match_Features]

    //! [Check_Identity]
    // Check identity using Cosine similarity.
    if (cos_score >= COSINE_THRESHOLD)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities;";
    }
    std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << COSINE_THRESHOLD << ". (higher value means higher similarity, max 1.0)" << std::endl;

    // Check identity using L2-Norm distance.
    if (l2norm_score <= L2NORM_THRESHOLD)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities.";
    }
    std::cout << " L2-Norm Distance: " << l2norm_score << ", threshold: " << L2NORM_THRESHOLD << ". (lower value means higher similarity, min 0.0)" << std::endl;
    //! [Check_Identity]

    visualize(image1, boxes1[0], image2, boxes2[0], cos_score);

    cv::imshow("face recognizer (image1)", image1);
    cv::imshow("face recognizer (image2)", image2);
    cv::waitKey(0);

    return 0;
}
