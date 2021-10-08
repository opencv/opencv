// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

#include "opencv2/objdetect.hpp"


using namespace cv;
using namespace std;


int main(int argc, char ** argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<det_onnx_path> "
                  << "<reg_onnx_path> "
                  << "<image1>"
                  << "<image2>\n";
        return -1;
    }

    String det_onnx_path = argv[1];
    String reg_onnx_path = argv[2];
    String image1_path = argv[3];
    String image2_path = argv[4];
    std::cout<<image1_path<<" "<<image2_path<<std::endl;
    Mat image1 = imread(image1_path);
    Mat image2 = imread(image2_path);

    float score_thresh = 0.9f;
    float nms_thresh = 0.3f;
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;
    int top_k = 5000;

    // Initialize FaceDetector
    Ptr<FaceDetectorYN> faceDetector;

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", image1.size(), score_thresh, nms_thresh, top_k);
    Mat faces_1;
    faceDetector->detect(image1, faces_1);
    if (faces_1.rows < 1)
    {
        std::cerr << "Cannot find a face in " << image1_path << "\n";
        return -1;
    }

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", image2.size(), score_thresh, nms_thresh, top_k);
    Mat faces_2;
    faceDetector->detect(image2, faces_2);
    if (faces_2.rows < 1)
    {
        std::cerr << "Cannot find a face in " << image2_path << "\n";
        return -1;
    }

    // Initialize FaceRecognizerSF
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");


    Mat aligned_face1, aligned_face2;
    faceRecognizer->alignCrop(image1, faces_1.row(0), aligned_face1);
    faceRecognizer->alignCrop(image2, faces_2.row(0), aligned_face2);

    Mat feature1, feature2;
    faceRecognizer->feature(aligned_face1, feature1);
    feature1 = feature1.clone();
    faceRecognizer->feature(aligned_face2, feature2);
    feature2 = feature2.clone();

    double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
    double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

    if(cos_score >= cosine_similar_thresh)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities;";
    }
    std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

    if(L2_score <= l2norm_similar_thresh)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities.";
    }
    std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";

    return 0;
}
