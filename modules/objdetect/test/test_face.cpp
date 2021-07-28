// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test{ namespace {

// label format:
//   image_name
//   num_face
//   face_1
//   face_..
//   face_num
std::map<String, Mat> blobFromTXT(const String& path, int numCoords)
{
    std::ifstream ifs(path.c_str());
    CV_Assert(ifs.is_open());

    std::map<String, Mat> gt;

    Mat faces;
    int faceNum = -1;
    int faceCount = 0;
    for (String line, key; getline(ifs, line); )
    {
        std::istringstream iss(line);
        if (line.find(".png") != std::string::npos)
        {
            // Get filename
            iss >> key;
        }
        else if (line.find(" ") == std::string::npos)
        {
            // Get the number of faces
            iss >> faceNum;
        }
        else
        {
            // Get faces
            Mat face(1, numCoords, CV_32FC1);
            for (int j = 0; j < numCoords; j++)
            {
                iss >> face.at<float>(0, j);
            }
            faces.push_back(face);
            faceCount++;
        }

        if (faceCount == faceNum)
        {
            // Store faces
            gt[key] = faces;

            faces.release();
            faceNum = -1;
            faceCount = 0;
        }
    }

    return gt;
}

typedef testing::TestWithParam<String> Objdetect_face_detect;
TEST(Objdetect_face_detect, regression)
{
    // Pre-set params
    float scoreThreshold = 0.7;
    double matchThreshold = 0.9;
    double l2disThreshold = 5;
    int numLM = 5;
    int numCoords = 4 + 2 * numLM;

    // Load ground truth labels
    std::map<String, Mat> gt = blobFromTXT(findDataFile("dnn_face/face_detection/cascades_label.txt", true), numCoords);
    // for (auto item: gt)
    // {
    //     std::cout << item.first << " " << item.second.size() << std::endl;
    // }

    // Initialize detector
    String model = findDataFile("dnn_face/face_detection/yunet.onnx", true);
    Ptr<FaceDetector> faceDetector = FaceDetector::create(model, "", Size(300, 300));
    faceDetector->setScoreThreshold(0.7);

    // Detect and match
    for (auto item: gt)
    {
        String imagePath = findDataFile("cascadeandhog/images/" + item.first, true);
        Mat image = imread(imagePath);

        // Set input size
        faceDetector->setInputSize(image.size());

        // Run detection
        Mat faces;
        faceDetector->detect(image, faces);
        // std::cout << item.first << " " << item.second.rows << " " << faces.rows << std::endl;

        // Match bboxes and landmarks
        std::vector<bool> matchedItem(item.second.rows, false);
        for (int i = 0; i < faces.rows; i++)
        {
            if (faces.at<float>(i, numCoords) < scoreThreshold)
                continue;

            bool boxMatched = false;
            std::vector<bool> lmMatched(numLM, false);
            cv::Rect2f resBox(faces.at<float>(i, 0), faces.at<float>(i, 1), faces.at<float>(i, 2), faces.at<float>(i, 3));
            for (int j = 0; j < item.second.rows && !boxMatched; j++)
            {
                if (matchedItem[j])
                    continue;

                // Retrieve bbox and compare IoU
                cv::Rect2f gtBox(item.second.at<float>(j, 0), item.second.at<float>(j, 1), item.second.at<float>(j, 2), item.second.at<float>(j, 3));
                double interArea = (resBox & gtBox).area();
                double iou = interArea / (resBox.area() + gtBox.area() - interArea);
                if (iou >= matchThreshold)
                {
                    boxMatched = true;
                    matchedItem[j] = true;
                }

                // Match landmarks if bbox is matched
                if (!boxMatched)
                    continue;
                for (int lmIdx = 0; lmIdx < numLM; lmIdx++)
                {
                    float gtX = item.second.at<float>(j, 4 + 2 * lmIdx);
                    float gtY = item.second.at<float>(j, 4 + 2 * lmIdx + 1);
                    float resX = faces.at<float>(i, 4 + 2 * lmIdx);
                    float resY = faces.at<float>(i, 4 + 2 * lmIdx + 1);
                    float l2dis = cv::sqrt((gtX - resX) * (gtX - resX) + (gtY - resY) * (gtY - resY));

                    if (l2dis <= l2disThreshold)
                    {
                        lmMatched[lmIdx] = true;
                    }
                }
            }
            EXPECT_TRUE(boxMatched) << "In image " << item.first << ", cannot match resBox " << resBox << " with any ground truth.";
            if (boxMatched)
            {
                EXPECT_TRUE(std::all_of(lmMatched.begin(), lmMatched.end(), [](bool v) { return v; })) << "In image " << item.first << ", resBox " << resBox << " matched but its landmarks failed to match.";
            }
        }
    }
}


}} // namespace ; namespace opencv_test