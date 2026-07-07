// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace opencv_test { namespace {

#ifdef HAVE_OPENCV_DNN
static std::vector<uchar> readModel(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    CV_Assert(stream.is_open());
    return std::vector<uchar>((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
}

template<typename Operation>
static bool succeeds(const Operation& operation)
{
    try
    {
        operation();
        return true;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
}

TEST(Objdetect_face, dnn_engine_selection)
{
    const std::string model = findDataFile("dnn/onnx/models/face_recognizer_fast.onnx", true);
    const std::string detectorModel = findDataFile("dnn/onnx/models/yunet-202303.onnx", true);
    const std::vector<uchar> buffer = readModel(model);
    const std::vector<uchar> detectorBuffer = readModel(detectorModel);
    const std::vector<uchar> emptyBuffer;
    const Size inputSize(32, 32);

    // Existing API remains equivalent to explicit default selection.
    EXPECT_NO_THROW(FaceDetectorYN::create(detectorModel, "", inputSize));
    EXPECT_NO_THROW(FaceDetectorYN::create(detectorModel, "", inputSize, 0.9f, 0.3f, 5000, 0, 0,
                                           dnn::ENGINE_AUTO));
    EXPECT_NO_THROW(FaceDetectorYN::create("onnx", detectorBuffer, emptyBuffer, inputSize));
    EXPECT_NO_THROW(FaceDetectorYN::create("onnx", detectorBuffer, emptyBuffer, inputSize,
                                           0.9f, 0.3f, 5000, 0, 0, dnn::ENGINE_AUTO));
    EXPECT_NO_THROW(FaceRecognizerSF::create(model, ""));
    EXPECT_NO_THROW(FaceRecognizerSF::create(model, "", 0, 0, dnn::ENGINE_AUTO));
    EXPECT_NO_THROW(FaceRecognizerSF::create("onnx", buffer, emptyBuffer));
    EXPECT_NO_THROW(FaceRecognizerSF::create("onnx", buffer, emptyBuffer, 0, 0,
                                             dnn::ENGINE_AUTO));

    const int engines[] = { dnn::ENGINE_CLASSIC, dnn::ENGINE_NEW,
                            dnn::ENGINE_AUTO, dnn::ENGINE_ORT };
    for (int engine : engines)
    {
        SCOPED_TRACE(format("engine=%d", engine));

        // Direct DNN calls are the behavior oracle for each face-factory route.
        const bool directFile = succeeds([&]() { dnn::readNet(model, "", "", engine); });
        EXPECT_EQ(directFile, succeeds([&]() {
            FaceRecognizerSF::create(model, "", 0, 0, engine);
        }));
        const bool directBuffer = succeeds([&]() {
            dnn::readNet("onnx", buffer, emptyBuffer, engine);
        });
        EXPECT_EQ(directBuffer, succeeds([&]() {
            FaceRecognizerSF::create("onnx", buffer, emptyBuffer, 0, 0, engine);
        }));

        const bool directDetectorFile = succeeds([&]() {
            dnn::readNet(detectorModel, "", "", engine);
        });
        // Classic YuNet preserves the factory's existing fixed-input-shape error.
        // This also makes selection of Classic observable without inspecting Net internals.
        if (engine == dnn::ENGINE_CLASSIC)
            EXPECT_THROW(FaceDetectorYN::create(detectorModel, "", inputSize,
                                                0.9f, 0.3f, 5000, 0, 0, engine), cv::Exception);
        else
            EXPECT_EQ(directDetectorFile, succeeds([&]() {
                FaceDetectorYN::create(detectorModel, "", inputSize,
                                       0.9f, 0.3f, 5000, 0, 0, engine);
            }));

        const bool directDetectorBuffer = succeeds([&]() {
            dnn::readNet("onnx", detectorBuffer, emptyBuffer, engine);
        });
        if (engine == dnn::ENGINE_CLASSIC)
            EXPECT_THROW(FaceDetectorYN::create("onnx", detectorBuffer, emptyBuffer, inputSize,
                                                0.9f, 0.3f, 5000, 0, 0, engine), cv::Exception);
        else
            EXPECT_EQ(directDetectorBuffer, succeeds([&]() {
                FaceDetectorYN::create("onnx", detectorBuffer, emptyBuffer, inputSize,
                                       0.9f, 0.3f, 5000, 0, 0, engine);
            }));
    }
}

TEST(Objdetect_face, invalid_engine_propagation)
{
    const std::string model = findDataFile("dnn/onnx/models/face_recognizer_fast.onnx", true);
    const std::string detectorModel = findDataFile("dnn/onnx/models/yunet-202303.onnx", true);
    const std::vector<uchar> buffer = readModel(model);
    const std::vector<uchar> detectorBuffer = readModel(detectorModel);
    const std::vector<uchar> emptyBuffer;
    const Size inputSize(32, 32);
    // EngineType values 1 through 4 are valid; zero is unsupported.
    const int invalidEngine = 0;

    EXPECT_THROW(dnn::readNet(model, "", "", invalidEngine), cv::Exception);
    EXPECT_THROW(dnn::readNet("onnx", buffer, emptyBuffer, invalidEngine), cv::Exception);
    EXPECT_THROW(FaceRecognizerSF::create(model, "", 0, 0, invalidEngine), cv::Exception);
    EXPECT_THROW(FaceRecognizerSF::create("onnx", buffer, emptyBuffer, 0, 0, invalidEngine),
                 cv::Exception);

    EXPECT_THROW(dnn::readNet(detectorModel, "", "", invalidEngine), cv::Exception);
    EXPECT_THROW(dnn::readNet("onnx", detectorBuffer, emptyBuffer, invalidEngine), cv::Exception);
    EXPECT_THROW(FaceDetectorYN::create(detectorModel, "", inputSize,
                                        0.9f, 0.3f, 5000, 0, 0, invalidEngine), cv::Exception);
    EXPECT_THROW(FaceDetectorYN::create("onnx", detectorBuffer, emptyBuffer, inputSize,
                                        0.9f, 0.3f, 5000, 0, 0, invalidEngine), cv::Exception);
}
#endif  // HAVE_OPENCV_DNN

// label format:
//   image_name
//   num_face
//   face_1
//   face_..
//   face_num
std::map<std::string, Mat> blobFromTXT(const std::string& path, int numCoords)
{
    std::ifstream ifs(path.c_str());
    CV_Assert(ifs.is_open());

    std::map<std::string, Mat> gt;

    Mat faces;
    int faceNum = -1;
    int faceCount = 0;
    for (std::string line, key; getline(ifs, line); )
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

TEST(Objdetect_face_detection, regression)
{
    // Pre-set params
    float scoreThreshold = 0.7f;
    float matchThreshold = 0.7f;
    float l2disThreshold = 15.0f;
    int numLM = 5;
    int numCoords = 4 + 2 * numLM;

    // Load ground truth labels
    std::map<std::string, Mat> gt = blobFromTXT(findDataFile("dnn_face/detection/cascades_labels.txt"), numCoords);

    // Initialize detector
    std::string model = findDataFile("dnn/onnx/models/yunet-202303.onnx", false);
    Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(model, "", Size(300, 300));
    faceDetector->setScoreThreshold(0.7f);

    // Detect and match
    for (auto item: gt)
    {
        std::string imagePath = findDataFile("cascadeandhog/images/" + item.first);
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
                break;
            }
            EXPECT_TRUE(boxMatched) << "In image " << item.first << ", cannot match resBox " << resBox << " with any ground truth.";
            if (boxMatched)
            {
                EXPECT_TRUE(std::all_of(lmMatched.begin(), lmMatched.end(), [](bool v) { return v; })) << "In image " << item.first << ", resBox " << resBox << " matched but its landmarks failed to match.";
            }
        }
    }
}

TEST(Objdetect_face_recognition, regression)
{
    // Pre-set params
    float score_thresh = 0.9f;
    float nms_thresh = 0.3f;
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;

    // Load ground truth labels
    std::ifstream ifs(findDataFile("dnn_face/recognition/cascades_label.txt").c_str());
    CV_Assert(ifs.is_open());

    std::set<std::string> fSet;
    std::map<std::string, Mat> featureMap;
    std::map<std::pair<std::string, std::string>, int> gtMap;


    for (std::string line, key; getline(ifs, line);)
    {
        std::string fname1, fname2;
        int label;
        std::istringstream iss(line);
        iss>>fname1>>fname2>>label;
        // std::cout<<fname1<<" "<<fname2<<" "<<label<<std::endl;

        fSet.insert(fname1);
        fSet.insert(fname2);
        gtMap[std::make_pair(fname1, fname2)] = label;
    }

    // Initialize detector
    std::string detect_model = findDataFile("dnn/onnx/models/yunet-202303.onnx", false);
    Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(detect_model, "", Size(150, 150), score_thresh, nms_thresh);

    std::string recog_model = findDataFile("dnn/onnx/models/face_recognizer_fast.onnx", false);
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(recog_model, "");

    // Detect and match
    for (auto fname: fSet)
    {
        std::string imagePath = findDataFile("dnn_face/recognition/" + fname);
        Mat image = imread(imagePath);

        Mat faces;
        faceDetector->detect(image, faces);

        ASSERT_EQ(faces.rows, 1);

        Mat aligned_face;
        faceRecognizer->alignCrop(image, faces.row(0), aligned_face);

        Mat feature;
        faceRecognizer->feature(aligned_face, feature);

        featureMap[fname] = feature.clone();
    }

    for (auto item: gtMap)
    {
        Mat feature1 = featureMap[item.first.first];
        Mat feature2 = featureMap[item.first.second];
        int label = item.second;

        double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
        double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

        EXPECT_TRUE(label == 0 ? cos_score <= cosine_similar_thresh : cos_score > cosine_similar_thresh) << "Cosine match result of images " << item.first.first << " and " << item.first.second << " is different from ground truth (score: "<< cos_score <<";Thresh: "<< cosine_similar_thresh <<").";
        EXPECT_TRUE(label == 0 ? L2_score > l2norm_similar_thresh : L2_score <= l2norm_similar_thresh) << "L2norm match result of images " << item.first.first << " and " << item.first.second << " is different from ground truth (score: "<< L2_score <<";Thresh: "<< l2norm_similar_thresh <<").";
    }
}

}} // namespace
