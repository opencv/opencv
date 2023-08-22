#include "test_precomp.hpp"
#include "opencv2/video/tracking.hpp"
#include <unordered_map>

// #include <fstream> // Include the header for file input/output

namespace opencv_test { namespace {


class CV_ByteTrackerTest : public cvtest::BaseTest
{
public:
    CV_ByteTrackerTest();

protected:
    void run(int);
};

CV_ByteTrackerTest::CV_ByteTrackerTest()
{
}

void CV_ByteTrackerTest::run(int)
{
    int code = cvtest::TS::OK;

    // Create ByteTracker instance with parameters
    cv::ByteTracker::Params params;
    params.frameRate = 30;
    params.frameBuffer = 30;
    cv::Ptr<cv::ByteTracker> tracker = cv::ByteTracker::create(params);
    cv::Mat trackedResultMat;
    cv::Mat referenceResultMat;
    std::string referenceLine;

    // Read detections from a file
    std::ifstream detectionFile("detections.txt");
    std::ifstream referenceFile("reference_results.txt");
    if (!detectionFile.is_open() || !referenceFile.is_open())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    while (std::getline(referenceFile, referenceLine))
    {
        // Parse the reference line to extract information
        std::istringstream iss(referenceLine);
        int frame, trackId;
        float x, y, width, height, score, classId;
        iss >> frame >> trackId >> x >> y >> width >> height >> score >> classId;

        cv::Mat row(1, 8, CV_32F);
        row.at<float>(0, 0) = frame;
        row.at<float>(0, 1) = trackId;
        row.at<float>(0, 2) = x;
        row.at<float>(0, 3) = y;
        row.at<float>(0, 4) = width;
        row.at<float>(0, 5) = height;
        row.at<float>(0, 6) = score;
        row.at<float>(0, 7) = classId;

        // Append the reference bounding box to the referenceResultMat
        if (!referenceResultMat.empty())
        {
            referenceResultMat.push_back(row);
        }
        else
        {
            referenceResultMat = row.clone();
        }
    }

    referenceFile.close();

    // Create a map to store detections grouped by frame
    std::map<int, std::vector<cv::Rect>> detectionsByFrame;

    cv::Mat frameDetection;
    std::string line;

    std::unordered_map<int, cv::Mat> frameDetections;

    while (std::getline(detectionFile, line))
    {
        int frame, trackId, classId;
        float x, y, width, height, score;
        sscanf(line.c_str(), "%d %d %f %f %f %f %f %d", &frame, &trackId, &x, &y, &width, &height, &score, &classId);
        cv::Mat detectionRow(1, 6, CV_32F);
        detectionRow.at<float>(0, 0) = x;
        detectionRow.at<float>(0, 1) = y;
        detectionRow.at<float>(0, 2) = width;
        detectionRow.at<float>(0, 3) = height;
        detectionRow.at<float>(0, 4) = classId;
        detectionRow.at<float>(0, 5) = score;

        if (frameDetections.find(frame) == frameDetections.end())
        {
            frameDetections[frame] = detectionRow;
        }
        else
        {
            cv::Mat &frameMatrix = frameDetections[frame];
            cv::vconcat(frameMatrix, detectionRow, frameMatrix);
        }
    }
    detectionFile.close();

    bool result = false;
    for (const auto &pair : frameDetections)
    {
        cv::Mat detectionsMat = pair.second;

        cv::Mat trackedObjects;
        bool ok = tracker->update(detectionsMat, trackedObjects);
        result |= ok;
        if (trackedResultMat.empty())
        {
            trackedResultMat = trackedObjects.clone();
        }
        else
        {
            cv::vconcat(trackedResultMat, trackedObjects, trackedResultMat);
        }
    }


    // Compare the trackedObjects with the referenceBoundingBox
    ASSERT_EQ(result, true);

    float eps = 0.001;
    ASSERT_EQ(trackedResultMat.size(), referenceResultMat.size());
    for (int i = 0; i < trackedResultMat.rows; ++i)
    {
        for (int j = 0; j < trackedResultMat.cols; ++j)
        {
            ASSERT_NEAR(trackedResultMat.at<float>(i,j), referenceResultMat.at<float>(i,j), eps);
        }

    }

    if (code < 0)
        ts->set_failed_test_info(code);
}

TEST(Video_ByteTracker, accuracy){ CV_ByteTrackerTest test; test.safe_run(); }

}}// namespace
