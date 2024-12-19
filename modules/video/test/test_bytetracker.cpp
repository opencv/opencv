#include "test_precomp.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video.hpp"
#include <map>
#include <string>
#include <fstream> // Include the header for file input/output

namespace opencv_test { namespace {

static std::string getDataDir() { return TS::ptr()->get_data_path(); }

std::string getDetections() { return getDataDir() + "bytetracker/detFile.txt"; }
std::string getReference() { return getDataDir() + "bytetracker/newRef.txt"; }

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
    std::map<int,cv::Mat> trackedMap;
    std::map<int,cv::Mat> referenceMap;
    std::string referenceLine;

    // Read detections from a file
    std::ifstream detectionFile(getDetections());
    std::ifstream referenceFile(getReference());

    if (!detectionFile.is_open() || !referenceFile.is_open())
    {
        cout<<"Invalid test data";
        cout<<detectionFile.is_open();
        cout<<"\n"<<referenceFile.is_open();
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    cv::Mat frameRows;

    while (std::getline(referenceFile, referenceLine))
    {
        // Parse the reference line to extract information
        //std::istringstream iss(referenceLine);
        int frame, trackId, classId;
        float x, y, width, height, score;
        sscanf(referenceLine.c_str(), "%d,%d,%f,%f,%f,%f,%f,%d", &frame, &trackId, &x, &y, &width, &height, &score, &classId);

        cv::Mat row(1, 8, CV_32F);
        row.at<float>(0, 0) = static_cast<float>(frame);
        row.at<float>(0, 1) = static_cast<float>(trackId);
        row.at<float>(0, 2) = x;
        row.at<float>(0, 3) = y;
        row.at<float>(0, 4) = width;
        row.at<float>(0, 5) = height;
        row.at<float>(0, 6) = score;
        row.at<float>(0, 7) = 0; //actually im receiving a -1 because the model is trained with only people

            // Append the reference bounding box to the referenceResultMat
        if (referenceMap.find(frame) == referenceMap.end())
        {
            // Define the range of indexes to append
            int startIdx = 2;
            int endIdx = 5;

            // Iterate through the specified range of indexes and append to the map
            for (int idx = startIdx; idx <= endIdx; ++idx)
            {
                referenceMap[idx] = row.at<int>(0, idx);
            }
        }
        else
        {
            cv::Mat newRow(1, 4, CV_32F);
            for (int i = 0; i < newRow.cols; ++i)
            {
                newRow.at<int>(0, i) = static_cast<int>(row.at<float>(0,2+i));
                referenceMap[frame].push_back(newRow);
            }
        }

    }

    referenceFile.close();

    //cout<<referenceResultMat;

    // Create a map to store detections grouped by frame
    std::map<int, std::vector<cv::Rect>> detectionsByFrame;

    cv::Mat frameDetection;
    std::string line;

    std::map<int, cv::Mat> frameDetections;

    while (std::getline(detectionFile, line))
    {
        int frame, trackId, classId;
        float x, y, width, height, score;
        //std::cout << "\nLine content: " << line << "\n";
        //std::cout<<line.size()<<"\n";
        sscanf(line.c_str(), "%d,%d,%f,%f,%f,%f,%f,%d", &frame, &trackId, &x, &y, &width, &height, &score, &classId);

        cv::Mat detectionRow(1, 6, CV_32F);

        detectionRow.at<float>(0, 0) = x;
        detectionRow.at<float>(0, 1) = y;
        detectionRow.at<float>(0, 2) = width;
        detectionRow.at<float>(0, 3) = height;
        detectionRow.at<float>(0, 4) = static_cast<float>(classId);
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

        for (int i =0; i < trackedObjects.rows; ++i)
        {
            cv::Mat row = trackedObjects.row(i);
            float x = row.at<float>(0, 0);
            float y = row.at<float>(0, 1);
            float width = row.at<float>(0, 2);
            float height = row.at<float>(0, 3);
            float classId = row.at<float>(0, 4);
            float score = row.at<float>(0, 5);
            float trackId = row.at<float>(0, 6);

            cv::Mat outputRow(1, 8, CV_32F);
            cv::Mat mapRow(1 ,4 ,CV_32F);
            outputRow.at<float>(0, 0) = static_cast<float>(pair.first);
            outputRow.at<float>(0, 1) = trackId+1; // Assuming i represents trackId
            outputRow.at<float>(0, 2) = x;
            outputRow.at<float>(0, 3) = y;
            outputRow.at<float>(0, 4) = width;
            outputRow.at<float>(0, 5) = height;
            outputRow.at<float>(0, 6) = score;
            outputRow.at<float>(0, 7) = classId;

            if (trackedMap.find(pair.first) == trackedMap.end())
            {
                // Define the range of indexes to append
                int startIdx = 2;
                int endIdx = 5;

                // Iterate through the specified range of indexes and append to the map
                for (int idx = startIdx; idx <= endIdx; ++idx)
                {
                    trackedMap[idx] = outputRow.at<int>(0, idx);
                }
            }
            else
            {
                cv::Mat newRow(1, 4, CV_32F);
                for (int j = 0; j < newRow.cols; ++j)
                {
                    newRow.at<int>(0, j) = static_cast<int>(row.at<float>(0,2+j));
                    trackedMap[pair.first].push_back(newRow);
                }
            }

            if (trackedResultMat.empty())
            {
                trackedResultMat = outputRow.clone();
            }
            else
            {
                cv::vconcat(trackedResultMat, outputRow, trackedResultMat);
            }
        }
    }
    ASSERT_EQ(result, true);

    //float eps = 30;
    //bool printed = false;
    //ASSERT_EQ(trackedResultMat.size(), referenceResultMat.size());

    // for (int i = 1; i < trackedResultMat.rows; ++i)
    // {
    //     /*
    //     if (cv::abs(trackedResultMat.at<float>(i,0) - referenceResultMat.at<float>(i,0)) > eps && !printed)
    //         {
    //             cout<<"\n"<<i<<" "<<0<<"\n"<<trackedResultMat.row(i)<<"\n"<<referenceResultMat.row(i);
    //             /*
    //             cout<<"\n";
    //             cout<<"\n"<<referenceResultMat.row(i+1);
    //             cout<<"\n"<<referenceResultMat.row(i+2);
    //             cout<<"\n"<<referenceResultMat.row(i+3);
    //             cout<<"\n"<<referenceResultMat.row(i+4);
    //             //cout<<"\n"<<referenceResultMat.row(5);
    //             //cout<<"\n"<<referenceResultMat.row(6);
    //             */
    //            //printed = true;
    //         //}

    //     //for (int j = 0; j < trackedResultMat.cols; ++j)
    //     for (int j = 2; j < 6; ++j)
    //     {
    //         if(trackedResultMat.at<float>(i,0) == 41 || trackedResultMat.at<float>(i,0) == 42)
    //         {
    //             cout<<"\n";
    //             cout<<"track"<<trackedResultMat.row(i)<<"\n";
    //             cout<<"ref"<<referenceResultMat.row(i)<<"\n";
    //         }

    //         if (cv::abs(trackedResultMat.at<float>(i,j) - referenceResultMat.at<float>(i,j)) > eps)
    //         {
    //             /*
    //             cout<<"\n"<<i<<" "<<j<<"\n"<<trackedResultMat.row(i)<<"\n"<<referenceResultMat.row(i);

    //             cout<<"\n";
    //             cout<<"\n"<<trackedResultMat.row(i-1);
    //             cout<<"\n"<<trackedResultMat.row(i+1);


    //             cout<<"\n";
    //             cout<<"\n"<<referenceResultMat.row(i-1);
    //             cout<<"\n"<<referenceResultMat.row(i+1);
    //             */

    //         }
    //         eps = cv::abs(0.2 * referenceResultMat.at<float>(i,j));
    //         //cout<<referenceResultMat.row(i);
    //         //cout<<trackedResultMat.row(i);
    //         if (j == 2 || j==3 || j==6)
    //         {
    //             ASSERT_NEAR(trackedResultMat.at<float>(i,j), referenceResultMat.at<float>(i,j), eps);
    //         }
    //     }

    // }
    int counter = 0;
    for (const auto& pair : referenceMap)
    {
        int i = pair.first;
        cv::Mat costMatrix;
        cv::Mat assignedPairs;
        cv::Mat referenceMat = referenceMap[i];
        cv::Mat trackedMat = trackedMap[i];
        costMatrix = tracker->getCostMatrix(trackedMat, referenceMat);
        lapjv(costMatrix, assignedPairs, 0.8f);
        for (int j = 0; j < assignedPairs.rows; j++)
        {
            for (int k = 0; k < assignedPairs.cols; k++)
            {
                if (assignedPairs.at<int>(j,k) == -1)
                {
                    counter++;
                }
                EXPECT_NE(assignedPairs.at<int>(j,k),-1);
            }

        }
    }
    if (counter == 0)
    {
        cout<<"All outputs were matched correctly";
    }

    if (code < 0)
        ts->set_failed_test_info(code);
}

TEST(Video_ByteTracker, accuracy){ CV_ByteTrackerTest test; test.safe_run(); }

}}// namespace
