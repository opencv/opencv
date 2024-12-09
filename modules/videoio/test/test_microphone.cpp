// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Usage: opencv_test_videoio --gtest_also_run_disabled_tests

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(DISABLED_videoio_micro, basic)
{
    int cursize = 0;
    int validSize = 0;
    Mat frame;

    std::vector<int> params { CAP_PROP_AUDIO_STREAM, 0, CAP_PROP_VIDEO_STREAM, -1 };
    VideoCapture cap(0, cv::CAP_MSMF, params);
    ASSERT_TRUE(cap.isOpened());

    int samplesPerSecond = (int)cap.get(cv::CAP_PROP_AUDIO_SAMPLES_PER_SECOND);
    const int audio_base_index = (int)cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX);

    const double cvTickFreq = cv::getTickFrequency();
    int64 sysTimePrev = cv::getTickCount();
    int64 sysTimeCurr = cv::getTickCount();

    cout << "Audio would be captured for the next 10 seconds" << endl;
    while ((sysTimeCurr-sysTimePrev)/cvTickFreq < 10)
    {
        if (cap.grab())
        {
            ASSERT_TRUE(cap.retrieve(frame, audio_base_index));
            sysTimeCurr = cv::getTickCount();
        }
    }
    validSize = samplesPerSecond*(int)((sysTimeCurr-sysTimePrev)/cvTickFreq);
    cursize = (int)cap.get(cv::CAP_PROP_AUDIO_POS);
    ASSERT_LT(validSize - cursize, cursize*0.05);
}

}} // namespace
