// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef perf::TestBaseWithParam<std::string> VideoCapture_Reading;
typedef perf::TestBaseWithParam<std::string> VideoCapture_Reading_1000_frame;

const string bunny_files[] = {
    "highgui/video/big_buck_bunny.avi",
    "highgui/video/big_buck_bunny.mov",
    "highgui/video/big_buck_bunny.mp4",
#ifndef HAVE_MSMF
    // MPEG2 is not supported by Media Foundation yet
    // http://social.msdn.microsoft.com/Forums/en-US/mediafoundationdevelopment/thread/39a36231-8c01-40af-9af5-3c105d684429
    "highgui/video/big_buck_bunny.mpg",
#endif
    "highgui/video/big_buck_bunny.wmv"
};

PERF_TEST_P(VideoCapture_Reading, ReadFile, testing::ValuesIn(bunny_files) )
{
  string filename = getDataPath(GetParam());

  VideoCapture cap;

  TEST_CYCLE() cap.open(filename);

  SANITY_CHECK_NOTHING();
}

PERF_TEST(VideoCapture_Reading_1000_frame, Get1000frame)
{
    VideoCapture cap1(0);
    VideoCapture cap2(2);

    ASSERT_TRUE(cap1.isOpened());
    ASSERT_TRUE(cap2.isOpened());

    Mat frame1;
    int ITERATION_COUNT = 500;

    //false start
    cap1>>frame1;
    cap2>>frame1;

    TEST_CYCLE() {
        for(int j = 0; j < ITERATION_COUNT; ++j)
        {
            cap1>>frame1;
            cap2>>frame1;
        }
    };
    SANITY_CHECK_NOTHING();
}

PERF_TEST(VideoCapture_Reading_1000_frame, GetWaitAny1000frame)
{
    VideoCapture cap1(0);
    VideoCapture cap2(2);

    ASSERT_TRUE(cap1.isOpened());
    ASSERT_TRUE(cap2.isOpened());

    std::vector<VideoCapture> VCM;

    VCM.push_back(cap1);
    VCM.push_back(cap2);

    std::vector<int> state;

    Mat frame1, frame2;
    std::vector<Mat> forMAt = {frame1, frame2};

    int ITERATION_COUNT = 500;
    int TIMEOUT = -1;

    //false start
    cap1>>frame1;
    cap2>>frame2;

    TEST_CYCLE() {
        int out = 0;
        for(int j = 0; j < ITERATION_COUNT; ++j)
        {
            VideoCapture::waitAny(VCM, state, TIMEOUT);
            for(int i = 0; i < 2; ++i)
            {
                if(state[i] == CAP_CAM_READY)
                {
                  EXPECT_TRUE(VCM[i].retrieve(forMAt[i]));
                }
            }
        }
    };
    SANITY_CHECK_NOTHING();
}

} // namespace
