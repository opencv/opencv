// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

namespace opencv_test {

//#define DEBUG_BGFG

using namespace testing;
using namespace cvtest;
using namespace perf;

namespace {

using namespace cv;

static void cvtFrameFmt(std::vector<Mat>& input, std::vector<Mat>& output)
{
    for(int i = 0; i< (int)(input.size()); i++)
    {
        cvtColor(input[i], output[i], COLOR_RGB2GRAY);
    }
}

static void prepareData(VideoCapture& cap, int cn, std::vector<Mat>& frame_buffer, int skipFrames = 0)
{
    std::vector<Mat> frame_buffer_init;
    int nFrame = (int)frame_buffer.size();
    for (int i = 0; i < skipFrames; i++)
    {
        cv::Mat frame;
        cap >> frame;
    }
    for (int i = 0; i < nFrame; i++)
    {
        cv::Mat frame;
        cap >> frame;
        ASSERT_FALSE(frame.empty());
        frame_buffer_init.push_back(frame);
    }

    if (cn == 1)
        cvtFrameFmt(frame_buffer_init, frame_buffer);
    else
        frame_buffer.swap(frame_buffer_init);
}

}}
