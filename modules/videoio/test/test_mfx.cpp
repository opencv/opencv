// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(videoio_mfx, read_invalid)
{
    if (!videoio_registry::hasBackend(CAP_INTEL_MFX))
        throw SkipTestException("MediaSDK backend was not found");

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open("nonexistent-file", CAP_INTEL_MFX));
    ASSERT_FALSE(cap.isOpened());
    Mat img;
    ASSERT_NO_THROW(cap >> img);
    ASSERT_TRUE(img.empty());
}

TEST(videoio_mfx, write_invalid)
{
    if (!videoio_registry::hasBackend(CAP_INTEL_MFX))
        throw SkipTestException("MediaSDK backend was not found");

    const string filename = cv::tempfile(".264");
    VideoWriter writer;
    bool res = true;
    ASSERT_NO_THROW(res = writer.open(filename, CAP_INTEL_MFX, VideoWriter::fourcc('H', '2', '6', '4'), 1, Size(641, 480), true));
    EXPECT_FALSE(res);
    EXPECT_FALSE(writer.isOpened());
    ASSERT_NO_THROW(res = writer.open(filename, CAP_INTEL_MFX, VideoWriter::fourcc('H', '2', '6', '4'), 1, Size(640, 481), true));
    EXPECT_FALSE(res);
    EXPECT_FALSE(writer.isOpened());
    ASSERT_NO_THROW(res = writer.open(filename, CAP_INTEL_MFX, VideoWriter::fourcc('A', 'B', 'C', 'D'), 1, Size(640, 480), true));
    EXPECT_FALSE(res);
    EXPECT_FALSE(writer.isOpened());
    ASSERT_NO_THROW(res = writer.open(String(), CAP_INTEL_MFX, VideoWriter::fourcc('H', '2', '6', '4'), 1, Size(640, 480), true));
    EXPECT_FALSE(res);
    EXPECT_FALSE(writer.isOpened());
    ASSERT_NO_THROW(res = writer.open(filename, CAP_INTEL_MFX, VideoWriter::fourcc('H', '2', '6', '4'), 0, Size(640, 480), true));
    EXPECT_FALSE(res);
    EXPECT_FALSE(writer.isOpened());

    ASSERT_NO_THROW(res = writer.open(filename, CAP_INTEL_MFX, VideoWriter::fourcc('H', '2', '6', '4'), 30, Size(640, 480), true));
    ASSERT_TRUE(res);
    ASSERT_TRUE(writer.isOpened());
    Mat t;
    // write some bad frames
    t = Mat(Size(1024, 768), CV_8UC3);
    EXPECT_NO_THROW(writer << t);
    t = Mat(Size(320, 240), CV_8UC3);
    EXPECT_NO_THROW(writer << t);
    t = Mat(Size(640, 480), CV_8UC2);
    EXPECT_NO_THROW(writer << t);

    // cleanup
    ASSERT_NO_THROW(writer.release());
    remove(filename.c_str());
}


//==================================================================================================

const int FRAME_COUNT = 20;

inline void generateFrame(int i, Mat & frame)
{
    ::generateFrame(i, FRAME_COUNT, frame);
}

inline int fourccByExt(const String &ext)
{
    if (ext == ".mpeg2")
        return VideoWriter::fourcc('M', 'P', 'G', '2');
    else if (ext == ".264")
        return VideoWriter::fourcc('H', '2', '6', '4');
    else if (ext == ".265")
        return VideoWriter::fourcc('H', '2', '6', '5');
    return -1;
}

//==================================================================================================

typedef tuple<Size, double, const char *> Size_FPS_Ext;
typedef testing::TestWithParam< Size_FPS_Ext > videoio_mfx;

TEST_P(videoio_mfx, read_write_raw)
{
    if (!videoio_registry::hasBackend(CAP_INTEL_MFX))
        throw SkipTestException("MediaSDK backend was not found");

    const Size FRAME_SIZE = get<0>(GetParam());
    const double FPS = get<1>(GetParam());
    const char *ext = get<2>(GetParam());
    const String filename = cv::tempfile(ext);
    const int fourcc = fourccByExt(ext);

    bool isColor = true;
    std::queue<Mat> goodFrames;

    // Write video
    VideoWriter writer;
    writer.open(filename, CAP_INTEL_MFX, fourcc, FPS, FRAME_SIZE, isColor);
    ASSERT_TRUE(writer.isOpened());
    Mat frame(FRAME_SIZE, CV_8UC3);
    for (int i = 0; i < FRAME_COUNT; ++i)
    {
        generateFrame(i, frame);
        goodFrames.push(frame.clone());
        writer << frame;
    }
    writer.release();
    EXPECT_FALSE(writer.isOpened());

    // Read video
    VideoCapture cap;
    cap.open(filename, CAP_INTEL_MFX);
    ASSERT_TRUE(cap.isOpened());
    EXPECT_EQ(FRAME_SIZE.width, cap.get(CAP_PROP_FRAME_WIDTH));
    EXPECT_EQ(FRAME_SIZE.height, cap.get(CAP_PROP_FRAME_HEIGHT));
    for (int i = 0; i < FRAME_COUNT; ++i)
    {
        ASSERT_TRUE(cap.read(frame));
        ASSERT_FALSE(frame.empty());
        ASSERT_EQ(FRAME_SIZE.width, frame.cols);
        ASSERT_EQ(FRAME_SIZE.height, frame.rows);
        // verify
        ASSERT_NE(goodFrames.size(), 0u);
        const Mat &goodFrame = goodFrames.front();
        EXPECT_EQ(goodFrame.depth(), frame.depth());
        EXPECT_EQ(goodFrame.channels(), frame.channels());
        EXPECT_EQ(goodFrame.type(), frame.type());
        double psnr = cvtest::PSNR(goodFrame, frame);
        if (fourcc == VideoWriter::fourcc('M', 'P', 'G', '2'))
            EXPECT_GT(psnr, 31); // experimentally chosen value
        else
            EXPECT_GT(psnr, 33); // experimentally chosen value
        goodFrames.pop();
    }
    EXPECT_FALSE(cap.read(frame));
    EXPECT_TRUE(frame.empty());
    cap.release();
    EXPECT_FALSE(cap.isOpened());
    remove(filename.c_str());
}

inline static std::string videoio_mfx_name_printer(const testing::TestParamInfo<videoio_mfx::ParamType>& info)
{
    std::ostringstream out;
    const Size sz = get<0>(info.param);
    const std::string ext = get<2>(info.param);
    out << sz.width << "x" << sz.height << "x" << get<1>(info.param) << "x" << ext.substr(1, ext.size() - 1);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(videoio, videoio_mfx,
                        testing::Combine(
                            testing::Values(Size(640, 480), Size(638, 478), Size(636, 476), Size(1920, 1080)),
                            testing::Values(1, 30, 100),
                            testing::Values(".mpeg2", ".264", ".265")),
                        videoio_mfx_name_printer);

}} // namespace
