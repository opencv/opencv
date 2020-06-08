/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_FFMPEG

using namespace std;

static const char* AVI_EXT = ".avi";
static const char* MP4_EXT = ".mp4";

class CV_FFmpegWriteBigVideoTest : public cvtest::BaseTest
{
    struct TestFormatEntry {
        int tag;
        const char* ext;
        bool required;
    };

    static long int getFileSize(string filename)
    {
        FILE *p_file = NULL;
        p_file = fopen(filename.c_str(), "rb");
        if (p_file == NULL)
            return -1;
        fseek(p_file, 0, SEEK_END);
        long int size = ftell(p_file);
        fclose(p_file);
        return size;
    }
public:
    void run(int)
    {
        const int img_r = 4096;
        const int img_c = 4096;
        const double fps0 = 15;
        const double time_sec = 1;

        const TestFormatEntry entries[] = {
            {0, AVI_EXT, true},
            //{VideoWriter::fourcc('D', 'I', 'V', '3'), AVI_EXT, true},
            //{VideoWriter::fourcc('D', 'I', 'V', 'X'), AVI_EXT, true},
            {VideoWriter::fourcc('D', 'X', '5', '0'), AVI_EXT, true},
            {VideoWriter::fourcc('F', 'L', 'V', '1'), AVI_EXT, true},
            {VideoWriter::fourcc('H', '2', '6', '1'), AVI_EXT, true},
            {VideoWriter::fourcc('H', '2', '6', '3'), AVI_EXT, true},
            {VideoWriter::fourcc('I', '4', '2', '0'), AVI_EXT, true},
            //{VideoWriter::fourcc('j', 'p', 'e', 'g'), AVI_EXT, true},
            {VideoWriter::fourcc('M', 'J', 'P', 'G'), AVI_EXT, true},
            {VideoWriter::fourcc('m', 'p', '4', 'v'), AVI_EXT, true},
            {VideoWriter::fourcc('M', 'P', 'E', 'G'), AVI_EXT, true},
            //{VideoWriter::fourcc('W', 'M', 'V', '1'), AVI_EXT, true},
            //{VideoWriter::fourcc('W', 'M', 'V', '2'), AVI_EXT, true},
            {VideoWriter::fourcc('X', 'V', 'I', 'D'), AVI_EXT, true},
            //{VideoWriter::fourcc('Y', 'U', 'Y', '2'), AVI_EXT, true},
            {VideoWriter::fourcc('H', '2', '6', '4'), MP4_EXT, false}
        };

        const size_t n = sizeof(entries)/sizeof(entries[0]);

        for (size_t j = 0; j < n; ++j)
        {
            int tag = entries[j].tag;
            const char* ext = entries[j].ext;
            string s = cv::format("%08x%s", tag, ext);

            const string filename = tempfile(s.c_str());

            try
            {
                double fps = fps0;
                Size frame_s = Size(img_c, img_r);

                if( tag == VideoWriter::fourcc('H', '2', '6', '1') )
                    frame_s = Size(352, 288);
                else if( tag == VideoWriter::fourcc('H', '2', '6', '3') )
                    frame_s = Size(704, 576);
                else if( tag == VideoWriter::fourcc('H', '2', '6', '4') )
                    // OpenH264 1.5.0 has resolution limitations, so lets use DCI 4K resolution
                    frame_s = Size(4096, 2160);
                /*else if( tag == CV_FOURCC('M', 'J', 'P', 'G') ||
                         tag == CV_FOURCC('j', 'p', 'e', 'g') )
                    frame_s = Size(1920, 1080);*/

                if( tag == VideoWriter::fourcc('M', 'P', 'E', 'G') )
                {
                    frame_s = Size(720, 576);
                    fps = 25;
                }

                VideoWriter writer(filename, CAP_FFMPEG, tag, fps, frame_s);

                if (writer.isOpened() == false)
                {
                    fprintf(stderr, "\n\nFile name: %s\n", filename.c_str());
                    fprintf(stderr, "Codec id: %d   Codec tag: %c%c%c%c\n", (int)j,
                               tag & 255, (tag >> 8) & 255, (tag >> 16) & 255, (tag >> 24) & 255);
                    fprintf(stderr, "Error: cannot create video file.\n");
                    if (entries[j].required)
                        ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                }
                else
                {
                    Mat img(frame_s, CV_8UC3, Scalar::all(0));
                    const int coeff = cvRound(min(frame_s.width, frame_s.height)/(fps0 * time_sec));

                    for (int i = 0 ; i < static_cast<int>(fps * time_sec); i++ )
                    {
                        //circle(img, Point2i(img_c / 2, img_r / 2), min(img_r, img_c) / 2 * (i + 1), Scalar(255, 0, 0, 0), 2);
                        rectangle(img, Point2i(coeff * i, coeff * i), Point2i(coeff * (i + 1), coeff * (i + 1)),
                                  Scalar::all(255 * (1.0 - static_cast<double>(i) / (fps * time_sec * 2) )), -1);
                        writer << img;
                    }

                    writer.release();
                    long int sz = getFileSize(filename);
                    if (sz < 0)
                    {
                        fprintf(stderr, "ERROR: File name: %s was not created\n", filename.c_str());
                        if (entries[j].required)
                            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                    }
                    else
                    {
                        if (sz < 8192)
                        {
                            fprintf(stderr, "ERROR: File name: %s is very small (data write problems?)\n", filename.c_str());
                            if (entries[j].required)
                                ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                        }
                        remove(filename.c_str());
                    }
                }
            }
            catch(...)
            {
                ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
            }
            ts->set_failed_test_info(cvtest::TS::OK);
        }
    }
};

TEST(Videoio_Video, ffmpeg_writebig) { CV_FFmpegWriteBigVideoTest test; test.safe_run(); }

class CV_FFmpegReadImageTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            string filename = ts->get_data_path() + "readwrite/ordinary.bmp";
            VideoCapture cap(filename, CAP_FFMPEG);
            Mat img0 = imread(filename, 1);
            Mat img, img_next;
            cap >> img;
            cap >> img_next;

            CV_Assert( !img0.empty() && !img.empty() && img_next.empty() );

            double diff = cvtest::norm(img0, img, CV_C);
            CV_Assert( diff == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Videoio_Video, ffmpeg_image) { CV_FFmpegReadImageTest test; test.safe_run(); }

#endif

#if defined(HAVE_FFMPEG)

typedef tuple<VideoCaptureAPIs, string, string, string, string, string> videoio_container_params_t;
typedef testing::TestWithParam< videoio_container_params_t > videoio_container;

TEST_P(videoio_container, read)
{
    const VideoCaptureAPIs api = get<0>(GetParam());

    //if (!videoio_registry::hasBackend(api))
    //    throw SkipTestException("Backend was not found");

    const string path = get<1>(GetParam());
    const string ext = get<2>(GetParam());
    const string ext_raw = get<3>(GetParam());
    const string codec = get<4>(GetParam());
    const string pixelFormat = get<5>(GetParam());
    const string fileName = path + "." + ext;
    const string fileNameOut = tempfile(cv::format("test_container_stream.%s", ext_raw.c_str()).c_str());

    // Write encoded video read using VideoContainer to tmp file
    size_t totalBytes = 0;
    {
        VideoCapture container(findDataFile(fileName), api);
        if (!container.isOpened())
            throw SkipTestException("Video stream is not supported");
        if (!container.set(CAP_PROP_FORMAT, -1))  // turn off video decoder (extract stream)
            throw SkipTestException("Fetching of RAW video streams is not supported");
        ASSERT_EQ(-1.f, container.get(CAP_PROP_FORMAT));  // check
        EXPECT_EQ(codec, fourccToString((int)container.get(CAP_PROP_FOURCC)));
        EXPECT_EQ(pixelFormat, fourccToString((int)container.get(CAP_PROP_CODEC_PIXEL_FORMAT)));

        std::ofstream file(fileNameOut.c_str(), ios::out | ios::trunc | std::ios::binary);
        Mat raw_data;
        while (true)
        {
            container >> raw_data;
            size_t size = raw_data.total();
            if (raw_data.empty())
                break;
            ASSERT_EQ(CV_8UC1, raw_data.type());
            ASSERT_LE(raw_data.dims, 2);
            ASSERT_EQ(raw_data.rows, 1);
            ASSERT_EQ((size_t)raw_data.cols, raw_data.total());
            ASSERT_TRUE(raw_data.isContinuous());
            totalBytes += size;
            file.write(reinterpret_cast<char*>(raw_data.data), size);
            ASSERT_FALSE(file.fail());
        }
        ASSERT_GE(totalBytes, (size_t)65536) << "Encoded stream is too small";
    }

    std::cout << "Checking extracted video stream: " << fileNameOut << " (size: " << totalBytes << " bytes)" << std::endl;

    // Check decoded frames read from original media are equal to frames decoded from tmp file
    {
        VideoCapture capReference(findDataFile(fileName), api);
        ASSERT_TRUE(capReference.isOpened());
        VideoCapture capActual(fileNameOut.c_str(), api);
        ASSERT_TRUE(capActual.isOpened());
        Mat reference, actual;
        int nframes = 0, n_err = 0;
        while (capReference.read(reference) && n_err < 3)
        {
            nframes++;
            ASSERT_TRUE(capActual.read(actual)) << nframes;
            EXPECT_EQ(0, cvtest::norm(actual, reference, NORM_INF)) << "frame=" << nframes << " err=" << ++n_err;
        }
        ASSERT_GT(nframes, 0);
    }

    ASSERT_EQ(0, remove(fileNameOut.c_str()));
}

const videoio_container_params_t videoio_container_params[] =
{
    videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h264", "h264", "h264", "I420"),
    videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h265", "h265", "hevc", "I420"),
    videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "mjpg.avi", "mjpg", "MJPG", "I420"),
    //videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h264.mkv", "mkv.h264", "h264", "I420"),
    //videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h265.mkv", "mkv.h265", "hevc", "I420"),
    //videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h264.mp4", "mp4.avc1", "avc1", "I420"),
    //videoio_container_params_t(CAP_FFMPEG, "video/big_buck_bunny", "h265.mp4", "mp4.hev1", "hev1", "I420"),
};

INSTANTIATE_TEST_CASE_P(/**/, videoio_container, testing::ValuesIn(videoio_container_params));

typedef tuple<string, string, int> videoio_skip_params_t;
typedef testing::TestWithParam< videoio_skip_params_t > videoio_skip;

TEST_P(videoio_skip, DISABLED_read)  // optional test, may fail in some configurations
{
#if CV_VERSION_MAJOR >= 4
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("Backend was not found");
#endif

    const string path = get<0>(GetParam());
    const string env = get<1>(GetParam());
    const int expectedFrameNumber = get<2>(GetParam());

    #ifdef _WIN32
        _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", env.c_str());
    #else
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", env.c_str(), 1);
    #endif
    VideoCapture container(findDataFile(path), CAP_FFMPEG);
    #ifdef _WIN32
        _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "");
    #else
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "", 1);
    #endif

    ASSERT_TRUE(container.isOpened());

    Mat reference;
    int nframes = 0, n_err = 0;
    while (container.isOpened())
    {
        if (container.read(reference))
            nframes++;
        else if (++n_err > 3)
            break;
    }
    EXPECT_EQ(expectedFrameNumber, nframes);
}

const videoio_skip_params_t videoio_skip_params[] =
{
    videoio_skip_params_t("video/big_buck_bunny.mp4", "", 125),
    videoio_skip_params_t("video/big_buck_bunny.mp4", "avdiscard;nonkey", 11)
};

INSTANTIATE_TEST_CASE_P(/**/, videoio_skip, testing::ValuesIn(videoio_skip_params));

//==========================================================================

//////////////////////////////// Parallel VideoWriters and VideoCaptures ////////////////////////////////////

class CreateVideoWriterInvoker :
    public ParallelLoopBody
{
public:
    const static Size FrameSize;
    static std::string TmpDirectory;

    CreateVideoWriterInvoker(std::vector< cv::Ptr<VideoWriter> >& _writers, std::vector<std::string>& _files) :
        writers(_writers), files(_files)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        for (int i = range.start; i != range.end; ++i)
        {
            std::ostringstream stream;
            stream << i << ".avi";
            std::string fileName = tempfile(stream.str().c_str());

            files[i] = fileName;
            writers[i] = makePtr<VideoWriter>(fileName, CAP_FFMPEG, VideoWriter::fourcc('X','V','I','D'), 25.0f, FrameSize);

            CV_Assert(writers[i]->isOpened());
        }
    }

private:
    std::vector< cv::Ptr<VideoWriter> >& writers;
    std::vector<std::string>& files;
};

std::string CreateVideoWriterInvoker::TmpDirectory;
const Size CreateVideoWriterInvoker::FrameSize(1020, 900);

class WriteVideo_Invoker :
    public ParallelLoopBody
{
public:
    enum { FrameCount = 300 };

    static const Scalar ObjectColor;
    static const Point Center;

    WriteVideo_Invoker(const std::vector< cv::Ptr<VideoWriter> >& _writers) :
        ParallelLoopBody(), writers(&_writers)
    {
    }

    static void GenerateFrame(Mat& frame, unsigned int i)
    {
        frame = Scalar::all(i % 255);

        std::string text = to_string(i);
        putText(frame, text, Point(50, Center.y), FONT_HERSHEY_SIMPLEX, 5.0, ObjectColor, 5, CV_AA);
        circle(frame, Center, i + 2, ObjectColor, 2, CV_AA);
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        for (int j = range.start; j < range.end; ++j)
        {
            VideoWriter* writer = writers->operator[](j);
            CV_Assert(writer != NULL);
            CV_Assert(writer->isOpened());

            Mat frame(CreateVideoWriterInvoker::FrameSize, CV_8UC3);
            for (unsigned int i = 0; i < FrameCount; ++i)
            {
                GenerateFrame(frame, i);
                writer->operator<< (frame);
            }
        }
    }

protected:
    static std::string to_string(unsigned int i)
    {
        std::stringstream stream(std::ios::out);
        stream << "frame #" << i;
        return stream.str();
    }

private:
    const std::vector< cv::Ptr<VideoWriter> >* writers;
};

const Scalar WriteVideo_Invoker::ObjectColor(Scalar::all(0));
const Point WriteVideo_Invoker::Center(CreateVideoWriterInvoker::FrameSize.height / 2,
    CreateVideoWriterInvoker::FrameSize.width / 2);

class CreateVideoCaptureInvoker :
    public ParallelLoopBody
{
public:
    CreateVideoCaptureInvoker(std::vector< cv::Ptr<VideoCapture> >& _readers, const std::vector<std::string>& _files) :
        ParallelLoopBody(), readers(&_readers), files(&_files)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        for (int i = range.start; i != range.end; ++i)
        {
            readers->operator[](i) = makePtr<VideoCapture>(files->operator[](i), CAP_FFMPEG);
            CV_Assert(readers->operator[](i)->isOpened());
        }
    }
private:
    std::vector< cv::Ptr<VideoCapture> >* readers;
    const std::vector<std::string>* files;
};

class ReadImageAndTest :
    public ParallelLoopBody
{
public:
    ReadImageAndTest(const std::vector< cv::Ptr<VideoCapture> >& _readers, cvtest::TS* _ts) :
        ParallelLoopBody(), readers(&_readers), ts(_ts)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        for (int j = range.start; j < range.end; ++j)
        {
            VideoCapture* capture = readers->operator[](j).get();
            CV_Assert(capture != NULL);
            CV_Assert(capture->isOpened());

            const static double eps = 23.0;
            unsigned int frameCount = static_cast<unsigned int>(capture->get(CAP_PROP_FRAME_COUNT));
            CV_Assert(frameCount == WriteVideo_Invoker::FrameCount);
            Mat reference(CreateVideoWriterInvoker::FrameSize, CV_8UC3);

            for (unsigned int i = 0; i < frameCount && next; ++i)
            {
                SCOPED_TRACE(cv::format("frame=%d/%d", (int)i, (int)frameCount));

                Mat actual;
                (*capture) >> actual;

                WriteVideo_Invoker::GenerateFrame(reference, i);

                EXPECT_EQ(reference.cols, actual.cols);
                EXPECT_EQ(reference.rows, actual.rows);
                EXPECT_EQ(reference.depth(), actual.depth());
                EXPECT_EQ(reference.channels(), actual.channels());

                double psnr = cvtest::PSNR(actual, reference);
                if (psnr < eps)
                {
    #define SUM cvtest::TS::SUMMARY
                    ts->printf(SUM, "\nPSNR: %lf\n", psnr);
                    ts->printf(SUM, "Video #: %d\n", range.start);
                    ts->printf(SUM, "Frame #: %d\n", i);
    #undef SUM
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    ts->set_gtest_status();

                    Mat diff;
                    absdiff(actual, reference, diff);

                    EXPECT_EQ(countNonZero(diff.reshape(1) > 1), 0);

                    next = false;
                }
            }
        }
    }

    static bool next;

private:
    const std::vector< cv::Ptr<VideoCapture> >* readers;
    cvtest::TS* ts;
};

bool ReadImageAndTest::next;

TEST(Videoio_Video_parallel_writers_and_readers, accuracy)
{
    const unsigned int threadsCount = 4;
    cvtest::TS* ts = cvtest::TS::ptr();

    // creating VideoWriters
    std::vector< cv::Ptr<VideoWriter> > writers(threadsCount);
    Range range(0, threadsCount);
    std::vector<std::string> files(threadsCount);
    CreateVideoWriterInvoker invoker1(writers, files);
    parallel_for_(range, invoker1);

    // write a video
    parallel_for_(range, WriteVideo_Invoker(writers));

    // deleting the writers
    writers.clear();

    std::vector<cv::Ptr<VideoCapture> > readers(threadsCount);
    CreateVideoCaptureInvoker invoker2(readers, files);
    parallel_for_(range, invoker2);

    ReadImageAndTest::next = true;

    parallel_for_(range, ReadImageAndTest(readers, ts));

    // deleting tmp video files
    for (std::vector<std::string>::const_iterator i = files.begin(), end = files.end(); i != end; ++i)
    {
        int code = remove(i->c_str());
        if (code == 1)
            std::cerr << "Couldn't delete " << *i << std::endl;
    }

    // delete the readers
    readers.clear();
}

typedef std::pair<VideoCaptureProperties, double> cap_property_t;
typedef std::vector<cap_property_t> cap_properties_t;
typedef std::pair<std::string, cap_properties_t> ffmpeg_cap_properties_param_t;
typedef testing::TestWithParam<ffmpeg_cap_properties_param_t> ffmpeg_cap_properties;

#ifdef _WIN32
namespace {
::testing::AssertionResult IsOneOf(double value, double expected1, double expected2)
{
    // internal floating point class is used to perform accurate floating point types comparison
    typedef ::testing::internal::FloatingPoint<double> FloatingPoint;

    FloatingPoint val(value);
    if (val.AlmostEquals(FloatingPoint(expected1)) || val.AlmostEquals(FloatingPoint(expected2)))
    {
        return ::testing::AssertionSuccess();
    }
    else
    {
        return ::testing::AssertionFailure()
               << value << " is neither  equal to " << expected1 << " nor " << expected2;
    }
}
}
#endif

TEST_P(ffmpeg_cap_properties, can_read_property)
{
    ffmpeg_cap_properties_param_t parameters = GetParam();
    const std::string path = parameters.first;
    const cap_properties_t properties = parameters.second;

    VideoCapture cap(findDataFile(path), CAP_FFMPEG);
    ASSERT_TRUE(cap.isOpened()) << "Can not open " << findDataFile(path);

    for (std::size_t i = 0; i < properties.size(); ++i)
    {
        const cap_property_t& prop = properties[i];
        const double actualValue = cap.get(static_cast<int>(prop.first));
    #ifndef _WIN32
        EXPECT_DOUBLE_EQ(actualValue, prop.second)
            << "Property " << static_cast<int>(prop.first) << " has wrong value";
    #else
        EXPECT_TRUE(IsOneOf(actualValue, prop.second, 0.0))
            << "Property " << static_cast<int>(prop.first) << " has wrong value";
    #endif
    }
}

cap_properties_t loadBigBuckBunnyFFProbeResults() {
    cap_property_t properties[] = { cap_property_t(CAP_PROP_BITRATE, 5851.),
                                    cap_property_t(CAP_PROP_FPS, 24.),
                                    cap_property_t(CAP_PROP_FRAME_HEIGHT, 384.),
                                    cap_property_t(CAP_PROP_FRAME_WIDTH, 672.) };
    return cap_properties_t(properties, properties + sizeof(properties) / sizeof(cap_property_t));
}

const ffmpeg_cap_properties_param_t videoio_ffmpeg_properties[] = {
    ffmpeg_cap_properties_param_t("video/big_buck_bunny.avi", loadBigBuckBunnyFFProbeResults())
};

INSTANTIATE_TEST_CASE_P(videoio, ffmpeg_cap_properties, testing::ValuesIn(videoio_ffmpeg_properties));

// related issue: https://github.com/opencv/opencv/issues/15499
TEST(videoio, mp4_orientation_meta_auto)
{
    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/big_buck_bunny_rotated.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, CAP_FFMPEG));
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << CAP_FFMPEG << std::endl;

    cap.set(CAP_PROP_ORIENTATION_AUTO, true);
    if (cap.get(CAP_PROP_ORIENTATION_AUTO) == 0)
        throw SkipTestException("FFmpeg frame rotation metadata is not supported");

    Size actual;
    EXPECT_NO_THROW(actual = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                                    (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    EXPECT_EQ(384, actual.width);
    EXPECT_EQ(672, actual.height);

    Mat frame;

    cap >> frame;

    ASSERT_EQ(384, frame.cols);
    ASSERT_EQ(672, frame.rows);
}

// related issue: https://github.com/opencv/opencv/issues/15499
TEST(videoio, mp4_orientation_no_rotation)
{
    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/big_buck_bunny_rotated.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, CAP_FFMPEG));
    cap.set(CAP_PROP_ORIENTATION_AUTO, 0);
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << CAP_FFMPEG << std::endl;
    ASSERT_FALSE(cap.get(CAP_PROP_ORIENTATION_AUTO));

    Size actual;
    EXPECT_NO_THROW(actual = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                                    (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    EXPECT_EQ(672, actual.width);
    EXPECT_EQ(384, actual.height);

    Mat frame;

    cap >> frame;

    ASSERT_EQ(672, frame.cols);
    ASSERT_EQ(384, frame.rows);
}

#endif
}} // namespace
