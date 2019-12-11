// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

static inline long long getFileSize(const string &filename)
{
    ifstream f(filename, ios_base::in | ios_base::binary);
    f.seekg(0, ios_base::end);
    return f.tellg();
}

typedef tuple<string, string, Size> FourCC_Ext_Size;
typedef testing::TestWithParam< FourCC_Ext_Size > videoio_ffmpeg;

TEST_P(videoio_ffmpeg, write_big)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string fourcc = get<0>(GetParam());
    const string ext = get<1>(GetParam());
    const Size sz = get<2>(GetParam());
    const double time_sec = 1;
    const double fps = 25;

    ostringstream buf;
    buf << "write_big_" << fourcc << "." << ext;
    const string filename = tempfile(buf.str().c_str());

    VideoWriter writer(filename, CAP_FFMPEG, fourccFromString(fourcc), fps, sz);
    if (ext == "mp4" && fourcc == "H264" && !writer.isOpened())
    {
        throw cvtest::SkipTestException("H264/mp4 codec is not supported - SKIP");
    }
    ASSERT_TRUE(writer.isOpened());
    Mat img(sz, CV_8UC3, Scalar::all(0));
    const int coeff = cvRound(min(sz.width, sz.height)/(fps * time_sec));
    for (int i = 0 ; i < static_cast<int>(fps * time_sec); i++ )
    {
        rectangle(img,
                  Point2i(coeff * i, coeff * i),
                  Point2i(coeff * (i + 1), coeff * (i + 1)),
                  Scalar::all(255 * (1.0 - static_cast<double>(i) / (fps * time_sec * 2))),
                  -1);
        writer << img;
    }
    writer.release();
    EXPECT_GT(getFileSize(filename), 8192);
    remove(filename.c_str());
}

static const Size bigSize(4096, 4096);

const FourCC_Ext_Size entries[] =
{
    make_tuple("", "avi", bigSize),
    make_tuple("DX50", "avi", bigSize),
    make_tuple("FLV1", "avi", bigSize),
    make_tuple("H261", "avi", Size(352, 288)),
    make_tuple("H263", "avi", Size(704, 576)),
    make_tuple("I420", "avi", bigSize),
    make_tuple("MJPG", "avi", bigSize),
    make_tuple("mp4v", "avi", bigSize),
    make_tuple("MPEG", "avi", Size(720, 576)),
    make_tuple("XVID", "avi", bigSize),
    make_tuple("H264", "mp4", Size(4096, 2160))
};

INSTANTIATE_TEST_CASE_P(videoio, videoio_ffmpeg, testing::ValuesIn(entries));

//==========================================================================

TEST(videoio_ffmpeg, image)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = findDataFile("readwrite/ordinary.bmp");
    Mat image = imread(filename, IMREAD_COLOR);
    ASSERT_FALSE(image.empty());
    VideoCapture cap(filename, CAP_FFMPEG);
    ASSERT_TRUE(cap.isOpened());
    Mat frame1, frame2;
    cap >> frame1 >> frame2;
    ASSERT_FALSE(frame1.empty());
    ASSERT_TRUE(frame2.empty());
    ASSERT_EQ(0, cvtest::norm(image, frame1, NORM_INF));
}

//==========================================================================

typedef tuple<VideoCaptureAPIs, string, string, string, string, string> videoio_container_params_t;
typedef testing::TestWithParam< videoio_container_params_t > videoio_container;

TEST_P(videoio_container, read)
{
    const VideoCaptureAPIs api = get<0>(GetParam());

    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("Backend was not found");

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

//==========================================================================

static void generateFrame(Mat &frame, unsigned int i, const Point &center, const Scalar &color)
{
    frame = Scalar::all(i % 255);
    stringstream buf(ios::out);
    buf << "frame #" << i;
    putText(frame, buf.str(), Point(50, center.y), FONT_HERSHEY_SIMPLEX, 5.0, color, 5, CV_AA);
    circle(frame, center, i + 2, color, 2, CV_AA);
}

TEST(videoio_ffmpeg, parallel)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const int NUM = 4;
    const int GRAN = 4;
    const Range R(0, NUM);
    const Size sz(1020, 900);
    const int frameNum = 300;
    const Scalar color(Scalar::all(0));
    const Point center(sz.height / 2, sz.width / 2);

    // Generate filenames
    vector<string> files;
    for (int i = 0; i < NUM; ++i)
    {
        ostringstream stream;
        stream << i << ".avi";
        files.push_back(tempfile(stream.str().c_str()));
    }
    // Write videos
    {
        vector< Ptr<VideoWriter> > writers(NUM);
        auto makeWriters = [&](const Range &r)
        {
            for (int i = r.start; i != r.end; ++i)
                writers[i] = makePtr<VideoWriter>(files[i],
                                                  CAP_FFMPEG,
                                                  VideoWriter::fourcc('X','V','I','D'),
                                                  25.0f,
                                                  sz);
        };
        parallel_for_(R, makeWriters, GRAN);
        for(int i = 0; i < NUM; ++i)
        {
            ASSERT_TRUE(writers[i]);
            ASSERT_TRUE(writers[i]->isOpened());
        }
        auto writeFrames = [&](const Range &r)
        {
            for (int j = r.start; j < r.end; ++j)
            {
                Mat frame(sz, CV_8UC3);
                for (int i = 0; i < frameNum; ++i)
                {
                    generateFrame(frame, i, center, color);
                    writers[j]->write(frame);
                }
            }
        };
        parallel_for_(R, writeFrames, GRAN);
    }
    // Read videos
    {
        vector< Ptr<VideoCapture> > readers(NUM);
        auto makeCaptures = [&](const Range &r)
        {
            for (int i = r.start; i != r.end; ++i)
                readers[i] = makePtr<VideoCapture>(files[i], CAP_FFMPEG);
        };
        parallel_for_(R, makeCaptures, GRAN);
        for(int i = 0; i < NUM; ++i)
        {
            ASSERT_TRUE(readers[i]);
            ASSERT_TRUE(readers[i]->isOpened());
        }
        auto readFrames = [&](const Range &r)
        {
            for (int j = r.start; j < r.end; ++j)
            {
                Mat reference(sz, CV_8UC3);
                for (int i = 0; i < frameNum; ++i)
                {
                    Mat actual;
                    EXPECT_TRUE(readers[j]->read(actual));
                    EXPECT_FALSE(actual.empty());
                    generateFrame(reference, i, center, color);
                    EXPECT_EQ(reference.size(), actual.size());
                    EXPECT_EQ(reference.depth(), actual.depth());
                    EXPECT_EQ(reference.channels(), actual.channels());
                    EXPECT_GE(cvtest::PSNR(actual, reference), 35.0) << "cap" << j << ", frame " << i;
                }
            }
        };
        parallel_for_(R, readFrames, GRAN);
    }
    // Remove files
    for(int i = 0; i < NUM; ++i)
    {
        remove(files[i].c_str());
    }
}

}} // namespace
