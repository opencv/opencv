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
#include "opencv2/highgui.hpp"

using namespace cv;

#ifdef HAVE_FFMPEG

using namespace std;

class CV_FFmpegWriteBigVideoTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        const int img_r = 4096;
        const int img_c = 4096;
        const double fps0 = 15;
        const double time_sec = 1;

        const int tags[] = {
            0,
            //VideoWriter::fourcc('D', 'I', 'V', '3'),
            //VideoWriter::fourcc('D', 'I', 'V', 'X'),
            VideoWriter::fourcc('D', 'X', '5', '0'),
            VideoWriter::fourcc('F', 'L', 'V', '1'),
            VideoWriter::fourcc('H', '2', '6', '1'),
            VideoWriter::fourcc('H', '2', '6', '3'),
            VideoWriter::fourcc('I', '4', '2', '0'),
            //VideoWriter::fourcc('j', 'p', 'e', 'g'),
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            VideoWriter::fourcc('m', 'p', '4', 'v'),
            VideoWriter::fourcc('M', 'P', 'E', 'G'),
            //VideoWriter::fourcc('W', 'M', 'V', '1'),
            //VideoWriter::fourcc('W', 'M', 'V', '2'),
            VideoWriter::fourcc('X', 'V', 'I', 'D'),
            //VideoWriter::fourcc('Y', 'U', 'Y', '2'),
        };

        const size_t n = sizeof(tags)/sizeof(tags[0]);

        bool created = false;

        for (size_t j = 0; j < n; ++j)
        {
            int tag = tags[j];
            stringstream s;
            s << tag;

            const string filename = tempfile((s.str()+".avi").c_str());

            try
            {
                double fps = fps0;
                Size frame_s = Size(img_c, img_r);

                if( tag == VideoWriter::fourcc('H', '2', '6', '1') )
                    frame_s = Size(352, 288);
                else if( tag == VideoWriter::fourcc('H', '2', '6', '3') )
                    frame_s = Size(704, 576);
                /*else if( tag == CV_FOURCC('M', 'J', 'P', 'G') ||
                         tag == CV_FOURCC('j', 'p', 'e', 'g') )
                    frame_s = Size(1920, 1080);*/

                if( tag == VideoWriter::fourcc('M', 'P', 'E', 'G') )
                {
                    frame_s = Size(720, 576);
                    fps = 25;
                }

                VideoWriter writer(filename, tag, fps, frame_s);

                if (writer.isOpened() == false)
                {
                    ts->printf(ts->LOG, "\n\nFile name: %s\n", filename.c_str());
                    ts->printf(ts->LOG, "Codec id: %d   Codec tag: %c%c%c%c\n", j,
                               tag & 255, (tag >> 8) & 255, (tag >> 16) & 255, (tag >> 24) & 255);
                    ts->printf(ts->LOG, "Error: cannot create video file.");
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

                    if (!created) created = true;
                    else remove(filename.c_str());
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

TEST(Highgui_Video, ffmpeg_writebig) { CV_FFmpegWriteBigVideoTest test; test.safe_run(); }

class CV_FFmpegReadImageTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            string filename = ts->get_data_path() + "readwrite/ordinary.bmp";
            VideoCapture cap(filename);
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

TEST(Highgui_Video, ffmpeg_image) { CV_FFmpegReadImageTest test; test.safe_run(); }

#endif

#if defined(HAVE_FFMPEG)

//////////////////////////////// Parallel VideoWriters and VideoCaptures ////////////////////////////////////

class CreateVideoWriterInvoker :
    public ParallelLoopBody
{
public:
    const static Size FrameSize;
    static std::string TmpDirectory;

    CreateVideoWriterInvoker(std::vector<VideoWriter*>& _writers, std::vector<std::string>& _files) :
        ParallelLoopBody(), writers(&_writers), files(&_files)
    {
    }

    virtual void operator() (const Range& range) const
    {
        for (int i = range.start; i != range.end; ++i)
        {
            std::ostringstream stream;
            stream << i << ".avi";
            std::string fileName = tempfile(stream.str().c_str());

            files->operator[](i) = fileName;
            writers->operator[](i) = new VideoWriter(fileName, VideoWriter::fourcc('X','V','I','D'), 25.0f, FrameSize);

            CV_Assert(writers->operator[](i)->isOpened());
        }
    }

private:
    std::vector<VideoWriter*>* writers;
    std::vector<std::string>* files;
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

    WriteVideo_Invoker(const std::vector<VideoWriter*>& _writers) :
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

    virtual void operator() (const Range& range) const
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
    const std::vector<VideoWriter*>* writers;
};

const Scalar WriteVideo_Invoker::ObjectColor(Scalar::all(0));
const Point WriteVideo_Invoker::Center(CreateVideoWriterInvoker::FrameSize.height / 2,
    CreateVideoWriterInvoker::FrameSize.width / 2);

class CreateVideoCaptureInvoker :
    public ParallelLoopBody
{
public:
    CreateVideoCaptureInvoker(std::vector<VideoCapture*>& _readers, const std::vector<std::string>& _files) :
        ParallelLoopBody(), readers(&_readers), files(&_files)
    {
    }

    virtual void operator() (const Range& range) const
    {
        for (int i = range.start; i != range.end; ++i)
        {
            readers->operator[](i) = new VideoCapture(files->operator[](i));
            CV_Assert(readers->operator[](i)->isOpened());
        }
    }
private:
    std::vector<VideoCapture*>* readers;
    const std::vector<std::string>* files;
};

class ReadImageAndTest :
    public ParallelLoopBody
{
public:
    ReadImageAndTest(const std::vector<VideoCapture*>& _readers, cvtest::TS* _ts) :
        ParallelLoopBody(), readers(&_readers), ts(_ts)
    {
    }

    virtual void operator() (const Range& range) const
    {
        for (int j = range.start; j < range.end; ++j)
        {
            VideoCapture* capture = readers->operator[](j);
            CV_Assert(capture != NULL);
            CV_Assert(capture->isOpened());

            const static double eps = 23.0;
            unsigned int frameCount = static_cast<unsigned int>(capture->get(CAP_PROP_FRAME_COUNT));
            CV_Assert(frameCount == WriteVideo_Invoker::FrameCount);
            Mat reference(CreateVideoWriterInvoker::FrameSize, CV_8UC3);

            for (unsigned int i = 0; i < frameCount && next; ++i)
            {
                Mat actual;
                (*capture) >> actual;

                WriteVideo_Invoker::GenerateFrame(reference, i);

                EXPECT_EQ(reference.cols, actual.cols);
                EXPECT_EQ(reference.rows, actual.rows);
                EXPECT_EQ(reference.depth(), actual.depth());
                EXPECT_EQ(reference.channels(), actual.channels());

                double psnr = PSNR(actual, reference);
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
    const std::vector<VideoCapture*>* readers;
    cvtest::TS* ts;
};

bool ReadImageAndTest::next;

TEST(Highgui_Video_parallel_writers_and_readers, accuracy)
{
    const unsigned int threadsCount = 4;
    cvtest::TS* ts = cvtest::TS::ptr();

    // creating VideoWriters
    std::vector<VideoWriter*> writers(threadsCount);
    Range range(0, threadsCount);
    std::vector<std::string> files(threadsCount);
    CreateVideoWriterInvoker invoker1(writers, files);
    parallel_for_(range, invoker1);

    // write a video
    parallel_for_(range, WriteVideo_Invoker(writers));

    // deleting the writers
    for (std::vector<VideoWriter*>::iterator i = writers.begin(), end = writers.end(); i != end; ++i)
        delete *i;
    writers.clear();

    std::vector<VideoCapture*> readers(threadsCount);
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
}

#endif
