#include <iostream>
#include <sstream>
#include <stdexcept>
#include <deque>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "utility_lib.h"

using namespace std;
using namespace cv;

ImageSource::ImageSource(const string& path, int flags) : img_(imread(path, flags))
{ 
    CV_Assert(!img_.empty()); 
}

void ImageSource::next(Mat& frame) 
{ 
    frame = img_; 
}

VideoSource::VideoSource(const string& path) : vc_(path), path_(path)
{ 
    CV_Assert(vc_.isOpened()); 
}

void VideoSource::next(Mat& frame)
{
    vc_ >> frame;

    if (frame.empty())
    {
        vc_.open(path_);
        vc_ >> frame;
    }
}    

CameraSource::CameraSource(int device, int width, int height) : vc_(device)
{
    if (width != -1) 
        vc_.set(CV_CAP_PROP_FRAME_WIDTH, width);

    if (height != -1) 
        vc_.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if (!vc_.isOpened())
    {
        ostringstream msg;
        msg << "Can't open camera with dev. ID = " << device;
        throw runtime_error(msg.str());
    }
}

void CameraSource::next(Mat& frame)
{
    vc_ >> frame;
}

namespace
{
    class PairFrameSource_2 : public PairFrameSource
    {
    public:
        PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1);

        void next(Mat& frame0, Mat& frame1);

    private:
        Ptr<FrameSource> source0_;
        Ptr<FrameSource> source1_;
    };

    PairFrameSource_2::PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1) :
        source0_(source0), source1_(source1)
    {
        CV_Assert(!source0_.empty());
        CV_Assert(!source1_.empty());
    }

    void PairFrameSource_2::next(Mat& frame0, Mat& frame1)
    {
        source0_->next(frame0);
        source1_->next(frame1);
    }

    class PairFrameSource_1 : public PairFrameSource
    {
    public:
        PairFrameSource_1(const Ptr<FrameSource>& source, int offset);

        void next(Mat& frame0, Mat& frame1);

    private:
        Ptr<FrameSource> source_;
        deque<Mat> frames_;
    };

    PairFrameSource_1::PairFrameSource_1(const Ptr<FrameSource>& source, int offset) :
        source_(source)
    {
        CV_Assert(!source_.empty());

        Mat temp;
        for (int i = 0; i < offset; ++i)
        {
            source_->next(temp);
            frames_.push_back(temp.clone());
        }
    }

    void PairFrameSource_1::next(Mat& frame0, Mat& frame1)
    {
        source_->next(frame1);
        frames_.push_back(frame1.clone());
        frame0 = frames_.front();
        frames_.pop_front();
    }
}

Ptr<PairFrameSource> PairFrameSource::get(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1)
{
    return new PairFrameSource_2(source0, source1);
}

Ptr<PairFrameSource> PairFrameSource::get(const Ptr<FrameSource>& source, int offset)
{
    return new PairFrameSource_1(source, offset);
}

void makeGray(const Mat& src, Mat& dst)
{
    if (src.channels() == 1)
        dst = src;
    else if (src.channels() == 3)
        cvtColor(src, dst, COLOR_BGR2GRAY);
    else if (src.channels() == 4)
        cvtColor(src, dst, COLOR_BGRA2GRAY);
    else
    {
        ostringstream msg;
        msg << "Can't convert image to gray: unsupported channels = " << src.channels();
        throw runtime_error(msg.str());
    }
}

void printText(Mat& img, const string& msg, int lineOffsY, Scalar fontColor)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;

    Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;

    putText(img, msg, org, fontFace, fontScale, Scalar::all(0), 5 * fontThickness / 2, 16);
    putText(img, msg, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

void BaseApp::run(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (parseCmdArgs(i, argc, argv))
            continue;

        if (parseFrameSourcesCmdArgs(i, argc, argv))
            continue;

        if (parseHelpCmdArg(i, argc, argv))
            return;

        ostringstream msg;
        msg << "Unknown command line argument: " << argv[i];
        throw runtime_error(msg.str());
    }

    if (gpu::getCudaEnabledDeviceCount() == 0)
        throw runtime_error("No GPU found or the library is compiled without GPU support"); 

    process();
}

bool BaseApp::parseCmdArgs(int& i, int argc, const char* argv[])
{
    return false;
}

void BaseApp::printHelp()
{
    cout << "\nFrame Source Flags:\n"
         << "  -i <img_path>\n"
         << "       Image source path.\n"
         << "  -v <video_path>\n"
         << "       Video source path.\n"
         << "  -c <device_ID>\n"
         << "       Camera device ID\n"
         << "  -w <camera_frame_width>\n"
         << "  -h <camera_frame_height>\n";
}

bool BaseApp::processKey(int key)
{
    if ((key & 0xff) == 27 /*escape*/)
    {
        exited = true;
        return true;
    }

    return false;
}

bool BaseApp::parseHelpCmdArg(int& i, int argc, const char* argv[])
{
    CV_Assert(i < argc);

    string arg(argv[i]);

    if (arg == "--help" || arg == "-h")
    {
        printHelp();
        return true;
    }

    return false;
}

bool BaseApp::parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "-i") 
    {
        ++i;

        if (i >= argc)
            throw runtime_error("Missing file name after -i");

        sources.push_back(new ImageSource(argv[i]));
    }
    else if (arg == "-v")
    {
        ++i;

        if (i >= argc)
            throw runtime_error("Missing file name after -v");

        sources.push_back(new VideoSource(argv[i]));
    }
    else if (arg == "-w") 
    {
        ++i;

        if (i >= argc)
            throw runtime_error("Missing value after -w");

        frame_width = atoi(argv[i]);
    }
    else if (arg == "-h")
    {
        ++i;

        if (i >= argc)
            throw runtime_error("Missing value after -h");

        frame_height = atoi(argv[i]);
    }
    else if (arg == "-c") 
    {
        ++i;

        if (i >= argc)
            throw runtime_error("Missing value after -c");

        sources.push_back(new CameraSource(atoi(argv[i]), frame_width, frame_height));
    }
    else 
        return false;

    return true;
}
