#include <iostream>
#include <deque>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

////////////////////////////////////
// ImageSource

ImageSource::ImageSource(const string& path, int flags)
{
    img_ = imread(path, flags);

    if (img_.empty())
        THROW_EXCEPTION("Can't open " << path << " image");
}

void ImageSource::next(Mat& frame)
{
    frame = img_;
}

void ImageSource::reset()
{
}

////////////////////////////////////
// VideoSource

VideoSource::VideoSource(const string& path) : vc_(path), path_(path)
{
    if (!vc_.isOpened())
        THROW_EXCEPTION("Can't open " << path << " video");
}

void VideoSource::next(Mat& frame)
{
    vc_ >> frame;

    if (frame.empty())
    {
        reset();
        vc_ >> frame;
    }
}

void VideoSource::reset()
{
    vc_.release();
    vc_.open(path_);
}

////////////////////////////////////
// ImagesVideoSource

ImagesVideoSource::ImagesVideoSource(const string& path) : VideoSource(path), looped(false), prev(0.0)
{
}

void ImagesVideoSource::next(Mat& frame)
{
    if (!looped)
        VideoSource::next(frame);

    if (prev >= 1)
        looped = true;

    prev = vc_.get(CV_CAP_PROP_POS_AVI_RATIO);
}

////////////////////////////////////
// CameraSource

CameraSource::CameraSource(int device, int width, int height) : vc_(device)
{
    if (!vc_.isOpened())
        THROW_EXCEPTION("Can't open camera with ID = " << device);

    if (width != -1)
        vc_.set(CV_CAP_PROP_FRAME_WIDTH, width);

    if (height != -1)
        vc_.set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

void CameraSource::next(Mat& frame)
{
    vc_ >> frame;
}

void CameraSource::reset()
{
}

////////////////////////////////////
// PairFrameSource

namespace
{
    class PairFrameSource_2 : public PairFrameSource
    {
    public:
        PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1);

        void next(Mat& frame0, Mat& frame1);
        void reset();

    private:
        Ptr<FrameSource> source0_;
        Ptr<FrameSource> source1_;
    };

    PairFrameSource_2::PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1) :
        source0_(source0), source1_(source1)
    {
        CV_Assert( !source0_.empty() );
        CV_Assert( !source1_.empty() );
    }

    void PairFrameSource_2::next(Mat& frame0, Mat& frame1)
    {
        source0_->next(frame0);
        source1_->next(frame1);
    }

    void PairFrameSource_2::reset()
    {
        source0_->reset();
        source1_->reset();
    }

    class PairFrameSource_1 : public PairFrameSource
    {
    public:
        PairFrameSource_1(const Ptr<FrameSource>& source, int offset);

        void next(Mat& frame0, Mat& frame1);
        void reset();

    private:
        Ptr<FrameSource> source_;
        int offset_;
        deque<Mat> frames_;
    };

    PairFrameSource_1::PairFrameSource_1(const Ptr<FrameSource>& source, int offset) : source_(source), offset_(offset)
    {
        CV_Assert( !source_.empty() );

        reset();
    }

    void PairFrameSource_1::next(Mat& frame0, Mat& frame1)
    {
        source_->next(frame1);
        frames_.push_back(frame1.clone());
        frame0 = frames_.front();
        frames_.pop_front();
    }

    void PairFrameSource_1::reset()
    {
        source_->reset();

        frames_.clear();
        Mat temp;
        for (int i = 0; i < offset_; ++i)
        {
            source_->next(temp);
            frames_.push_back(temp.clone());
        }
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

////////////////////////////////////
// Auxiliary functions

void makeGray(const Mat& src, Mat& dst)
{
    CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

    if (src.channels() == 1)
        dst = src;
    else if (src.channels() == 3)
        cvtColor(src, dst, COLOR_BGR2GRAY);
    else
        cvtColor(src, dst, COLOR_BGRA2GRAY);
}

void printText(Mat& img, const string& msg, int lineOffsY, Scalar fontColor, double fontScale)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    int fontThickness = 2;

    Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;

    putText(img, msg, org, fontFace, fontScale, Scalar(0,0,0,255), 5 * fontThickness / 2, 16);
    putText(img, msg, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

////////////////////////////////////
// BaseApp

BaseApp::BaseApp() : exited(false), frame_width_(-1), frame_height_(-1), device_(0)
{
}

void BaseApp::run(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (parseCmdArgs(i, argc, argv))
            continue;

        if (parseFrameSourcesCmdArgs(i, argc, argv))
            continue;

        if (parseGpuDeviceCmdArgs(i, argc, argv))
            continue;

        if (parseHelpCmdArg(i, argc, argv))
            return;

        THROW_EXCEPTION("Unknown command line argument: " << argv[i]);
    }

    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices == 0)
        THROW_EXCEPTION("No GPU found or the OpenCV library was compiled without CUDA support");

    if (device_ < 0 || device_ >= num_devices)
        THROW_EXCEPTION("Incorrect device ID : " << device_);

    DeviceInfo dev_info(device_);
    if (!dev_info.isCompatible())
        THROW_EXCEPTION("GPU module wasn't built for GPU #" << device_ << " " << dev_info.name() << ", CC " << dev_info.majorVersion() << '.' << dev_info.minorVersion());

    cout << "Initializing device...\n" << endl;
    setDevice(device_);
    GpuMat m(10, 10, CV_8U);
    m.release();

    printShortCudaDeviceInfo(device_);

    cout << endl;

    process();
}

bool BaseApp::parseCmdArgs(int& i, int argc, const char* argv[])
{
    return false;
}

bool BaseApp::parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "-i" || arg == "--image")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        sources.push_back(new ImageSource(argv[i]));
    }
    else if (arg == "-v" || arg == "--video")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        sources.push_back(new VideoSource(argv[i]));
    }
    else if (arg == "-fw" || arg == "--frame-width")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        frame_width_ = atoi(argv[i]);
    }
    else if (arg == "-fh" || arg == "--frame-height")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        frame_height_ = atoi(argv[i]);
    }
    else if (arg == "-c" || arg == "--camera")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        sources.push_back(new CameraSource(atoi(argv[i]), frame_width_, frame_height_));
    }
    else
        return false;

    return true;
}

bool BaseApp::parseGpuDeviceCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "-d" || arg == "--device")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        device_ = atoi(argv[i]);
    }
    else
        return false;

    return true;
}

bool BaseApp::parseHelpCmdArg(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "-h" || arg == "--help")
    {
        printHelp();
        return true;
    }

    return false;
}

void BaseApp::printHelp()
{
    cout << endl
         << "Frame Source Flags:" << endl
         << "  -i|--image <img_path>" << endl
         << "       Image source path." << endl
         << "  -v|--video <video_path>" << endl
         << "       Video source path." << endl
         << "  -c|--camera <device_ID>" << endl
         << "       Camera device ID" << endl
         << "  -fw|--frame-width <camera_frame_width>" << endl
         << "       Camera frame width" << endl
         << "  -fh|--frame-height <camera_frame_height>" << endl
         << "       Camera frame height" << endl
         << endl
         << "Device Flags:" << endl
         << "  -d|--device <device_id>" << endl
         << "       Device ID" << endl;
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
