#include <iostream>
#include <deque>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

////////////////////////////////////
// ImageSource

namespace
{
    class ImageSource : public FrameSource
    {
    public:
        explicit ImageSource(const string& fileName, int flags = 1);

        void next(Mat& frame);

        void reset();

    private:
        Mat img_;
    };

    ImageSource::ImageSource(const string& fileName, int flags)
    {
        img_ = imread(fileName, flags);

        if (img_.empty())
            THROW_EXCEPTION("Can't open " << fileName << " image");
    }

    void ImageSource::next(Mat& frame)
    {
        frame = img_;
    }

    void ImageSource::reset()
    {
    }
}

Ptr<FrameSource> FrameSource::image(const string& fileName, int flags)
{
    return new ImageSource(fileName, flags);
}

////////////////////////////////////
// VideoSource

namespace
{
    class VideoSource : public FrameSource
    {
    public:
        explicit VideoSource(const string& fileName);

        void next(Mat& frame);

        void reset();

    protected:
        string fileName_;
        VideoCapture vc_;
    };

    VideoSource::VideoSource(const string& fileName) : fileName_(fileName)
    {
        if (!vc_.open(fileName))
            THROW_EXCEPTION("Can't open " << fileName << " video");
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
        vc_.open(fileName_);
    }
}

Ptr<FrameSource> FrameSource::video(const string& fileName)
{
    return new VideoSource(fileName);
}

////////////////////////////////////
// CameraSource

namespace
{
    class CameraSource : public FrameSource
    {
    public:
        explicit CameraSource(int device, int width = -1, int height = -1);

        void next(Mat& frame);

        void reset();

    private:
        VideoCapture vc_;
    };

    CameraSource::CameraSource(int device, int width, int height)
    {
        if (!vc_.open(device))
            THROW_EXCEPTION("Can't open camera with ID = " << device);

        if (width > 0)
            vc_.set(CV_CAP_PROP_FRAME_WIDTH, width);

        if (height > 0)
            vc_.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    }

    void CameraSource::next(Mat& frame)
    {
        vc_ >> frame;
    }

    void CameraSource::reset()
    {
    }
}

Ptr<FrameSource> FrameSource::camera(int device, int width, int height)
{
    return new CameraSource(device, width, height);
}

////////////////////////////////////
// ImagesVideoSource

namespace
{
    class ImagesPatternSource : public FrameSource
    {
    public:
        explicit ImagesPatternSource(const string& pattern);

        void next(Mat& frame);

        void reset();

    private:
        string pattern_;
        VideoCapture vc_;
        bool looped_;
        double prev_;
    };

    ImagesPatternSource::ImagesPatternSource(const string& pattern) : pattern_(pattern), looped_(false), prev_(0.0)
    {
        if (!vc_.open(pattern))
            THROW_EXCEPTION("Can't open " << pattern << " pattern");
    }

    void ImagesPatternSource::next(Mat& frame)
    {
        if (!looped_)
        {
            vc_ >> frame;

            if (frame.empty())
            {
                reset();
                vc_ >> frame;
            }
        }

        if (prev_ >= 1)
            looped_ = true;

        prev_ = vc_.get(CV_CAP_PROP_POS_AVI_RATIO);
    }

    void ImagesPatternSource::reset()
    {
        vc_.release();
        vc_.open(pattern_);
    }
}

Ptr<FrameSource> FrameSource::imagesPattern(const string& pattern)
{
    return new ImagesPatternSource(pattern);
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

Ptr<PairFrameSource> PairFrameSource::create(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1)
{
    return new PairFrameSource_2(source0, source1);
}

Ptr<PairFrameSource> PairFrameSource::create(const Ptr<FrameSource>& source, int offset)
{
    return new PairFrameSource_1(source, offset);
}

////////////////////////////////////
// Auxiliary functions

void makeGray(const InputArray src, OutputArray dst)
{
    const int scn = src.channels();

    CV_DbgAssert( scn == 1 || scn == 3 || scn == 4 );

    if (src.kind() == _InputArray::GPU_MAT)
    {
        if (scn == 1)
            dst.getGpuMatRef() = src.getGpuMat();
        else if (scn == 3)
            gpu::cvtColor(src.getGpuMat(), dst.getGpuMatRef(), COLOR_BGR2GRAY);
        else
            gpu::cvtColor(src.getGpuMat(), dst.getGpuMatRef(), COLOR_BGRA2GRAY);
    }
    else
    {
        if (scn == 1)
            dst.getMatRef() = src.getMat();
        else if (scn == 3)
            cvtColor(src, dst, COLOR_BGR2GRAY);
        else
            cvtColor(src, dst, COLOR_BGRA2GRAY);
    }
}

void printText(Mat& img, const string& msg, int lineOffsY, Scalar fontColor, double fontScale)
{
    const int fontFace = FONT_HERSHEY_DUPLEX;
    const int fontThickness = 2;

    const Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;

    putText(img, msg, org, fontFace, fontScale, Scalar(0,0,0,255), 5 * fontThickness / 2, 16);
    putText(img, msg, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

////////////////////////////////////
// BaseApp

BaseApp::BaseApp() : frame_width_(-1), frame_height_(-1), device_(0), active_(true)
{
}

void BaseApp::run(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (parseAppCmdArgs(i, argc, argv))
            continue;

        if (parseFrameSourcesCmdArgs(i, argc, argv))
            continue;

        if (parseGpuDeviceCmdArgs(i, argc, argv))
            continue;

        if (parseHelpCmdArg(i, argc, argv))
            return;

        THROW_EXCEPTION("Unknown command line argument: " << argv[i]);
    }

    const int num_devices = getCudaEnabledDeviceCount();
    if (num_devices <= 0)
        THROW_EXCEPTION("No GPU found or the OpenCV library was compiled without CUDA support");

    if (device_ < 0 || device_ >= num_devices)
        THROW_EXCEPTION("Incorrect device ID : " << device_);

    DeviceInfo dev_info(device_);
    if (!dev_info.isCompatible())
        THROW_EXCEPTION("GPU module wasn't built for GPU #" << device_ << " " << dev_info.name() << ", CC " << dev_info.majorVersion() << '.' << dev_info.minorVersion());

    cout << "Initializing device... \n" << endl;
    setDevice(device_);
    printShortCudaDeviceInfo(device_);

    cout << endl;

    runAppLogic();
}

void BaseApp::wait(int delay)
{
    const int key = waitKey(delay);

    if ((key & 0xff) == 27 /*escape*/)
    {
        active_ = false;
        return;
    }

    processAppKey(key);
}

void BaseApp::processAppKey(int)
{
}

void BaseApp::printAppHelp()
{
}

bool BaseApp::parseAppCmdArgs(int&, int, const char*[])
{
    return false;
}

bool BaseApp::parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "--image")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        sources_.push_back(FrameSource::image(argv[i]));
    }
    else if (arg == "--video")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        sources_.push_back(FrameSource::video(argv[i]));
    }
    else if (arg == "--frame-width")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        frame_width_ = atoi(argv[i]);
    }
    else if (arg == "--frame-height")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        frame_height_ = atoi(argv[i]);
    }
    else if (arg == "--camera")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        sources_.push_back(FrameSource::camera(atoi(argv[i]), frame_width_, frame_height_));
    }
    else
        return false;

    return true;
}

bool BaseApp::parseGpuDeviceCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "--device")
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
    printAppHelp();

    cout << "Source Options: \n"
         << "  --image <img_path> \n"
         << "       Image source path. \n"
         << "  --video <video_path> \n"
         << "       Video source path. \n"
         << "  --camera <device_ID> \n"
         << "       Camera device ID \n"
         << "  --frame-width <camera_frame_width> \n"
         << "       Camera frame width \n"
         << "  --frame-height <camera_frame_height> \n"
         << "       Camera frame height \n" << endl;

    cout << "Device Options: \n"
         << "  --device <device_id> \n"
         << "       Device ID" << endl;
}
