#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <string>
#include <sstream>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

////////////////////////////////////
// FrameSource

class FrameSource
{
public:
    virtual ~FrameSource() {}

    virtual void next(cv::Mat& frame) = 0;
    virtual void reset() = 0;
};

class ImageSource : public FrameSource
{
public:
    explicit ImageSource(const std::string& path, int flags = 1);

    void next(cv::Mat& frame);
    void reset();

private:
    cv::Mat img_;
};

class VideoSource : public FrameSource
{
public:
    explicit VideoSource(const std::string& path);

    void next(cv::Mat& frame);
    void reset();

protected:
    cv::VideoCapture vc_;
    std::string path_;
};

class ImagesVideoSource : public VideoSource
{
public:
    explicit ImagesVideoSource(const std::string& path);

    void next(cv::Mat& frame);

private:
    bool looped;
    double prev;
};

class CameraSource : public FrameSource
{
public:
    explicit CameraSource(int device, int width = -1, int height = -1);

    void next(cv::Mat& frame);
    void reset();

private:
    cv::VideoCapture vc_;
};

class PairFrameSource
{
public:
    virtual ~PairFrameSource() {}

    virtual void next(cv::Mat& frame0, cv::Mat& frame1) = 0;
    virtual void reset() = 0;

    static cv::Ptr<PairFrameSource> get(const cv::Ptr<FrameSource>& source0, const cv::Ptr<FrameSource>& source1);

    static cv::Ptr<PairFrameSource> get(const cv::Ptr<FrameSource>& source, int offset);
};

////////////////////////////////////
// Auxiliary functions

void makeGray(const cv::Mat& src, cv::Mat& dst);

void printText(cv::Mat& img, const std::string& msg, int lineOffsY, cv::Scalar fontColor = CV_RGB(118, 185, 0), double fontScale = 0.8);

#define THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg ; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

////////////////////////////////////
// BaseApp

class BaseApp
{
public:
    BaseApp();

    void run(int argc, const char* argv[]);

protected:
    virtual void process() = 0;
    virtual bool parseCmdArgs(int& i, int argc, const char* argv[]);
    virtual void printHelp();
    virtual bool processKey(int key);

    bool exited;
    std::vector< cv::Ptr<FrameSource> > sources;

private:
    bool parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[]);
    bool parseGpuDeviceCmdArgs(int& i, int argc, const char* argv[]);
    bool parseHelpCmdArg(int& i, int argc, const char* argv[]);

    int device_;
    int frame_width_;
    int frame_height_;
};

#define RUN_APP(App) \
    int main(int argc, const char* argv[]) \
    { \
        try \
        { \
            App app; \
            app.run(argc, argv); \
        } \
        catch (const std::exception &e) \
        { \
            std::cout << "Error: " << e.what() << std::endl; \
            return -1; \
        } \
        return 0; \
    }

#endif // __UTILITY_H__
