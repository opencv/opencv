#ifndef __UTILITY_LIB_H__
#define __UTILITY_LIB_H__

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class FrameSource
{
public:
    virtual ~FrameSource() {}

    virtual void next(cv::Mat& frame) = 0;
};

class ImageSource : public FrameSource
{
public:
    explicit ImageSource(const std::string& path, int flags = 1);

    void next(cv::Mat& frame);

private:
    cv::Mat img_;
};

class VideoSource : public FrameSource
{
public:
    explicit VideoSource(const std::string& path);

    void next(cv::Mat& frame);

protected:
    cv::VideoCapture vc_;
    std::string path_;
};

class ImagesVideoSource : public VideoSource
{
public:
    explicit ImagesVideoSource(const std::string& path) : VideoSource(path), looped(false), prev(0.0){}

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

private:
    cv::VideoCapture vc_;
};

class PairFrameSource
{
public:
    virtual ~PairFrameSource() {}

    virtual void next(cv::Mat& frame0, cv::Mat& frame1) = 0;
    
    static cv::Ptr<PairFrameSource> get(const cv::Ptr<FrameSource>& source0, const cv::Ptr<FrameSource>& source1);

    static cv::Ptr<PairFrameSource> get(const cv::Ptr<FrameSource>& source, int offset);
};

void makeGray(const cv::Mat& src, cv::Mat& dst);

void printText(cv::Mat& img, const std::string& msg, int lineOffsY, cv::Scalar fontColor = CV_RGB(118, 185, 0), double fontScale = 0.8);

class BaseApp
{
public:
    BaseApp() : exited(false), frame_width(-1), frame_height(-1), device_(0) {}

    void run(int argc, const char* argv[]);

protected:
    virtual void process() = 0;
    virtual bool parseCmdArgs(int& i, int argc, const char* argv[]);
    virtual void printHelp();
    virtual bool processKey(int key);

    bool exited;
    std::vector< cv::Ptr<FrameSource> > sources;
    
    int frame_width;
    int frame_height;

    virtual bool parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[]);
private:
    bool parseHelpCmdArg(int& i, int argc, const char* argv[]);
    bool parseGpuDeviceCmdArgs(int& i, int argc, const char* argv[]);

    int device_;
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

#endif // __UTILITY_LIB_H__
