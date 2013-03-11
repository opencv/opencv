#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

#include <string>
#include <sstream>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>

////////////////////////////////////
// FrameSource

class FrameSource
{
public:
    virtual ~FrameSource() {}

    virtual void next(cv::Mat& frame) = 0;

    virtual void reset() = 0;

    static cv::Ptr<FrameSource> image(const std::string& fileName, int flags = 1);
    static cv::Ptr<FrameSource> video(const std::string& fileName);
    static cv::Ptr<FrameSource> camera(int device, int width = -1, int height = -1);
    static cv::Ptr<FrameSource> imagesPattern(const std::string& pattern);
};

class PairFrameSource
{
public:
    virtual ~PairFrameSource() {}

    virtual void next(cv::Mat& frame0, cv::Mat& frame1) = 0;

    virtual void reset() = 0;

    static cv::Ptr<PairFrameSource> create(const cv::Ptr<FrameSource>& source0, const cv::Ptr<FrameSource>& source1);
    static cv::Ptr<PairFrameSource> create(const cv::Ptr<FrameSource>& source, int offset);
};

////////////////////////////////////
// Auxiliary functions

void makeGray(cv::InputArray src, cv::OutputArray dst);

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

    bool isActive() const { return active_; }
    void wait(int delay = 0);

protected:
    virtual void runAppLogic() = 0;
    virtual void processAppKey(int key);
    virtual void printAppHelp();
    virtual bool parseAppCmdArgs(int& i, int argc, const char* argv[]);

    std::vector< cv::Ptr<FrameSource> > sources_;

private:
    bool parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[]);
    bool parseGpuDeviceCmdArgs(int& i, int argc, const char* argv[]);
    bool parseHelpCmdArg(int& i, int argc, const char* argv[]);
    void printHelp();

    int device_;
    int frame_width_;
    int frame_height_;
    bool active_;
};

#define RUN_APP(App) \
    int main(int argc, const char* argv[]) \
    { \
        try \
        { \
            App app; \
            app.run(argc, argv); \
        } \
        catch (const std::exception& e) \
        { \
            std::cerr << "Error: " << e.what() << std::endl; \
            return -1; \
        } \
        return 0; \
    }

#endif // __UTILITY_HPP__
