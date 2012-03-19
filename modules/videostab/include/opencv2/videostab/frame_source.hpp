#ifndef __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__
#define __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__

#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace cv
{
namespace videostab
{

class IFrameSource
{
public:
    virtual ~IFrameSource() {}
    virtual void reset() = 0;
    virtual Mat nextFrame() = 0;
};

class NullFrameSource : public IFrameSource
{
public:
    virtual void reset() {}
    virtual Mat nextFrame() { return Mat(); }
};

class VideoFileSource : public IFrameSource
{
public:
    VideoFileSource(const std::string &path, bool volatileFrame = false);
    virtual void reset();
    virtual Mat nextFrame();

    int frameCount() { return reader_.get(CV_CAP_PROP_FRAME_COUNT); }
    double fps() { return reader_.get(CV_CAP_PROP_FPS); }

private:
    std::string path_;
    bool volatileFrame_;
    VideoCapture reader_;
};

} // namespace videostab
} // namespace cv

#endif
