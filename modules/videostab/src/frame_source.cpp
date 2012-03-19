#include "precomp.hpp"
#include "opencv2/videostab/frame_source.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

VideoFileSource::VideoFileSource(const string &path, bool volatileFrame)
    : path_(path), volatileFrame_(volatileFrame) { reset(); }


void VideoFileSource::reset()
{
    reader_.release();
    reader_.open(path_);
    if (!reader_.isOpened())
        throw runtime_error("can't open file: " + path_);
}


Mat VideoFileSource::nextFrame()
{
    Mat frame;
    reader_ >> frame;
    return volatileFrame_ ? frame : frame.clone();
}

} // namespace videostab
} // namespace cv
