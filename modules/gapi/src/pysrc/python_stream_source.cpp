#include <opencv2/gapi/pysrc/python_stream_source.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core.hpp>

namespace cv {
namespace gapi {
namespace wip {

cv::Ptr<cv::gapi::wip::IStreamSource> make_py_src(const cv::Ptr<cv::gapi::wip::IStreamSource>& src)
{
    return src;
}

} // namespace wip
} // namespace gapi
} // namespace cv
