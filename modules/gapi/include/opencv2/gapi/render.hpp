#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <string>
#include <vector>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/own/exports.hpp>
#include <opencv2/gapi/own/scalar.hpp>

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

struct Text
{
    std::string text;
    cv::Point   point;
    int         ff;
    double      fs;
    cv::Scalar  color;
    int         thick;
    int         lt;
    bool        bottom_left_origin;
};

struct Rect
{
    cv::Rect   rect;
    cv::Scalar color;
    int        thick;
    int        lt;
    int        shift;
};

using Prim  = util::variant<Text, Rect>;
using Prims = std::vector<Prim>;

GAPI_EXPORTS void render(cv::Mat& bgrx, const Prims& prims);

// FIXME Specify the signature for NV12 case
GAPI_EXPORTS void render(cv::Mat& y_plane, cv::Mat& uv_plane , const Prims& prims);


} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_RENDER_HPP
