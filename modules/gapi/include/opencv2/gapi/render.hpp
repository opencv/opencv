#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <string>
#include <vector>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/own/exports.hpp>

namespace cv
{
namespace gapi
{

struct TextEvent
{
    std::string text;
    float x;
    float y;
    int ff;
    double fs;
    cv::Scalar color;
    int thick;
    int lt;
    bool bottom_left_origin;
};

struct RectEvent
{
    float x;
    float y;
    int widht;
    int height;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
};

using DrawEvent  = util::variant<TextEvent, RectEvent>;
using DrawEvents = std::vector<DrawEvent>;

GAPI_EXPORTS void render(cv::Mat& bgrx, const std::vector<DrawEvent>& events);

// FIXME Specify the signature for NV12 case
GAPI_EXPORTS void render(cv::Mat& y_plane, cv::Mat& uv_plane , const std::vector<DrawEvent>& events);

} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_RENDER_HPP
