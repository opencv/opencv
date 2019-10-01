#include <vector>
#include "render_priv.hpp"

#ifndef OPENCV_RENDER_OCV_HPP
#define OPENCV_RENDER_OCV_HPP

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
void GAPI_EXPORTS drawPrimitivesOCVYUV(cv::Mat &yuv, const Prims &prims);
void GAPI_EXPORTS drawPrimitivesOCVBGR(cv::Mat &bgr, const Prims &prims);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_RENDER_OCV_HPP
