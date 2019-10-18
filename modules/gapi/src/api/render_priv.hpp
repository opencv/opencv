// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_RENDER_PRIV_HPP
#define OPENCV_RENDER_PRIV_HPP

#include <opencv2/gapi/render/render.hpp>

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
GAPI_EXPORTS void BGR2NV12(const cv::Mat& bgr, cv::Mat& y_plane, cv::Mat& uv_plane);

void blendImage(const cv::Mat& img,
                const cv::Mat& alpha,
                const cv::Point& org,
                cv::Mat background);

class IBitmaskCreator
{
public:
    virtual int createMask(cv::Mat&) = 0;
    virtual const cv::Size& computeMaskSize() = 0;
    virtual void setMaskParams(const cv::gapi::wip::draw::Text& text) = 0;
    virtual ~IBitmaskCreator() = default;
};

template<typename T, typename... Args>
std::unique_ptr<IBitmaskCreator> make_mask_creator(Args&&... args)
{
    return std::unique_ptr<IBitmaskCreator>(new T(std::forward<Args>(args)...));
}

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_RENDER_PRIV_HPP
