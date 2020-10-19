// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifdef HAVE_FREETYPE

#ifndef OPENCV_FT_RENDER_PRIV_HPP
#define OPENCV_FT_RENDER_PRIV_HPP

#include "api/ft_render.hpp"

#include <ft2build.h>
#include FT_FREETYPE_H

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

class FTTextRender::Priv
{
public:
    explicit Priv(const std::string& path);

    cv::Size getTextSize(const std::wstring& text, int fh, int* baseline);
    void putText(cv::Mat& mat, const std::wstring& text, const cv::Point& org, int fh);

    ~Priv();

private:
    FT_Library    m_library;
    FT_Face       m_face;
};

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_FT_RENDER_PRIV_HPP
#endif // HAVE_FREETYPE
