// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_FREETYPE_MASK_CREATOR_HPP
#define OPENCV_FREETYPE_MASK_CREATOR_HPP

#ifdef HAVE_FREETYPE

#include "api/render_priv.hpp"

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

class FreeTypeBitmaskCreator : public IBitmaskCreator
{
public:
    FreeTypeBitmaskCreator(const std::string& path)
    {
        if (FT_Init_FreeType(&m_library) < 0)
        {
            util::throw_error(std::runtime_error("Failed to init FreeType library"));
        }

        if (FT_New_Face(m_library, path.c_str(), 0, &m_face))
        {
            util::throw_error(std::runtime_error("Failed to set font"));
        }
    }

    virtual cv::Size computeMaskSize() override
    {
        FT_GlyphSlot slot = m_face->glyph;
        int bmp_w        = 0;
        int max_baseline = 0;

        m_glyphs.resize(m_text.text.size());
        m_pos.reserve(m_text.text.size());

        for (size_t i = 0; i < m_text.text.size(); ++i)
        {
            FT_Load_Char(m_face, m_text.text[i], FT_LOAD_RENDER);
            FT_Bitmap *bitmap = &(slot->bitmap);

            cv::Mat(bitmap->rows, bitmap->width, CV_8UC1, bitmap->buffer, bitmap->pitch).copyTo(m_glyphs[i]);

            int gl_bottom_pad = (slot->metrics.height - slot->metrics.horiBearingY) >> 6;
            max_baseline = std::max(max_baseline, gl_bottom_pad);
            m_max_glyph_top = std::max(m_max_glyph_top, slot->bitmap_top);

            // FIXME why bitmap->left is negative ?
            int gl_x_pad = slot->bitmap_left > 0 ? slot->bitmap_left + bmp_w - 1 : bmp_w;
            auto shift = (slot->advance.x >> 6);


            // FIXME why bitmap->width > shift slot->advance.x ?
            if (shift < bitmap->width)
            {
                gl_x_pad = bmp_w;
                bmp_w += bitmap->width;
            }
            else
            {
                bmp_w += shift;
            }

            m_pos.emplace_back(gl_x_pad, slot->bitmap_top);
        }

        int bmp_h = max_baseline + m_max_glyph_top;
        m_mask_size = cv::Size(bmp_w, bmp_h);
        m_baseline = max_baseline;

        return m_mask_size;
    }

    int virtual createMask(cv::Mat& mask) override
    {
        mask = cv::Scalar(0);
        for (size_t i = 0; i < m_text.text.size(); ++i)
        {
            cv::Rect glyph_roi(m_pos[i].x, m_max_glyph_top - m_pos[i].y, m_glyphs[i].cols, m_glyphs[i].rows);
            cv::Mat roi = mask(glyph_roi);
            m_glyphs[i].copyTo(roi);
        }
        return m_baseline;
    }

    void virtual setMaskParams(const cv::gapi::wip::draw::Text& text) override
    {
        m_text = text;

        // Convert OpenCV scale to Freetype text height
        auto sz = cv::getTextSize(text.text, FONT_HERSHEY_SIMPLEX, text.fs, 1, nullptr);
        int font_height = static_cast<int>(sz.height * 3.0 / 2); // 3.0 / 2 for better aligning with OpenCV

        FT_Set_Pixel_Sizes(m_face, font_height, font_height);
    }

    virtual ~FreeTypeBitmaskCreator() override
    {
        FT_Done_Face(m_face);
        FT_Done_FreeType(m_library);
    }

private:
    FT_Library    m_library;
    FT_Face       m_face;

    std::vector<cv::Mat> m_glyphs;
    std::vector<cv::Point> m_pos;

    cv::gapi::wip::draw::Text m_text;
    int m_baseline = 0;
    int m_max_glyph_top = 0;
    cv::Size m_mask_size;
};

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_FREETYPE
#endif // OPENCV_FREETYPE_MASK_CREATOR_HPP
