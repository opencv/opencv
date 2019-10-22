#ifdef HAVE_FREETYPE

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_FREETYPE_MASK_CREATOR_HPP
#define OPENCV_FREETYPE_MASK_CREATOR_HPP

#include "api/render_priv.hpp"

#include <ft2build.h>
#include FT_FREETYPE_H

namespace cv   {
namespace gapi {
namespace wip  {
namespace draw {

class FreeTypeBitmaskCreator : public IBitmaskCreator
{
public:
    FreeTypeBitmaskCreator(const std::string& path)
    {
        FT_Init_FreeType(&m_library);
        FT_New_Face(m_library, path.c_str(), 0, &m_face);
    }

    virtual const cv::Size& computeMaskSize() override
    {
        m_slot = m_face->glyph;
        m_mask_size = cv::Size(0, 0);
        int bmp_w          = 0;
        int max_bottom_pad = 0;
        m_max_top_pad    = 0;

        m_glyphs.resize(m_text.text.size());
        m_pos.reserve(m_text.text.size());

        for (size_t i = 0; i < m_text.text.size(); ++i)
        {
            FT_Load_Char(m_face, m_text.text[i], FT_LOAD_RENDER);
            FT_Bitmap *bitmap = &(m_slot->bitmap);

            cv::Mat(bitmap->rows, bitmap->width, CV_8UC1, bitmap->buffer, bitmap->pitch).copyTo(m_glyphs[i]);

            int gl_bottom_pad = (m_slot->metrics.height - m_slot->metrics.horiBearingY) >> 6;
            max_bottom_pad = std::max(max_bottom_pad, gl_bottom_pad);
            m_max_top_pad = std::max(m_max_top_pad, m_slot->bitmap_top);

            // FIXME why bitmap->left is negative ?
            int gl_x_pad = m_slot->bitmap_left > 0 ? m_slot->bitmap_left + bmp_w - 1 : bmp_w;
            auto shift = (m_slot->advance.x >> 6);


            // FIXME why bitmap->width > shift m_slot->advance.x ?
            if (shift < bitmap->width)
            {
                gl_x_pad = bmp_w;
                bmp_w += bitmap->width;
            }
            else
            {
                bmp_w += shift;
            }

            m_pos.emplace_back(gl_x_pad, m_slot->bitmap_top);
        }

        int bmp_h = max_bottom_pad + m_max_top_pad;
        m_mask_size = cv::Size(bmp_w, bmp_h);
        m_baseline = max_bottom_pad;

        return m_mask_size;
    }

    int virtual createMask(cv::Mat& mask) override
    {
        mask = cv::Scalar(0);
        for (size_t i = 0; i < m_text.text.size(); ++i)
        {
            cv::Rect glyph_roi(m_pos[i].x, m_max_top_pad - m_pos[i].y, m_glyphs[i].cols, m_glyphs[i].rows);
            cv::Mat roi = mask(glyph_roi);
            m_glyphs[i].copyTo(roi);
        }
        return m_baseline;
    }

    void virtual setMaskParams(const cv::gapi::wip::draw::Text& text) override
    {
        m_text = text;

        // Convert OpenCV scale to Freetype text height
        //auto sz = cv::getTextSize(text.text, text.ff, text.fs, text.thick, nullptr);
        //int font_height = static_cast<int>(sz.height * 3.0 / 2); // 3.0 / 2 for better aligning with OpenCV
        int font_height = 50;

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
    FT_GlyphSlot  m_slot;

    std::vector<cv::Mat> m_glyphs;
    std::vector<cv::Point> m_pos;

    cv::gapi::wip::draw::Text m_text;
    int m_baseline;
    int m_max_top_pad;
    cv::Size m_mask_size;
};

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_FREETYPE_MASK_CREATOR_HPP

#endif // HAVE_FREETYPE
