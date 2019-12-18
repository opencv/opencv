// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "precomp.hpp"

#ifdef HAVE_FREETYPE

#include "api/ft_render.hpp"
#include "api/ft_render_priv.hpp"

#include <opencv2/gapi/util/throw.hpp>
#include <opencv2/gapi/own/assert.hpp>

cv::gapi::wip::draw::FTTextRender::Priv::Priv(const std::string& path)
{
    if (FT_Init_FreeType(&m_library) != 0)
    {
        cv::util::throw_error(std::runtime_error("Failed to initialize FT"));
    }

    if (FT_New_Face(m_library, path.c_str(), 0, &m_face))
    {
        FT_Done_FreeType(m_library);
        cv::util::throw_error(std::runtime_error("Failed to create a font face"));
    }
}

cv::Size cv::gapi::wip::draw::FTTextRender::Priv::getTextSize(const std::wstring& text, int fh, int* baseline)
{
    //
    //
    //
    //   ^                                                           diff between size and advance(2)
    //   |       ______________               width        width    |<->|
    //   |      |      **      |            |<------>| <------------|--->
    //   |      |     *  *     |            |________| |____________|___|________
    //   | left |    *    *    |       left |* * * * | |   * * * * *|   |     ^   ^
    //   |<---->|   ** ** **   |     <----->|*      *| |       *    |   |  t  |   |
    //   |      |  *        *  |     |      |*      *| |       *    |   |  o  | h |
    //   |      | *          * |     |      |* * * * | |       *   (1)  |  p  | e |  baseline
    //   O------|*------------*|-----O----- |*-------|-|----O--*----O---|-----*-i-|------------>
    //   |      |______________|     |      |*       | |*   |  *    |   |     ^ g |
    //   |      |              |     |      |*       | |*   |  *    |   |  b  | h |
    //   |      |    width     |     |      |*       | |*   |  *    |   |  o  | t |
    //   |      |<------------>|     |      |*       | | * *|*      |   |  t  |   |
    //   |                           |      |________| |____|_______|___|_____|___*
    //   |         advance           |       advance   |    |advance| (advance maybe less than width)
    //   <---------------------------><----------------|----><------>
    //                                                 |left| (left maybe is negative)
    //                                                 |<-->|
    //
    //
    //   O                                       - The pen position for any time
    //
    //   left (m_face->glyph->bitmap_left)       - The horizontal distance from the current pen position to the glyph's left bbox edge.
    //
    //   advance (m_face->glyph->advance.x >> 6) - The horizontal distance to increment (for left-to-right writing)
    //                                              or decrement (for right-to-left writing) the pen position after a
    //                                              glyph has been rendered when processing text
    //
    //   widht (bitmap->width)                   - The width of glyph
    //
    //
    //   Algorihm to compute size of the text bounding box:
    //
    //   1) Go through all symbols and shift pen position and save glyph parameters (left, advance, width)
    //      If left + pen postion < 0 set left to 0. For example it's maybe happened
    //      if we print first letter 'J' or any other letter with negative 'left'
    //      We want to render glyph in pen position + left, so we must't allow it to be negative
    //
    //   2) If width == 0 we must to skip this symbol and don't save parameters for him.
    //      For example width == 0 for space sometimes
    //
    //   3) Also we compute max top and max bottom it's required for compute baseline
    //
    //   3) At the end we'll get the pen position for the symbol next to the last.
    //      See (1) on picture.
    //
    //   4) As we can see the last pen position is isn't horizontal size yet.
    //      We need to check if the glyph goes beyound the last position of the pen
    //      To do this we can:
    //      a) Return to the previous position -advance
    //      b) Shift on left value +left
    //      c) Shift on width of the last glyph
    //
    //      Compare result position with pen position and choose max
    //
    //      We can compute diff and check if diff > 0 pen.x += diff.
    //      See (2) on picture.
    //
    //  5) Return size. Complete!!!
    //
    // See also about freetype glyph metrics:
    // https://www.freetype.org/freetype2/docs/glyphs/glyphs-3.html

    GAPI_Assert(!FT_Set_Pixel_Sizes(m_face, fh, fh) &&
                "Failed to set pixel size");

    cv::Point pen(0, 0);

    int max_bot      = 0;
    int max_top      = 0;
    int last_advance = 0;
    int last_width   = 0;
    int last_left    = 0;

    for (const auto& wc : text)
    {
        GAPI_Assert(!FT_Load_Char(m_face, wc, FT_LOAD_RENDER) &&
                    "Failed to load char");

        FT_Bitmap *bitmap = &(m_face->glyph->bitmap);

        int left    = m_face->glyph->bitmap_left;
        int advance = (m_face->glyph->advance.x >> 6);
        int width   = bitmap->width;

        // NB: Read (1) paragraph of algorithm description
        if (pen.x + left < 0)
        {
            left = 0;
        }

        int bot = (m_face->glyph->metrics.height - m_face->glyph->metrics.horiBearingY) >> 6;
        max_bot = std::max(max_bot, bot);
        max_top = std::max(max_top, m_face->glyph->bitmap_top);

        // NB: Read (2) paragraph of algorithm description
        if (width != 0)
        {
            last_width = width;
            last_advance = advance;
            last_left    = left;
        }

        pen.x += advance;
    }

    // NB: Read (4) paragraph of algorithm description
    int diff = (last_width + last_left) - last_advance;
    pen.x += (diff > 0) ? diff : 0;

    if (baseline)
    {
        *baseline = max_bot;
    }

    return {pen.x, max_bot + max_top};
}

void cv::gapi::wip::draw::FTTextRender::Priv::putText(cv::Mat& mat,
                                                       const std::wstring& text,
                                                       const cv::Point& org,
                                                       int fh)
{
    GAPI_Assert(!FT_Set_Pixel_Sizes(m_face, fh, fh) &&
                "Failed to set pixel size");

    cv::Point pen = org;
    for (const auto& wc : text)
    {
        GAPI_Assert(!FT_Load_Char(m_face, wc, FT_LOAD_RENDER) &&
                    "Failed to load char");
        FT_Bitmap *bitmap = &(m_face->glyph->bitmap);

        cv::Mat glyph(bitmap->rows, bitmap->width, CV_8UC1, bitmap->buffer, bitmap->pitch);

        int left    = m_face->glyph->bitmap_left;
        int top     = m_face->glyph->bitmap_top;
        int advance = (m_face->glyph->advance.x >> 6);

        if (pen.x + left < 0)
        {
            left = 0;
        }

        cv::Rect rect(pen.x + left, org.y - top, glyph.cols, glyph.rows);

        auto roi = mat(rect);
        roi += glyph;
        pen.x += advance;
    }
}

cv::gapi::wip::draw::FTTextRender::Priv::~Priv()
{
    FT_Done_Face(m_face);
    FT_Done_FreeType(m_library);
}

cv::gapi::wip::draw::FTTextRender::FTTextRender(const std::string& path)
    : m_priv(new Priv(path))
{
}

cv::Size cv::gapi::wip::draw::FTTextRender::getTextSize(const std::wstring& text,
                                                         int fh,
                                                         int* baseline)
{
    return m_priv->getTextSize(text, fh, baseline);
}

void cv::gapi::wip::draw::FTTextRender::putText(cv::Mat& mat,
                                                 const std::wstring& text,
                                                 const cv::Point& org,
                                                 int fh)
{
    m_priv->putText(mat, text, org, fh);
}

#endif // HAVE_FREETYPE
