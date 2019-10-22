// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_OCV_MASK_CREATOR_HPP
#define OPENCV_OCV_MASK_CREATOR_HPP

#include "api/render_priv.hpp"

namespace cv   {
namespace gapi {
namespace wip  {
namespace draw {

class OCVBitmaskCreator : public IBitmaskCreator
{
public:
    virtual const cv::Size& computeMaskSize() override
    {
        auto ff   = FONT_HERSHEY_COMPLEX;
        m_thick = 1;

        auto sz = cv::getTextSize(m_text.text, FONT_HERSHEY_COMPLEX,
                                  m_text.fs, m_thick,
                                  &m_baseline);

        m_baseline += m_thick;

        m_mask_size = {sz.width, sz.height + m_baseline};

        return m_mask_size;
    }

    int virtual createMask(cv::Mat& mask) override
    {
        // Mask must be allocate outside
        GAPI_Assert(mask.size() == m_mask_size);
        GAPI_Assert(mask.type() == CV_8UC1);

        mask = cv::Scalar::all(0);

        cv::Point org((mask.cols - mask.size().width) / 2,
                      (mask.rows + mask.size().height - 2 * m_baseline) / 2);

        cv::putText(mask, m_text.text, org, FONT_HERSHEY_COMPLEX,
                    m_text.fs, 255, m_thick);

        return m_baseline;
    }

    void virtual setMaskParams(const cv::gapi::wip::draw::Text& text) override
    {
        m_text = text;
    }

private:
    cv::gapi::wip::draw::Text m_text;
    int m_baseline;
    int m_thick;
    cv::Size m_mask_size;
};

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_OCV_MASK_CREATOR_HPP
