// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

namespace cv {

// Matches cvtColor's RGBA<->mRGBA fixed-point rounding, so results stay bit-consistent.
static inline uchar mulDiv255(uchar a, uchar b)
{
    return (uchar)((a * b + 128) / 255);
}

class AlphaCompositeInvoker : public ParallelLoopBody
{
public:
    AlphaCompositeInvoker(const Mat& overlay, const Mat& background, Mat& dst)
        : overlay_(overlay), background_(background), dst_(dst), bgChannels_(background.channels())
    {
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int width = overlay_.cols;
        for (int y = range.start; y < range.end; ++y)
        {
            const uchar* ov = overlay_.ptr<uchar>(y);
            const uchar* bg = background_.ptr<uchar>(y);
            uchar* d = dst_.ptr<uchar>(y);

            if (bgChannels_ == 4)
            {
                for (int x = 0; x < width; ++x, ov += 4, bg += 4, d += 4)
                {
                    uchar as = ov[3];
                    uchar ad = bg[3];
                    uchar invAs = (uchar)(255 - as);
                    d[0] = saturate_cast<uchar>(ov[0] + mulDiv255(bg[0], invAs));
                    d[1] = saturate_cast<uchar>(ov[1] + mulDiv255(bg[1], invAs));
                    d[2] = saturate_cast<uchar>(ov[2] + mulDiv255(bg[2], invAs));
                    d[3] = saturate_cast<uchar>(as + mulDiv255(ad, invAs));
                }
            }
            else
            {
                for (int x = 0; x < width; ++x, ov += 4, bg += 3, d += 3)
                {
                    uchar invAs = (uchar)(255 - ov[3]);
                    d[0] = saturate_cast<uchar>(ov[0] + mulDiv255(bg[0], invAs));
                    d[1] = saturate_cast<uchar>(ov[1] + mulDiv255(bg[1], invAs));
                    d[2] = saturate_cast<uchar>(ov[2] + mulDiv255(bg[2], invAs));
                }
            }
        }
    }

private:
    const Mat& overlay_;
    const Mat& background_;
    Mat& dst_;
    int bgChannels_;
};

} // namespace cv

void cv::alphaComposite(InputArray _overlay, InputArray _background, OutputArray _dst, bool premultiplied)
{
    CV_INSTRUMENT_REGION();

    int bgType = _background.type();
    int bgChannels = CV_MAT_CN(bgType);

    CV_Assert(_overlay.type() == CV_8UC4);
    CV_Assert(CV_MAT_DEPTH(bgType) == CV_8U && (bgChannels == 3 || bgChannels == 4));
    CV_Assert(_overlay.size() == _background.size());

    Mat overlay = _overlay.getMat();
    Mat background = _background.getMat();

    Mat overlayPremul;
    if (premultiplied)
        overlayPremul = overlay;
    else
        cvtColor(overlay, overlayPremul, COLOR_RGBA2mRGBA);

    Mat backgroundPremul;
    if (bgChannels == 3 || premultiplied)
        backgroundPremul = background;
    else
        cvtColor(background, backgroundPremul, COLOR_RGBA2mRGBA);

    _dst.create(background.size(), bgType);
    Mat dst = _dst.getMat();

    // 4-channel straight-alpha output needs a temp buffer before un-premultiplying into dst.
    bool needsUnpremultiply = (bgChannels == 4 && !premultiplied);
    Mat composited = needsUnpremultiply ? Mat(background.size(), bgType) : dst;

    AlphaCompositeInvoker invoker(overlayPremul, backgroundPremul, composited);
    parallel_for_(Range(0, background.rows), invoker, background.total() / (double)(1 << 16));

    if (needsUnpremultiply)
        cvtColor(composited, dst, COLOR_mRGBA2RGBA);
}
