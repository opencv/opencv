// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

namespace cv {

// Matches cvtColor's RGBA<->mRGBA fixed-point rounding, so results stay bit-consistent.
static inline int mulDiv255(int a, int b)
{
    return (a * b + 128) / 255;
}

// One Porter-Duff weight, F = K0*255 + K1*a, where a is the *other* operand's alpha.
template<int K0, int K1>
struct CompositeTerm
{
    static inline int apply(int c, int f) { return mulDiv255(c, f); }
};

// F is identically 255, so the multiplication drops out. Keeps OVER as cheap as a bare copy.
template<>
struct CompositeTerm<1, 0>
{
    static inline int apply(int c, int) { return c; }
};

typedef void (*CompositeRowFunc)(const uchar* overlay, const uchar* background, uchar* dst, int width);

template<int CN, int A0, int A1, int B0, int B1>
static void compositeRow(const uchar* ov, const uchar* bg, uchar* d, int width)
{
    typedef CompositeTerm<A0, A1> OverlayTerm;
    typedef CompositeTerm<B0, B1> BackgroundTerm;

    for (int x = 0; x < width; ++x, ov += 4, bg += CN, d += CN)
    {
        int as = ov[3];
        int ad = (CN == 4) ? bg[3] : 255;
        int fa = A0 * 255 + A1 * ad;
        int fb = B0 * 255 + B1 * as;
        d[0] = saturate_cast<uchar>(OverlayTerm::apply(ov[0], fa) + BackgroundTerm::apply(bg[0], fb));
        d[1] = saturate_cast<uchar>(OverlayTerm::apply(ov[1], fa) + BackgroundTerm::apply(bg[1], fb));
        d[2] = saturate_cast<uchar>(OverlayTerm::apply(ov[2], fa) + BackgroundTerm::apply(bg[2], fb));
        if (CN == 4)
            d[3] = saturate_cast<uchar>(OverlayTerm::apply(as, fa) + BackgroundTerm::apply(ad, fb));
    }
}

template<int A0, int A1, int B0, int B1>
static CompositeRowFunc rowFuncFor(int bgChannels)
{
    return bgChannels == 4 ? &compositeRow<4, A0, A1, B0, B1> : &compositeRow<3, A0, A1, B0, B1>;
}

// F_a is a function of the background's alpha and F_b of the overlay's, as in the W3C table.
static CompositeRowFunc compositeRowFunc(int op, int bgChannels)
{
    switch (op)
    {                                   //      F_a       F_b
    case ALPHA_COMPOSITE_CLEAR:     return rowFuncFor< 0, 0,  0, 0>(bgChannels);
    case ALPHA_COMPOSITE_SOURCE:    return rowFuncFor< 1, 0,  0, 0>(bgChannels);
    case ALPHA_COMPOSITE_DEST:      return rowFuncFor< 0, 0,  1, 0>(bgChannels);
    case ALPHA_COMPOSITE_OVER:      return rowFuncFor< 1, 0,  1,-1>(bgChannels);
    case ALPHA_COMPOSITE_DEST_OVER: return rowFuncFor< 1,-1,  1, 0>(bgChannels);
    case ALPHA_COMPOSITE_IN:        return rowFuncFor< 0, 1,  0, 0>(bgChannels);
    case ALPHA_COMPOSITE_DEST_IN:   return rowFuncFor< 0, 0,  0, 1>(bgChannels);
    case ALPHA_COMPOSITE_OUT:       return rowFuncFor< 1,-1,  0, 0>(bgChannels);
    case ALPHA_COMPOSITE_DEST_OUT:  return rowFuncFor< 0, 0,  1,-1>(bgChannels);
    case ALPHA_COMPOSITE_ATOP:      return rowFuncFor< 0, 1,  1,-1>(bgChannels);
    case ALPHA_COMPOSITE_DEST_ATOP: return rowFuncFor< 1,-1,  0, 1>(bgChannels);
    case ALPHA_COMPOSITE_XOR:       return rowFuncFor< 1,-1,  1,-1>(bgChannels);
    case ALPHA_COMPOSITE_PLUS:      return rowFuncFor< 1, 0,  1, 0>(bgChannels);
    default:
        CV_Error_(Error::StsBadArg, ("unknown alpha compositing operator: %d", op));
    }
}

class AlphaCompositeInvoker : public ParallelLoopBody
{
public:
    AlphaCompositeInvoker(const Mat& overlay, const Mat& background, Mat& dst, CompositeRowFunc rowFunc)
        : overlay_(overlay), background_(background), dst_(dst), rowFunc_(rowFunc)
    {
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        for (int y = range.start; y < range.end; ++y)
            rowFunc_(overlay_.ptr<uchar>(y), background_.ptr<uchar>(y), dst_.ptr<uchar>(y), overlay_.cols);
    }

private:
    const Mat& overlay_;
    const Mat& background_;
    Mat& dst_;
    CompositeRowFunc rowFunc_;
};

} // namespace cv

void cv::alphaComposite(InputArray _overlay, InputArray _background, OutputArray _dst,
                        int op, bool premultiplied)
{
    CV_INSTRUMENT_REGION();

    int bgType = _background.type();
    int bgChannels = CV_MAT_CN(bgType);

    CV_Assert(_overlay.type() == CV_8UC4);
    CV_Assert(CV_MAT_DEPTH(bgType) == CV_8U && (bgChannels == 3 || bgChannels == 4));
    CV_Assert(_overlay.size() == _background.size());

    CompositeRowFunc rowFunc = compositeRowFunc(op, bgChannels);

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

    AlphaCompositeInvoker invoker(overlayPremul, backgroundPremul, composited, rowFunc);
    parallel_for_(Range(0, background.rows), invoker, background.total() / (double)(1 << 16));

    if (needsUnpremultiply)
        cvtColor(composited, dst, COLOR_mRGBA2RGBA);
}
