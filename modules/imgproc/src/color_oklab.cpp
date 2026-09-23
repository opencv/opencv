// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/********************************* COPYRIGHT NOTICE *******************************\
  The Oklab colour space and the conversion formulas implemented below are due to
  Bjorn Ottosson, "A perceptual color space for image processing":
  https://bottosson.github.io/posts/oklab/
\**********************************************************************************/

#include "precomp.hpp"
#include "color.hpp"

namespace cv
{

// sRGB electro-optical transfer function and its inverse, applied per channel to values
// already normalized to [0, 1].
static inline float oklabSRGB2Linear(float c)
{
    return c <= 0.04045f ? c * (1.f / 12.92f)
                         : std::pow((c + 0.055f) * (1.f / 1.055f), 2.4f);
}

static inline float oklabLinear2SRGB(float c)
{
    c = std::max(c, 0.f);
    return c <= 0.0031308f ? c * 12.92f
                          : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

// linear sRGB -> Oklab. R, G, B are linear-light (gamma already removed) and in [0, 1]
// for in-gamut colors; L comes out in [0, 1], a and b are unbounded but stay roughly
// within [-0.5, 0.5] for the sRGB gamut.
static inline void linearRGB2Oklab(float R, float G, float B, float& L, float& a, float& b)
{
    float l = 0.4122214708f*R + 0.5363325363f*G + 0.0514459929f*B;
    float m = 0.2119034982f*R + 0.6806995451f*G + 0.1073969566f*B;
    float s = 0.0883024619f*R + 0.2817188376f*G + 0.6299787005f*B;

    float l_ = cubeRoot(l), m_ = cubeRoot(m), s_ = cubeRoot(s);

    L = 0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_;
    a = 1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_;
    b = 0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_;
}

// Oklab -> linear sRGB, the exact inverse of linearRGB2Oklab above.
static inline void oklab2LinearRGB(float L, float a, float b, float& R, float& G, float& B)
{
    float l_ = L + 0.3963377774f*a + 0.2158037573f*b;
    float m_ = L - 0.1055613458f*a - 0.0638541728f*b;
    float s_ = L - 0.0894841775f*a - 1.2914855480f*b;

    float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;

    R = +4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s;
    G = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s;
    B = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s;
}

struct RGB2Oklab_f
{
    typedef float channel_type;

    RGB2Oklab_f(int _srccn, int _blueIdx) : srccn(_srccn), blueIdx(_blueIdx) {}

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();
        int scn = srccn, bidx = blueIdx;
        for (int i = 0; i < n; i++, src += scn, dst += 3)
        {
            float B = oklabSRGB2Linear(src[bidx]);
            float G = oklabSRGB2Linear(src[1]);
            float R = oklabSRGB2Linear(src[bidx ^ 2]);
            float L, a, b;
            linearRGB2Oklab(R, G, B, L, a, b);
            dst[0] = L; dst[1] = a; dst[2] = b;
        }
    }
    int srccn, blueIdx;
};

struct Oklab2RGB_f
{
    typedef float channel_type;

    Oklab2RGB_f(int _dstcn, int _blueIdx) : dstcn(_dstcn), blueIdx(_blueIdx) {}

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();
        int dcn = dstcn, bidx = blueIdx;
        float alpha = ColorChannel<float>::max();
        for (int i = 0; i < n; i++, src += 3, dst += dcn)
        {
            float R, G, B;
            oklab2LinearRGB(src[0], src[1], src[2], R, G, B);
            dst[bidx] = oklabLinear2SRGB(B);
            dst[1] = oklabLinear2SRGB(G);
            dst[bidx ^ 2] = oklabLinear2SRGB(R);
            if (dcn == 4)
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
};

// 8-bit encoding: L in [0, 1] scales directly to [0, 255]; a and b use the same *255
// scale but are offset by 128 (mirroring how CIE Lab's a*/b* are stored as uchar), so
// the sRGB gamut -- roughly a, b in [-0.4, 0.4] -- lands well within the byte's range
// without saturating except for extreme, highly-saturated out-of-gamut inputs.
struct RGB2Oklab_b
{
    typedef uchar channel_type;

    RGB2Oklab_b(int _srccn, int _blueIdx) : srccn(_srccn), blueIdx(_blueIdx) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();
        int scn = srccn, bidx = blueIdx;
        static const float inv255 = 1.f / 255.f;
        for (int i = 0; i < n; i++, src += scn, dst += 3)
        {
            float B = oklabSRGB2Linear(src[bidx] * inv255);
            float G = oklabSRGB2Linear(src[1] * inv255);
            float R = oklabSRGB2Linear(src[bidx ^ 2] * inv255);
            float L, a, b;
            linearRGB2Oklab(R, G, B, L, a, b);
            dst[0] = saturate_cast<uchar>(cvRound(L * 255.f));
            dst[1] = saturate_cast<uchar>(cvRound(a * 255.f) + 128);
            dst[2] = saturate_cast<uchar>(cvRound(b * 255.f) + 128);
        }
    }
    int srccn, blueIdx;
};

struct Oklab2RGB_b
{
    typedef uchar channel_type;

    Oklab2RGB_b(int _dstcn, int _blueIdx) : dstcn(_dstcn), blueIdx(_blueIdx) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();
        int dcn = dstcn, bidx = blueIdx;
        uchar alpha = ColorChannel<uchar>::max();
        static const float inv255 = 1.f / 255.f;
        for (int i = 0; i < n; i++, src += 3, dst += dcn)
        {
            float L = src[0] * inv255;
            float a = (src[1] - 128) * inv255;
            float b = (src[2] - 128) * inv255;
            float R, G, B;
            oklab2LinearRGB(L, a, b, R, G, B);
            dst[bidx] = saturate_cast<uchar>(cvRound(oklabLinear2SRGB(B) * 255.f));
            dst[1] = saturate_cast<uchar>(cvRound(oklabLinear2SRGB(G) * 255.f));
            dst[bidx ^ 2] = saturate_cast<uchar>(cvRound(oklabLinear2SRGB(R) * 255.f));
            if (dcn == 4)
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
};

void cvtColorBGR2Oklab(InputArray _src, OutputArray _dst, bool swapb)
{
    CvtHelper<Set<3, 4>, Set<3>, Set<CV_8U, CV_32F>> h(_src, _dst, 3);

    int blueIdx = swapb ? 2 : 0;
    if (h.depth == CV_8U)
        CvtColorLoop(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     RGB2Oklab_b(h.scn, blueIdx));
    else
        CvtColorLoop(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     RGB2Oklab_f(h.scn, blueIdx));
}

void cvtColorOklab2BGR(InputArray _src, OutputArray _dst, int dcn, bool swapb)
{
    if (dcn <= 0) dcn = 3;
    CvtHelper<Set<3>, Set<3, 4>, Set<CV_8U, CV_32F>> h(_src, _dst, dcn);

    int blueIdx = swapb ? 2 : 0;
    if (h.depth == CV_8U)
        CvtColorLoop(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     Oklab2RGB_b(dcn, blueIdx));
    else
        CvtColorLoop(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     Oklab2RGB_f(dcn, blueIdx));
}

} // namespace cv
