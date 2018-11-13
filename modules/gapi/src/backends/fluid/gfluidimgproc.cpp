// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "precomp.hpp"

#include "opencv2/gapi/own/assert.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/imgproc/types_c.h"

#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

#include "opencv2/gapi/own/types.hpp"

#include "opencv2/gapi/fluid/gfluidbuffer.hpp"
#include "opencv2/gapi/fluid/gfluidkernel.hpp"

#include "gfluidbuffer_priv.hpp"
#include "gfluidbackend.hpp"
#include "gfluidimgproc.hpp"
#include "gfluidutils.hpp"

#include "gfluidimgproc_func.hpp"

#include <opencv2/core/hal/intrin.hpp>

#include <cmath>
#include <cstdlib>

namespace cv {
namespace gapi {
namespace fluid {

//----------------------------------
//
// Fluid kernels: RGB2Gray, BGR2Gray
//
//----------------------------------

// Y' = 0.299*R' + 0.587*G' + 0.114*B'
// U' = (B' - Y')*0.492
// V' = (R' - Y')*0.877
static const float coef_rgb2yuv_bt601[5] = {0.299f, 0.587f, 0.114f, 0.492f, 0.877f};

// R' = Y' + 1.140*V'
// G' = Y' - 0.394*U' - 0.581*V'
// B' = Y' + 2.032*U'
static const float coef_yuv2rgb_bt601[4] = {1.140f, -0.394f, -0.581f, 2.032f};

static void run_rgb2gray(Buffer &dst, const View &src, float coef_r, float coef_g, float coef_b)
{
    GAPI_Assert(src.meta().depth == CV_8U);
    GAPI_Assert(dst.meta().depth == CV_8U);
    GAPI_Assert(src.meta().chan == 3);
    GAPI_Assert(dst.meta().chan == 1);
    GAPI_Assert(src.length() == dst.length());

    const auto *in  = src.InLine<uchar>(0);
          auto *out = dst.OutLine<uchar>();

    int width = dst.length();

    // TODO: Vectorize for SIMD
    for (int w=0; w < width; w++)
    {
        uchar r = in[3*w    ];
        uchar g = in[3*w + 1];
        uchar b = in[3*w + 2];
        float result = coef_r*r + coef_g*g + coef_b*b;
        out[w] = saturate<uchar>(result, roundf);
    }
}

GAPI_FLUID_KERNEL(GFluidRGB2GrayCustom, cv::gapi::imgproc::GRGB2GrayCustom, false)
{
    static const int Window = 1;

    static void run(const View &src, float coef_r, float coef_g, float coef_b, Buffer &dst)
    {
        run_rgb2gray(dst, src, coef_r, coef_g, coef_b);
    }
};

GAPI_FLUID_KERNEL(GFluidRGB2Gray, cv::gapi::imgproc::GRGB2Gray, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        float coef_r = coef_rgb2yuv_bt601[0];
        float coef_g = coef_rgb2yuv_bt601[1];
        float coef_b = coef_rgb2yuv_bt601[2];
        run_rgb2gray(dst, src, coef_r, coef_g, coef_b);
    }
};

GAPI_FLUID_KERNEL(GFluidBGR2Gray, cv::gapi::imgproc::GBGR2Gray, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        float coef_r = coef_rgb2yuv_bt601[0];
        float coef_g = coef_rgb2yuv_bt601[1];
        float coef_b = coef_rgb2yuv_bt601[2];
        run_rgb2gray(dst, src, coef_b, coef_g, coef_r);
    }
};

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, YUV-to-RGB
//
//--------------------------------------

static void run_rgb2yuv(Buffer &dst, const View &src, const float coef[5])
{
    GAPI_Assert(src.meta().depth == CV_8U);
    GAPI_Assert(dst.meta().depth == CV_8U);
    GAPI_Assert(src.meta().chan == 3);
    GAPI_Assert(dst.meta().chan == 3);
    GAPI_Assert(src.length() == dst.length());

    const auto *in  = src.InLine<uchar>(0);
          auto *out = dst.OutLine<uchar>();

    int width = dst.length();

    // TODO: Vectorize for SIMD
    for (int w=0; w < width; w++)
    {
        uchar r = in[3*w    ];
        uchar g = in[3*w + 1];
        uchar b = in[3*w + 2];
        float y = coef[0]*r + coef[1]*g + coef[2]*b;
        float u = coef[3]*(b - y) + 128;
        float v = coef[4]*(r - y) + 128;
        out[3*w    ] = saturate<uchar>(y, roundf);
        out[3*w + 1] = saturate<uchar>(u, roundf);
        out[3*w + 2] = saturate<uchar>(v, roundf);
    }
}

static void run_yuv2rgb(Buffer &dst, const View &src, const float coef[4])
{
    GAPI_Assert(src.meta().depth == CV_8U);
    GAPI_Assert(dst.meta().depth == CV_8U);
    GAPI_Assert(src.meta().chan == 3);
    GAPI_Assert(dst.meta().chan == 3);
    GAPI_Assert(src.length() == dst.length());

    const auto *in  = src.InLine<uchar>(0);
          auto *out = dst.OutLine<uchar>();

    int width = dst.length();

    // TODO: Vectorize for SIMD
    for (int w=0; w < width; w++)
    {
        uchar y = in[3*w    ];
        int   u = in[3*w + 1] - 128;
        int   v = in[3*w + 2] - 128;
        float r = y             + coef[0]*v;
        float g = y + coef[1]*u + coef[2]*v;
        float b = y + coef[3]*u;
        out[3*w    ] = saturate<uchar>(r, roundf);
        out[3*w + 1] = saturate<uchar>(g, roundf);
        out[3*w + 2] = saturate<uchar>(b, roundf);
    }
}

GAPI_FLUID_KERNEL(GFluidRGB2YUV, cv::gapi::imgproc::GRGB2YUV, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        run_rgb2yuv(dst, src, coef_rgb2yuv_bt601);
    }
};

GAPI_FLUID_KERNEL(GFluidYUV2RGB, cv::gapi::imgproc::GYUV2RGB, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        run_yuv2rgb(dst, src, coef_yuv2rgb_bt601);
    }
};

//--------------------------------------
//
// Fluid kernels: RGB-to-Lab, BGR-to-LUV
//
//--------------------------------------

enum LabLUV { LL_Lab, LL_LUV };

// gamma-correction (inverse) for sRGB, 1/gamma=2.4 for inverse, like for Mac OS (?)
static inline float f_gamma(float x)
{
    return x <= 0.04045f ? x*(1.f/12.92f) : std::pow((x + 0.055f)*(1/1.055f), 2.4f);
}

// saturate into interval [0, 1]
static inline float clip01(float value)
{
    return value < 0? 0:
           value > 1? 1:
           value;
}

static inline void f_rgb2xyz(float  R, float  G, float  B,
                             float& X, float& Y, float& Z)
{
    X = clip01(0.412453f*R + 0.357580f*G + 0.180423f*B);
    Y = clip01(0.212671f*R + 0.715160f*G + 0.072169f*B);
    Z = clip01(0.019334f*R + 0.119193f*G + 0.950227f*B);
}

static inline void f_xyz2lab(float  X, float  Y, float  Z,
                             float& L, float& a, float& b)
{
    // CIE XYZ values of reference white point for D65 illuminant
    static const float Xn = 0.950456f, Yn = 1.f, Zn = 1.088754f;

    // Other coefficients below:
    // 7.787f    = (29/3)^3/(29*4)
    // 0.008856f = (6/29)^3
    // 903.3     = (29/3)^3

    float x = X/Xn, y = Y/Yn, z = Z/Zn;

    auto f = [](float t){ return t>0.008856f? std::cbrt(t): (7.787f*t + 16.f/116.f); };

    float fx = f(x), fy = f(y), fz = f(z);

    L = y > 0.008856f ? (116.f*std::cbrt(y) - 16.f) : (903.3f * y);
    a = 500.f * (fx - fy);
    b = 200.f * (fy - fz);
}

static inline void f_xyz2luv(float  X, float  Y, float  Z,
                             float& L, float& u, float& v)
{
    static const float un = 0.19793943f, vn = 0.46831096f;

    float u1 = 4*X / (X + 15*Y + 3*Z);
    float v1 = 9*Y / (X + 15*Y + 3*Z);

    L = Y > 0.008856f ? (116.f*std::cbrt(Y) - 16.f) : (903.3f * Y);
    u = 13*L * (u1 - un);
    v = 13*L * (v1 - vn);
}

// compile-time parameters: output format (Lab/LUV),
// and position of blue channel in BGR/RGB (0 or 2)
template<LabLUV labluv, int blue=0>
static void run_rgb2labluv(Buffer &dst, const View &src)
{
    GAPI_Assert(src.meta().depth == CV_8U);
    GAPI_Assert(dst.meta().depth == CV_8U);
    GAPI_Assert(src.meta().chan == 3);
    GAPI_Assert(dst.meta().chan == 3);
    GAPI_Assert(src.length() == dst.length());

    const auto *in  = src.InLine<uchar>(0);
          auto *out = dst.OutLine<uchar>();

    int width = dst.length();

    for (int w=0; w < width; w++)
    {
        float R, G, B;
        B = in[3*w +    blue ] / 255.f;
        G = in[3*w +    1    ] / 255.f;
        R = in[3*w + (2^blue)] / 255.f;

        B = f_gamma( B );
        G = f_gamma( G );
        R = f_gamma( R );

        float X, Y, Z;
        f_rgb2xyz(R, G, B, X, Y, Z);

        // compile-time `if`
        if (LL_Lab == labluv)
        {
            float L, a, b;
            f_xyz2lab(X, Y, Z, L, a, b);

            out[3*w    ] = saturate<uchar>(L * 255.f/100, roundf);
            out[3*w + 1] = saturate<uchar>(a + 128, roundf);
            out[3*w + 2] = saturate<uchar>(b + 128, roundf);
        }
        else if (LL_LUV == labluv)
        {
            float L, u, v;
            f_xyz2luv(X, Y, Z, L, u, v);

            out[3*w    ] = saturate<uchar>( L        * 255.f/100, roundf);
            out[3*w + 1] = saturate<uchar>((u + 134) * 255.f/354, roundf);
            out[3*w + 2] = saturate<uchar>((v + 140) * 255.f/262, roundf);
        }
        else
            CV_Error(cv::Error::StsBadArg, "unsupported color conversion");;
    }
}

GAPI_FLUID_KERNEL(GFluidRGB2Lab, cv::gapi::imgproc::GRGB2Lab, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        static const int blue = 2; // RGB: 0=red, 1=green, 2=blue
        run_rgb2labluv<LL_Lab, blue>(dst, src);
    }
};

GAPI_FLUID_KERNEL(GFluidBGR2LUV, cv::gapi::imgproc::GBGR2LUV, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        static const int blue = 0; // BGR: 0=blue, 1=green, 2=red
        run_rgb2labluv<LL_LUV, blue>(dst, src);
    }
};

//-------------------------------
//
// Fluid kernels: blur, boxFilter
//
//-------------------------------

static const int maxKernelSize = 9;

template<typename DST, typename SRC>
static void run_boxfilter(Buffer &dst, const View &src, const cv::Size &kernelSize,
                          const cv::Point& /* anchor */, bool normalize)
{
    GAPI_Assert(kernelSize.width <= maxKernelSize);
    GAPI_Assert(kernelSize.width == kernelSize.height);

    int kernel = kernelSize.width;
    int border = (kernel - 1) / 2;

    const SRC *in[ maxKernelSize ];
          DST *out;

    for (int i=0; i < kernel; i++)
    {
        in[i] = src.InLine<SRC>(i - border);
    }

    out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    GAPI_DbgAssert(chan <= 4);

    for (int w=0; w < width; w++)
    {
        float sum[4] = {0, 0, 0, 0};

        for (int i=0; i < kernel; i++)
        {
            for (int j=0; j < kernel; j++)
            {
                for (int c=0; c < chan; c++)
                    sum[c] += in[i][(w + j - border)*chan + c];
            }
        }

        for (int c=0; c < chan; c++)
        {
            float result = normalize? sum[c]/(kernel * kernel) : sum[c];

            out[w*chan + c] = saturate<DST>(result, rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidBlur, cv::gapi::imgproc::GBlur, false)
{
    static const int Window = 3;

    static void run(const View &src, const cv::Size& kernelSize, const cv::Point& anchor,
                    int /* borderType */, const cv::Scalar& /* borderValue */, Buffer &dst)
    {
        // TODO: support sizes 3, 5, 7, 9, ...
        GAPI_Assert(kernelSize.width  == 3 && kernelSize.height == 3);

        // TODO: suport non-trivial anchor
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        static const bool normalize = true;

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_(ushort, ushort, run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_( short,  short, run_boxfilter, dst, src, kernelSize, anchor, normalize);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                            const cv::Size    & /* kernelSize */,
                            const cv::Point   & /* anchor */,
                                      int          borderType,
                            const cv::Scalar  &    borderValue)
    {
        return { borderType, borderValue};
    }
};

GAPI_FLUID_KERNEL(GFluidBoxFilter, cv::gapi::imgproc::GBoxFilter, false)
{
    static const int Window = 3;

    static void run(const     View  &    src,
                              int     /* ddepth */,
                    const cv::Size  &    kernelSize,
                    const cv::Point &   anchor,
                              bool       normalize,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst)
    {
        // TODO: support sizes 3, 5, 7, 9, ...
        GAPI_Assert(kernelSize.width  == 3 && kernelSize.height == 3);

        // TODO: suport non-trivial anchor
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_(ushort, ushort, run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_( short,  short, run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_( float, uchar , run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_( float, ushort, run_boxfilter, dst, src, kernelSize, anchor, normalize);
        UNARY_( float,  short, run_boxfilter, dst, src, kernelSize, anchor, normalize);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                                      int       /* ddepth */,
                            const cv::Size    & /* kernelSize */,
                            const cv::Point   & /* anchor */,
                                      bool      /* normalize */,
                                      int          borderType,
                            const cv::Scalar  &    borderValue)
    {
        return { borderType, borderValue};
    }
};

//-------------------------
//
// Fluid kernels: sepFilter
//
//-------------------------

template<typename T>
static void getKernel(T k[], const cv::Mat& kernel)
{
    GAPI_Assert(kernel.channels() == 1);

    int depth = CV_MAT_DEPTH(kernel.type());
    int cols = kernel.cols;
    int rows = kernel.rows;

    switch ( depth )
    {
    case CV_8U:
        for (int h=0; h < rows; h++)
        for (int w=0; w < cols; w++)
            k[h*cols + w] = static_cast<T>( kernel.at<uchar>(h, w) );
        break;
    case CV_16U:
        for (int h=0; h < rows; h++)
        for (int w=0; w < cols; w++)
            k[h*cols + w] = static_cast<T>( kernel.at<ushort>(h, w) );
        break;
    case CV_16S:
        for (int h=0; h < rows; h++)
        for (int w=0; w < cols; w++)
            k[h*cols + w] = static_cast<T>( kernel.at<short>(h, w) );
        break;
    case CV_32F:
        for (int h=0; h < rows; h++)
        for (int w=0; w < cols; w++)
            k[h*cols + w] = static_cast<T>( kernel.at<float>(h, w) );
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported kernel type");
    }
}

template<typename DST, typename SRC>
static void run_sepfilter(Buffer& dst, const View& src,
                          const float kx[], int kxLen,
                          const float ky[], int kyLen,
                          const cv::Point& /* anchor */,
                          float delta=0)
{
    static const int maxLines = 9;
    GAPI_Assert(kyLen <= maxLines);

    const SRC *in[ maxLines ];
          DST *out;

    int border = (kyLen - 1) / 2;
    for (int i=0; i < kyLen; i++)
    {
        in[i] = src.InLine<SRC>(i - border);
    }

    out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    for (int w=0; w < width; w++)
    {
        // TODO: make this cycle innermost
        for (int c=0; c < chan; c++)
        {
            float sum=0;

            for (int i=0; i < kyLen; i++)
            {
                float sumi=0;

                for (int j=0; j < kxLen; j++)
                {
                    sumi += in[i][(w + j - border)*chan + c] * kx[j];
                }

                sum += sumi * ky[i];
            }

            float result = sum + delta;

            out[w*chan + c] = saturate<DST>(result, rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidSepFilter, cv::gapi::imgproc::GSepFilter, true)
{
    static const int Window = 3;

    static void run(const     View&      src,
                              int     /* ddepth */,
                    const cv::Mat&       kernX,
                    const cv::Mat&       kernY,
                    const cv::Point&     anchor,
                    const cv::Scalar&    delta_,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        // TODO: support non-trivial anchors
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        // TODO: support kernel heights 3, 5, 7, 9, ...
        GAPI_Assert((kernY.rows == 1 || kernY.cols == 1)  && (kernY.cols * kernY.rows == 3));
        GAPI_Assert((kernX.rows == 1 || kernX.cols == 1));

        int kxLen = kernX.rows * kernX.cols;
        int kyLen = kernY.rows * kernY.cols;

        float *kx = scratch.OutLine<float>();
        float *ky = kx + kxLen;

        float delta = static_cast<float>(delta_[0]);

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_sepfilter, dst, src, kx, kxLen, ky, kyLen, anchor, delta);
        UNARY_(ushort, ushort, run_sepfilter, dst, src, kx, kxLen, ky, kyLen, anchor, delta);
        UNARY_( short,  short, run_sepfilter, dst, src, kx, kxLen, ky, kyLen, anchor, delta);
        UNARY_( float,  float, run_sepfilter, dst, src, kx, kxLen, ky, kyLen, anchor, delta);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc& /* in */,
                                  int       /* ddepth */,
                            const Mat     &    kernX,
                            const Mat     &    kernY,
                            const Point   & /* anchor */,
                            const Scalar  & /* delta */,
                                  int       /* borderType */,
                            const Scalar  & /* borderValue */,
                                  Buffer  &    scratch)
    {
        int kxLen = kernX.rows * kernX.cols;
        int kyLen = kernY.rows * kernY.cols;

        cv::gapi::own::Size bufsize(kxLen + kyLen, 1);
        GMatDesc bufdesc = {CV_32F, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        // FIXME: move to resetScratch stage ?
        float *kx = scratch.OutLine<float>();
        float *ky = kx + kxLen;
        getKernel(kx, kernX);
        getKernel(ky, kernY);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                                      int       /* ddepth */,
                            const cv::Mat&      /* kernX */,
                            const cv::Mat&      /* kernY */,
                            const cv::Point&    /* anchor */,
                            const cv::Scalar&   /* delta */,
                                      int          borderType,
                            const cv::Scalar&      borderValue)
    {
        return { borderType, borderValue};
    }
};

//----------------------------
//
// Fluid kernels: gaussianBlur
//
//----------------------------

GAPI_FLUID_KERNEL(GFluidGaussBlur, cv::gapi::imgproc::GGaussBlur, true)
{
    // TODO: support kernel height 3, 5, 7, 9, ...
    static const int Window = 3;

    static void run(const     View  &    src,
                    const cv::Size  &    ksize,
                              double  /* sigmaX */,
                              double  /* sigmaY */,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        GAPI_Assert(ksize.height == 3);

        int kxsize = ksize.width;
        int kysize = ksize.height;

        auto *kx = scratch.OutLine<float>(); // cached kernX data
        auto *ky = kx + kxsize;              // cached kernY data

        auto  anchor = cv::Point(-1, -1);
        float delta = 0.f;

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_sepfilter, dst, src, kx, kxsize, ky, kysize, anchor, delta);
        UNARY_(ushort, ushort, run_sepfilter, dst, src, kx, kxsize, ky, kysize, anchor, delta);
        UNARY_( short,  short, run_sepfilter, dst, src, kx, kxsize, ky, kysize, anchor, delta);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc& /* in */,
                            const cv::Size &   ksize,
                                  double       sigmaX,
                                  double       sigmaY,
                                  int       /* borderType */,
                            const cv::Scalar  & /* borderValue */,
                                  Buffer  &    scratch)
    {
        int kxsize = ksize.width;
        int kysize = ksize.height;

        cv::gapi::own::Size bufsize(kxsize + kysize, 1);
        GMatDesc bufdesc = {CV_32F, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        // FIXME: fill buffer at resetScratch stage?

        if (sigmaX == 0)
            sigmaX = 0.3 * ((kxsize - 1)/2. - 1) + 0.8;

        if (sigmaY == 0)
            sigmaY = sigmaX;

        Mat kernX = getGaussianKernel(kxsize, sigmaX, CV_32F);

        Mat kernY = kernX;
        if (sigmaY != sigmaX)
            kernY = getGaussianKernel(kysize, sigmaY, CV_32F);

        auto *kx = scratch.OutLine<float>();
        auto *ky = kx + kxsize;

        getKernel(kx, kernX);
        getKernel(ky, kernY);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                            const cv::Size    & /* ksize */,
                                      double    /* sigmaX */,
                                      double    /* sigmaY */,
                                      int          borderType,
                            const cv::Scalar  &    borderValue)
    {
        return { borderType, borderValue};
    }
};

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

template<typename DST, typename SRC>
static void run_sobel(Buffer& dst,
                const View  & src,
                const float   kx[],
                const float   ky[],
                      int     ksize,
                      float   scale,  // default: 1
                      float   delta,  // default: 0
                      float  *buf[])
{
    static const int kmax = 11;
    GAPI_Assert(ksize <= kmax);

    const SRC *in[ kmax ];
          DST *out;

    int border = (ksize - 1) / 2;
    for (int i=0; i < ksize; i++)
    {
        in[i] = src.InLine<SRC>(i - border);
    }

    out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    GAPI_DbgAssert(ksize == 3);
//  float buf[3][width * chan];

    int y  = dst.y();
    int y0 = dst.priv().writeStart();
//  int y1 = dst.priv().writeEnd();

    run_sobel_row(out, in, width, chan, kx, ky, border, scale, delta, buf, y, y0);
}

GAPI_FLUID_KERNEL(GFluidSobel, cv::gapi::imgproc::GSobel, true)
{
    static const int Window = 3;

    static void run(const     View  &    src,
                              int     /* ddepth */,
                              int     /* dx */,
                              int     /* dy */,
                              int        ksize,
                              double    _scale,
                              double    _delta,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        // TODO: support kernel height 3, 5, 7, 9, ...
        GAPI_Assert(ksize == 3 || ksize == CV_SCHARR);

        int ksz = (ksize == CV_SCHARR)? 3: ksize;

        auto *kx = scratch.OutLine<float>();
        auto *ky = kx + ksz;

        int width = dst.meta().size.width;
        int chan  = dst.meta().chan;

        float *buf[3];
        buf[0] = ky + ksz;
        buf[1] = buf[0] + width*chan;
        buf[2] = buf[1] + width*chan;

        auto scale = static_cast<float>(_scale);
        auto delta = static_cast<float>(_delta);

        //     DST     SRC     OP         __VA_ARGS__
        UNARY_(uchar , uchar , run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_(ushort, ushort, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( short, uchar , run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( short, ushort, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( short,  short, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( float, uchar , run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( float, ushort, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( float,  short, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);
        UNARY_( float,  float, run_sobel, dst, src, kx, ky, ksz, scale, delta, buf);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc&    in,
                                  int       /* ddepth */,
                                  int          dx,
                                  int          dy,
                                  int          ksize,
                                  double    /* scale */,
                                  double    /* delta */,
                                  int       /* borderType */,
                            const Scalar  & /* borderValue */,
                                  Buffer  &    scratch)
    {
        // TODO: support kernel height 3, 5, 7, 9, ...
        GAPI_Assert(ksize == 3 || ksize == CV_SCHARR);
        int ksz = (ksize == CV_SCHARR) ? 3 : ksize;

        int width = in.size.width;
        int chan  = in.chan;

        int buflen = ksz + ksz            // kernels: kx, ky
                   + ksz * width * chan;  // working buffers

        cv::gapi::own::Size bufsize(buflen, 1);
        GMatDesc bufdesc = {CV_32F, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        auto *kx = scratch.OutLine<float>();
        auto *ky = kx + ksz;

        Mat kxmat(1, ksize, CV_32FC1, kx);
        Mat kymat(ksize, 1, CV_32FC1, ky);
        getDerivKernels(kxmat, kymat, dx, dy, ksize);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                                      int       /* ddepth */,
                                      int       /* dx */,
                                      int       /* dy */,
                                      int       /* ksize */,
                                      double    /* scale */,
                                      double    /* delta */,
                                      int          borderType,
                            const cv::Scalar  &    borderValue)
    {
        return {borderType, borderValue};
    }
};

//------------------------
//
// Fluid kernels: filter2D
//
//------------------------

template<typename DST, typename SRC>
static void run_filter2d(Buffer& dst, const View& src,
                         const float k[], int k_rows, int k_cols,
                         const cv::Point& /* anchor */,
                         float delta=0)
{
    static const int maxLines = 9;
    GAPI_Assert(k_rows <= maxLines);

    const SRC *in[ maxLines ];
          DST *out;

    int border_x = (k_cols - 1) / 2;
    int border_y = (k_rows - 1) / 2;

    for (int i=0; i < k_rows; i++)
    {
        in[i] = src.InLine<SRC>(i - border_y);
    }

    out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    for (int w=0; w < width; w++)
    {
        // TODO: make this cycle innermost
        for (int c=0; c < chan; c++)
        {
            float sum = 0;

            for (int i=0; i < k_rows; i++)
            for (int j=0; j < k_cols; j++)
            {
                sum += in[i][(w + j - border_x)*chan + c] * k[k_cols*i + j];
            }

            float result = sum + delta;

            out[w*chan + c] = saturate<DST>(result, rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidFilter2D, cv::gapi::imgproc::GFilter2D, true)
{
    static const int Window = 3;

    static void run(const     View  &    src,
                              int     /* ddepth */,
                    const cv::Mat   &    kernel,
                    const cv::Point &    anchor,
                    const cv::Scalar&    delta_,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        // TODO: support non-trivial anchors
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        // TODO: support kernel heights 3, 5, 7, 9, ...
        GAPI_Assert(kernel.rows == 3 && kernel.cols == 3);

        float delta = static_cast<float>(delta_[0]);

        int k_rows = kernel.rows;
        int k_cols = kernel.cols;
        const float *k = scratch.OutLine<float>(); // copy of kernel.data

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_(ushort, ushort, run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_( short,  short, run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_( float, uchar , run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_( float, ushort, run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_( float,  short, run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);
        UNARY_( float,  float, run_filter2d, dst, src, k, k_rows, k_cols, anchor, delta);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const cv::GMatDesc& /* in */,
                                      int       /* ddepth */,
                            const cv::Mat     &    kernel,
                            const cv::Point   & /* anchor */,
                            const cv::Scalar  & /* delta */,
                                      int       /* borderType */,
                            const cv::Scalar  & /* borderValue */,
                                      Buffer  &    scratch)
    {
        cv::gapi::own::Size bufsize(kernel.rows * kernel.cols, 1);
        GMatDesc bufdesc = {CV_32F, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        // FIXME: move to resetScratch stage ?
        float *data = scratch.OutLine<float>();
        getKernel(data, kernel);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                                      int       /* ddepth */,
                            const cv::Mat&      /* kernel */,
                            const cv::Point&    /* anchor */,
                            const cv::Scalar&   /* delta */,
                                      int          borderType,
                            const cv::Scalar&      borderValue)
    {
        return { borderType, borderValue};
    }
};

//-----------------------------
//
// Fluid kernels: erode, dilate
//
//-----------------------------

enum Morphology { M_ERODE, M_DILATE };

template<typename DST, typename SRC>
static void run_morphology(          Buffer&    dst,
                           const     View  &    src,
                           const     uchar      k[],
                                     int        k_rows,
                                     int        k_cols,
                           const cv::Point & /* anchor */,
                                     Morphology morphology)
{
    static const int maxLines = 9;
    GAPI_Assert(k_rows <= maxLines);

    const SRC *in[ maxLines ];
          DST *out;

    int border_x = (k_cols - 1) / 2;
    int border_y = (k_rows - 1) / 2;

    for (int i=0; i < k_rows; i++)
    {
        in[i] = src.InLine<SRC>(i - border_y);
    }

    out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    for (int w=0; w < width; w++)
    {
        // TODO: make this cycle innermost
        for (int c=0; c < chan; c++)
        {
            SRC result=0;
            if (M_ERODE == morphology)
            {
                result = std::numeric_limits<SRC>::max();
            }
            else if (M_DILATE == morphology)
            {
                result = std::numeric_limits<SRC>::min();
            }
            else
                CV_Error(cv::Error::StsBadArg, "unsupported morphology operation");

            for (int i=0; i < k_rows; i++)
            for (int j=0; j < k_cols; j++)
            {
                if ( k[k_cols*i + j] )
                {
                    if (M_ERODE == morphology)
                    {
                        result = std::min(result, in[i][(w + j - border_x)*chan + c]);
                    }
                    else if (M_DILATE == morphology)
                    {
                        result = std::max(result, in[i][(w + j - border_x)*chan + c]);
                    }
                    else
                        CV_Error(cv::Error::StsBadArg, "unsupported morphology operation");
                }
            }

            out[w*chan + c] = saturate<DST>(result, rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidErode, cv::gapi::imgproc::GErode, true)
{
    static const int Window = 3;

    static void run(const     View  &    src,
                    const cv::Mat   &    kernel,
                    const cv::Point &    anchor,
                              int        iterations,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        // TODO: support non-trivial anchors
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        // TODO: support kernel heights 3, 5, 7, 9, ...
        GAPI_Assert(kernel.rows == 3 && kernel.cols == 3);

        // TODO: support iterations > 1
        GAPI_Assert(iterations == 1);

        int k_rows = kernel.rows;
        int k_cols = kernel.cols;

        auto *k = scratch.OutLine<uchar>(); // copy of kernel.data

        //     DST     SRC     OP              __VA_ARGS__
        UNARY_(uchar , uchar , run_morphology, dst, src, k, k_rows, k_cols, anchor, M_ERODE);
        UNARY_(ushort, ushort, run_morphology, dst, src, k, k_rows, k_cols, anchor, M_ERODE);
        UNARY_( short,  short, run_morphology, dst, src, k, k_rows, k_cols, anchor, M_ERODE);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc& /* in */,
                            const Mat     &    kernel,
                            const Point   & /* anchor */,
                              int           /* iterations */,
                                  int       /* borderType */,
                            const cv::Scalar  & /* borderValue */,
                                  Buffer  &    scratch)
    {
        int k_rows = kernel.rows;
        int k_cols = kernel.cols;

        cv::gapi::own::Size bufsize(k_rows * k_cols, 1);
        GMatDesc bufdesc = {CV_8U, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        // FIXME: move to resetScratch stage ?
        auto *k = scratch.OutLine<uchar>();
        getKernel(k, kernel);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                            const cv::Mat   &   /* kernel */,
                            const cv::Point &   /* anchor */,
                                      int       /* iterations */,
                                      int          borderType,
                            const cv::Scalar&      borderValue)
    {
    #if 1
        // TODO: saturate borderValue to image type in general case (not only maximal border)
        GAPI_Assert(borderType == cv::BORDER_CONSTANT && borderValue[0] == DBL_MAX);
        return { borderType, cv::gapi::own::Scalar::all(INT_MAX) };
    #else
        return { borderType, borderValue };
    #endif
    }
};

GAPI_FLUID_KERNEL(GFluidDilate, cv::gapi::imgproc::GDilate, true)
{
    static const int Window = 3;

    static void run(const     View  &    src,
                    const cv::Mat   &    kernel,
                    const cv::Point &    anchor,
                              int        iterations,
                              int     /* borderType */,
                    const cv::Scalar& /* borderValue */,
                              Buffer&    dst,
                              Buffer&    scratch)
    {
        // TODO: support non-trivial anchors
        GAPI_Assert(anchor.x == -1 && anchor.y == -1);

        // TODO: support kernel heights 3, 5, 7, 9, ...
        GAPI_Assert(kernel.rows == 3 && kernel.cols == 3);

        // TODO: support iterations > 1
        GAPI_Assert(iterations == 1);

        int k_rows = kernel.rows;
        int k_cols = kernel.cols;

        auto *k = scratch.OutLine<uchar>(); // copy of kernel.data

        //     DST     SRC     OP              __VA_ARGS__
        UNARY_(uchar , uchar , run_morphology, dst, src, k, k_rows, k_cols, anchor, M_DILATE);
        UNARY_(ushort, ushort, run_morphology, dst, src, k, k_rows, k_cols, anchor, M_DILATE);
        UNARY_( short,  short, run_morphology, dst, src, k, k_rows, k_cols, anchor, M_DILATE);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc& /* in */,
                            const Mat     &    kernel,
                            const Point   & /* anchor */,
                              int           /* iterations */,
                                  int       /* borderType */,
                            const cv::Scalar  & /* borderValue */,
                                  Buffer  &    scratch)
    {
        int k_rows = kernel.rows;
        int k_cols = kernel.cols;

        cv::gapi::own::Size bufsize(k_rows * k_cols, 1);
        GMatDesc bufdesc = {CV_8U, 1, bufsize};
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);

        // FIXME: move to resetScratch stage ?
        auto *k = scratch.OutLine<uchar>();
        getKernel(k, kernel);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                            const cv::Mat   &   /* kernel */,
                            const cv::Point &   /* anchor */,
                                      int       /* iterations */,
                                      int       borderType,
                            const cv::Scalar&   borderValue)
    {
    #if 1
        // TODO: fix borderValue for Dilate in general case (not only minimal border)
        GAPI_Assert(borderType == cv::BORDER_CONSTANT && borderValue[0] == DBL_MAX);
        return { borderType, cv::gapi::own::Scalar::all(INT_MIN) };
    #else
        return { borderType, borderValue };
    #endif
    }
};

//--------------------------
//
// Fluid kernels: medianBlur
//
//--------------------------

template<typename DST, typename SRC>
static void run_medianblur(      Buffer& dst,
                           const View  & src,
                                 int     ksize)
{
    static const int kmax = 9;
    GAPI_Assert(ksize <= kmax);

    const SRC *in[ kmax ];
          DST *out;

    int border = (ksize - 1) / 2;

    for (int i=0; i < ksize; i++)
    {
        in[i] = src.InLine<SRC>(i - border);
    }

    out = dst.OutLine<DST>(0);

    int width = dst.length();
    int chan  = dst.meta().chan;

    for (int w=0; w < width; w++)
    {
        // TODO: make this cycle innermost
        for (int c=0; c < chan; c++)
        {
            SRC neighbours[kmax * kmax];

            for (int i=0; i < ksize; i++)
            for (int j=0; j < ksize; j++)
            {
                neighbours[i*ksize + j] = in[i][(w + j - border)*chan + c];
            }

            int length = ksize * ksize;
            std::nth_element(neighbours, neighbours + length/2, neighbours + length);

            out[w*chan + c] = saturate<DST>(neighbours[length/2], rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidMedianBlur, cv::gapi::imgproc::GMedianBlur, false)
{
    static const int Window = 3;

    static void run(const View  & src,
                          int     ksize,
                          Buffer& dst)
    {
        // TODO: support kernel sizes: 3, 5, 7, 9, ...
        GAPI_Assert(ksize == 3);

        //     DST     SRC     OP              __VA_ARGS__
        UNARY_(uchar , uchar , run_medianblur, dst, src, ksize);
        UNARY_(ushort, ushort, run_medianblur, dst, src, ksize);
        UNARY_( short,  short, run_medianblur, dst, src, ksize);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static Border getBorder(const cv::GMatDesc& /* src */,
                                      int       /* ksize */)
    {
        int  borderType  = cv::BORDER_REPLICATE;
        auto borderValue = cv::Scalar();
        return { borderType, borderValue };
    }
};

} // namespace fliud
} // namespace gapi
} // namespace cv

cv::gapi::GKernelPackage cv::gapi::imgproc::fluid::kernels()
{
    using namespace cv::gapi::fluid;

    return cv::gapi::kernels
    <   GFluidBGR2Gray
      , GFluidRGB2Gray
      , GFluidRGB2GrayCustom
      , GFluidRGB2YUV
      , GFluidYUV2RGB
      , GFluidRGB2Lab
      , GFluidBGR2LUV
      , GFluidBlur
      , GFluidSepFilter
      , GFluidBoxFilter
      , GFluidFilter2D
      , GFluidErode
      , GFluidDilate
      , GFluidMedianBlur
      , GFluidGaussBlur
      , GFluidSobel
    #if 0
      , GFluidCanny        -- not fluid (?)
      , GFluidEqualizeHist -- not fluid
    #endif
    >();
}

#endif // !defined(GAPI_STANDALONE)
