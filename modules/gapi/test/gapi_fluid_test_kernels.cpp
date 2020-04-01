// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"

#include <iomanip>
#include <vector>
#include "gapi_fluid_test_kernels.hpp"
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/own/saturate.hpp>

namespace cv
{
namespace gapi_test_kernels
{

GAPI_FLUID_KERNEL(FAddSimple, TAddSimple, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View   &a,
                    const cv::gapi::fluid::View   &b,
                          cv::gapi::fluid::Buffer &o)
    {
        // std::cout << "AddSimple {{{\n";
        // std::cout << "  a - "; a.debug(std::cout);
        // std::cout << "  b - "; b.debug(std::cout);
        // std::cout << "  o - "; o.debug(std::cout);

        const uint8_t* in1 = a.InLine<uint8_t>(0);
        const uint8_t* in2 = b.InLine<uint8_t>(0);
              uint8_t* out = o.OutLine<uint8_t>();

        // std::cout << "a: ";
        // for (int i = 0, w = a.length(); i < w; i++)
        // {
        //     std::cout << std::setw(4) << int(in1[i]);
        // }
        // std::cout << "\n";

        // std::cout << "b: ";
        // for (int i = 0, w = a.length(); i < w; i++)
        // {
        //     std::cout << std::setw(4) << int(in2[i]);
        // }
        // std::cout << "\n";

        for (int i = 0, w = a.length(); i < w; i++)
        {
            out[i] = in1[i] + in2[i];
        }

        // std::cout << "}}} " << std::endl;;
    }
};

GAPI_FLUID_KERNEL(FAddCSimple, TAddCSimple, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &in,
                    const int                      cval,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            //std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                //std::cout << std::setw(4) << int(in_row[i]);
                //FIXME: it seems that over kernels might need it as well
                out_row[i] = cv::gapi::own::saturate<uint8_t>(in_row[i] + cval);
            }
            //std::cout << std::endl;
        }
    }
};

GAPI_FLUID_KERNEL(FAddScalar, TAddScalar, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &in,
                    const cv::Scalar              &cval,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row[i]);
                out_row[i] = static_cast<uint8_t>(in_row[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

GAPI_FLUID_KERNEL(FAddScalarToMat, TAddScalarToMat, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::Scalar              &cval,
                    const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row[i]);
                out_row[i] = static_cast<uint8_t>(in_row[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

template<int kernelSize, int lpi = 1>
static void runBlur(const cv::gapi::fluid::View& src, cv::gapi::fluid::Buffer& dst)
{
    const auto borderSize = (kernelSize - 1) / 2;
    const unsigned char* ins[kernelSize];

    for (int l = 0; l < lpi; l++)
    {
        for (int i = 0; i < kernelSize; i++)
        {
            ins[i] = src.InLine<unsigned char>(i - borderSize + l);
        }

        auto out = dst.OutLine<unsigned char>(l);
        const auto width = dst.length();

        for (int w = 0; w < width; w++)
        {
            float res = 0.0f;
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = -borderSize; j < borderSize + 1; j++)
                {
                    res += ins[i][w+j];
                }
            }
            out[w] = static_cast<unsigned char>(std::rint(res / (kernelSize * kernelSize)));
        }
    }
}

GAPI_FLUID_KERNEL(FBlur1x1, TBlur1x1, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }
};

GAPI_FLUID_KERNEL(FBlur3x3, TBlur3x3, false)
{
    static const int Window = 3;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, borderValue};
    }
};

GAPI_FLUID_KERNEL(FBlur5x5, TBlur5x5, false)
{
    static const int Window = 5;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, borderValue};
    }
};

GAPI_FLUID_KERNEL(FBlur3x3_2lpi, TBlur3x3_2lpi, false)
{
    static const int Window = 3;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window, LPI>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, borderValue};
    }
};

GAPI_FLUID_KERNEL(FBlur5x5_2lpi, TBlur5x5_2lpi, false)
{
    static const int Window = 5;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window, LPI>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, borderValue};
    }
};

GAPI_FLUID_KERNEL(FIdentity, TId, false)
{
    static const int Window = 3;

    static void run(const cv::gapi::fluid::View   &a,
                          cv::gapi::fluid::Buffer &o)
    {
        const uint8_t* in[3] = {
            a.InLine<uint8_t>(-1),
            a.InLine<uint8_t>( 0),
            a.InLine<uint8_t>(+1)
        };
        uint8_t* out = o.OutLine<uint8_t>();

        // ReadFunction3x3(in, a.length());
        for (int i = 0, w = a.length(); i < w; i++)
        {
            out[i] = in[1][i];
        }
    }

    static gapi::fluid::Border getBorder(const cv::GMatDesc &)
    {
        return { cv::BORDER_REPLICATE, cv::Scalar{} };
    }
};

GAPI_FLUID_KERNEL(FId7x7, TId7x7, false)
{
    static const int Window = 7;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &a,
                          cv::gapi::fluid::Buffer &o)
    {
        for (int l = 0, lpi = o.lpi(); l < lpi; l++)
        {
            const uint8_t* in[Window] = {
                a.InLine<uint8_t>(-3 + l),
                a.InLine<uint8_t>(-2 + l),
                a.InLine<uint8_t>(-1 + l),
                a.InLine<uint8_t>( 0 + l),
                a.InLine<uint8_t>(+1 + l),
                a.InLine<uint8_t>(+2 + l),
                a.InLine<uint8_t>(+3 + l),
            };
            uint8_t* out = o.OutLine<uint8_t>(l);

            // std::cout << "Id7x7 " << l << " of " << lpi << " {{{\n";
            // std::cout << "  a - "; a.debug(std::cout);
            // std::cout << "  o - "; o.debug(std::cout);
            // std::cout << "}}} " << std::endl;;

            // // std::cout << "Id7x7 at " << a.y() << "/L" << l <<  " {{{" << std::endl;
            // for (int j = 0; j < Window; j++)
            // {
            //     // std::cout << std::setw(2) << j-(Window-1)/2 << ": ";
            //     for (int i = 0, w = a.length(); i < w; i++)
            //         std::cout << std::setw(4) << int(in[j][i]);
            //     std::cout << std::endl;
            // }
            // std::cout << "}}}" << std::endl;

            for (int i = 0, w = a.length(); i < w; i++)
                out[i] = in[(Window-1)/2][i];
        }
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc&/* src*/)
    {
        return { cv::BORDER_REPLICATE, cv::Scalar{} };
    }
};

GAPI_FLUID_KERNEL(FPlusRow0, TPlusRow0, true)
{
    static const int Window = 1;

    static void initScratch(const cv::GMatDesc            &in,
                                  cv::gapi::fluid::Buffer &scratch)
    {
        cv::Size scratch_size{in.size.width, 1};
        cv::gapi::fluid::Buffer buffer(in.withSize(scratch_size));
        scratch = std::move(buffer);
    }

    static void resetScratch(cv::gapi::fluid::Buffer &scratch)
    {
        // FIXME: only 1 line can be used!
        uint8_t* out_row = scratch.OutLine<uint8_t>();
        for (int i = 0, w = scratch.length(); i < w; i++)
        {
            out_row[i] = 0;
        }
    }

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &out,
                          cv::gapi::fluid::Buffer &scratch)
    {
        const uint8_t* in_row  = in     .InLine <uint8_t>(0);
              uint8_t* out_row = out    .OutLine<uint8_t>();
              uint8_t* tmp_row = scratch.OutLine<uint8_t>();

        if (in.y() == 0)
        {
            // Copy 1st row to scratch buffer
            for (int i = 0, w = in.length(); i < w; i++)
            {
                out_row[i] = in_row[i];
                tmp_row[i] = in_row[i];
            }
        }
        else
        {
            // Output is 1st row + in
            for (int i = 0, w = in.length(); i < w; i++)
            {
                out_row[i] = in_row[i] + tmp_row[i];
            }
        }
    }
};

static void split3Row(const cv::gapi::fluid::View   &in,
                      cv::gapi::fluid::Buffer &o1,
                      cv::gapi::fluid::Buffer &o2,
                      cv::gapi::fluid::Buffer &o3)
{
    for (int l = 0; l < o1.lpi(); l++)
    {
        // std::cout << "Split3  {{{\n";
        // std::cout << "  a - "; in.debug(std::cout);
        // std::cout << "  1 - "; o1.debug(std::cout);
        // std::cout << "  2 - "; o2.debug(std::cout);
        // std::cout << "  3 - "; o3.debug(std::cout);
        // std::cout << "}}} " << std::endl;;

        const uint8_t* in_rgb = in.InLine<uint8_t>(l);
              uint8_t* out_r  = o1.OutLine<uint8_t>(l);
              uint8_t* out_g  = o2.OutLine<uint8_t>(l);
              uint8_t* out_b  = o3.OutLine<uint8_t>(l);

        for (int i = 0, w = in.length(); i < w; i++)
        {
            out_r[i] = in_rgb[3*i];
            out_g[i] = in_rgb[3*i+1];
            out_b[i] = in_rgb[3*i+2];
        }
    }
}

GAPI_FLUID_KERNEL(FTestSplit3, cv::gapi::core::GSplit3, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &o1,
                          cv::gapi::fluid::Buffer &o2,
                          cv::gapi::fluid::Buffer &o3)
    {
        split3Row(in, o1, o2, o3);
    }
};

GAPI_FLUID_KERNEL(FTestSplit3_4lpi, TSplit3_4lpi, false)
{
    static const int Window = 1;
    static const int LPI = 4;

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &o1,
                          cv::gapi::fluid::Buffer &o2,
                          cv::gapi::fluid::Buffer &o3)
    {
        split3Row(in, o1, o2, o3);
    }
};

std::tuple<GMat, GMat, GMat> split3_4lpi(const GMat& src)
{
    return TSplit3_4lpi::on(src);
}

GAPI_FLUID_KERNEL(FSum2MatsAndScalar, TSum2MatsAndScalar, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &a,
                    const cv::Scalar              &cval,
                    const cv::gapi::fluid::View   &b,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row1  = a .InLine <uint8_t>(l);
            const uint8_t* in_row2  = b .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = a.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row1[i]);
                std::cout << std::setw(4) << int(in_row2[i]);
                out_row[i] = static_cast<uint8_t>(in_row1[i] + in_row2[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

GAPI_FLUID_KERNEL(FEqualizeHist, TEqualizeHist, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &mat,
                    const std::vector<int>        &arr,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = mat.InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);

            for (int i = 0, w = mat.length(); i < w; i++)
            {
                out_row[i] = static_cast<uint8_t>(arr[in_row[i]]);
            }
        }
    }
};

GAPI_OCV_KERNEL(OCVCalcHist, TCalcHist)
{
    static void run(const cv::Mat& in, std::vector<int>& out)
    {
        out = std::vector<int>(256, 0);

        // Calculate normalized accumulated integral transformation array for gapi
        for(int i = 0; i < in.rows; ++i)
            for(int j = 0; j < in.cols; ++j)
                ++out[in.at<uint8_t>(i, j)];

        for(unsigned int i = 1; i < out.size(); ++i)
            out[i] += out[i-1];

        int size = in.size().width * in.size().height;
        int min = size;
        for(unsigned int i = 0; i < out.size(); ++i)
            if(out[i] != 0 && out[i] < min)
                min = out[i];

        for(auto & el : out)
        {
            // General histogram equalization formula
            el = cvRound(((float)(el - min) / (float)(size - min))*255);
        }
    }
};

static const int ITUR_BT_601_CY = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;
static const int ITUR_BT_601_SHIFT = 20;

static inline void uvToRGBuv(const uchar u, const uchar v, int& ruv, int& guv, int& buv)
{
    int uu, vv;
    uu = int(u) - 128;
    vv = int(v) - 128;

    ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * vv;
    guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu;
    buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * uu;
}

static inline void yRGBuvToRGB(const uchar vy, const int ruv, const int guv, const int buv,
                                uchar& r, uchar& g, uchar& b)
{
    int y = std::max(0, vy - 16) * ITUR_BT_601_CY;
    r = saturate_cast<uchar>((y + ruv) >> ITUR_BT_601_SHIFT);
    g = saturate_cast<uchar>((y + guv) >> ITUR_BT_601_SHIFT);
    b = saturate_cast<uchar>((y + buv) >> ITUR_BT_601_SHIFT);
}

GAPI_FLUID_KERNEL(FNV12toRGB, cv::gapi::imgproc::GNV12toRGB, false)
{
    static const int Window = 1;
    static const int LPI    = 2;
    static const auto Kind = GFluidKernel::Kind::YUV420toRGB;

    static void run(const cv::gapi::fluid::View   &in1,
                    const cv::gapi::fluid::View   &in2,
                          cv::gapi::fluid::Buffer &out)
    {
        const auto w = out.length();

        GAPI_Assert(w % 2 == 0);
        GAPI_Assert(out.lpi() == 2);

        const uchar* uv_row = in2.InLineB(0);
        const uchar*   y_rows[] = {in1. InLineB(0), in1. InLineB(1)};
              uchar* out_rows[] = {out.OutLineB(0), out.OutLineB(1)};

        for (int i = 0; i < w/2; i++)
        {
            uchar u = uv_row[2*i];
            uchar v = uv_row[2*i + 1];
            int ruv, guv, buv;
            uvToRGBuv(u, v, ruv, guv, buv);

            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++)
                {
                    uchar vy = y_rows[y][2*i + x];
                    uchar r, g, b;
                    yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                    out_rows[y][3*(2*i + x)]     = r;
                    out_rows[y][3*(2*i + x) + 1] = g;
                    out_rows[y][3*(2*i + x) + 2] = b;
                }
            }
        }
    }
};


GAPI_FLUID_KERNEL(FMerge3_4lpi, TMerge3_4lpi, false)
{
    static const int Window = 1;
    static const int LPI = 4;

    static void run(const cv::gapi::fluid::View &src1,
                    const cv::gapi::fluid::View &src2,
                    const cv::gapi::fluid::View &src3,
                          cv::gapi::fluid::Buffer &dst)
    {
        for (int l = 0; l < dst.lpi(); l++)
        {
            const auto *in1 = src1.InLine<uchar>(l);
            const auto *in2 = src2.InLine<uchar>(l);
            const auto *in3 = src3.InLine<uchar>(l);
            auto *out = dst.OutLine<uchar>(l);

            for (int w = 0; w < dst.length(); w++)
            {
                out[3*w    ] = in1[w];
                out[3*w + 1] = in2[w];
                out[3*w + 2] = in3[w];
            }
        }
    }
};

GMat merge3_4lpi(const GMat& src1, const GMat& src2, const GMat& src3)
{
    return TMerge3_4lpi::on(src1, src2, src3);
}

cv::gapi::GKernelPackage fluidTestPackage = cv::gapi::kernels
        <FAddSimple
        ,FAddCSimple
        ,FAddScalar
        ,FAddScalarToMat
        ,FBlur1x1
        ,FBlur3x3
        ,FBlur5x5
        ,FBlur3x3_2lpi
        ,FBlur5x5_2lpi
        ,FIdentity
        ,FId7x7
        ,FMerge3_4lpi
        ,FNV12toRGB
        ,FPlusRow0
        ,FSum2MatsAndScalar
        ,FTestSplit3
        ,FTestSplit3_4lpi
        ,FEqualizeHist
        ,OCVCalcHist
        >();
} // namespace gapi_test_kernels
} // namespace cv
