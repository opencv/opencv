// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "gapi_fluid_test_kernels.hpp"

namespace opencv_test
{

using namespace cv::gapi_test_kernels;

G_TYPED_KERNEL(TCopy, <GMat(GMat)>, "test.fluid.copy")
{
    static GMatDesc outMeta(const cv::GMatDesc &in) {
        return in;
    }
};

GAPI_FLUID_KERNEL(FCopy, TCopy, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &out)
    {
        const uint8_t* in_row  = in .InLine <uint8_t>(0);
        uint8_t* out_row = out.OutLine<uint8_t>();

        for (int i = 0, w = in.length(); i < w; i++)
        {
            //std::cout << std::setw(4) << int(in_row[i]);
            out_row[i] = in_row[i];
        }
        //std::cout << std::endl;
    }
};

GAPI_FLUID_KERNEL(FResizeNN1Lpi, cv::gapi::core::GResize, false)
{
    static const int Window = 1;
    static const auto Kind = GFluidKernel::Kind::Resize;

    static void run(const cv::gapi::fluid::View& in, cv::Size /*sz*/, double /*fx*/, double /*fy*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out)

    {
        auto length = out.length();
        double vRatio = (double)in.meta().size.height / out.meta().size.height;
        double hRatio = (double)in.length() / length;
        auto y = out.y();
        auto inY = in.y();

        for (int l = 0; l < out.lpi(); l++)
        {
            auto sy = static_cast<int>((y+l) * vRatio);
            int idx = sy - inY;

            const auto src = in.InLine <unsigned char>(idx);
            auto dst = out.OutLine<unsigned char>(l);

            for (int x = 0; x < length; x++)
            {
                auto inX = static_cast<int>(x * hRatio);
                dst[x] = src[inX];
            }
        }
    }
};

namespace
{
namespace func
{
template <class Mapper>
void initScratch(const cv::GMatDesc& in, cv::Size outSz, cv::gapi::fluid::Buffer &scratch)
{
    CV_Assert(in.depth == CV_8U && in.chan == 1);

    cv::Size scratch_size{static_cast<int>(outSz.width * sizeof(typename Mapper::Unit)), 1};

    cv::GMatDesc desc;
    desc.chan  = 1;
    desc.depth = CV_8UC1;
    desc.size  = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    auto mapX = scratch.OutLine<typename Mapper::Unit>();
    double hRatio = (double)in.size.width / outSz.width;

    for (int x = 0, w = outSz.width; x < w; x++)
    {
        mapX[x] = Mapper::map(hRatio, 0, in.size.width, x);
    }
}

template <class Mapper>
inline void calcRow(const cv::gapi::fluid::View& in, cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch)
{
    double vRatio = (double)in.meta().size.height / out.meta().size.height;
    auto mapX = scratch.OutLine<typename Mapper::Unit>();
    auto inY = in.y();
    auto inH = in.meta().size.height;
    auto outY = out.y();
    auto length = out.length();

    for (int l = 0; l < out.lpi(); l++)
    {
        auto mapY = Mapper::map(vRatio, inY, inH, outY + l);

        const auto src0 = in.InLine <unsigned char>(mapY.s0);
        const auto src1 = in.InLine <unsigned char>(mapY.s1);

        auto dst = out.OutLine<unsigned char>(l);

        for (int x = 0; x < length; x++)
        {
            auto alpha0 = mapX[x].alpha0;
            auto alpha1 = mapX[x].alpha1;
            auto sx0 = mapX[x].s0;
            auto sx1 = mapX[x].s1;

            int res0 = src0[sx0]*alpha0 + src0[sx1]*alpha1;
            int res1 = src1[sx0]*alpha0 + src1[sx1]*alpha1;

            dst[x] = uchar(( ((mapY.alpha0 * (res0 >> 4)) >> 16) + ((mapY.alpha1 * (res1 >> 4)) >> 16) + 2)>>2);
        }
    }
}
} // namespace func

constexpr static const int INTER_RESIZE_COEF_BITS = 11;
constexpr static const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

namespace linear
{
struct Mapper
{
    struct Unit
    {
        short alpha0;
        short alpha1;
        int   s0;
        int   s1;
    };

    static inline Unit map(double ratio, int start, int max, int outCoord)
    {
        auto f = static_cast<float>((outCoord + 0.5f) * ratio - 0.5f);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.s0 = std::max(s - start, 0);
        u.s1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>((1.0f - f) * INTER_RESIZE_COEF_SCALE);
        u.alpha1 = saturate_cast<short>((f) * INTER_RESIZE_COEF_SCALE);

        return u;
    }
};

} // namespace linear

namespace areaUpscale
{
struct Mapper
{
    struct Unit
    {
        short alpha0;
        short alpha1;
        int   s0;
        int   s1;
    };

    static inline Unit map(double ratio, int start, int max, int outCoord)
    {
        int s = cvFloor(outCoord*ratio);
        float f = (float)((outCoord+1) - (s+1)/ratio);
        f = f <= 0 ? 0.f : f - cvFloor(f);

        Unit u;

        u.s0 = std::max(s - start, 0);
        u.s1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>((1.0f - f) * INTER_RESIZE_COEF_SCALE);
        u.alpha1 = saturate_cast<short>((f) * INTER_RESIZE_COEF_SCALE);

        return u;
    }
};
} // namespace areaUpscale
} // anonymous namespace

GAPI_FLUID_KERNEL(FResizeLinear1Lpi, cv::gapi::core::GResize, true)
{
    static const int Window = 1;
    static const auto Kind = GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            cv::Size outSz, double /*fx*/, double /*fy*/, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch)
    {
        func::initScratch<linear::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/)
    {}

    static void run(const cv::gapi::fluid::View& in, cv::Size /*sz*/, double /*fx*/, double /*fy*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch)

    {
        func::calcRow<linear::Mapper>(in, out, scratch);
    }
};

namespace
{
// FIXME
// Move to some common place (to reuse/align with ResizeAgent)
auto startInCoord = [](int outCoord, double ratio) {
    return static_cast<int>(outCoord * ratio + 1e-3);
};
auto endInCoord = [](int outCoord, double ratio) {
    return static_cast<int>(std::ceil((outCoord + 1) * ratio - 1e-3));
};
} // namespace

GAPI_FLUID_KERNEL(FResizeArea1Lpi, cv::gapi::core::GResize, false)
{
    static const int Window = 1;
    static const auto Kind = GFluidKernel::Kind::Resize;

    static void run(const cv::gapi::fluid::View& in, cv::Size /*sz*/, double /*fx*/, double /*fy*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out)

    {
        auto firstOutLineIdx = out.y();
        auto firstViewLineIdx = in.y();
        auto length = out.length();
        double vRatio = (double)in.meta().size.height / out.meta().size.height;
        double hRatio = (double)in.length() / length;

        for (int l = 0; l < out.lpi(); l++)
        {
            int outY = firstOutLineIdx + l;
            int startY = startInCoord(outY, vRatio);
            int endY   = endInCoord  (outY, vRatio);

            auto dst = out.OutLine<unsigned char>(l);

            for (int x = 0; x < length; x++)
            {
                float res = 0.0;

                int startX = startInCoord(x, hRatio);
                int endX   = endInCoord  (x, hRatio);

                for (int inY = startY; inY < endY; inY++)
                {
                    double startCoordY = inY / vRatio;
                    double endCoordY = startCoordY + 1/vRatio;

                    if (startCoordY < outY) startCoordY = outY;
                    if (endCoordY > outY + 1) endCoordY = outY + 1;

                    float fracY = static_cast<float>((inY == startY || inY == endY - 1) ? endCoordY - startCoordY : 1/vRatio);

                    const auto src = in.InLine <unsigned char>(inY - firstViewLineIdx);

                    float rowSum = 0.0f;

                    for (int inX = startX; inX < endX; inX++)
                    {
                        double startCoordX = inX / hRatio;
                        double endCoordX = startCoordX + 1/hRatio;

                        if (startCoordX < x) startCoordX = x;
                        if (endCoordX > x + 1) endCoordX = x + 1;

                        float fracX = static_cast<float>((inX == startX || inX == endX - 1) ? endCoordX - startCoordX : 1/hRatio);

                        rowSum += src[inX] * fracX;
                    }
                    res += rowSum * fracY;
                }
                dst[x] = static_cast<unsigned char>(std::rint(res));
            }
        }
    }
};

GAPI_FLUID_KERNEL(FResizeAreaUpscale1Lpi, cv::gapi::core::GResize, true)
{
    static const int Window = 1;
    static const auto Kind = GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            cv::Size outSz, double /*fx*/, double /*fy*/, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch)
    {
        func::initScratch<areaUpscale::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/)
    {}

    static void run(const cv::gapi::fluid::View& in, cv::Size /*sz*/, double /*fx*/, double /*fy*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch)
    {
        func::calcRow<areaUpscale::Mapper>(in, out, scratch);
    }
};

#define ADD_RESIZE_KERNEL_WITH_LPI(interp, lpi, scratch)                                                                           \
struct Resize##interp##lpi##LpiHelper : public FResize##interp##1Lpi { static const int LPI = lpi; };                              \
struct FResize##interp##lpi##Lpi : public cv::GFluidKernelImpl<Resize##interp##lpi##LpiHelper, cv::gapi::core::GResize, scratch>{};

ADD_RESIZE_KERNEL_WITH_LPI(NN, 2, false)
ADD_RESIZE_KERNEL_WITH_LPI(NN, 3, false)
ADD_RESIZE_KERNEL_WITH_LPI(NN, 4, false)

ADD_RESIZE_KERNEL_WITH_LPI(Linear, 2, true)
ADD_RESIZE_KERNEL_WITH_LPI(Linear, 3, true)
ADD_RESIZE_KERNEL_WITH_LPI(Linear, 4, true)

ADD_RESIZE_KERNEL_WITH_LPI(Area, 2, false)
ADD_RESIZE_KERNEL_WITH_LPI(Area, 3, false)
ADD_RESIZE_KERNEL_WITH_LPI(Area, 4, false)

ADD_RESIZE_KERNEL_WITH_LPI(AreaUpscale, 2, true)
ADD_RESIZE_KERNEL_WITH_LPI(AreaUpscale, 3, true)
ADD_RESIZE_KERNEL_WITH_LPI(AreaUpscale, 4, true)
#undef ADD_RESIZE_KERNEL_WITH_LPI

static auto fluidResizeTestPackage = [](int interpolation, cv::Size szIn, cv::Size szOut, int lpi = 1)
{
    using namespace cv;
    using namespace cv::gapi;
    bool upscale = szIn.width < szOut.width || szIn.height < szOut.height;

#define RESIZE_CASE(interp, lpi) \
    case lpi: pkg = kernels<FCopy, FResize##interp##lpi##Lpi>(); break;

#define RESIZE_SWITCH(interp)   \
    switch(lpi)                 \
    {                           \
    RESIZE_CASE(interp, 1)      \
    RESIZE_CASE(interp, 2)      \
    RESIZE_CASE(interp, 3)      \
    RESIZE_CASE(interp, 4)      \
    default: CV_Assert(false);  \
    }

    GKernelPackage pkg;
    switch (interpolation)
    {
    case INTER_NEAREST: RESIZE_SWITCH(NN); break;
    case INTER_LINEAR:  RESIZE_SWITCH(Linear); break;
    case INTER_AREA:
    {
        if (upscale)
        {
            RESIZE_SWITCH(AreaUpscale)
        }
        else
        {
            RESIZE_SWITCH(Area);
        }
    }break;
    default: CV_Assert(false);
    }
    return combine(pkg, fluidTestPackage, unite_policy::KEEP);

#undef RESIZE_SWITCH
#undef RESIZE_CASE
};

struct ResizeTestFluid : public TestWithParam<std::tuple<int, int, cv::Size, std::tuple<cv::Size, cv::Rect>, int, double>> {};
TEST_P(ResizeTestFluid, SanityTest)
{
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    int lpi = 0;
    double tolerance = 0.0;
    cv::Rect outRoi;
    std::tuple<cv::Size, cv::Rect> outSizeAndRoi;
    std::tie(type, interp, sz_in, outSizeAndRoi, lpi, tolerance) = GetParam();
    std::tie(sz_out, outRoi) = outSizeAndRoi;
    if (outRoi == cv::Rect{}) outRoi = {0,0,sz_out.width,sz_out.height};
    if (outRoi.width == 0) outRoi.width = sz_out.width;
    double fx = 0, fy = 0;

    cv::Mat in_mat1 (sz_in, type );
    cv::Scalar mean = cv::Scalar(127);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Mat out_mat = cv::Mat::zeros(sz_out, type);
    cv::Mat out_mat_ocv = cv::Mat::zeros(sz_out, type);

    cv::GMat in;
    auto mid = TBlur3x3::on(in, cv::BORDER_REPLICATE, {});
    auto out = cv::gapi::resize(mid, sz_out, fx, fy, interp);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat, cv::compile_args(GFluidOutputRois{{outRoi}}, fluidResizeTestPackage(interp, sz_in, sz_out, lpi)));

    cv::Mat mid_mat;
    cv::blur(in_mat1, mid_mat, {3,3}, {-1,-1},  cv::BORDER_REPLICATE);
    cv::resize(mid_mat, out_mat_ocv, sz_out, fx, fy, interp);

    cv::Mat absDiff;
    cv::absdiff(out_mat(outRoi), out_mat_ocv(outRoi), absDiff);
    EXPECT_EQ(0, cv::countNonZero(absDiff > tolerance));
}

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR),
                                Values(cv::Size(8, 7),
                                       cv::Size(8, 8),
                                       cv::Size(8, 64),
                                       cv::Size(8, 25),
                                       cv::Size(16, 8),
                                       cv::Size(16, 7)),
                                Values(std::make_tuple(cv::Size(5, 4), cv::Rect{}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 0, 0, 2}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 1, 0, 2}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 2, 0, 2}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 0, 0, 3}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 2, 0, 2}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 4, 0, 3}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 0, 0, 3}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 1, 0, 2}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 3, 0, 1})),
                                Values(1, 2, 3, 4), // lpi
                                Values(0.0)));

INSTANTIATE_TEST_CASE_P(ResizeAreaTestCPU, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_AREA),
                                Values(cv::Size(8, 7),
                                       cv::Size(8, 8),
                                       cv::Size(8, 64),
                                       cv::Size(8, 25),
                                       cv::Size(16, 8),
                                       cv::Size(16, 7)),
                                Values(std::make_tuple(cv::Size(5, 4), cv::Rect{}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 0, 0, 2}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 1, 0, 2}),
                                       std::make_tuple(cv::Size(5, 4), cv::Rect{0, 2, 0, 2}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 0, 0, 3}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 2, 0, 2}),
                                       std::make_tuple(cv::Size(7, 7), cv::Rect{0, 4, 0, 3}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 0, 0, 3}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 1, 0, 2}),
                                       std::make_tuple(cv::Size(8, 4), cv::Rect{0, 3, 0, 1})),
                                Values(1, 2, 3, 4), // lpi
                                // Actually this tolerance only for cases where OpenCV
                                // uses ResizeAreaFast
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(ResizeUpscaleTestCPU, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(1, 5),
                                       cv::Size(3, 5),
                                       cv::Size(7, 5),
                                       cv::Size(1, 7),
                                       cv::Size(3, 7),
                                       cv::Size(7, 7)),
                                Values(std::make_tuple(cv::Size(8, 8), cv::Rect{0,0,8,2}),
                                       std::make_tuple(cv::Size(8, 8), cv::Rect{0,2,8,2}),
                                       std::make_tuple(cv::Size(8, 8), cv::Rect{0,4,8,2}),
                                       std::make_tuple(cv::Size(8, 8), cv::Rect{0,6,8,2}),
                                       std::make_tuple(cv::Size(8, 8), cv::Rect{0,0,8,8}),
                                       std::make_tuple(cv::Size(16, 8), cv::Rect{}),
                                       std::make_tuple(cv::Size(16, 64), cv::Rect{0, 0,16,16}),
                                       std::make_tuple(cv::Size(16, 64), cv::Rect{0,16,16,16}),
                                       std::make_tuple(cv::Size(16, 64), cv::Rect{0,32,16,16}),
                                       std::make_tuple(cv::Size(16, 64), cv::Rect{0,48,16,16}),
                                       std::make_tuple(cv::Size(16, 64), cv::Rect{0, 0,16,64}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0, 0,16, 7}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0, 7,16, 6}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0,13,16, 6}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0,19,16, 6}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0, 0,16, 7}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0, 7,16, 7}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0,14,16, 7}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0,21,16, 4}),
                                       std::make_tuple(cv::Size(16, 25), cv::Rect{0, 0,16,25}),
                                       std::make_tuple(cv::Size(16, 7), cv::Rect{}),
                                       std::make_tuple(cv::Size(16, 8), cv::Rect{})),
                                Values(1, 2, 3, 4), // lpi
                                Values(0.0)));

INSTANTIATE_TEST_CASE_P(ResizeUpscaleOneDimDownscaleAnother, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(6, 6),
                                       cv::Size(8, 7),
                                       cv::Size(8, 8),
                                       cv::Size(8, 10),
                                       cv::Size(10, 8),
                                       cv::Size(10, 7)),
                                Values(std::make_tuple(cv::Size(11, 5), cv::Rect{}),
                                       std::make_tuple(cv::Size(11, 5), cv::Rect{0, 0, 0, 2}),
                                       std::make_tuple(cv::Size(11, 5), cv::Rect{0, 2, 0, 2}),
                                       std::make_tuple(cv::Size(11, 5), cv::Rect{0, 4, 0, 1}),
                                       std::make_tuple(cv::Size(12, 2), cv::Rect{}),
                                       std::make_tuple(cv::Size(12, 2), cv::Rect{0, 0, 0, 1}),
                                       std::make_tuple(cv::Size(12, 2), cv::Rect{0, 1, 0, 1}),
                                       std::make_tuple(cv::Size(23, 3), cv::Rect{}),
                                       std::make_tuple(cv::Size(23, 3), cv::Rect{0, 0, 0, 1}),
                                       std::make_tuple(cv::Size(23, 3), cv::Rect{0, 1, 0, 1}),
                                       std::make_tuple(cv::Size(23, 3), cv::Rect{0, 2, 0, 1}),
                                       std::make_tuple(cv::Size(3, 24), cv::Rect{}),
                                       std::make_tuple(cv::Size(3, 24), cv::Rect{0,  0, 0, 6}),
                                       std::make_tuple(cv::Size(3, 24), cv::Rect{0,  6, 0, 6}),
                                       std::make_tuple(cv::Size(3, 24), cv::Rect{0, 12, 0, 6}),
                                       std::make_tuple(cv::Size(3, 24), cv::Rect{0, 18, 0, 6}),
                                       std::make_tuple(cv::Size(5, 11), cv::Rect{}),
                                       std::make_tuple(cv::Size(5, 11), cv::Rect{0, 0, 0, 3}),
                                       std::make_tuple(cv::Size(5, 11), cv::Rect{0, 3, 0, 3}),
                                       std::make_tuple(cv::Size(5, 11), cv::Rect{0, 6, 0, 3}),
                                       std::make_tuple(cv::Size(5, 11), cv::Rect{0, 9, 0, 2})),
                                Values(1, 2, 3, 4), // lpi
                                Values(0.0)));

INSTANTIATE_TEST_CASE_P(Resize400_384TestCPU, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(128, 400)),
                                Values(std::make_tuple(cv::Size(128, 384), cv::Rect{})),
                                Values(1, 2, 3, 4), // lpi
                                Values(0.0)));

INSTANTIATE_TEST_CASE_P(Resize220_400TestCPU, ResizeTestFluid,
                        Combine(Values(CV_8UC1),
                                Values(cv::INTER_LINEAR),
                                Values(cv::Size(220, 220)),
                                Values(std::make_tuple(cv::Size(400, 400), cv::Rect{})),
                                Values(1, 2, 3, 4), // lpi
                                Values(0.0)));

static auto cvBlur = [](const cv::Mat& in, cv::Mat& out, int kernelSize)
{
    if (kernelSize == 1)
    {
        out = in;
    }
    else
    {
        cv::blur(in, out, {kernelSize, kernelSize});
    }
};

using SizesWithRois = std::tuple<cv::Size, cv::Rect, cv::Size, cv::Rect>;
struct ResizeAndAnotherReaderTest : public TestWithParam<std::tuple<int, int, bool, SizesWithRois>>{};
TEST_P(ResizeAndAnotherReaderTest, SanityTest)
{
    bool readFromInput = false;
    int interp = -1, kernelSize = -1;
    SizesWithRois sizesWithRois;
    std::tie(interp, kernelSize, readFromInput, sizesWithRois) = GetParam();

    cv::Size sz,  resizedSz;
    cv::Rect roi, resizedRoi;
    std::tie(sz, roi, resizedSz, resizedRoi) = sizesWithRois;

    cv::Mat in_mat(sz, CV_8UC1);
    cv::Scalar mean = cv::Scalar(127);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);

    cv::Mat gapi_resize_out = cv::Mat::zeros(resizedSz, CV_8UC1);
    cv::Mat gapi_blur_out = cv::Mat::zeros(sz, CV_8UC1);

    auto blur = kernelSize == 1 ? &TBlur1x1::on : kernelSize == 3 ? &TBlur3x3::on : &TBlur5x5::on;

    cv::GMat in, resize_out, blur_out;

    if (readFromInput)
    {
        resize_out = gapi::resize(in, resizedSz, 0, 0, interp);
        blur_out   = blur(in, cv::BORDER_DEFAULT, {});
    }
    else
    {
        auto mid   = TCopy::on(in);
        resize_out = gapi::resize(mid, resizedSz, 0, 0, interp);
        blur_out   = blur(mid, cv::BORDER_DEFAULT, {});
    }

    cv::GComputation c(GIn(in), GOut(resize_out, blur_out));
    c.apply(gin(in_mat), gout(gapi_resize_out, gapi_blur_out), cv::compile_args(GFluidOutputRois{{resizedRoi, roi}},
                                                                                fluidResizeTestPackage(interp, sz, resizedSz)));

    cv::Mat ocv_resize_out = cv::Mat::zeros(resizedSz, CV_8UC1);
    cv::resize(in_mat, ocv_resize_out, resizedSz, 0, 0, interp);
    cv::Mat ocv_blur_out = cv::Mat::zeros(sz, CV_8UC1);
    cvBlur(in_mat, ocv_blur_out, kernelSize);

    EXPECT_EQ(0, cv::countNonZero(gapi_resize_out(resizedRoi) != ocv_resize_out(resizedRoi)));
    EXPECT_EQ(0, cv::countNonZero(gapi_blur_out(roi) != ocv_blur_out(roi)));
}

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeAndAnotherReaderTest,
                        Combine(Values(cv::INTER_NEAREST, cv::INTER_LINEAR),
                                Values(1, 3, 5),
                                testing::Bool(), // Read from input directly or place a copy node at start
                                Values(std::make_tuple(cv::Size{8,8}, cv::Rect{0,0,8,8},
                                                       cv::Size{4,4}, cv::Rect{0,0,4,4}),
                                       std::make_tuple(cv::Size{8,8}, cv::Rect{0,0,8,2},
                                                       cv::Size{4,4}, cv::Rect{0,0,4,1}),
                                       std::make_tuple(cv::Size{8,8}, cv::Rect{0,2,8,4},
                                                       cv::Size{4,4}, cv::Rect{0,1,4,2}),
                                       std::make_tuple(cv::Size{8,8}, cv::Rect{0,4,8,4},
                                                       cv::Size{4,4}, cv::Rect{0,2,4,2}),
                                       std::make_tuple(cv::Size{64,64}, cv::Rect{0, 0,64,64},
                                                       cv::Size{49,49}, cv::Rect{0, 0,49,49}),
                                       std::make_tuple(cv::Size{64,64}, cv::Rect{0, 0,64,15},
                                                       cv::Size{49,49}, cv::Rect{0, 0,49,11}),
                                       std::make_tuple(cv::Size{64,64}, cv::Rect{0,11,64,23},
                                                       cv::Size{49,49}, cv::Rect{0, 9,49,17}),
                                       std::make_tuple(cv::Size{64,64}, cv::Rect{0,50,64,14},
                                                       cv::Size{49,49}, cv::Rect{0,39,49,10}))));

struct BlursAfterResizeTest : public TestWithParam<std::tuple<int, int, int, bool, std::tuple<cv::Size, cv::Size, cv::Rect>>>{};
TEST_P(BlursAfterResizeTest, SanityTest)
{
    bool readFromInput = false;
    int interp = -1, kernelSize1 = -1, kernelSize2 = -1;
    std::tuple<cv::Size, cv::Size, cv::Rect> sizesWithRoi;
    std::tie(interp, kernelSize1, kernelSize2, readFromInput, sizesWithRoi) = GetParam();

    cv::Size inSz,  outSz;
    cv::Rect outRoi;
    std::tie(inSz, outSz, outRoi) = sizesWithRoi;

    cv::Mat in_mat(inSz, CV_8UC1);
    cv::Scalar mean = cv::Scalar(127);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);
    cv::Mat gapi_out1 = cv::Mat::zeros(outSz, CV_8UC1);
    cv::Mat gapi_out2 = cv::Mat::zeros(outSz, CV_8UC1);

    auto blur1 = kernelSize1 == 1 ? &TBlur1x1::on : kernelSize1 == 3 ? &TBlur3x3::on : &TBlur5x5::on;
    auto blur2 = kernelSize2 == 1 ? &TBlur1x1::on : kernelSize2 == 3 ? &TBlur3x3::on : &TBlur5x5::on;

    cv::GMat in, out1, out2;
    if (readFromInput)
    {
        auto resized = gapi::resize(in, outSz, 0, 0, interp);
        out1 = blur1(resized, cv::BORDER_DEFAULT, {});
        out2 = blur2(resized, cv::BORDER_DEFAULT, {});
    }
    else
    {
        auto mid = TCopy::on(in);
        auto resized = gapi::resize(mid, outSz, 0, 0, interp);
        out1 = blur1(resized, cv::BORDER_DEFAULT, {});
        out2 = blur2(resized, cv::BORDER_DEFAULT, {});
    }

    cv::GComputation c(GIn(in), GOut(out1, out2));
    c.apply(gin(in_mat), gout(gapi_out1, gapi_out2), cv::compile_args(GFluidOutputRois{{outRoi, outRoi}},
                                                                      fluidResizeTestPackage(interp, inSz, outSz)));

    cv::Mat ocv_out1 = cv::Mat::zeros(outSz, CV_8UC1);
    cv::Mat ocv_out2 = cv::Mat::zeros(outSz, CV_8UC1);
    cv::Mat resized = cv::Mat::zeros(outSz, CV_8UC1);
    cv::resize(in_mat, resized, outSz, 0, 0, interp);
    cvBlur(resized, ocv_out1, kernelSize1);
    cvBlur(resized, ocv_out2, kernelSize2);

    EXPECT_EQ(0, cv::countNonZero(gapi_out1(outRoi) != ocv_out1(outRoi)));
    EXPECT_EQ(0, cv::countNonZero(gapi_out2(outRoi) != ocv_out2(outRoi)));
}

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, BlursAfterResizeTest,
                        Combine(Values(cv::INTER_NEAREST, cv::INTER_LINEAR),
                                Values(1, 3, 5),
                                Values(1, 3, 5),
                                testing::Bool(), // Read from input directly or place a copy node at start
                                Values(std::make_tuple(cv::Size{8,8},
                                                       cv::Size{4,4}, cv::Rect{0,0,4,4}),
                                       std::make_tuple(cv::Size{8,8},
                                                       cv::Size{4,4}, cv::Rect{0,0,4,1}),
                                       std::make_tuple(cv::Size{8,8},
                                                       cv::Size{4,4}, cv::Rect{0,1,4,2}),
                                       std::make_tuple(cv::Size{8,8},
                                                       cv::Size{4,4}, cv::Rect{0,2,4,2}),
                                       std::make_tuple(cv::Size{64,64},
                                                       cv::Size{49,49}, cv::Rect{0, 0,49,49}),
                                       std::make_tuple(cv::Size{64,64},
                                                       cv::Size{49,49}, cv::Rect{0, 0,49,11}),
                                       std::make_tuple(cv::Size{64,64},
                                                       cv::Size{49,49}, cv::Rect{0, 9,49,17}),
                                       std::make_tuple(cv::Size{64,64},
                                                       cv::Size{49,49}, cv::Rect{0,39,49,10}))));

} // namespace opencv_test
