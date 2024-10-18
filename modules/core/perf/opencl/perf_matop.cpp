// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// SetTo ////////////////////////

typedef Size_MatType SetToFixture;

OCL_PERF_TEST_P(SetToFixture, SetTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar s = Scalar::all(17);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    declare.in(src, WARMUP_RNG).out(src);

    OCL_TEST_CYCLE() src.setTo(s);

    SANITY_CHECK(src);
}

///////////// SetTo with mask ////////////////////////

typedef Size_MatType SetToFixture;

OCL_PERF_TEST_P(SetToFixture, SetToWithMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar s = Scalar::all(17);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), mask(srcSize, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(src);

    OCL_TEST_CYCLE() src.setTo(s, mask);

    SANITY_CHECK(src);
}

///////////// ConvertTo ////////////////////////

typedef Size_MatType ConvertToFixture;

OCL_PERF_TEST_P(ConvertToFixture, ConvertTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ddepth = CV_MAT_DEPTH(type) == CV_8U ? CV_32F : CV_8U,
        cn = CV_MAT_CN(type), dtype = CV_MAKE_TYPE(ddepth, cn);

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type), dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK(dst);
}


static Size convertFP16_srcSize(4000, 4000);

OCL_PERF_TEST(Core, ConvertFP32FP16MatMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_32F;
    const int dtype = CV_16F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    Mat src(srcSize, type);
    Mat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP32FP16MatUMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_32F;
    const int dtype = CV_16F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    Mat src(srcSize, type);
    UMat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP32FP16UMatMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_32F;
    const int dtype = CV_16F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type);
    Mat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP32FP16UMatUMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_32F;
    const int dtype = CV_16F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type);
    UMat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP16FP32MatMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_16F;
    const int dtype = CV_32F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    Mat src(srcSize, type);
    Mat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP16FP32MatUMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_16F;
    const int dtype = CV_32F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    Mat src(srcSize, type);
    UMat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP16FP32UMatMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_16F;
    const int dtype = CV_32F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type);
    Mat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST(Core, ConvertFP16FP32UMatUMat)
{
    const Size srcSize = convertFP16_srcSize;
    const int type = CV_16F;
    const int dtype = CV_32F;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type);
    UMat dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK_NOTHING();
}


///////////// CopyTo ////////////////////////

typedef Size_MatType CopyToFixture;

OCL_PERF_TEST_P(CopyToFixture, CopyTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.copyTo(dst);

    SANITY_CHECK(dst);
}

///////////// CopyTo with mask ////////////////////////

typedef Size_MatType CopyToFixture;

OCL_PERF_TEST_P(CopyToFixture, CopyToWithMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type), mask(srcSize, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.copyTo(dst, mask);

    SANITY_CHECK(dst);
}

OCL_PERF_TEST_P(CopyToFixture, CopyToWithMaskUninit,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3), OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst, mask(srcSize, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG);

    for ( ;  next(); )
    {
        dst.release();
        startTimer();
        src.copyTo(dst, mask);
        cvtest::ocl::perf::safeFinish();
        stopTimer();
    }

    SANITY_CHECK(dst);
}



enum ROIType {
    ROI_FULL,
    ROI_2_RECT,
    ROI_2_TOP,  // contiguous memory block
    ROI_2_LEFT,
    ROI_4,
    ROI_16,
};
static Rect getROI(enum ROIType t, const Size& sz)
{
    switch (t)
    {
        case ROI_FULL: return Rect(0, 0, sz.width, sz.height);
        case ROI_2_RECT: return Rect(0, 0, sz.width * 71 / 100, sz.height * 71 / 100);  // 71 = sqrt(1/2) * 100
        case ROI_2_TOP: return Rect(0, 0, sz.width, sz.height / 2);  // 71 = sqrt(1/2) * 100
        case ROI_2_LEFT: return Rect(0, 0, sz.width / 2, sz.height);  // 71 = sqrt(1/2) * 100
        case ROI_4: return Rect(0, 0, sz.width / 2, sz.height / 2);
        case ROI_16: return Rect(0, 0, sz.width / 4, sz.height / 4);
    }
    CV_Assert(false);
}

typedef TestBaseWithParam< tuple<cv::Size, MatType, ROIType> > OpenCLBuffer;

static inline void PrintTo(const tuple<cv::Size, MatType, enum ROIType>& v, std::ostream* os)
{
    *os << "(" << get<0>(v) << ", " << typeToString(get<1>(v)) << ", ";
    enum ROIType roiType = get<2>(v);
    if (roiType == ROI_FULL)
        *os << "ROI_100_FULL";
    else if (roiType == ROI_2_RECT)
        *os << "ROI_050_RECT_HALF";
    else if (roiType == ROI_2_TOP)
        *os << "ROI_050_TOP_HALF";
    else if (roiType == ROI_2_LEFT)
        *os << "ROI_050_LEFT_HALF";
    else if (roiType == ROI_4)
        *os << "ROI_025_1/4";
    else
        *os << "ROI_012_1/16";
    *os << ")";
}

PERF_TEST_P_(OpenCLBuffer, cpu_write)
{
    const Size srcSize = get<0>(GetParam());
    const int type = get<1>(GetParam());
    const Rect roi = getROI(get<2>(GetParam()), srcSize);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    declare.in(src(roi), WARMUP_NONE);

    OCL_TEST_CYCLE()
    {
        Mat m = src(roi).getMat(ACCESS_WRITE);
        m.setTo(Scalar(1, 2, 3, 4));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(OpenCLBuffer, cpu_read)
{
    const Size srcSize = get<0>(GetParam());
    const int type = get<1>(GetParam());
    const Rect roi = getROI(get<2>(GetParam()), srcSize);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type, Scalar(1, 2, 3, 4));
    declare.in(src(roi), WARMUP_NONE);

    OCL_TEST_CYCLE()
    {
        unsigned counter = 0;
        Mat m = src(roi).getMat(ACCESS_READ);
        for (int y = 0; y < m.rows; y++)
        {
            uchar* ptr = m.ptr(y);
            size_t width_bytes = m.cols * m.elemSize();
            for (size_t x_bytes = 0; x_bytes < width_bytes; x_bytes++)
                counter += (unsigned)(ptr[x_bytes]);
        }
        (void)counter; // To avoid -Wunused-but-set-variable
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(OpenCLBuffer, cpu_update)
{
    const Size srcSize = get<0>(GetParam());
    const int type = get<1>(GetParam());
    const Rect roi = getROI(get<2>(GetParam()), srcSize);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type, Scalar(1, 2, 3, 4));
    declare.in(src(roi), WARMUP_NONE);

    OCL_TEST_CYCLE()
    {
        Mat m = src(roi).getMat(ACCESS_READ | ACCESS_WRITE);
        for (int y = 0; y < m.rows; y++)
        {
            uchar* ptr = m.ptr(y);
            size_t width_bytes = m.cols * m.elemSize();
            for (size_t x_bytes = 0; x_bytes < width_bytes; x_bytes++)
                ptr[x_bytes] += 1;
        }
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*FULL*/, OpenCLBuffer,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p, sz2160p),
        testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4),
        testing::Values(ROI_FULL)
    )
);

INSTANTIATE_TEST_CASE_P(ROI, OpenCLBuffer,
    testing::Combine(
        testing::Values(sz1080p, sz2160p),
        testing::Values(CV_8UC1),
        testing::Values(ROI_16, ROI_4, ROI_2_RECT, ROI_2_LEFT, ROI_2_TOP, ROI_FULL)
    )
);


} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
