// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#ifndef __OPENCV_FAST_NLMEANS_DENOISING_OPENCL_HPP__
#define __OPENCV_FAST_NLMEANS_DENOISING_OPENCL_HPP__

#include "opencl_kernels_photo.hpp"

#ifdef HAVE_OPENCL

namespace cv {

enum
{
    BLOCK_ROWS = 32,
    BLOCK_COLS = 32,
    CTA_SIZE_INTEL = 64,
    CTA_SIZE_DEFAULT = 256
};

template <typename FT, typename ST, typename WT>
static bool ocl_calcAlmostDist2Weight(UMat & almostDist2Weight,
                                      int searchWindowSize, int templateWindowSize,
                                      const FT *h, int hn, int cn, int normType,
                                      int & almostTemplateWindowSizeSqBinShift)
{
    const WT maxEstimateSumValue = searchWindowSize * searchWindowSize *
        std::numeric_limits<ST>::max();
    int fixedPointMult = (int)std::min<WT>(std::numeric_limits<WT>::max() / maxEstimateSumValue,
                                           std::numeric_limits<int>::max());
    int depth = DataType<FT>::depth;
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (depth == CV_64F && !doubleSupport)
        return false;

    // precalc weight for every possible l2 dist between blocks
    // additional optimization of precalced weights to replace division(averaging) by binary shift
    CV_Assert(templateWindowSize <= 46340); // sqrt(INT_MAX)
    int templateWindowSizeSq = templateWindowSize * templateWindowSize;
    almostTemplateWindowSizeSqBinShift = getNearestPowerOf2(templateWindowSizeSq);
    FT almostDist2ActualDistMultiplier = (FT)(1 << almostTemplateWindowSizeSqBinShift) / templateWindowSizeSq;

    const FT WEIGHT_THRESHOLD = 1e-3f;
    WT maxDist = normType == NORM_L1 ? (WT)std::numeric_limits<ST>::max() * cn :
        (WT)std::numeric_limits<ST>::max() * (WT)std::numeric_limits<ST>::max() * cn;
    int almostMaxDist = (int)(maxDist / almostDist2ActualDistMultiplier + 1);
    FT den[4];
    CV_Assert(hn > 0 && hn <= 4);
    for (int i=0; i<hn; i++)
        den[i] = 1.0f / (h[i] * h[i] * cn);

    almostDist2Weight.create(1, almostMaxDist, CV_32SC(hn == 3 ? 4 : hn));

    char buf[40];
    ocl::Kernel k("calcAlmostDist2Weight", ocl::photo::nlmeans_oclsrc,
                  format("-D OP_CALC_WEIGHTS -D FT=%s -D w_t=%s"
                         " -D wlut_t=%s -D convert_wlut_t=%s%s%s",
                         ocl::typeToStr(depth), ocl::typeToStr(CV_MAKE_TYPE(depth, hn)),
                         ocl::typeToStr(CV_32SC(hn)), ocl::convertTypeStr(depth, CV_32S, hn, buf),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         normType == NORM_L1 ? " -D ABS" : ""));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::PtrWriteOnly(almostDist2Weight), almostMaxDist,
           almostDist2ActualDistMultiplier, fixedPointMult,
           ocl::KernelArg::Constant(den, (hn == 3 ? 4 : hn)*sizeof(FT)), WEIGHT_THRESHOLD);

    size_t globalsize[1] = { (size_t)almostMaxDist };
    return k.run(1, globalsize, NULL, false);
}

static bool ocl_fastNlMeansDenoising(InputArray _src, OutputArray _dst, const float *h, int hn,
                                     int templateWindowSize, int searchWindowSize, int normType)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int ctaSize = ocl::Device::getDefault().isIntel() ? CTA_SIZE_INTEL : CTA_SIZE_DEFAULT;
    Size size = _src.size();

    if (cn < 1 || cn > 4 || ((normType != NORM_L2 || depth != CV_8U) &&
                             (normType != NORM_L1 || (depth != CV_8U && depth != CV_16U))))
        return false;

    int templateWindowHalfWize = templateWindowSize / 2;
    int searchWindowHalfSize = searchWindowSize / 2;
    templateWindowSize  = templateWindowHalfWize * 2 + 1;
    searchWindowSize = searchWindowHalfSize * 2 + 1;
    int nblocksx = divUp(size.width, BLOCK_COLS), nblocksy = divUp(size.height, BLOCK_ROWS);
    int almostTemplateWindowSizeSqBinShift = -1;

    char buf[4][40];
    String opts = format("-D OP_CALC_FASTNLMEANS -D TEMPLATE_SIZE=%d -D SEARCH_SIZE=%d"
                         " -D pixel_t=%s -D int_t=%s -D wlut_t=%s"
                         " -D weight_t=%s -D convert_weight_t=%s -D sum_t=%s -D convert_sum_t=%s"
                         " -D BLOCK_COLS=%d -D BLOCK_ROWS=%d"
                         " -D CTA_SIZE=%d -D TEMPLATE_SIZE2=%d -D SEARCH_SIZE2=%d"
                         " -D convert_int_t=%s -D cn=%d -D psz=%d -D convert_pixel_t=%s%s",
                         templateWindowSize, searchWindowSize,
                         ocl::typeToStr(type), ocl::typeToStr(CV_32SC(cn)),
                         ocl::typeToStr(CV_32SC(hn)),
                         depth == CV_8U ? ocl::typeToStr(CV_32SC(hn)) :
                         format("long%s", hn > 1 ? format("%d", hn).c_str() : "").c_str(),
                         depth == CV_8U ? ocl::convertTypeStr(CV_32S, CV_32S, hn, buf[0]) :
                         format("convert_long%s", hn > 1 ? format("%d", hn).c_str() : "").c_str(),
                         depth == CV_8U ? ocl::typeToStr(CV_32SC(cn)) :
                         format("long%s", cn > 1 ? format("%d", cn).c_str() : "").c_str(),
                         depth == CV_8U ? ocl::convertTypeStr(depth, CV_32S, cn, buf[1]) :
                         format("convert_long%s", cn > 1 ? format("%d", cn).c_str() : "").c_str(),
                         BLOCK_COLS, BLOCK_ROWS,
                         ctaSize, templateWindowHalfWize, searchWindowHalfSize,
                         ocl::convertTypeStr(depth, CV_32S, cn, buf[2]), cn,
                         (depth == CV_8U ? sizeof(uchar) : sizeof(ushort)) * (cn == 3 ? 4 : cn),
                         ocl::convertTypeStr(CV_32S, depth, cn, buf[3]),
                         normType == NORM_L1 ? " -D ABS" : "");

    ocl::Kernel k("fastNlMeansDenoising", ocl::photo::nlmeans_oclsrc, opts);
    if (k.empty())
        return false;

    UMat almostDist2Weight;
    if ((depth == CV_8U &&
         !ocl_calcAlmostDist2Weight<float, uchar, int>(almostDist2Weight,
                                                       searchWindowSize, templateWindowSize,
                                                       h, hn, cn, normType,
                                                       almostTemplateWindowSizeSqBinShift)) ||
        (depth == CV_16U &&
         !ocl_calcAlmostDist2Weight<float, ushort, int64>(almostDist2Weight,
                                                          searchWindowSize, templateWindowSize,
                                                          h, hn, cn, normType,
                                                          almostTemplateWindowSizeSqBinShift)))
        return false;
    CV_Assert(almostTemplateWindowSizeSqBinShift >= 0);

    UMat srcex;
    int borderSize = searchWindowHalfSize + templateWindowHalfWize;
    if (cn == 3) {
        srcex.create(size.height + 2*borderSize, size.width + 2*borderSize, CV_MAKE_TYPE(depth, 4));
        UMat src(srcex, Rect(borderSize, borderSize, size.width, size.height));
        int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(std::vector<UMat>(1, _src.getUMat()), std::vector<UMat>(1, src), from_to, 3);
        copyMakeBorder(src, srcex, borderSize, borderSize, borderSize, borderSize,
                       BORDER_DEFAULT|BORDER_ISOLATED); // create borders in place
    }
    else
        copyMakeBorder(_src, srcex, borderSize, borderSize, borderSize, borderSize, BORDER_DEFAULT);

    _dst.create(size, type);
    UMat dst;
    if (cn == 3)
        dst.create(size, CV_MAKE_TYPE(depth, 4));
    else
        dst = _dst.getUMat();

    int searchWindowSizeSq = searchWindowSize * searchWindowSize;
    Size upColSumSize(size.width, searchWindowSizeSq * nblocksy);
    Size colSumSize(nblocksx * templateWindowSize, searchWindowSizeSq * nblocksy);
    UMat buffer(upColSumSize + colSumSize, CV_32SC(cn));

    srcex = srcex(Rect(Point(borderSize, borderSize), size));
    k.args(ocl::KernelArg::ReadOnlyNoSize(srcex), ocl::KernelArg::WriteOnly(dst),
           ocl::KernelArg::PtrReadOnly(almostDist2Weight),
           ocl::KernelArg::PtrReadOnly(buffer), almostTemplateWindowSizeSqBinShift);

    size_t globalsize[2] = { (size_t)nblocksx * ctaSize, (size_t)nblocksy }, localsize[2] = { (size_t)ctaSize, 1 };
    if (!k.run(2, globalsize, localsize, false)) return false;

    if (cn == 3) {
        int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(std::vector<UMat>(1, dst), std::vector<UMat>(1, _dst.getUMat()), from_to, 3);
    }

    return true;
}

static bool ocl_fastNlMeansDenoisingColored( InputArray _src, OutputArray _dst,
                                      float h, float hForColorComponents,
                                      int templateWindowSize, int searchWindowSize)
{
    UMat src = _src.getUMat();
    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    UMat src_lab;
    cvtColor(src, src_lab, COLOR_LBGR2Lab);

    UMat l(src.size(), CV_8U);
    UMat ab(src.size(), CV_8UC2);
    std::vector<UMat> l_ab(2), l_ab_denoised(2);
    l_ab[0] = l;
    l_ab[1] = ab;
    l_ab_denoised[0].create(src.size(), CV_8U);
    l_ab_denoised[1].create(src.size(), CV_8UC2);

    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(std::vector<UMat>(1, src_lab), l_ab, from_to, 3);

    fastNlMeansDenoising(l_ab[0], l_ab_denoised[0], h, templateWindowSize, searchWindowSize);
    fastNlMeansDenoising(l_ab[1], l_ab_denoised[1], hForColorComponents, templateWindowSize, searchWindowSize);

    UMat dst_lab(src.size(), CV_8UC3);
    mixChannels(l_ab_denoised, std::vector<UMat>(1, dst_lab), from_to, 3);

    cvtColor(dst_lab, dst, COLOR_Lab2LBGR, src.channels());
    return true;
}

}

#endif
#endif
