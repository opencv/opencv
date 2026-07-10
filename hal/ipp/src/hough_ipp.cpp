// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <algorithm>

#if IPP_VERSION_X100 >= 810 && !DISABLE_IPP_HOUGH

int ipp_hal_houghLines(const uchar* src_data, size_t src_step, int width, int height,
                       float rho, float theta, int threshold, int numangle,
                       double min_theta, double max_theta,
                       int lines_max, float** out_lines, int* out_count)
{
    CV_HAL_CHECK_USE_IPP();

    *out_lines = NULL;
    *out_count = 0;

    IppiSize srcSize = { width, height };
    IppPointPolar delta = { rho, theta };
    int max_rho = width + height;
    int min_rho = -max_rho;
    IppPointPolar dstRoi[2] = {{(Ipp32f)min_rho, (Ipp32f)min_theta}, {(Ipp32f)max_rho, (Ipp32f)max_theta}};

    // countNonZero equivalent; HAL cannot call cv::countNonZero (no opencv_core link)
    Ipp32s zeros = 0;
    if (CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, (const Ipp8u*)src_data, (int)src_step, srcSize, &zeros, 0, 0) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    int nz = width * height - zeros;

    int ipp_linesMax = std::min(lines_max, nz * numangle / threshold);
    Ipp32f* lines = (Ipp32f*)ippsMalloc_8u_L((size_t)ipp_linesMax * 2 * sizeof(Ipp32f));
    if (!lines && ipp_linesMax > 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int bufferSize = 0;
    int linesCount = 0;
    IppStatus ok = ippiHoughLineGetSize_8u_C1R(srcSize, delta, ipp_linesMax, &bufferSize);
    Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
    if (ok >= 0) { ok = CV_INSTRUMENT_FUN_IPP(ippiHoughLine_Region_8u32f_C1R, src_data, (int)src_step, srcSize, (IppPointPolar*)lines, dstRoi, ipp_linesMax, &linesCount, delta, threshold, buffer); }
    ippsFree(buffer);
    if (ok >= 0)
    {
        *out_lines = lines;
        *out_count = linesCount;
        return CV_HAL_ERROR_OK;
    }
    ippsFree(lines);
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_houghLinesProbabilistic(const uchar* src_data, size_t src_step, int width, int height,
                                    float rho, float theta, int threshold,
                                    int line_length, int line_gap,
                                    int numangle, int numrho,
                                    int lines_max, int** out_lines, int* out_count)
{
    CV_HAL_CHECK_USE_IPP();

    *out_lines = NULL;
    *out_count = 0;

    IppiSize srcSize = { width, height };
    IppPointPolar delta = { rho, theta };
    int ipp_linesMax = std::min(lines_max, numangle * numrho);
    Ipp32s* lines = (Ipp32s*)ippsMalloc_8u_L((size_t)ipp_linesMax * 4 * sizeof(Ipp32s));
    if (!lines && ipp_linesMax > 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiHoughProbSpec* pSpec = NULL;
    int bufferSize = 0, specSize = 0;
    int linesCount = 0;
    IppStatus ok = ippiHoughProbLineGetSize_8u_C1R(srcSize, delta, &specSize, &bufferSize);
    Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
    pSpec = (IppiHoughProbSpec*)ippsMalloc_8u_L(specSize);
    if (ok >= 0) ok = ippiHoughProbLineInit_8u32f_C1R(srcSize, delta, ippAlgHintNone, pSpec);
    if (ok >= 0) ok = CV_INSTRUMENT_FUN_IPP(ippiHoughProbLine_8u32f_C1R, src_data, (int)src_step, srcSize, threshold, line_length, line_gap, (IppiPoint*)lines, ipp_linesMax, &linesCount, buffer, pSpec);
    ippsFree(pSpec);
    ippsFree(buffer);
    if (ok >= 0)
    {
        *out_lines = lines;
        *out_count = linesCount;
        return CV_HAL_ERROR_OK;
    }
    ippsFree(lines);
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

void ipp_hal_houghLinesFree(void* lines)
{
    ippsFree(lines);
}

#endif
