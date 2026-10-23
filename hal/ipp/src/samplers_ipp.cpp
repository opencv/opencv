// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 700

using namespace cv;

int ipp_hal_getRectSubPix(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                          int patch_type, uchar* patch_data, size_t patch_step, int patch_width, int patch_height,
                          double center_x, double center_y)
{
    CV_HAL_CHECK_USE_IPP();

    int ddepth = CV_MAT_DEPTH(patch_type);

    // ippiCopySubpixIntersect replicates the border incorrectly in some cases, see
    // https://github.com/opencv/opencv/issues/30301. The window position is computed the same way as IPP does,
    // including (int)(x + w) - w, which is not equal to cvFloor() for some inputs.
    float fx = (float)center_x - 0.5f * (patch_width - 1);
    float fy = (float)center_y - 0.5f * (patch_height - 1);
    int sx = (int)(fx + patch_width) - patch_width;
    int sy = (int)(fy + patch_height) - patch_height;

    // the window crosses the top edge and reaches the last column: the top rows get src(0, cols-2)
    // instead of src(0, cols-1), and for 1-column images IPP reads outside of the image
    if( sy < 0 && sx >= src_width - patch_width )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;


    // 8u->8u, the window lies totally to the left of the image: the fixed-point weights overflow
    if( src_type == CV_8UC1 && ddepth == CV_8U && fx + patch_width < 0 )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    typedef IppStatus (CV_STDCALL *ippiGetRectSubPixFunc)( const void* src, int src_step,
                                                           IppiSize src_size, void* dst,
                                                           int dst_step, IppiSize win_size,
                                                           IppiPoint_32f center,
                                                           IppiPoint* minpt, IppiPoint* maxpt );

    ippiGetRectSubPixFunc ippiCopySubpixIntersect =
        src_type == CV_8UC1 && ddepth == CV_8U ? (ippiGetRectSubPixFunc)ippiCopySubpixIntersect_8u_C1R :
        src_type == CV_8UC1 && ddepth == CV_32F ? (ippiGetRectSubPixFunc)ippiCopySubpixIntersect_8u32f_C1R :
        src_type == CV_32FC1 && ddepth == CV_32F ? (ippiGetRectSubPixFunc)ippiCopySubpixIntersect_32f_C1R : 0;

    if( ippiCopySubpixIntersect )
    {
        IppiPoint minpt = {0, 0}, maxpt = {0, 0};
        IppiPoint_32f icenter = {(float)center_x, (float)center_y};
        IppiSize src_size = {src_width, src_height}, win_size = {patch_width, patch_height};

        if( CV_INSTRUMENT_FUN_IPP(ippiCopySubpixIntersect, src_data, (int)src_step, src_size, patch_data,
                                  (int)patch_step, win_size, icenter, &minpt, &maxpt) >= 0 )
            return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif
