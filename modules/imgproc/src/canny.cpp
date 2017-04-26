/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <queue>

#include "opencv2/core/openvx/ovx_defs.hpp"

#ifdef _MSC_VER
#pragma warning( disable: 4127 ) // conditional expression is constant
#endif

namespace cv
{

static void CannyImpl(Mat& dx_, Mat& dy_, Mat& _dst, double low_thresh, double high_thresh, bool L2gradient);


#ifdef HAVE_IPP
static bool ipp_Canny(const Mat& src , const Mat& dx_, const Mat& dy_, Mat& dst, float low,  float high, bool L2gradient, int aperture_size)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

#if IPP_DISABLE_PERF_CANNY_MT
    if(cv::getNumThreads()>1)
        return false;
#endif

    ::ipp::IwiSize size(dst.cols, dst.rows);
    IppDataType    type     = ippiGetDataType(dst.depth());
    int            channels = dst.channels();
    IppNormType    norm     = (L2gradient)?ippNormL2:ippNormL1;

    if(size.width <= 3 || size.height <= 3)
        return false;

    if(channels != 1)
        return false;

    if(type != ipp8u)
        return false;

    if(src.empty())
    {
        try
        {
            ::ipp::IwiImage iwSrcDx;
            ::ipp::IwiImage iwSrcDy;
            ::ipp::IwiImage iwDst;

            ippiGetImage(dx_, iwSrcDx);
            ippiGetImage(dy_, iwSrcDy);
            ippiGetImage(dst, iwDst);

            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterCannyDeriv, &iwSrcDx, &iwSrcDy, &iwDst, norm, low, high);
        }
        catch (::ipp::IwException ex)
        {
            return false;
        }
    }
    else
    {
        IppiMaskSize kernel;

        if(aperture_size == 3)
            kernel = ippMskSize3x3;
        else if(aperture_size == 5)
            kernel = ippMskSize5x5;
        else
            return false;

        try
        {
            ::ipp::IwiImage iwSrc;
            ::ipp::IwiImage iwDst;

            ippiGetImage(src, iwSrc);
            ippiGetImage(dst, iwDst);

            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterCanny, &iwSrc, &iwDst, ippFilterSobel, kernel, norm, low, high, ippBorderRepl);
        }
        catch (::ipp::IwException)
        {
            return false;
        }
    }

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dx_); CV_UNUSED(dy_); CV_UNUSED(dst); CV_UNUSED(low); CV_UNUSED(high); CV_UNUSED(L2gradient); CV_UNUSED(aperture_size);
    return false;
#endif
}
#endif

#ifdef HAVE_OPENCL

template <bool useCustomDeriv>
static bool ocl_Canny(InputArray _src, const UMat& dx_, const UMat& dy_, OutputArray _dst, float low_thresh, float high_thresh,
                      int aperture_size, bool L2gradient, int cn, const Size & size)
{
    CV_INSTRUMENT_REGION_OPENCL()

    UMat map;

    const ocl::Device &dev = ocl::Device::getDefault();
    int max_wg_size = (int)dev.maxWorkGroupSize();

    int lSizeX = 32;
    int lSizeY = max_wg_size / 32;

    if (lSizeY == 0)
    {
        lSizeX = 16;
        lSizeY = max_wg_size / 16;
    }
    if (lSizeY == 0)
    {
        lSizeY = 1;
    }

    if (L2gradient)
    {
        low_thresh = std::min(32767.0f, low_thresh);
        high_thresh = std::min(32767.0f, high_thresh);

        if (low_thresh > 0)
            low_thresh *= low_thresh;
        if (high_thresh > 0)
            high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh), high = cvFloor(high_thresh);

    if (!useCustomDeriv &&
        aperture_size == 3 && !_src.isSubmatrix())
    {
        /*
            stage1_with_sobel:
                Sobel operator
                Calc magnitudes
                Non maxima suppression
                Double thresholding
        */
        char cvt[40];
        ocl::Kernel with_sobel("stage1_with_sobel", ocl::imgproc::canny_oclsrc,
                               format("-D WITH_SOBEL -D cn=%d -D TYPE=%s -D convert_floatN=%s -D floatN=%s -D GRP_SIZEX=%d -D GRP_SIZEY=%d%s",
                                      cn, ocl::memopTypeToStr(_src.depth()),
                                      ocl::convertTypeStr(_src.depth(), CV_32F, cn, cvt),
                                      ocl::typeToStr(CV_MAKE_TYPE(CV_32F, cn)),
                                      lSizeX, lSizeY,
                                      L2gradient ? " -D L2GRAD" : ""));
        if (with_sobel.empty())
            return false;

        UMat src = _src.getUMat();
        map.create(size, CV_32S);
        with_sobel.args(ocl::KernelArg::ReadOnly(src),
                        ocl::KernelArg::WriteOnlyNoSize(map),
                        (float) low, (float) high);

        size_t globalsize[2] = { (size_t)size.width, (size_t)size.height },
                localsize[2] = { (size_t)lSizeX, (size_t)lSizeY };

        if (!with_sobel.run(2, globalsize, localsize, false))
            return false;
    }
    else
    {
        /*
            stage1_without_sobel:
                Calc magnitudes
                Non maxima suppression
                Double thresholding
        */
        UMat dx, dy;
        if (!useCustomDeriv)
        {
            Sobel(_src, dx, CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
            Sobel(_src, dy, CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);
        }
        else
        {
            dx = dx_;
            dy = dy_;
        }

        ocl::Kernel without_sobel("stage1_without_sobel", ocl::imgproc::canny_oclsrc,
                                    format("-D WITHOUT_SOBEL -D cn=%d -D GRP_SIZEX=%d -D GRP_SIZEY=%d%s",
                                           cn, lSizeX, lSizeY, L2gradient ? " -D L2GRAD" : ""));
        if (without_sobel.empty())
            return false;

        map.create(size, CV_32S);
        without_sobel.args(ocl::KernelArg::ReadOnlyNoSize(dx), ocl::KernelArg::ReadOnlyNoSize(dy),
                           ocl::KernelArg::WriteOnly(map),
                           low, high);

        size_t globalsize[2] = { (size_t)size.width, (size_t)size.height },
                localsize[2] = { (size_t)lSizeX, (size_t)lSizeY };

        if (!without_sobel.run(2, globalsize, localsize, false))
            return false;
    }

    int PIX_PER_WI = 8;
    /*
        stage2:
            hysteresis (add weak edges if they are connected with strong edges)
    */

    int sizey = lSizeY / PIX_PER_WI;
    if (sizey == 0)
        sizey = 1;

    size_t globalsize[2] = { (size_t)size.width, ((size_t)size.height + PIX_PER_WI - 1) / PIX_PER_WI }, localsize[2] = { (size_t)lSizeX, (size_t)sizey };

    ocl::Kernel edgesHysteresis("stage2_hysteresis", ocl::imgproc::canny_oclsrc,
                                format("-D STAGE2 -D PIX_PER_WI=%d -D LOCAL_X=%d -D LOCAL_Y=%d",
                                PIX_PER_WI, lSizeX, sizey));

    if (edgesHysteresis.empty())
        return false;

    edgesHysteresis.args(ocl::KernelArg::ReadWrite(map));
    if (!edgesHysteresis.run(2, globalsize, localsize, false))
        return false;

    // get edges

    ocl::Kernel getEdgesKernel("getEdges", ocl::imgproc::canny_oclsrc,
                                format("-D GET_EDGES -D PIX_PER_WI=%d", PIX_PER_WI));
    if (getEdgesKernel.empty())
        return false;

    _dst.create(size, CV_8UC1);
    UMat dst = _dst.getUMat();

    getEdgesKernel.args(ocl::KernelArg::ReadOnly(map), ocl::KernelArg::WriteOnlyNoSize(dst));

    return getEdgesKernel.run(2, globalsize, NULL, false);
}

#endif

class parallelCanny : public ParallelLoopBody
{

public:
    parallelCanny(const Mat& _src, uchar* _map, int _low, int _high, int _aperture_size, bool _L2gradient, std::queue<uchar*> *borderPeaksParallel) :
        src(_src), map(_map), low(_low), high(_high), aperture_size(_aperture_size), L2gradient(_L2gradient), _borderPeaksParallel(borderPeaksParallel)
    {
    }

    ~parallelCanny()
    {
    }

    parallelCanny& operator=(const parallelCanny&) { return *this; }

    void operator()(const Range &boundaries) const
    {
#if CV_SIMD128
        bool haveSIMD = hasSIMD128();
#endif

        const int type = src.type(), cn = CV_MAT_CN(type);

        Mat dx, dy;
        std::queue<uchar*> borderPeaksLocal;

        ptrdiff_t mapstep = src.cols + 2;

        // In sobel transform we calculate ksize2 extra lines for the first and last rows of each slice
        // because IPPDerivSobel expects only isolated ROIs, in contrast with the opencv version which
        // uses the pixels outside of the ROI to form a border.
        //
        // TODO: statement above is not true anymore, so adjustments may be required
        int ksize2 = aperture_size / 2;
        // If Scharr filter: aperture_size is 3 and ksize2 is 1
        if(aperture_size == -1)
        {
            ksize2 = 1;
        }

        if (boundaries.start == 0 && boundaries.end == src.rows)
        {
            Mat tempdx(boundaries.end - boundaries.start + 2, src.cols, CV_16SC(cn));
            Mat tempdy(boundaries.end - boundaries.start + 2, src.cols, CV_16SC(cn));

            memset(tempdx.ptr<short>(0), 0, cn * src.cols*sizeof(short));
            memset(tempdy.ptr<short>(0), 0, cn * src.cols*sizeof(short));
            memset(tempdx.ptr<short>(tempdx.rows - 1), 0, cn * src.cols*sizeof(short));
            memset(tempdy.ptr<short>(tempdy.rows - 1), 0, cn * src.cols*sizeof(short));

            Sobel(src, tempdx.rowRange(1, tempdx.rows - 1), CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
            Sobel(src, tempdy.rowRange(1, tempdy.rows - 1), CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

            dx = tempdx;
            dy = tempdy;
        }
        else if (boundaries.start == 0)
        {
            Mat tempdx(boundaries.end - boundaries.start + 2 + ksize2, src.cols, CV_16SC(cn));
            Mat tempdy(boundaries.end - boundaries.start + 2 + ksize2, src.cols, CV_16SC(cn));

            memset(tempdx.ptr<short>(0), 0, cn * src.cols*sizeof(short));
            memset(tempdy.ptr<short>(0), 0, cn * src.cols*sizeof(short));

            Sobel(src.rowRange(boundaries.start, boundaries.end + 1 + ksize2), tempdx.rowRange(1, tempdx.rows),
                    CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
            Sobel(src.rowRange(boundaries.start, boundaries.end + 1 + ksize2), tempdy.rowRange(1, tempdy.rows),
                    CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

            dx = tempdx.rowRange(0, tempdx.rows - ksize2);
            dy = tempdy.rowRange(0, tempdy.rows - ksize2);
        }
        else if (boundaries.end == src.rows)
        {
            Mat tempdx(boundaries.end - boundaries.start + 2 + ksize2, src.cols, CV_16SC(cn));
            Mat tempdy(boundaries.end - boundaries.start + 2 + ksize2, src.cols, CV_16SC(cn));

            memset(tempdx.ptr<short>(tempdx.rows - 1), 0, cn * src.cols*sizeof(short));
            memset(tempdy.ptr<short>(tempdy.rows - 1), 0, cn * src.cols*sizeof(short));

            Sobel(src.rowRange(boundaries.start - 1 - ksize2, boundaries.end), tempdx.rowRange(0, tempdx.rows - 1),
                    CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
            Sobel(src.rowRange(boundaries.start - 1 - ksize2, boundaries.end), tempdy.rowRange(0, tempdy.rows - 1),
                    CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

            dx = tempdx.rowRange(ksize2, tempdx.rows);
            dy = tempdy.rowRange(ksize2, tempdy.rows);
        }
        else
        {
            Mat tempdx(boundaries.end - boundaries.start + 2 + 2*ksize2, src.cols, CV_16SC(cn));
            Mat tempdy(boundaries.end - boundaries.start + 2 + 2*ksize2, src.cols, CV_16SC(cn));

            Sobel(src.rowRange(boundaries.start - 1 - ksize2, boundaries.end + 1 + ksize2), tempdx,
                    CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
            Sobel(src.rowRange(boundaries.start - 1 - ksize2, boundaries.end + 1 + ksize2), tempdy,
                    CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

            dx = tempdx.rowRange(ksize2, tempdx.rows - ksize2);
            dy = tempdy.rowRange(ksize2, tempdy.rows - ksize2);
        }

        int maxsize = std::max(1 << 10, src.cols * (boundaries.end - boundaries.start) / 10);
        std::vector<uchar*> stack(maxsize);
        uchar **stack_top = &stack[0];
        uchar **stack_bottom = &stack[0];

        AutoBuffer<uchar> buffer(cn * mapstep * 3 * sizeof(int));

        int* mag_buf[3];
        mag_buf[0] = (int*)(uchar*)buffer;
        mag_buf[1] = mag_buf[0] + mapstep*cn;
        mag_buf[2] = mag_buf[1] + mapstep*cn;

        // calculate magnitude and angle of gradient, perform non-maxima suppression.
        // fill the map with one of the following values:
        //   0 - the pixel might belong to an edge
        //   1 - the pixel can not belong to an edge
        //   2 - the pixel does belong to an edge
        for (int i = boundaries.start - 1; i <= boundaries.end; i++)
        {
            int* _norm = mag_buf[(i > boundaries.start) - (i == boundaries.start - 1) + 1] + 1;

            short* _dx = dx.ptr<short>(i - boundaries.start + 1);
            short* _dy = dy.ptr<short>(i - boundaries.start + 1);

            if (!L2gradient)
            {
                int j = 0, width = src.cols * cn;
#if CV_SIMD128
                if (haveSIMD)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        v_int16x8 v_dx = v_load((const short *)(_dx + j));
                        v_int16x8 v_dy = v_load((const short *)(_dy + j));

                        v_dx = v_reinterpret_as_s16(v_abs(v_dx));
                        v_dy = v_reinterpret_as_s16(v_abs(v_dy));

                        v_int32x4 v_dx_ml;
                        v_int32x4 v_dy_ml;
                        v_int32x4 v_dx_mh;
                        v_int32x4 v_dy_mh;
                        v_expand(v_dx, v_dx_ml, v_dx_mh);
                        v_expand(v_dy, v_dy_ml, v_dy_mh);

                        v_store((int *)(_norm + j), v_dx_ml + v_dy_ml);
                        v_store((int *)(_norm + j + 4), v_dx_mh + v_dy_mh);
                    }
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = src.cols * cn;
#if CV_SIMD128
                if (haveSIMD)
                {
                   for ( ; j <= width - 8; j += 8)
                    {
                        v_int16x8 v_dx = v_load((const short*)(_dx + j));
                        v_int16x8 v_dy = v_load((const short*)(_dy + j));

                        v_int32x4 v_dxp_low, v_dxp_high;
                        v_int32x4 v_dyp_low, v_dyp_high;
                        v_expand(v_dx, v_dxp_low, v_dxp_high);
                        v_expand(v_dy, v_dyp_low, v_dyp_high);

                        v_store((int *)(_norm + j), v_dxp_low*v_dxp_low+v_dyp_low*v_dyp_low);
                        v_store((int *)(_norm + j + 4), v_dxp_high*v_dxp_high+v_dyp_high*v_dyp_high);
                    }
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }

            if (cn > 1)
            {
                for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[src.cols] = 0;

            // at the very beginning we do not have a complete ring
            // buffer of 3 magnitude rows for non-maxima suppression
            if (i <= boundaries.start)
                continue;

            uchar* _map = map + mapstep*i + 1;
            _map[-1] = _map[src.cols] = 1;

            int* _mag = mag_buf[1] + 1; // take the central row
            ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
            ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

            const short* _x = dx.ptr<short>(i - boundaries.start);
            const short* _y = dy.ptr<short>(i - boundaries.start);

            if ((stack_top - stack_bottom) + src.cols > maxsize)
            {
                int sz = (int)(stack_top - stack_bottom);
                maxsize = std::max(maxsize * 3/2, sz + src.cols);
                stack.resize(maxsize);
                stack_bottom = &stack[0];
                stack_top = stack_bottom + sz;
            }

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

#define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

            int prev_flag = 0, j = 0;
#if CV_SIMD128
            if (haveSIMD)
            {
                v_int32x4 v_low = v_setall_s32(low);
                v_int8x16 v_one = v_setall_s8(1);

                for (; j <= src.cols - 16; j += 16)
                {
                    v_int32x4 v_m1 = v_load((const int*)(_mag + j));
                    v_int32x4 v_m2 = v_load((const int*)(_mag + j + 4));
                    v_int32x4 v_m3 = v_load((const int*)(_mag + j + 8));
                    v_int32x4 v_m4 = v_load((const int*)(_mag + j + 12));

                    v_store((signed char*)(_map + j), v_one);

                    v_int32x4 v_cmp1 = v_m1 > v_low;
                    v_int32x4 v_cmp2 = v_m2 > v_low;
                    v_int32x4 v_cmp3 = v_m3 > v_low;
                    v_int32x4 v_cmp4 = v_m4 > v_low;

                    v_int16x8 v_cmp80 = v_pack(v_cmp1, v_cmp2);
                    v_int16x8 v_cmp81 = v_pack(v_cmp3, v_cmp4);

                    v_int8x16 v_cmp = v_pack(v_cmp80, v_cmp81);
                    unsigned int mask = v_signmask(v_cmp);

                    if (mask)
                    {
                        int m, k = j;

                        for (; mask; ++k, mask >>= 1)
                        {
                            if (mask & 0x00000001)
                            {
                                m = _mag[k];
                                int xs = _x[k];
                                int ys = _y[k];
                                int x = std::abs(xs);
                                int y = std::abs(ys) << CANNY_SHIFT;

                                int tg22x = x * TG22;

                                if (y < tg22x)
                                {
                                    if (m > _mag[k - 1] && m >= _mag[k + 1]) goto _canny_push_sse;
                                }
                                else
                                {
                                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                                    if (y > tg67x)
                                    {
                                        if (m > _mag[k + magstep2] && m >= _mag[k + magstep1]) goto _canny_push_sse;
                                    } else
                                    {
                                        int s = (xs ^ ys) < 0 ? -1 : 1;
                                        if (m > _mag[k + magstep2 - s] && m > _mag[k + magstep1 + s]) goto _canny_push_sse;
                                    }
                                }
                            }

                            prev_flag = 0;
                            continue;

_canny_push_sse:
                            // _map[k-mapstep] is short-circuited at the start because previous thread is
                            // responsible for initializing it.
                            if (m > high && !prev_flag  && (i <= boundaries.start + 1 || _map[k - mapstep] != 2))
                            {
                                CANNY_PUSH(_map + k);
                                prev_flag = 1;
                            } else
                                _map[k] = 0;

                        }

                        if (prev_flag && ((k < j+16) || (k < src.cols && _mag[k] <= high)))
                            prev_flag = 0;
                    }
                }
            }
#endif
            for (; j < src.cols; j++)
            {
                int m = _mag[j];

                if (m > low)
                {
                    int xs = _x[j];
                    int ys = _y[j];
                    int x = std::abs(xs);
                    int y = std::abs(ys) << CANNY_SHIFT;

                    int tg22x = x * TG22;

                    if (y < tg22x)
                    {
                        if (m > _mag[j-1] && m >= _mag[j+1]) goto _canny_push;
                    }
                    else
                    {
                        int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                        if (y > tg67x)
                        {
                            if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto _canny_push;
                        }
                        else
                        {
                            int s = (xs ^ ys) < 0 ? -1 : 1;
                            if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto _canny_push;
                        }
                    }
                }

                prev_flag = 0;
                _map[j] = uchar(1);
                continue;

_canny_push:
                // _map[j-mapstep] is short-circuited at the start because previous thread is
                // responsible for initializing it.
                if (!prev_flag && m > high && (i <= boundaries.start+1 || _map[j-mapstep] != 2) )
                {
                    CANNY_PUSH(_map + j);
                    prev_flag = 1;
                }
                else
                    _map[j] = 0;
            }

            // scroll the ring buffer
            _mag = mag_buf[0];
            mag_buf[0] = mag_buf[1];
            mag_buf[1] = mag_buf[2];
            mag_buf[2] = _mag;
        }

        // now track the edges (hysteresis thresholding)
        while (stack_top > stack_bottom)
        {
            if ((stack_top - stack_bottom) + 8 > maxsize)
            {
                int sz = (int)(stack_top - stack_bottom);
                maxsize = maxsize * 3/2;
                stack.resize(maxsize);
                stack_bottom = &stack[0];
                stack_top = stack_bottom + sz;
            }

            uchar* m;
            CANNY_POP(m);

            // Stops thresholding from expanding to other slices by sending pixels in the borders of each
            // slice in a queue to be serially processed later.
            if ( (m < map + (boundaries.start + 2) * mapstep) || (m >= map + boundaries.end * mapstep) )
            {
                borderPeaksLocal.push(m);
                continue;
            }

            if (!m[-1])         CANNY_PUSH(m - 1);
            if (!m[1])          CANNY_PUSH(m + 1);
            if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
            if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
            if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
            if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
            if (!m[mapstep])    CANNY_PUSH(m + mapstep);
            if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
        }

        AutoLock lock(mutex);
        while (!borderPeaksLocal.empty()) {
            _borderPeaksParallel->push(borderPeaksLocal.front());
            borderPeaksLocal.pop();
        }
    }

private:
    const Mat& src;
    uchar* map;
    int low, high, aperture_size;
    bool L2gradient;
    std::queue<uchar*> *_borderPeaksParallel;
    mutable Mutex mutex;
};

class finalPass : public ParallelLoopBody
{

public:
    finalPass(uchar *_map, Mat &_dst, ptrdiff_t _mapstep) :
        map(_map), dst(_dst), mapstep(_mapstep) {}

    ~finalPass() {}

    finalPass& operator=(const finalPass&) {return *this;}

    void operator()(const Range &boundaries) const
    {
        // the final pass, form the final image
        const uchar* pmap = map + mapstep + 1 + (ptrdiff_t)(mapstep * boundaries.start);
        uchar* pdst = dst.ptr() + (ptrdiff_t)(dst.step * boundaries.start);

#if CV_SIMD128
        bool haveSIMD = hasSIMD128();
#endif

        for (int i = boundaries.start; i < boundaries.end; i++, pmap += mapstep, pdst += dst.step)
        {
            int j = 0;
#if CV_SIMD128
            if(haveSIMD) {
                const v_int8x16 v_zero = v_setzero_s8();

                for(; j <= dst.cols - 32; j += 32) {
                    v_uint8x16 v_pmap1 = v_load((const unsigned char*)(pmap + j));
                    v_uint8x16 v_pmap2 = v_load((const unsigned char*)(pmap + j + 16));

                    v_uint16x8 v_pmaplo1;
                    v_uint16x8 v_pmaphi1;
                    v_uint16x8 v_pmaplo2;
                    v_uint16x8 v_pmaphi2;
                    v_expand(v_pmap1, v_pmaplo1, v_pmaphi1);
                    v_expand(v_pmap2, v_pmaplo2, v_pmaphi2);

                    v_pmaplo1 = v_pmaplo1 >> 1;
                    v_pmaphi1 = v_pmaphi1 >> 1;
                    v_pmaplo2 = v_pmaplo2 >> 1;
                    v_pmaphi2 = v_pmaphi2 >> 1;

                    v_pmap1 = v_pack(v_pmaplo1, v_pmaphi1);
                    v_pmap2 = v_pack(v_pmaplo2, v_pmaphi2);

                    v_pmap1 = v_reinterpret_as_u8(v_zero - v_reinterpret_as_s8(v_pmap1));
                    v_pmap2 = v_reinterpret_as_u8(v_zero - v_reinterpret_as_s8(v_pmap2));

                    v_store((pdst + j), v_pmap1);
                    v_store((pdst + j + 16), v_pmap2);
                }

                for(; j <= dst.cols - 16; j += 16) {
                    v_uint8x16 v_pmap = v_load((const unsigned char*)(pmap + j));

                    v_uint16x8 v_pmaplo;
                    v_uint16x8 v_pmaphi;
                    v_expand(v_pmap, v_pmaplo, v_pmaphi);

                    v_pmaplo = v_pmaplo >> 1;
                    v_pmaphi = v_pmaphi >> 1;

                    v_pmap = v_pack(v_pmaplo, v_pmaphi);
                    v_pmap = v_reinterpret_as_u8(v_zero - v_reinterpret_as_s8(v_pmap));

                    v_store((pdst + j), v_pmap);
                }
            }
#endif
            for (; j < dst.cols; j++)
                pdst[j] = (uchar)-(pmap[j] >> 1);
        }
    }

private:
    uchar *map;
    Mat &dst;
    ptrdiff_t mapstep;
};

#ifdef HAVE_OPENVX
static bool openvx_canny(const Mat& src, Mat& dst, int loVal, int hiVal, int kSize, bool useL2)
{
    using namespace ivx;

    Context context = ovx::getOpenVXContext();
    try
    {
    Image _src = Image::createFromHandle(
                context,
                Image::matTypeToFormat(src.type()),
                Image::createAddressing(src),
                src.data );
    Image _dst = Image::createFromHandle(
                context,
                Image::matTypeToFormat(dst.type()),
                Image::createAddressing(dst),
                dst.data );
    Threshold threshold = Threshold::createRange(context, VX_TYPE_UINT8, saturate_cast<uchar>(loVal), saturate_cast<uchar>(hiVal));

#if 0
    // the code below is disabled because vxuCannyEdgeDetector()
    // ignores context attribute VX_CONTEXT_IMMEDIATE_BORDER

    // FIXME: may fail in multithread case
    border_t prevBorder = context.immediateBorder();
    context.setImmediateBorder(VX_BORDER_REPLICATE);
    IVX_CHECK_STATUS( vxuCannyEdgeDetector(context, _src, threshold, kSize, (useL2 ? VX_NORM_L2 : VX_NORM_L1), _dst) );
    context.setImmediateBorder(prevBorder);
#else
    // alternative code without vxuCannyEdgeDetector()
    Graph graph = Graph::create(context);
    ivx::Node node = ivx::Node(vxCannyEdgeDetectorNode(graph, _src, threshold, kSize, (useL2 ? VX_NORM_L2 : VX_NORM_L1), _dst) );
    node.setBorder(VX_BORDER_REPLICATE);
    graph.verify();
    graph.process();
#endif

#ifdef VX_VERSION_1_1
    _src.swapHandle();
    _dst.swapHandle();
#endif
    }
    catch(const WrapperError& e)
    {
        VX_DbgThrow(e.what());
    }
    catch(const RuntimeError& e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif // HAVE_OPENVX

void Canny( InputArray _src, OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, bool L2gradient )
{
    CV_INSTRUMENT_REGION()

    const int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    const Size size = _src.size();

    CV_Assert( depth == CV_8U );
    _dst.create(size, CV_8U);

    if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
    {
        // backward compatibility
        aperture_size &= ~CV_CANNY_L2_GRADIENT;
        L2gradient = true;
    }

    if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
        CV_Error(CV_StsBadFlag, "Aperture size should be odd between 3 and 7");

    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);

    CV_OCL_RUN(_dst.isUMat() && (cn == 1 || cn == 3),
               ocl_Canny<false>(_src, UMat(), UMat(), _dst, (float)low_thresh, (float)high_thresh, aperture_size, L2gradient, cn, size))

    Mat src = _src.getMat(), dst = _dst.getMat();

    CV_OVX_RUN(
        false && /* disabling due to accuracy issues */
            src.type() == CV_8UC1 &&
            !src.isSubmatrix() &&
            src.cols >= aperture_size &&
            src.rows >= aperture_size,
        openvx_canny(
            src,
            dst,
            cvFloor(low_thresh),
            cvFloor(high_thresh),
            aperture_size,
            L2gradient ) )

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::canny(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
        return;
#endif

    CV_IPP_RUN_FAST(ipp_Canny(src, Mat(), Mat(), dst, (float)low_thresh, (float)high_thresh, L2gradient, aperture_size))

    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;
        if (high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = src.cols + 2;
    AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

    uchar *map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    // Minimum number of threads should be 1, maximum should not exceed number of CPU's, because of overhead
    int numOfThreads = std::max(1, std::min(getNumThreads(), getNumberOfCPUs()));

    // Make a fallback for pictures with too few rows.
    int grainSize = src.rows / numOfThreads;
    int ksize2 = aperture_size / 2;
    // If Scharr filter: aperture size is 3, ksize2 is 1
    if(aperture_size == -1)
    {
        ksize2 = 1;
    }

    int minGrainSize = 2 * (ksize2 + 1);
    if (grainSize < minGrainSize)
    {
        numOfThreads = std::max(1, src.rows / minGrainSize);
    }

    std::queue<uchar*> borderPeaksParallel;

    parallel_for_(Range(0, src.rows), parallelCanny(src, map, low, high, aperture_size, L2gradient, &borderPeaksParallel), numOfThreads);

#define CANNY_PUSH_SERIAL(d)    *(d) = uchar(2), borderPeaksParallel.push(d)

    // now track the edges (hysteresis thresholding)
    uchar* m;
    while (!borderPeaksParallel.empty())
    {
        m = borderPeaksParallel.front();
        borderPeaksParallel.pop();
        if (!m[-1])         CANNY_PUSH_SERIAL(m - 1);
        if (!m[1])          CANNY_PUSH_SERIAL(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH_SERIAL(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH_SERIAL(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH_SERIAL(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH_SERIAL(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH_SERIAL(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH_SERIAL(m + mapstep + 1);
    }

    parallel_for_(Range(0, dst.rows), finalPass(map, dst, mapstep), dst.total()/(double)(1<<16));
}

void Canny( InputArray _dx, InputArray _dy, OutputArray _dst,
                double low_thresh, double high_thresh,
                bool L2gradient )
{
    CV_INSTRUMENT_REGION()

    CV_Assert(_dx.dims() == 2);
    CV_Assert(_dx.type() == CV_16SC1 || _dx.type() == CV_16SC3);
    CV_Assert(_dy.type() == _dx.type());
    CV_Assert(_dx.sameSize(_dy));

    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);

    const int cn = _dx.channels();
    const Size size = _dx.size();

    CV_OCL_RUN(_dst.isUMat(),
               ocl_Canny<true>(UMat(), _dx.getUMat(), _dy.getUMat(), _dst, (float)low_thresh, (float)high_thresh, 0, L2gradient, cn, size))

    _dst.create(size, CV_8U);
    Mat dst = _dst.getMat();

    Mat dx = _dx.getMat();
    Mat dy = _dy.getMat();

    CV_IPP_RUN_FAST(ipp_Canny(Mat(), dx, dy, dst, (float)low_thresh, (float)high_thresh, L2gradient, 0))

    if (cn > 1)
    {
        dx = dx.clone();
        dy = dy.clone();
    }
    CannyImpl(dx, dy, dst, low_thresh, high_thresh, L2gradient);
}

static void CannyImpl(Mat& dx, Mat& dy, Mat& dst,
    double low_thresh, double high_thresh, bool L2gradient)
{
    const int cn = dx.channels();
    const int cols = dx.cols, rows = dx.rows;

    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;
        if (high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = cols + 2;
    AutoBuffer<uchar> buffer((cols+2)*(rows+2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, cols * rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

#if CV_SIMD128
    bool haveSIMD = hasSIMD128();
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            if (!L2gradient)
            {
                int j = 0, width = cols * cn;
#if CV_SIMD128
                if (haveSIMD)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        v_int16x8 v_dx = v_load((const short*)(_dx + j));
                        v_int16x8 v_dy = v_load((const short*)(_dy + j));

                        v_int32x4 v_dx0, v_dx1, v_dy0, v_dy1;
                        v_expand(v_dx, v_dx0, v_dx1);
                        v_expand(v_dy, v_dy0, v_dy1);

                        v_dx0 = v_reinterpret_as_s32(v_abs(v_dx0));
                        v_dx1 = v_reinterpret_as_s32(v_abs(v_dx1));
                        v_dy0 = v_reinterpret_as_s32(v_abs(v_dy0));
                        v_dy1 = v_reinterpret_as_s32(v_abs(v_dy1));

                        v_store(_norm + j, v_dx0 + v_dy0);
                        v_store(_norm + j + 4, v_dx1 + v_dy1);
                    }
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = cols * cn;
#if CV_SIMD128
                if (haveSIMD)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        v_int16x8 v_dx = v_load((const short*)(_dx + j));
                        v_int16x8 v_dy = v_load((const short*)(_dy + j));

                        v_int16x8 v_dx_dy0, v_dx_dy1;
                        v_zip(v_dx, v_dy, v_dx_dy0, v_dx_dy1);

                        v_int32x4 v_dst0 = v_dotprod(v_dx_dy0, v_dx_dy0);
                        v_int32x4 v_dst1 = v_dotprod(v_dx_dy1, v_dx_dy1);

                        v_store(_norm + j, v_dst0);
                        v_store(_norm + j + 4, v_dst1);
                    }
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }

            if (cn > 1)
            {
                for(int j = 0, jn = 0; j < cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if ((stack_top - stack_bottom) + cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3/2, sz + cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

#define CANNY_SHIFT 15
        const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

        int prev_flag = 0, j = 0;
#if CV_SIMD128
        if (haveSIMD)
        {
            v_int32x4 v_low = v_setall_s32(low);
            v_int8x16 v_one = v_setall_s8(1);

            for (; j <= cols - 16; j += 16)
            {
                v_int32x4 v_m1 = v_load((const int*)(_mag + j));
                v_int32x4 v_m2 = v_load((const int*)(_mag + j + 4));
                v_int32x4 v_m3 = v_load((const int*)(_mag + j + 8));
                v_int32x4 v_m4 = v_load((const int*)(_mag + j + 12));

                v_store((signed char*)(_map + j), v_one);

                v_int32x4 v_cmp1 = v_m1 > v_low;
                v_int32x4 v_cmp2 = v_m2 > v_low;
                v_int32x4 v_cmp3 = v_m3 > v_low;
                v_int32x4 v_cmp4 = v_m4 > v_low;

                v_int16x8 v_cmp80 = v_pack(v_cmp1, v_cmp2);
                v_int16x8 v_cmp81 = v_pack(v_cmp3, v_cmp4);

                v_int8x16 v_cmp = v_pack(v_cmp80, v_cmp81);
                unsigned int mask = v_signmask(v_cmp);

                if (mask)
                {
                    int m, k = j;

                    for (; mask; ++k, mask >>= 1)
                    {
                        if (mask & 0x00000001)
                        {
                            m = _mag[k];
                            int xs = _x[k];
                            int ys = _y[k];
                            int x = std::abs(xs);
                            int y = std::abs(ys) << CANNY_SHIFT;

                            int tg22x = x * TG22;

                            if (y < tg22x)
                            {
                                if (m > _mag[k - 1] && m >= _mag[k + 1]) goto ocv_canny_push_sse;
                            }
                            else
                            {
                                int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                                if (y > tg67x)
                                {
                                    if (m > _mag[k + magstep2] && m >= _mag[k + magstep1]) goto ocv_canny_push_sse;
                                } else
                                {
                                    int s = (xs ^ ys) < 0 ? -1 : 1;
                                    if (m > _mag[k + magstep2 - s] && m > _mag[k + magstep1 + s]) goto ocv_canny_push_sse;
                                }
                            }
                        }

                        prev_flag = 0;
                        continue;

ocv_canny_push_sse:
                        // _map[k-mapstep] is short-circuited at the start because previous thread is
                        // responsible for initializing it.
                        if (!prev_flag && m > high && _map[k-mapstep] != 2)
                        {
                            CANNY_PUSH(_map + k);
                            prev_flag = 1;
                        } else
                            _map[k] = 0;

                    }

                    if (prev_flag && ((k < j+16) || (k < cols && _mag[k] <= high)))
                        prev_flag = 0;
                }
            }
        }
#endif
        for (; j < cols; j++)
        {
            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j-1] && m >= _mag[j+1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
__ocv_canny_push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }

    parallel_for_(Range(0, dst.rows), finalPass(map, dst, mapstep), dst.total()/(double)(1<<16));
}

} // namespace cv

void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(image), dst = cv::cvarrToMat(edges);
    CV_Assert( src.size == dst.size && src.depth() == CV_8U && dst.type() == CV_8U );

    cv::Canny(src, dst, threshold1, threshold2, aperture_size & 255,
              (aperture_size & CV_CANNY_L2_GRADIENT) != 0);
}

/* End of file. */
