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


#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
#define USE_IPP_CANNY 1
#else
#undef USE_IPP_CANNY
#endif


namespace cv
{

#ifdef USE_IPP_CANNY
static bool ippCanny(const Mat& _src, Mat& _dst, float low,  float high)
{
    int size = 0, size1 = 0;
    IppiSize roi = { _src.cols, _src.rows };

    if (ippiFilterSobelNegVertGetBufferSize_8u16s_C1R(roi, ippMskSize3x3, &size) < 0)
        return false;
    if (ippiFilterSobelHorizGetBufferSize_8u16s_C1R(roi, ippMskSize3x3, &size1) < 0)
        return false;
    size = std::max(size, size1);

    if (ippiCannyGetSize(roi, &size1) < 0)
        return false;
    size = std::max(size, size1);

    AutoBuffer<uchar> buf(size + 64);
    uchar* buffer = alignPtr((uchar*)buf, 32);

    Mat _dx(_src.rows, _src.cols, CV_16S);
    if( ippiFilterSobelNegVertBorder_8u16s_C1R(_src.ptr(), (int)_src.step,
                    _dx.ptr<short>(), (int)_dx.step, roi,
                    ippMskSize3x3, ippBorderRepl, 0, buffer) < 0 )
        return false;

    Mat _dy(_src.rows, _src.cols, CV_16S);
    if( ippiFilterSobelHorizBorder_8u16s_C1R(_src.ptr(), (int)_src.step,
                    _dy.ptr<short>(), (int)_dy.step, roi,
                    ippMskSize3x3, ippBorderRepl, 0, buffer) < 0 )
        return false;

    if( ippiCanny_16s8u_C1R(_dx.ptr<short>(), (int)_dx.step,
                               _dy.ptr<short>(), (int)_dy.step,
                              _dst.ptr(), (int)_dst.step, roi, low, high, buffer) < 0 )
        return false;
    return true;
}
#endif

#ifdef HAVE_OPENCL

static bool ocl_Canny(InputArray _src, OutputArray _dst, float low_thresh, float high_thresh,
                      int aperture_size, bool L2gradient, int cn, const Size & size)
{
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

    if (aperture_size == 3 && !_src.isSubmatrix())
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

        size_t globalsize[2] = { size.width, size.height },
                localsize[2] = { lSizeX, lSizeY };

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
        Sobel(_src, dx, CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
        Sobel(_src, dy, CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

        ocl::Kernel without_sobel("stage1_without_sobel", ocl::imgproc::canny_oclsrc,
                                    format("-D WITHOUT_SOBEL -D cn=%d -D GRP_SIZEX=%d -D GRP_SIZEY=%d%s",
                                           cn, lSizeX, lSizeY, L2gradient ? " -D L2GRAD" : ""));
        if (without_sobel.empty())
            return false;

        map.create(size, CV_32S);
        without_sobel.args(ocl::KernelArg::ReadOnlyNoSize(dx), ocl::KernelArg::ReadOnlyNoSize(dy),
                           ocl::KernelArg::WriteOnly(map),
                           low, high);

        size_t globalsize[2] = { size.width, size.height },
                localsize[2] = { lSizeX, lSizeY };

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

    size_t globalsize[2] = { size.width, (size.height + PIX_PER_WI - 1) / PIX_PER_WI }, localsize[2] = { lSizeX, sizey };

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

#ifdef HAVE_TBB

// Queue with peaks that will processed serially.
static tbb::concurrent_queue<uchar*> borderPeaks;

class tbbCanny
{
public:
    tbbCanny(const Range _boundaries, const Mat& _src, uchar* _map, int _low,
            int _high, int _aperture_size, bool _L2gradient)
        : boundaries(_boundaries), src(_src), map(_map), low(_low), high(_high),
          aperture_size(_aperture_size), L2gradient(_L2gradient)
    {}

    // This parallel version of Canny algorithm splits the src image in threadsNumber horizontal slices.
    // The first row of each slice contains the last row of the previous slice and
    // the last row of each slice contains the first row of the next slice
    // so that each slice is independent and no mutexes are required.
    void operator()() const
    {
#if CV_SSE2
        bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

        const int type = src.type(), cn = CV_MAT_CN(type);

        Mat dx, dy;

        ptrdiff_t mapstep = src.cols + 2;

        // In sobel transform we calculate ksize2 extra lines for the first and last rows of each slice
        // because IPPDerivSobel expects only isolated ROIs, in contrast with the opencv version which
        // uses the pixels outside of the ROI to form a border.
        uchar ksize2 = aperture_size / 2;

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
#if CV_SSE2
                if (haveSSE2)
                {
                    __m128i v_zero = _mm_setzero_si128();
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));
                        v_dx = _mm_max_epi16(v_dx, _mm_sub_epi16(v_zero, v_dx));
                        v_dy = _mm_max_epi16(v_dy, _mm_sub_epi16(v_zero, v_dy));

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx, v_zero), _mm_unpacklo_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx, v_zero), _mm_unpackhi_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    vst1q_s32(_norm + j, vaddq_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_dx))),
                                                   vabsq_s32(vmovl_s16(vget_low_s16(v_dy)))));
                    vst1q_s32(_norm + j + 4, vaddq_s32(vabsq_s32(vmovl_s16(vget_high_s16(v_dx))),
                                                       vabsq_s32(vmovl_s16(vget_high_s16(v_dy)))));
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = src.cols * cn;
#if CV_SSE2
                if (haveSSE2)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                        __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                        __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                    int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j, v_dst);

                    v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                    v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j + 4, v_dst);
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

            int prev_flag = 0;
            bool canny_push = false;
            for (int j = 0; j < src.cols; j++)
            {
                #define CANNY_SHIFT 15
                const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

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
                        if (m > _mag[j-1] && m >= _mag[j+1]) canny_push = true;
                    }
                    else
                    {
                        int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                        if (y > tg67x)
                        {
                            if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) canny_push = true;
                        }
                        else
                        {
                            int s = (xs ^ ys) < 0 ? -1 : 1;
                            if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) canny_push = true;
                        }
                    }
                }
                if (!canny_push)
                {
                    prev_flag = 0;
                    _map[j] = uchar(1);
                    continue;
                }
                else
                {
                    // _map[j-mapstep] is short-circuited at the start because previous thread is
                    // responsible for initializing it.
                    if (!prev_flag && m > high && (i <= boundaries.start+1 || _map[j-mapstep] != 2) )
                    {
                        CANNY_PUSH(_map + j);
                        prev_flag = 1;
                    }
                    else
                        _map[j] = 0;

                    canny_push = false;
                }
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
                borderPeaks.push(m);
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
    }

private:
    const Range boundaries;
    const Mat& src;
    uchar* map;
    int low;
    int high;
    int aperture_size;
    bool L2gradient;
};

#endif

} // namespace cv

void cv::Canny( InputArray _src, OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, bool L2gradient )
{
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
        CV_Error(CV_StsBadFlag, "Aperture size should be odd");

    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);

    CV_OCL_RUN(_dst.isUMat() && (cn == 1 || cn == 3),
               ocl_Canny(_src, _dst, (float)low_thresh, (float)high_thresh, aperture_size, L2gradient, cn, size))

    Mat src = _src.getMat(), dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::canny(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
        return;
#endif

#ifdef USE_IPP_CANNY
    CV_IPP_CHECK()
    {
        if( aperture_size == 3 && !L2gradient && 1 == cn )
        {
            if (ippCanny(src, dst, (float)low_thresh, (float)high_thresh))
            {
                CV_IMPL_ADD(CV_IMPL_IPP);
                return;
            }
            setIppErrorStatus();
        }
    }
#endif

#ifdef HAVE_TBB

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
AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2));

uchar* map = (uchar*)buffer;
memset(map, 1, mapstep);
memset(map + mapstep*(src.rows + 1), 1, mapstep);

int threadsNumber = tbb::task_scheduler_init::default_num_threads();
int grainSize = src.rows / threadsNumber;

// Make a fallback for pictures with too few rows.
uchar ksize2 = aperture_size / 2;
int minGrainSize = 1 + ksize2;
int maxGrainSize = src.rows - 2 - 2*ksize2;
if ( !( minGrainSize <= grainSize && grainSize <= maxGrainSize ) )
{
    threadsNumber = 1;
    grainSize = src.rows;
}

tbb::task_group g;

for (int i = 0; i < threadsNumber; ++i)
{
    if (i < threadsNumber - 1)
        g.run(tbbCanny(Range(i * grainSize, (i + 1) * grainSize), src, map, low, high, aperture_size, L2gradient));
    else
        g.run(tbbCanny(Range(i * grainSize, src.rows), src, map, low, high, aperture_size, L2gradient));
}

g.wait();

#define CANNY_PUSH_SERIAL(d)    *(d) = uchar(2), borderPeaks.push(d)

// now track the edges (hysteresis thresholding)
uchar* m;
while (borderPeaks.try_pop(m))
{
    if (!m[-1])         CANNY_PUSH_SERIAL(m - 1);
    if (!m[1])          CANNY_PUSH_SERIAL(m + 1);
    if (!m[-mapstep-1]) CANNY_PUSH_SERIAL(m - mapstep - 1);
    if (!m[-mapstep])   CANNY_PUSH_SERIAL(m - mapstep);
    if (!m[-mapstep+1]) CANNY_PUSH_SERIAL(m - mapstep + 1);
    if (!m[mapstep-1])  CANNY_PUSH_SERIAL(m + mapstep - 1);
    if (!m[mapstep])    CANNY_PUSH_SERIAL(m + mapstep);
    if (!m[mapstep+1])  CANNY_PUSH_SERIAL(m + mapstep + 1);
}

#else

    Mat dx(src.rows, src.cols, CV_16SC(cn));
    Mat dy(src.rows, src.cols, CV_16SC(cn));

    Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE);
    Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE);

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

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
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

#if CV_SSE2
    bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            if (!L2gradient)
            {
                int j = 0, width = src.cols * cn;
#if CV_SSE2
                if (haveSSE2)
                {
                    __m128i v_zero = _mm_setzero_si128();
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));
                        v_dx = _mm_max_epi16(v_dx, _mm_sub_epi16(v_zero, v_dx));
                        v_dy = _mm_max_epi16(v_dy, _mm_sub_epi16(v_zero, v_dy));

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx, v_zero), _mm_unpacklo_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx, v_zero), _mm_unpackhi_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    vst1q_s32(_norm + j, vaddq_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_dx))),
                                                   vabsq_s32(vmovl_s16(vget_low_s16(v_dy)))));
                    vst1q_s32(_norm + j + 4, vaddq_s32(vabsq_s32(vmovl_s16(vget_high_s16(v_dx))),
                                                       vabsq_s32(vmovl_s16(vget_high_s16(v_dy)))));
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = src.cols * cn;
#if CV_SSE2
                if (haveSSE2)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                        __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                        __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                    int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j, v_dst);

                    v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                    v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j + 4, v_dst);
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
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3/2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

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

#endif

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(image), dst = cv::cvarrToMat(edges);
    CV_Assert( src.size == dst.size && src.depth() == CV_8U && dst.type() == CV_8U );

    cv::Canny(src, dst, threshold1, threshold2, aperture_size & 255,
              (aperture_size & CV_CANNY_L2_GRADIENT) != 0);
}

/* End of file. */
