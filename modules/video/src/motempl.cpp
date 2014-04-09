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
#include "opencl_kernels.hpp"

#ifdef HAVE_OPENCL

namespace  cv {

static bool ocl_updateMotionHistory( InputArray _silhouette, InputOutputArray _mhi,
                                     float timestamp, float delbound )
{
    ocl::Kernel k("updateMotionHistory", ocl::video::updatemotionhistory_oclsrc);
    if (k.empty())
        return false;

    UMat silh = _silhouette.getUMat(), mhi = _mhi.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(silh), ocl::KernelArg::ReadWrite(mhi),
           timestamp, delbound);

    size_t globalsize[2] = { silh.cols, silh.rows };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::updateMotionHistory( InputArray _silhouette, InputOutputArray _mhi,
                              double timestamp, double duration )
{
    CV_Assert( _silhouette.type() == CV_8UC1 && _mhi.type() == CV_32FC1 );
    CV_Assert( _silhouette.sameSize(_mhi) );

    float ts = (float)timestamp;
    float delbound = (float)(timestamp - duration);

    CV_OCL_RUN(_mhi.isUMat() && _mhi.dims() <= 2,
               ocl_updateMotionHistory(_silhouette, _mhi, ts, delbound))

    Mat silh = _silhouette.getMat(), mhi = _mhi.getMat();
    Size size = silh.size();
#if defined(HAVE_IPP) && !defined(HAVE_IPP_ICV_ONLY)
    int silhstep = (int)silh.step, mhistep = (int)mhi.step;
#endif

    if( silh.isContinuous() && mhi.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
#if defined(HAVE_IPP) && !defined(HAVE_IPP_ICV_ONLY)
        silhstep = (int)silh.total();
        mhistep = (int)mhi.total() * sizeof(Ipp32f);
#endif
    }

#if defined(HAVE_IPP) && !defined(HAVE_IPP_ICV_ONLY)
    IppStatus status = ippiUpdateMotionHistory_8u32f_C1IR((const Ipp8u *)silh.data, silhstep, (Ipp32f *)mhi.data, mhistep,
                                                          ippiSize(size.width, size.height), (Ipp32f)timestamp, (Ipp32f)duration);
    if (status >= 0)
        return;
#endif

#if CV_SSE2
    volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2);
#endif

    for(int y = 0; y < size.height; y++ )
    {
        const uchar* silhData = silh.ptr<uchar>(y);
        float* mhiData = mhi.ptr<float>(y);
        int x = 0;

#if CV_SSE2
        if( useSIMD )
        {
            __m128 ts4 = _mm_set1_ps(ts), db4 = _mm_set1_ps(delbound);
            for( ; x <= size.width - 8; x += 8 )
            {
                __m128i z = _mm_setzero_si128();
                __m128i s = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(silhData + x)), z);
                __m128 s0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(s, z)), s1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s, z));
                __m128 v0 = _mm_loadu_ps(mhiData + x), v1 = _mm_loadu_ps(mhiData + x + 4);
                __m128 fz = _mm_setzero_ps();

                v0 = _mm_and_ps(v0, _mm_cmpge_ps(v0, db4));
                v1 = _mm_and_ps(v1, _mm_cmpge_ps(v1, db4));

                __m128 m0 = _mm_and_ps(_mm_xor_ps(v0, ts4), _mm_cmpneq_ps(s0, fz));
                __m128 m1 = _mm_and_ps(_mm_xor_ps(v1, ts4), _mm_cmpneq_ps(s1, fz));

                v0 = _mm_xor_ps(v0, m0);
                v1 = _mm_xor_ps(v1, m1);

                _mm_storeu_ps(mhiData + x, v0);
                _mm_storeu_ps(mhiData + x + 4, v1);
            }
        }
#endif

        for( ; x < size.width; x++ )
        {
            float val = mhiData[x];
            val = silhData[x] ? ts : val < delbound ? 0 : val;
            mhiData[x] = val;
        }
    }
}


void cv::calcMotionGradient( InputArray _mhi, OutputArray _mask,
                             OutputArray _orientation,
                             double delta1, double delta2,
                             int aperture_size )
{
    static int runcase = 0; runcase++;

    Mat mhi = _mhi.getMat();
    Size size = mhi.size();

    _mask.create(size, CV_8U);
    _orientation.create(size, CV_32F);

    Mat mask = _mask.getMat();
    Mat orient = _orientation.getMat();

    if( aperture_size < 3 || aperture_size > 7 || (aperture_size & 1) == 0 )
        CV_Error( Error::StsOutOfRange, "aperture_size must be 3, 5 or 7" );

    if( delta1 <= 0 || delta2 <= 0 )
        CV_Error( Error::StsOutOfRange, "both delta's must be positive" );

    if( mhi.type() != CV_32FC1 )
        CV_Error( Error::StsUnsupportedFormat,
                 "MHI must be single-channel floating-point images" );

    if( orient.data == mhi.data )
    {
        _orientation.release();
        _orientation.create(size, CV_32F);
        orient = _orientation.getMat();
    }

    if( delta1 > delta2 )
        std::swap(delta1, delta2);

    float gradient_epsilon = 1e-4f * aperture_size * aperture_size;
    float min_delta = (float)delta1;
    float max_delta = (float)delta2;

    Mat dX_min, dY_max;

    // calc Dx and Dy
    Sobel( mhi, dX_min, CV_32F, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE );
    Sobel( mhi, dY_max, CV_32F, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE );

    int x, y;

    if( mhi.isContinuous() && orient.isContinuous() && mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    // calc gradient
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        fastAtan2(dY_max_row, dX_min_row, orient_row, size.width, true);

        // make orientation zero where the gradient is very small
        for( x = 0; x < size.width; x++ )
        {
            float dY = dY_max_row[x];
            float dX = dX_min_row[x];

            if( std::abs(dX) < gradient_epsilon && std::abs(dY) < gradient_epsilon )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
            else
                mask_row[x] = (uchar)1;
        }
    }

    erode( mhi, dX_min, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );
    dilate( mhi, dY_max, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );

    // mask off pixels which have little motion difference in their neighborhood
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        for( x = 0; x < size.width; x++ )
        {
            float d0 = dY_max_row[x] - dX_min_row[x];

            if( mask_row[x] == 0 || d0 < min_delta || max_delta < d0 )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
        }
    }
}

double cv::calcGlobalOrientation( InputArray _orientation, InputArray _mask,
                                  InputArray _mhi, double /*timestamp*/,
                                  double duration )
{
    Mat orient = _orientation.getMat(), mask = _mask.getMat(), mhi = _mhi.getMat();
    Size size = mhi.size();

    CV_Assert( mask.type() == CV_8U && orient.type() == CV_32F && mhi.type() == CV_32F );
    CV_Assert( mask.size() == size && orient.size() == size );
    CV_Assert( duration > 0 );

    int histSize = 12;
    float _ranges[] = { 0.f, 360.f };
    const float* ranges = _ranges;
    Mat hist;

    calcHist(&orient, 1, 0, mask, hist, 1, &histSize, &ranges);

    // find the maximum index (the dominant orientation)
    Point baseOrientPt;
    minMaxLoc(hist, 0, 0, 0, &baseOrientPt);
    float fbaseOrient = (baseOrientPt.x + baseOrientPt.y)*360.f/histSize;

    // override timestamp with the maximum value in MHI
    double timestamp = 0;
    minMaxLoc( mhi, 0, &timestamp, 0, 0, mask );

    // find the shift relative to the dominant orientation as weighted sum of relative angles
    float a = (float)(254. / 255. / duration);
    float b = (float)(1. - timestamp * a);
    float delbound = (float)(timestamp - duration);

    if( mhi.isContinuous() && mask.isContinuous() && orient.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    /*
     a = 254/(255*dt)
     b = 1 - t*a = 1 - 254*t/(255*dur) =
     (255*dt - 254*t)/(255*dt) =
     (dt - (t - dt)*254)/(255*dt);
     --------------------------------------------------------
     ax + b = 254*x/(255*dt) + (dt - (t - dt)*254)/(255*dt) =
     (254*x + dt - (t - dt)*254)/(255*dt) =
     ((x - (t - dt))*254 + dt)/(255*dt) =
     (((x - (t - dt))/dt)*254 + 1)/255 = (((x - low_time)/dt)*254 + 1)/255
     */
    float shiftOrient = 0, shiftWeight = 0;
    for( int y = 0; y < size.height; y++ )
    {
        const float* mhiptr = mhi.ptr<float>(y);
        const float* oriptr = orient.ptr<float>(y);
        const uchar* maskptr = mask.ptr<uchar>(y);

        for( int x = 0; x < size.width; x++ )
        {
            if( maskptr[x] != 0 && mhiptr[x] > delbound )
            {
                /*
                 orient in 0..360, base_orient in 0..360
                 -> (rel_angle = orient - base_orient) in -360..360.
                 rel_angle is translated to -180..180
                 */
                float weight = mhiptr[x] * a + b;
                float relAngle = oriptr[x] - fbaseOrient;

                relAngle += (relAngle < -180 ? 360 : 0);
                relAngle += (relAngle > 180 ? -360 : 0);

                if( fabs(relAngle) < 45 )
                {
                    shiftOrient += weight * relAngle;
                    shiftWeight += weight;
                }
            }
        }
    }

    // add the dominant orientation and the relative shift
    if( shiftWeight == 0 )
        shiftWeight = 0.01f;

    fbaseOrient += shiftOrient / shiftWeight;
    fbaseOrient -= (fbaseOrient < 360 ? 0 : 360);
    fbaseOrient += (fbaseOrient >= 0 ? 0 : 360);

    return fbaseOrient;
}


void cv::segmentMotion(InputArray _mhi, OutputArray _segmask,
                       std::vector<Rect>& boundingRects,
                       double timestamp, double segThresh)
{
    Mat mhi = _mhi.getMat();

    _segmask.create(mhi.size(), CV_32F);
    Mat segmask = _segmask.getMat();
    segmask = Scalar::all(0);

    CV_Assert( mhi.type() == CV_32F );
    CV_Assert( segThresh >= 0 );

    Mat mask = Mat::zeros( mhi.rows + 2, mhi.cols + 2, CV_8UC1 );

    int x, y;

    // protect zero mhi pixels from floodfill.
    for( y = 0; y < mhi.rows; y++ )
    {
        const float* mhiptr = mhi.ptr<float>(y);
        uchar* maskptr = mask.ptr<uchar>(y+1) + 1;

        for( x = 0; x < mhi.cols; x++ )
        {
            if( mhiptr[x] == 0 )
                maskptr[x] = 1;
        }
    }

    float ts = (float)timestamp;
    float comp_idx = 1.f;

    for( y = 0; y < mhi.rows; y++ )
    {
        float* mhiptr = mhi.ptr<float>(y);
        uchar* maskptr = mask.ptr<uchar>(y+1) + 1;

        for( x = 0; x < mhi.cols; x++ )
        {
            if( mhiptr[x] == ts && maskptr[x] == 0 )
            {
                Rect cc;
                floodFill( mhi, mask, Point(x,y), Scalar::all(0),
                           &cc, Scalar::all(segThresh), Scalar::all(segThresh),
                           FLOODFILL_MASK_ONLY + 2*256 + 4 );

                for( int y1 = 0; y1 < cc.height; y1++ )
                {
                    float* segmaskptr = segmask.ptr<float>(cc.y + y1) + cc.x;
                    uchar* maskptr1 = mask.ptr<uchar>(cc.y + y1 + 1) + cc.x + 1;

                    for( int x1 = 0; x1 < cc.width; x1++ )
                    {
                        if( maskptr1[x1] > 1 )
                        {
                            maskptr1[x1] = 1;
                            segmaskptr[x1] = comp_idx;
                        }
                    }
                }
                comp_idx += 1.f;
                boundingRects.push_back(cc);
            }
        }
    }
}


/* End of file. */
