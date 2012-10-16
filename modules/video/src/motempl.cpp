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


/* motion templates */
CV_IMPL void
cvUpdateMotionHistory( const void* silhouette, void* mhimg,
                       double timestamp, double mhi_duration )
{
    CvMat  silhstub, *silh = cvGetMat(silhouette, &silhstub);
    CvMat  mhistub, *mhi = cvGetMat(mhimg, &mhistub);

    if( !CV_IS_MASK_ARR( silh ))
        CV_Error( CV_StsBadMask, "" );

    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( mhi, silh ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    CvSize size = cvGetMatSize( mhi );

    if( CV_IS_MAT_CONT( mhi->type & silh->type ))
    {
        size.width *= size.height;
        size.height = 1;
    }

    float ts = (float)timestamp;
    float delbound = (float)(timestamp - mhi_duration);
    int x, y;
#if CV_SSE2
    volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2);
#endif

    for( y = 0; y < size.height; y++ )
    {
        const uchar* silhData = silh->data.ptr + silh->step*y;
        float* mhiData = (float*)(mhi->data.ptr + mhi->step*y);
        x = 0;

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


CV_IMPL void
cvCalcMotionGradient( const CvArr* mhiimg, CvArr* maskimg,
                      CvArr* orientation,
                      double delta1, double delta2,
                      int aperture_size )
{
    cv::Ptr<CvMat> dX_min, dY_max;

    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(maskimg, &maskstub);
    CvMat  orientstub, *orient = cvGetMat(orientation, &orientstub);
    CvMat  dX_min_row, dY_max_row, orient_row, mask_row;
    CvSize size;
    int x, y;

    float  gradient_epsilon = 1e-4f * aperture_size * aperture_size;
    float  min_delta, max_delta;

    if( !CV_IS_MASK_ARR( mask ))
        CV_Error( CV_StsBadMask, "" );

    if( aperture_size < 3 || aperture_size > 7 || (aperture_size & 1) == 0 )
        CV_Error( CV_StsOutOfRange, "aperture_size must be 3, 5 or 7" );

    if( delta1 <= 0 || delta2 <= 0 )
        CV_Error( CV_StsOutOfRange, "both delta's must be positive" );

    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( orient->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat,
        "MHI and orientation must be single-channel floating-point images" );

    if( !CV_ARE_SIZES_EQ( mhi, mask ) || !CV_ARE_SIZES_EQ( orient, mhi ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( orient->data.ptr == mhi->data.ptr )
        CV_Error( CV_StsInplaceNotSupported, "orientation image must be different from MHI" );

    if( delta1 > delta2 )
    {
        double t;
        CV_SWAP( delta1, delta2, t );
    }

    size = cvGetMatSize( mhi );
    min_delta = (float)delta1;
    max_delta = (float)delta2;
    dX_min = cvCreateMat( mhi->rows, mhi->cols, CV_32F );
    dY_max = cvCreateMat( mhi->rows, mhi->cols, CV_32F );

    // calc Dx and Dy
    cvSobel( mhi, dX_min, 1, 0, aperture_size );
    cvSobel( mhi, dY_max, 0, 1, aperture_size );
    cvGetRow( dX_min, &dX_min_row, 0 );
    cvGetRow( dY_max, &dY_max_row, 0 );
    cvGetRow( orient, &orient_row, 0 );
    cvGetRow( mask, &mask_row, 0 );

    // calc gradient
    for( y = 0; y < size.height; y++ )
    {
        dX_min_row.data.ptr = dX_min->data.ptr + y*dX_min->step;
        dY_max_row.data.ptr = dY_max->data.ptr + y*dY_max->step;
        orient_row.data.ptr = orient->data.ptr + y*orient->step;
        mask_row.data.ptr = mask->data.ptr + y*mask->step;
        cvCartToPolar( &dX_min_row, &dY_max_row, 0, &orient_row, 1 );

        // make orientation zero where the gradient is very small
        for( x = 0; x < size.width; x++ )
        {
            float dY = dY_max_row.data.fl[x];
            float dX = dX_min_row.data.fl[x];

            if( fabs(dX) < gradient_epsilon && fabs(dY) < gradient_epsilon )
            {
                mask_row.data.ptr[x] = 0;
                orient_row.data.i[x] = 0;
            }
            else
                mask_row.data.ptr[x] = 1;
        }
    }

    cvErode( mhi, dX_min, 0, (aperture_size-1)/2);
    cvDilate( mhi, dY_max, 0, (aperture_size-1)/2);

    // mask off pixels which have little motion difference in their neighborhood
    for( y = 0; y < size.height; y++ )
    {
        dX_min_row.data.ptr = dX_min->data.ptr + y*dX_min->step;
        dY_max_row.data.ptr = dY_max->data.ptr + y*dY_max->step;
        mask_row.data.ptr = mask->data.ptr + y*mask->step;
        orient_row.data.ptr = orient->data.ptr + y*orient->step;

        for( x = 0; x < size.width; x++ )
        {
            float d0 = dY_max_row.data.fl[x] - dX_min_row.data.fl[x];

            if( mask_row.data.ptr[x] == 0 || d0 < min_delta || max_delta < d0 )
            {
                mask_row.data.ptr[x] = 0;
                orient_row.data.i[x] = 0;
            }
        }
    }
}


CV_IMPL double
cvCalcGlobalOrientation( const void* orientation, const void* maskimg, const void* mhiimg,
                         double curr_mhi_timestamp, double mhi_duration )
{
    int hist_size = 12;
    cv::Ptr<CvHistogram> hist;

    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(maskimg, &maskstub);
    CvMat  orientstub, *orient = cvGetMat(orientation, &orientstub);
    void*  _orient;
    float _ranges[] = { 0, 360 };
    float* ranges = _ranges;
    int base_orient;
    float shift_orient = 0, shift_weight = 0;
    float a, b, fbase_orient;
    float delbound;
    CvMat mhi_row, mask_row, orient_row;
    int x, y, mhi_rows, mhi_cols;

    if( !CV_IS_MASK_ARR( mask ))
        CV_Error( CV_StsBadMask, "" );

    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( orient->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat,
        "MHI and orientation must be single-channel floating-point images" );

    if( !CV_ARE_SIZES_EQ( mhi, mask ) || !CV_ARE_SIZES_EQ( orient, mhi ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( mhi_duration <= 0 )
        CV_Error( CV_StsOutOfRange, "MHI duration must be positive" );

    if( orient->data.ptr == mhi->data.ptr )
        CV_Error( CV_StsInplaceNotSupported, "orientation image must be different from MHI" );

    // calculate histogram of different orientation values
    hist = cvCreateHist( 1, &hist_size, CV_HIST_ARRAY, &ranges );
    _orient = orient;
    cvCalcArrHist( &_orient, hist, 0, mask );

    // find the maximum index (the dominant orientation)
    cvGetMinMaxHistValue( hist, 0, 0, 0, &base_orient );
    fbase_orient = base_orient*360.f/hist_size;

    // override timestamp with the maximum value in MHI
    cvMinMaxLoc( mhi, 0, &curr_mhi_timestamp, 0, 0, mask );

    // find the shift relative to the dominant orientation as weighted sum of relative angles
    a = (float)(254. / 255. / mhi_duration);
    b = (float)(1. - curr_mhi_timestamp * a);
    delbound = (float)(curr_mhi_timestamp - mhi_duration);
    mhi_rows = mhi->rows;
    mhi_cols = mhi->cols;

    if( CV_IS_MAT_CONT( mhi->type & mask->type & orient->type ))
    {
        mhi_cols *= mhi_rows;
        mhi_rows = 1;
    }

    cvGetRow( mhi, &mhi_row, 0 );
    cvGetRow( mask, &mask_row, 0 );
    cvGetRow( orient, &orient_row, 0 );

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
    for( y = 0; y < mhi_rows; y++ )
    {
        mhi_row.data.ptr = mhi->data.ptr + mhi->step*y;
        mask_row.data.ptr = mask->data.ptr + mask->step*y;
        orient_row.data.ptr = orient->data.ptr + orient->step*y;

        for( x = 0; x < mhi_cols; x++ )
            if( mask_row.data.ptr[x] != 0 && mhi_row.data.fl[x] > delbound )
            {
                /*
                   orient in 0..360, base_orient in 0..360
                   -> (rel_angle = orient - base_orient) in -360..360.
                   rel_angle is translated to -180..180
                 */
                float weight = mhi_row.data.fl[x] * a + b;
                float rel_angle = orient_row.data.fl[x] - fbase_orient;

                rel_angle += (rel_angle < -180 ? 360 : 0);
                rel_angle += (rel_angle > 180 ? -360 : 0);

                if( fabs(rel_angle) < 45 )
                {
                    shift_orient += weight * rel_angle;
                    shift_weight += weight;
                }
            }
    }

    // add the dominant orientation and the relative shift
    if( shift_weight == 0 )
        shift_weight = 0.01f;

    fbase_orient += shift_orient / shift_weight;
    fbase_orient -= (fbase_orient < 360 ? 0 : 360);
    fbase_orient += (fbase_orient >= 0 ? 0 : 360);

    return fbase_orient;
}


CV_IMPL CvSeq*
cvSegmentMotion( const CvArr* mhiimg, CvArr* segmask, CvMemStorage* storage,
                 double timestamp, double seg_thresh )
{
    CvSeq* components = 0;
    cv::Ptr<CvMat> mask8u;

    CvMat  mhistub, *mhi = cvGetMat(mhiimg, &mhistub);
    CvMat  maskstub, *mask = cvGetMat(segmask, &maskstub);
    Cv32suf v, comp_idx;
    int stub_val, ts;
    int x, y;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL memory storage" );

    mhi = cvGetMat( mhi, &mhistub );
    mask = cvGetMat( mask, &maskstub );

    if( CV_MAT_TYPE( mhi->type ) != CV_32FC1 || CV_MAT_TYPE( mask->type ) != CV_32FC1 )
        CV_Error( CV_BadDepth, "Both MHI and the destination mask" );

    if( !CV_ARE_SIZES_EQ( mhi, mask ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    mask8u = cvCreateMat( mhi->rows + 2, mhi->cols + 2, CV_8UC1 );
    cvZero( mask8u );
    cvZero( mask );
    components = cvCreateSeq( CV_SEQ_KIND_GENERIC, sizeof(CvSeq),
                              sizeof(CvConnectedComp), storage );

    v.f = (float)timestamp; ts = v.i;
    v.f = FLT_MAX*0.1f; stub_val = v.i;
    comp_idx.f = 1;

    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == 0 )
                mhi_row[x] = stub_val;
        }
    }

    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        uchar* mask8u_row = mask8u->data.ptr + (y+1)*mask8u->step + 1;

        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == ts && mask8u_row[x] == 0 )
            {
                CvConnectedComp comp;
                int x1, y1;
                CvScalar _seg_thresh = cvRealScalar(seg_thresh);
                CvPoint seed = cvPoint(x,y);

                cvFloodFill( mhi, seed, cvRealScalar(0), _seg_thresh, _seg_thresh,
                            &comp, CV_FLOODFILL_MASK_ONLY + 2*256 + 4, mask8u );

                for( y1 = 0; y1 < comp.rect.height; y1++ )
                {
                    int* mask_row1 = (int*)(mask->data.ptr +
                                    (comp.rect.y + y1)*mask->step) + comp.rect.x;
                    uchar* mask8u_row1 = mask8u->data.ptr +
                                    (comp.rect.y + y1+1)*mask8u->step + comp.rect.x+1;

                    for( x1 = 0; x1 < comp.rect.width; x1++ )
                    {
                        if( mask8u_row1[x1] > 1 )
                        {
                            mask8u_row1[x1] = 1;
                            mask_row1[x1] = comp_idx.i;
                        }
                    }
                }
                comp_idx.f++;
                cvSeqPush( components, &comp );
            }
        }
    }

    for( y = 0; y < mhi->rows; y++ )
    {
        int* mhi_row = (int*)(mhi->data.ptr + y*mhi->step);
        for( x = 0; x < mhi->cols; x++ )
        {
            if( mhi_row[x] == stub_val )
                mhi_row[x] = 0;
        }
    }

    return components;
}


void cv::updateMotionHistory( InputArray _silhouette, InputOutputArray _mhi,
                              double timestamp, double duration )
{
    Mat silhouette = _silhouette.getMat();
    CvMat c_silhouette = silhouette, c_mhi = _mhi.getMat();
    cvUpdateMotionHistory( &c_silhouette, &c_mhi, timestamp, duration );
}

void cv::calcMotionGradient( InputArray _mhi, OutputArray _mask,
                             OutputArray _orientation,
                             double delta1, double delta2,
                             int aperture_size )
{
    Mat mhi = _mhi.getMat();
    _mask.create(mhi.size(), CV_8U);
    _orientation.create(mhi.size(), CV_32F);
    CvMat c_mhi = mhi, c_mask = _mask.getMat(), c_orientation = _orientation.getMat();
    cvCalcMotionGradient(&c_mhi, &c_mask, &c_orientation, delta1, delta2, aperture_size);
}

double cv::calcGlobalOrientation( InputArray _orientation, InputArray _mask,
                                  InputArray _mhi, double timestamp,
                                  double duration )
{
    Mat orientation = _orientation.getMat(), mask = _mask.getMat(), mhi = _mhi.getMat();
    CvMat c_orientation = orientation, c_mask = mask, c_mhi = mhi;
    return cvCalcGlobalOrientation(&c_orientation, &c_mask, &c_mhi, timestamp, duration);
}

void cv::segmentMotion(InputArray _mhi, OutputArray _segmask,
                       vector<Rect>& boundingRects,
                       double timestamp, double segThresh)
{
    Mat mhi = _mhi.getMat();
    _segmask.create(mhi.size(), CV_32F);
    CvMat c_mhi = mhi, c_segmask = _segmask.getMat();
    Ptr<CvMemStorage> storage = cvCreateMemStorage();
    Seq<CvConnectedComp> comps = cvSegmentMotion(&c_mhi, &c_segmask, storage, timestamp, segThresh);
    Seq<CvConnectedComp>::const_iterator it(comps);
    size_t i, ncomps = comps.size();
    boundingRects.resize(ncomps);
    for( i = 0; i < ncomps; i++, ++it)
        boundingRects[i] = (*it).rect;
}

/* End of file. */
