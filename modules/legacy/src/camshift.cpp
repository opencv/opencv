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
#include "opencv2/video/tracking_c.h"

CvCamShiftTracker::CvCamShiftTracker()
{
    int i;

    memset( &m_box, 0, sizeof(m_box));
    memset( &m_comp, 0, sizeof(m_comp));
    memset( m_color_planes, 0, sizeof(m_color_planes));
    m_threshold = 0;

    for( i = 0; i < CV_MAX_DIM; i++ )
    {
        m_min_ch_val[i] = 0;
        m_max_ch_val[i] = 255;
        m_hist_ranges[i] = m_hist_ranges_data[i];
        m_hist_ranges[i][0] = 0.f;
        m_hist_ranges[i][1] = 256.f;
    }

    m_hist = 0;
    m_back_project = 0;
    m_temp = 0;
    m_mask = 0;
}


CvCamShiftTracker::~CvCamShiftTracker()
{
    int i;

    cvReleaseHist( &m_hist );
    for( i = 0; i < CV_MAX_DIM; i++ )
        cvReleaseImage( &m_color_planes[i] );
    cvReleaseImage( &m_back_project );
    cvReleaseImage( &m_temp );
    cvReleaseImage( &m_mask );
}


void
CvCamShiftTracker::color_transform( const IplImage* image )
{
    CvSize size = cvGetSize(image);
    uchar* color_data = 0, *mask = 0;
    uchar* planes[CV_MAX_DIM];
    int x, color_step = 0, plane_step = 0, mask_step;
    int dims[CV_MAX_DIM];
    int i, n = get_hist_dims(dims);

    assert( image->nChannels == 3 && m_hist != 0 );

    if( !m_temp || !m_mask || !m_color_planes[0] || !m_color_planes[n-1] || !m_back_project ||
        m_temp->width != size.width || m_temp->height != size.height ||
        m_temp->nChannels != 3 )
    {
        cvReleaseImage( &m_temp );
        m_temp = cvCreateImage( size, IPL_DEPTH_8U, 3 );
        cvReleaseImage( &m_mask );
        m_mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );
        cvReleaseImage( &m_back_project );
        m_back_project = cvCreateImage( size, IPL_DEPTH_8U, 1 );
        for( i = 0; i < CV_MAX_DIM; i++ )
        {
            cvReleaseImage( &m_color_planes[i] );
            if( i < n )
                m_color_planes[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
        }
    }

    cvCvtColor( image, m_temp, CV_BGR2HSV );
    cvGetRawData( m_temp, &color_data, &color_step, &size );
    cvGetRawData( m_mask, &mask, &mask_step, &size );

    for( i = 0; i < n; i++ )
        cvGetRawData( m_color_planes[i], &planes[i], &plane_step, &size );

    for( ; size.height--; color_data += color_step, mask += mask_step )
    {
        for( x = 0; x < size.width; x++ )
        {
            int val0 = color_data[x*3];
            int val1 = color_data[x*3+1];
            int val2 = color_data[x*3+2];
            if( m_min_ch_val[0] <= val0 && val0 <= m_max_ch_val[0] &&
                m_min_ch_val[1] <= val1 && val1 <= m_max_ch_val[1] &&
                m_min_ch_val[2] <= val2 && val2 <= m_max_ch_val[2] )
            {
                // hue is written to the 0-th plane, saturation - to the 1-st one,
                // so 1d histogram will automagically correspond to hue-based tracking,
                // 2d histogram - to saturation-based tracking.
                planes[0][x] = (uchar)val0;
                if( n > 1 )
                    planes[1][x] = (uchar)val1;
                if( n > 2 )
                    planes[2][x] = (uchar)val2;

                mask[x] = (uchar)255;
            }
            else
            {
                planes[0][x] = 0;
                if( n > 1 )
                    planes[1][x] = 0;
                if( n > 2 )
                    planes[2][x] = 0;
                mask[x] = 0;
            }
        }
        for( i = 0; i < n; i++ )
            planes[i] += plane_step;
    }
}


bool
CvCamShiftTracker::update_histogram( const IplImage* cur_frame )
{
    float max_val = 0;
    int i, dims;

    if( m_comp.rect.width == 0 || m_comp.rect.height == 0 ||
        m_hist == 0 )
    {
        assert(0);
        return false;
    }

    color_transform(cur_frame);

    dims = cvGetDims( m_hist->bins );
    for( i = 0; i < dims; i++ )
        cvSetImageROI( m_color_planes[i], m_comp.rect );
    cvSetImageROI( m_mask, m_comp.rect );

    cvSetHistBinRanges( m_hist, m_hist_ranges, 1 );
    cvCalcHist( m_color_planes, m_hist, 0, m_mask );

    for( i = 0; i < dims; i++ )
        cvSetImageROI( m_color_planes[i], m_comp.rect );

    for( i = 0; i < dims; i++ )
        cvResetImageROI( m_color_planes[i] );
    cvResetImageROI( m_mask );

    cvGetMinMaxHistValue( m_hist, 0, &max_val );
    cvScale( m_hist->bins, m_hist->bins, max_val ? 255. / max_val : 0. );

    return max_val != 0;
}


void
CvCamShiftTracker::reset_histogram()
{
    if( m_hist )
        cvClearHist( m_hist );
}


bool
CvCamShiftTracker::track_object( const IplImage* cur_frame )
{
    CvRect rect;
    CvSize bp_size;

    union
    {
        void** arr;
        IplImage** img;
    } u;

    if( m_comp.rect.width == 0 || m_comp.rect.height == 0 ||
        m_hist == 0 )
    {
        return false;
    }

    color_transform( cur_frame );
    u.img = m_color_planes;
    cvCalcArrBackProject( u.arr, m_back_project, m_hist );
    cvAnd( m_back_project, m_mask, m_back_project );

    rect = m_comp.rect;
    bp_size = cvGetSize( m_back_project );
    if( rect.x < 0 )
        rect.x = 0;
    if( rect.x + rect.width > bp_size.width )
        rect.width = bp_size.width - rect.x;
    if( rect.y < 0 )
        rect.y = 0;
    if( rect.y + rect.height > bp_size.height )
        rect.height = bp_size.height - rect.y;

    cvCamShift( m_back_project, rect,
                cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ),
                &m_comp, &m_box );

    if( m_comp.rect.width == 0 || m_comp.rect.height == 0 )
        m_comp.rect = rect; // do not allow tracker to loose the object

    return m_comp.rect.width != 0 && m_comp.rect.height != 0;
}


bool
CvCamShiftTracker::set_hist_dims( int c_dims, int *dims )
{
    if( (unsigned)(c_dims-1) >= (unsigned)CV_MAX_DIM || dims == 0 )
        return false;

    if( m_hist )
    {
        int dims2[CV_MAX_DIM];
        int c_dims2 = cvGetDims( m_hist->bins, dims2 );

        if( c_dims2 == c_dims && memcmp( dims, dims2, c_dims*sizeof(dims[0])) == 0 )
            return true;

        cvReleaseHist( &m_hist );
    }

    m_hist = cvCreateHist( c_dims, dims, CV_HIST_ARRAY, 0, 0 );

    return true;
}


bool
CvCamShiftTracker::set_hist_bin_range( int channel, int min_val, int max_val )
{
    if( (unsigned)channel >= (unsigned)CV_MAX_DIM ||
        min_val >= max_val || min_val < 0 || max_val > 256 )
    {
        assert(0);
        return false;
    }

    m_hist_ranges[channel][0] = (float)min_val;
    m_hist_ranges[channel][1] = (float)max_val;

    return true;
}

/* End of file. */
