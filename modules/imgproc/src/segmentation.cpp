/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

/****************************************************************************************\
*                                       Watershed                                        *
\****************************************************************************************/

namespace cv
{
// A node represents a pixel to label
struct WSNode
{
    int next;
    int mask_ofs;
    int img_ofs;
};

// Queue for WSNodes
struct WSQueue
{
    WSQueue() { first = last = 0; }
    int first, last;
};


static int
allocWSNodes( std::vector<WSNode>& storage )
{
    int sz = (int)storage.size();
    int newsz = MAX(128, sz*3/2);

    storage.resize(newsz);
    if( sz == 0 )
    {
        storage[0].next = 0;
        sz = 1;
    }
    for( int i = sz; i < newsz-1; i++ )
        storage[i].next = i+1;
    storage[newsz-1].next = 0;
    return sz;
}

}


void cv::watershed( InputArray _src, InputOutputArray _markers )
{
    CV_INSTRUMENT_REGION();

    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    const int WSHED = -1; // Pixel belongs to watershed

    // possible bit values = 2^8
    const int NQ = 256;

    Mat src = _src.getMat(), dst = _markers.getMat();
    Size size = src.size();

    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
    #define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
    #define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
    #define ws_push(idx,mofs,iofs)          \
    {                                       \
        if( !free_node )                    \
            free_node = allocWSNodes( storage );\
        node = free_node;                   \
        free_node = storage[free_node].next;\
        storage[node].next = 0;             \
        storage[node].mask_ofs = mofs;      \
        storage[node].img_ofs = iofs;       \
        if( q[idx].last )                   \
            storage[q[idx].last].next=node; \
        else                                \
            q[idx].first = node;            \
        q[idx].last = node;                 \
    }

    // Get next node from queue idx
    #define ws_pop(idx,mofs,iofs)           \
    {                                       \
        node = q[idx].first;                \
        q[idx].first = storage[node].next;  \
        if( !storage[node].next )           \
            q[idx].last = 0;                \
        storage[node].next = free_node;     \
        free_node = node;                   \
        mofs = storage[node].mask_ofs;      \
        iofs = storage[node].img_ofs;       \
    }

    // Get highest absolute channel difference in diff
    #define c_diff(ptr1,ptr2,diff)           \
    {                                        \
        db = std::abs((ptr1)[0] - (ptr2)[0]);\
        dg = std::abs((ptr1)[1] - (ptr2)[1]);\
        dr = std::abs((ptr1)[2] - (ptr2)[2]);\
        diff = ws_max(db,dg);                \
        diff = ws_max(diff,dr);              \
        CV_Assert( 0 <= diff && diff <= 255 );  \
    }

    CV_Assert( src.type() == CV_8UC3 && dst.type() == CV_32SC1 );
    CV_Assert( src.size() == dst.size() );

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step/sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for( i = 0; i < 256; i++ )
        subs_tab[i] = 0;
    for( i = 256; i <= 512; i++ )
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for( j = 0; j < size.width; j++ )
        mask[j] = mask[j + mstep*(size.height-1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for( i = 1; i < size.height-1; i++ )
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width-1] = WSHED; // boundary pixels

        for( j = 1; j < size.width-1; j++ )
        {
            int* m = mask + j;
            if( m[0] < 0 ) m[0] = 0;
            if( m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0) )
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j*3;
                int idx = 256, t;
                if( m[-1] > 0 )
                    c_diff( ptr, ptr - 3, idx );
                if( m[1] > 0 )
                {
                    c_diff( ptr, ptr + 3, t );
                    idx = ws_min( idx, t );
                }
                if( m[-mstep] > 0 )
                {
                    c_diff( ptr, ptr - istep, t );
                    idx = ws_min( idx, t );
                }
                if( m[mstep] > 0 )
                {
                    c_diff( ptr, ptr + istep, t );
                    idx = ws_min( idx, t );
                }

                // Add to according queue
                CV_Assert( 0 <= idx && idx <= 255 );
                ws_push( idx, i*mstep + j, i*istep + j*3 );
                m[0] = IN_QUEUE;
            }
        }
    }

    // find the first non-empty queue
    for( i = 0; i < NQ; i++ )
        if( q[i].first )
            break;

    // if there is no markers, exit immediately
    if( i == NQ )
        return;

    active_queue = i;
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for(;;)
    {
        int mofs, iofs;
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if( q[active_queue].first == 0 )
        {
            for( i = active_queue+1; i < NQ; i++ )
                if( q[i].first )
                    break;
            if( i == NQ )
                break;
            active_queue = i;
        }

        // Get next node
        ws_pop( active_queue, mofs, iofs );

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;

        // Check surrounding pixels for labels
        // to determine label for current pixel
        t = m[-1]; // Left
        if( t > 0 ) lab = t;
        t = m[1]; // Right
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[-mstep]; // Top
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[mstep]; // Bottom
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }

        // Set label to current pixel in marker image
        CV_Assert( lab != 0 );
        m[0] = lab;

        if( lab == WSHED )
            continue;

        // Add adjacent, unlabeled pixels to corresponding queue
        if( m[-1] == 0 )
        {
            c_diff( ptr, ptr - 3, t );
            ws_push( t, mofs - 1, iofs - 3 );
            active_queue = ws_min( active_queue, t );
            m[-1] = IN_QUEUE;
        }
        if( m[1] == 0 )
        {
            c_diff( ptr, ptr + 3, t );
            ws_push( t, mofs + 1, iofs + 3 );
            active_queue = ws_min( active_queue, t );
            m[1] = IN_QUEUE;
        }
        if( m[-mstep] == 0 )
        {
            c_diff( ptr, ptr - istep, t );
            ws_push( t, mofs - mstep, iofs - istep );
            active_queue = ws_min( active_queue, t );
            m[-mstep] = IN_QUEUE;
        }
        if( m[mstep] == 0 )
        {
            c_diff( ptr, ptr + istep, t );
            ws_push( t, mofs + mstep, iofs + istep );
            active_queue = ws_min( active_queue, t );
            m[mstep] = IN_QUEUE;
        }
    }
}


/****************************************************************************************\
*                                         Meanshift                                      *
\****************************************************************************************/


void cv::pyrMeanShiftFiltering( InputArray _src, OutputArray _dst,
                                double sp0, double sr, int max_level,
                                TermCriteria termcrit )
{
    CV_INSTRUMENT_REGION();

    Mat src0 = _src.getMat();

    if( src0.empty() )
        return;

    _dst.create( src0.size(), src0.type() );
    Mat dst0 = _dst.getMat();

    const int cn = 3;
    const int MAX_LEVELS = 8;

    if( (unsigned)max_level > (unsigned)MAX_LEVELS )
        CV_Error( cv::Error::StsOutOfRange, "The number of pyramid levels is too large or negative" );

    std::vector<cv::Mat> src_pyramid(max_level+1);
    std::vector<cv::Mat> dst_pyramid(max_level+1);
    cv::Mat mask0;
    int i, j, level;
    //uchar* submask = 0;

    #define cdiff(ofs0) (tab[c0-dptr[ofs0]+255] + \
        tab[c1-dptr[(ofs0)+1]+255] + tab[c2-dptr[(ofs0)+2]+255] >= isr22)

    double sr2 = sr * sr;
    int isr2 = cvRound(sr2), isr22 = MAX(isr2,16);
    int tab[768];


    if( src0.type() != CV_8UC3 )
        CV_Error( cv::Error::StsUnsupportedFormat, "Only 8-bit, 3-channel images are supported" );

    if( src0.type() != dst0.type() )
        CV_Error( cv::Error::StsUnmatchedFormats, "The input and output images must have the same type" );

    if( src0.size() != dst0.size() )
        CV_Error( cv::Error::StsUnmatchedSizes, "The input and output images must have the same size" );

    if( !(termcrit.type & TermCriteria::MAX_ITER) )
        termcrit.maxCount = 5;
    termcrit.maxCount = MAX(termcrit.maxCount,1);
    termcrit.maxCount = MIN(termcrit.maxCount,100);
    if( !(termcrit.type & TermCriteria::EPS) )
        termcrit.epsilon = 1.f;
    termcrit.epsilon = MAX(termcrit.epsilon, 0.f);

    for( i = 0; i < 768; i++ )
        tab[i] = (i - 255)*(i - 255);

    // 1. construct pyramid
    src_pyramid[0] = src0;
    dst_pyramid[0] = dst0;
    for( level = 1; level <= max_level; level++ )
    {
        src_pyramid[level].create( (src_pyramid[level-1].rows+1)/2,
                        (src_pyramid[level-1].cols+1)/2, src_pyramid[level-1].type() );
        dst_pyramid[level].create( src_pyramid[level].rows,
                        src_pyramid[level].cols, src_pyramid[level].type() );
        cv::pyrDown( src_pyramid[level-1], src_pyramid[level], src_pyramid[level].size() );
    }

    mask0.create(src0.rows, src0.cols, CV_8UC1);
    //CV_CALL( submask = (uchar*)cvAlloc( (sp+2)*(sp+2) ));

    // 2. apply meanshift, starting from the pyramid top (i.e. the smallest layer)
    for( level = max_level; level >= 0; level-- )
    {
        cv::Mat src = src_pyramid[level];
        cv::Size size = src.size();
        const uchar* sptr = src.ptr();
        int sstep = (int)src.step;
        uchar* dptr;
        int dstep;
        float sp = (float)(sp0 / (1 << level));
        sp = MAX( sp, 1 );

        cv::Mat m;
        if( level < max_level )
        {
            cv::Size size1 = dst_pyramid[level+1].size();
            m = cv::Mat(size.height, size.width, CV_8UC1, mask0.ptr());
            dstep = (int)dst_pyramid[level+1].step;
            dptr = dst_pyramid[level+1].ptr() + dstep + cn;
            cv::pyrUp( dst_pyramid[level+1], dst_pyramid[level], dst_pyramid[level].size() );
            m.setTo(cv::Scalar::all(0));

            for( i = 1; i < size1.height-1; i++, dptr += dstep - (size1.width-2)*3)
            {
                uchar* mask = m.ptr(1 + i * 2);
                for( j = 1; j < size1.width-1; j++, dptr += cn )
                {
                    int c0 = dptr[0], c1 = dptr[1], c2 = dptr[2];
                    mask[j*2 - 1] = cdiff(-3) || cdiff(3) || cdiff(-dstep-3) || cdiff(-dstep) ||
                        cdiff(-dstep+3) || cdiff(dstep-3) || cdiff(dstep) || cdiff(dstep+3);
                }
            }

            cv::dilate( m, m, cv::Mat() );
        }

        dptr = dst_pyramid[level].ptr();
        dstep = (int)dst_pyramid[level].step;

        for( i = 0; i < size.height; i++, sptr += sstep - size.width*3,
                                          dptr += dstep - size.width*3
        )
        {
            uchar* mask = m.empty() ? NULL : m.ptr(i);
            for( j = 0; j < size.width; j++, sptr += 3, dptr += 3 )
            {
                int x0 = j, y0 = i, x1, y1, iter;
                int c0, c1, c2;

                if( mask && !mask[j] )
                    continue;

                c0 = sptr[0], c1 = sptr[1], c2 = sptr[2];

                // iterate meanshift procedure
                for( iter = 0; iter < termcrit.maxCount; iter++ )
                {
                    const uchar* ptr;
                    int x, y, count = 0;
                    int minx, miny, maxx, maxy;
                    int s0 = 0, s1 = 0, s2 = 0, sx = 0, sy = 0;
                    double icount;
                    int stop_flag;

                    //mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
                    minx = cvRound(x0 - sp); minx = MAX(minx, 0);
                    miny = cvRound(y0 - sp); miny = MAX(miny, 0);
                    maxx = cvRound(x0 + sp); maxx = MIN(maxx, size.width-1);
                    maxy = cvRound(y0 + sp); maxy = MIN(maxy, size.height-1);
                    ptr = sptr + (miny - i)*sstep + (minx - j)*3;

                    for( y = miny; y <= maxy; y++, ptr += sstep - (maxx-minx+1)*3 )
                    {
                        int row_count = 0;
                        x = minx;
                        #if CV_ENABLE_UNROLLED
                        for( ; x + 3 <= maxx; x += 4, ptr += 12 )
                        {
                            int t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x; row_count++;
                            }
                            t0 = ptr[3], t1 = ptr[4], t2 = ptr[5];
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x+1; row_count++;
                            }
                            t0 = ptr[6], t1 = ptr[7], t2 = ptr[8];
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x+2; row_count++;
                            }
                            t0 = ptr[9], t1 = ptr[10], t2 = ptr[11];
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x+3; row_count++;
                            }
                        }
                        #endif
                        for( ; x <= maxx; x++, ptr += 3 )
                        {
                            int t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];
                            if( tab[t0-c0+255] + tab[t1-c1+255] + tab[t2-c2+255] <= isr2 )
                            {
                                s0 += t0; s1 += t1; s2 += t2;
                                sx += x; row_count++;
                            }
                        }
                        count += row_count;
                        sy += y*row_count;
                    }

                    if( count == 0 )
                        break;

                    icount = 1./count;
                    x1 = cvRound(sx*icount);
                    y1 = cvRound(sy*icount);
                    s0 = cvRound(s0*icount);
                    s1 = cvRound(s1*icount);
                    s2 = cvRound(s2*icount);

                    stop_flag = (x0 == x1 && y0 == y1) || std::abs(x1-x0) + std::abs(y1-y0) +
                        tab[s0 - c0 + 255] + tab[s1 - c1 + 255] +
                        tab[s2 - c2 + 255] <= termcrit.epsilon;

                    x0 = x1; y0 = y1;
                    c0 = s0; c1 = s1; c2 = s2;

                    if( stop_flag )
                        break;
                }

                dptr[0] = (uchar)c0;
                dptr[1] = (uchar)c1;
                dptr[2] = (uchar)c2;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void cvWatershed( const CvArr* _src, CvArr* _markers )
{
    cv::Mat src = cv::cvarrToMat(_src), markers = cv::cvarrToMat(_markers);
    cv::watershed(src, markers);
}


CV_IMPL void
cvPyrMeanShiftFiltering( const CvArr* srcarr, CvArr* dstarr,
                        double sp0, double sr, int max_level,
                        CvTermCriteria termcrit )
{
    cv::Mat src = cv::cvarrToMat(srcarr);
    const cv::Mat dst = cv::cvarrToMat(dstarr);

    cv::pyrMeanShiftFiltering(src, dst, sp0, sr, max_level, termcrit);
}
