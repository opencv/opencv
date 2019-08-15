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
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <algorithm>
#include <iterator>

namespace cv
{

// Classical Hough Transform
struct LinePolar
{
    float rho;
    float angle;
};


struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

static void
createTrigTable( int numangle, double min_theta, double theta_step,
                 float irho, float *tabSin, float *tabCos )
{
    float ang = static_cast<float>(min_theta);
    for(int n = 0; n < numangle; ang += (float)theta_step, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }
}

static void
findLocalMaximums( int numrho, int numangle, int threshold,
                   const int *accum, std::vector<int>& sort_buf )
{
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf.push_back(base);
        }
}

/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
static void
HoughLinesStandard( InputArray src, OutputArray lines, int type,
                    float rho, float theta,
                    int threshold, int linesMax,
                    double min_theta, double max_theta )
{
    CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Internal error");

    Mat img = src.getMat();

    int i, j;
    float irho = 1 / rho;

    CV_Assert( img.type() == CV_8UC1 );
    CV_Assert( linesMax > 0 );

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;

    int max_rho = width + height;
    int min_rho = -max_rho;

    CV_CheckGE(max_theta, min_theta, "max_theta must be greater than min_theta");

    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound(((max_rho - min_rho) + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    if (type == CV_32FC2 && CV_IPP_CHECK_COND)
    {
        IppiSize srcSize = { width, height };
        IppPointPolar delta = { rho, theta };
        IppPointPolar dstRoi[2] = {{(Ipp32f) min_rho, (Ipp32f) min_theta},{(Ipp32f) max_rho, (Ipp32f) max_theta}};
        int bufferSize;
        int nz = countNonZero(img);
        int ipp_linesMax = std::min(linesMax, nz*numangle/threshold);
        int linesCount = 0;
        std::vector<Vec2f> _lines(ipp_linesMax);
        IppStatus ok = ippiHoughLineGetSize_8u_C1R(srcSize, delta, ipp_linesMax, &bufferSize);
        Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
        if (ok >= 0) {ok = CV_INSTRUMENT_FUN_IPP(ippiHoughLine_Region_8u32f_C1R, image, step, srcSize, (IppPointPolar*) &_lines[0], dstRoi, ipp_linesMax, &linesCount, delta, threshold, buffer);};
        ippsFree(buffer);
        if (ok >= 0)
        {
            lines.create(linesCount, 1, CV_32FC2);
            Mat(linesCount, 1, CV_32FC2, &_lines[0]).copyTo(lines);
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif


    Mat _accum = Mat::zeros( (numangle+2), (numrho+2), CV_32SC1 );
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum.ptr<int>();
    float *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

    // create sin and cos table
    createTrigTable( numangle, min_theta, theta,
                     irho, tabSin, tabCos);

    // stage 1. fill accumulator
    for( i = 0; i < height; i++ )
        for( j = 0; j < width; j++ )
        {
            if( image[i * step + j] != 0 )
                for(int n = 0; n < numangle; n++ )
                {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }

    // stage 2. find local maximums
    findLocalMaximums( numrho, numangle, threshold, accum, _sort_buf );

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1./(numrho+2);

    lines.create(linesMax, 1, type);
    Mat _lines = lines.getMat();
    for( i = 0; i < linesMax; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        if (type == CV_32FC2)
        {
            _lines.at<Vec2f>(i) = Vec2f(line.rho, line.angle);
        }
        else
        {
            CV_DbgAssert(type == CV_32FC3);
            _lines.at<Vec3f>(i) = Vec3f(line.rho, line.angle, (float)accum[idx]);
        }
    }
}


// Multi-Scale variant of Classical Hough Transform

struct hough_index
{
    hough_index() : value(0), rho(0.f), theta(0.f) {}
    hough_index(int _val, float _rho, float _theta)
    : value(_val), rho(_rho), theta(_theta) {}

    int value;
    float rho, theta;
};


static void
HoughLinesSDiv( InputArray image, OutputArray lines, int type,
                float rho, float theta, int threshold,
                int srn, int stn, int linesMax,
                double min_theta, double max_theta )
{
    CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Internal error");

    #define _POINT(row, column)\
        (image_src[(row)*step+(column)])

    Mat img = image.getMat();
    int index, i;
    int ri, ti, ti1, ti0;
    int row, col;
    float r, t;                 /* Current rho and theta */
    float rv;                   /* Some temporary rho value */

    int fn = 0;
    float xc, yc;

    const float d2r = (float)(CV_PI / 180);
    int sfn = srn * stn;
    int fi;
    int count;
    int cmax = 0;

    std::vector<hough_index> lst;

    CV_Assert( img.type() == CV_8UC1 );
    CV_Assert( linesMax > 0 );

    threshold = MIN( threshold, 255 );

    const uchar* image_src = img.ptr();
    int step = (int)img.step;
    int w = img.cols;
    int h = img.rows;

    float irho = 1 / rho;
    float itheta = 1 / theta;
    float srho = rho / srn;
    float stheta = theta / stn;
    float isrho = 1 / srho;
    float istheta = 1 / stheta;

    int rn = cvFloor( std::sqrt( (double)w * w + (double)h * h ) * irho );
    int tn = cvFloor( 2 * CV_PI * itheta );

    lst.push_back(hough_index(threshold, -1.f, 0.f));

    // Precalculate sin table
    std::vector<float> _sinTable( 5 * tn * stn );
    float* sinTable = &_sinTable[0];

    for( index = 0; index < 5 * tn * stn; index++ )
        sinTable[index] = (float)cos( stheta * index * 0.2f );

    std::vector<uchar> _caccum(rn * tn, (uchar)0);
    uchar* caccum = &_caccum[0];

    // Counting all feature pixels
    for( row = 0; row < h; row++ )
        for( col = 0; col < w; col++ )
            fn += _POINT( row, col ) != 0;

    std::vector<int> _x(fn), _y(fn);
    int* x = &_x[0], *y = &_y[0];

    // Full Hough Transform (it's accumulator update part)
    fi = 0;
    for( row = 0; row < h; row++ )
    {
        for( col = 0; col < w; col++ )
        {
            if( _POINT( row, col ))
            {
                int halftn;
                float r0;
                float scale_factor;
                int iprev = -1;
                float phi, phi1;
                float theta_it;     // Value of theta for iterating

                // Remember the feature point
                x[fi] = col;
                y[fi] = row;
                fi++;

                yc = (float) row + 0.5f;
                xc = (float) col + 0.5f;

                /* Update the accumulator */
                t = (float) fabs( cvFastArctan( yc, xc ) * d2r );
                r = (float) std::sqrt( (double)xc * xc + (double)yc * yc );
                r0 = r * irho;
                ti0 = cvFloor( (t + CV_PI*0.5) * itheta );

                caccum[ti0]++;

                theta_it = rho / r;
                theta_it = theta_it < theta ? theta_it : theta;
                scale_factor = theta_it * itheta;
                halftn = cvFloor( CV_PI / theta_it );
                for( ti1 = 1, phi = theta_it - (float)(CV_PI*0.5), phi1 = (theta_it + t) * itheta;
                     ti1 < halftn; ti1++, phi += theta_it, phi1 += scale_factor )
                {
                    rv = r0 * std::cos( phi );
                    i = (int)rv * tn;
                    i += cvFloor( phi1 );
                    assert( i >= 0 );
                    assert( i < rn * tn );
                    caccum[i] = (uchar) (caccum[i] + ((i ^ iprev) != 0));
                    iprev = i;
                    if( cmax < caccum[i] )
                        cmax = caccum[i];
                }
            }
        }
    }

    // Starting additional analysis
    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti] > threshold )
                count++;
        }
    }

    if( count * 100 > rn * tn )
    {
        HoughLinesStandard( image, lines, type, rho, theta, threshold, linesMax, min_theta, max_theta );
        return;
    }

    std::vector<uchar> _buffer(srn * stn + 2);
    uchar* buffer = &_buffer[0];
    uchar* mcaccum = buffer + 1;

    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti] > threshold )
            {
                count++;
                memset( mcaccum, 0, sfn * sizeof( uchar ));

                for( index = 0; index < fn; index++ )
                {
                    int ti2;
                    float r0;

                    yc = (float) y[index] + 0.5f;
                    xc = (float) x[index] + 0.5f;

                    // Update the accumulator
                    t = (float) fabs( cvFastArctan( yc, xc ) * d2r );
                    r = (float) std::sqrt( (double)xc * xc + (double)yc * yc ) * isrho;
                    ti0 = cvFloor( (t + CV_PI * 0.5) * istheta );
                    ti2 = (ti * stn - ti0) * 5;
                    r0 = (float) ri *srn;

                    for( ti1 = 0; ti1 < stn; ti1++, ti2 += 5 )
                    {
                        rv = r * sinTable[(int) (std::abs( ti2 ))] - r0;
                        i = cvFloor( rv ) * stn + ti1;

                        i = CV_IMAX( i, -1 );
                        i = CV_IMIN( i, sfn );
                        mcaccum[i]++;
                        assert( i >= -1 );
                        assert( i <= sfn );
                    }
                }

                // Find peaks in maccum...
                for( index = 0; index < sfn; index++ )
                {
                    int pos = (int)(lst.size() - 1);
                    if( pos < 0 || lst[pos].value < mcaccum[index] )
                    {
                        hough_index vi(mcaccum[index],
                                       index / stn * srho + ri * rho,
                                       index % stn * stheta + ti * theta - (float)(CV_PI*0.5));
                        lst.push_back(vi);
                        for( ; pos >= 0; pos-- )
                        {
                            if( lst[pos].value > vi.value )
                                break;
                            lst[pos+1] = lst[pos];
                        }
                        lst[pos+1] = vi;
                        if( (int)lst.size() > linesMax )
                            lst.pop_back();
                    }
                }
            }
        }
    }

    lines.create((int)lst.size(), 1, type);
    Mat _lines = lines.getMat();
    for( size_t idx = 0; idx < lst.size(); idx++ )
    {
        if( lst[idx].rho < 0 )
            continue;
        if (type == CV_32FC2)
        {
            _lines.at<Vec2f>((int)idx) = Vec2f(lst[idx].rho, lst[idx].theta);
        }
        else
        {
            CV_DbgAssert(type == CV_32FC3);
            _lines.at<Vec3f>((int)idx) = Vec3f(lst[idx].rho, lst[idx].theta, (float)lst[idx].value);
        }
    }
}


/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

static void
HoughLinesProbabilistic( Mat& image,
                         float rho, float theta, int threshold,
                         int lineLength, int lineGap,
                         std::vector<Vec4i>& lines, int linesMax )
{
    Point pt;
    float irho = 1 / rho;
    RNG rng((uint64)-1);

    CV_Assert( image.type() == CV_8UC1 );

    int width = image.cols;
    int height = image.rows;

    int numangle = cvRound(CV_PI / theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    CV_IPP_CHECK()
    {
        IppiSize srcSize = { width, height };
        IppPointPolar delta = { rho, theta };
        IppiHoughProbSpec* pSpec;
        int bufferSize, specSize;
        int ipp_linesMax = std::min(linesMax, numangle*numrho);
        int linesCount = 0;
        lines.resize(ipp_linesMax);
        IppStatus ok = ippiHoughProbLineGetSize_8u_C1R(srcSize, delta, &specSize, &bufferSize);
        Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
        pSpec = (IppiHoughProbSpec*) ippsMalloc_8u_L(specSize);
        if (ok >= 0) ok = ippiHoughProbLineInit_8u32f_C1R(srcSize, delta, ippAlgHintNone, pSpec);
        if (ok >= 0) {ok = CV_INSTRUMENT_FUN_IPP(ippiHoughProbLine_8u32f_C1R, image.data, (int)image.step, srcSize, threshold, lineLength, lineGap, (IppiPoint*) &lines[0], ipp_linesMax, &linesCount, buffer, pSpec);};

        ippsFree(pSpec);
        ippsFree(buffer);
        if (ok >= 0)
        {
            lines.resize(linesCount);
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        lines.clear();
        setIppErrorStatus();
    }
#endif

    Mat accum = Mat::zeros( numangle, numrho, CV_32SC1 );
    Mat mask( height, width, CV_8UC1 );
    std::vector<float> trigtab(numangle*2);

    for( int n = 0; n < numangle; n++ )
    {
        trigtab[n*2] = (float)(cos((double)n*theta) * irho);
        trigtab[n*2+1] = (float)(sin((double)n*theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.ptr();
    std::vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for( pt.y = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] )
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();

    // stage 2. process all the points in random order
    for( ; count > 0; count-- )
    {
        // choose random point out of the remaining ones
        int idx = rng.uniform(0, count);
        int max_val = threshold-1, max_n = 0;
        Point point = nzloc[idx];
        Point line_end[2];
        float a, b;
        int* adata = accum.ptr<int>();
        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count-1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if( !mdata0[i*width + j] )
            continue;

        // update accumulator, find the most probable line
        for( int n = 0; n < numangle; n++, adata += numrho )
        {
            int r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n*2+1];
        b = ttab[max_n*2];
        x0 = j;
        y0 = i;
        if( fabs(a) > fabs(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 2; k++ )
        {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    std::abs(line_end[1].y - line_end[0].y) >= lineLength;

        for( k = 0; k < 2; k++ )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    if( good_line )
                    {
                        adata = accum.ptr<int>();
                        for( int n = 0; n < numangle; n++, adata += numrho )
                        {
                            int r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line )
        {
            Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
            lines.push_back(lr);
            if( (int)lines.size() >= linesMax )
                return;
        }
    }
}

#ifdef HAVE_OPENCL

#define OCL_MAX_LINES 4096

static bool ocl_makePointsList(InputArray _src, OutputArray _pointsList, InputOutputArray _counters)
{
    UMat src = _src.getUMat();
    _pointsList.create(1, (int) src.total(), CV_32SC1);
    UMat pointsList = _pointsList.getUMat();
    UMat counters = _counters.getUMat();
    ocl::Device dev = ocl::Device::getDefault();

    const int pixPerWI = 16;
    int workgroup_size = min((int) dev.maxWorkGroupSize(), (src.cols + pixPerWI - 1)/pixPerWI);
    ocl::Kernel pointListKernel("make_point_list", ocl::imgproc::hough_lines_oclsrc,
                                format("-D MAKE_POINTS_LIST -D GROUP_SIZE=%d -D LOCAL_SIZE=%d", workgroup_size, src.cols));
    if (pointListKernel.empty())
        return false;

    pointListKernel.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnlyNoSize(pointsList),
                         ocl::KernelArg::PtrWriteOnly(counters));

    size_t localThreads[2]  = { (size_t)workgroup_size, 1 };
    size_t globalThreads[2] = { (size_t)workgroup_size, (size_t)src.rows };

    return pointListKernel.run(2, globalThreads, localThreads, false);
}

static bool ocl_fillAccum(InputArray _pointsList, OutputArray _accum, int total_points, double rho, double theta, int numrho, int numangle)
{
    UMat pointsList = _pointsList.getUMat();
    _accum.create(numangle + 2, numrho + 2, CV_32SC1);
    UMat accum = _accum.getUMat();
    ocl::Device dev = ocl::Device::getDefault();

    float irho = (float) (1 / rho);
    int workgroup_size = min((int) dev.maxWorkGroupSize(), total_points);

    ocl::Kernel fillAccumKernel;
    size_t localThreads[2];
    size_t globalThreads[2];

    size_t local_memory_needed = (numrho + 2)*sizeof(int);
    if (local_memory_needed > dev.localMemSize())
    {
        accum.setTo(Scalar::all(0));
        fillAccumKernel.create("fill_accum_global", ocl::imgproc::hough_lines_oclsrc,
                                format("-D FILL_ACCUM_GLOBAL"));
        if (fillAccumKernel.empty())
            return false;
        globalThreads[0] = workgroup_size; globalThreads[1] = numangle;
        fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
                        total_points, irho, (float) theta, numrho, numangle);
        return fillAccumKernel.run(2, globalThreads, NULL, false);
    }
    else
    {
        fillAccumKernel.create("fill_accum_local", ocl::imgproc::hough_lines_oclsrc,
                                format("-D FILL_ACCUM_LOCAL -D LOCAL_SIZE=%d -D BUFFER_SIZE=%d", workgroup_size, numrho + 2));
        if (fillAccumKernel.empty())
            return false;
        localThreads[0] = workgroup_size; localThreads[1] = 1;
        globalThreads[0] = workgroup_size; globalThreads[1] = numangle+2;
        fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
                        total_points, irho, (float) theta, numrho, numangle);
        return fillAccumKernel.run(2, globalThreads, localThreads, false);
    }
}

static bool ocl_HoughLines(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
                           double min_theta, double max_theta)
{
    CV_Assert(_src.type() == CV_8UC1);

    if (max_theta < 0 || max_theta > CV_PI ) {
        CV_Error( Error::StsBadArg, "max_theta must fall between 0 and pi" );
    }
    if (min_theta < 0 || min_theta > max_theta ) {
        CV_Error( Error::StsBadArg, "min_theta must fall between 0 and max_theta" );
    }
    if (!(rho > 0 && theta > 0)) {
        CV_Error( Error::StsBadArg, "rho and theta must be greater 0" );
    }

    UMat src = _src.getUMat();
    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

    UMat pointsList;
    UMat counters(1, 2, CV_32SC1, Scalar::all(0));

    if (!ocl_makePointsList(src, pointsList, counters))
        return false;

    int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
    if (total_points <= 0)
    {
        _lines.release();
        return true;
    }

    UMat accum;
    if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
        return false;

    const int pixPerWI = 8;
    ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
                               format("-D GET_LINES"));
    if (getLinesKernel.empty())
        return false;

    int linesMax = threshold > 0 ? min(total_points*numangle/threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
    UMat lines(linesMax, 1, CV_32FC2);

    getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::WriteOnlyNoSize(lines),
                        ocl::KernelArg::PtrWriteOnly(counters), linesMax, threshold, (float) rho, (float) theta);

    size_t globalThreads[2] = { ((size_t)numrho + pixPerWI - 1)/pixPerWI, (size_t)numangle };
    if (!getLinesKernel.run(2, globalThreads, NULL, false))
        return false;

    int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
    if (total_lines > 0)
        _lines.assign(lines.rowRange(Range(0, total_lines)));
    else
        _lines.release();
    return true;
}

static bool ocl_HoughLinesP(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
                           double minLineLength, double maxGap)
{
    CV_Assert(_src.type() == CV_8UC1);

    if (!(rho > 0 && theta > 0)) {
        CV_Error( Error::StsBadArg, "rho and theta must be greater 0" );
    }

    UMat src = _src.getUMat();
    int numangle = cvRound(CV_PI / theta);
    int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

    UMat pointsList;
    UMat counters(1, 2, CV_32SC1, Scalar::all(0));

    if (!ocl_makePointsList(src, pointsList, counters))
        return false;

    int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
    if (total_points <= 0)
    {
        _lines.release();
        return true;
    }

    UMat accum;
    if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
        return false;

    ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
                               format("-D GET_LINES_PROBABOLISTIC"));
    if (getLinesKernel.empty())
        return false;

    int linesMax = threshold > 0 ? min(total_points*numangle/threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
    UMat lines(linesMax, 1, CV_32SC4);

    getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::ReadOnly(src),
                        ocl::KernelArg::WriteOnlyNoSize(lines), ocl::KernelArg::PtrWriteOnly(counters),
                        linesMax, threshold, (int) minLineLength, (int) maxGap, (float) rho, (float) theta);

    size_t globalThreads[2] = { (size_t)numrho, (size_t)numangle };
    if (!getLinesKernel.run(2, globalThreads, NULL, false))
        return false;

    int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
    if (total_lines > 0)
        _lines.assign(lines.rowRange(Range(0, total_lines)));
    else
        _lines.release();

    return true;
}

#endif /* HAVE_OPENCL */

void HoughLines( InputArray _image, OutputArray lines,
                 double rho, double theta, int threshold,
                 double srn, double stn, double min_theta, double max_theta )
{
    CV_INSTRUMENT_REGION();

    int type = CV_32FC2;
    if (lines.fixedType())
    {
        type = lines.type();
        CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Wrong type of output lines");
    }

    CV_OCL_RUN(srn == 0 && stn == 0 && _image.isUMat() && lines.isUMat() && type == CV_32FC2,
               ocl_HoughLines(_image, lines, rho, theta, threshold, min_theta, max_theta));

    if( srn == 0 && stn == 0 )
        HoughLinesStandard(_image, lines, type, (float)rho, (float)theta, threshold, INT_MAX, min_theta, max_theta );
    else
        HoughLinesSDiv(_image, lines, type, (float)rho, (float)theta, threshold, cvRound(srn), cvRound(stn), INT_MAX, min_theta, max_theta);
}


void HoughLinesP(InputArray _image, OutputArray _lines,
                 double rho, double theta, int threshold,
                 double minLineLength, double maxGap )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_image.isUMat() && _lines.isUMat(),
               ocl_HoughLinesP(_image, _lines, rho, theta, threshold, minLineLength, maxGap));

    Mat image = _image.getMat();
    std::vector<Vec4i> lines;
    HoughLinesProbabilistic(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX);
    Mat(lines).copyTo(_lines);
}

void HoughLinesPointSet( InputArray _point, OutputArray _lines, int lines_max, int threshold,
                         double min_rho, double max_rho, double rho_step,
                         double min_theta, double max_theta, double theta_step )
{
    std::vector<Vec3d> lines;
    std::vector<Point2f> point;
    _point.copyTo(point);

    CV_Assert( _point.type() == CV_32FC2 || _point.type() == CV_32SC2 );
    if( lines_max <= 0 ) {
        CV_Error( Error::StsBadArg, "lines_max must be greater than 0" );
    }
    if( threshold < 0) {
        CV_Error( Error::StsBadArg, "threshold must be greater than 0" );
    }
    if( ((max_rho - min_rho) <= 0) || ((max_theta - min_theta) <= 0) ) {
        CV_Error( Error::StsBadArg, "max must be greater than min" );
    }
    if( ((rho_step <= 0)) || ((theta_step <= 0)) ) {
        CV_Error( Error::StsBadArg, "step must be greater than 0" );
    }

    int i;
    float irho = 1 / (float)rho_step;
    float irho_min = ((float)min_rho * irho);
    int numangle = cvRound((max_theta - min_theta) / theta_step);
    int numrho = cvRound((max_rho - min_rho + 1) / rho_step);

    Mat _accum = Mat::zeros( (numangle+2), (numrho+2), CV_32SC1 );
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum.ptr<int>();
    float *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

    // create sin and cos table
    createTrigTable( numangle, min_theta, theta_step,
                     irho, tabSin, tabCos );

    // stage 1. fill accumulator
    for( i = 0; i < (int)point.size(); i++ )
        for(int n = 0; n < numangle; n++ )
        {
            int r = cvRound( point.at(i).x  * tabCos[n] + point.at(i).y * tabSin[n] - irho_min);
            accum[(n+1) * (numrho+2) + r+1]++;
        }

    // stage 2. find local maximums
    findLocalMaximums( numrho, numangle, threshold, accum, _sort_buf );

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    lines_max = std::min(lines_max, (int)_sort_buf.size());
    double scale = 1./(numrho+2);
    for( i = 0; i < lines_max; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = static_cast<float>(min_rho) + r * (float)rho_step;
        line.angle = static_cast<float>(min_theta) + n * (float)theta_step;
        lines.push_back(Vec3d((double)accum[idx], (double)line.rho, (double)line.angle));
    }

    Mat(lines).copyTo(_lines);
}

/****************************************************************************************\
*                                     Circle Detection                                   *
\****************************************************************************************/

struct EstimatedCircle
{
    EstimatedCircle(Vec3f _c, int _accum) :
        c(_c), accum(_accum) {}
    Vec3f c;
    int accum;
};

static bool cmpAccum(const EstimatedCircle& left, const EstimatedCircle& right)
{
    // Compare everything so the order is completely deterministic
    // Larger accum first
    if (left.accum > right.accum)
        return true;
    else if (left.accum < right.accum)
        return false;
    // Larger radius first
    else if (left.c[2] > right.c[2])
        return true;
    else if (left.c[2] < right.c[2])
        return false;
    // Smaller X
    else if (left.c[0] < right.c[0])
        return true;
    else if (left.c[0] > right.c[0])
        return false;
    // Smaller Y
    else if (left.c[1] < right.c[1])
        return true;
    else if (left.c[1] > right.c[1])
        return false;
    // Identical - neither object is less than the other
    else
        return false;
}

static inline Vec3f GetCircle(const EstimatedCircle& est)
{
    return est.c;
}

static inline Vec4f GetCircle4f(const EstimatedCircle& est)
{
    return Vec4f(est.c[0], est.c[1], est.c[2], (float)est.accum);
}

class NZPointList : public std::vector<Point>
{
private:
    NZPointList(const NZPointList& other); // non-copyable

public:
    NZPointList(int reserveSize = 256)
    {
        reserve(reserveSize);
    }
};

class NZPointSet
{
private:
    NZPointSet(const NZPointSet& other); // non-copyable

public:
    Mat_<uchar> positions;

    NZPointSet(int rows, int cols) :
        positions(rows, cols, (uchar)0)
    {
    }

    void insert(const Point& pt)
    {
        positions(pt) = 1;
    }

    void insert(const NZPointSet& from)
    {
        cv::bitwise_or(from.positions, positions, positions);
    }

    void toList(NZPointList& list) const
    {
        for (int y = 0; y < positions.rows; y++)
        {
            const uchar *ptr = positions.ptr<uchar>(y, 0);
            for (int x = 0; x < positions.cols; x++)
            {
                if (ptr[x])
                {
                    list.push_back(Point(x, y));
                }
            }
        }
    }
};

class HoughCirclesAccumInvoker : public ParallelLoopBody
{
public:
    HoughCirclesAccumInvoker(const Mat &_edges, const Mat &_dx, const Mat &_dy, int _minRadius, int _maxRadius, float _idp,
                             std::vector<Mat>& _accumVec, NZPointSet& _nz, Mutex& _mtx) :
        edges(_edges), dx(_dx), dy(_dy), minRadius(_minRadius), maxRadius(_maxRadius), idp(_idp),
        accumVec(_accumVec), nz(_nz), mutex(_mtx)
    {
        acols = cvCeil(edges.cols * idp), arows = cvCeil(edges.rows * idp);
        astep = acols + 2;
    }

    ~HoughCirclesAccumInvoker() { }

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        Mat accumLocal = Mat(arows + 2, acols + 2, CV_32SC1, Scalar::all(0));
        int *adataLocal = accumLocal.ptr<int>();
        NZPointSet nzLocal(nz.positions.rows, nz.positions.cols);
        int startRow = boundaries.start;
        int endRow = boundaries.end;
        int numCols = edges.cols;

        if(edges.isContinuous() && dx.isContinuous() && dy.isContinuous())
        {
            numCols *= (boundaries.end - boundaries.start);
            endRow = boundaries.start + 1;
        }

        // Accumulate circle evidence for each edge pixel
        for(int y = startRow; y < endRow; ++y )
        {
            const uchar* edgeData = edges.ptr<const uchar>(y);
            const short* dxData = dx.ptr<const short>(y);
            const short* dyData = dy.ptr<const short>(y);
            int x = 0;

            for(; x < numCols; ++x )
            {
#if CV_SIMD
                {
                    v_uint8 v_zero = vx_setzero_u8();

                    for(; x <= numCols - 2*v_uint8::nlanes; x += 2*v_uint8::nlanes) {
                        v_uint8 v_edge1 = (vx_load(edgeData + x                  ) != v_zero);
                        v_uint8 v_edge2 = (vx_load(edgeData + x + v_uint8::nlanes) != v_zero);

                        if(v_check_any(v_edge1))
                        {
                            x += v_scan_forward(v_edge1);
                            goto _next_step;
                        }

                        if(v_check_any(v_edge2))
                        {
                            x += v_uint8::nlanes + v_scan_forward(v_edge2);
                            goto _next_step;
                        }
                    }
                }
#endif
                for(; x < numCols && !edgeData[x]; ++x)
                    ;

                if(x == numCols)
                    continue;
#if CV_SIMD
_next_step:
#endif
                float vx, vy;
                int sx, sy, x0, y0, x1, y1;

                vx = dxData[x];
                vy = dyData[x];

                if(vx == 0 && vy == 0)
                    continue;

                float mag = std::sqrt(vx*vx+vy*vy);

                if(mag < 1.0f)
                    continue;

                Point pt = Point(x % edges.cols, y + x / edges.cols);
                nzLocal.insert(pt);

                sx = cvRound((vx * idp) * 1024 / mag);
                sy = cvRound((vy * idp) * 1024 / mag);

                x0 = cvRound((pt.x * idp) * 1024);
                y0 = cvRound((pt.y * idp) * 1024);

                // Step from min_radius to max_radius in both directions of the gradient
                for(int k1 = 0; k1 < 2; k1++ )
                {
                    x1 = x0 + minRadius * sx;
                    y1 = y0 + minRadius * sy;

                    for(int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, r++ )
                    {
                        int x2 = x1 >> 10, y2 = y1 >> 10;
                        if( (unsigned)x2 >= (unsigned)acols ||
                            (unsigned)y2 >= (unsigned)arows )
                            break;

                        adataLocal[y2*astep + x2]++;
                    }

                    sx = -sx; sy = -sy;
                }
            }
        }

        { // TODO Try using TLSContainers
            AutoLock lock(mutex);
            accumVec.push_back(accumLocal);
            nz.insert(nzLocal);
        }
    }

private:
    const Mat &edges, &dx, &dy;
    int minRadius, maxRadius;
    float idp;
    std::vector<Mat>& accumVec;
    NZPointSet& nz;

    int acols, arows, astep;

    Mutex& mutex;
};

class HoughCirclesFindCentersInvoker : public ParallelLoopBody
{
public:
    HoughCirclesFindCentersInvoker(const Mat &_accum, std::vector<int> &_centers, int _accThreshold, Mutex& _mutex) :
        accum(_accum), centers(_centers), accThreshold(_accThreshold), _lock(_mutex)
    {
        acols = accum.cols;
        arows = accum.rows;
        adata = accum.ptr<int>();
    }

    ~HoughCirclesFindCentersInvoker() {}

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        int startRow = boundaries.start;
        int endRow = boundaries.end;
        std::vector<int> centersLocal;
        bool singleThread = (boundaries == Range(1, accum.rows - 1));

        startRow = max(1, startRow);
        endRow = min(arows - 1, endRow);

        //Find possible circle centers
        for(int y = startRow; y < endRow; ++y )
        {
            int x = 1;
            int base = y * acols + x;

            for(; x < acols - 1; ++x, ++base )
            {
                if( adata[base] > accThreshold &&
                    adata[base] > adata[base-1] && adata[base] >= adata[base+1] &&
                    adata[base] > adata[base-acols] && adata[base] >= adata[base+acols] )
                    centersLocal.push_back(base);
            }
        }

        if(!centersLocal.empty())
        {
            if(singleThread)
                centers = centersLocal;
            else
            {
                AutoLock alock(_lock);
                centers.insert(centers.end(), centersLocal.begin(), centersLocal.end());
            }
        }
    }

private:
    const Mat &accum;
    std::vector<int> &centers;
    int accThreshold;

    int acols, arows;
    const int *adata;
    Mutex& _lock;
};

template<typename T>
static bool CheckDistance(const std::vector<T> &circles, size_t endIdx, const T& circle, float minDist2)
{
    bool goodPoint = true;
    for (uint j = 0; j < endIdx; ++j)
    {
        T pt = circles[j];
        float distX = circle[0] - pt[0], distY = circle[1] - pt[1];
        if (distX * distX + distY * distY < minDist2)
        {
            goodPoint = false;
            break;
        }
    }
    return goodPoint;
}

static void GetCircleCenters(const std::vector<int> &centers, std::vector<Vec3f> &circles, int acols, float minDist, float dr)
{
    size_t centerCnt = centers.size();
    float minDist2 = minDist * minDist;
    for (size_t i = 0; i < centerCnt; ++i)
    {
        int center = centers[i];
        int y = center / acols;
        int x = center - y * acols;
        Vec3f circle = Vec3f((x + 0.5f) * dr, (y + 0.5f) * dr, 0);

        bool goodPoint = CheckDistance(circles, circles.size(), circle, minDist2);
        if (goodPoint)
            circles.push_back(circle);
    }
}

static void GetCircleCenters(const std::vector<int> &centers, std::vector<Vec4f> &circles, int acols, float minDist, float dr)
{
    size_t centerCnt = centers.size();
    float minDist2 = minDist * minDist;
    for (size_t i = 0; i < centerCnt; ++i)
    {
        int center = centers[i];
        int y = center / acols;
        int x = center - y * acols;
        Vec4f circle = Vec4f((x + 0.5f) * dr, (y + 0.5f) * dr, 0, (float)center);

        bool goodPoint = CheckDistance(circles, circles.size(), circle, minDist2);
        if (goodPoint)
            circles.push_back(circle);
    }
}

template<typename T>
static void RemoveOverlaps(std::vector<T>& circles, float minDist)
{
    if (circles.size() <= 1u)
        return;
    float minDist2 = minDist * minDist;
    size_t endIdx = 1;
    for (size_t i = 1; i < circles.size(); ++i)
    {
        T circle = circles[i];
        if (CheckDistance(circles, endIdx, circle, minDist2))
        {
            circles[endIdx] = circle;
            ++endIdx;
        }
    }
    circles.resize(endIdx);
}

static void CreateCircles(const std::vector<EstimatedCircle>& circlesEst, std::vector<Vec3f>& circles)
{
    std::transform(circlesEst.begin(), circlesEst.end(), std::back_inserter(circles), GetCircle);
}

static void CreateCircles(const std::vector<EstimatedCircle>& circlesEst, std::vector<Vec4f>& circles)
{
    std::transform(circlesEst.begin(), circlesEst.end(), std::back_inserter(circles), GetCircle4f);
}

template<class NZPoints>
class HoughCircleEstimateRadiusInvoker : public ParallelLoopBody
{
public:
    HoughCircleEstimateRadiusInvoker(const NZPoints &_nz, int _nzSz, const std::vector<int> &_centers, std::vector<EstimatedCircle> &_circlesEst,
                                     int _acols, int _accThreshold, int _minRadius, int _maxRadius,
                                     float _dp, Mutex& _mutex) :
        nz(_nz), nzSz(_nzSz), centers(_centers), circlesEst(_circlesEst), acols(_acols), accThreshold(_accThreshold),
        minRadius(_minRadius), maxRadius(_maxRadius), dr(_dp), _lock(_mutex)
    {
        minRadius2 = (float)minRadius * minRadius;
        maxRadius2 = (float)maxRadius * maxRadius;
        centerSz = (int)centers.size();
        CV_Assert(nzSz > 0);
    }

    ~HoughCircleEstimateRadiusInvoker() {}

protected:
    inline int filterCircles(const Point2f& curCenter, float* ddata) const;

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        std::vector<EstimatedCircle> circlesLocal;
        const int nBinsPerDr = 10;
        int nBins = cvRound((maxRadius - minRadius)/dr*nBinsPerDr);
        AutoBuffer<int> bins(nBins);
        AutoBuffer<float> distBuf(nzSz), distSqrtBuf(nzSz);
        float *ddata = distBuf.data();
        float *dSqrtData = distSqrtBuf.data();

        bool singleThread = (boundaries == Range(0, centerSz));
        int i = boundaries.start;

        // For each found possible center
        // Estimate radius and check support
        for(; i < boundaries.end; ++i)
        {
            int ofs = centers[i];
            int y = ofs / acols;
            int x = ofs - y * acols;

            //Calculate circle's center in pixels
            Point2f curCenter = Point2f((x + 0.5f) * dr, (y + 0.5f) * dr);
            int nzCount = filterCircles(curCenter, ddata);

            int maxCount = 0;
            float rBest = 0;
            if(nzCount)
            {
                Mat_<float> distMat(1, nzCount, ddata);
                Mat_<float> distSqrtMat(1, nzCount, dSqrtData);
                sqrt(distMat, distSqrtMat);

                memset(bins.data(), 0, sizeof(bins[0])*bins.size());
                for(int k = 0; k < nzCount; k++)
                {
                    int bin = std::max(0, std::min(nBins-1, cvRound((dSqrtData[k] - minRadius)/dr*nBinsPerDr)));
                    bins[bin]++;
                }

                for(int j = nBins - 1; j > 0; j--)
                {
                    if(bins[j])
                    {
                        int upbin = j;
                        int curCount = 0;
                        for(; j > upbin - nBinsPerDr && j >= 0; j--)
                        {
                            curCount += bins[j];
                        }
                        float rCur = (upbin + j)/2.f /nBinsPerDr * dr + minRadius;
                        if((curCount * rBest >= maxCount * rCur) ||
                            (rBest < FLT_EPSILON && curCount >= maxCount))
                        {
                            rBest = rCur;
                            maxCount = curCount;
                        }
                    }
                }
            }

            // Check if the circle has enough support
            if(maxCount > accThreshold)
            {
                circlesLocal.push_back(EstimatedCircle(Vec3f(curCenter.x, curCenter.y, rBest), maxCount));
            }
        }

        if(!circlesLocal.empty())
        {
            std::sort(circlesLocal.begin(), circlesLocal.end(), cmpAccum);
            if(singleThread)
            {
                std::swap(circlesEst, circlesLocal);
            }
            else
            {
                AutoLock alock(_lock);
                if (circlesEst.empty())
                    std::swap(circlesEst, circlesLocal);
                else
                    circlesEst.insert(circlesEst.end(), circlesLocal.begin(), circlesLocal.end());
            }
        }
    }

private:
    const NZPoints &nz;
    int nzSz;
    const std::vector<int> &centers;
    std::vector<EstimatedCircle> &circlesEst;
    int acols, accThreshold, minRadius, maxRadius;
    float dr;
    int centerSz;
    float minRadius2, maxRadius2;
    Mutex& _lock;
};

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointList>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Point* nz_ = &nz[0];
    int j = 0;
#if CV_SIMD
    {
        const v_float32 v_minRadius2 = vx_setall_f32(minRadius2);
        const v_float32 v_maxRadius2 = vx_setall_f32(maxRadius2);

        v_float32 v_curCenterX = vx_setall_f32(curCenter.x);
        v_float32 v_curCenterY = vx_setall_f32(curCenter.y);

        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[v_float32::nlanes];
        int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[v_int32::nlanes];
        for(; j <= nzSz - v_float32::nlanes; j += v_float32::nlanes)
        {
            v_float32 v_nzX, v_nzY;
            v_load_deinterleave((const float*)&nz_[j], v_nzX, v_nzY); // FIXIT use proper datatype

            v_float32 v_x = v_cvt_f32(v_reinterpret_as_s32(v_nzX));
            v_float32 v_y = v_cvt_f32(v_reinterpret_as_s32(v_nzY));

            v_float32 v_dx = v_x - v_curCenterX;
            v_float32 v_dy = v_y - v_curCenterY;

            v_float32 v_r2 = (v_dx * v_dx) + (v_dy * v_dy);
            v_float32 vmask = (v_minRadius2 <= v_r2) & (v_r2 <= v_maxRadius2);
            if (v_check_any(vmask))
            {
                v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                v_store_aligned(rbuf, v_r2);
                for (int i = 0; i < v_int32::nlanes; ++i)
                    if (rmask[i]) ddata[nzCount++] = rbuf[i];
            }
        }
    }
#endif

    // Estimate best radius
    for(; j < nzSz; ++j)
    {
        const Point pt = nz_[j];
        float _dx = curCenter.x - pt.x, _dy = curCenter.y - pt.y;
        float _r2 = _dx * _dx + _dy * _dy;

        if(minRadius2 <= _r2 && _r2 <= maxRadius2)
        {
            ddata[nzCount++] = _r2;
        }
    }
    return nzCount;
}

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointSet>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Mat_<uchar>& positions = nz.positions;

    const int rOuter = maxRadius + 1;
    const Range xOuter = Range(std::max(int(curCenter.x - rOuter), 0), std::min(int(curCenter.x + rOuter), positions.cols));
    const Range yOuter = Range(std::max(int(curCenter.y - rOuter), 0), std::min(int(curCenter.y + rOuter), positions.rows));

#if CV_SIMD
    float v_seq[v_float32::nlanes];
    for (int i = 0; i < v_float32::nlanes; ++i)
        v_seq[i] = (float)i;
    const v_float32 v_minRadius2 = vx_setall_f32(minRadius2);
    const v_float32 v_maxRadius2 = vx_setall_f32(maxRadius2);
    const v_float32 v_curCenterX_0123 = vx_setall_f32(curCenter.x) - vx_load(v_seq);
#endif

    for (int y = yOuter.start; y < yOuter.end; y++)
    {
        const uchar* ptr = positions.ptr(y, 0);
        float dy = curCenter.y - y;
        float dy2 = dy * dy;

        int x = xOuter.start;
#if CV_SIMD
        {
            const v_float32 v_dy2 = vx_setall_f32(dy2);
            const v_uint32 v_zero_u32 = vx_setall_u32(0);
            float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[v_float32::nlanes];
            int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[v_int32::nlanes];
            for (; x <= xOuter.end - v_float32::nlanes; x += v_float32::nlanes)
            {
                v_uint32 v_mask = vx_load_expand_q(ptr + x);
                v_mask = v_mask != v_zero_u32;

                v_float32 v_x = v_cvt_f32(vx_setall_s32(x));
                v_float32 v_dx = v_x - v_curCenterX_0123;

                v_float32 v_r2 = (v_dx * v_dx) + v_dy2;
                v_float32 vmask = (v_minRadius2 <= v_r2) & (v_r2 <= v_maxRadius2) & v_reinterpret_as_f32(v_mask);
                if (v_check_any(vmask))
                {
                    v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                    v_store_aligned(rbuf, v_r2);
                    for (int i = 0; i < v_int32::nlanes; ++i)
                        if (rmask[i]) ddata[nzCount++] = rbuf[i];
                }
            }
        }
#endif
        for (; x < xOuter.end; x++)
        {
            if (ptr[x])
            {
                float _dx = curCenter.x - x;
                float _r2 = _dx * _dx + dy2;
                if(minRadius2 <= _r2 && _r2 <= maxRadius2)
                {
                    ddata[nzCount++] = _r2;
                }
            }
        }
    }
    return nzCount;
}

template <typename CircleType>
static void HoughCirclesGradient(InputArray _image, OutputArray _circles,
                                 float dp, float minDist,
                                 int minRadius, int maxRadius, int cannyThreshold,
                                 int accThreshold, int maxCircles, int kernelSize, bool centersOnly)
{
    CV_Assert(kernelSize == -1 || kernelSize == 3 || kernelSize == 5 || kernelSize == 7);

    dp = max(dp, 1.f);
    float idp = 1.f/dp;

    Mat edges, dx, dy;

    Sobel(_image, dx, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE);
    Sobel(_image, dy, CV_16S, 0, 1, kernelSize, 1, 0, BORDER_REPLICATE);
    Canny(dx, dy, edges, std::max(1, cannyThreshold / 2), cannyThreshold, false);

    Mutex mtx;
    int numThreads = std::max(1, getNumThreads());
    std::vector<Mat> accumVec;
    NZPointSet nz(_image.rows(), _image.cols());
    parallel_for_(Range(0, edges.rows),
                  HoughCirclesAccumInvoker(edges, dx, dy, minRadius, maxRadius, idp, accumVec, nz, mtx),
                  numThreads);
    int nzSz = cv::countNonZero(nz.positions);
    if(nzSz <= 0)
        return;

    Mat accum = accumVec[0];
    for(size_t i = 1; i < accumVec.size(); i++)
    {
        accum += accumVec[i];
    }
    accumVec.clear();

    std::vector<int> centers;

    // 4 rows when multithreaded because there is a bit overhead
    // and on the other side there are some row ranges where centers are concentrated
    parallel_for_(Range(1, accum.rows - 1),
                  HoughCirclesFindCentersInvoker(accum, centers, accThreshold, mtx),
                  (numThreads > 1) ? ((accum.rows - 2) / 4) : 1);

    int centerCnt = (int)centers.size();
    if(centerCnt == 0)
        return;

    std::sort(centers.begin(), centers.end(), hough_cmp_gt(accum.ptr<int>()));

    std::vector<CircleType> circles;
    circles.reserve(256);
    if (centersOnly)
    {
        // Just get the circle centers
        GetCircleCenters(centers, circles, accum.cols, minDist, dp);
    }
    else
    {
        std::vector<EstimatedCircle> circlesEst;
        if (nzSz < maxRadius * maxRadius)
        {
            // Faster to use a list
            NZPointList nzList(nzSz);
            nz.toList(nzList);
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointList>(nzList, nzSz, centers, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }
        else
        {
            // Faster to use a matrix
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointSet>(nz, nzSz, centers, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }

        // Sort by accumulator value
        std::sort(circlesEst.begin(), circlesEst.end(), cmpAccum);

        // Create Circles
        CreateCircles(circlesEst, circles);
        RemoveOverlaps(circles, minDist);
    }

    if (circles.size() > 0)
    {
        int numCircles = std::min(maxCircles, int(circles.size()));
        Mat(1, numCircles, cv::traits::Type<CircleType>::value, &circles[0]).copyTo(_circles);
        return;
    }
}

static void HoughCircles( InputArray _image, OutputArray _circles,
                          int method, double dp, double minDist,
                          double param1, double param2,
                          int minRadius, int maxRadius,
                          int maxCircles, double param3 )
{
    CV_INSTRUMENT_REGION();

    int type = CV_32FC3;
    if( _circles.fixedType() )
    {
        type = _circles.type();
        CV_CheckType(type, type == CV_32FC3 || type == CV_32FC4, "Wrong type of output circles");
    }

    CV_Assert(!_image.empty() && _image.type() == CV_8UC1 && (_image.isMat() || _image.isUMat()));
    CV_Assert(_circles.isMat() || _circles.isVector());

    if( dp <= 0 || minDist <= 0 || param1 <= 0 || param2 <= 0)
        CV_Error( Error::StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    int cannyThresh = cvRound(param1), accThresh = cvRound(param2), kernelSize = cvRound(param3);

    minRadius = std::max(0, minRadius);

    if(maxCircles < 0)
        maxCircles = INT_MAX;

    bool centersOnly = (maxRadius < 0);

    if( maxRadius <= 0 )
        maxRadius = std::max( _image.rows(), _image.cols() );
    else if( maxRadius <= minRadius )
        maxRadius = minRadius + 2;

    switch( method )
    {
    case CV_HOUGH_GRADIENT:
        if (type == CV_32FC3)
            HoughCirclesGradient<Vec3f>(_image, _circles, (float)dp, (float)minDist,
                                        minRadius, maxRadius, cannyThresh,
                                        accThresh, maxCircles, kernelSize, centersOnly);
        else if (type == CV_32FC4)
            HoughCirclesGradient<Vec4f>(_image, _circles, (float)dp, (float)minDist,
                                        minRadius, maxRadius, cannyThresh,
                                        accThresh, maxCircles, kernelSize, centersOnly);
        else
            CV_Error(Error::StsError, "Internal error");
        break;
    default:
        CV_Error( Error::StsBadArg, "Unrecognized method id. Actually only CV_HOUGH_GRADIENT is supported." );
    }
}

void HoughCircles( InputArray _image, OutputArray _circles,
                   int method, double dp, double minDist,
                   double param1, double param2,
                   int minRadius, int maxRadius )
{
    HoughCircles(_image, _circles, method, dp, minDist, param1, param2, minRadius, maxRadius, -1, 3);
}
} // \namespace cv


/* Wrapper function for standard hough transform */
CV_IMPL CvSeq*
cvHoughLines2( CvArr* src_image, void* lineStorage, int method,
               double rho, double theta, int threshold,
               double param1, double param2,
               double min_theta, double max_theta )
{
    cv::Mat image = cv::cvarrToMat(src_image);
    std::vector<cv::Vec2f> l2;
    std::vector<cv::Vec4i> l4;

    CvMat* mat = 0;
    CvSeq* lines = 0;
    CvSeq lines_header;
    CvSeqBlock lines_block;
    int lineType, elemSize;
    int linesMax = INT_MAX;
    int iparam1, iparam2;

    if( !lineStorage )
        CV_Error(cv::Error::StsNullPtr, "NULL destination" );

    if( rho <= 0 || theta <= 0 || threshold <= 0 )
        CV_Error( cv::Error::StsOutOfRange, "rho, theta and threshold must be positive" );

    if( method != CV_HOUGH_PROBABILISTIC )
    {
        lineType = CV_32FC2;
        elemSize = sizeof(float)*2;
    }
    else
    {
        lineType = CV_32SC4;
        elemSize = sizeof(int)*4;
    }

    bool isStorage = isStorageOrMat(lineStorage);

    if( isStorage )
    {
        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
    }
    else
    {
        mat = (CvMat*)lineStorage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        if( CV_MAT_TYPE( mat->type ) != lineType )
            CV_Error( CV_StsBadArg,
            "The destination matrix data type is inappropriate, see the manual" );

        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
        linesMax = lines->total;
        cvClearSeq( lines );
    }

    iparam1 = cvRound(param1);
    iparam2 = cvRound(param2);

    switch( method )
    {
    case CV_HOUGH_STANDARD:
        HoughLinesStandard( image, l2, CV_32FC2, (float)rho,
                (float)theta, threshold, linesMax, min_theta, max_theta );
        break;
    case CV_HOUGH_MULTI_SCALE:
        HoughLinesSDiv( image, l2, CV_32FC2, (float)rho, (float)theta,
                threshold, iparam1, iparam2, linesMax, min_theta, max_theta );
        break;
    case CV_HOUGH_PROBABILISTIC:
        HoughLinesProbabilistic( image, (float)rho, (float)theta,
                threshold, iparam1, iparam2, l4, linesMax );
        break;
    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }

    int nlines = (int)(l2.size() + l4.size());

    if( !isStorage )
    {
        if( mat->cols > mat->rows )
            mat->cols = nlines;
        else
            mat->rows = nlines;
    }

    if( nlines )
    {
        cv::Mat lx = method == CV_HOUGH_STANDARD || method == CV_HOUGH_MULTI_SCALE ?
            cv::Mat(nlines, 1, CV_32FC2, &l2[0]) : cv::Mat(nlines, 1, CV_32SC4, &l4[0]);

        if (isStorage)
        {
            cvSeqPushMulti(lines, lx.ptr(), nlines);
        }
        else
        {
            cv::Mat dst(nlines, 1, lx.type(), mat->data.ptr);
            lx.copyTo(dst);
        }
    }

    if( isStorage )
        return lines;
    return 0;
}


CV_IMPL CvSeq*
cvHoughCircles( CvArr* src_image, void* circle_storage,
                int method, double dp, double min_dist,
                double param1, double param2,
                int min_radius, int max_radius )
{
    CvSeq* circles = NULL;
    int circles_max = INT_MAX;
    cv::Mat src = cv::cvarrToMat(src_image), circles_mat;

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    bool isStorage = isStorageOrMat(circle_storage);

    if(isStorage)
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else
    {
        CvSeq circles_header;
        CvSeqBlock circles_block;
        CvMat *mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
                      "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }

    cv::HoughCircles(src, circles_mat, method, dp, min_dist, param1, param2, min_radius, max_radius, circles_max, 3);
    cvSeqPushMulti(circles, circles_mat.data, (int)circles_mat.total());
    return circles;
}

/* End of file. */
