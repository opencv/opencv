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
#include "opencl_kernels_imgproc.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <functional>

namespace cv
{

struct greaterThanPtr :
        public std::binary_function<const float *, const float *, bool>
{
    bool operator () (const float * a, const float * b) const
    { return *a > *b; }
};

#ifdef HAVE_OPENCL

struct Corner
{
    float val;
    short y;
    short x;

    bool operator < (const Corner & c) const
    {  return val > c.val; }
};

static bool ocl_goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray _mask, int blockSize,
                                     bool useHarrisDetector, double harrisK )
{
    UMat eig, maxEigenValue;
    if( useHarrisDetector )
        cornerHarris( _image, eig, blockSize, 3, harrisK );
    else
        cornerMinEigenVal( _image, eig, blockSize, 3 );

    Size imgsize = _image.size();
    std::vector<Corner> tmpCorners;
    size_t total, i, j, ncorners = 0, possibleCornersCount =
            std::max(1024, static_cast<int>(imgsize.area() * 0.1));
    bool haveMask = !_mask.empty();

    // find threshold
    {
        CV_Assert(eig.type() == CV_32FC1);
        int dbsize = ocl::Device::getDefault().maxComputeUnits();
        size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

        int wgs2_aligned = 1;
        while (wgs2_aligned < (int)wgs)
            wgs2_aligned <<= 1;
        wgs2_aligned >>= 1;

        ocl::Kernel k("maxEigenVal", ocl::imgproc::gftt_oclsrc,
                      format("-D OP_MAX_EIGEN_VAL -D WGS=%d -D groupnum=%d -D WGS2_ALIGNED=%d%s",
                             (int)wgs, dbsize, wgs2_aligned, haveMask ? " -D HAVE_MASK" : ""));
        if (k.empty())
            return false;

        UMat mask = _mask.getUMat();
        maxEigenValue.create(1, dbsize, CV_32FC1);

        ocl::KernelArg eigarg = ocl::KernelArg::ReadOnlyNoSize(eig),
                dbarg = ocl::KernelArg::PtrWriteOnly(maxEigenValue),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);

        if (haveMask)
            k.args(eigarg, eig.cols, (int)eig.total(), dbarg, maskarg);
        else
            k.args(eigarg, eig.cols, (int)eig.total(), dbarg);

        size_t globalsize = dbsize * wgs;
        if (!k.run(1, &globalsize, &wgs, false))
            return false;

        ocl::Kernel k2("maxEigenValTask", ocl::imgproc::gftt_oclsrc,
                       format("-D OP_MAX_EIGEN_VAL -D WGS=%d -D WGS2_ALIGNED=%d -D groupnum=%d",
                              wgs, wgs2_aligned, dbsize));
        if (k2.empty())
            return false;

        k2.args(dbarg, (float)qualityLevel);

        if (!k2.runTask(false))
            return false;
    }

    // collect list of pointers to features - put them into temporary image
    {
        ocl::Kernel k("findCorners", ocl::imgproc::gftt_oclsrc,
                      format("-D OP_FIND_CORNERS%s", haveMask ? " -D HAVE_MASK" : ""));
        if (k.empty())
            return false;

        UMat counter(1, 1, CV_32SC1, Scalar::all(0)),
            corners(1, (int)possibleCornersCount, CV_32FC2, Scalar::all(-1));
        CV_Assert(sizeof(Corner) == corners.elemSize());

        ocl::KernelArg eigarg = ocl::KernelArg::ReadOnlyNoSize(eig),
                cornersarg = ocl::KernelArg::PtrWriteOnly(corners),
                counterarg = ocl::KernelArg::PtrReadWrite(counter),
                thresholdarg = ocl::KernelArg::PtrReadOnly(maxEigenValue);

        if (!haveMask)
            k.args(eigarg, cornersarg, counterarg,
                   eig.rows - 2, eig.cols - 2, thresholdarg,
                   (int)possibleCornersCount);
        else
        {
            UMat mask = _mask.getUMat();
            k.args(eigarg, ocl::KernelArg::ReadOnlyNoSize(mask),
                   cornersarg, counterarg, eig.rows - 2, eig.cols - 2,
                   thresholdarg, (int)possibleCornersCount);
        }

        size_t globalsize[2] = { eig.cols - 2, eig.rows - 2 };
        if (!k.run(2, globalsize, NULL, false))
            return false;

        total = std::min<size_t>(counter.getMat(ACCESS_READ).at<int>(0, 0), possibleCornersCount);
        if (total == 0)
        {
            _corners.release();
            return true;
        }

        tmpCorners.resize(total);

        Mat mcorners(1, (int)total, CV_32FC2, &tmpCorners[0]);
        corners.colRange(0, (int)total).copyTo(mcorners);
    }
    std::sort(tmpCorners.begin(), tmpCorners.end());

    std::vector<Point2f> corners;
    corners.reserve(total);

    if (minDistance >= 1)
    {
         // Partition the image into larger grids
        int w = imgsize.width, h = imgsize.height;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);
        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            const Corner & c = tmpCorners[i];
            bool good = true;

            int x_cell = c.x / cell_size;
            int y_cell = c.y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for( int yy = y1; yy <= y2; yy++ )
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector<Point2f> &m = grid[yy * grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = c.x - m[j].x;
                            float dy = c.y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)c.x, (float)c.y));

                corners.push_back(Point2f((float)c.x, (float)c.y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            const Corner & c = tmpCorners[i];

            corners.push_back(Point2f((float)c.x, (float)c.y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
    return true;
}

#endif

}

void cv::goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                              int maxCorners, double qualityLevel, double minDistance,
                              InputArray _mask, int blockSize,
                              bool useHarrisDetector, double harrisK )
{
    CV_Assert( qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0 );
    CV_Assert( _mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)) );

    CV_OCL_RUN(_image.dims() <= 2 && _image.isUMat(),
               ocl_goodFeaturesToTrack(_image, _corners, maxCorners, qualityLevel, minDistance,
                                    _mask, blockSize, useHarrisDetector, harrisK))

    Mat image = _image.getMat(), eig, tmp;
    if( useHarrisDetector )
        cornerHarris( image, eig, blockSize, 3, harrisK );
    else
        cornerMinEigenVal( image, eig, blockSize, 3 );

    double maxVal = 0;
    minMaxLoc( eig, 0, &maxVal, 0, 0, _mask );
    threshold( eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO );
    dilate( eig, tmp, Mat());

    Size imgsize = image.size();
    std::vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    Mat mask = _mask.getMat();
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr() );

    std::vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (minDistance >= 1)
    {
         // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(Point2f((float)x, (float)y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

CV_IMPL void
cvGoodFeaturesToTrack( const void* _image, void*, void*,
                       CvPoint2D32f* _corners, int *_corner_count,
                       double quality_level, double min_distance,
                       const void* _maskImage, int block_size,
                       int use_harris, double harris_k )
{
    cv::Mat image = cv::cvarrToMat(_image), mask;
    std::vector<cv::Point2f> corners;

    if( _maskImage )
        mask = cv::cvarrToMat(_maskImage);

    CV_Assert( _corners && _corner_count );
    cv::goodFeaturesToTrack( image, corners, *_corner_count, quality_level,
        min_distance, mask, block_size, use_harris != 0, harris_k );

    size_t i, ncorners = corners.size();
    for( i = 0; i < ncorners; i++ )
        _corners[i] = corners[i];
    *_corner_count = (int)ncorners;
}

/* End of file. */
