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

#include "opencv2/core/openvx/ovx_defs.hpp"

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
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

#ifdef HAVE_OPENCL

struct Corner
{
    float val;
    short y;
    short x;

    bool operator < (const Corner & c) const
    // Ensure a fully deterministic result of the sort
    {  return (val > c.val) ? true : (val < c.val) ? false : (y > c.y) ? true : (y < c.y) ? false : (x > c.x); }
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
    size_t total, i, j, ncorners = 0, possibleCornersCount =
            std::max(1024, static_cast<int>(imgsize.area() * 0.1));
    bool haveMask = !_mask.empty();
    UMat corners_buffer(1, (int)possibleCornersCount + 1, CV_32FC2);
    CV_Assert(sizeof(Corner) == corners_buffer.elemSize());
    Mat tmpCorners;

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
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
                cornersarg = ocl::KernelArg::PtrWriteOnly(corners_buffer);

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

        k2.args(dbarg, (float)qualityLevel, cornersarg);

        if (!k2.runTask(false))
            return false;
    }

    // collect list of pointers to features - put them into temporary image
    {
        ocl::Kernel k("findCorners", ocl::imgproc::gftt_oclsrc,
                      format("-D OP_FIND_CORNERS%s", haveMask ? " -D HAVE_MASK" : ""));
        if (k.empty())
            return false;

        ocl::KernelArg eigarg = ocl::KernelArg::ReadOnlyNoSize(eig),
                cornersarg = ocl::KernelArg::PtrWriteOnly(corners_buffer),
                thresholdarg = ocl::KernelArg::PtrReadOnly(maxEigenValue);

        if (!haveMask)
            k.args(eigarg, cornersarg, eig.rows - 2, eig.cols - 2, thresholdarg,
                  (int)possibleCornersCount);
        else
        {
            UMat mask = _mask.getUMat();
            k.args(eigarg, ocl::KernelArg::ReadOnlyNoSize(mask),
                   cornersarg, eig.rows - 2, eig.cols - 2,
                   thresholdarg, (int)possibleCornersCount);
        }

        size_t globalsize[2] = { (size_t)eig.cols - 2, (size_t)eig.rows - 2 };
        if (!k.run(2, globalsize, NULL, false))
            return false;

        tmpCorners = corners_buffer.getMat(ACCESS_RW);
        total = std::min<size_t>(tmpCorners.at<Vec2i>(0, 0)[0], possibleCornersCount);
        if (total == 0)
        {
            _corners.release();
            return true;
        }
    }

    Corner* corner_ptr = tmpCorners.ptr<Corner>() + 1;
    std::sort(corner_ptr, corner_ptr + total);

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
            const Corner & c = corner_ptr[i];
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
            const Corner & c = corner_ptr[i];

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

#ifdef HAVE_OPENVX
struct VxKeypointsComparator
{
    bool operator () (const vx_keypoint_t& a, const vx_keypoint_t& b)
    {
        return a.strength > b.strength;
    }
};

static bool openvx_harris(Mat image, OutputArray _corners,
                          int _maxCorners, double _qualityLevel, double _minDistance,
                          int _blockSize, double _harrisK)
{
    using namespace ivx;

    if(image.type() != CV_8UC1) return false;

    //OpenVX implementations don't have to provide other sizes
    if(!(_blockSize == 3 || _blockSize == 5 || _blockSize == 7)) return false;

    try
    {
        Context context = Context::create();

        Image ovxImage = Image::createFromHandle(context, Image::matTypeToFormat(image.type()),
                                                 Image::createAddressing(image), image.data);
        //The minimum threshold which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
        //set to 0, we'll filter it later by threshold
        ivx::Scalar strengthThresh = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, 0);

        //The gradient window size to use on the input.
        vx_int32 gradientSize = 3;

        //The block window size used to compute the harris corner score
        vx_int32 blockSize = _blockSize;

        //The scalar sensitivity threshold k from the Harris-Stephens equation
        ivx::Scalar sensivity = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, _harrisK);

        //The radial Euclidean distance for non-maximum suppression
        ivx::Scalar minDistance = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, _minDistance);

        vx_size capacity = image.cols * image.rows;
        Array corners = Array::create(context, VX_TYPE_KEYPOINT, capacity);
        ivx::Scalar numCorners = ivx::Scalar::create<VX_TYPE_SIZE>(context, 0);

        IVX_CHECK_STATUS(vxuHarrisCorners(context, ovxImage, strengthThresh, minDistance, sensivity,
                                          gradientSize, blockSize, corners, numCorners));

        std::vector<vx_keypoint_t> vxKeypoints;
        corners.copyTo(vxKeypoints);

        std::sort(vxKeypoints.begin(), vxKeypoints.end(), VxKeypointsComparator());

        vx_float32 maxStrength = 0.0f;
        if(vxKeypoints.size() > 0)
            maxStrength = vxKeypoints[0].strength;
        size_t maxKeypoints = min((size_t)_maxCorners, vxKeypoints.size());
        std::vector<Point2f> keypoints;
        keypoints.reserve(maxKeypoints);
        for(size_t i = 0; i < maxKeypoints; i++)
        {
            vx_keypoint_t kp = vxKeypoints[i];
            if(kp.strength < maxStrength*_qualityLevel) break;
            keypoints.push_back(Point2f((float)kp.x, (float)kp.y));
        }

        Mat(keypoints).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        ovxImage.swapHandle();
#endif
    }
    catch (RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}

#endif

}

void cv::goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                              int maxCorners, double qualityLevel, double minDistance,
                              InputArray _mask, int blockSize,
                              bool useHarrisDetector, double harrisK )
{
    CV_INSTRUMENT_REGION()

    CV_Assert( qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0 );
    CV_Assert( _mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)) );

    CV_OCL_RUN(_image.dims() <= 2 && _image.isUMat(),
               ocl_goodFeaturesToTrack(_image, _corners, maxCorners, qualityLevel, minDistance,
                                    _mask, blockSize, useHarrisDetector, harrisK))

    Mat image = _image.getMat(), eig, tmp;
    if (image.empty())
    {
        _corners.release();
        return;
    }

    // Disabled due to bad accuracy
    CV_OVX_RUN(false && useHarrisDetector && _mask.empty(),
               openvx_harris(image, _corners, maxCorners, qualityLevel, minDistance, blockSize, harrisK))

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

    std::vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr() );

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
            {
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
