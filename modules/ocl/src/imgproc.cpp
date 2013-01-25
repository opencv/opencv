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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Rock Li, Rock.Li@amd.com
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wu Zailong, bullet@yeah.net
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
#include <iomanip>

using namespace cv;
using namespace cv::ocl;
using namespace std;

#if !defined (HAVE_OPENCL)


void cv::ocl::meanShiftFiltering(const oclMat &, oclMat &, int, int, TermCriteria)
{
    throw_nogpu();
}
void cv::ocl::meanShiftProc(const oclMat &, oclMat &, oclMat &, int, int, TermCriteria)
{
    throw_nogpu();
}
double cv::ocl::threshold(const oclMat &, oclMat &, double, int)
{
    throw_nogpu();
    return 0.0;
}
void cv::ocl::resize(const oclMat &, oclMat &, Size, double, double, int)
{
    throw_nogpu();
}
void cv::ocl::remap(const oclMat &, oclMat &, oclMat &, oclMat &, int, int , const Scalar &)
{
    throw_nogpu();
}

void cv::ocl::copyMakeBorder(const oclMat &, oclMat &, int, int, int, int, const Scalar &)
{
    throw_nogpu();
}
void cv::ocl::warpAffine(const oclMat &, oclMat &, const Mat &, Size, int)
{
    throw_nogpu();
}
void cv::ocl::warpPerspective(const oclMat &, oclMat &, const Mat &, Size, int)
{
    throw_nogpu();
}
void cv::ocl::integral(const oclMat &, oclMat &, oclMat &)
{
    throw_nogpu();
}
void cv::ocl::calcHist(const oclMat &, oclMat &hist)
{
    throw_nogpu();
}
void cv::ocl::bilateralFilter(const oclMat &, oclMat &, int, double, double, int)
{
    throw_nogpu();
}
void cv::ocl::convolve(const oclMat &, const oclMat &, oclMat &)
{
    throw_nogpu();
}

#else /* !defined (HAVE_OPENCL) */

namespace cv
{
    namespace ocl
    {

        ////////////////////////////////////OpenCL kernel strings//////////////////////////
        extern const char *meanShift;
        extern const char *imgproc_copymakeboder;
        extern const char *imgproc_median;
        extern const char *imgproc_threshold;
        extern const char *imgproc_resize;
        extern const char *imgproc_remap;
        extern const char *imgproc_warpAffine;
        extern const char *imgproc_warpPerspective;
        extern const char *imgproc_integral_sum;
        extern const char *imgproc_integral;
        extern const char *imgproc_histogram;
        extern const char *imgproc_bilateral;
        extern const char *imgproc_calcHarris;
        extern const char *imgproc_calcMinEigenVal;
        extern const char *imgproc_convolve;
        ////////////////////////////////////OpenCL call wrappers////////////////////////////

        template <typename T> struct index_and_sizeof;
        template <> struct index_and_sizeof<char>
        {
            enum { index = 1 };
        };
        template <> struct index_and_sizeof<unsigned char>
        {
            enum { index = 2 };
        };
        template <> struct index_and_sizeof<short>
        {
            enum { index = 3 };
        };
        template <> struct index_and_sizeof<unsigned short>
        {
            enum { index = 4 };
        };
        template <> struct index_and_sizeof<int>
        {
            enum { index = 5 };
        };
        template <> struct index_and_sizeof<float>
        {
            enum { index = 6 };
        };
        template <> struct index_and_sizeof<double>
        {
            enum { index = 7 };
        };

        /////////////////////////////////////////////////////////////////////////////////////
        // threshold

        typedef void (*gpuThresh_t)(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type);

        static void threshold_8u(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type)
        {
            CV_Assert( (src.cols == dst.cols) && (src.rows == dst.rows) );
            Context *clCxt = src.clCxt;

            uchar thresh_uchar = cvFloor(thresh);
            uchar max_val = cvRound(maxVal);
            string kernelName = "threshold";

            size_t cols = (dst.cols + (dst.offset % 16) + 15) / 16;
            size_t bSizeX = 16, bSizeY = 16;
            size_t gSizeX = cols % bSizeX == 0 ? cols : (cols + bSizeX - 1) / bSizeX * bSizeX;
            size_t gSizeY = dst.rows;
            size_t globalThreads[3] = {gSizeX, gSizeY, 1};
            size_t localThreads[3] = {bSizeX, bSizeY, 1};

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair(sizeof(cl_mem), &src.data));
            args.push_back( make_pair(sizeof(cl_mem), &dst.data));
            args.push_back( make_pair(sizeof(cl_int), (void *)&src.offset));
            args.push_back( make_pair(sizeof(cl_int), (void *)&src.step));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.offset));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.step));
            args.push_back( make_pair(sizeof(cl_uchar), (void *)&thresh_uchar));
            args.push_back( make_pair(sizeof(cl_uchar), (void *)&max_val));
            args.push_back( make_pair(sizeof(cl_int), (void *)&type));
            openCLExecuteKernel(clCxt, &imgproc_threshold, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
        }

        static void threshold_32f(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type)
        {
            CV_Assert( (src.cols == dst.cols) && (src.rows == dst.rows) );
            Context *clCxt = src.clCxt;

            float thresh_f = thresh;
            float max_val = maxVal;
            int dst_offset = (dst.offset >> 2);
            int dst_step = (dst.step >> 2);
            int src_offset = (src.offset >> 2);
            int src_step = (src.step >> 2);

            string kernelName = "threshold";

            size_t cols = (dst.cols + (dst_offset & 3) + 3) / 4;
            //size_t cols = dst.cols;
            size_t bSizeX = 16, bSizeY = 16;
            size_t gSizeX = cols % bSizeX == 0 ? cols : (cols + bSizeX - 1) / bSizeX * bSizeX;
            size_t gSizeY = dst.rows;
            size_t globalThreads[3] = {gSizeX, gSizeY, 1};
            size_t localThreads[3] = {bSizeX, bSizeY, 1};

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair(sizeof(cl_mem), &src.data));
            args.push_back( make_pair(sizeof(cl_mem), &dst.data));
            args.push_back( make_pair(sizeof(cl_int), (void *)&src_offset));
            args.push_back( make_pair(sizeof(cl_int), (void *)&src_step));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst_offset));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst_step));
            args.push_back( make_pair(sizeof(cl_float), (void *)&thresh_f));
            args.push_back( make_pair(sizeof(cl_float), (void *)&max_val));
            args.push_back( make_pair(sizeof(cl_int), (void *)&type));
            openCLExecuteKernel(clCxt, &imgproc_threshold, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());

        }

        //threshold: support 8UC1 and 32FC1 data type and five threshold type
        double threshold(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type)
        {
            //TODO: These limitations shall be removed later.
            CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
            CV_Assert(type == THRESH_BINARY || type == THRESH_BINARY_INV || type == THRESH_TRUNC
                      || type == THRESH_TOZERO || type == THRESH_TOZERO_INV );

            static const gpuThresh_t gpuThresh_callers[2] = {threshold_8u, threshold_32f};

            dst.create( src.size(), src.type() );
            gpuThresh_callers[(src.type() == CV_32FC1)](src, dst, thresh, maxVal, type);

            return thresh;
        }
        ////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////   remap   //////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        void remap( const oclMat &src, oclMat &dst, oclMat &map1, oclMat &map2, int interpolation, int borderType, const Scalar &borderValue )
        {
            Context *clCxt = src.clCxt;
            CV_Assert(interpolation == INTER_LINEAR || interpolation == INTER_NEAREST
                      || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4);
            CV_Assert((map1.type() == CV_16SC2 && !map2.data) || (map1.type() == CV_32FC2 && !map2.data) || (map1.type() == CV_32FC1 && map2.type() == CV_32FC1));
            CV_Assert(!map2.data || map2.size() == map1.size());
            CV_Assert(dst.size() == map1.size());

            dst.create(map1.size(), src.type());


            string kernelName;

            if( map1.type() == CV_32FC2 && !map2.data )
            {
                if(interpolation == INTER_LINEAR && borderType == BORDER_CONSTANT)
                    kernelName = "remapLNFConstant";
                else if(interpolation == INTER_NEAREST && borderType == BORDER_CONSTANT)
                    kernelName = "remapNNFConstant";
            }
            else if(map1.type() == CV_16SC2 && !map2.data)
            {
                if(interpolation == INTER_LINEAR && borderType == BORDER_CONSTANT)
                    kernelName = "remapLNSConstant";
                else if(interpolation == INTER_NEAREST && borderType == BORDER_CONSTANT)
                    kernelName = "remapNNSConstant";

            }
            else if(map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
            {
                if(interpolation == INTER_LINEAR && borderType == BORDER_CONSTANT)
                    kernelName = "remapLNF1Constant";
                else if (interpolation == INTER_NEAREST && borderType == BORDER_CONSTANT)
                    kernelName = "remapNNF1Constant";
            }

            //int channels = dst.oclchannels();
            //int depth = dst.depth();
            //int type = src.type();
            size_t blkSizeX = 16, blkSizeY = 16;
            size_t glbSizeX;
            int cols = dst.cols;
            if(src.type() == CV_8UC1)
            {
                cols = (dst.cols + dst.offset % 4 + 3) / 4;
                glbSizeX = cols % blkSizeX == 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;

            }
            else if(src.type() == CV_32FC1 && interpolation == INTER_LINEAR)
            {
                cols = (dst.cols + (dst.offset >> 2) % 4 + 3) / 4;
                glbSizeX = cols % blkSizeX == 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;
            }
            else
            {
                glbSizeX = dst.cols % blkSizeX == 0 ? dst.cols : (dst.cols / blkSizeX + 1) * blkSizeX;

            }

            size_t glbSizeY = dst.rows % blkSizeY == 0 ? dst.rows : (dst.rows / blkSizeY + 1) * blkSizeY;
            size_t globalThreads[3] = {glbSizeX, glbSizeY, 1};
            size_t localThreads[3] = {blkSizeX, blkSizeY, 1};


            vector< pair<size_t, const void *> > args;
            if(map1.channels() == 2)
            {
                args.push_back( make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&map1.data));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&cols));
                if(src.clCxt -> impl -> double_support != 0)
                {
                    args.push_back( make_pair(sizeof(cl_double4), (void *)&borderValue));
                }
                else
                {
                    float borderFloat[4] = {(float)borderValue[0], (float)borderValue[1], (float)borderValue[2], (float)borderValue[3]};
                    args.push_back( make_pair(sizeof(cl_float4), (void *)&borderFloat));
                }
            }
            if(map1.channels() == 1)
            {
                args.push_back( make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&map1.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&map2.data));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.offset));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.step));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&map1.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&cols));
                if(src.clCxt -> impl -> double_support != 0)
                {
                    args.push_back( make_pair(sizeof(cl_double4), (void *)&borderValue));
                }
                else
                {
                    float borderFloat[4] = {(float)borderValue[0], (float)borderValue[1], (float)borderValue[2], (float)borderValue[3]};
                    args.push_back( make_pair(sizeof(cl_float4), (void *)&borderFloat));
                }
            }
            openCLExecuteKernel(clCxt, &imgproc_remap, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // resize

        static void resize_gpu( const oclMat &src, oclMat &dst, double fx, double fy, int interpolation)
        {
            CV_Assert( (src.channels() == dst.channels()) );
            Context *clCxt = src.clCxt;
            float ifx = 1. / fx;
            float ify = 1. / fy;
            double ifx_d = 1. / fx;
            double ify_d = 1. / fy;
            int srcStep_in_pixel = src.step1() / src.oclchannels();
            int srcoffset_in_pixel = src.offset / src.elemSize();
            int dstStep_in_pixel = dst.step1() / dst.oclchannels();
            int dstoffset_in_pixel = dst.offset / dst.elemSize();
            //printf("%d %d\n",src.step1() , dst.elemSize());
            string kernelName;
            if(interpolation == INTER_LINEAR)
                kernelName = "resizeLN";
            else if(interpolation == INTER_NEAREST)
                kernelName = "resizeNN";

            //TODO: improve this kernel
            size_t blkSizeX = 16, blkSizeY = 16;
            size_t glbSizeX;
            if(src.type() == CV_8UC1)
            {
                size_t cols = (dst.cols + dst.offset % 4 + 3) / 4;
                glbSizeX = cols % blkSizeX == 0 && cols != 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;
            }
            else
            {
                glbSizeX = dst.cols % blkSizeX == 0 && dst.cols != 0 ? dst.cols : (dst.cols / blkSizeX + 1) * blkSizeX;
            }
            size_t glbSizeY = dst.rows % blkSizeY == 0 && dst.rows != 0 ? dst.rows : (dst.rows / blkSizeY + 1) * blkSizeY;
            size_t globalThreads[3] = {glbSizeX, glbSizeY, 1};
            size_t localThreads[3] = {blkSizeX, blkSizeY, 1};

            vector< pair<size_t, const void *> > args;
            if(interpolation == INTER_NEAREST)
            {
                args.push_back( make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dstoffset_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&srcoffset_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dstStep_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&srcStep_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
                if(src.clCxt -> impl -> double_support != 0)
                {
                    args.push_back( make_pair(sizeof(cl_double), (void *)&ifx_d));
                    args.push_back( make_pair(sizeof(cl_double), (void *)&ify_d));
                }
                else
                {
                    args.push_back( make_pair(sizeof(cl_float), (void *)&ifx));
                    args.push_back( make_pair(sizeof(cl_float), (void *)&ify));
                }
            }
            else
            {
                args.push_back( make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dstoffset_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&srcoffset_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dstStep_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&srcStep_in_pixel));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back( make_pair(sizeof(cl_float), (void *)&ifx));
                args.push_back( make_pair(sizeof(cl_float), (void *)&ify));
            }

            openCLExecuteKernel(clCxt, &imgproc_resize, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
        }


        void resize(const oclMat &src, oclMat &dst, Size dsize,
                    double fx, double fy, int interpolation)
        {
            CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4
                      || src.type() == CV_32FC1 || src.type() == CV_32FC3 || src.type() == CV_32FC4);
            CV_Assert(interpolation == INTER_LINEAR || interpolation == INTER_NEAREST);
            CV_Assert( src.size().area() > 0 );
            CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );

            if(!(dsize == Size()) && (fx > 0 && fy > 0))
            {
                if(dsize.width != (int)(src.cols * fx) || dsize.height != (int)(src.rows * fy))
                {
                    CV_Error(CV_StsUnmatchedSizes, "invalid dsize and fx, fy!");
                }
            }
            if( dsize == Size() )
            {
                dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
            }
            else
            {
                fx = (double)dsize.width / src.cols;
                fy = (double)dsize.height / src.rows;
            }

            dst.create(dsize, src.type());

            if( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR )
            {
                resize_gpu( src, dst, fx, fy, interpolation);
                return;
            }
            CV_Error(CV_StsUnsupportedFormat, "Non-supported interpolation method");
        }


        ////////////////////////////////////////////////////////////////////////
        // medianFilter
        void medianFilter(const oclMat &src, oclMat &dst, int m)
        {
            CV_Assert( m % 2 == 1 && m > 1 );
            CV_Assert( m <= 5 || src.depth() == CV_8U );
            CV_Assert( src.cols <= dst.cols && src.rows <= dst.rows );

            if(src.data == dst.data)
            {
                oclMat src1;
                src.copyTo(src1);
                return medianFilter(src1, dst, m);
            }

            int srcStep = src.step1() / src.oclchannels();
            int dstStep = dst.step1() / dst.oclchannels();
            int srcOffset = src.offset / src.oclchannels() / src.elemSize1();
            int dstOffset = dst.offset / dst.oclchannels() / dst.elemSize1();

            Context *clCxt = src.clCxt;
            string kernelName = "medianFilter";


            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data));
            args.push_back( make_pair( sizeof(cl_int), (void *)&srcOffset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dstOffset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&src.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&srcStep));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dstStep));

            size_t globalThreads[3] = {(src.cols + 18) / 16 * 16, (src.rows + 15) / 16 * 16, 1};
            size_t localThreads[3] = {16, 16, 1};

            if(m == 3)
            {
                string kernelName = "medianFilter3";
                openCLExecuteKernel(clCxt, &imgproc_median, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
            }
            else if(m == 5)
            {
                string kernelName = "medianFilter5";
                openCLExecuteKernel(clCxt, &imgproc_median, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
            }
            else
            {
                CV_Error(CV_StsUnsupportedFormat, "Non-supported filter length");
                //string kernelName = "medianFilter";
                //args.push_back( make_pair( sizeof(cl_int),(void*)&m));

                //openCLExecuteKernel(clCxt,&imgproc_median,kernelName,globalThreads,localThreads,args,src.oclchannels(),-1);
            }

        }

        ////////////////////////////////////////////////////////////////////////
        // copyMakeBorder
        void copyMakeBorder(const oclMat &src, oclMat &dst, int top, int bottom, int left, int right, int bordertype, const Scalar &scalar)
        {
            //CV_Assert(src.oclchannels() != 2);
            CV_Assert(top >= 0 && bottom >= 0 && left >= 0 && right >= 0);
            if((dst.cols != dst.wholecols) || (dst.rows != dst.wholerows)) //has roi
            {
                if(((bordertype & cv::BORDER_ISOLATED) == 0) &&
                        (bordertype != cv::BORDER_CONSTANT) &&
                        (bordertype != cv::BORDER_REPLICATE))
                {
                    CV_Error(CV_StsBadArg, "unsupported border type");
                }
            }
            bordertype &= ~cv::BORDER_ISOLATED;
            if((bordertype == cv::BORDER_REFLECT) || (bordertype == cv::BORDER_WRAP))
            {
                CV_Assert((src.cols >= left) && (src.cols >= right) && (src.rows >= top) && (src.rows >= bottom));
            }
            if(bordertype == cv::BORDER_REFLECT_101)
            {
                CV_Assert((src.cols > left) && (src.cols > right) && (src.rows > top) && (src.rows > bottom));
            }
            dst.create(src.rows + top + bottom, src.cols + left + right, src.type());
            int srcStep = src.step1() / src.oclchannels();
            int dstStep = dst.step1() / dst.oclchannels();
            int srcOffset = src.offset / src.elemSize();
            int dstOffset = dst.offset / dst.elemSize();
            int __bordertype[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101};
            const char *borderstr[] = {"BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101"};
            size_t bordertype_index;
            for(bordertype_index = 0; bordertype_index < sizeof(__bordertype) / sizeof(int); bordertype_index++)
            {
                if(__bordertype[bordertype_index] == bordertype)
                    break;
            }
            if(bordertype_index == sizeof(__bordertype) / sizeof(int))
            {
                CV_Error(CV_StsBadArg, "unsupported border type");
            }
            string kernelName = "copymakeborder";
            size_t localThreads[3] = {16, 16, 1};
            size_t globalThreads[3] = {(dst.cols + localThreads[0] - 1) / localThreads[0] *localThreads[0],
                                       (dst.rows + localThreads[1] - 1) / localThreads[1] *localThreads[1], 1
                                      };

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&src.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&srcStep));
            args.push_back( make_pair( sizeof(cl_int), (void *)&srcOffset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dstStep));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dstOffset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&top));
            args.push_back( make_pair( sizeof(cl_int), (void *)&left));
            char compile_option[64];
            union sc
            {
                cl_uchar4 uval;
                cl_char4  cval;
                cl_ushort4 usval;
                cl_short4 shval;
                cl_int4 ival;
                cl_float4 fval;
                cl_double4 dval;
            } val;
            switch(dst.depth())
            {
            case CV_8U:
                val.uval.s[0] = saturate_cast<uchar>(scalar.val[0]);
                val.uval.s[1] = saturate_cast<uchar>(scalar.val[1]);
                val.uval.s[2] = saturate_cast<uchar>(scalar.val[2]);
                val.uval.s[3] = saturate_cast<uchar>(scalar.val[3]);
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=uchar -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_uchar) , (void *)&val.uval.s[0] ));
                    if(((dst.offset & 3) == 0) && ((dst.cols & 3) == 0))
                    {
                        kernelName = "copymakeborder_C1_D0";
                        globalThreads[0] = (dst.cols / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
                    }
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=uchar4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_uchar4) , (void *)&val.uval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_8S:
                val.cval.s[0] = saturate_cast<char>(scalar.val[0]);
                val.cval.s[1] = saturate_cast<char>(scalar.val[1]);
                val.cval.s[2] = saturate_cast<char>(scalar.val[2]);
                val.cval.s[3] = saturate_cast<char>(scalar.val[3]);
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=char -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_char) , (void *)&val.cval.s[0] ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=char4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_char4) , (void *)&val.cval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_16U:
                val.usval.s[0] = saturate_cast<ushort>(scalar.val[0]);
                val.usval.s[1] = saturate_cast<ushort>(scalar.val[1]);
                val.usval.s[2] = saturate_cast<ushort>(scalar.val[2]);
                val.usval.s[3] = saturate_cast<ushort>(scalar.val[3]);
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=ushort -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_ushort) , (void *)&val.usval.s[0] ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=ushort4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_ushort4) , (void *)&val.usval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_16S:
                val.shval.s[0] = saturate_cast<short>(scalar.val[0]);
                val.shval.s[1] = saturate_cast<short>(scalar.val[1]);
                val.shval.s[2] = saturate_cast<short>(scalar.val[2]);
                val.shval.s[3] = saturate_cast<short>(scalar.val[3]);
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=short -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_short) , (void *)&val.shval.s[0] ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=short4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_short4) , (void *)&val.shval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_32S:
                val.ival.s[0] = saturate_cast<int>(scalar.val[0]);
                val.ival.s[1] = saturate_cast<int>(scalar.val[1]);
                val.ival.s[2] = saturate_cast<int>(scalar.val[2]);
                val.ival.s[3] = saturate_cast<int>(scalar.val[3]);
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=int -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_int) , (void *)&val.ival.s[0] ));
                    break;
                case 2:
                    sprintf(compile_option, "-D GENTYPE=int2 -D %s", borderstr[bordertype_index]);
                    cl_int2 i2val;
                    i2val.s[0] = val.ival.s[0];
                    i2val.s[1] = val.ival.s[1];
                    args.push_back( make_pair( sizeof(cl_int2) , (void *)&i2val ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=int4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_int4) , (void *)&val.ival ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_32F:
                val.fval.s[0] = scalar.val[0];
                val.fval.s[1] = scalar.val[1];
                val.fval.s[2] = scalar.val[2];
                val.fval.s[3] = scalar.val[3];
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=float -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_float) , (void *)&val.fval.s[0] ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=float4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_float4) , (void *)&val.fval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            case CV_64F:
                val.dval.s[0] = scalar.val[0];
                val.dval.s[1] = scalar.val[1];
                val.dval.s[2] = scalar.val[2];
                val.dval.s[3] = scalar.val[3];
                switch(dst.oclchannels())
                {
                case 1:
                    sprintf(compile_option, "-D GENTYPE=double -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_double) , (void *)&val.dval.s[0] ));
                    break;
                case 4:
                    sprintf(compile_option, "-D GENTYPE=double4 -D %s", borderstr[bordertype_index]);
                    args.push_back( make_pair( sizeof(cl_double4) , (void *)&val.dval ));
                    break;
                default:
                    CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
                }
                break;
            default:
                CV_Error(CV_StsUnsupportedFormat, "unknown depth");
            }

            openCLExecuteKernel(src.clCxt, &imgproc_copymakeboder, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
            //uchar* cputemp=new uchar[32*dst.wholerows];
            ////int* cpudata=new int[this->step*this->wholerows/sizeof(int)];
            //openCLSafeCall(clEnqueueReadBuffer(src.clCxt->impl->clCmdQueue, (cl_mem)dst.data, CL_TRUE,
            //						0, 32*dst.wholerows, cputemp, 0, NULL, NULL));
            //for(int i=0;i<dst.wholerows;i++)
            //{
            //	for(int j=0;j<dst.wholecols;j++)
            //	{
            //		cout<< (int)cputemp[i*32+j]<<" ";
            //	}
            //	cout<<endl;
            //}
            //delete []cputemp;
        }

        ////////////////////////////////////////////////////////////////////////
        // warp

        namespace
        {
#define F double

            void convert_coeffs(F *M)
            {
                double D = M[0] * M[4] - M[1] * M[3];
                D = D != 0 ? 1. / D : 0;
                double A11 = M[4] * D, A22 = M[0] * D;
                M[0] = A11;
                M[1] *= -D;
                M[3] *= -D;
                M[4] = A22;
                double b1 = -M[0] * M[2] - M[1] * M[5];
                double b2 = -M[3] * M[2] - M[4] * M[5];
                M[2] = b1;
                M[5] = b2;
            }

            double invert(double *M)
            {
#define Sd(y,x) (Sd[y*3+x])
#define Dd(y,x) (Dd[y*3+x])
#define det3(m)    (m(0,0)*(m(1,1)*m(2,2) - m(1,2)*m(2,1)) -  \
                    m(0,1)*(m(1,0)*m(2,2) - m(1,2)*m(2,0)) +  \
                    m(0,2)*(m(1,0)*m(2,1) - m(1,1)*m(2,0)))
                double *Sd = M;
                double *Dd = M;
                double d = det3(Sd);
                double result = 0;
                if( d != 0)
                {
                    double t[9];
                    result = d;
                    d = 1. / d;

                    t[0] = (Sd(1, 1) * Sd(2, 2) - Sd(1, 2) * Sd(2, 1)) * d;
                    t[1] = (Sd(0, 2) * Sd(2, 1) - Sd(0, 1) * Sd(2, 2)) * d;
                    t[2] = (Sd(0, 1) * Sd(1, 2) - Sd(0, 2) * Sd(1, 1)) * d;

                    t[3] = (Sd(1, 2) * Sd(2, 0) - Sd(1, 0) * Sd(2, 2)) * d;
                    t[4] = (Sd(0, 0) * Sd(2, 2) - Sd(0, 2) * Sd(2, 0)) * d;
                    t[5] = (Sd(0, 2) * Sd(1, 0) - Sd(0, 0) * Sd(1, 2)) * d;

                    t[6] = (Sd(1, 0) * Sd(2, 1) - Sd(1, 1) * Sd(2, 0)) * d;
                    t[7] = (Sd(0, 1) * Sd(2, 0) - Sd(0, 0) * Sd(2, 1)) * d;
                    t[8] = (Sd(0, 0) * Sd(1, 1) - Sd(0, 1) * Sd(1, 0)) * d;

                    Dd(0, 0) = t[0];
                    Dd(0, 1) = t[1];
                    Dd(0, 2) = t[2];
                    Dd(1, 0) = t[3];
                    Dd(1, 1) = t[4];
                    Dd(1, 2) = t[5];
                    Dd(2, 0) = t[6];
                    Dd(2, 1) = t[7];
                    Dd(2, 2) = t[8];
                }
                return result;
            }

            void warpAffine_gpu(const oclMat &src, oclMat &dst, F coeffs[2][3], int interpolation)
            {
                CV_Assert( (src.oclchannels() == dst.oclchannels()) );
                int srcStep = src.step1();
                int dstStep = dst.step1();
                float float_coeffs[2][3];
                cl_mem coeffs_cm;

                Context *clCxt = src.clCxt;
                string s[3] = {"NN", "Linear", "Cubic"};
                string kernelName = "warpAffine" + s[interpolation];


                if(src.clCxt -> impl -> double_support != 0)
                {
                    cl_int st;
                    coeffs_cm = clCreateBuffer( clCxt->impl->clContext, CL_MEM_READ_WRITE, sizeof(F) * 2 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(clCxt->impl->clCmdQueue, (cl_mem)coeffs_cm, 1, 0, sizeof(F) * 2 * 3, coeffs, 0, 0, 0));
                }
                else
                {
                    cl_int st;
                    for(int m = 0; m < 2; m++)
                        for(int n = 0; n < 3; n++)
                        {
                            float_coeffs[m][n] = coeffs[m][n];
                        }
                    coeffs_cm = clCreateBuffer( clCxt->impl->clContext, CL_MEM_READ_WRITE, sizeof(float) * 2 * 3, NULL, &st );
                    openCLSafeCall(clEnqueueWriteBuffer(clCxt->impl->clCmdQueue, (cl_mem)coeffs_cm, 1, 0, sizeof(float) * 2 * 3, float_coeffs, 0, 0, 0));

                }
                //TODO: improve this kernel
                size_t blkSizeX = 16, blkSizeY = 16;
                size_t glbSizeX;
                size_t cols;
                //if(src.type() == CV_8UC1 && interpolation != 2)
                if(src.type() == CV_8UC1 && interpolation != 2)
                {
                    cols = (dst.cols + dst.offset % 4 + 3) / 4;
                    glbSizeX = cols % blkSizeX == 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;
                }
                else
                {
                    cols = dst.cols;
                    glbSizeX = dst.cols % blkSizeX == 0 ? dst.cols : (dst.cols / blkSizeX + 1) * blkSizeX;
                }
                size_t glbSizeY = dst.rows % blkSizeY == 0 ? dst.rows : (dst.rows / blkSizeY + 1) * blkSizeY;
                size_t globalThreads[3] = {glbSizeX, glbSizeY, 1};
                size_t localThreads[3] = {blkSizeX, blkSizeY, 1};

                vector< pair<size_t, const void *> > args;

                args.push_back(make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back(make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back(make_pair(sizeof(cl_int), (void *)&srcStep));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dstStep));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back(make_pair(sizeof(cl_mem), (void *)&coeffs_cm));
                args.push_back(make_pair(sizeof(cl_int), (void *)&cols));

                openCLExecuteKernel(clCxt, &imgproc_warpAffine, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
                openCLSafeCall(clReleaseMemObject(coeffs_cm));
            }


            void warpPerspective_gpu(const oclMat &src, oclMat &dst, double coeffs[3][3], int interpolation)
            {
                CV_Assert( (src.oclchannels() == dst.oclchannels()) );
                int srcStep = src.step1();
                int dstStep = dst.step1();
                float float_coeffs[3][3];
                cl_mem coeffs_cm;

                Context *clCxt = src.clCxt;
                string s[3] = {"NN", "Linear", "Cubic"};
                string kernelName = "warpPerspective" + s[interpolation];

                if(src.clCxt -> impl -> double_support != 0)
                {
                    cl_int st;
                    coeffs_cm = clCreateBuffer( clCxt->impl->clContext, CL_MEM_READ_WRITE, sizeof(double) * 3 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(clCxt->impl->clCmdQueue, (cl_mem)coeffs_cm, 1, 0, sizeof(double) * 3 * 3, coeffs, 0, 0, 0));
                }
                else
                {
                    cl_int st;
                    for(int m = 0; m < 3; m++)
                        for(int n = 0; n < 3; n++)
                            float_coeffs[m][n] = coeffs[m][n];

                    coeffs_cm = clCreateBuffer( clCxt->impl->clContext, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(clCxt->impl->clCmdQueue, (cl_mem)coeffs_cm, 1, 0, sizeof(float) * 3 * 3, float_coeffs, 0, 0, 0));
                }
                //TODO: improve this kernel
                size_t blkSizeX = 16, blkSizeY = 16;
                size_t glbSizeX;
                size_t cols;
                if(src.type() == CV_8UC1 && interpolation == 0)
                {
                    cols = (dst.cols + dst.offset % 4 + 3) / 4;
                    glbSizeX = cols % blkSizeX == 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;
                }
                else
                    /*
                    */
                {
                    cols = dst.cols;
                    glbSizeX = dst.cols % blkSizeX == 0 ? dst.cols : (dst.cols / blkSizeX + 1) * blkSizeX;
                }
                size_t glbSizeY = dst.rows % blkSizeY == 0 ? dst.rows : (dst.rows / blkSizeY + 1) * blkSizeY;
                size_t globalThreads[3] = {glbSizeX, glbSizeY, 1};
                size_t localThreads[3] = {blkSizeX, blkSizeY, 1};

                vector< pair<size_t, const void *> > args;

                args.push_back(make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back(make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back(make_pair(sizeof(cl_int), (void *)&srcStep));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dstStep));
                args.push_back(make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back(make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back(make_pair(sizeof(cl_mem), (void *)&coeffs_cm));
                args.push_back(make_pair(sizeof(cl_int), (void *)&cols));

                openCLExecuteKernel(clCxt, &imgproc_warpPerspective, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
                openCLSafeCall(clReleaseMemObject(coeffs_cm));
            }
        }

        void warpAffine(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags)
        {
            int interpolation = flags & INTER_MAX;

            CV_Assert((src.depth() == CV_8U  || src.depth() == CV_32F) && src.oclchannels() != 2 && src.oclchannels() != 3);
            CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

            dst.create(dsize, src.type());

            CV_Assert(M.rows == 2 && M.cols == 3);

            int warpInd = (flags & WARP_INVERSE_MAP) >> 4;
            F coeffs[2][3];

            double coeffsM[2*3];
            Mat coeffsMat(2, 3, CV_64F, (void *)coeffsM);
            M.convertTo(coeffsMat, coeffsMat.type());
            if(!warpInd)
            {
                convert_coeffs(coeffsM);
            }

            for(int i = 0; i < 2; ++i)
                for(int j = 0; j < 3; ++j)
                    coeffs[i][j] = coeffsM[i*3+j];

            warpAffine_gpu(src, dst, coeffs, interpolation);
        }

        void warpPerspective(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags)
        {
            int interpolation = flags & INTER_MAX;

            CV_Assert((src.depth() == CV_8U  || src.depth() == CV_32F) && src.oclchannels() != 2 && src.oclchannels() != 3);
            CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

            dst.create(dsize, src.type());


            CV_Assert(M.rows == 3 && M.cols == 3);

            int warpInd = (flags & WARP_INVERSE_MAP) >> 4;
            double coeffs[3][3];

            double coeffsM[3*3];
            Mat coeffsMat(3, 3, CV_64F, (void *)coeffsM);
            M.convertTo(coeffsMat, coeffsMat.type());
            if(!warpInd)
            {
                invert(coeffsM);
            }

            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    coeffs[i][j] = coeffsM[i*3+j];

            warpPerspective_gpu(src, dst, coeffs, interpolation);
        }


        ////////////////////////////////////////////////////////////////////////
        // integral

        void integral(const oclMat &src, oclMat &sum, oclMat &sqsum)
        {
            CV_Assert(src.type() == CV_8UC1);
            if(src.clCxt->impl->double_support == 0 && src.depth() == CV_64F)
            {
                CV_Error(CV_GpuNotSupported, "select device don't support double");
            }
            int vlen = 4;
            int offset = src.offset / vlen;
            int pre_invalid = src.offset % vlen;
            int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;

            oclMat t_sum , t_sqsum;
            t_sum.create(src.cols, src.rows, CV_32SC1);
            t_sqsum.create(src.cols, src.rows, CV_32FC1);

            int w = src.cols + 1, h = src.rows + 1;
            sum.create(h, w, CV_32SC1);
            sqsum.create(h, w, CV_32FC1);
            int sum_offset = sum.offset / vlen, sqsum_offset = sqsum.offset / vlen;

            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sqsum.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&pre_invalid ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.step));
            size_t gt[3] = {((vcols + 1) / 2) * 256, 1, 1}, lt[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_cols", gt, lt, args, -1, -1);
            args.clear();
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sqsum.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&sum.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&sqsum.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sum.step));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sqsum.step));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sum_offset));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sqsum_offset));
            size_t gt2[3] = {t_sum.cols  * 32, 1, 1}, lt2[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_rows", gt2, lt2, args, -1, -1);
            //cout << "tested" << endl;
        }
        void integral(const oclMat &src, oclMat &sum)
        {
            CV_Assert(src.type() == CV_8UC1);
            int vlen = 4;
            int offset = src.offset / vlen;
            int pre_invalid = src.offset % vlen;
            int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;

            oclMat t_sum;
            t_sum.create(src.cols, src.rows, CV_32SC1);

            int w = src.cols + 1, h = src.rows + 1;
            sum.create(h, w, CV_32SC1);
            int sum_offset = sum.offset / vlen;

            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&pre_invalid ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.step));
            size_t gt[3] = {((vcols + 1) / 2) * 256, 1, 1}, lt[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral_sum, "integral_sum_cols", gt, lt, args, -1, -1);
            args.clear();
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&sum.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&t_sum.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sum.step));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sum_offset));
            size_t gt2[3] = {t_sum.cols  * 32, 1, 1}, lt2[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral_sum, "integral_sum_rows", gt2, lt2, args, -1, -1);
            //cout << "tested" << endl;
        }

        /////////////////////// corner //////////////////////////////
        static void extractCovData(const oclMat &src, oclMat &Dx, oclMat &Dy,
                            int blockSize, int ksize, int borderType)
        {
            CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
            double scale = static_cast<double>(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;
            if (ksize < 0)
                scale *= 2.;

            if (src.depth() == CV_8U)
            {
                scale *= 255.;
                scale = 1. / scale;
            }
            else
            {
                scale = 1. / scale;
            }
            if (ksize > 0)
            {
                Sobel(src, Dx, CV_32F, 1, 0, ksize, scale, 0, borderType);
                Sobel(src, Dy, CV_32F, 0, 1, ksize, scale, 0, borderType);
            }
            else
            {
                Scharr(src, Dx, CV_32F, 1, 0, scale, 0, borderType);
                Scharr(src, Dy, CV_32F, 0, 1, scale, 0, borderType);
            }
            CV_Assert(Dx.offset == 0 && Dy.offset == 0);
        }

        static void corner_ocl(const char *src_str, string kernelName, int block_size, float k, oclMat &Dx, oclMat &Dy,
                        oclMat &dst, int border_type)
        {
            char borderType[30];
            switch (border_type)
            {
            case cv::BORDER_CONSTANT:
                sprintf(borderType, "BORDER_CONSTANT");
                break;
            case cv::BORDER_REFLECT101:
                sprintf(borderType, "BORDER_REFLECT101");
                break;
            case cv::BORDER_REFLECT:
                sprintf(borderType, "BORDER_REFLECT");
                break;
            case cv::BORDER_REPLICATE:
                sprintf(borderType, "BORDER_REPLICATE");
                break;
            default:
                cout << "BORDER type is not supported!" << endl;
            }
            char build_options[150];
            sprintf(build_options, "-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s",
                    block_size / 2, block_size / 2, block_size, block_size, borderType);

            size_t blockSizeX = 256, blockSizeY = 1;
            size_t gSize = blockSizeX - block_size / 2 * 2;
            size_t globalSizeX = (Dx.cols) % gSize == 0 ? Dx.cols / gSize * blockSizeX : (Dx.cols / gSize + 1) * blockSizeX;
            size_t rows_per_thread = 2;
            size_t globalSizeY = ((Dx.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ?
                                 ((Dx.rows + rows_per_thread - 1) / rows_per_thread) :
                                 (((Dx.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

            size_t gt[3] = { globalSizeX, globalSizeY, 1 };
            size_t lt[3]  = { blockSizeX, blockSizeY, 1 };
            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&Dx.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&Dy.data));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dx.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dx.wholerows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dx.wholecols ));
            args.push_back( make_pair(sizeof(cl_int), (void *)&Dx.step));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dy.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dy.wholerows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&Dy.wholecols ));
            args.push_back( make_pair(sizeof(cl_int), (void *)&Dy.step));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.offset));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.cols));
            args.push_back( make_pair(sizeof(cl_int), (void *)&dst.step));
            args.push_back( make_pair( sizeof(cl_float) , (void *)&k));
            openCLExecuteKernel(dst.clCxt, &src_str, kernelName, gt, lt, args, -1, -1, build_options);
        }

        void cornerHarris(const oclMat &src, oclMat &dst, int blockSize, int ksize,
                          double k, int borderType)
        {
            if(src.clCxt->impl->double_support == 0 && src.depth() == CV_64F)
            {
                CV_Error(CV_GpuNotSupported, "select device don't support double");
            }
            CV_Assert(src.cols >= blockSize / 2 && src.rows >= blockSize / 2);
            oclMat Dx, Dy;
            CV_Assert(borderType == cv::BORDER_CONSTANT || borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);
            extractCovData(src, Dx, Dy, blockSize, ksize, borderType);
            dst.create(src.size(), CV_32F);
            corner_ocl(imgproc_calcHarris, "calcHarris", blockSize, static_cast<float>(k), Dx, Dy, dst, borderType);
        }

        void cornerMinEigenVal(const oclMat &src, oclMat &dst, int blockSize, int ksize, int borderType)
        {
            if(src.clCxt->impl->double_support == 0 && src.depth() == CV_64F)
            {
                CV_Error(CV_GpuNotSupported, "select device don't support double");
            }
            CV_Assert(src.cols >= blockSize / 2 && src.rows >= blockSize / 2);
            oclMat Dx, Dy;
            CV_Assert(borderType == cv::BORDER_CONSTANT || borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);
            extractCovData(src, Dx, Dy, blockSize, ksize, borderType);
            dst.create(src.size(), CV_32F);
            corner_ocl(imgproc_calcMinEigenVal, "calcMinEigenVal", blockSize, 0, Dx, Dy, dst, borderType);
        }
        /////////////////////////////////// MeanShiftfiltering ///////////////////////////////////////////////
        static void meanShiftFiltering_gpu(const oclMat &src, oclMat dst, int sp, int sr, int maxIter, float eps)
        {
            CV_Assert( (src.cols == dst.cols) && (src.rows == dst.rows) );
            CV_Assert( !(dst.step & 0x3) );
            Context *clCxt = src.clCxt;

            //Arrange the NDRange
            int col = src.cols, row = src.rows;
            int ltx = 16, lty = 8;
            if(src.cols % ltx != 0)
                col = (col / ltx + 1) * ltx;
            if(src.rows % lty != 0)
                row = (row / lty + 1) * lty;

            size_t globalThreads[3] = {col, row, 1};
            size_t localThreads[3]  = {ltx, lty, 1};

            //set args
            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.step ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sp ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sr ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&maxIter ));
            args.push_back( make_pair( sizeof(cl_float) , (void *)&eps ));
            openCLExecuteKernel(clCxt, &meanShift, "meanshift_kernel", globalThreads, localThreads, args, -1, -1);
        }

        void meanShiftFiltering(const oclMat &src, oclMat &dst, int sp, int sr, TermCriteria criteria)
        {
            if( src.empty() )
                CV_Error( CV_StsBadArg, "The input image is empty" );

            if( src.depth() != CV_8U || src.oclchannels() != 4 )
                CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

            //            if(src.clCxt->impl->double_support == 0)
            //            {
            //                CV_Error( CV_GpuNotSupported, "Selected device doesn't support double, so a deviation exists.\nIf the accuracy is acceptable, the error can be ignored.\n");
            //            }

            dst.create( src.size(), CV_8UC4 );

            if( !(criteria.type & TermCriteria::MAX_ITER) )
                criteria.maxCount = 5;

            int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

            float eps;
            if( !(criteria.type & TermCriteria::EPS) )
                eps = 1.f;
            eps = (float)std::max(criteria.epsilon, 0.0);

            meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps);

        }

        static void meanShiftProc_gpu(const oclMat &src, oclMat dstr, oclMat dstsp, int sp, int sr, int maxIter, float eps)
        {
            //sanity checks
            CV_Assert( (src.cols == dstr.cols) && (src.rows == dstr.rows) &&
                       (src.rows == dstsp.rows) && (src.cols == dstsp.cols));
            CV_Assert( !(dstsp.step & 0x3) );
            Context *clCxt = src.clCxt;

            //Arrange the NDRange
            int col = src.cols, row = src.rows;
            int ltx = 16, lty = 8;
            if(src.cols % ltx != 0)
                col = (col / ltx + 1) * ltx;
            if(src.rows % lty != 0)
                row = (row / lty + 1) * lty;

            size_t globalThreads[3] = {col, row, 1};
            size_t localThreads[3]  = {ltx, lty, 1};

            //set args
            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&dstr.data ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&dstsp.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstr.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstsp.step ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstr.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstsp.offset ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstr.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstr.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sp ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&sr ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&maxIter ));
            args.push_back( make_pair( sizeof(cl_float) , (void *)&eps ));
            openCLExecuteKernel(clCxt, &meanShift, "meanshiftproc_kernel", globalThreads, localThreads, args, -1, -1);
        }

        void meanShiftProc(const oclMat &src, oclMat &dstr, oclMat &dstsp, int sp, int sr, TermCriteria criteria)
        {
            if( src.empty() )
                CV_Error( CV_StsBadArg, "The input image is empty" );

            if( src.depth() != CV_8U || src.oclchannels() != 4 )
                CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

            //            if(src.clCxt->impl->double_support == 0)
            //            {
            //                CV_Error( CV_GpuNotSupported, "Selected device doesn't support double, so a deviation exists.\nIf the accuracy is acceptable, the error can be ignored.\n");
            //            }

            dstr.create( src.size(), CV_8UC4 );
            dstsp.create( src.size(), CV_16SC2 );

            if( !(criteria.type & TermCriteria::MAX_ITER) )
                criteria.maxCount = 5;

            int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

            float eps;
            if( !(criteria.type & TermCriteria::EPS) )
                eps = 1.f;
            eps = (float)std::max(criteria.epsilon, 0.0);

            meanShiftProc_gpu(src, dstr, dstsp, sp, sr, maxIter, eps);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////hist///////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace histograms
        {
            const int PARTIAL_HISTOGRAM256_COUNT = 256;
            const int HISTOGRAM256_BIN_COUNT = 256;
        }
        ///////////////////////////////calcHist/////////////////////////////////////////////////////////////////
        static void calc_sub_hist(const oclMat &mat_src, const oclMat &mat_sub_hist)
        {
            using namespace histograms;

            Context  *clCxt = mat_src.clCxt;
            int depth = mat_src.depth();

            string kernelName = "calc_sub_hist";

            size_t localThreads[3]  = { HISTOGRAM256_BIN_COUNT, 1, 1 };
            size_t globalThreads[3] = { PARTIAL_HISTOGRAM256_COUNT *localThreads[0], 1, 1};

            int dataWidth = 16;
            int dataWidth_bits = 4;
            int mask = dataWidth - 1;

            int cols = mat_src.cols * mat_src.oclchannels();
            int src_offset = mat_src.offset;
            int hist_step = mat_sub_hist.step >> 2;
            int left_col = 0, right_col = 0;

            if(cols >= dataWidth * 2 - 1)
            {
                left_col = dataWidth - (src_offset & mask);
                left_col &= mask;
                src_offset += left_col;
                cols -= left_col;
                right_col = cols & mask;
                cols -= right_col;
            }
            else
            {
                left_col = cols;
                right_col = 0;
                cols = 0;
                globalThreads[0] = 0;
            }

            vector<pair<size_t , const void *> > args;
            if(globalThreads[0] != 0)
            {
                int tempcols = cols >> dataWidth_bits;
                int inc_x = globalThreads[0] % tempcols;
                int inc_y = globalThreads[0] / tempcols;
                src_offset >>= dataWidth_bits;
                int src_step = mat_src.step >> dataWidth_bits;
                int datacount = tempcols * mat_src.rows;
                args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_src.data));
                args.push_back( make_pair( sizeof(cl_int), (void *)&src_step));
                args.push_back( make_pair( sizeof(cl_int), (void *)&src_offset));
                args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_sub_hist.data));
                args.push_back( make_pair( sizeof(cl_int), (void *)&datacount));
                args.push_back( make_pair( sizeof(cl_int), (void *)&tempcols));
                args.push_back( make_pair( sizeof(cl_int), (void *)&inc_x));
                args.push_back( make_pair( sizeof(cl_int), (void *)&inc_y));
                args.push_back( make_pair( sizeof(cl_int), (void *)&hist_step));
                openCLExecuteKernel(clCxt, &imgproc_histogram, kernelName, globalThreads, localThreads, args, -1, depth);
            }
            if(left_col != 0 || right_col != 0)
            {
                kernelName = "calc_sub_hist_border";
                src_offset = mat_src.offset;
                localThreads[0] = 1;
                localThreads[1] = 256;
                globalThreads[0] = left_col + right_col;
                globalThreads[1] = (mat_src.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];

                args.clear();
                args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_src.data));
                args.push_back( make_pair( sizeof(cl_int), (void *)&mat_src.step));
                args.push_back( make_pair( sizeof(cl_int), (void *)&src_offset));
                args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_sub_hist.data));
                args.push_back( make_pair( sizeof(cl_int), (void *)&left_col));
                args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
                args.push_back( make_pair( sizeof(cl_int), (void *)&mat_src.rows));
                args.push_back( make_pair( sizeof(cl_int), (void *)&hist_step));
                openCLExecuteKernel(clCxt, &imgproc_histogram, kernelName, globalThreads, localThreads, args, -1, depth);
            }
        }
        static void merge_sub_hist(const oclMat &sub_hist, oclMat &mat_hist)
        {
            using namespace histograms;

            Context  *clCxt = sub_hist.clCxt;
            string kernelName = "merge_hist";

            size_t localThreads[3]  = { 256, 1, 1 };
            size_t globalThreads[3] = { HISTOGRAM256_BIN_COUNT *localThreads[0], 1, 1};
            int src_step = sub_hist.step >> 2;
            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&sub_hist.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_hist.data));
            args.push_back( make_pair( sizeof(cl_int), (void *)&src_step));
            openCLExecuteKernel(clCxt, &imgproc_histogram, kernelName, globalThreads, localThreads, args, -1, -1);
        }
        void calcHist(const oclMat &mat_src, oclMat &mat_hist)
        {
            using namespace histograms;
            CV_Assert(mat_src.type() == CV_8UC1);
            mat_hist.create(1, 256, CV_32SC1);

            oclMat buf(PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_BIN_COUNT, CV_32SC1);
            buf.setTo(0);

            calc_sub_hist(mat_src, buf);
            merge_sub_hist(buf, mat_hist);
        }
        ///////////////////////////////////equalizeHist/////////////////////////////////////////////////////
        void equalizeHist(const oclMat &mat_src, oclMat &mat_dst)
        {
            mat_dst.create(mat_src.rows, mat_src.cols, CV_8UC1);

            oclMat mat_hist(1, 256, CV_32SC1);
            //mat_hist.setTo(0);
            calcHist(mat_src, mat_hist);

            Context *clCxt = mat_src.clCxt;
            string kernelName = "calLUT";
            size_t localThreads[3] = { 256, 1, 1};
            size_t globalThreads[3] = { 256, 1, 1};
            oclMat lut(1, 256, CV_8UC1);
            vector<pair<size_t , const void *> > args;
            float scale = 255.f / (mat_src.rows * mat_src.cols);
            args.push_back( make_pair( sizeof(cl_mem), (void *)&lut.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&mat_hist.data));
            args.push_back( make_pair( sizeof(cl_float), (void *)&scale));
            openCLExecuteKernel(clCxt, &imgproc_histogram, kernelName, globalThreads, localThreads, args, -1, -1);
            LUT(mat_src, lut, mat_dst);
        }
        //////////////////////////////////bilateralFilter////////////////////////////////////////////////////
        static void
        oclbilateralFilter_8u( const oclMat &src, oclMat &dst, int d,
                               double sigma_color, double sigma_space,
                               int borderType )
        {
            int cn = src.channels();
            int i, j, maxk, radius;

            CV_Assert( (src.channels() == 1 || src.channels() == 3) &&
                       src.type() == dst.type() && src.size() == dst.size() &&
                       src.data != dst.data );

            if( sigma_color <= 0 )
                sigma_color = 1;
            if( sigma_space <= 0 )
                sigma_space = 1;

            double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
            double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

            if( d <= 0 )
                radius = cvRound(sigma_space * 1.5);
            else
                radius = d / 2;
            radius = MAX(radius, 1);
            d = radius * 2 + 1;

            oclMat temp;
            copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

            vector<float> _color_weight(cn * 256);
            vector<float> _space_weight(d * d);
            vector<int> _space_ofs(d * d);
            float *color_weight = &_color_weight[0];
            float *space_weight = &_space_weight[0];
            int *space_ofs = &_space_ofs[0];
            int dst_step_in_pixel = dst.step / dst.elemSize();
            int dst_offset_in_pixel = dst.offset / dst.elemSize();
            int temp_step_in_pixel = temp.step / temp.elemSize();
            // initialize color-related bilateral filter coefficients
            for( i = 0; i < 256 * cn; i++ )
                color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

            // initialize space-related bilateral filter coefficients
            for( i = -radius, maxk = 0; i <= radius; i++ )
                for( j = -radius; j <= radius; j++ )
                {
                    double r = std::sqrt((double)i * i + (double)j * j);
                    if( r > radius )
                        continue;
                    space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
                    space_ofs[maxk++] = (int)(i * temp_step_in_pixel + j);
                }
            oclMat oclcolor_weight(1, cn * 256, CV_32FC1, color_weight);
            oclMat oclspace_weight(1, d * d, CV_32FC1, space_weight);
            oclMat oclspace_ofs(1, d * d, CV_32SC1, space_ofs);

            string kernelName = "bilateral";
            size_t localThreads[3]  = { 16, 16, 1 };
            size_t globalThreads[3] = { (dst.cols + localThreads[0] - 1) / localThreads[0] *localThreads[0],
                                        (dst.rows + localThreads[1] - 1) / localThreads[1] *localThreads[1],
                                        1
                                      };
            if((dst.type() == CV_8UC1) && ((dst.offset & 3) == 0) && ((dst.cols & 3) == 0))
            {
                kernelName = "bilateral2";
                globalThreads[0] = (dst.cols / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            }
            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&temp.data ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst.rows ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst.cols ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&maxk ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&radius ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst_step_in_pixel ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&dst_offset_in_pixel ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&temp_step_in_pixel ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&temp.rows ));
            args.push_back( make_pair( sizeof(cl_int), (void *)&temp.cols ));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&oclcolor_weight.data ));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&oclspace_weight.data ));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&oclspace_ofs.data ));
            openCLExecuteKernel(src.clCxt, &imgproc_bilateral, kernelName, globalThreads, localThreads, args, dst.oclchannels(), dst.depth());
        }
        void bilateralFilter(const oclMat &src, oclMat &dst, int radius, double sigmaclr, double sigmaspc, int borderType)
        {

            dst.create( src.size(), src.type() );
            if( src.depth() == CV_8U )
                oclbilateralFilter_8u( src, dst, radius, sigmaclr, sigmaspc, borderType );
            else
                CV_Error( CV_StsUnsupportedFormat,
                          "Bilateral filtering is only implemented for 8uimages" );
        }

    }
}
//////////////////////////////////convolve////////////////////////////////////////////////////
inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}
static void convolve_run(const oclMat &src, const oclMat &temp1, oclMat &dst, string kernelName, const char **kernelString)
{
    CV_Assert(src.depth() == CV_32FC1);
    CV_Assert(temp1.depth() == CV_32F);
    CV_Assert(temp1.cols <= 17 && temp1.rows <= 17);

    dst.create(src.size(), src.type());

    CV_Assert(src.cols == dst.cols && src.rows == dst.rows);
    CV_Assert(src.type() == dst.type());

    Context  *clCxt = src.clCxt;
    int channels = dst.oclchannels();
    int depth = dst.depth();

    size_t vector_length = 1;
    int offset_cols = ((dst.offset % dst.step) / dst.elemSize1()) & (vector_length - 1);
    int cols = divUp(dst.cols * channels + offset_cols, vector_length);
    int rows = dst.rows;

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { divUp(cols, localThreads[0]) *localThreads[0],
                                divUp(rows, localThreads[1]) *localThreads[1],
                                1
                              };

    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&temp1.data ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&temp1.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&temp1.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&temp1.cols ));

    openCLExecuteKernel(clCxt, kernelString, kernelName, globalThreads, localThreads, args, -1, depth);
}
void cv::ocl::convolve(const oclMat &x, const oclMat &t, oclMat &y)
{
    CV_Assert(x.depth() == CV_32F);
    CV_Assert(t.depth() == CV_32F);
    CV_Assert(x.type() == y.type() && x.size() == y.size());
    y.create(x.size(), x.type());
    string kernelName = "convolve";

    convolve_run(x, t, y, kernelName, &imgproc_convolve);
}
#endif /* !defined (HAVE_OPENCL) */
