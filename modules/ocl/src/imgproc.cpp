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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
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
//    Wenju He, wenju@multicorewareinc.com
//    Peng Xiao, pengxiao@outlook.com
//    Sen Liu, swjtuls1987@126.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
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

        static std::vector<uchar> scalarToVector(const cv::Scalar & sc, int depth, int ocn, int cn)
        {
            CV_Assert(ocn == cn || (ocn == 4 && cn == 3));

            static const int sizeMap[] = { sizeof(uchar), sizeof(char), sizeof(ushort),
                                       sizeof(short), sizeof(int), sizeof(float), sizeof(double) };

            int elemSize1 = sizeMap[depth];
            int bufSize = elemSize1 * ocn;
            std::vector<uchar> _buf(bufSize);
            uchar * buf = &_buf[0];
            scalarToRawData(sc, buf, CV_MAKE_TYPE(depth, cn));
            memset(buf + elemSize1 * cn, 0, (ocn - cn) * elemSize1);

            return _buf;
        }

        static void threshold_runner(const oclMat &src, oclMat &dst, double thresh, double maxVal, int thresholdType)
        {
            bool ival = src.depth() < CV_32F;
            int cn = src.channels(), vecSize = 4, depth = src.depth();
            std::vector<uchar> thresholdValue = scalarToVector(cv::Scalar::all(ival ? cvFloor(thresh) : thresh), dst.depth(),
                                                               dst.oclchannels(), dst.channels());
            std::vector<uchar> maxValue = scalarToVector(cv::Scalar::all(maxVal), dst.depth(), dst.oclchannels(), dst.channels());

            const char * const thresholdMap[] = { "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC",
                                                  "THRESH_TOZERO", "THRESH_TOZERO_INV" };
            const char * const channelMap[] = { "", "", "2", "4", "4" };
            const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
            std::string buildOptions = format("-D T=%s%s -D %s", typeMap[depth], channelMap[cn], thresholdMap[thresholdType]);

            int elemSize = src.elemSize();
            int src_step = src.step / elemSize, src_offset = src.offset / elemSize;
            int dst_step = dst.step / elemSize, dst_offset = dst.offset / elemSize;

            std::vector< std::pair<size_t, const void *> > args;
            args.push_back( std::make_pair(sizeof(cl_mem), (void *)&src.data));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src_offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src_step));
            args.push_back( std::make_pair(sizeof(cl_mem), (void *)&dst.data));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst_offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst_step));
            args.push_back( std::make_pair(thresholdValue.size(), (void *)&thresholdValue[0]));
            args.push_back( std::make_pair(maxValue.size(), (void *)&maxValue[0]));

            int max_index = dst.cols, cols = dst.cols;
            if (cn == 1 && vecSize > 1)
            {
                CV_Assert(((vecSize - 1) & vecSize) == 0 && vecSize <= 16);
                cols = divUp(cols, vecSize);
                buildOptions += format(" -D VECTORIZED -D VT=%s%d -D VLOADN=vload%d -D VECSIZE=%d -D VSTOREN=vstore%d",
                                       typeMap[depth], vecSize, vecSize, vecSize, vecSize);

                int vecSizeBytes = vecSize * dst.elemSize1();
                if ((dst.offset % dst.step) % vecSizeBytes == 0 && dst.step % vecSizeBytes == 0)
                    buildOptions += " -D DST_ALIGNED";
                if ((src.offset % src.step) % vecSizeBytes == 0 && src.step % vecSizeBytes == 0)
                    buildOptions += " -D SRC_ALIGNED";

                args.push_back( std::make_pair(sizeof(cl_int), (void *)&max_index));
            }

            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&cols));

            size_t localThreads[3] = { 16, 16, 1 };
            size_t globalThreads[3] = { cols, dst.rows, 1 };

            openCLExecuteKernel(src.clCxt, &imgproc_threshold, "threshold", globalThreads, localThreads, args,
                                -1, -1, buildOptions.c_str());
        }

        double threshold(const oclMat &src, oclMat &dst, double thresh, double maxVal, int thresholdType)
        {
            CV_Assert(thresholdType == THRESH_BINARY || thresholdType == THRESH_BINARY_INV || thresholdType == THRESH_TRUNC
                      || thresholdType == THRESH_TOZERO || thresholdType == THRESH_TOZERO_INV);

            dst.create(src.size(), src.type());
            threshold_runner(src, dst, thresh, maxVal, thresholdType);

            return thresh;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////   remap   //////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        void remap( const oclMat &src, oclMat &dst, oclMat &map1, oclMat &map2, int interpolation, int borderType, const Scalar &borderValue )
        {
            Context *clCxt = src.clCxt;
            bool supportsDouble = clCxt->supportsFeature(FEATURE_CL_DOUBLE);
            if (!supportsDouble && src.depth() == CV_64F)
            {
                CV_Error(CV_OpenCLDoubleNotSupported, "Selected device does not support double");
                return;
            }

            if (map1.empty())
                map1.swap(map2);

            CV_Assert(interpolation == INTER_LINEAR || interpolation == INTER_NEAREST
                      /*|| interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4*/);
            CV_Assert((map1.type() == CV_16SC2 && (map2.empty() || (interpolation == INTER_NEAREST &&
                                                                    (map2.type() == CV_16UC1 || map2.type() == CV_16SC1)) )) ||
                      (map1.type() == CV_32FC2 && !map2.data) ||
                      (map1.type() == CV_32FC1 && map2.type() == CV_32FC1));
            CV_Assert(!map2.data || map2.size() == map1.size());
            CV_Assert(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE || borderType == BORDER_WRAP
                      || borderType == BORDER_REFLECT_101 || borderType == BORDER_REFLECT);

            dst.create(map1.size(), src.type());

            const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
            const char * const channelMap[] = { "", "", "2", "4", "4" };
            const char * const interMap[] = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LINEAR", "INTER_LANCZOS" };
            const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                                   "BORDER_REFLECT_101", "BORDER_TRANSPARENT" };

            String kernelName = "remap";
            if (map1.type() == CV_32FC2 && map2.empty())
                kernelName = kernelName + "_32FC2";
            else if (map1.type() == CV_16SC2)
            {
                kernelName = kernelName + "_16SC2";
                if (!map2.empty())
                    kernelName = kernelName + "_16UC1";
            }
            else if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
                kernelName = kernelName + "_2_32FC1";
            else
                CV_Error(Error::StsBadArg, "Unsupported map types");

            int ocn = dst.oclchannels();
            size_t localThreads[3] = { 16, 16, 1};
            size_t globalThreads[3] = { dst.cols, dst.rows, 1};

            Mat scalar(1, 1, CV_MAKE_TYPE(dst.depth(), ocn), borderValue);
            String buildOptions = format("-D %s -D %s -D T=%s%s", interMap[interpolation],
                                         borderMap[borderType], typeMap[src.depth()], channelMap[ocn]);

            if (interpolation != INTER_NEAREST)
            {
                int wdepth = std::max(CV_32F, dst.depth());
                buildOptions = buildOptions
                              + format(" -D WT=%s%s -D convertToT=convert_%s%s%s -D convertToWT=convert_%s%s"
                                       " -D convertToWT2=convert_%s2 -D WT2=%s2",
                                       typeMap[wdepth], channelMap[ocn],
                                       typeMap[src.depth()], channelMap[ocn], src.depth() < CV_32F ? "_sat_rte" : "",
                                       typeMap[wdepth], channelMap[ocn],
                                       typeMap[wdepth], typeMap[wdepth]);
            }

            int src_step = src.step / src.elemSize(), src_offset = src.offset / src.elemSize();
            int map1_step = map1.step / map1.elemSize(), map1_offset = map1.offset / map1.elemSize();
            int map2_step = map2.step / map2.elemSize(), map2_offset = map2.offset / map2.elemSize();
            int dst_step = dst.step / dst.elemSize(), dst_offset = dst.offset / dst.elemSize();

            std::vector< std::pair<size_t, const void *> > args;
            args.push_back( std::make_pair(sizeof(cl_mem), (void *)&src.data));
            args.push_back( std::make_pair(sizeof(cl_mem), (void *)&dst.data));
            args.push_back( std::make_pair(sizeof(cl_mem), (void *)&map1.data));
            if (!map2.empty())
                args.push_back( std::make_pair(sizeof(cl_mem), (void *)&map2.data));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src_offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst_offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&map1_offset));
            if (!map2.empty())
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&map2_offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src_step));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst_step));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&map1_step));
            if (!map2.empty())
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&map2_step));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.cols));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.rows));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.cols));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( std::make_pair(scalar.elemSize(), (void *)scalar.data));

            openCLExecuteKernel(clCxt, &imgproc_remap, kernelName, globalThreads, localThreads, args, -1, -1, buildOptions.c_str());
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

            String kernelName;
            if (interpolation == INTER_LINEAR)
                kernelName = "resizeLN";
            else if (interpolation == INTER_NEAREST)
                kernelName = "resizeNN";

            //TODO: improve this kernel
            size_t blkSizeX = 16, blkSizeY = 16;
            size_t glbSizeX;
            if (src.type() == CV_8UC1)
            {
                size_t cols = (dst.cols + dst.offset % 4 + 3) / 4;
                glbSizeX = cols % blkSizeX == 0 && cols != 0 ? cols : (cols / blkSizeX + 1) * blkSizeX;
            }
            else
                glbSizeX = dst.cols % blkSizeX == 0 && dst.cols != 0 ? dst.cols : (dst.cols / blkSizeX + 1) * blkSizeX;

            size_t glbSizeY = dst.rows % blkSizeY == 0 && dst.rows != 0 ? dst.rows : (dst.rows / blkSizeY + 1) * blkSizeY;
            size_t globalThreads[3] = {glbSizeX, glbSizeY, 1};
            size_t localThreads[3] = {blkSizeX, blkSizeY, 1};

            std::vector< std::pair<size_t, const void *> > args;
            if (interpolation == INTER_NEAREST)
            {
                args.push_back( std::make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( std::make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dstoffset_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&srcoffset_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dstStep_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&srcStep_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.rows));
                if (src.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
                {
                    args.push_back( std::make_pair(sizeof(cl_double), (void *)&ifx_d));
                    args.push_back( std::make_pair(sizeof(cl_double), (void *)&ify_d));
                }
                else
                {
                    args.push_back( std::make_pair(sizeof(cl_float), (void *)&ifx));
                    args.push_back( std::make_pair(sizeof(cl_float), (void *)&ify));
                }
            }
            else
            {
                args.push_back( std::make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back( std::make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dstoffset_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&srcoffset_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dstStep_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&srcStep_in_pixel));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back( std::make_pair(sizeof(cl_float), (void *)&ifx));
                args.push_back( std::make_pair(sizeof(cl_float), (void *)&ify));
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

            if (!(dsize == Size()) && (fx > 0 && fy > 0))
                if (dsize.width != (int)(src.cols * fx) || dsize.height != (int)(src.rows * fy))
                    CV_Error(Error::StsUnmatchedSizes, "invalid dsize and fx, fy!");

            if ( dsize == Size() )
                dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
            else
            {
                fx = (double)dsize.width / src.cols;
                fy = (double)dsize.height / src.rows;
            }

            dst.create(dsize, src.type());

            if ( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR )
            {
                resize_gpu( src, dst, fx, fy, interpolation);
                return;
            }

            CV_Error(Error::StsUnsupportedFormat, "Non-supported interpolation method");
        }

        ////////////////////////////////////////////////////////////////////////
        // medianFilter

        void medianFilter(const oclMat &src, oclMat &dst, int m)
        {
            CV_Assert( m % 2 == 1 && m > 1 );
            CV_Assert( (src.depth() == CV_8U || src.depth() == CV_32F) && (src.channels() == 1 || src.channels() == 4));
            dst.create(src.size(), src.type());

            int srcStep = src.step / src.elemSize(), dstStep = dst.step / dst.elemSize();
            int srcOffset = src.offset /  src.elemSize(), dstOffset = dst.offset / dst.elemSize();

            Context *clCxt = src.clCxt;

            std::vector< std::pair<size_t, const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcOffset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstOffset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcStep));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstStep));

            size_t globalThreads[3] = {(src.cols + 18) / 16 * 16, (src.rows + 15) / 16 * 16, 1};
            size_t localThreads[3] = {16, 16, 1};

            if (m == 3)
            {
                String kernelName = "medianFilter3";
                openCLExecuteKernel(clCxt, &imgproc_median, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
            }
            else if (m == 5)
            {
                String kernelName = "medianFilter5";
                openCLExecuteKernel(clCxt, &imgproc_median, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
            }
            else
                CV_Error(Error::StsBadArg, "Non-supported filter length");
        }

        ////////////////////////////////////////////////////////////////////////
        // copyMakeBorder

        void copyMakeBorder(const oclMat &src, oclMat &dst, int top, int bottom, int left, int right, int bordertype, const Scalar &scalar)
        {
            if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
            {
                CV_Error(Error::OpenCLDoubleNotSupported, "Selected device does not support double");
                return;
            }

            oclMat _src = src;

            CV_Assert(top >= 0 && bottom >= 0 && left >= 0 && right >= 0);

            if( (_src.wholecols != _src.cols || _src.wholerows != _src.rows) && (bordertype & BORDER_ISOLATED) == 0 )
            {
                Size wholeSize;
                Point ofs;
                _src.locateROI(wholeSize, ofs);
                int dtop = std::min(ofs.y, top);
                int dbottom = std::min(wholeSize.height - _src.rows - ofs.y, bottom);
                int dleft = std::min(ofs.x, left);
                int dright = std::min(wholeSize.width - _src.cols - ofs.x, right);
                _src.adjustROI(dtop, dbottom, dleft, dright);
                top -= dtop;
                left -= dleft;
                bottom -= dbottom;
                right -= dright;
            }
            bordertype &= ~cv::BORDER_ISOLATED;

            dst.create(_src.rows + top + bottom, _src.cols + left + right, _src.type());
            int srcStep = _src.step / _src.elemSize(),  dstStep = dst.step / dst.elemSize();
            int srcOffset = _src.offset / _src.elemSize(), dstOffset = dst.offset / dst.elemSize();
            int depth = _src.depth(), ochannels = _src.oclchannels();

            int __bordertype[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101 };
            const char *borderstr[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" };

            int bordertype_index = -1;
            for (int i = 0, end = sizeof(__bordertype) / sizeof(int); i < end; i++)
                if (__bordertype[i] == bordertype)
                {
                    bordertype_index = i;
                    break;
                }
            if (bordertype_index < 0)
                CV_Error(Error::StsBadArg, "Unsupported border type");

            size_t localThreads[3] = { 16, 16, 1 };
            size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

            std::vector< std::pair<size_t, const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&_src.data));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.cols));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&_src.cols));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&_src.rows));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcStep));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcOffset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstStep));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstOffset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&top));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&left));

            const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
            const char * const channelMap[] = { "", "", "2", "4", "4" };
            std::string buildOptions = format("-D GENTYPE=%s%s -D %s",
                                              typeMap[depth], channelMap[ochannels],
                                              borderstr[bordertype_index]);

            int cn = src.channels(), ocn = src.oclchannels();
            int bufSize = src.elemSize1() * ocn;
            AutoBuffer<uchar> _buf(bufSize);
            uchar * buf = (uchar *)_buf;
            scalarToRawData(scalar, buf, dst.type());
            memset(buf + src.elemSize1() * cn, 0, (ocn - cn) * src.elemSize1());

            args.push_back( std::make_pair( bufSize , (void *)buf ));

            openCLExecuteKernel(src.clCxt, &imgproc_copymakeboder, "copymakeborder", globalThreads,
                                localThreads, args, -1, -1, buildOptions.c_str());
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
                if ( d != 0)
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
                String s[3] = {"NN", "Linear", "Cubic"};
                String kernelName = "warpAffine" + s[interpolation];

                if (src.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
                {
                    cl_int st;
                    coeffs_cm = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE, sizeof(F) * 2 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(), (cl_mem)coeffs_cm, 1, 0,
                                                        sizeof(F) * 2 * 3, coeffs, 0, 0, 0));
                }
                else
                {
                    cl_int st;
                    for(int m = 0; m < 2; m++)
                        for(int n = 0; n < 3; n++)
                            float_coeffs[m][n] = coeffs[m][n];

                    coeffs_cm = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE, sizeof(float) * 2 * 3, NULL, &st );
                    openCLSafeCall(clEnqueueWriteBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(), (cl_mem)coeffs_cm,
                                                        1, 0, sizeof(float) * 2 * 3, float_coeffs, 0, 0, 0));

                }
                //TODO: improve this kernel
                size_t blkSizeX = 16, blkSizeY = 16;
                size_t glbSizeX;
                size_t cols;

                if (src.type() == CV_8UC1 && interpolation != 2)
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

                std::vector< std::pair<size_t, const void *> > args;

                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcStep));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstStep));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&coeffs_cm));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&cols));

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
                String s[3] = {"NN", "Linear", "Cubic"};
                String kernelName = "warpPerspective" + s[interpolation];

                if (src.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
                {
                    cl_int st;
                    coeffs_cm = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE, sizeof(double) * 3 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(), (cl_mem)coeffs_cm, 1, 0,
                                                        sizeof(double) * 3 * 3, coeffs, 0, 0, 0));
                }
                else
                {
                    cl_int st;
                    for(int m = 0; m < 3; m++)
                        for(int n = 0; n < 3; n++)
                            float_coeffs[m][n] = coeffs[m][n];

                    coeffs_cm = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE, sizeof(float) * 3 * 3, NULL, &st );
                    openCLVerifyCall(st);
                    openCLSafeCall(clEnqueueWriteBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(), (cl_mem)coeffs_cm, 1, 0,
                                                        sizeof(float) * 3 * 3, float_coeffs, 0, 0, 0));
                }

                //TODO: improve this kernel
                size_t blkSizeX = 16, blkSizeY = 16;
                size_t glbSizeX;
                size_t cols;
                if (src.type() == CV_8UC1 && interpolation == 0)
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

                std::vector< std::pair<size_t, const void *> > args;

                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcStep));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstStep));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
                args.push_back(std::make_pair(sizeof(cl_mem), (void *)&coeffs_cm));
                args.push_back(std::make_pair(sizeof(cl_int), (void *)&cols));

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
            if (!warpInd)
                convert_coeffs(coeffsM);

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
            if (!warpInd)
                invert(coeffsM);

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
            if (!src.clCxt->supportsFeature(ocl::FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
            {
                CV_Error(Error::OpenCLDoubleNotSupported, "Select device doesn't support double");
                return;
            }

            int vlen = 4;
            int offset = src.offset / vlen;
            int pre_invalid = src.offset % vlen;
            int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;

            oclMat t_sum , t_sqsum;
            int w = src.cols + 1, h = src.rows + 1;
            int depth = src.depth() == CV_8U ? CV_32S : CV_64F;
            int type = CV_MAKE_TYPE(depth, 1);

            t_sum.create(src.cols, src.rows, type);
            sum.create(h, w, type);

            t_sqsum.create(src.cols, src.rows, CV_32FC1);
            sqsum.create(h, w, CV_32FC1);

            int sum_offset = sum.offset / vlen;
            int sqsum_offset = sqsum.offset / vlen;

            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sqsum.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&pre_invalid ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.step));
            size_t gt[3] = {((vcols + 1) / 2) * 256, 1, 1}, lt[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_cols", gt, lt, args, -1, depth);

            args.clear();
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sqsum.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&sum.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&sqsum.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sum.step));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sqsum.step));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sum_offset));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sqsum_offset));
            size_t gt2[3] = {t_sum.cols  * 32, 1, 1}, lt2[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_rows", gt2, lt2, args, -1, depth);
        }

        void integral(const oclMat &src, oclMat &sum)
        {
            CV_Assert(src.type() == CV_8UC1);
            int vlen = 4;
            int offset = src.offset / vlen;
            int pre_invalid = src.offset % vlen;
            int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;

            oclMat t_sum;
            int w = src.cols + 1, h = src.rows + 1;
            int depth = src.depth() == CV_8U ? CV_32S : CV_32F;
            int type = CV_MAKE_TYPE(depth, 1);

            t_sum.create(src.cols, src.rows, type);
            sum.create(h, w, type);

            int sum_offset = sum.offset / vlen;
            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&pre_invalid ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.step));
            size_t gt[3] = {((vcols + 1) / 2) * 256, 1, 1}, lt[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral_sum, "integral_sum_cols", gt, lt, args, -1, depth);

            args.clear();
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&t_sum.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&sum.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t_sum.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sum.step));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sum_offset));
            size_t gt2[3] = {t_sum.cols  * 32, 1, 1}, lt2[3] = {256, 1, 1};
            openCLExecuteKernel(src.clCxt, &imgproc_integral_sum, "integral_sum_rows", gt2, lt2, args, -1, depth);
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
                scale = 1. / scale;

            if (ksize > 0)
            {
                Context* clCxt = Context::getContext();
                if(clCxt->supportsFeature(FEATURE_CL_INTEL_DEVICE) && src.type() == CV_8UC1 &&
                    src.cols % 8 == 0 && src.rows % 8 == 0 &&
                    ksize==3 &&
                    (borderType ==cv::BORDER_REFLECT ||
                     borderType == cv::BORDER_REPLICATE ||
                     borderType ==cv::BORDER_REFLECT101 ||
                     borderType ==cv::BORDER_WRAP))
                {
                    Dx.create(src.size(), CV_32FC1);
                    Dy.create(src.size(), CV_32FC1);

                    const unsigned int block_x = 8;
                    const unsigned int block_y = 8;

                    unsigned int src_pitch = src.step;
                    unsigned int dst_pitch = Dx.cols;

                    float _scale = scale;

                    std::vector<std::pair<size_t , const void *> > args;
                    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
                    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&Dx.data ));
                    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&Dy.data ));
                    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
                    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
                    args.push_back( std::make_pair( sizeof(cl_uint) , (void *)&src_pitch ));
                    args.push_back( std::make_pair( sizeof(cl_uint) , (void *)&dst_pitch ));
                    args.push_back( std::make_pair( sizeof(cl_float) , (void *)&_scale ));
                    size_t gt2[3] = {src.cols, src.rows, 1}, lt2[3] = {block_x, block_y, 1};

                    String option = "-D BLK_X=8 -D BLK_Y=8";
                    switch(borderType)
                    {
                    case cv::BORDER_REPLICATE:
                        option = option + " -D BORDER_REPLICATE";
                        break;
                    case cv::BORDER_REFLECT:
                        option = option + " -D BORDER_REFLECT";
                        break;
                    case cv::BORDER_REFLECT101:
                        option = option + " -D BORDER_REFLECT101";
                        break;
                    case cv::BORDER_WRAP:
                        option = option + " -D BORDER_WRAP";
                        break;
                    }
                    openCLExecuteKernel(src.clCxt, &imgproc_sobel3, "sobel3", gt2, lt2, args, -1, -1, option.c_str() );
                }
                else
                {
                    Sobel(src, Dx, CV_32F, 1, 0, ksize, scale, 0, borderType);
                    Sobel(src, Dy, CV_32F, 0, 1, ksize, scale, 0, borderType);
                }
            }
            else
            {
                Scharr(src, Dx, CV_32F, 1, 0, scale, 0, borderType);
                Scharr(src, Dy, CV_32F, 0, 1, scale, 0, borderType);
            }
            CV_Assert(Dx.offset == 0 && Dy.offset == 0);
        }

        static void corner_ocl(const cv::ocl::ProgramEntry* source, String kernelName, int block_size, float k, oclMat &Dx, oclMat &Dy,
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
                CV_Error(Error::StsBadFlag, "BORDER type is not supported!");
            }

            std::string buildOptions = format("-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s",
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
            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&Dx.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&Dy.data));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dx.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dx.wholerows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dx.wholecols ));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&Dx.step));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dy.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dy.wholerows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&Dy.wholecols ));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&Dy.step));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.offset));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.rows));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.cols));
            args.push_back( std::make_pair(sizeof(cl_int), (void *)&dst.step));
            args.push_back( std::make_pair( sizeof(cl_float) , (void *)&k));

            openCLExecuteKernel(dst.clCxt, source, kernelName, gt, lt, args, -1, -1, buildOptions.c_str());
        }

        void cornerHarris(const oclMat &src, oclMat &dst, int blockSize, int ksize,
                          double k, int borderType)
        {
            oclMat dx, dy;
            cornerHarris_dxdy(src, dst, dx, dy, blockSize, ksize, k, borderType);
        }

        void cornerHarris_dxdy(const oclMat &src, oclMat &dst, oclMat &dx, oclMat &dy, int blockSize, int ksize,
                          double k, int borderType)
        {
            if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
            {
                CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
                return;
            }

            CV_Assert(borderType == cv::BORDER_CONSTANT || borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE
                      || borderType == cv::BORDER_REFLECT);

            extractCovData(src, dx, dy, blockSize, ksize, borderType);
            dst.create(src.size(), CV_32FC1);
            corner_ocl(&imgproc_calcHarris, "calcHarris", blockSize, static_cast<float>(k), dx, dy, dst, borderType);
        }

        void cornerMinEigenVal(const oclMat &src, oclMat &dst, int blockSize, int ksize, int borderType)
        {
            oclMat dx, dy;
            cornerMinEigenVal_dxdy(src, dst, dx, dy, blockSize, ksize, borderType);
        }

        void cornerMinEigenVal_dxdy(const oclMat &src, oclMat &dst, oclMat &dx, oclMat &dy, int blockSize, int ksize, int borderType)
        {
            if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
            {
                CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
                return;
            }

            CV_Assert(borderType == cv::BORDER_CONSTANT || borderType == cv::BORDER_REFLECT101 ||
                      borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);

            extractCovData(src, dx, dy, blockSize, ksize, borderType);
            dst.create(src.size(), CV_32F);

            corner_ocl(&imgproc_calcMinEigenVal, "calcMinEigenVal", blockSize, 0, dx, dy, dst, borderType);
        }

        /////////////////////////////////// MeanShiftfiltering ///////////////////////////////////////////////

        static void meanShiftFiltering_gpu(const oclMat &src, oclMat dst, int sp, int sr, int maxIter, float eps)
        {
            CV_Assert( (src.cols == dst.cols) && (src.rows == dst.rows) );
            CV_Assert( !(dst.step & 0x3) );

            //Arrange the NDRange
            int col = src.cols, row = src.rows;
            int ltx = 16, lty = 8;
            if (src.cols % ltx != 0)
                col = (col / ltx + 1) * ltx;
            if (src.rows % lty != 0)
                row = (row / lty + 1) * lty;

            size_t globalThreads[3] = {col, row, 1};
            size_t localThreads[3]  = {ltx, lty, 1};

            //set args
            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.step ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sp ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sr ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&maxIter ));
            args.push_back( std::make_pair( sizeof(cl_float) , (void *)&eps ));

            openCLExecuteKernel(src.clCxt, &meanShift, "meanshift_kernel", globalThreads, localThreads, args, -1, -1);
        }

        void meanShiftFiltering(const oclMat &src, oclMat &dst, int sp, int sr, TermCriteria criteria)
        {
            if (src.empty())
                CV_Error(Error::StsBadArg, "The input image is empty");

            if ( src.depth() != CV_8U || src.oclchannels() != 4 )
                CV_Error(Error::StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported");

            dst.create( src.size(), CV_8UC4 );

            if ( !(criteria.type & TermCriteria::MAX_ITER) )
                criteria.maxCount = 5;

            int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

            float eps;
            if ( !(criteria.type & TermCriteria::EPS) )
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

            //Arrange the NDRange
            int col = src.cols, row = src.rows;
            int ltx = 16, lty = 8;
            if (src.cols % ltx != 0)
                col = (col / ltx + 1) * ltx;
            if (src.rows % lty != 0)
                row = (row / lty + 1) * lty;

            size_t globalThreads[3] = {col, row, 1};
            size_t localThreads[3]  = {ltx, lty, 1};

            //set args
            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dstr.data ));
            args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dstsp.data ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstr.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstsp.step ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstr.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstsp.offset ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstr.cols ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstr.rows ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sp ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sr ));
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&maxIter ));
            args.push_back( std::make_pair( sizeof(cl_float) , (void *)&eps ));

            openCLExecuteKernel(src.clCxt, &meanShift, "meanshiftproc_kernel", globalThreads, localThreads, args, -1, -1);
        }

        void meanShiftProc(const oclMat &src, oclMat &dstr, oclMat &dstsp, int sp, int sr, TermCriteria criteria)
        {
            if (src.empty())
                CV_Error(Error::StsBadArg, "The input image is empty");

            if ( src.depth() != CV_8U || src.oclchannels() != 4 )
                CV_Error(Error::StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported");

//            if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
//            {
//                CV_Error(Error::OpenCLDoubleNotSupportedNotSupported, "Selected device doesn't support double, so a deviation exists.\nIf the accuracy is acceptable, the error can be ignored.\n");
//                return;
//            }

            dstr.create( src.size(), CV_8UC4 );
            dstsp.create( src.size(), CV_16SC2 );

            if ( !(criteria.type & TermCriteria::MAX_ITER) )
                criteria.maxCount = 5;

            int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

            float eps;
            if ( !(criteria.type & TermCriteria::EPS) )
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

            int depth = mat_src.depth();

            size_t localThreads[3]  = { HISTOGRAM256_BIN_COUNT, 1, 1 };
            size_t globalThreads[3] = { PARTIAL_HISTOGRAM256_COUNT *localThreads[0], 1, 1};

            int dataWidth = 16;
            int dataWidth_bits = 4;
            int mask = dataWidth - 1;

            int cols = mat_src.cols * mat_src.oclchannels();
            int src_offset = mat_src.offset;
            int hist_step = mat_sub_hist.step >> 2;
            int left_col = 0, right_col = 0;

            if (cols >= dataWidth * 2 - 1)
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

            std::vector<std::pair<size_t , const void *> > args;
            if (globalThreads[0] != 0)
            {
                int tempcols = cols >> dataWidth_bits;
                int inc_x = globalThreads[0] % tempcols;
                int inc_y = globalThreads[0] / tempcols;
                src_offset >>= dataWidth_bits;
                int src_step = mat_src.step >> dataWidth_bits;
                int datacount = tempcols * mat_src.rows;

                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_sub_hist.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&datacount));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&tempcols));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&inc_x));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&inc_y));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&hist_step));

                openCLExecuteKernel(mat_src.clCxt, &imgproc_histogram, "calc_sub_hist", globalThreads, localThreads, args, -1, depth);
            }

            if (left_col != 0 || right_col != 0)
            {
                src_offset = mat_src.offset;
                localThreads[0] = 1;
                localThreads[1] = 256;
                globalThreads[0] = left_col + right_col;
                globalThreads[1] = mat_src.rows;

                args.clear();
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src.step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_sub_hist.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&left_col));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src.rows));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&hist_step));

                openCLExecuteKernel(mat_src.clCxt, &imgproc_histogram, "calc_sub_hist_border", globalThreads, localThreads, args, -1, depth);
            }
        }

        static void merge_sub_hist(const oclMat &sub_hist, oclMat &mat_hist)
        {
            using namespace histograms;

            size_t localThreads[3]  = { 256, 1, 1 };
            size_t globalThreads[3] = { HISTOGRAM256_BIN_COUNT *localThreads[0], 1, 1};
            int src_step = sub_hist.step >> 2;

            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&sub_hist.data));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_hist.data));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step));

            openCLExecuteKernel(sub_hist.clCxt, &imgproc_histogram, "merge_hist", globalThreads, localThreads, args, -1, -1);
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

            calcHist(mat_src, mat_hist);

            size_t localThreads[3] = { 256, 1, 1};
            size_t globalThreads[3] = { 256, 1, 1};
            oclMat lut(1, 256, CV_8UC1);
            int total = mat_src.rows * mat_src.cols;

            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&lut.data));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_hist.data));
            args.push_back( std::make_pair( sizeof(int), (void *)&total));

            openCLExecuteKernel(mat_src.clCxt, &imgproc_histogram, "calLUT", globalThreads, localThreads, args, -1, -1);
            LUT(mat_src, lut, mat_dst);
        }

        ////////////////////////////////////////////////////////////////////////
        // CLAHE
        namespace clahe
        {
            static void calcLut(const oclMat &src, oclMat &dst,
                const int tilesX, const int tilesY, const cv::Size tileSize,
                const int clipLimit, const float lutScale)
            {
                cl_int2 tile_size;
                tile_size.s[0] = tileSize.width;
                tile_size.s[1] = tileSize.height;

                std::vector<std::pair<size_t , const void *> > args;
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.step ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&tile_size ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&tilesX ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&clipLimit ));
                args.push_back( std::make_pair( sizeof(cl_float), (void *)&lutScale ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.offset ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));

                String kernelName = "calcLut";
                size_t localThreads[3]  = { 32, 8, 1 };
                size_t globalThreads[3] = { tilesX * localThreads[0], tilesY * localThreads[1], 1 };
                bool is_cpu = isCpuDevice();
                if (is_cpu)
                    openCLExecuteKernel(Context::getContext(), &imgproc_clahe, kernelName, globalThreads, localThreads, args, -1, -1, (char*)"-D CPU");
                else
                {
                    cl_kernel kernel = openCLGetKernelFromSource(Context::getContext(), &imgproc_clahe, kernelName);
                    int wave_size = (int)queryWaveFrontSize(kernel);
                    openCLSafeCall(clReleaseKernel(kernel));

                    std::string opt = format("-D WAVE_SIZE=%d", wave_size);
                    openCLExecuteKernel(Context::getContext(), &imgproc_clahe, kernelName, globalThreads, localThreads, args, -1, -1, opt.c_str());
                }
            }

            static void transform(const oclMat &src, oclMat &dst, const oclMat &lut,
                const int tilesX, const int tilesY, const Size & tileSize)
            {
                cl_int2 tile_size;
                tile_size.s[0] = tileSize.width;
                tile_size.s[1] = tileSize.height;

                std::vector<std::pair<size_t , const void *> > args;
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&lut.data ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.step ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&lut.step ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows ));
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&tile_size ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&tilesX ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&tilesY ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.offset ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&lut.offset ));

                size_t localThreads[3]  = { 32, 8, 1 };
                size_t globalThreads[3] = { src.cols, src.rows, 1 };

                openCLExecuteKernel(Context::getContext(), &imgproc_clahe, "transform", globalThreads, localThreads, args, -1, -1);
            }
        }

        namespace
        {
            class CLAHE_Impl : public cv::CLAHE
            {
            public:
                CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

                cv::AlgorithmInfo* info() const;

                void apply(cv::InputArray src, cv::OutputArray dst);

                void setClipLimit(double clipLimit);
                double getClipLimit() const;

                void setTilesGridSize(cv::Size tileGridSize);
                cv::Size getTilesGridSize() const;

                void collectGarbage();

            private:
                double clipLimit_;
                int tilesX_;
                int tilesY_;

                oclMat srcExt_;
                oclMat lut_;
            };

            CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
                clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
            {
            }

            CV_INIT_ALGORITHM(CLAHE_Impl, "CLAHE_OCL",
                obj.info()->addParam(obj, "clipLimit", obj.clipLimit_);
                obj.info()->addParam(obj, "tilesX", obj.tilesX_);
                obj.info()->addParam(obj, "tilesY", obj.tilesY_))

            void CLAHE_Impl::apply(cv::InputArray src_raw, cv::OutputArray dst_raw)
            {
                oclMat& src = getOclMatRef(src_raw);
                oclMat& dst = getOclMatRef(dst_raw);
                CV_Assert( src.type() == CV_8UC1 );

                dst.create( src.size(), src.type() );

                const int histSize = 256;

                ensureSizeIsEnough(tilesX_ * tilesY_, histSize, CV_8UC1, lut_);

                cv::Size tileSize;
                oclMat srcForLut;

                if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0)
                {
                    tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
                    srcForLut = src;
                }
                else
                {
                    ocl::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0,
                                            tilesX_ - (src.cols % tilesX_), BORDER_REFLECT_101, Scalar::all(0));

                    tileSize = Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
                    srcForLut = srcExt_;
                }

                const int tileSizeTotal = tileSize.area();
                const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

                int clipLimit = 0;
                if (clipLimit_ > 0.0)
                {
                    clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
                    clipLimit = std::max(clipLimit, 1);
                }

                clahe::calcLut(srcForLut, lut_, tilesX_, tilesY_, tileSize, clipLimit, lutScale);
                clahe::transform(src, dst, lut_, tilesX_, tilesY_, tileSize);
            }

            void CLAHE_Impl::setClipLimit(double clipLimit)
            {
                clipLimit_ = clipLimit;
            }

            double CLAHE_Impl::getClipLimit() const
            {
                return clipLimit_;
            }

            void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
            {
                tilesX_ = tileGridSize.width;
                tilesY_ = tileGridSize.height;
            }

            cv::Size CLAHE_Impl::getTilesGridSize() const
            {
                return cv::Size(tilesX_, tilesY_);
            }

            void CLAHE_Impl::collectGarbage()
            {
                srcExt_.release();
                lut_.release();
            }
        }

        cv::Ptr<cv::CLAHE> createCLAHE(double clipLimit, cv::Size tileGridSize)
        {
            return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
        }

        //////////////////////////////////bilateralFilter////////////////////////////////////////////////////

        static void oclbilateralFilter_8u( const oclMat &src, oclMat &dst, int d,
                               double sigma_color, double sigma_space,
                               int borderType )
        {
            int cn = src.channels();
            int i, j, maxk, radius;

            CV_Assert( (src.channels() == 1 || src.channels() == 3) &&
                       src.type() == dst.type() && src.size() == dst.size() &&
                       src.data != dst.data );

            if ( sigma_color <= 0 )
                sigma_color = 1;
            if ( sigma_space <= 0 )
                sigma_space = 1;

            double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
            double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

            if ( d <= 0 )
                radius = cvRound(sigma_space * 1.5);
            else
                radius = d / 2;
            radius = MAX(radius, 1);
            d = radius * 2 + 1;

            oclMat temp;
            copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

            std::vector<float> _color_weight(cn * 256);
            std::vector<float> _space_weight(d * d);
            std::vector<int> _space_ofs(d * d);
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
                    if ( r > radius )
                        continue;
                    space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
                    space_ofs[maxk++] = (int)(i * temp_step_in_pixel + j);
                }

            oclMat oclcolor_weight(1, cn * 256, CV_32FC1, color_weight);
            oclMat oclspace_weight(1, d * d, CV_32FC1, space_weight);
            oclMat oclspace_ofs(1, d * d, CV_32SC1, space_ofs);

            String kernelName = "bilateral";
            size_t localThreads[3]  = { 16, 16, 1 };
            size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

            if ((dst.type() == CV_8UC1) && ((dst.offset & 3) == 0) && ((dst.cols & 3) == 0))
            {
                kernelName = "bilateral2";
                globalThreads[0] = dst.cols >> 2;
            }

            std::vector<std::pair<size_t , const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&temp.data ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.cols ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&maxk ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&radius ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step_in_pixel ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_offset_in_pixel ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp_step_in_pixel ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp.rows ));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp.cols ));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&oclcolor_weight.data ));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&oclspace_weight.data ));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&oclspace_ofs.data ));

            openCLExecuteKernel(src.clCxt, &imgproc_bilateral, kernelName, globalThreads, localThreads, args, dst.oclchannels(), dst.depth());
        }

        void bilateralFilter(const oclMat &src, oclMat &dst, int radius, double sigmaclr, double sigmaspc, int borderType)
        {
            dst.create( src.size(), src.type() );
            if ( src.depth() == CV_8U )
                oclbilateralFilter_8u( src, dst, radius, sigmaclr, sigmaspc, borderType );
            else
                CV_Error(Error::StsUnsupportedFormat, "Bilateral filtering is only implemented for CV_8U images");
        }

    }
}
//////////////////////////////////mulSpectrums////////////////////////////////////////////////////
void cv::ocl::mulSpectrums(const oclMat &a, const oclMat &b, oclMat &c, int /*flags*/, float scale, bool conjB)
{
    CV_Assert(a.type() == CV_32FC2);
    CV_Assert(b.type() == CV_32FC2);

    c.create(a.size(), CV_32FC2);

    size_t lt[3]  = { 16, 16, 1 };
    size_t gt[3]  = { a.cols, a.rows, 1 };

    String kernelName = conjB ? "mulAndScaleSpectrumsKernel_CONJ":"mulAndScaleSpectrumsKernel";

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&a.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&b.data ));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&scale));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&c.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&a.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&a.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&a.step ));

    Context *clCxt = Context::getContext();
    openCLExecuteKernel(clCxt, &imgproc_mulAndScaleSpectrums, kernelName, gt, lt, args, -1, -1);
}
//////////////////////////////////convolve////////////////////////////////////////////////////
// ported from CUDA module
void cv::ocl::ConvolveBuf::create(Size image_size, Size templ_size)
{
    result_size = Size(image_size.width - templ_size.width + 1,
                       image_size.height - templ_size.height + 1);

    block_size = user_block_size;
    if (user_block_size.width == 0 || user_block_size.height == 0)
        block_size = estimateBlockSize(result_size, templ_size);

    dft_size.width  = 1 << int(ceil(std::log(block_size.width + templ_size.width - 1.) / std::log(2.)));
    dft_size.height = 1 << int(ceil(std::log(block_size.height + templ_size.height - 1.) / std::log(2.)));

    // CUFFT has hard-coded kernels for power-of-2 sizes (up to 8192),
    // see CUDA Toolkit 4.1 CUFFT Library Programming Guide
    //if (dft_size.width > 8192)
    dft_size.width = getOptimalDFTSize(block_size.width + templ_size.width - 1.);
    //if (dft_size.height > 8192)
    dft_size.height = getOptimalDFTSize(block_size.height + templ_size.height - 1.);

    // To avoid wasting time doing small DFTs
    dft_size.width = std::max(dft_size.width, 512);
    dft_size.height = std::max(dft_size.height, 512);

    image_block.create(dft_size, CV_32F);
    templ_block.create(dft_size, CV_32F);
    result_data.create(dft_size, CV_32F);

    //spect_len = dft_size.height * (dft_size.width / 2 + 1);
    image_spect.create(dft_size.height, dft_size.width / 2 + 1, CV_32FC2);
    templ_spect.create(dft_size.height, dft_size.width / 2 + 1, CV_32FC2);
    result_spect.create(dft_size.height, dft_size.width / 2 + 1, CV_32FC2);

    // Use maximum result matrix block size for the estimated DFT block size
    block_size.width = std::min(dft_size.width - templ_size.width + 1, result_size.width);
    block_size.height = std::min(dft_size.height - templ_size.height + 1, result_size.height);
}

Size cv::ocl::ConvolveBuf::estimateBlockSize(Size result_size, Size /*templ_size*/)
{
    int width = (result_size.width + 2) / 3;
    int height = (result_size.height + 2) / 3;
    width = std::min(width, result_size.width);
    height = std::min(height, result_size.height);
    return Size(width, height);
}

static void convolve_run_fft(const oclMat &image, const oclMat &templ, oclMat &result, bool ccorr, ConvolveBuf& buf)
{
#if defined HAVE_CLAMDFFT
    CV_Assert(image.type() == CV_32F);
    CV_Assert(templ.type() == CV_32F);

    buf.create(image.size(), templ.size());
    result.create(buf.result_size, CV_32F);

    Size& block_size = buf.block_size;
    Size& dft_size = buf.dft_size;

    oclMat& image_block = buf.image_block;
    oclMat& templ_block = buf.templ_block;
    oclMat& result_data = buf.result_data;

    oclMat& image_spect = buf.image_spect;
    oclMat& templ_spect = buf.templ_spect;
    oclMat& result_spect = buf.result_spect;

    oclMat templ_roi = templ;
    copyMakeBorder(templ_roi, templ_block, 0, templ_block.rows - templ_roi.rows, 0,
                   templ_block.cols - templ_roi.cols, 0, Scalar());

    cv::ocl::dft(templ_block, templ_spect, dft_size);

    // Process all blocks of the result matrix
    for (int y = 0; y < result.rows; y += block_size.height)
    {
        for (int x = 0; x < result.cols; x += block_size.width)
        {
            Size image_roi_size(std::min(x + dft_size.width, image.cols) - x,
                                std::min(y + dft_size.height, image.rows) - y);
            Rect roi0(x, y, image_roi_size.width, image_roi_size.height);

            oclMat image_roi(image, roi0);

            copyMakeBorder(image_roi, image_block, 0, image_block.rows - image_roi.rows,
                           0, image_block.cols - image_roi.cols, 0, Scalar());

            cv::ocl::dft(image_block, image_spect, dft_size);

            mulSpectrums(image_spect, templ_spect, result_spect, 0,
                                 1.f / dft_size.area(), ccorr);

            cv::ocl::dft(result_spect, result_data, dft_size, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

            Size result_roi_size(std::min(x + block_size.width, result.cols) - x,
                                 std::min(y + block_size.height, result.rows) - y);

            Rect roi1(x, y, result_roi_size.width, result_roi_size.height);
            Rect roi2(0, 0, result_roi_size.width, result_roi_size.height);

            oclMat result_roi(result, roi1);
            oclMat result_block(result_data, roi2);

            result_block.copyTo(result_roi);
        }
    }

#else
    CV_Error(Error::OpenCLNoAMDBlasFft, "OpenCL DFT is not implemented");
#define UNUSED(x) (void)(x);
    UNUSED(image) UNUSED(templ) UNUSED(result) UNUSED(ccorr) UNUSED(buf)
#undef UNUSED
#endif
}

static void convolve_run(const oclMat &src, const oclMat &temp1, oclMat &dst, String kernelName, const cv::ocl::ProgramEntry* source)
{
    CV_Assert(src.depth() == CV_32FC1);
    CV_Assert(temp1.depth() == CV_32F);
    CV_Assert(temp1.cols <= 17 && temp1.rows <= 17);

    dst.create(src.size(), src.type());

    CV_Assert(src.cols == dst.cols && src.rows == dst.rows);
    CV_Assert(src.type() == dst.type());

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    int src_step = src.step / src.elemSize(), src_offset = src.offset / src.elemSize();
    int dst_step = dst.step / dst.elemSize(), dst_offset = dst.offset / dst.elemSize();
    int temp1_step = temp1.step / temp1.elemSize(), temp1_offset = temp1.offset / temp1.elemSize();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&temp1.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp1_step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp1.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp1.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&temp1_offset ));

    openCLExecuteKernel(src.clCxt, source, kernelName, globalThreads, localThreads, args, -1, dst.depth());
}

void cv::ocl::convolve(const oclMat &x, const oclMat &t, oclMat &y, bool ccorr)
{
    CV_Assert(x.depth() == CV_32F);
    CV_Assert(t.depth() == CV_32F);
    y.create(x.size(), x.type());
    String kernelName = "convolve";
    if(t.cols > 17 || t.rows > 17)
    {
        ConvolveBuf buf;
        convolve_run_fft(x, t, y, ccorr, buf);
    }
    else
    {
        CV_Assert(ccorr == false);
        convolve_run(x, t, y, kernelName, &imgproc_convolve);
    }
}
void cv::ocl::convolve(const oclMat &image, const oclMat &templ, oclMat &result, bool ccorr, ConvolveBuf& buf)
{
    result.create(image.size(), image.type());
    convolve_run_fft(image, templ, result, ccorr, buf);
}
