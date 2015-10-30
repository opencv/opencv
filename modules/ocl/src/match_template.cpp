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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
        void matchTemplate_SQDIFF(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void matchTemplate_SQDIFF_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void convolve_32F(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void matchTemplate_CCORR(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void matchTemplate_CCORR_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void matchTemplate_CCOFF(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);

        void matchTemplate_CCOFF_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf);


        void matchTemplateNaive_SQDIFF(
            const oclMat &image, const oclMat &templ, oclMat &result, int cn);

        void matchTemplateNaive_CCORR(
            const oclMat &image, const oclMat &templ, oclMat &result, int cn);

        void extractFirstChannel_32F(
            const oclMat &image, oclMat &result);

        // Evaluates optimal template's area threshold. If
        // template's area is less  than the threshold, we use naive match
        // template version, otherwise FFT-based (if available)
        static bool useNaive(int , int , Size )
        {
            // FIXME!
            //   always use naive until convolve is imported
            return true;
        }

        //////////////////////////////////////////////////////////////////////
        // SQDIFF
        void matchTemplate_SQDIFF(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf & buf)
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            if (useNaive(CV_TM_SQDIFF, image.depth(), templ.size()))
            {
                matchTemplateNaive_SQDIFF(image, templ, result, image.oclchannels());
                return;
            }
            else
            {
                buf.image_sqsums.resize(1);

                // TODO, add double support for ocl::integral
                // use CPU integral temporarily
                Mat sums, sqsums;
                cv::integral(Mat(image.reshape(1)), sums, sqsums);
                buf.image_sqsums[0] = sqsums;

                unsigned long long templ_sqsum = (unsigned long long)sqrSum(templ.reshape(1))[0];
                matchTemplate_CCORR(image, templ, result, buf);

                //port CUDA's matchTemplatePrepared_SQDIFF_8U
                Context *clCxt = image.clCxt;
                string kernelName = "matchTemplate_Prepared_SQDIFF";
                vector< pair<size_t, const void *> > args;

                args.push_back( make_pair( sizeof(cl_mem), (void *)&buf.image_sqsums[0].data));
                args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data));
                args.push_back( make_pair( sizeof(cl_ulong), (void *)&templ_sqsum));
                args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows));
                args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols));
                args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows));
                args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols));
                args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sqsums[0].offset));
                args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sqsums[0].step));
                args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
                args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

                size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
                size_t localThreads[3]  = {16, 16, 1};

                const char * build_opt = image.oclchannels() == 4 ? "-D CN4" : "";
                openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, 1, CV_8U, build_opt);
            }
        }

        void matchTemplate_SQDIFF_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf)
        {
            matchTemplate_CCORR(image, templ, result, buf);
            buf.image_sums.resize(1);

            integral(image.reshape(1), buf.image_sums[0]);

            unsigned long long templ_sqsum = (unsigned long long)sqrSum(templ.reshape(1))[0];

            Context *clCxt = image.clCxt;
            string kernelName = "matchTemplate_Prepared_SQDIFF_NORMED";
            vector< pair<size_t, const void *> > args;

            args.push_back( make_pair( sizeof(cl_mem), (void *)&buf.image_sums[0].data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data));
            args.push_back( make_pair( sizeof(cl_ulong), (void *)&templ_sqsum));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sums[0].offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sums[0].step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, 1, CV_8U);
        }

        void matchTemplateNaive_SQDIFF(
            const oclMat &image, const oclMat &templ, oclMat &result, int)
        {
            CV_Assert((image.depth() == CV_8U && templ.depth() == CV_8U )
                      || ((image.depth() == CV_32F && templ.depth() == CV_32F) && result.depth() == CV_32F)
                     );
            CV_Assert(image.oclchannels() == templ.oclchannels() && (image.oclchannels() == 1 || image.oclchannels() == 4) && result.oclchannels() == 1);
            CV_Assert(result.rows == image.rows - templ.rows + 1 && result.cols == image.cols - templ.cols + 1);

            Context *clCxt = image.clCxt;
            string kernelName = "matchTemplate_Naive_SQDIFF";

            vector< pair<size_t, const void *> > args;

            args.push_back( make_pair( sizeof(cl_mem), (void *)&image.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&templ.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, image.oclchannels(), image.depth());
        }

        //////////////////////////////////////////////////////////////////////
        // CCORR
        void convolve_32F(
            const oclMat &, const oclMat &, oclMat &, MatchTemplateBuf &)
        {
            CV_Error(-1, "convolve is not fully implemented yet");
        }

        void matchTemplate_CCORR(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf)
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            if (useNaive(CV_TM_CCORR, image.depth(), templ.size()))
            {
                matchTemplateNaive_CCORR(image, templ, result, image.oclchannels());
                return;
            }
            else
            {
                if(image.depth() == CV_8U && templ.depth() == CV_8U)
                {
                    image.convertTo(buf.imagef, CV_32F);
                    templ.convertTo(buf.templf, CV_32F);
                    convolve_32F(buf.imagef, buf.templf, result, buf);
                }
                else
                {
                    convolve_32F(image, templ, result, buf);
                }
            }
        }

        void matchTemplate_CCORR_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf)
        {
            matchTemplate_CCORR(image, templ, result, buf);
            buf.image_sums.resize(1);
            buf.image_sqsums.resize(1);

            integral(image.reshape(1), buf.image_sums[0], buf.image_sqsums[0]);

            unsigned long long templ_sqsum = (unsigned long long)sqrSum(templ.reshape(1))[0];

            Context *clCxt = image.clCxt;
            string kernelName = "normalizeKernel";
            vector< pair<size_t, const void *> > args;

            args.push_back( make_pair( sizeof(cl_mem), (void *)&buf.image_sqsums[0].data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data));
            args.push_back( make_pair( sizeof(cl_ulong), (void *)&templ_sqsum));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sqsums[0].offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&buf.image_sqsums[0].step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, 1, CV_8U);
        }

        void matchTemplateNaive_CCORR(
            const oclMat &image, const oclMat &templ, oclMat &result, int)
        {
            CV_Assert((image.depth() == CV_8U && templ.depth() == CV_8U )
                      || ((image.depth() == CV_32F && templ.depth() == CV_32F) && result.depth() == CV_32F)
                     );
            CV_Assert(image.oclchannels() == templ.oclchannels() && (image.oclchannels() == 1 || image.oclchannels() == 4) && result.oclchannels() == 1);
            CV_Assert(result.rows == image.rows - templ.rows + 1 && result.cols == image.cols - templ.cols + 1);

            Context *clCxt = image.clCxt;
            string kernelName = "matchTemplate_Naive_CCORR";

            vector< pair<size_t, const void *> > args;

            args.push_back( make_pair( sizeof(cl_mem), (void *)&image.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&templ.data));
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, image.oclchannels(), image.depth());
        }
        //////////////////////////////////////////////////////////////////////
        // CCOFF
        void matchTemplate_CCOFF(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf)
        {
            CV_Assert(image.depth() == CV_8U && templ.depth() == CV_8U);

            matchTemplate_CCORR(image, templ, result, buf);

            Context *clCxt = image.clCxt;
            string kernelName;

            kernelName = "matchTemplate_Prepared_CCOFF";
            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));
            Vec4f templ_sum = Vec4f::all(0);
            // to be continued in the following section
            if(image.oclchannels() == 1)
            {
                buf.image_sums.resize(1);
                integral(image, buf.image_sums[0]);

                templ_sum[0] = (float)sum(templ)[0] / templ.size().area();
                args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[0].data) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].offset) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].step) );
                args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[0]) );
            }
            else
            {

                split(image, buf.images);
                templ_sum = sum(templ) / templ.size().area();
                buf.image_sums.resize(buf.images.size());


                for(int i = 0; i < image.oclchannels(); i ++)
                {
                    integral(buf.images[i], buf.image_sums[i]);
                }
                switch(image.oclchannels())
                {
                case 4:
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[0].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[1].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[2].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[3].data) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].offset) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].step) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[0]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[1]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[2]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[3]) );
                    break;
                default:
                    CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
                    break;
                }
            }
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, image.oclchannels(), image.depth());
        }

        void matchTemplate_CCOFF_NORMED(
            const oclMat &image, const oclMat &templ, oclMat &result, MatchTemplateBuf &buf)
        {
            image.convertTo(buf.imagef, CV_32F);
            templ.convertTo(buf.templf, CV_32F);

            matchTemplate_CCORR(buf.imagef, buf.templf, result, buf);
            float scale = 1.f / templ.size().area();

            Context *clCxt = image.clCxt;
            string kernelName;

            kernelName = "matchTemplate_Prepared_CCOFF_NORMED";
            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&templ.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));
            args.push_back( make_pair( sizeof(cl_float), (void *)&scale) );

            Vec4f templ_sum   = Vec4f::all(0);
            Vec4f templ_sqsum = Vec4f::all(0);
            // to be continued in the following section
            if(image.oclchannels() == 1)
            {
                buf.image_sums.resize(1);
                buf.image_sqsums.resize(1);
                integral(image, buf.image_sums[0], buf.image_sqsums[0]);

                templ_sum[0]   = (float)sum(templ)[0];

                templ_sqsum[0] = sqrSum(templ)[0];

                templ_sqsum[0] -= scale * templ_sum[0] * templ_sum[0];
                templ_sum[0]   *= scale;

                args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[0].data) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].offset) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].step) );
                args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sqsums[0].data) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sqsums[0].offset) );
                args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sqsums[0].step) );
                args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[0]) );
                args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sqsum[0]) );
            }
            else
            {

                split(image, buf.images);
                templ_sum   = sum(templ);

                templ_sqsum = sqrSum(templ);

                templ_sqsum -= scale * templ_sum * templ_sum;

                float templ_sqsum_sum = 0;
                for(int i = 0; i < image.oclchannels(); i ++)
                {
                    templ_sqsum_sum += templ_sqsum[i] - scale * templ_sum[i] * templ_sum[i];
                }
                templ_sum   *= scale;
                buf.image_sums.resize(buf.images.size());
                buf.image_sqsums.resize(buf.images.size());

                for(int i = 0; i < image.oclchannels(); i ++)
                {
                    integral(buf.images[i], buf.image_sums[i], buf.image_sqsums[i]);
                }

                switch(image.oclchannels())
                {
                case 4:
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[0].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[1].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[2].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sums[3].data) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].offset) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sums[0].step) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sqsums[0].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sqsums[1].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sqsums[2].data) );
                    args.push_back( make_pair( sizeof(cl_mem),  (void *)&buf.image_sqsums[3].data) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sqsums[0].offset) );
                    args.push_back( make_pair( sizeof(cl_int),  (void *)&buf.image_sqsums[0].step) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[0]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[1]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[2]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sum[3]) );
                    args.push_back( make_pair( sizeof(cl_float), (void *)&templ_sqsum_sum) );
                    break;
                default:
                    CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
                    break;
                }
            }
            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, image.oclchannels(), image.depth());
        }
        void extractFirstChannel_32F(const oclMat &image, oclMat &result)
        {
            Context *clCxt = image.clCxt;
            string kernelName;

            kernelName = "extractFirstChannel";
            size_t globalThreads[3] = {(size_t)result.cols, (size_t)result.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};

            vector< pair<size_t, const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem), (void *)&image.data) );
            args.push_back( make_pair( sizeof(cl_mem), (void *)&result.data) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.rows) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.cols) );
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.offset));
            args.push_back( make_pair( sizeof(cl_int), (void *)&image.step));
            args.push_back( make_pair( sizeof(cl_int), (void *)&result.step));

            openCLExecuteKernel(clCxt, &match_template, kernelName, globalThreads, localThreads, args, -1, -1);
        }
    }/*ocl*/
} /*cv*/

void cv::ocl::matchTemplate(const oclMat &image, const oclMat &templ, oclMat &result, int method)
{
    MatchTemplateBuf buf;
    matchTemplate(image, templ, result, method, buf);
}
void cv::ocl::matchTemplate(const oclMat &image, const oclMat &templ, oclMat &result, int method, MatchTemplateBuf &buf)
{
    CV_Assert(image.type() == templ.type());
    CV_Assert(image.cols >= templ.cols && image.rows >= templ.rows);

    typedef void (*Caller)(const oclMat &, const oclMat &, oclMat &, MatchTemplateBuf &);

    const Caller callers[] =
    {
        ::matchTemplate_SQDIFF, ::matchTemplate_SQDIFF_NORMED,
        ::matchTemplate_CCORR, ::matchTemplate_CCORR_NORMED,
        ::matchTemplate_CCOFF, ::matchTemplate_CCOFF_NORMED
    };

    Caller caller = callers[method];
    CV_Assert(caller);
    caller(image, templ, result, buf);
}
