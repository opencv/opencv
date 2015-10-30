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
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma,  jin@multicorewareinc.com
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
//     and/or other Materials provided with the distribution.
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

#if defined _MSC_VER
#define snprintf sprintf_s
#endif
namespace cv
{
    namespace ocl
    {
        // The function calculates center of gravity and the central second order moments
        static void icvCompleteMomentState( CvMoments* lmoments )
        {
            double cx = 0, cy = 0;
            double mu20, mu11, mu02;

            assert( lmoments != 0 );
            lmoments->inv_sqrt_m00 = 0;

            if( fabs(lmoments->m00) > DBL_EPSILON )
            {
                double inv_m00 = 1. / lmoments->m00;
                cx = lmoments->m10 * inv_m00;
                cy = lmoments->m01 * inv_m00;
                lmoments->inv_sqrt_m00 = std::sqrt( fabs(inv_m00) );
            }

            // mu20 = m20 - m10*cx
            mu20 = lmoments->m20 - lmoments->m10 * cx;
            // mu11 = m11 - m10*cy
            mu11 = lmoments->m11 - lmoments->m10 * cy;
            // mu02 = m02 - m01*cy
            mu02 = lmoments->m02 - lmoments->m01 * cy;

            lmoments->mu20 = mu20;
            lmoments->mu11 = mu11;
            lmoments->mu02 = mu02;

            // mu30 = m30 - cx*(3*mu20 + cx*m10)
            lmoments->mu30 = lmoments->m30 - cx * (3 * mu20 + cx * lmoments->m10);
            mu11 += mu11;
            // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
            lmoments->mu21 = lmoments->m21 - cx * (mu11 + cx * lmoments->m01) - cy * mu20;
            // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
            lmoments->mu12 = lmoments->m12 - cy * (mu11 + cy * lmoments->m10) - cx * mu02;
            // mu03 = m03 - cy*(3*mu02 + cy*m01)
            lmoments->mu03 = lmoments->m03 - cy * (3 * mu02 + cy * lmoments->m01);
        }


        static void icvContourMoments( CvSeq* contour, CvMoments* mom )
        {
            if( contour->total )
            {
                CvSeqReader reader;
                int lpt = contour->total;
                double a00, a10, a01, a20, a11, a02, a30, a21, a12, a03;

                cvStartReadSeq( contour, &reader, 0 );

                size_t reader_size = lpt << 1;
                cv::Mat reader_mat(1,reader_size,CV_32FC1);

                bool is_float = CV_SEQ_ELTYPE(contour) == CV_32FC2;

                if (!cv::ocl::Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE) && is_float)
                {
                    CV_Error(CV_StsUnsupportedFormat, "Moments - double is not supported by your GPU!");
                }

                if( is_float )
                {
                    for(size_t i = 0; i < reader_size; ++i)
                    {
                        reader_mat.at<float>(0, i++) = ((CvPoint2D32f*)(reader.ptr))->x;
                        reader_mat.at<float>(0, i) = ((CvPoint2D32f*)(reader.ptr))->y;
                        CV_NEXT_SEQ_ELEM( contour->elem_size, reader );
                    }
                }
                else
                {
                    for(size_t i = 0; i < reader_size; ++i)
                    {
                        reader_mat.at<float>(0, i++) = ((CvPoint*)(reader.ptr))->x;
                        reader_mat.at<float>(0, i) = ((CvPoint*)(reader.ptr))->y;
                        CV_NEXT_SEQ_ELEM( contour->elem_size, reader );
                    }
                }

                cv::ocl::oclMat dst_a(10, lpt, CV_64FC1);
                cv::ocl::oclMat reader_oclmat(reader_mat);
                int llength = std::min(lpt,128);
                size_t localThreads[3]  = { (size_t)llength, 1, 1};
                size_t globalThreads[3] = { (size_t)lpt, 1, 1};
                vector<pair<size_t , const void *> > args;
                args.push_back( make_pair( sizeof(cl_int) , (void *)&contour->total ));
                args.push_back( make_pair( sizeof(cl_mem) , (void *)&reader_oclmat.data ));
                args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst_a.data ));
                cl_int dst_step = (cl_int)dst_a.step;
                args.push_back( make_pair( sizeof(cl_int) , (void *)&dst_step ));

                char builOption[128];
                snprintf(builOption, 128, "-D CV_8UC1");

                openCLExecuteKernel(dst_a.clCxt, &moments, "icvContourMoments", globalThreads, localThreads, args, -1, -1, builOption);

                cv::Mat dst(dst_a);
                a00 = a10 = a01 = a20 = a11 = a02 = a30 = a21 = a12 = a03 = 0.0;
                if (!cv::ocl::Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
                {
                    for (int i = 0; i < contour->total; ++i)
                    {
                        a00 += dst.at<cl_long>(0, i);
                        a10 += dst.at<cl_long>(1, i);
                        a01 += dst.at<cl_long>(2, i);
                        a20 += dst.at<cl_long>(3, i);
                        a11 += dst.at<cl_long>(4, i);
                        a02 += dst.at<cl_long>(5, i);
                        a30 += dst.at<cl_long>(6, i);
                        a21 += dst.at<cl_long>(7, i);
                        a12 += dst.at<cl_long>(8, i);
                        a03 += dst.at<cl_long>(9, i);
                    }
                }
                else
                {
                    a00 = cv::sum(dst.row(0))[0];
                    a10 = cv::sum(dst.row(1))[0];
                    a01 = cv::sum(dst.row(2))[0];
                    a20 = cv::sum(dst.row(3))[0];
                    a11 = cv::sum(dst.row(4))[0];
                    a02 = cv::sum(dst.row(5))[0];
                    a30 = cv::sum(dst.row(6))[0];
                    a21 = cv::sum(dst.row(7))[0];
                    a12 = cv::sum(dst.row(8))[0];
                    a03 = cv::sum(dst.row(9))[0];
                }

                double db1_2, db1_6, db1_12, db1_24, db1_20, db1_60;
                if( fabs(a00) > FLT_EPSILON )
                {
                    if( a00 > 0 )
                    {
                        db1_2 = 0.5;
                        db1_6 = 0.16666666666666666666666666666667;
                        db1_12 = 0.083333333333333333333333333333333;
                        db1_24 = 0.041666666666666666666666666666667;
                        db1_20 = 0.05;
                        db1_60 = 0.016666666666666666666666666666667;
                    }
                    else
                    {
                        db1_2 = -0.5;
                        db1_6 = -0.16666666666666666666666666666667;
                        db1_12 = -0.083333333333333333333333333333333;
                        db1_24 = -0.041666666666666666666666666666667;
                        db1_20 = -0.05;
                        db1_60 = -0.016666666666666666666666666666667;
                    }

                    // spatial moments
                    mom->m00 = a00 * db1_2;
                    mom->m10 = a10 * db1_6;
                    mom->m01 = a01 * db1_6;
                    mom->m20 = a20 * db1_12;
                    mom->m11 = a11 * db1_24;
                    mom->m02 = a02 * db1_12;
                    mom->m30 = a30 * db1_20;
                    mom->m21 = a21 * db1_60;
                    mom->m12 = a12 * db1_60;
                    mom->m03 = a03 * db1_20;

                    icvCompleteMomentState( mom );
                }
            }
        }

        Moments ocl_moments(oclMat& src, bool binary) //for image
        {
            CV_Assert(src.oclchannels() == 1);
            if(src.type() == CV_64FC1 && !Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
            {
                CV_Error(CV_StsUnsupportedFormat, "Moments - double is not supported by your GPU!");
            }

            if(binary)
            {
                oclMat mask;
                if(src.type() != CV_8UC1)
                {
                    src.convertTo(mask, CV_8UC1);
                }
                oclMat src8u(src.size(), CV_8UC1);
                src8u.setTo(Scalar(255), mask);
                src = src8u;
            }
            const int TILE_SIZE = 256;

            CvMoments mom;
            memset(&mom, 0, sizeof(mom));

            cv::Size size = src.size();
            int blockx, blocky;
            blockx = (size.width + TILE_SIZE - 1)/TILE_SIZE;
            blocky = (size.height + TILE_SIZE - 1)/TILE_SIZE;

            oclMat dst_m;
            int tile_height = TILE_SIZE;

            size_t localThreads[3]  = {1, (size_t)tile_height, 1};
            size_t globalThreads[3] = {(size_t)blockx, (size_t)size.height, 1};

            if(Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
            {
                dst_m.create(blocky * 10, blockx, CV_64FC1);
            }else
            {
                dst_m.create(blocky * 10, blockx, CV_32FC1);
            }

            int src_step = (int)(src.step/src.elemSize());
            int dstm_step = (int)(dst_m.step/dst_m.elemSize());

            vector<pair<size_t , const void *> > args;
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&src_step ));
            args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst_m.data ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dst_m.cols ));
            args.push_back( make_pair( sizeof(cl_int) , (void *)&dstm_step ));

            int binary_;
            if(binary)
                binary_ = 1;
            else
                binary_ = 0;
            args.push_back( make_pair( sizeof(cl_int) , (void *)&binary_));

            char builOption[128];
            if(binary || src.type() == CV_8UC1)
            {
                snprintf(builOption, 128, "-D CV_8UC1");
            }else if(src.type() == CV_16UC1)
            {
                snprintf(builOption, 128, "-D CV_16UC1");
            }else if(src.type() == CV_16SC1)
            {
                snprintf(builOption, 128, "-D CV_16SC1");
            }else if(src.type() == CV_32FC1)
            {
                snprintf(builOption, 128, "-D CV_32FC1");
            }else if(src.type() == CV_64FC1)
            {
                snprintf(builOption, 128, "-D CV_64FC1");
            }else
            {
                CV_Error( CV_StsUnsupportedFormat, "" );
            }

            openCLExecuteKernel(Context::getContext(), &moments, "CvMoments", globalThreads, localThreads, args, -1, -1, builOption);

            Mat tmp(dst_m);
            tmp.convertTo(tmp, CV_64FC1);

            double tmp_m[10] = {0};

            for(int j = 0; j < tmp.rows; j += 10)
            {
                for(int i = 0; i < tmp.cols; i++)
                {
                    tmp_m[0] += tmp.at<double>(j, i);
                    tmp_m[1] += tmp.at<double>(j + 1, i);
                    tmp_m[2] += tmp.at<double>(j + 2, i);
                    tmp_m[3] += tmp.at<double>(j + 3, i);
                    tmp_m[4] += tmp.at<double>(j + 4, i);
                    tmp_m[5] += tmp.at<double>(j + 5, i);
                    tmp_m[6] += tmp.at<double>(j + 6, i);
                    tmp_m[7] += tmp.at<double>(j + 7, i);
                    tmp_m[8] += tmp.at<double>(j + 8, i);
                    tmp_m[9] += tmp.at<double>(j + 9, i);
                }
            }

            mom.m00 = tmp_m[0];
            mom.m10 = tmp_m[1];
            mom.m01 = tmp_m[2];
            mom.m20 = tmp_m[3];
            mom.m11 = tmp_m[4];
            mom.m02 = tmp_m[5];
            mom.m30 = tmp_m[6];
            mom.m21 = tmp_m[7];
            mom.m12 = tmp_m[8];
            mom.m03 = tmp_m[9];
            icvCompleteMomentState( &mom );
            return mom;
        }

        Moments ocl_moments(InputArray _contour) //for contour
        {
            CvMoments mom;
            memset(&mom, 0, sizeof(mom));

            Mat arr = _contour.getMat();
            CvMat c_array = arr;

            const void* array = &c_array;

            CvSeq* contour = 0;
            if( CV_IS_SEQ( array ))
            {
                contour = (CvSeq*)(array);
                if( !CV_IS_SEQ_POINT_SET( contour ))
                    CV_Error( CV_StsBadArg, "The passed sequence is not a valid contour" );
            }

            int type, coi = 0;

            CvMat stub, *mat = (CvMat*)(array);
            CvContour contourHeader;
            CvSeqBlock block;

            if( !contour )
            {
                mat = cvGetMat( mat, &stub, &coi );
                type = CV_MAT_TYPE( mat->type );

                if( type == CV_32SC2 || type == CV_32FC2 )
                {
                    contour = cvPointSeqFromMat(
                        CV_SEQ_KIND_CURVE | CV_SEQ_FLAG_CLOSED,
                        mat, &contourHeader, &block );
                }
            }

            CV_Assert(contour);

            icvContourMoments(contour, &mom);
            return mom;
        }
    }
}
