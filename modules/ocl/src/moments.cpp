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

#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include "opencl_kernels.hpp"

namespace cv
{
namespace ocl
{
// The function calculates center of gravity and the central second order moments
static void icvCompleteMomentState( CvMoments* moments )
{
    double cx = 0, cy = 0;
    double mu20, mu11, mu02;

    assert( moments != 0 );
    moments->inv_sqrt_m00 = 0;

    if( fabs(moments->m00) > DBL_EPSILON )
    {
        double inv_m00 = 1. / moments->m00;
        cx = moments->m10 * inv_m00;
        cy = moments->m01 * inv_m00;
        moments->inv_sqrt_m00 = std::sqrt( fabs(inv_m00) );
    }

    // mu20 = m20 - m10*cx
    mu20 = moments->m20 - moments->m10 * cx;
    // mu11 = m11 - m10*cy
    mu11 = moments->m11 - moments->m10 * cy;
    // mu02 = m02 - m01*cy
    mu02 = moments->m02 - moments->m01 * cy;

    moments->mu20 = mu20;
    moments->mu11 = mu11;
    moments->mu02 = mu02;

    // mu30 = m30 - cx*(3*mu20 + cx*m10)
    moments->mu30 = moments->m30 - cx * (3 * mu20 + cx * moments->m10);
    mu11 += mu11;
    // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
    moments->mu21 = moments->m21 - cx * (mu11 + cx * moments->m01) - cy * mu20;
    // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
    moments->mu12 = moments->m12 - cy * (mu11 + cy * moments->m10) - cx * mu02;
    // mu03 = m03 - cy*(3*mu02 + cy*m01)
    moments->mu03 = moments->m03 - cy * (3 * mu02 + cy * moments->m01);
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
        size_t localThreads[3]  = { llength, 1, 1};
        size_t globalThreads[3] = { lpt, 1, 1};
        std::vector<std::pair<size_t , const void *> > args;
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&contour->total ));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&reader_oclmat.data ));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst_a.data ));
        cl_int dst_step = (cl_int)dst_a.step;
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step ));

        openCLExecuteKernel2(dst_a.clCxt, &moments, "icvContourMoments", globalThreads, localThreads, args, -1, -1);

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

static void ocl_cvMoments( const void* array, CvMoments* mom, int binary )
{
    const int TILE_SIZE = 256;
    int type, depth, cn, coi = 0;
    CvMat stub, *mat = (CvMat*)array;
    CvContour contourHeader;
    CvSeq* contour = 0;
    CvSeqBlock block;
    if( CV_IS_SEQ( array ))
    {
        contour = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( contour ))
            CV_Error( CV_StsBadArg, "The passed sequence is not a valid contour" );
    }

    if( !mom )
        CV_Error( CV_StsNullPtr, "" );

    memset( mom, 0, sizeof(*mom));

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
    if( contour )
    {
        icvContourMoments( contour, mom );
        return;
    }

    type = CV_MAT_TYPE( mat->type );
    depth = CV_MAT_DEPTH( type );
    cn = CV_MAT_CN( type );

    cv::Size size = cvGetMatSize( mat );
    if( cn > 1 && coi == 0 )
        CV_Error( CV_StsBadArg, "Invalid image type" );

    if( size.width <= 0 || size.height <= 0 )
        return;

    cv::Mat src0 = cv::cvarrToMat(mat);
    cv::ocl::oclMat src(src0);
    cv::Size tileSize;
    int blockx,blocky;
    if(size.width%TILE_SIZE == 0)
        blockx = size.width/TILE_SIZE;
    else
        blockx = size.width/TILE_SIZE + 1;
    if(size.height%TILE_SIZE == 0)
        blocky = size.height/TILE_SIZE;
    else
        blocky = size.height/TILE_SIZE + 1;
    oclMat dst_m(blocky * 10, blockx, CV_64FC1);
    oclMat sum(1, 10, CV_64FC1);
    int tile_width  = std::min(size.width,TILE_SIZE);
    int tile_height = std::min(size.height,TILE_SIZE);
    size_t localThreads[3]  = { tile_height, 1, 1};
    size_t globalThreads[3] = { size.height, blockx, 1};
    std::vector<std::pair<size_t , const void *> > args,args_sum;
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst_m.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_m.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_m.step ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&blocky ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&depth ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cn ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&coi ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&binary ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&TILE_SIZE ));
    openCLExecuteKernel2(Context::getContext(), &moments, "CvMoments", globalThreads, localThreads, args, -1, depth);

    size_t localThreadss[3]  = { 128, 1, 1};
    size_t globalThreadss[3] = { 128, 1, 1};
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&tile_height ));
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&tile_width ));
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&TILE_SIZE ));
    args_sum.push_back( std::make_pair( sizeof(cl_mem) , (void *)&sum.data ));
    args_sum.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst_m.data ));
    args_sum.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_m.step ));
    openCLExecuteKernel2(Context::getContext(), &moments, "dst_sum", globalThreadss, localThreadss, args_sum, -1, -1);

    Mat dstsum(sum);
    mom->m00 = dstsum.at<double>(0, 0);
    mom->m10 = dstsum.at<double>(0, 1);
    mom->m01 = dstsum.at<double>(0, 2);
    mom->m20 = dstsum.at<double>(0, 3);
    mom->m11 = dstsum.at<double>(0, 4);
    mom->m02 = dstsum.at<double>(0, 5);
    mom->m30 = dstsum.at<double>(0, 6);
    mom->m21 = dstsum.at<double>(0, 7);
    mom->m12 = dstsum.at<double>(0, 8);
    mom->m03 = dstsum.at<double>(0, 9);

    icvCompleteMomentState( mom );
}


Moments ocl_moments( InputArray _array, bool binaryImage )
{
    CvMoments om;
    Mat arr = _array.getMat();
    CvMat c_array = arr;
    ocl_cvMoments(&c_array, &om, binaryImage);
    return om;
}

}

}
