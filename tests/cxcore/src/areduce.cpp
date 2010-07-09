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

#include "cxcoretest.h"

using namespace cv;


class CV_ReduceTest : public CvTest
{
public:
    CV_ReduceTest() : CvTest( "reduce", "reduce" ) {}
protected:
    void run( int);
    int checkOp( const Mat& src, int dstType, int opType, const Mat& opRes, int dim, double eps );
    int checkCase( int srcType, int dstType, int dim, Size sz );
    int checkDim( int dim, Size sz );
    int checkSize( Size sz );
};

template<class Type>
void testReduce( const Mat& src, Mat& sum, Mat& avg, Mat& max, Mat& min, int dim )
{
    assert( src.channels() == 1 );
    if( dim == 0 ) // row
    {
        sum.create( 1, src.cols, CV_64FC1 ); 
        max.create( 1, src.cols, CV_64FC1 );
        min.create( 1, src.cols, CV_64FC1 );
    }
    else
    {
        sum.create( src.rows, 1, CV_64FC1 ); 
        max.create( src.rows, 1, CV_64FC1 );
        min.create( src.rows, 1, CV_64FC1 );
    }
    sum.setTo(Scalar(0));
    max.setTo(Scalar(-DBL_MAX));
    min.setTo(Scalar(DBL_MAX));
    
    const Mat_<Type>& src_ = src;
    Mat_<double>& sum_ = (Mat_<double>&)sum;
    Mat_<double>& min_ = (Mat_<double>&)min;
    Mat_<double>& max_ = (Mat_<double>&)max;

    if( dim == 0 )
    {
        for( int ri = 0; ri < src.rows; ri++ )
        {
            for( int ci = 0; ci < src.cols; ci++ )
            {
                sum_(0, ci) += src_(ri, ci);
                max_(0, ci) = std::max( max_(0, ci), (double)src_(ri, ci) );
                min_(0, ci) = std::min( min_(0, ci), (double)src_(ri, ci) );
            }
        }
    }
    else
    {
        for( int ci = 0; ci < src.cols; ci++ )
        {
            for( int ri = 0; ri < src.rows; ri++ )
            {
                sum_(ri, 0) += src_(ri, ci);
                max_(ri, 0) = std::max( max_(ri, 0), (double)src_(ri, ci) );
                min_(ri, 0) = std::min( min_(ri, 0), (double)src_(ri, ci) );
            }
        }
    }
    sum.convertTo( avg, CV_64FC1 );
    avg = avg * (1.0 / (dim==0 ? (double)src.rows : (double)src.cols));
}

void getMatTypeStr( int type, string& str)
{
    str = type == CV_8UC1 ? "CV_8UC1" :
          type == CV_8SC1 ? "CV_8SC1" :
          type == CV_16UC1 ? "CV_16UC1" :
          type == CV_16SC1 ? "CV_16SC1" :
          type == CV_32SC1 ? "CV_32SC1" :
          type == CV_32FC1 ? "CV_32FC1" :
          type == CV_64FC1 ? "CV_64FC1" : "unsupported matrix type";
}

int CV_ReduceTest::checkOp( const Mat& src, int dstType, int opType, const Mat& opRes, int dim, double eps )
{
    int srcType = src.type();
    bool support = false;
    if( opType == CV_REDUCE_SUM || opType == CV_REDUCE_AVG )
    {
        if( srcType == CV_8U && (dstType == CV_32S || dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_16U && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_16S && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_32F && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_64F && dstType == CV_64F)
            support = true;
    }
    else if( opType == CV_REDUCE_MAX )
    {
        if( srcType == CV_8U && dstType == CV_8U )
            support = true;
        if( srcType == CV_32F && dstType == CV_32F )
            support = true;
        if( srcType == CV_64F && dstType == CV_64F )
            support = true;
    }
    else if( opType == CV_REDUCE_MIN )
    {
        if( srcType == CV_8U && dstType == CV_8U)
            support = true;
        if( srcType == CV_32F && dstType == CV_32F)
            support = true;
        if( srcType == CV_64F && dstType == CV_64F)
            support = true;
    }
    if( !support )
        return CvTS::OK;

    assert( opRes.type() == CV_64FC1 );
    Mat _dst, dst;
    reduce( src, _dst, dim, opType, dstType );
    _dst.convertTo( dst, CV_64FC1 );
    if( norm( opRes, dst, NORM_INF ) > eps )
    {
        char msg[100];
        const char* opTypeStr = opType == CV_REDUCE_SUM ? "CV_REDUCE_SUM" :
            opType == CV_REDUCE_AVG ? "CV_REDUCE_AVG" :
            opType == CV_REDUCE_MAX ? "CV_REDUCE_MAX" :
            opType == CV_REDUCE_MIN ? "CV_REDUCE_MIN" : "unknown operation type";
        string srcTypeStr, dstTypeStr;
        getMatTypeStr( src.type(), srcTypeStr );
        getMatTypeStr( dstType, dstTypeStr );
        const char* dimStr = dim == 0 ? "ROWS" : "COLS";

        sprintf( msg, "bad accuracy with srcType = %s, dstType = %s, opType = %s, dim = %s",
            srcTypeStr.c_str(), dstTypeStr.c_str(), opTypeStr, dimStr );
        ts->printf( CvTS::LOG, msg );
        return CvTS::FAIL_BAD_ACCURACY;
    }
    return CvTS::OK;
}

int CV_ReduceTest::checkCase( int srcType, int dstType, int dim, Size sz )
{
    int code = CvTS::OK, tempCode;
    Mat src, sum, avg, max, min;

    src.create( sz, srcType );
    randu( src, Scalar(0), Scalar(100) );

    if( srcType == CV_8UC1 )
        testReduce<uchar>( src, sum, avg, max, min, dim );
    else if( srcType == CV_8SC1 )
        testReduce<char>( src, sum, avg, max, min, dim );
    else if( srcType == CV_16UC1 )
        testReduce<unsigned short int>( src, sum, avg, max, min, dim );
    else if( srcType == CV_16SC1 )
        testReduce<short int>( src, sum, avg, max, min, dim );
    else if( srcType == CV_32SC1 )
        testReduce<int>( src, sum, avg, max, min, dim );
    else if( srcType == CV_32FC1 )
        testReduce<float>( src, sum, avg, max, min, dim );
    else if( srcType == CV_64FC1 )
        testReduce<double>( src, sum, avg, max, min, dim );
    else 
        assert( 0 );

    // 1. sum
    tempCode = checkOp( src, dstType, CV_REDUCE_SUM, sum, dim, 
        srcType == CV_32FC1 && dstType == CV_32FC1 ? 0.05 : FLT_EPSILON );
    code = tempCode != CvTS::OK ? tempCode : code;

    // 2. avg
    tempCode = checkOp( src, dstType, CV_REDUCE_AVG, avg, dim, 
        dstType == CV_32SC1 ? 0.6 : 0.00007 );
    code = tempCode != CvTS::OK ? tempCode : code;

    // 3. max
    tempCode = checkOp( src, dstType, CV_REDUCE_MAX, max, dim, FLT_EPSILON );
    code = tempCode != CvTS::OK ? tempCode : code;

    // 4. min
    tempCode = checkOp( src, dstType, CV_REDUCE_MIN, min, dim, FLT_EPSILON );
    code = tempCode != CvTS::OK ? tempCode : code;
    
    return code;
}

int CV_ReduceTest::checkDim( int dim, Size sz )
{
    int code = CvTS::OK, tempCode;

    // CV_8UC1
    tempCode = checkCase( CV_8UC1, CV_8UC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_32SC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_32FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_64FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    // CV_16UC1
    tempCode = checkCase( CV_16UC1, CV_32FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_16UC1, CV_64FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    // CV_16SC1
    tempCode = checkCase( CV_16SC1, CV_32FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_16SC1, CV_64FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    // CV_32FC1
    tempCode = checkCase( CV_32FC1, CV_32FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkCase( CV_32FC1, CV_64FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    // CV_64FC1
    tempCode = checkCase( CV_64FC1, CV_64FC1, dim, sz );
    code = tempCode != CvTS::OK ? tempCode : code;

    return code;
}

int CV_ReduceTest::checkSize( Size sz )
{
    int code = CvTS::OK, tempCode;

    tempCode = checkDim( 0, sz ); // rows
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkDim( 1, sz ); // cols 
    code = tempCode != CvTS::OK ? tempCode : code;

    return code;
}

void CV_ReduceTest::run( int )
{
    int code = CvTS::OK, tempCode;
    
    tempCode = checkSize( Size(1,1) );
    code = tempCode != CvTS::OK ? tempCode : code;
    
    tempCode = checkSize( Size(1,100) );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkSize( Size(100,1) );
    code = tempCode != CvTS::OK ? tempCode : code;

    tempCode = checkSize( Size(1000,500) );
    code = tempCode != CvTS::OK ? tempCode : code;

    ts->set_failed_test_info( code );
}

CV_ReduceTest reduce_test;
