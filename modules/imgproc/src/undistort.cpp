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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

namespace cv
{

Mat getDefaultNewCameraMatrix( const Mat& cameraMatrix, Size imgsize,
                               bool centerPrincipalPoint )
{
    if( !centerPrincipalPoint && cameraMatrix.type() == CV_64F )
        return cameraMatrix;
    
    Mat newCameraMatrix;
    cameraMatrix.convertTo(newCameraMatrix, CV_64F);
    if( centerPrincipalPoint )
    {
        ((double*)newCameraMatrix.data)[2] = (imgsize.width-1)*0.5;
        ((double*)newCameraMatrix.data)[5] = (imgsize.height-1)*0.5;
    }
    return newCameraMatrix;
}

void initUndistortRectifyMap( const Mat& _cameraMatrix, const Mat& _distCoeffs,
                              const Mat& matR, const Mat& _newCameraMatrix,
                              Size size, int m1type, Mat& map1, Mat& map2 )
{
    if( m1type <= 0 )
        m1type = CV_16SC2;
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );
    map1.create( size, m1type );
    if( m1type != CV_32FC2 )
        map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
    else
        map2.release();

    Mat_<double> R = Mat_<double>::eye(3, 3), distCoeffs;
    Mat_<double> A = Mat_<double>(_cameraMatrix), Ar;

    if( _newCameraMatrix.data )
        Ar = Mat_<double>(_newCameraMatrix);
    else
        Ar = getDefaultNewCameraMatrix( A, size, true );

    if( matR.data )
        R = Mat_<double>(matR);

    if( _distCoeffs.data )
        distCoeffs = Mat_<double>(_distCoeffs);
    else
    {
        distCoeffs.create(5, 1);
        distCoeffs = 0.;
    }

    CV_Assert( A.size() == Size(3,3) && A.size() == R.size() );
    CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));
    Mat_<double> iR = (Ar.colRange(0,3)*R).inv(DECOMP_LU);
    const double* ir = &iR(0,0);

    double u0 = A(0, 2),  v0 = A(1, 2);
    double fx = A(0, 0),  fy = A(1, 1);

    CV_Assert( distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(1, 5) ||
               distCoeffs.size() == Size(4, 1) || distCoeffs.size() == Size(5, 1));

    if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() )
        distCoeffs = distCoeffs.t();

    double k1 = ((double*)distCoeffs.data)[0];
    double k2 = ((double*)distCoeffs.data)[1];
    double p1 = ((double*)distCoeffs.data)[2];
    double p2 = ((double*)distCoeffs.data)[3];
    double k3 = distCoeffs.cols + distCoeffs.rows - 1 == 5 ? ((double*)distCoeffs.data)[4] : 0.;

    for( int i = 0; i < size.height; i++ )
    {
        float* m1f = (float*)(map1.data + map1.step*i);
        float* m2f = (float*)(map2.data + map2.step*i);
        short* m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;
        double _x = i*ir[1] + ir[2], _y = i*ir[4] + ir[5], _w = i*ir[7] + ir[8];

        for( int j = 0; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6] )
        {
            double w = 1./_w, x = _x*w, y = _y*w;
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2, _2xy = 2*x*y;
            double kr = 1 + ((k3*r2 + k2)*r2 + k1)*r2;
            double u = fx*(x*kr + p1*_2xy + p2*(r2 + 2*x2)) + u0;
            double v = fy*(y*kr + p1*(r2 + 2*y2) + p2*_2xy) + v0;
            if( m1type == CV_16SC2 )
            {
                int iu = saturate_cast<int>(u*INTER_TAB_SIZE);
                int iv = saturate_cast<int>(v*INTER_TAB_SIZE);
                m1[j*2] = (short)(iu >> INTER_BITS);
                m1[j*2+1] = (short)(iv >> INTER_BITS);
                m2[j] = (ushort)((iv & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }
            else
            {
                m1f[j*2] = (float)u;
                m1f[j*2+1] = (float)v;
            }
        }
    }
}


void undistort( const Mat& src, Mat& dst, const Mat& _cameraMatrix,
                const Mat& _distCoeffs, const Mat& _newCameraMatrix )
{
    dst.create( src.size(), src.type() );
    CV_Assert( dst.data != src.data );

    int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
    Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

    Mat_<double> A, distCoeffs, Ar, I = Mat_<double>::eye(3,3);

    _cameraMatrix.convertTo(A, CV_64F);
    if( _distCoeffs.data )
        distCoeffs = Mat_<double>(_distCoeffs);
    else
    {
        distCoeffs.create(5, 1);
        distCoeffs = 0.;
    }

    if( _newCameraMatrix.data )
        _newCameraMatrix.convertTo(Ar, CV_64F);
    else
        A.copyTo(Ar);

    double v0 = Ar(1, 2);
    for( int y = 0; y < src.rows; y += stripe_size0 )
    {
        int stripe_size = std::min( stripe_size0, src.rows - y );
        Ar(1, 2) = v0 - y;
        Mat map1_part = map1.rowRange(0, stripe_size),
            map2_part = map2.rowRange(0, stripe_size),
            dst_part = dst.rowRange(y, y + stripe_size);

        initUndistortRectifyMap( A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
                                 map1_part.type(), map1_part, map2_part );
        remap( src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT );
    }
}

}


CV_IMPL void
cvUndistort2( const CvArr* srcarr, CvArr* dstarr, const CvMat* Aarr, const CvMat* dist_coeffs, const CvMat* newAarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs = cv::cvarrToMat(dist_coeffs), newA;
    if( newAarr )
        newA = cv::cvarrToMat(newAarr);

    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
    cv::undistort( src, dst, A, distCoeffs, newA );
}


CV_IMPL void cvInitUndistortMap( const CvMat* Aarr, const CvMat* dist_coeffs,
                                 CvArr* mapxarr, CvArr* mapyarr )
{
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs = cv::cvarrToMat(dist_coeffs);
    cv::Mat mapx = cv::cvarrToMat(mapxarr), mapy, mapx0 = mapx, mapy0;

    if( mapyarr )
        mapy0 = mapy = cv::cvarrToMat(mapyarr);

    cv::initUndistortRectifyMap( A, distCoeffs, cv::Mat(), A,
                                 mapx.size(), mapx.type(), mapx, mapy );
    CV_Assert( mapx0.data == mapx.data && mapy0.data == mapy.data );
}

void
cvInitUndistortRectifyMap( const CvMat* Aarr, const CvMat* dist_coeffs,
    const CvMat *Rarr, const CvMat* ArArr, CvArr* mapxarr, CvArr* mapyarr )
{
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs, R, Ar;
    cv::Mat mapx = cv::cvarrToMat(mapxarr), mapy, mapx0 = mapx, mapy0;

    if( mapyarr )
        mapy0 = mapy = cv::cvarrToMat(mapyarr);

    if( dist_coeffs )
        distCoeffs = cv::cvarrToMat(dist_coeffs);
    if( Rarr )
        R = cv::cvarrToMat(Rarr);
    if( ArArr )
        Ar = cv::cvarrToMat(ArArr);

    cv::initUndistortRectifyMap( A, distCoeffs, R, Ar, mapx.size(), mapx.type(), mapx, mapy );
    CV_Assert( mapx0.data == mapx.data && mapy0.data == mapy.data );
}


void cvUndistortPoints( const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
                   const CvMat* _distCoeffs,
                   const CvMat* matR, const CvMat* matP )
{
    double A[3][3], RR[3][3], k[5]={0,0,0,0,0}, fx, fy, ifx, ify, cx, cy;
    CvMat matA=cvMat(3, 3, CV_64F, A), _Dk;
    CvMat _RR=cvMat(3, 3, CV_64F, RR);
    const CvPoint2D32f* srcf;
    const CvPoint2D64f* srcd;
    CvPoint2D32f* dstf;
    CvPoint2D64f* dstd;
    int stype, dtype;
    int sstep, dstep;
    int i, j, n, iters = 1;

    CV_Assert( CV_IS_MAT(_src) && CV_IS_MAT(_dst) &&
        (_src->rows == 1 || _src->cols == 1) &&
        (_dst->rows == 1 || _dst->cols == 1) &&
        _src->cols + _src->rows - 1 == _dst->rows + _dst->cols - 1 &&
        (CV_MAT_TYPE(_src->type) == CV_32FC2 || CV_MAT_TYPE(_src->type) == CV_64FC2) &&
        (CV_MAT_TYPE(_dst->type) == CV_32FC2 || CV_MAT_TYPE(_dst->type) == CV_64FC2));

    CV_Assert( CV_IS_MAT(_cameraMatrix) &&
        _cameraMatrix->rows == 3 && _cameraMatrix->cols == 3 );

    cvConvert( _cameraMatrix, &matA );

    if( _distCoeffs )
    {
        CV_Assert( CV_IS_MAT(_distCoeffs) &&
            (_distCoeffs->rows == 1 || _distCoeffs->cols == 1) &&
            (_distCoeffs->rows*_distCoeffs->cols == 4 ||
            _distCoeffs->rows*_distCoeffs->cols == 5) );

        _Dk = cvMat( _distCoeffs->rows, _distCoeffs->cols,
            CV_MAKETYPE(CV_64F,CV_MAT_CN(_distCoeffs->type)), k);
        
        cvConvert( _distCoeffs, &_Dk );
        iters = 5;
    }

    if( matR )
    {
        CV_Assert( CV_IS_MAT(matR) && matR->rows == 3 && matR->cols == 3 );
        cvConvert( matR, &_RR );
    }
    else
        cvSetIdentity(&_RR);

    if( matP )
    {
        double PP[3][3];
        CvMat _P3x3, _PP=cvMat(3, 3, CV_64F, PP);
        CV_Assert( CV_IS_MAT(matP) && matP->rows == 3 && (matP->cols == 3 || matP->cols == 4));
        cvConvert( cvGetCols(matP, &_P3x3, 0, 3), &_PP );
        cvMatMul( &_PP, &_RR, &_RR );
    }

    srcf = (const CvPoint2D32f*)_src->data.ptr;
    srcd = (const CvPoint2D64f*)_src->data.ptr;
    dstf = (CvPoint2D32f*)_dst->data.ptr;
    dstd = (CvPoint2D64f*)_dst->data.ptr;
    stype = CV_MAT_TYPE(_src->type);
    dtype = CV_MAT_TYPE(_dst->type);
    sstep = _src->rows == 1 ? 1 : _src->step/CV_ELEM_SIZE(stype);
    dstep = _dst->rows == 1 ? 1 : _dst->step/CV_ELEM_SIZE(dtype);

    n = _src->rows + _src->cols - 1;

    fx = A[0][0];
    fy = A[1][1];
    ifx = 1./fx;
    ify = 1./fy;
    cx = A[0][2];
    cy = A[1][2];

    for( i = 0; i < n; i++ )
    {
        double x, y, x0, y0;
        if( stype == CV_32FC2 )
        {
            x = srcf[i*sstep].x;
            y = srcf[i*sstep].y;
        }
        else
        {
            x = srcd[i*sstep].x;
            y = srcd[i*sstep].y;
        }

        x0 = x = (x - cx)*ifx;
        y0 = y = (y - cy)*ify;

        // compensate distortion iteratively
        for( j = 0; j < iters; j++ )
        {
            double r2 = x*x + y*y;
            double icdist = 1./(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
            double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
            double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
            x = (x0 - deltaX)*icdist;
            y = (y0 - deltaY)*icdist;
        }

        double xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
        double yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
        double ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
        x = xx*ww;
        y = yy*ww;

        if( dtype == CV_32FC2 )
        {
            dstf[i*dstep].x = (float)x;
            dstf[i*dstep].y = (float)y;
        }
        else
        {
            dstd[i*dstep].x = x;
            dstd[i*dstep].y = y;
        }
    }
}


void cv::undistortPoints( const Mat& src, Mat& dst,
                          const Mat& cameraMatrix, const Mat& distCoeffs,
                          const Mat& R, const Mat& P )
{
    CV_Assert( src.isContinuous() && src.depth() == CV_32F &&
              ((src.rows == 1 && src.channels() == 2) || src.cols*src.channels() == 2));
    
    dst.create(src.size(), src.type());
    CvMat _src = src, _dst = dst, _cameraMatrix = cameraMatrix;
    CvMat matR, matP, _distCoeffs, *pR=0, *pP=0, *pD=0;
    if( R.data )
        pR = &(matR = R);
    if( P.data )
        pP = &(matP = P);
    if( distCoeffs.data )
        pD = &(_distCoeffs = distCoeffs);
    cvUndistortPoints(&_src, &_dst, &_cameraMatrix, pD, pR, pP);
}

void cv::undistortPoints( const Mat& src, std::vector<Point2f>& dst,
                          const Mat& cameraMatrix, const Mat& distCoeffs,
                          const Mat& R, const Mat& P )
{
    size_t sz = src.cols*src.rows*src.channels()/2;
    CV_Assert( src.isContinuous() && src.depth() == CV_32F &&
               ((src.rows == 1 && src.channels() == 2) || src.cols*src.channels() == 2));
    
    dst.resize(sz);
    CvMat _src = src, _dst = Mat(dst), _cameraMatrix = cameraMatrix;
    CvMat matR, matP, _distCoeffs, *pR=0, *pP=0, *pD=0;
    if( R.data )
        pR = &(matR = R);
    if( P.data )
        pP = &(matP = P);
    if( distCoeffs.data )
        pD = &(_distCoeffs = distCoeffs);
    cvUndistortPoints(&_src, &_dst, &_cameraMatrix, pD, pR, pP);
}

/*  End of file  */
