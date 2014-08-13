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

cv::Mat cv::getDefaultNewCameraMatrix( InputArray _cameraMatrix, Size imgsize,
                               bool centerPrincipalPoint )
{
    Mat cameraMatrix = _cameraMatrix.getMat();
    if( !centerPrincipalPoint && cameraMatrix.type() == CV_64F )
        return cameraMatrix;

    Mat newCameraMatrix;
    cameraMatrix.convertTo(newCameraMatrix, CV_64F);
    if( centerPrincipalPoint )
    {
        newCameraMatrix.ptr<double>()[2] = (imgsize.width-1)*0.5;
        newCameraMatrix.ptr<double>()[5] = (imgsize.height-1)*0.5;
    }
    return newCameraMatrix;
}

void cv::initUndistortRectifyMap( InputArray _cameraMatrix, InputArray _distCoeffs,
                              InputArray _matR, InputArray _newCameraMatrix,
                              Size size, int m1type, OutputArray _map1, OutputArray _map2 )
{
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();
    Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    if( m1type <= 0 )
        m1type = CV_16SC2;
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );
    _map1.create( size, m1type );
    Mat map1 = _map1.getMat(), map2;
    if( m1type != CV_32FC2 )
    {
        _map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        map2 = _map2.getMat();
    }
    else
        _map2.release();

    Mat_<double> R = Mat_<double>::eye(3, 3);
    Mat_<double> A = Mat_<double>(cameraMatrix), Ar;

    if( !newCameraMatrix.empty() )
        Ar = Mat_<double>(newCameraMatrix);
    else
        Ar = getDefaultNewCameraMatrix( A, size, true );

    if( !matR.empty() )
        R = Mat_<double>(matR);

    if( !distCoeffs.empty() )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(12, 1, CV_64F);
        distCoeffs = 0.;
    }

    CV_Assert( A.size() == Size(3,3) && A.size() == R.size() );
    CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));
    Mat_<double> iR = (Ar.colRange(0,3)*R).inv(DECOMP_LU);
    const double* ir = &iR(0,0);

    double u0 = A(0, 2),  v0 = A(1, 2);
    double fx = A(0, 0),  fy = A(1, 1);

    CV_Assert( distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
               distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
               distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1) ||
               distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1));

    if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() )
        distCoeffs = distCoeffs.t();

    const double* const distPtr = distCoeffs.ptr<double>();
    double k1 = distPtr[0];
    double k2 = distPtr[1];
    double p1 = distPtr[2];
    double p2 = distPtr[3];
    double k3 = distCoeffs.cols + distCoeffs.rows - 1 >= 5 ? distPtr[4] : 0.;
    double k4 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? distPtr[5] : 0.;
    double k5 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? distPtr[6] : 0.;
    double k6 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? distPtr[7] : 0.;
    double s1 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? distPtr[8] : 0.;
    double s2 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? distPtr[9] : 0.;
    double s3 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? distPtr[10] : 0.;
    double s4 = distCoeffs.cols + distCoeffs.rows - 1 >= 12 ? distPtr[11] : 0.;

    for( int i = 0; i < size.height; i++ )
    {
        float* m1f = map1.ptr<float>(i);
        float* m2f = map2.ptr<float>(i);
        short* m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;
        double _x = i*ir[1] + ir[2], _y = i*ir[4] + ir[5], _w = i*ir[7] + ir[8];

        for( int j = 0; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6] )
        {
            double w = 1./_w, x = _x*w, y = _y*w;
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2, _2xy = 2*x*y;
            double kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2);
            double u = fx*(x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+s2*r2*r2) + u0;
            double v = fy*(y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+s4*r2*r2) + v0;
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


void cv::undistort( InputArray _src, OutputArray _dst, InputArray _cameraMatrix,
                    InputArray _distCoeffs, InputArray _newCameraMatrix )
{
    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    CV_Assert( dst.data != src.data );

    int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
    Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

    Mat_<double> A, Ar, I = Mat_<double>::eye(3,3);

    cameraMatrix.convertTo(A, CV_64F);
    if( !distCoeffs.empty() )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(5, 1, CV_64F);
        distCoeffs = 0.;
    }

    if( !newCameraMatrix.empty() )
        newCameraMatrix.convertTo(Ar, CV_64F);
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
    double A[3][3], RR[3][3], k[12]={0,0,0,0,0,0,0,0,0,0,0}, fx, fy, ifx, ify, cx, cy;
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
             _distCoeffs->rows*_distCoeffs->cols == 5 ||
             _distCoeffs->rows*_distCoeffs->cols == 8 ||
             _distCoeffs->rows*_distCoeffs->cols == 12));

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
            double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
            double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
            double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
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


void cv::undistortPoints( InputArray _src, OutputArray _dst,
                          InputArray _cameraMatrix,
                          InputArray _distCoeffs,
                          InputArray _Rmat,
                          InputArray _Pmat )
{
    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), R = _Rmat.getMat(), P = _Pmat.getMat();

    CV_Assert( src.isContinuous() && (src.depth() == CV_32F || src.depth() == CV_64F) &&
              ((src.rows == 1 && src.channels() == 2) || src.cols*src.channels() == 2));

    _dst.create(src.size(), src.type(), -1, true);
    Mat dst = _dst.getMat();

    CvMat _csrc = src, _cdst = dst, _ccameraMatrix = cameraMatrix;
    CvMat matR, matP, _cdistCoeffs, *pR=0, *pP=0, *pD=0;
    if( !R.empty() )
        pR = &(matR = R);
    if( !P.empty() )
        pP = &(matP = P);
    if( !distCoeffs.empty() )
        pD = &(_cdistCoeffs = distCoeffs);
    cvUndistortPoints(&_csrc, &_cdst, &_ccameraMatrix, pD, pR, pP);
}

namespace cv
{

static Point2f mapPointSpherical(const Point2f& p, float alpha, Vec4d* J, int projType)
{
    double x = p.x, y = p.y;
    double beta = 1 + 2*alpha;
    double v = x*x + y*y + 1, iv = 1/v;
    double u = std::sqrt(beta*v + alpha*alpha);

    double k = (u - alpha)*iv;
    double kv = (v*beta/u - (u - alpha)*2)*iv*iv;
    double kx = kv*x, ky = kv*y;

    if( projType == PROJ_SPHERICAL_ORTHO )
    {
        if(J)
            *J = Vec4d(kx*x + k, kx*y, ky*x, ky*y + k);
        return Point2f((float)(x*k), (float)(y*k));
    }
    if( projType == PROJ_SPHERICAL_EQRECT )
    {
        // equirectangular
        double iR = 1/(alpha + 1);
        double x1 = std::max(std::min(x*k*iR, 1.), -1.);
        double y1 = std::max(std::min(y*k*iR, 1.), -1.);

        if(J)
        {
            double fx1 = iR/std::sqrt(1 - x1*x1);
            double fy1 = iR/std::sqrt(1 - y1*y1);
            *J = Vec4d(fx1*(kx*x + k), fx1*ky*x, fy1*kx*y, fy1*(ky*y + k));
        }
        return Point2f((float)asin(x1), (float)asin(y1));
    }
    CV_Error(CV_StsBadArg, "Unknown projection type");
    return Point2f();
}


static Point2f invMapPointSpherical(Point2f _p, float alpha, int projType)
{
    static int avgiter = 0, avgn = 0;

    double eps = 1e-12;
    Vec2d p(_p.x, _p.y), q(_p.x, _p.y), err;
    Vec4d J;
    int i, maxiter = 5;

    for( i = 0; i < maxiter; i++ )
    {
        Point2f p1 = mapPointSpherical(Point2f((float)q[0], (float)q[1]), alpha, &J, projType);
        err = Vec2d(p1.x, p1.y) - p;
        if( err[0]*err[0] + err[1]*err[1] < eps )
            break;

        Vec4d JtJ(J[0]*J[0] + J[2]*J[2], J[0]*J[1] + J[2]*J[3],
                  J[0]*J[1] + J[2]*J[3], J[1]*J[1] + J[3]*J[3]);
        double d = JtJ[0]*JtJ[3] - JtJ[1]*JtJ[2];
        d = d ? 1./d : 0;
        Vec4d iJtJ(JtJ[3]*d, -JtJ[1]*d, -JtJ[2]*d, JtJ[0]*d);
        Vec2d JtErr(J[0]*err[0] + J[2]*err[1], J[1]*err[0] + J[3]*err[1]);

        q -= Vec2d(iJtJ[0]*JtErr[0] + iJtJ[1]*JtErr[1], iJtJ[2]*JtErr[0] + iJtJ[3]*JtErr[1]);
        //Matx22d J(kx*x + k, kx*y, ky*x, ky*y + k);
        //q -= Vec2d((J.t()*J).inv()*(J.t()*err));
    }

    if( i < maxiter )
    {
        avgiter += i;
        avgn++;
        if( avgn == 1500 )
            printf("avg iters = %g\n", (double)avgiter/avgn);
    }

    return i < maxiter ? Point2f((float)q[0], (float)q[1]) : Point2f(-FLT_MAX, -FLT_MAX);
}

}

float cv::initWideAngleProjMap( InputArray _cameraMatrix0, InputArray _distCoeffs0,
                            Size imageSize, int destImageWidth, int m1type,
                            OutputArray _map1, OutputArray _map2, int projType, double _alpha )
{
    Mat cameraMatrix0 = _cameraMatrix0.getMat(), distCoeffs0 = _distCoeffs0.getMat();
    double k[12] = {0,0,0,0,0,0,0,0,0,0,0}, M[9]={0,0,0,0,0,0,0,0,0};
    Mat distCoeffs(distCoeffs0.rows, distCoeffs0.cols, CV_MAKETYPE(CV_64F,distCoeffs0.channels()), k);
    Mat cameraMatrix(3,3,CV_64F,M);
    Point2f scenter((float)cameraMatrix.at<double>(0,2), (float)cameraMatrix.at<double>(1,2));
    Point2f dcenter((destImageWidth-1)*0.5f, 0.f);
    float xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX;
    int N = 9;
    std::vector<Point2f> uvec(1), vvec(1);
    Mat I = Mat::eye(3,3,CV_64F);
    float alpha = (float)_alpha;

    int ndcoeffs = distCoeffs0.cols*distCoeffs0.rows*distCoeffs0.channels();
    CV_Assert((distCoeffs0.cols == 1 || distCoeffs0.rows == 1) &&
              (ndcoeffs == 4 || ndcoeffs == 5 || ndcoeffs == 8));
    CV_Assert(cameraMatrix0.size() == Size(3,3));
    distCoeffs0.convertTo(distCoeffs,CV_64F);
    cameraMatrix0.convertTo(cameraMatrix,CV_64F);

    alpha = std::min(alpha, 0.999f);

    for( int i = 0; i < N; i++ )
        for( int j = 0; j < N; j++ )
        {
            Point2f p((float)j*imageSize.width/(N-1), (float)i*imageSize.height/(N-1));
            uvec[0] = p;
            undistortPoints(uvec, vvec, cameraMatrix, distCoeffs, I, I);
            Point2f q = mapPointSpherical(vvec[0], alpha, 0, projType);
            if( xmin > q.x ) xmin = q.x;
            if( xmax < q.x ) xmax = q.x;
            if( ymin > q.y ) ymin = q.y;
            if( ymax < q.y ) ymax = q.y;
        }

    float scale = (float)std::min(dcenter.x/fabs(xmax), dcenter.x/fabs(xmin));
    Size dsize(destImageWidth, cvCeil(std::max(scale*fabs(ymin)*2, scale*fabs(ymax)*2)));
    dcenter.y = (dsize.height - 1)*0.5f;

    Mat mapxy(dsize, CV_32FC2);
    double k1 = k[0], k2 = k[1], k3 = k[2], p1 = k[3], p2 = k[4], k4 = k[5], k5 = k[6], k6 = k[7], s1 = k[8], s2 = k[9], s3 = k[10], s4 = k[11];
    double fx = cameraMatrix.at<double>(0,0), fy = cameraMatrix.at<double>(1,1), cx = scenter.x, cy = scenter.y;

    for( int y = 0; y < dsize.height; y++ )
    {
        Point2f* mxy = mapxy.ptr<Point2f>(y);
        for( int x = 0; x < dsize.width; x++ )
        {
            Point2f p = (Point2f((float)x, (float)y) - dcenter)*(1.f/scale);
            Point2f q = invMapPointSpherical(p, alpha, projType);
            if( q.x <= -FLT_MAX && q.y <= -FLT_MAX )
            {
                mxy[x] = Point2f(-1.f, -1.f);
                continue;
            }
            double x2 = q.x*q.x, y2 = q.y*q.y;
            double r2 = x2 + y2, _2xy = 2*q.x*q.y;
            double kr = 1 + ((k3*r2 + k2)*r2 + k1)*r2/(1 + ((k6*r2 + k5)*r2 + k4)*r2);
            double u = fx*(q.x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+ s2*r2*r2) + cx;
            double v = fy*(q.y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+ s4*r2*r2) + cy;

            mxy[x] = Point2f((float)u, (float)v);
        }
    }

    if(m1type == CV_32FC2)
    {
        _map1.create(mapxy.size(), mapxy.type());
        Mat map1 = _map1.getMat();
        mapxy.copyTo(map1);
        _map2.release();
    }
    else
        convertMaps(mapxy, Mat(), _map1, _map2, m1type, false);

    return scale;
}

/*  End of file  */
