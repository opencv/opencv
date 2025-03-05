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

#include "opencv2/core/types.hpp"
#include "precomp.hpp"
#include "distortion_model.hpp"

#include "undistort.simd.hpp"
#include "undistort.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

Mat getDefaultNewCameraMatrix( InputArray _cameraMatrix, Size imgsize,
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

static Ptr<ParallelLoopBody> getInitUndistortRectifyMapComputer(Size _size, Mat &_map1, Mat &_map2, int _m1type,
                                                         const double* _ir, Matx33d &_matTilt,
                                                         double _u0, double _v0, double _fx, double _fy,
                                                         double _k1, double _k2, double _p1, double _p2,
                                                         double _k3, double _k4, double _k5, double _k6,
                                                         double _s1, double _s2, double _s3, double _s4)
{
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(getInitUndistortRectifyMapComputer, (_size, _map1, _map2, _m1type, _ir, _matTilt, _u0, _v0, _fx, _fy, _k1, _k2, _p1, _p2, _k3, _k4, _k5, _k6, _s1, _s2, _s3, _s4),
        CV_CPU_DISPATCH_MODES_ALL);
}

void initUndistortRectifyMap( InputArray _cameraMatrix, InputArray _distCoeffs,
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
        distCoeffs.create(14, 1, CV_64F);
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
               distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1) ||
               distCoeffs.size() == Size(1, 14) || distCoeffs.size() == Size(14, 1));

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
    double tauX = distCoeffs.cols + distCoeffs.rows - 1 >= 14 ? distPtr[12] : 0.;
    double tauY = distCoeffs.cols + distCoeffs.rows - 1 >= 14 ? distPtr[13] : 0.;

    // Matrix for trapezoidal distortion of tilted image sensor
    Matx33d matTilt = Matx33d::eye();
    computeTiltProjectionMatrix(tauX, tauY, &matTilt);

    parallel_for_(Range(0, size.height), *getInitUndistortRectifyMapComputer(
        size, map1, map2, m1type, ir, matTilt, u0, v0,
        fx, fy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4));
}

void initInverseRectificationMap( InputArray _cameraMatrix, InputArray _distCoeffs,
                              InputArray _matR, InputArray _newCameraMatrix,
                              const Size& size, int m1type, OutputArray _map1, OutputArray _map2 )
{
    // Parameters
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();
    Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    // Check m1type validity
    if( m1type <= 0 )
        m1type = CV_16SC2;
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );

    // Init Maps
    _map1.create( size, m1type );
    Mat map1 = _map1.getMat(), map2;
    if( m1type != CV_32FC2 )
    {
        _map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        map2 = _map2.getMat();
    }
    else {
        _map2.release();
    }

    // Init camera intrinsics
    Mat_<double> A = Mat_<double>(cameraMatrix), Ar;
    if( !newCameraMatrix.empty() )
        Ar = Mat_<double>(newCameraMatrix);
    else
        Ar = getDefaultNewCameraMatrix( A, size, true );
    CV_Assert( A.size() == Size(3,3) );
    CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));

    // Init rotation matrix
    Mat_<double> R = Mat_<double>::eye(3, 3);
    if( !matR.empty() )
    {
        R = Mat_<double>(matR);
        //Note, do not inverse
    }
    CV_Assert( Size(3,3) == R.size() );

    // Init distortion vector
    if( !distCoeffs.empty() ){
        distCoeffs = Mat_<double>(distCoeffs);

        // Fix distortion vector orientation
        if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() ) {
            distCoeffs = distCoeffs.t();
        }
    }

    // Validate distortion vector size
    CV_Assert(  distCoeffs.empty() || // Empty allows cv::undistortPoints to skip distortion
                distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
                distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
                distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1) ||
                distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1) ||
                distCoeffs.size() == Size(1, 14) || distCoeffs.size() == Size(14, 1));

    // Create objectPoints
    std::vector<cv::Point2i> p2i_objPoints;
    std::vector<cv::Point2f> p2f_objPoints;
    for (int r = 0; r < size.height; r++)
    {
        for (int c = 0; c < size.width; c++)
        {
            p2i_objPoints.push_back(cv::Point2i(c, r));
            p2f_objPoints.push_back(cv::Point2f(static_cast<float>(c), static_cast<float>(r)));
        }
    }

    // Undistort
    std::vector<cv::Point2f> p2f_objPoints_undistorted;
    undistortPoints(
        p2f_objPoints,
        p2f_objPoints_undistorted,
        A,
        distCoeffs,
        cv::Mat::eye(cv::Size(3, 3), CV_64FC1), // R
        cv::Mat::eye(cv::Size(3, 3), CV_64FC1) // P = New K
    );

    // Rectify
    std::vector<cv::Point2f> p2f_sourcePoints_pinHole;
    perspectiveTransform(
        p2f_objPoints_undistorted,
        p2f_sourcePoints_pinHole,
        R
    );

    // Project points back to camera coordinates.
    std::vector<cv::Point2f> p2f_sourcePoints;
    undistortPoints(
        p2f_sourcePoints_pinHole,
        p2f_sourcePoints,
        cv::Mat::eye(cv::Size(3, 3), CV_32FC1), // K
        cv::Mat::zeros(cv::Size(1, 4), CV_32FC1), // Distortion
        cv::Mat::eye(cv::Size(3, 3), CV_32FC1), // R
        Ar // New K
    );

    // Copy to map
    if (m1type == CV_16SC2) {
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<Vec2s>(p2i_objPoints[i].y, p2i_objPoints[i].x) = Vec2s(saturate_cast<short>(p2f_sourcePoints[i].x), saturate_cast<short>(p2f_sourcePoints[i].y));
        }
    } else if (m1type == CV_32FC2) {
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<Vec2f>(p2i_objPoints[i].y, p2i_objPoints[i].x) = Vec2f(p2f_sourcePoints[i]);
        }
    } else { // m1type == CV_32FC1
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<float>(p2i_objPoints[i].y, p2i_objPoints[i].x) = p2f_sourcePoints[i].x;
            map2.at<float>(p2i_objPoints[i].y, p2i_objPoints[i].x) = p2f_sourcePoints[i].y;
        }
    }
}

void undistort( InputArray _src, OutputArray _dst, InputArray _cameraMatrix,
                InputArray _distCoeffs, InputArray _newCameraMatrix )
{
    CV_INSTRUMENT_REGION();

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

static void undistortPointsInternal( const Mat& _src, Mat& _dst, const Mat& _cameraMatrix,
                   const Mat& _distCoeffs, const Mat& matR, const Mat& matP, TermCriteria criteria)
{
    CV_Assert(criteria.isValid());
    double A[3][3], RR[3][3], k[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    Mat matA(3, 3, CV_64F, A), _Dk;
    Mat _RR(3, 3, CV_64F, RR);
    cv::Matx33d invMatTilt = cv::Matx33d::eye();
    cv::Matx33d matTilt = cv::Matx33d::eye();
    bool haveDistCoeffs = !_distCoeffs.empty();

    CV_Assert( (_src.rows == 1 || _src.cols == 1) &&
        (_dst.rows == 1 || _dst.cols == 1) &&
        _src.cols + _src.rows - 1 == _dst.rows + _dst.cols - 1 &&
        (_src.type() == CV_32FC2 || _src.type() == CV_64FC2) &&
        (_dst.type() == CV_32FC2 || _dst.type() == CV_64FC2));

    CV_Assert( _cameraMatrix.rows == 3 && _cameraMatrix.cols == 3 && _cameraMatrix.channels() == 1 );
    _cameraMatrix.convertTo(matA, CV_64F);

    if( haveDistCoeffs )
    {
        CV_Assert(
            (_distCoeffs.rows == 1 || _distCoeffs.cols == 1) &&
            (_distCoeffs.rows*_distCoeffs.cols == 4 ||
             _distCoeffs.rows*_distCoeffs.cols == 5 ||
             _distCoeffs.rows*_distCoeffs.cols == 8 ||
             _distCoeffs.rows*_distCoeffs.cols == 12 ||
             _distCoeffs.rows*_distCoeffs.cols == 14));

        _Dk = Mat( _distCoeffs.rows, _distCoeffs.cols,
            CV_MAKETYPE(CV_64F,_distCoeffs.channels()), k);
        _distCoeffs.convertTo(_Dk, CV_64F);
        CV_Assert(_Dk.ptr<double>() == k);
        if (k[12] != 0 || k[13] != 0)
        {
            computeTiltProjectionMatrix<double>(k[12], k[13], NULL, NULL, NULL, &invMatTilt);
            computeTiltProjectionMatrix<double>(k[12], k[13], &matTilt, NULL, NULL);
        }
    }

    if( !matR.empty() )
    {
        CV_Assert( matR.rows == 3 && matR.cols == 3 && matR.channels() == 1 );
        matR.convertTo(_RR, CV_64F);
        CV_Assert(_RR.ptr<double>() == &RR[0][0]);
    }
    else
        setIdentity(_RR);

    if( !matP.empty() )
    {
        double PP[3][3];
        Mat _PP(3, 3, CV_64F, PP);
        CV_Assert( matP.rows == 3 && (matP.cols == 3 || matP.cols == 4));
        matP.colRange(0, 3).convertTo(_PP, CV_64F);
        CV_Assert(_PP.ptr<double>() == &PP[0][0]);
        _RR = _PP*_RR;
    }

    const Point2f* srcf = (const Point2f*)_src.data;
    const Point2d* srcd = (const Point2d*)_src.data;
    Point2f* dstf = (Point2f*)_dst.data;
    Point2d* dstd = (Point2d*)_dst.data;
    int stype = _src.type();
    int dtype = _dst.type();
    int sstep = _src.rows == 1 ? 1 : (int)(_src.step/_src.elemSize());
    int dstep = _dst.rows == 1 ? 1 : (int)(_dst.step/_dst.elemSize());

    double fx = A[0][0];
    double fy = A[1][1];
    double ifx = 1./fx;
    double ify = 1./fy;
    double cx = A[0][2];
    double cy = A[1][2];

    int n = _src.rows + _src.cols - 1;
    for( int i = 0; i < n; i++ )
    {
        double x, y, x0 = 0, y0 = 0, u, v;
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
        u = x; v = y;
        x = (x - cx)*ifx;
        y = (y - cy)*ify;

        if( haveDistCoeffs ) {
            // compensate tilt distortion
            cv::Vec3d vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);
            double invProj = vecUntilt(2) ? 1./vecUntilt(2) : 1;
            x0 = x = invProj * vecUntilt(0);
            y0 = y = invProj * vecUntilt(1);

            double error = std::numeric_limits<double>::max();
            // compensate distortion iteratively

            for( int j = 0; ; j++ )
            {
                if ((criteria.type & TermCriteria::COUNT) && j >= criteria.maxCount)
                    break;
                if ((criteria.type & TermCriteria::EPS) && error < criteria.epsilon)
                    break;
                double r2 = x*x + y*y;
                double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
                if (icdist < 0)  // test: undistortPoints.regression_14583
                {
                    x = (u - cx)*ifx;
                    y = (v - cy)*ify;
                    break;
                }
                double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
                double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
                x = (x0 - deltaX)*icdist;
                y = (y0 - deltaY)*icdist;

                if(criteria.type & TermCriteria::EPS)
                {
                    double r4, r6, a1, a2, a3, cdist, icdist2;
                    double xd, yd, xd0, yd0;
                    Vec3d vecTilt;

                    r2 = x*x + y*y;
                    r4 = r2*r2;
                    r6 = r4*r2;
                    a1 = 2*x*y;
                    a2 = r2 + 2*x*x;
                    a3 = r2 + 2*y*y;
                    cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
                    icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
                    xd0 = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
                    yd0 = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

                    vecTilt = matTilt*cv::Vec3d(xd0, yd0, 1);
                    invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
                    xd = invProj * vecTilt(0);
                    yd = invProj * vecTilt(1);

                    double x_proj = xd*fx + cx;
                    double y_proj = yd*fy + cy;

                    error = sqrt( pow(x_proj - u, 2) + pow(y_proj - v, 2) );
                }
            }
        }

        if( !matR.empty() || !matP.empty() )
        {
            double xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
            double yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
            double ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
            x = xx*ww;
            y = yy*ww;
        }

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

void undistortPoints(InputArray _src, OutputArray _dst,
                     InputArray _cameraMatrix,
                     InputArray _distCoeffs,
                     InputArray _Rmat,
                     InputArray _Pmat,
                     TermCriteria criteria)
{
    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), R = _Rmat.getMat(), P = _Pmat.getMat();

    int npoints = src.checkVector(2), depth = src.depth();
    if (npoints < 0)
        src = src.t();
    npoints = src.checkVector(2);
    CV_Assert(npoints >= 0 && src.isContinuous() && (depth == CV_32F || depth == CV_64F));

    if (src.cols == 2)
        src = src.reshape(2);

    _dst.create(npoints, 1, CV_MAKETYPE(depth, 2), -1, true);
    Mat dst = _dst.getMat();

    undistortPointsInternal(src, dst, cameraMatrix, distCoeffs, R, P, criteria);
}

void undistortImagePoints(InputArray src, OutputArray dst, InputArray cameraMatrix, InputArray distCoeffs, TermCriteria termCriteria)
{
    undistortPoints(src, dst, cameraMatrix, distCoeffs, noArray(), cameraMatrix, termCriteria);
}

static Point2f mapPointSpherical(const Point2f& p, float alpha, Vec4d* J, enum UndistortTypes projType)
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
        return Point2f((float)std::asin(x1), (float)std::asin(y1));
    }
    CV_Error(Error::StsBadArg, "Unknown projection type");
}


static Point2f invMapPointSpherical(Point2f _p, float alpha, enum UndistortTypes projType)
{
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

    return i < maxiter ? Point2f((float)q[0], (float)q[1]) : Point2f(-FLT_MAX, -FLT_MAX);
}

float initWideAngleProjMap(InputArray _cameraMatrix0, InputArray _distCoeffs0,
                           Size imageSize, int destImageWidth, int m1type,
                           OutputArray _map1, OutputArray _map2,
                           enum UndistortTypes projType, double _alpha)
{
    Mat cameraMatrix0 = _cameraMatrix0.getMat(), distCoeffs0 = _distCoeffs0.getMat();
    double k[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0}, M[9]={0,0,0,0,0,0,0,0,0};
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
              (ndcoeffs == 4 || ndcoeffs == 5 || ndcoeffs == 8 || ndcoeffs == 12 || ndcoeffs == 14));
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
    Matx33d matTilt;
    computeTiltProjectionMatrix(k[12], k[13], &matTilt);

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
            double xd = (q.x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+ s2*r2*r2);
            double yd = (q.y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+ s4*r2*r2);
            cv::Vec3d vecTilt = matTilt*cv::Vec3d(xd, yd, 1);
            double invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
            double u = fx*invProj*vecTilt(0) + cx;
            double v = fy*invProj*vecTilt(1) + cy;

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

}
/*  End of file  */
