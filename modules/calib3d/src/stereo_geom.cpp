// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

void reprojectImageTo3D( InputArray _disparity, OutputArray __3dImage,
                         InputArray _Qmat, bool handleMissingValues, int dtype )
{
    CV_INSTRUMENT_REGION();

    Mat disparity = _disparity.getMat(), Q = _Qmat.getMat();
    int stype = disparity.type();

    CV_Assert( stype == CV_8UC1 || stype == CV_16SC1 ||
               stype == CV_32SC1 || stype == CV_32FC1 );
    CV_Assert( Q.size() == Size(4,4) );

    if( dtype >= 0 )
        dtype = CV_MAKETYPE(CV_MAT_DEPTH(dtype), 3);

    if( __3dImage.fixedType() )
    {
        int dtype_ = __3dImage.type();
        CV_Assert( dtype == -1 || dtype == dtype_ );
        dtype = dtype_;
    }

    if( dtype < 0 )
        dtype = CV_32FC3;
    else
        CV_Assert( dtype == CV_16SC3 || dtype == CV_32SC3 || dtype == CV_32FC3 );

    __3dImage.create(disparity.size(), dtype);
    Mat _3dImage = __3dImage.getMat();

    const float bigZ = 10000.f;
    Matx44d _Q;
    Q.convertTo(_Q, CV_64F);

    int x, cols = disparity.cols;
    CV_Assert( cols >= 0 );

    std::vector<float> _sbuf(cols);
    std::vector<Vec3f> _dbuf(cols);
    float* sbuf = &_sbuf[0];
    Vec3f* dbuf = &_dbuf[0];
    double minDisparity = FLT_MAX;

    // NOTE: here we quietly assume that at least one pixel in the disparity map is not defined.
    // and we set the corresponding Z's to some fixed big value.
    if( handleMissingValues )
        cv::minMaxIdx( disparity, &minDisparity, 0, 0, 0 );

    for( int y = 0; y < disparity.rows; y++ )
    {
        float* sptr = sbuf;
        Vec3f* dptr = dbuf;

        if( stype == CV_8UC1 )
        {
            const uchar* sptr0 = disparity.ptr<uchar>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else if( stype == CV_16SC1 )
        {
            const short* sptr0 = disparity.ptr<short>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else if( stype == CV_32SC1 )
        {
            const int* sptr0 = disparity.ptr<int>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else
            sptr = disparity.ptr<float>(y);

        if( dtype == CV_32FC3 )
            dptr = _3dImage.ptr<Vec3f>(y);

        for( x = 0; x < cols; x++)
        {
            double d = sptr[x];
            Vec4d homg_pt = _Q*Vec4d(x, y, d, 1.0);
            dptr[x] = Vec3d(homg_pt.val);
            dptr[x] /= homg_pt[3];

            if( fabs(d-minDisparity) <= FLT_EPSILON )
                dptr[x][2] = bigZ;
        }

        if( dtype == CV_16SC3 )
        {
            Vec3s* dptr0 = _3dImage.ptr<Vec3s>(y);
            for( x = 0; x < cols; x++ )
            {
                dptr0[x] = dptr[x];
            }
        }
        else if( dtype == CV_32SC3 )
        {
            Vec3i* dptr0 = _3dImage.ptr<Vec3i>(y);
            for( x = 0; x < cols; x++ )
            {
                dptr0[x] = dptr[x];
            }
        }
    }
}

void stereoRectify( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                    InputArray _cameraMatrix2, InputArray _distCoeffs2,
                    Size imageSize, InputArray R, InputArray T,
                    OutputArray _R1, OutputArray _R2,
                    OutputArray _P1, OutputArray _P2,
                    OutputArray _Qmat, int flags,
                    double alpha, Size newImgSize,
                    Rect* roi1, Rect* roi2 )
{
    Mat matR = Mat_<double>(R.getMat()), matT = Mat_<double>(T.getMat());

    Mat om, r_r;
    Mat Z = Mat::zeros(3, 1, CV_64F);
    double nx = imageSize.width, ny = imageSize.height;

    if( matR.rows == 3 && matR.cols == 3 )
        Rodrigues(matR, om);          // get vector rotation
    else
        matR.copyTo(om);
    om *= -0.5; // get average rotation
    Rodrigues(om, r_r);
    Mat t = r_r * matT; // rotate cameras to same orientation by averaging

    int idx = fabs(t.at<double>(0)) > fabs(t.at<double>(1)) ? 0 : 1;
    double c = t.at<double>(idx), nt = norm(t, NORM_L2);
    double _uu[3]={0, 0, 0};
    _uu[idx] = c > 0 ? 1 : -1;

    CV_Assert(nt > 0.0);

    // calculate global Z rotation
    Mat ww = t.cross(Mat(3, 1, CV_64F, _uu)), wR;
    double nw = norm(ww, NORM_L2);
    if (nw > 0.0)
        ww *= std::acos(fabs(c)/nt)/nw;
    Rodrigues(ww, wR);

    Mat Ri;
    // apply to both views
    gemm(wR, r_r, 1, Mat(), 0, Ri, GEMM_2_T);
    Ri.copyTo(_R1);
    gemm(wR, r_r, 1, Mat(), 0, Ri, 0);
    Ri.copyTo(_R2);
    t = Ri * matT;

    // calculate projection/camera matrices
    // these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    Point2d cc_new[2]={};

    newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imageSize;
    const double ratio_x = (double)newImgSize.width / imageSize.width / 2;
    const double ratio_y = (double)newImgSize.height / imageSize.height / 2;
    const double ratio = idx == 1 ? ratio_x : ratio_y;

    Mat cameraMatrix1 = Mat_<double>(_cameraMatrix1.getMat());
    Mat cameraMatrix2 = Mat_<double>(_cameraMatrix2.getMat());
    Mat distCoeffs1, distCoeffs2;
    if (!_distCoeffs1.empty())
        distCoeffs1 = Mat_<double>(_distCoeffs1.getMat());
    if (!_distCoeffs2.empty())
        distCoeffs2 = Mat_<double>(_distCoeffs2.getMat());

    double fc_new = (cameraMatrix1.at<double>(idx ^ 1, idx ^ 1) + cameraMatrix2.at<double>(idx ^ 1, idx ^ 1)) * ratio;

    for( int k = 0; k < 2; k++ )
    {
        const Mat& A = k == 0 ? cameraMatrix1 : cameraMatrix2;
        const Mat& Dk = k == 0 ? distCoeffs1 : distCoeffs2;
        Point2f _pts[4] = {};
        Point3f _pts_3[4] = {};
        Mat pts(1, 4, CV_32FC2, _pts);
        Mat pts_3(1, 4, CV_32FC3, _pts_3);

        for( int i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx-1));
            _pts[i].y = (float)(j*(ny-1));
        }
        undistortPoints(pts, pts, A, Dk, Mat(), Mat());
        convertPointsToHomogeneous(pts, pts_3);

        // Change the camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3] = {{fc_new, 0, 0}, {0, fc_new, 0}, {0, 0, 1}};
        Mat A_tmp(3, 3, CV_64F, _a_tmp);
        projectPoints(pts_3, (k == 0 ? _R1 : _R2), Z, A_tmp, Mat(), pts);
        Scalar avg = mean(pts);
        cc_new[k].x = (nx-1)/2 - avg.val[0];
        cc_new[k].y = (ny-1)/2 - avg.val[1];
    }

    // vertical focal length must be the same for both images to keep the epipolar constraint
    // (for horizontal epipolar lines -- TBD: check for vertical epipolar lines)
    // use fy for fx also, for simplicity

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & CALIB_ZERO_DISPARITY )
    {
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    }
    else if( idx == 0 ) // horizontal stereo
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    else // vertical stereo
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;

    double t_idx = t.at<double>(idx);

    Mat pp = Mat::zeros(3, 4, CV_64F);
    pp.at<double>(0, 0) = pp.at<double>(1, 1) = fc_new;
    pp.at<double>(0, 2) = cc_new[0].x;
    pp.at<double>(1, 2) = cc_new[0].y;
    pp.at<double>(2, 2) = 1.;
    pp.copyTo(_P1);

    pp.at<double>(0, 2) = cc_new[1].x;
    pp.at<double>(1, 2) = cc_new[1].y;
    pp.at<double>(idx, 3) = t_idx*fc_new; // baseline * focal length
    pp.copyTo(_P2);

    alpha = MIN(alpha, 1.);

    cv::Rect_<double> inner1, inner2, outer1, outer2;
    getUndistortRectangles(cameraMatrix1, distCoeffs1, _R1, _P1, imageSize, inner1, outer1);
    getUndistortRectangles(cameraMatrix2, distCoeffs2, _R2, _P2, imageSize, inner2, outer2);

    {
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = cc_new[0].x;
    double cy1_0 = cc_new[0].y;
    double cx2_0 = cc_new[1].x;
    double cy2_0 = cc_new[1].y;
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - 1 - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - 1 - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max((double)cx2/(cx2_0 - inner2.x), (double)cy2/(cy2_0 - inner2.y)),
                         (double)(newImgSize.width - 1 - cx2)/(inner2.x + inner2.width - cx2_0)),
                     (double)(newImgSize.height - 1 - cy2)/(inner2.y + inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - 1 - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - 1 - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - 1 - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - 1 - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1 - alpha) + s1*alpha;
    }

    fc_new *= s;
    cc_new[0] = Point2d(cx1, cy1);
    cc_new[1] = Point2d(cx2, cy2);

    pp.at<double>(0, 0) = pp.at<double>(1, 1) = fc_new;
    pp.at<double>(0, 2) = cx2;
    pp.at<double>(1, 2) = cy2;
    pp.at<double>(idx, 3) *= s;
    pp.copyTo(_P2);

    pp.at<double>(0, 2) = cx1;
    pp.at<double>(1, 2) = cy1;
    pp.at<double>(idx, 3) = 0.;
    pp.copyTo(_P1);

    if(roi1)
    {
        *roi1 =
            cv::Rect(cvCeil((inner1.x - cx1_0)*s + cx1),
                     cvCeil((inner1.y - cy1_0)*s + cy1),
                     cvFloor(inner1.width*s), cvFloor(inner1.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        ;
    }

    if(roi2)
    {
        *roi2 =
            cv::Rect(cvCeil((inner2.x - cx2_0)*s + cx2),
                     cvCeil((inner2.y - cy2_0)*s + cy2),
                     cvFloor(inner2.width*s), cvFloor(inner2.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        ;
    }
    }

    if( _Qmat.needed() )
    {
        double q[] =
        {
            1, 0, 0, -cc_new[0].x,
            0, 1, 0, -cc_new[0].y,
            0, 0, 0, fc_new,
            0, 0, -1./t_idx,
            (idx == 0 ? cc_new[0].x - cc_new[1].x : cc_new[0].y - cc_new[1].y)/t_idx
        };
        Mat Q(4, 4, CV_64F, q);
        Q.copyTo(_Qmat);
    }
}

/*
CV_IMPL int cvStereoRectifyUncalibrated(
    const CvMat* _points1, const CvMat* _points2,
    const CvMat* F0, CvSize imgSize,
    CvMat* _H1, CvMat* _H2, double threshold )
*/
bool stereoRectifyUncalibrated( InputArray _points1, InputArray _points2,
                                InputArray _Fmat, Size imgSize,
                                OutputArray _Hmat1, OutputArray _Hmat2, double threshold )
{
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    CV_Assert( points1.size() == points2.size() );

    int npoints = points1.checkVector(2);
    CV_Assert(npoints > 0);

    Mat _m1, _m2;

    points1.convertTo(_m1, CV_64F);
    points2.convertTo(_m2, CV_64F);
    _m1 = _m1.reshape(2, 1);
    _m2 = _m2.reshape(2, 1);

    Mat F0 = _Fmat.getMat(), F, Wdiag, U, Vt;
    F0.convertTo(F, CV_64F);

    SVDecomp(F, Wdiag, U, Vt, 0);
    Wdiag.at<double>(2) = 0.;
    Mat W = Mat::diag(Wdiag), UW;
    gemm(U, W, 1, Mat(), 0, UW);
    gemm(UW, Vt, 1, Mat(), 0, F);

    double cx = cvRound( (imgSize.width-1)*0.5 );
    double cy = cvRound( (imgSize.height-1)*0.5 );

    if( threshold > 0 )
    {
        Mat _lines1, _lines2;
        computeCorrespondEpilines(_m1, 1, F, _lines1);
        computeCorrespondEpilines(_m2, 2, F, _lines2);
        CV_Assert(_m1.isContinuous() && _m2.isContinuous() &&
                  _lines1.isContinuous() && _lines2.isContinuous());
        Point2d* m1 = (Point2d*)_m1.data;
        Point2d* m2 = (Point2d*)_m2.data;
        Point3d* lines1 = (Point3d*)_lines1.data;
        Point3d* lines2 = (Point3d*)_lines2.data;

        // measure distance from points to the corresponding epilines, mark outliers
        int i, j;
        for( i = j = 0; i < npoints; i++ )
        {
            if( fabs(m1[i].x*lines2[i].x +
                     m1[i].y*lines2[i].y +
                     lines2[i].z) <= threshold &&
                fabs(m2[i].x*lines1[i].x +
                     m2[i].y*lines1[i].y +
                     lines1[i].z) <= threshold )
            {
                if( j < i )
                {
                    m1[j] = m1[i];
                    m2[j] = m2[i];
                }
                j++;
            }
        }

        npoints = j;
        if( npoints == 0 )
            return false;
        _m1.cols = _m2.cols = npoints;
    }

    Mat E2 = U.col(2).clone();
    if (E2.at<double>(2) < 0)
        E2 *= -1.0;

    double t[] =
    {
        1, 0, -cx,
        0, 1, -cy,
        0, 0, 1
    };
    Mat T(3, 3, CV_64F, t);
    E2 = T*E2;

    double* e2 = (double*)E2.data;
    int mirror = e2[0] < 0;
    double d = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1]);
    d = MAX(d, DBL_EPSILON);
    double alpha = e2[0]/d;
    double beta = e2[1]/d;
    double r[] =
    {
        alpha, beta, 0,
        -beta, alpha, 0,
        0, 0, 1
    };
    Mat R(3, 3, CV_64F, r);
    T = R*T;
    E2 = R*E2;
    double invf = fabs(e2[2]) < 1e-6*fabs(e2[0]) ? 0 : -e2[2]/e2[0];
    double k[] =
    {
        1, 0, 0,
        0, 1, 0,
        invf, 0, 1
    };
    Mat K(3, 3, CV_64F, k);
    Mat H2 = K*T;
    E2 = K*E2;

    double it[] =
    {
        1, 0, cx,
        0, 1, cy,
        0, 0, 1
    };
    Mat iT( 3, 3, CV_64F, it );
    H2 = iT*H2;

    U.col(2).copyTo(E2);
    if (E2.at<double>(2) < 0)
        E2 *= -1.0;

    double e2_x[] =
    {
        0, -e2[2], e2[1],
       e2[2], 0, -e2[0],
       -e2[1], e2[0], 0
    };
    double e2_111[] =
    {
        e2[0], e2[0], e2[0],
        e2[1], e2[1], e2[1],
        e2[2], e2[2], e2[2],
    };
    Mat E2_x(3, 3, CV_64F, e2_x);
    Mat E2_111(3, 3, CV_64F, e2_111);
    Mat H0 = E2_x*F + E2_111;
    H0 = H2*H0;
    Mat E1(3, 1, CV_64F, (double*)Vt.data+6);
    E1 = H0*E1;

    perspectiveTransform( _m1, _m1, H0 );
    perspectiveTransform( _m2, _m2, H2 );
    Mat A, X;
    convertPointsToHomogeneous(_m1, A);
    A.convertTo(A, CV_64F);
    A = A.reshape(1, npoints);
    Mat BxBy = _m2.reshape(1, npoints);
    Mat B = BxBy.col(0);
    solve(A, B, X, DECOMP_SVD);
    CV_Assert(X.isContinuous());
    double* x = X.ptr<double>();

    double ha[] =
    {
        x[0], x[1], x[2],
        0, 1, 0,
        0, 0, 1
    };
    Mat Ha(3, 3, CV_64F, ha);
    Mat H1 = Ha*H0;
    perspectiveTransform( _m1, _m1, Ha );

    if( mirror )
    {
        double mm[] = { -1, 0, cx*2, 0, -1, cy*2, 0, 0, 1 };
        Mat MM(3, 3, CV_64F, mm);
        H1 = MM*H1;
        H2 = MM*H2;
    }

    H1.copyTo(_Hmat1);
    H2.copyTo(_Hmat2);
    return true;
}


static void adjust3rdMatrix(InputArrayOfArrays _imgpt1_0,
                            InputArrayOfArrays _imgpt3_0,
                            const Mat& cameraMatrix1, const Mat& distCoeffs1,
                            const Mat& cameraMatrix3, const Mat& distCoeffs3,
                            const Mat& R1, const Mat& R3, const Mat& P1, Mat& P3 )
{
    size_t n1 = _imgpt1_0.total(), n3 = _imgpt3_0.total();
    std::vector<Point2f> imgpt1, imgpt3;

    for( int i = 0; i < (int)std::min(n1, n3); i++ )
    {
        Mat pt1 = _imgpt1_0.getMat(i), pt3 = _imgpt3_0.getMat(i);
        int ni1 = pt1.checkVector(2, CV_32F), ni3 = pt3.checkVector(2, CV_32F);
        CV_Assert( ni1 > 0 && ni1 == ni3 );
        const Point2f* pt1data = pt1.ptr<Point2f>();
        const Point2f* pt3data = pt3.ptr<Point2f>();
        std::copy(pt1data, pt1data + ni1, std::back_inserter(imgpt1));
        std::copy(pt3data, pt3data + ni3, std::back_inserter(imgpt3));
    }

    undistortPoints(imgpt1, imgpt1, cameraMatrix1, distCoeffs1, R1, P1);
    undistortPoints(imgpt3, imgpt3, cameraMatrix3, distCoeffs3, R3, P3);

    double y1_ = 0, y2_ = 0, y1y1_ = 0, y1y2_ = 0;
    size_t n = imgpt1.size();
    CV_DbgAssert(n > 0);

    for( size_t i = 0; i < n; i++ )
    {
        double y1 = imgpt3[i].y, y2 = imgpt1[i].y;

        y1_ += y1; y2_ += y2;
        y1y1_ += y1*y1; y1y2_ += y1*y2;
    }

    y1_ /= n;
    y2_ /= n;
    y1y1_ /= n;
    y1y2_ /= n;

    double a = (y1y2_ - y1_*y2_)/(y1y1_ - y1_*y1_);
    double b = y2_ - a*y1_;

    P3.at<double>(0,0) *= a;
    P3.at<double>(1,1) *= a;
    P3.at<double>(0,2) = P3.at<double>(0,2)*a;
    P3.at<double>(1,2) = P3.at<double>(1,2)*a + b;
    P3.at<double>(0,3) *= a;
    P3.at<double>(1,3) *= a;
}

float rectify3Collinear( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                   InputArray _cameraMatrix2, InputArray _distCoeffs2,
                   InputArray _cameraMatrix3, InputArray _distCoeffs3,
                   InputArrayOfArrays _imgpt1,
                   InputArrayOfArrays _imgpt3,
                   Size imageSize, InputArray _Rmat12, InputArray _Tmat12,
                   InputArray _Rmat13, InputArray _Tmat13,
                   OutputArray _Rmat1, OutputArray _Rmat2, OutputArray _Rmat3,
                   OutputArray _Pmat1, OutputArray _Pmat2, OutputArray _Pmat3,
                   OutputArray _Qmat,
                   double alpha, Size newImgSize,
                   Rect* roi1, Rect* roi2, int flags )
{
    // first, rectify the 1-2 stereo pair
    stereoRectify( _cameraMatrix1, _distCoeffs1, _cameraMatrix2, _distCoeffs2,
                   imageSize, _Rmat12, _Tmat12, _Rmat1, _Rmat2, _Pmat1, _Pmat2, _Qmat,
                   flags, alpha, newImgSize, roi1, roi2 );

    Mat R12 = _Rmat12.getMat(), R13 = _Rmat13.getMat(), T12 = _Tmat12.getMat(), T13 = _Tmat13.getMat();

    _Rmat3.create(3, 3, CV_64F);
    _Pmat3.create(3, 4, CV_64F);

    Mat P1 = _Pmat1.getMat(), P2 = _Pmat2.getMat();
    Mat R3 = _Rmat3.getMat(), P3 = _Pmat3.getMat();

    // recompute rectification transforms for cameras 1 & 2.
    Mat om, r_r, r_r13;

    if( R13.size() != Size(3,3) )
        Rodrigues(R13, r_r13);
    else
        R13.copyTo(r_r13);

    if( R12.size() == Size(3,3) )
        Rodrigues(R12, om);
    else
        R12.copyTo(om);

    om *= -0.5;
    Rodrigues(om, r_r); // rotate cameras to same orientation by averaging
    Mat_<double> t12 = r_r * T12;

    int idx = fabs(t12(0,0)) > fabs(t12(1,0)) ? 0 : 1;
    double c = t12(idx,0), nt = norm(t12, NORM_L2);
    CV_Assert(fabs(nt) > 0);
    Mat_<double> uu = Mat_<double>::zeros(3,1);
    uu(idx, 0) = c > 0 ? 1 : -1;

    // calculate global Z rotation
    Mat_<double> ww = t12.cross(uu), wR;
    double nw = norm(ww, NORM_L2);
    CV_Assert(fabs(nw) > 0);
    ww *= std::acos(fabs(c)/nt)/nw;
    Rodrigues(ww, wR);

    // now rotate camera 3 to make its optical axis parallel to cameras 1 and 2.
    R3 = wR*r_r.t()*r_r13.t();
    Mat_<double> t13 = R3 * T13;

    P2.copyTo(P3);
    Mat t = P3.col(3);
    t13.copyTo(t);
    P3.at<double>(0,3) *= P3.at<double>(0,0);
    P3.at<double>(1,3) *= P3.at<double>(1,1);

    if( !_imgpt1.empty() && !_imgpt3.empty() )
        adjust3rdMatrix(_imgpt1, _imgpt3, _cameraMatrix1.getMat(), _distCoeffs1.getMat(),
                        _cameraMatrix3.getMat(), _distCoeffs3.getMat(), _Rmat1.getMat(), R3, P1, P3);

    return (float)((P3.at<double>(idx,3)/P3.at<double>(idx,idx))/
                   (P2.at<double>(idx,3)/P2.at<double>(idx,idx)));
}

void cv::fisheye::stereoRectify( InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size& imageSize,
        InputArray _R, InputArray _tvec, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2,
        OutputArray Q, int flags, const Size& newImageSize, double balance, double fov_scale)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((_R.size() == Size(3, 3) || _R.total() * _R.channels() == 3) && (_R.depth() == CV_32F || _R.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));


    Mat aaa = _tvec.getMat().reshape(3, 1);

    Vec3d rvec; // Rodrigues vector
    if (_R.size() == Size(3, 3))
    {
        Matx33d rmat;
        _R.getMat().convertTo(rmat, CV_64F);
        rvec = Affine3d(rmat).rvec();
    }
    else if (_R.total() * _R.channels() == 3)
        _R.getMat().convertTo(rvec, CV_64F);

    Vec3d tvec;
    _tvec.getMat().convertTo(tvec, CV_64F);

    // rectification algorithm
    rvec *= -0.5;              // get average rotation

    Matx33d r_r;
    Rodrigues(rvec, r_r);  // rotate cameras to same orientation by averaging

    Vec3d t = r_r * tvec;
    Vec3d uu(t[0] > 0 ? 1 : -1, 0, 0);

    // calculate global Z rotation
    Vec3d ww = t.cross(uu);
    double nw = norm(ww);
    if (nw > 0.0)
        ww *= std::acos(fabs(t[0])/cv::norm(t))/nw;

    Matx33d wr;
    Rodrigues(ww, wr);

    // apply to both views
    Matx33d ri1 = wr * r_r.t();
    Mat(ri1, false).convertTo(R1, R1.empty() ? CV_64F : R1.type());
    Matx33d ri2 = wr * r_r;
    Mat(ri2, false).convertTo(R2, R2.empty() ? CV_64F : R2.type());
    Vec3d tnew = ri2 * tvec;

    // calculate projection/camera matrices. these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    Matx33d newK1, newK2;
    fisheye::estimateNewCameraMatrixForUndistortRectify(K1, D1, imageSize, R1, newK1, balance, newImageSize, fov_scale);
    fisheye::estimateNewCameraMatrixForUndistortRectify(K2, D2, imageSize, R2, newK2, balance, newImageSize, fov_scale);

    double fc_new = std::min(newK1(1,1), newK2(1,1));
    Point2d cc_new[2] = { Vec2d(newK1(0, 2), newK1(1, 2)), Vec2d(newK2(0, 2), newK2(1, 2)) };

    // Vertical focal length must be the same for both images to keep the epipolar constraint use fy for fx also.
    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & CALIB_ZERO_DISPARITY )
        cc_new[0] = cc_new[1] = (cc_new[0] + cc_new[1]) * 0.5;
    else
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;

    Mat(Matx34d(fc_new, 0, cc_new[0].x, 0,
                0, fc_new, cc_new[0].y, 0,
                0,      0,           1, 0), false).convertTo(P1, P1.empty() ? CV_64F : P1.type());

    Mat(Matx34d(fc_new, 0, cc_new[1].x, tnew[0]*fc_new, // baseline * focal length;,
                0, fc_new, cc_new[1].y,              0,
                0,      0,           1,              0), false).convertTo(P2, P2.empty() ? CV_64F : P2.type());

    if (Q.needed())
        Mat(Matx44d(1, 0, 0,           -cc_new[0].x,
                    0, 1, 0,           -cc_new[0].y,
                    0, 0, 0,            fc_new,
                    0, 0, -1./tnew[0], (cc_new[0].x - cc_new[1].x)/tnew[0]), false).convertTo(Q, Q.empty() ? CV_64F : Q.depth());
}

}
