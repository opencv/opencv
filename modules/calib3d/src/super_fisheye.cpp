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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#include "super_fisheye.hpp"
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::projectPoints

void cv::super_fisheye::projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
                                InputArray K, InputArray D, double alpha, OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    projectPoints(objectPoints, imagePoints, affine.rvec(), affine.translation(), K, D, alpha, jacobian);
}

void cv::super_fisheye::projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray _rvec,
                                InputArray _tvec, InputArray _K, InputArray _D, double alpha, OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    // will support only 3-channel data now for points
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
    size_t n = objectPoints.total();

    CV_Assert(_rvec.total() * _rvec.channels() == 3 && (_rvec.depth() == CV_32F || _rvec.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));
    CV_Assert(_tvec.getMat().isContinuous() && _rvec.getMat().isContinuous());

    Vec3d om = _rvec.depth() == CV_32F ? (Vec3d)*_rvec.getMat().ptr<Vec3f>() : *_rvec.getMat().ptr<Vec3d>();
    Vec3d T  = _tvec.depth() == CV_32F ? (Vec3d)*_tvec.getMat().ptr<Vec3f>() : *_tvec.getMat().ptr<Vec3d>();

    CV_Assert(_K.size() == Size(3,3) && (_K.type() == CV_32F || _K.type() == CV_64F) && _D.type() == _K.type() && _D.total() == 4);

    cv::Vec2d f, c;
    if (_K.depth() == CV_32F)
    {

        Matx33f K = _K.getMat();
        f = Vec2f(K(0, 0), K(1, 1));
        c = Vec2f(K(0, 2), K(1, 2));
    }
    else
    {
        Matx33d K = _K.getMat();
        f = Vec2d(K(0, 0), K(1, 1));
        c = Vec2d(K(0, 2), K(1, 2));
    }

    Vec4d k = _D.depth() == CV_32F ? (Vec4d)*_D.getMat().ptr<Vec4f>(): *_D.getMat().ptr<Vec4d>();

    const bool isJacobianNeeded = jacobian.needed();
    JacobianRow *Jn = 0;
    if (isJacobianNeeded)
    {
        int nvars = 2 + 2 + 1 + 4 + 3 + 3; // f, c, alpha, k, om, T,
        jacobian.create(2*(int)n, nvars, CV_64F);
        Jn = jacobian.getMat().ptr<JacobianRow>(0);
    }

    Matx33d R;
    Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);
    Affine3d aff(om, T);

    const Vec3f* Xf = objectPoints.getMat().ptr<Vec3f>();
    const Vec3d* Xd = objectPoints.getMat().ptr<Vec3d>();
    Vec2f *xpf = imagePoints.getMat().ptr<Vec2f>();
    Vec2d *xpd = imagePoints.getMat().ptr<Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        Vec3d Xi = objectPoints.depth() == CV_32F ? (Vec3d)Xf[i] : Xd[i];
        Vec3d Y = aff*Xi;

        if(fabs(Y[2]) < DBL_MIN)
        {
            if(Y[2] > 0)
                Y[2] = DBL_MIN;
            else
                Y[2] = -DBL_MIN;
        }
        Vec2d x(Y[0]/fabs(Y[2]), Y[1]/fabs(Y[2]));

        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        // Angle of the incoming ray:
        double theta = atan2(r, 1);

        if(Y[2] <= 0)
            theta += M_PI/2;

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        Vec2d xd1 = x * cdist;
        Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (objectPoints.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;

        if (isJacobianNeeded)
        {
            //Vec3d Xi = pdepth == CV_32F ? (Vec3d)Xf[i] : Xd[i];
            //Vec3d Y = aff*Xi;
            double dYdR[] = { Xi[0], Xi[1], Xi[2], 0, 0, 0, 0, 0, 0,
                              0, 0, 0, Xi[0], Xi[1], Xi[2], 0, 0, 0,
                              0, 0, 0, 0, 0, 0, Xi[0], Xi[1], Xi[2] };

            Matx33d dYdom_data = Matx<double, 3, 9>(dYdR) * dRdom.t();
            const Vec3d *dYdom = (Vec3d*)dYdom_data.val;

            Matx33d dYdT_data = Matx33d::eye();
            const Vec3d *dYdT = (Vec3d*)dYdT_data.val;

            //Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);
            Vec3d dxdom[2];
            dxdom[0] = (1.0/fabs(Y[2])) * dYdom[0] - x[0]/fabs(Y[2]) * dYdom[2];
            dxdom[1] = (1.0/fabs(Y[2])) * dYdom[1] - x[1]/fabs(Y[2]) * dYdom[2];

            Vec3d dxdT[2];
            dxdT[0]  = (1.0/fabs(Y[2])) * dYdT[0] - x[0]/fabs(Y[2]) * dYdT[2];
            dxdT[1]  = (1.0/fabs(Y[2])) * dYdT[1] - x[1]/fabs(Y[2]) * dYdT[2];

            //double r2 = x.dot(x);
            Vec3d dr2dom = 2 * x[0] * dxdom[0] + 2 * x[1] * dxdom[1];
            Vec3d dr2dT  = 2 * x[0] *  dxdT[0] + 2 * x[1] *  dxdT[1];

            //double r = std::sqrt(r2);
            double drdr2 = r > 1e-8 ? 1.0/(2*r) : 1;
            Vec3d drdom = drdr2 * dr2dom;
            Vec3d drdT  = drdr2 * dr2dT;

            // Angle of the incoming ray:
            //double theta = atan(r);
            double dthetadr = 1.0/(1+r2);
            if(Y[2] <= 0)
                dthetadr *= -1;
            Vec3d dthetadom = dthetadr * drdom;
            Vec3d dthetadT  = dthetadr *  drdT;

            //double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;
            double dtheta_ddtheta = 1 + 3*k[0]*theta2 + 5*k[1]*theta4 + 7*k[2]*theta6 + 9*k[3]*theta8;
            Vec3d dtheta_ddom = dtheta_ddtheta * dthetadom;
            Vec3d dtheta_ddT  = dtheta_ddtheta * dthetadT;
            Vec4d dtheta_ddk  = Vec4d(theta3, theta5, theta7, theta9);

            //double inv_r = r > 1e-8 ? 1.0/r : 1;
            //double cdist = r > 1e-8 ? theta_d / r : 1;
            Vec3d dcdistdom = inv_r * (dtheta_ddom - cdist*drdom);
            Vec3d dcdistdT  = inv_r * (dtheta_ddT  - cdist*drdT);
            Vec4d dcdistdk  = inv_r *  dtheta_ddk;

            //Vec2d xd1 = x * cdist;
            Vec4d dxd1dk[2];
            Vec3d dxd1dom[2], dxd1dT[2];
            dxd1dom[0] = x[0] * dcdistdom + cdist * dxdom[0];
            dxd1dom[1] = x[1] * dcdistdom + cdist * dxdom[1];
            dxd1dT[0]  = x[0] * dcdistdT  + cdist * dxdT[0];
            dxd1dT[1]  = x[1] * dcdistdT  + cdist * dxdT[1];
            dxd1dk[0]  = x[0] * dcdistdk;
            dxd1dk[1]  = x[1] * dcdistdk;

            //Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
            Vec4d dxd3dk[2];
            Vec3d dxd3dom[2], dxd3dT[2];
            dxd3dom[0] = dxd1dom[0] + alpha * dxd1dom[1];
            dxd3dom[1] = dxd1dom[1];
            dxd3dT[0]  = dxd1dT[0]  + alpha * dxd1dT[1];
            dxd3dT[1]  = dxd1dT[1];
            dxd3dk[0]  = dxd1dk[0]  + alpha * dxd1dk[1];
            dxd3dk[1]  = dxd1dk[1];

            Vec2d dxd3dalpha(xd1[1], 0);

            //final jacobian
            Jn[0].dom = f[0] * dxd3dom[0];
            Jn[1].dom = f[1] * dxd3dom[1];

            Jn[0].dT = f[0] * dxd3dT[0];
            Jn[1].dT = f[1] * dxd3dT[1];

            Jn[0].dk = f[0] * dxd3dk[0];
            Jn[1].dk = f[1] * dxd3dk[1];

            Jn[0].dalpha = f[0] * dxd3dalpha[0];
            Jn[1].dalpha = 0; //f[1] * dxd3dalpha[1];

            Jn[0].df = Vec2d(xd3[0], 0);
            Jn[1].df = Vec2d(0, xd3[1]);

            Jn[0].dc = Vec2d(1, 0);
            Jn[1].dc = Vec2d(0, 1);

            //step to jacobian rows for next point
            Jn += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void cv::super_fisheye::undistortPoints( InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray R, InputArray P)
{
    CV_INSTRUMENT_REGION();

    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
    }

    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    cv::Matx33d RR = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    if(!P.empty())
    {
        cv::Matx33d PP;
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
    cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    for(size_t i = 0; i < n; i++ )
    {
        Vec2d pi = sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i];  // image point
        Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double scale = 1.0;

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;

            const double EPS = 1e-8; // or std::numeric_limits<double>::epsilon();
            for (int j = 0; j < 10; j++)
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6, k3_theta8 = k[3] * theta8;
                /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
                double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                   (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
                theta = theta - theta_fix;
                if (fabs(theta_fix) < EPS)
                    break;
            }

            theta = min(theta, CV_PI/2.);

            scale = std::tan(theta) / theta_d;
        }

        Vec2d pu = pw * scale; //undistorted point

        // reproject
        Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::super_fisheye::estimateNewCameraMatrixForUndistortRectify

void cv::super_fisheye::estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
                                                             OutputArray P, double balance, const Size& new_size, double fov_scale)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( K.size() == Size(3, 3)       && (K.depth() == CV_32F || K.depth() == CV_64F));
    CV_Assert(D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || D.depth() == CV_64F)));

    int w = image_size.width, h = image_size.height;
    balance = std::min(std::max(balance, 0.0), 1.0);

    cv::Mat points(1, 4, CV_64FC2);
    Vec2d* pptr = points.ptr<Vec2d>();
    pptr[0] = Vec2d(w/2, 0);
    pptr[1] = Vec2d(w, h/2);
    pptr[2] = Vec2d(w/2, h);
    pptr[3] = Vec2d(0, h/2);

    super_fisheye::undistortPoints(points, points, K, D, R);
    cv::Scalar center_mass = mean(points);
    cv::Vec2d cn(center_mass.val);

    double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0,0)/K.getMat().at<float> (1,1)
                                                : K.getMat().at<double>(0,0)/K.getMat().at<double>(1,1);

    // convert to identity ratio
    cn[0] *= aspect_ratio;
    for(size_t i = 0; i < points.total(); ++i)
        pptr[i][1] *= aspect_ratio;

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    for(size_t i = 0; i < points.total(); ++i)
    {
        miny = std::min(miny, pptr[i][1]);
        maxy = std::max(maxy, pptr[i][1]);
        minx = std::min(minx, pptr[i][0]);
        maxx = std::max(maxx, pptr[i][0]);
    }

    double f1 = w * 0.5/(cn[0] - minx);
    double f2 = w * 0.5/(maxx - cn[0]);
    double f3 = h * 0.5 * aspect_ratio/(cn[1] - miny);
    double f4 = h * 0.5 * aspect_ratio/(maxy - cn[1]);

    double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
    double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

    double f = balance * fmin + (1.0 - balance) * fmax;
    f *= fov_scale > 0 ? 1.0/fov_scale : 1.0;

    cv::Vec2d new_f(f, f), new_c = -cn * f + Vec2d(w, h * aspect_ratio) * 0.5;

    // restore aspect ratio
    new_f[1] /= aspect_ratio;
    new_c[1] /= aspect_ratio;

    if (!new_size.empty())
    {
        double rx = new_size.width /(double)image_size.width;
        double ry = new_size.height/(double)image_size.height;

        new_f[0] *= rx;  new_f[1] *= ry;
        new_c[0] *= rx;  new_c[1] *= ry;
    }

    Mat(Matx33d(new_f[0], 0, new_c[0],
                0, new_f[1], new_c[1],
                0,        0,       1)).convertTo(P, P.empty() ? K.type() : P.type());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::super_fisheye::stereoRectify

void cv::super_fisheye::stereoRectify( InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size& imageSize,
                                 InputArray _R, InputArray _tvec, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags, const Size& newImageSize, double balance, double fov_scale)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((_R.size() == Size(3, 3) || _R.total() * _R.channels() == 3) && (_R.depth() == CV_32F || _R.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));


    cv::Mat aaa = _tvec.getMat().reshape(3, 1);

    Vec3d rvec; // Rodrigues vector
    if (_R.size() == Size(3, 3))
    {
        cv::Matx33d rmat;
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
        ww *= acos(fabs(t[0])/cv::norm(t))/nw;

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
    estimateNewCameraMatrixForUndistortRectify(K1, D1, imageSize, R1, newK1, balance, newImageSize, fov_scale);
    estimateNewCameraMatrixForUndistortRectify(K2, D2, imageSize, R2, newK2, balance, newImageSize, fov_scale);

    double fc_new = std::min(newK1(1,1), newK2(1,1));
    Point2d cc_new[2] = { Vec2d(newK1(0, 2), newK1(1, 2)), Vec2d(newK2(0, 2), newK2(1, 2)) };

    // Vertical focal length must be the same for both images to keep the epipolar constraint use fy for fx also.
    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & cv::CALIB_ZERO_DISPARITY )
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


void cv::foo::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
                                 cv::InputArray _rvec,cv::InputArray _tvec,
                                 const cv::internal::IntrinsicParams& param, cv::OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
    Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0],
              0,               param.f[1], param.c[1],
              0,                        0,         1);
    super_fisheye::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian);
}

cv::Mat cv::foo::NormalizePixels(const Mat& imagePoints, const cv::internal::IntrinsicParams& param)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

    Mat distorted((int)imagePoints.total(), 1, CV_64FC2), undistorted;
    const Vec2d* ptr   = imagePoints.ptr<Vec2d>();
    Vec2d* ptr_d = distorted.ptr<Vec2d>();
    for (size_t i = 0; i < imagePoints.total(); ++i)
    {
        ptr_d[i] = (ptr[i] - param.c).mul(Vec2d(1.0 / param.f[0], 1.0 / param.f[1]));
        ptr_d[i][0] -= param.alpha * ptr_d[i][1];
    }
    cv::super_fisheye::undistortPoints(distorted, undistorted, Matx33d::eye(), param.k);
    return undistorted;
}