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
#include "epnp.h"
#include "p3p.h"
#include "ap3p.h"
#include "ippe.hpp"
#include "sqpnp.hpp"
#include "usac.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv {

using namespace std;

#if !defined(NDEBUG) || defined(CV_STATIC_ANALYSIS)
static bool isPlanarObjectPoints(InputArray _objectPoints, double threshold)
{
    CV_CheckType(_objectPoints.type(), _objectPoints.type() == CV_32FC3 || _objectPoints.type() == CV_64FC3,
                 "Type of _objectPoints must be CV_32FC3 or CV_64FC3");
    Mat objectPoints;
    if (_objectPoints.type() == CV_32FC3)
    {
        _objectPoints.getMat().convertTo(objectPoints, CV_64F);
    }
    else
    {
        objectPoints = _objectPoints.getMat();
    }

    Scalar meanValues = mean(objectPoints);
    int nbPts = objectPoints.checkVector(3, CV_64F);
    Mat objectPointsCentred = objectPoints - meanValues;
    objectPointsCentred = objectPointsCentred.reshape(1, nbPts);

    Mat w, u, vt;
    Mat MM = objectPointsCentred.t() * objectPointsCentred;
    SVDecomp(MM, w, u, vt);

    return (w.at<double>(2) < w.at<double>(1) * threshold);
}

static bool approxEqual(double a, double b, double eps)
{
    return std::fabs(a-b) < eps;
}
#endif

void drawFrameAxes(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                   InputArray rvec, InputArray tvec, float length, int thickness)
{
    CV_INSTRUMENT_REGION();

    int type = image.type();
    int cn = CV_MAT_CN(type);
    CV_CheckType(type, cn == 1 || cn == 3 || cn == 4,
                 "Number of channels must be 1, 3 or 4" );

    CV_Assert(image.getMat().total() > 0);
    CV_Assert(length > 0);

    // project axes points
    std::vector<Point3f> axesPoints;
    axesPoints.push_back(Point3f(0, 0, 0));
    axesPoints.push_back(Point3f(length, 0, 0));
    axesPoints.push_back(Point3f(0, length, 0));
    axesPoints.push_back(Point3f(0, 0, length));
    std::vector<Point2f> imagePoints;
    projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // draw axes lines
    line(image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), thickness);
    line(image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), thickness);
    line(image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), thickness);
}

bool solvePnP( InputArray opoints, InputArray ipoints,
               InputArray cameraMatrix, InputArray distCoeffs,
               OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess, int flags )
{
    CV_INSTRUMENT_REGION();

    std::vector<Mat> rvecs, tvecs;
    int solutions = solvePnPGeneric(opoints, ipoints, cameraMatrix, distCoeffs, rvecs, tvecs, useExtrinsicGuess, (SolvePnPMethod)flags, rvec, tvec);

    if (solutions > 0)
    {
        int rdepth = rvec.empty() ? CV_64F : rvec.depth();
        int tdepth = tvec.empty() ? CV_64F : tvec.depth();
        rvecs[0].convertTo(rvec, rdepth);
        tvecs[0].convertTo(tvec, tdepth);
    }

    return solutions > 0;
}

class PnPRansacCallback CV_FINAL : public PointSetRegistrator::Callback
{

public:

    PnPRansacCallback(Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), int _flags=SOLVEPNP_ITERATIVE,
            bool _useExtrinsicGuess=false, Mat _rvec=Mat(), Mat _tvec=Mat() )
        : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), flags(_flags), useExtrinsicGuess(_useExtrinsicGuess),
          rvec(_rvec), tvec(_tvec) {}

    /* Pre: True */
    /* Post: compute _model with given points and return number of found models */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const CV_OVERRIDE
    {
        Mat opoints = _m1.getMat(), ipoints = _m2.getMat();
        Mat iter_rvec = rvec.clone();
        Mat iter_tvec = tvec.clone();
        bool correspondence = solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                            iter_rvec, iter_tvec, useExtrinsicGuess, flags );

        Mat _local_model;
        hconcat(iter_rvec, iter_tvec, _local_model);
        _local_model.copyTo(_model);

        return correspondence;
    }

    /* Pre: True */
    /* Post: fill _err with projection errors */
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const CV_OVERRIDE
    {

        Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

        int i, count = opoints.checkVector(3);
        Mat _rvec = model.col(0);
        Mat _tvec = model.col(1);


        Mat projpoints(count, 2, CV_32FC1);
        projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

        const Point2f* ipoints_ptr = ipoints.ptr<Point2f>();
        const Point2f* projpoints_ptr = projpoints.ptr<Point2f>();

        _err.create(count, 1, CV_32FC1);
        float* err = _err.getMat().ptr<float>();

        for ( i = 0; i < count; ++i)
            err[i] = (float)norm( Matx21f(ipoints_ptr[i] - projpoints_ptr[i]), NORM_L2SQR );

    }


    Mat cameraMatrix;
    Mat distCoeffs;
    int flags;
    bool useExtrinsicGuess;
    Mat rvec;
    Mat tvec;
};

bool solvePnPRansac(InputArray _opoints, InputArray _ipoints,
                    InputArray _cameraMatrix, InputArray _distCoeffs,
                    OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess,
                    int iterationsCount, float reprojectionError, double confidence,
                    OutputArray _inliers, int flags)
{
    CV_INSTRUMENT_REGION();

    if (flags >= USAC_DEFAULT && flags <= USAC_MAGSAC)
        return usac::solvePnPRansac(_opoints, _ipoints, _cameraMatrix, _distCoeffs,
            _rvec, _tvec, useExtrinsicGuess, iterationsCount, reprojectionError,
            confidence, _inliers, flags);

    Mat opoints0 = _opoints.getMat(), ipoints0 = _ipoints.getMat();
    Mat opoints, ipoints;
    if( opoints0.depth() == CV_64F || !opoints0.isContinuous() )
        opoints0.convertTo(opoints, CV_32F);
    else
        opoints = opoints0;
    if( ipoints0.depth() == CV_64F || !ipoints0.isContinuous() )
        ipoints0.convertTo(ipoints, CV_32F);
    else
        ipoints = ipoints0;

    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints >= 4 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

    CV_Assert(opoints.isContinuous());
    CV_Assert(opoints.depth() == CV_32F || opoints.depth() == CV_64F);
    CV_Assert((opoints.rows == 1 && opoints.channels() == 3) || opoints.cols*opoints.channels() == 3);
    CV_Assert(ipoints.isContinuous());
    CV_Assert(ipoints.depth() == CV_32F || ipoints.depth() == CV_64F);
    CV_Assert((ipoints.rows == 1 && ipoints.channels() == 2) || ipoints.cols*ipoints.channels() == 2);

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);

    Mat rvec = useExtrinsicGuess ? _rvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat tvec = useExtrinsicGuess ? _tvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();

    int model_points = 5;
    int ransac_kernel_method = SOLVEPNP_EPNP;

    if( flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P)
    {
        model_points = 4;
        ransac_kernel_method = flags;
    }
    else if( npoints == 4 )
    {
        model_points = 4;
        ransac_kernel_method = SOLVEPNP_P3P;
    }

    if( model_points == npoints )
    {
        opoints = opoints.reshape(3);
        ipoints = ipoints.reshape(2);

        bool result = solvePnP(opoints, ipoints, cameraMatrix, distCoeffs, _rvec, _tvec, useExtrinsicGuess, ransac_kernel_method);

        if(!result)
        {
            if( _inliers.needed() )
                _inliers.release();

            return false;
        }

        if(_inliers.needed())
        {
            _inliers.create(npoints, 1, CV_32S);
            Mat _local_inliers = _inliers.getMat();
            for(int i = 0; i < npoints; i++)
            {
                _local_inliers.at<int>(i) = i;
            }
        }

        return true;
    }

    Ptr<PointSetRegistrator::Callback> cb; // pointer to callback
    cb = makePtr<PnPRansacCallback>( cameraMatrix, distCoeffs, ransac_kernel_method, useExtrinsicGuess, rvec, tvec);

    double param1 = reprojectionError;                // reprojection error
    double param2 = confidence;                       // confidence
    int param3 = iterationsCount;                     // number maximum iterations

    Mat _local_model(3, 2, CV_64FC1);
    Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);

    // call Ransac
    int result = createRANSACPointSetRegistrator(cb, model_points,
        param1, param2, param3)->run(opoints, ipoints, _local_model, _mask_local_inliers);

    if( result <= 0 || _local_model.rows <= 0)
    {
        _rvec.assign(rvec);    // output rotation vector
        _tvec.assign(tvec);    // output translation vector

        if( _inliers.needed() )
            _inliers.release();

        return false;
    }

    std::vector<Point3d> opoints_inliers;
    std::vector<Point2d> ipoints_inliers;
    opoints = opoints.reshape(3);
    ipoints = ipoints.reshape(2);
    opoints.convertTo(opoints_inliers, CV_64F);
    ipoints.convertTo(ipoints_inliers, CV_64F);

    const uchar* mask = _mask_local_inliers.ptr<uchar>();
    int npoints1 = compressElems(&opoints_inliers[0], mask, 1, npoints);
    compressElems(&ipoints_inliers[0], mask, 1, npoints);

    opoints_inliers.resize(npoints1);
    ipoints_inliers.resize(npoints1);
    try
    {
       if (flags == SOLVEPNP_ITERATIVE && !useExtrinsicGuess)
       {
          rvec = _local_model.col(0).clone();
          tvec = _local_model.col(1).clone();
          useExtrinsicGuess = true;
       }
       result = solvePnP(opoints_inliers, ipoints_inliers, cameraMatrix,
                          distCoeffs, rvec, tvec, useExtrinsicGuess,
                          (flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P) ? SOLVEPNP_EPNP : flags) ? 1 : -1;
    }
    catch (const cv::Exception& e)
    {
        if (flags == SOLVEPNP_ITERATIVE &&
            npoints1 == 5 &&
            e.what() &&
            std::string(e.what()).find("DLT algorithm needs at least 6 points") != std::string::npos
        )
        {
            CV_LOG_INFO(NULL, "solvePnPRansac(): solvePnP stage to compute the final pose using points "
                "in the consensus set raised DLT 6 points exception, use result from MSS (Minimal Sample Sets) stage instead.");
            rvec = _local_model.col(0);    // output rotation vector
            tvec = _local_model.col(1);    // output translation vector
            result = 1;
        }
        else
        {
            // raise other exceptions
            throw;
        }
    }

    if (result <= 0)
    {
        _rvec.assign(_local_model.col(0));    // output rotation vector
        _tvec.assign(_local_model.col(1));    // output translation vector

        if (_inliers.needed())
            _inliers.release();

        CV_LOG_DEBUG(NULL, "solvePnPRansac(): solvePnP stage to compute the final pose using points in the consensus set failed. Return false");
        return false;
    }
    else
    {
        _rvec.assign(rvec);    // output rotation vector
        _tvec.assign(tvec);    // output translation vector
    }

    if(_inliers.needed())
    {
        Mat _local_inliers;
        for (int i = 0; i < npoints; ++i)
        {
            if((int)_mask_local_inliers.at<uchar>(i) != 0) // inliers mask
                _local_inliers.push_back(i);    // output inliers vector
        }
        _local_inliers.copyTo(_inliers);
    }
    return true;
}


bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
                     InputOutputArray cameraMatrix, InputArray distCoeffs,
                     OutputArray rvec, OutputArray tvec, OutputArray inliers,
                     const UsacParams &params) {
    Ptr<usac::Model> model_params;
    usac::setParameters(model_params, cameraMatrix.empty() ? usac::EstimationMethod::P6P :
        usac::EstimationMethod::P3P, params, inliers.needed());
    Ptr<usac::RansacOutput> ransac_output;
    if (usac::run(model_params, imagePoints, objectPoints,
            ransac_output, cameraMatrix, noArray(), distCoeffs, noArray())) {
        if (inliers.needed()) {
            const auto &inliers_mask = ransac_output->getInliersMask();
            Mat inliers_;
            for (int i = 0; i < (int)inliers_mask.size(); i++)
                if (inliers_mask[i])
                    inliers_.push_back(i);
            inliers_.copyTo(inliers);
        }
        const Mat &model = ransac_output->getModel();
        model.col(0).copyTo(rvec);
        model.col(1).copyTo(tvec);
        if (cameraMatrix.empty())
            model.colRange(2, 5).copyTo(cameraMatrix);
        return true;
    } else return false;
}


int solveP3P( InputArray _opoints, InputArray _ipoints,
              InputArray _cameraMatrix, InputArray _distCoeffs,
              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags) {
    CV_INSTRUMENT_REGION();

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );
    CV_Assert( npoints == 3 || npoints == 4 );
    CV_Assert( flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P );

    if (opoints.cols == 3)
        opoints = opoints.reshape(3);
    if (ipoints.cols == 2)
        ipoints = ipoints.reshape(2);

    Mat cameraMatrix0 = _cameraMatrix.getMat();
    Mat distCoeffs0 = _distCoeffs.getMat();
    Mat cameraMatrix = Mat_<double>(cameraMatrix0);
    Mat distCoeffs = Mat_<double>(distCoeffs0);

    Mat undistortedPoints;
    undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
    std::vector<Mat> Rs, ts, rvecs;

    int solutions = 0;
    if (flags == SOLVEPNP_P3P)
    {
        p3p P3Psolver(cameraMatrix);
        solutions = P3Psolver.solve(Rs, ts, opoints, undistortedPoints);
    }
    else if (flags == SOLVEPNP_AP3P)
    {
        ap3p P3Psolver(cameraMatrix);
        solutions = P3Psolver.solve(Rs, ts, opoints, undistortedPoints);
    }

    if (solutions == 0) {
        return 0;
    }

    Mat objPts, imgPts;
    opoints.convertTo(objPts, CV_64F);
    ipoints.convertTo(imgPts, CV_64F);
    if (imgPts.cols > 1)
    {
        imgPts = imgPts.reshape(1);
        imgPts = imgPts.t();
    }
    else
        imgPts = imgPts.reshape(1, 2*imgPts.rows);

    std::vector<double> reproj_errors(solutions);
    for (size_t i = 0; i < reproj_errors.size(); i++)
    {
        Mat rvec;
        Rodrigues(Rs[i], rvec);
        rvecs.push_back(rvec);

        Mat projPts;
        projectPoints(objPts, rvec, ts[i], _cameraMatrix, _distCoeffs, projPts);

        projPts = projPts.reshape(1, 2*projPts.rows);
        Mat err = imgPts - projPts;

        err = err.t() * err;
        reproj_errors[i] = err.at<double>(0,0);
    }

    //sort the solutions
    for (int i = 1; i < solutions; i++)
    {
        for (int j = i; j > 0 && reproj_errors[j-1] > reproj_errors[j]; j--)
        {
            std::swap(reproj_errors[j], reproj_errors[j-1]);
            std::swap(rvecs[j], rvecs[j-1]);
            std::swap(ts[j], ts[j-1]);
        }
    }

    int depthRot = _rvecs.fixedType() ? _rvecs.depth() : CV_64F;
    int depthTrans = _tvecs.fixedType() ? _tvecs.depth() : CV_64F;
    _rvecs.create(solutions, 1, CV_MAKETYPE(depthRot, _rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));
    _tvecs.create(solutions, 1, CV_MAKETYPE(depthTrans, _tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));

    for (int i = 0; i < solutions; i++)
    {
        Mat rvec0, tvec0;
        if (depthRot == CV_64F)
            rvec0 = rvecs[i];
        else
            rvecs[i].convertTo(rvec0, depthRot);

        if (depthTrans == CV_64F)
            tvec0 = ts[i];
        else
            ts[i].convertTo(tvec0, depthTrans);

        if (_rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR)
        {
            Mat rref = _rvecs.getMat_();

            if (_rvecs.depth() == CV_32F)
                rref.at<Vec3f>(0,i) = Vec3f(rvec0.at<float>(0,0), rvec0.at<float>(1,0), rvec0.at<float>(2,0));
            else
                rref.at<Vec3d>(0,i) = Vec3d(rvec0.at<double>(0,0), rvec0.at<double>(1,0), rvec0.at<double>(2,0));
        }
        else
        {
            _rvecs.getMatRef(i) = rvec0;
        }

        if (_tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR)
        {

            Mat tref = _tvecs.getMat_();

            if (_tvecs.depth() == CV_32F)
                tref.at<Vec3f>(0,i) = Vec3f(tvec0.at<float>(0,0), tvec0.at<float>(1,0), tvec0.at<float>(2,0));
            else
                tref.at<Vec3d>(0,i) = Vec3d(tvec0.at<double>(0,0), tvec0.at<double>(1,0), tvec0.at<double>(2,0));
        }
        else
        {
            _tvecs.getMatRef(i) = tvec0;
        }
    }

    return solutions;
}


/**
 * @brief Compute the Interaction matrix and the residuals for the current pose.
 * @param objectPoints 3D object points.
 * @param R Current estimated rotation matrix.
 * @param tvec Current estimated translation vector.
 * @param L Interaction matrix for a vector of point features.
 * @param s Residuals.
 */
static void computeInteractionMatrixAndResiduals(const Mat& objectPoints, const Mat& R, const Mat& tvec,
                                                 Mat& L, Mat& s)
{
    Mat objectPointsInCam;

    int npoints = objectPoints.rows;
    for (int i = 0; i < npoints; i++)
    {
        Mat curPt = objectPoints.row(i);
        objectPointsInCam = R * curPt.t() + tvec;

        double Zi = objectPointsInCam.at<double>(2,0);
        double xi = objectPointsInCam.at<double>(0,0) / Zi;
        double yi = objectPointsInCam.at<double>(1,0) / Zi;

        s.at<double>(2*i,0) = xi;
        s.at<double>(2*i+1,0) = yi;

        L.at<double>(2*i,0) = -1 / Zi;
        L.at<double>(2*i,1) = 0;
        L.at<double>(2*i,2) = xi / Zi;
        L.at<double>(2*i,3) = xi*yi;
        L.at<double>(2*i,4) = -(1 + xi*xi);
        L.at<double>(2*i,5) = yi;

        L.at<double>(2*i+1,0) = 0;
        L.at<double>(2*i+1,1) = -1 / Zi;
        L.at<double>(2*i+1,2) = yi / Zi;
        L.at<double>(2*i+1,3) = 1 + yi*yi;
        L.at<double>(2*i+1,4) = -xi*yi;
        L.at<double>(2*i+1,5) = -xi;
    }
}

/**
 * @brief The exponential map from se(3) to SE(3).
 * @param twist A twist (v, w) represents the velocity of a rigid body as an angular velocity
 * around an axis and a linear velocity along this axis.
 * @param R1 Resultant rotation matrix from the twist.
 * @param t1 Resultant translation vector from the twist.
 */
static void exponentialMapToSE3Inv(const Mat& twist, Mat& R1, Mat& t1)
{
    //see Exponential Map in http://ethaneade.com/lie.pdf
    /*
    \begin{align*}
    \boldsymbol{\delta} &= \left( \mathbf{u}, \boldsymbol{\omega} \right ) \in se(3) \\
    \mathbf{u}, \boldsymbol{\omega} &\in \mathbb{R}^3 \\
    \theta &= \sqrt{ \boldsymbol{\omega}^T \boldsymbol{\omega} } \\
    A &= \frac{\sin \theta}{\theta} \\
    B &= \frac{1 - \cos \theta}{\theta^2} \\
    C &= \frac{1-A}{\theta^2} \\
    \mathbf{R} &= \mathbf{I} + A \boldsymbol{\omega}_{\times} + B \boldsymbol{\omega}_{\times}^2 \\
    \mathbf{V} &= \mathbf{I} + B \boldsymbol{\omega}_{\times} + C \boldsymbol{\omega}_{\times}^2 \\
    \exp \begin{pmatrix}
    \mathbf{u} \\
    \boldsymbol{\omega}
    \end{pmatrix} &=
    \left(
    \begin{array}{c|c}
    \mathbf{R} & \mathbf{V} \mathbf{u} \\ \hline
    \mathbf{0} & 1
    \end{array}
    \right )
    \end{align*}
    */
    double vx = twist.at<double>(0,0);
    double vy = twist.at<double>(1,0);
    double vz = twist.at<double>(2,0);
    double wx = twist.at<double>(3,0);
    double wy = twist.at<double>(4,0);
    double wz = twist.at<double>(5,0);

    Matx31d rvec(wx, wy, wz);
    Mat R;
    Rodrigues(rvec, R);

    double theta = sqrt(wx*wx + wy*wy + wz*wz);
    double sinc = std::fabs(theta) < 1e-8 ? 1 : std::sin(theta) / theta;
    double mcosc = (std::fabs(theta) < 1e-8) ? 0.5 : (1- std::cos(theta)) / (theta*theta);
    double msinc = (std::abs(theta) < 1e-8) ? (1/6.0) : (1-sinc) / (theta*theta);

    Matx31d dt;
    dt(0) = vx*(sinc + wx*wx*msinc) + vy*(wx*wy*msinc - wz*mcosc) + vz*(wx*wz*msinc + wy*mcosc);
    dt(1) = vx*(wx*wy*msinc + wz*mcosc) + vy*(sinc + wy*wy*msinc) + vz*(wy*wz*msinc - wx*mcosc);
    dt(2) = vx*(wx*wz*msinc - wy*mcosc) + vy*(wy*wz*msinc + wx*mcosc) + vz*(sinc + wz*wz*msinc);

    R1 = R.t();
    t1 = -R1 * dt;
}

enum SolvePnPRefineMethod {
    SOLVEPNP_REFINE_LM   = 0,
    SOLVEPNP_REFINE_VVS  = 1
};

static void solvePnPRefine(InputArray _objectPoints, InputArray _imagePoints,
                           InputArray _cameraMatrix, InputArray _distCoeffs,
                           InputOutputArray _rvec, InputOutputArray _tvec,
                           SolvePnPRefineMethod _flags,
                           TermCriteria _criteria=TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 20, FLT_EPSILON),
                           double _vvslambda=1)
{
    CV_INSTRUMENT_REGION();

    Mat opoints_ = _objectPoints.getMat(), ipoints_ = _imagePoints.getMat();
    Mat opoints, ipoints;
    opoints_.convertTo(opoints, CV_64F);
    ipoints_.convertTo(ipoints, CV_64F);
    int npoints = opoints.checkVector(3, CV_64F);
    CV_Assert( npoints >= 3 && npoints == ipoints.checkVector(2, CV_64F) );
    CV_Assert( !_rvec.empty() && !_tvec.empty() );

    int rtype = _rvec.type(), ttype = _tvec.type();
    Size rsize = _rvec.size(), tsize = _tvec.size();
    CV_Assert( (rtype == CV_32FC1 || rtype == CV_64FC1) &&
               (ttype == CV_32FC1 || ttype == CV_64FC1) );
    CV_Assert( (rsize == Size(1, 3) || rsize == Size(3, 1)) &&
               (tsize == Size(1, 3) || tsize == Size(3, 1)) );

    Mat cameraMatrix0 = _cameraMatrix.getMat();
    Mat distCoeffs0 = _distCoeffs.getMat();
    Mat cameraMatrix = Mat_<double>(cameraMatrix0);
    Mat distCoeffs = Mat_<double>(distCoeffs0);

    if (_flags == SOLVEPNP_REFINE_LM)
    {
        Mat rvec0 = _rvec.getMat(), tvec0 = _tvec.getMat();
        Mat rvec, tvec;
        rvec0.convertTo(rvec, CV_64F);
        tvec0.convertTo(tvec, CV_64F);

        Mat params(6, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
        {
            params.at<double>(i,0) = rvec.at<double>(i,0);
            params.at<double>(i+3,0) = tvec.at<double>(i,0);
        }

        int npts = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
        Mat imagePoints0 = ipoints.reshape(1, npts * 2);
        auto solvePnPRefineLMCallback = [opoints, imagePoints0, npts, cameraMatrix, distCoeffs]
            (InputOutputArray _param, OutputArray _err, OutputArray _Jac) -> bool
        {
            Mat param = _param.getMat();
            _err.create(npts * 2, 1, CV_64FC1);

            if (_Jac.needed())
            {
                _Jac.create(npts * 2, param.rows, CV_64FC1);
            }

            Mat prvec = param(Rect(0, 0, 1, 3)), ptvec = param(Rect(0, 3, 1, 3));

            Mat J, projectedPts;
            projectPoints(opoints, prvec, ptvec, cameraMatrix, distCoeffs, projectedPts, _Jac.needed() ? J : noArray());

            if (_Jac.needed())
            {
                Mat Jac = _Jac.getMat();
                for (int i = 0; i < Jac.rows; i++)
                {
                    for (int j = 0; j < Jac.cols; j++)
                    {
                        Jac.at<double>(i, j) = J.at<double>(i, j);
                    }
                }
            }

            Mat err = _err.getMat();
            projectedPts = projectedPts.reshape(1, npts * 2);
            err = projectedPts - imagePoints0;

            return true;
        };
        LevMarq solver(params, solvePnPRefineLMCallback,
                       LevMarq::Settings()
                       .setMaxIterations((unsigned int)_criteria.maxCount)
                       .setStepNormTolerance((double)_criteria.epsilon)
                       .setSmallEnergyTolerance((double)_criteria.epsilon * (double)_criteria.epsilon)
                       .setGeodesic(true));
        solver.optimize();

        params.rowRange(0, 3).convertTo(rvec0, rvec0.depth());
        params.rowRange(3, 6).convertTo(tvec0, tvec0.depth());
    }
    else if (_flags == SOLVEPNP_REFINE_VVS)
    {
        Mat rvec0 = _rvec.getMat(), tvec0 = _tvec.getMat();
        Mat rvec, tvec;
        rvec0.convertTo(rvec, CV_64F);
        tvec0.convertTo(tvec, CV_64F);

        std::vector<Point2d> ipoints_normalized;
        undistortPoints(ipoints, ipoints_normalized, cameraMatrix, distCoeffs);
        Mat sd = Mat(ipoints_normalized).reshape(1, npoints*2);
        Mat objectPoints0 = opoints.reshape(1, npoints);
        Mat imagePoints0 = ipoints.reshape(1, npoints*2);
        Mat L(npoints*2, 6, CV_64FC1), s(npoints*2, 1, CV_64FC1);

        double residuals_1 = std::numeric_limits<double>::max(), residuals = 0;
        Mat err;
        Mat R;
        Rodrigues(rvec, R);
        for (int iter = 0; iter < _criteria.maxCount; iter++)
        {
            computeInteractionMatrixAndResiduals(objectPoints0, R, tvec, L, s);
            err = s - sd;

            Mat Lp = L.inv(cv::DECOMP_SVD);
            Mat dq = -_vvslambda * Lp * err;

            Mat R1, t1;
            exponentialMapToSE3Inv(dq, R1, t1);
            R = R1 * R;
            tvec = R1 * tvec + t1;

            residuals_1 = residuals;
            Mat res = err.t()*err;
            residuals = res.at<double>(0,0);

            if (std::fabs(residuals - residuals_1) < _criteria.epsilon)
                break;
        }

        Rodrigues(R, rvec);
        rvec.convertTo(rvec0, rvec0.depth());
        tvec.convertTo(tvec0, tvec0.depth());
    }
}

void solvePnPRefineLM(InputArray _objectPoints, InputArray _imagePoints,
                      InputArray _cameraMatrix, InputArray _distCoeffs,
                      InputOutputArray _rvec, InputOutputArray _tvec,
                      TermCriteria _criteria)
{
    CV_INSTRUMENT_REGION();
    solvePnPRefine(_objectPoints, _imagePoints, _cameraMatrix, _distCoeffs, _rvec, _tvec, SOLVEPNP_REFINE_LM, _criteria);
}

void solvePnPRefineVVS(InputArray _objectPoints, InputArray _imagePoints,
                       InputArray _cameraMatrix, InputArray _distCoeffs,
                       InputOutputArray _rvec, InputOutputArray _tvec,
                       TermCriteria _criteria, double _VVSlambda)
{
    CV_INSTRUMENT_REGION();
    solvePnPRefine(_objectPoints, _imagePoints, _cameraMatrix, _distCoeffs, _rvec, _tvec, SOLVEPNP_REFINE_VVS, _criteria, _VVSlambda);
}

int solvePnPGeneric( InputArray _opoints, InputArray _ipoints,
                     InputArray _cameraMatrix, InputArray _distCoeffs,
                     OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                     bool useExtrinsicGuess, int flags,
                     InputArray _rvec, InputArray _tvec,
                     OutputArray reprojectionError) {
    CV_INSTRUMENT_REGION();

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( ( (npoints >= 4) || (npoints == 3 && flags == SOLVEPNP_ITERATIVE && useExtrinsicGuess)
                || (npoints >= 3 && flags == SOLVEPNP_SQPNP) )
               && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

    opoints = opoints.reshape(3, npoints);
    ipoints = ipoints.reshape(2, npoints);

    if( flags != SOLVEPNP_ITERATIVE )
        useExtrinsicGuess = false;

    if (useExtrinsicGuess)
        CV_Assert( !_rvec.empty() && !_tvec.empty() );

    if( useExtrinsicGuess )
    {
        int rtype = _rvec.type(), ttype = _tvec.type();
        Size rsize = _rvec.size(), tsize = _tvec.size();
        CV_Assert( (rtype == CV_32FC1 || rtype == CV_64FC1) &&
                   (ttype == CV_32FC1 || ttype == CV_64FC1) );
        CV_Assert( (rsize == Size(1, 3) || rsize == Size(3, 1)) &&
                   (tsize == Size(1, 3) || tsize == Size(3, 1)) );
    }

    Mat cameraMatrix0 = _cameraMatrix.getMat();
    Mat distCoeffs0 = _distCoeffs.getMat();
    Mat cameraMatrix = Mat_<double>(cameraMatrix0);
    Mat distCoeffs = Mat_<double>(distCoeffs0);

    std::vector<Mat> vec_rvecs, vec_tvecs;
    if (flags == SOLVEPNP_EPNP)
    {
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        epnp PnP(cameraMatrix, opoints, undistortedPoints);

        Mat rvec, tvec, R;
        PnP.compute_pose(R, tvec);
        Rodrigues(R, rvec);

        vec_rvecs.push_back(rvec);
        vec_tvecs.push_back(tvec);
    }
    else if (flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P)
    {
        std::vector<Mat> rvecs, tvecs;
        solveP3P(opoints, ipoints, _cameraMatrix, _distCoeffs, rvecs, tvecs, flags);
        vec_rvecs.insert(vec_rvecs.end(), rvecs.begin(), rvecs.end());
        vec_tvecs.insert(vec_tvecs.end(), tvecs.begin(), tvecs.end());
    }
    else if (flags == SOLVEPNP_ITERATIVE)
    {
        Mat rvec, tvec;
        if (useExtrinsicGuess)
        {
            rvec = _rvec.getMat();
            tvec = _tvec.getMat();
        }
        else
        {
            rvec.create(3, 1, CV_64FC1);
            tvec.create(3, 1, CV_64FC1);
        }

        findExtrinsicCameraParams2(opoints, ipoints, cameraMatrix, distCoeffs,
                                   rvec, tvec, useExtrinsicGuess );
        vec_rvecs.push_back(rvec);
        vec_tvecs.push_back(tvec);
    }
    else if (flags == SOLVEPNP_IPPE)
    {
        CV_DbgAssert(isPlanarObjectPoints(opoints, 1e-3));
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);

        IPPE::PoseSolver poseSolver;
        Mat rvec1, tvec1, rvec2, tvec2;
        float reprojErr1, reprojErr2;
        try
        {
            poseSolver.solveGeneric(opoints, undistortedPoints, rvec1, tvec1, reprojErr1, rvec2, tvec2, reprojErr2);

            if (reprojErr1 < reprojErr2)
            {
                vec_rvecs.push_back(rvec1);
                vec_tvecs.push_back(tvec1);

                vec_rvecs.push_back(rvec2);
                vec_tvecs.push_back(tvec2);
            }
            else
            {
                vec_rvecs.push_back(rvec2);
                vec_tvecs.push_back(tvec2);

                vec_rvecs.push_back(rvec1);
                vec_tvecs.push_back(tvec1);
            }
        }
        catch (...) { }
    }
    else if (flags == SOLVEPNP_IPPE_SQUARE)
    {
        CV_Assert(npoints == 4);

#if !defined(NDEBUG) || defined(CV_STATIC_ANALYSIS)
        double Xs[4][3];
        if (opoints.depth() == CV_32F)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Xs[i][j] = opoints.ptr<Vec3f>(0)[i](j);
                }
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Xs[i][j] = opoints.ptr<Vec3d>(0)[i](j);
                }
            }
        }

        const double equalThreshold = 1e-9;
        //Z must be zero
        for (int i = 0; i < 4; i++)
        {
            CV_DbgCheck(Xs[i][2], approxEqual(Xs[i][2], 0, equalThreshold), "Z object point coordinate must be zero!");
        }
        //Y0 == Y1 && Y2 == Y3
        CV_DbgCheck(Xs[0][1], approxEqual(Xs[0][1], Xs[1][1], equalThreshold), "Object points must be: Y0 == Y1!");
        CV_DbgCheck(Xs[2][1], approxEqual(Xs[2][1], Xs[3][1], equalThreshold), "Object points must be: Y2 == Y3!");
        //X0 == X3 && X1 == X2
        CV_DbgCheck(Xs[0][0], approxEqual(Xs[0][0], Xs[3][0], equalThreshold), "Object points must be: X0 == X3!");
        CV_DbgCheck(Xs[1][0], approxEqual(Xs[1][0], Xs[2][0], equalThreshold), "Object points must be: X1 == X2!");
        //X1 == Y1 && X3 == Y3
        CV_DbgCheck(Xs[1][0], approxEqual(Xs[1][0], Xs[1][1], equalThreshold), "Object points must be: X1 == Y1!");
        CV_DbgCheck(Xs[3][0], approxEqual(Xs[3][0], Xs[3][1], equalThreshold), "Object points must be: X3 == Y3!");
#endif

        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);

        IPPE::PoseSolver poseSolver;
        Mat rvec1, tvec1, rvec2, tvec2;
        float reprojErr1, reprojErr2;
        try
        {
            poseSolver.solveSquare(opoints, undistortedPoints, rvec1, tvec1, reprojErr1, rvec2, tvec2, reprojErr2);

            if (reprojErr1 < reprojErr2)
            {
                vec_rvecs.push_back(rvec1);
                vec_tvecs.push_back(tvec1);

                vec_rvecs.push_back(rvec2);
                vec_tvecs.push_back(tvec2);
            }
            else
            {
                vec_rvecs.push_back(rvec2);
                vec_tvecs.push_back(tvec2);

                vec_rvecs.push_back(rvec1);
                vec_tvecs.push_back(tvec1);
            }
        } catch (...) { }
    }
    else if (flags == SOLVEPNP_SQPNP)
    {
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);

        sqpnp::PoseSolver solver;
        solver.solve(opoints, undistortedPoints, vec_rvecs, vec_tvecs);
    }
    else
        CV_Error(cv::Error::StsBadArg, "The flags argument must be one of SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, "
            "SOLVEPNP_EPNP, SOLVEPNP_AP3P, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE or SOLVEPNP_SQPNP");

    CV_Assert(vec_rvecs.size() == vec_tvecs.size());

    int solutions = static_cast<int>(vec_rvecs.size());

    int depthRot = _rvecs.fixedType() ? _rvecs.depth() : CV_64F;
    int depthTrans = _tvecs.fixedType() ? _tvecs.depth() : CV_64F;
    _rvecs.create(solutions, 1, CV_MAKETYPE(depthRot, _rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));
    _tvecs.create(solutions, 1, CV_MAKETYPE(depthTrans, _tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));

    for (int i = 0; i < solutions; i++)
    {
        Mat rvec0, tvec0;
        if (depthRot == CV_64F)
            rvec0 = vec_rvecs[i];
        else
            vec_rvecs[i].convertTo(rvec0, depthRot);

        if (depthTrans == CV_64F)
            tvec0 = vec_tvecs[i];
        else
            vec_tvecs[i].convertTo(tvec0, depthTrans);

        if (_rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR)
        {
            Mat rref = _rvecs.getMat_();

            if (_rvecs.depth() == CV_32F)
                rref.at<Vec3f>(0,i) = Vec3f(rvec0.at<float>(0,0), rvec0.at<float>(1,0), rvec0.at<float>(2,0));
            else
                rref.at<Vec3d>(0,i) = Vec3d(rvec0.at<double>(0,0), rvec0.at<double>(1,0), rvec0.at<double>(2,0));
        }
        else
        {
            _rvecs.getMatRef(i) = rvec0;
        }

        if (_tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR)
        {

            Mat tref = _tvecs.getMat_();

            if (_tvecs.depth() == CV_32F)
                tref.at<Vec3f>(0,i) = Vec3f(tvec0.at<float>(0,0), tvec0.at<float>(1,0), tvec0.at<float>(2,0));
            else
                tref.at<Vec3d>(0,i) = Vec3d(tvec0.at<double>(0,0), tvec0.at<double>(1,0), tvec0.at<double>(2,0));
        }
        else
        {
            _tvecs.getMatRef(i) = tvec0;
        }
    }

    if (reprojectionError.needed())
    {
        int type = (reprojectionError.fixedType() || !reprojectionError.empty())
                ? reprojectionError.type()
                : (max(_ipoints.depth(), _opoints.depth()) == CV_64F ? CV_64F : CV_32F);

        reprojectionError.create(solutions, 1, type);
        CV_CheckType(reprojectionError.type(), type == CV_32FC1 || type == CV_64FC1,
                     "Type of reprojectionError must be CV_32FC1 or CV_64FC1!");

        Mat objectPoints, imagePoints;
        if (opoints.depth() == CV_32F)
        {
            opoints.convertTo(objectPoints, CV_64F);
        }
        else
        {
            objectPoints = opoints;
        }
        if (ipoints.depth() == CV_32F)
        {
            ipoints.convertTo(imagePoints, CV_64F);
        }
        else
        {
            imagePoints = ipoints;
        }

        for (size_t i = 0; i < vec_rvecs.size(); i++)
        {
            Mat projectedPoints;
            projectPoints(objectPoints, vec_rvecs[i], vec_tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
            int nprojectedPoints = (int)projectedPoints.total();
            double rmse = norm(projectedPoints, imagePoints, NORM_L2) / sqrt(2*nprojectedPoints);

            Mat err = reprojectionError.getMat();
            if (type == CV_32F)
            {
                err.at<float>(static_cast<int>(i)) = static_cast<float>(rmse);
            }
            else
            {
                err.at<double>(static_cast<int>(i)) = rmse;
            }
        }
    }

    return solutions;
}

}
