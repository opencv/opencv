// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "calib3d_c_api.h"

using namespace cv;

Ptr<PnPRefinerLM> PnPRefinerLM::create()
{
    return makePtr<PnPRefinerLM>();
}

bool PnPRefinerLM::refine(InputArray opoints, InputArray ipoints, InputArray cameraMatrix, InputArray distortion, InputArray rVec, InputArray tVec, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    Mat rIn,tIn;
    rVec.copyTo(rIn);
    tVec.copyTo(tIn);

    CvMat c_objectPoints = cvMat(opoints.getMat()), c_imagePoints = cvMat(ipoints.getMat());
    CvMat c_cameraMatrix = cvMat(cameraMatrix.getMat()), c_distCoeffs = cvMat(distortion.getMat());
    CvMat c_rvec = cvMat(rIn), c_tvec = cvMat(tIn);
    cvFindExtrinsicCameraParams2(&c_objectPoints, &c_imagePoints, &c_cameraMatrix,
                                 (c_distCoeffs.rows && c_distCoeffs.cols) ? &c_distCoeffs : 0,
                                 &c_rvec, &c_tvec, true );

    rVecs.clear();
    tVecs.clear();
    rVecs.push_back(rIn);
    tVecs.push_back(tIn);
    return true;
}

class SolvePnPRefineLMCallback CV_FINAL : public LMSolver::Callback
{
public:
    SolvePnPRefineLMCallback(InputArray _opoints, InputArray _ipoints, InputArray _cameraMatrix, InputArray _distCoeffs)
    {
        objectPoints = _opoints.getMat();
        imagePoints = _ipoints.getMat();
        npoints = std::max(objectPoints.checkVector(3, CV_32F), objectPoints.checkVector(3, CV_64F));
        imagePoints0 = imagePoints.reshape(1, npoints*2);
        cameraMatrix = _cameraMatrix.getMat();
        distCoeffs = _distCoeffs.getMat();
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const CV_OVERRIDE
    {
        Mat param = _param.getMat();
        _err.create(npoints*2, 1, CV_64FC1);

        if(_Jac.needed())
        {
            _Jac.create(npoints*2, param.rows, CV_64FC1);
        }

        Mat rvec = param(Rect(0, 0, 1, 3)), tvec = param(Rect(0, 3, 1, 3));

        Mat J, projectedPts;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPts, _Jac.needed() ? J : noArray());

        if (_Jac.needed())
        {
            Mat Jac = _Jac.getMat();
            for (int i = 0; i < Jac.rows; i++)
            {
                for (int j = 0; j < Jac.cols; j++)
                {
                    Jac.at<double>(i,j) = J.at<double>(i,j);
                }
            }
        }

        Mat err = _err.getMat();
        projectedPts = projectedPts.reshape(1, npoints*2);
        err = projectedPts - imagePoints0;

        return true;
    }

    Mat objectPoints, imagePoints, imagePoints0;
    Mat cameraMatrix, distCoeffs;
    int npoints;
};

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
    double sinc = std::fabs(theta) < 1e-8 ? 1 : sin(theta) / theta;
    double mcosc = (std::fabs(theta) < 1e-8) ? 0.5 : (1-cos(theta)) / (theta*theta);
    double msinc = (std::abs(theta) < 1e-8) ? (1/6.0) : (1-sinc) / (theta*theta);

    Matx31d dt;
    dt(0) = vx*(sinc + wx*wx*msinc) + vy*(wx*wy*msinc - wz*mcosc) + vz*(wx*wz*msinc + wy*mcosc);
    dt(1) = vx*(wx*wy*msinc + wz*mcosc) + vy*(sinc + wy*wy*msinc) + vz*(wy*wz*msinc - wx*mcosc);
    dt(2) = vx*(wx*wz*msinc - wy*mcosc) + vy*(wy*wz*msinc + wx*mcosc) + vz*(sinc + wz*wz*msinc);

    R1 = R.t();
    t1 = -R1 * dt;
}

PnPRefinerLMcpp::PnPRefinerLMcpp(TermCriteria _criteria): criteria(_criteria)
{

}

Ptr<PnPRefinerLMcpp> PnPRefinerLMcpp::create(TermCriteria _criteria)
{
    return makePtr<PnPRefinerLMcpp>(_criteria);
}


bool PnPRefinerLMcpp::refine(InputArray _objectPoints, InputArray _imagePoints, InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec, InputArray _tvec, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
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

    LMSolver::create(makePtr<SolvePnPRefineLMCallback>(opoints, ipoints, cameraMatrix, distCoeffs), criteria.maxCount, criteria.epsilon)->run(params);

    params.rowRange(0, 3).convertTo(rvec0, rvec0.depth());
    params.rowRange(3, 6).convertTo(tvec0, tvec0.depth());
    rVecs.clear();
    rVecs.push_back(rvec0);
    tVecs.clear();
    tVecs.push_back(tvec0);
    return true;
}





PnPRefinerVVS::PnPRefinerVVS(TermCriteria _criteria, double _vvslambda): criteria(_criteria), vvslambda(_vvslambda)
{

}

Ptr<PnPRefinerVVS> PnPRefinerVVS::create(TermCriteria _criteria, double _vvslambda)
{
    return makePtr<PnPRefinerVVS>(_criteria,_vvslambda);
}


bool PnPRefinerVVS::refine(InputArray _objectPoints, InputArray _imagePoints, InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec, InputArray _tvec, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
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
    for (int iter = 0; iter < criteria.maxCount; iter++)
    {
        computeInteractionMatrixAndResiduals(objectPoints0, R, tvec, L, s);
        err = s - sd;

        Mat Lp = L.inv(cv::DECOMP_SVD);
        Mat dq = -vvslambda * Lp * err;

        Mat R1, t1;
        exponentialMapToSE3Inv(dq, R1, t1);
        R = R1 * R;
        tvec = R1 * tvec + t1;

        residuals_1 = residuals;
        Mat res = err.t()*err;
        residuals = res.at<double>(0,0);

        if (std::fabs(residuals - residuals_1) < criteria.epsilon)
            break;
    }

    Rodrigues(R, rvec);
    rvec.convertTo(rvec0, rvec0.depth());
    tvec.convertTo(tvec0, tvec0.depth());
    rVecs.clear();
    rVecs.push_back(rvec0);
    tVecs.clear();
    tVecs.push_back(tvec0);
    return true;
}
