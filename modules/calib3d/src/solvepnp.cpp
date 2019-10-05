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
#include "upnp.h"
#include "dls.h"
#include "epnp.h"
#include "p3p.h"
#include "ap3p.h"
#include "ippe.hpp"
#include "opencv2/calib3d/calib3d_c.h"

namespace cv
{
#if defined _DEBUG || defined CV_STATIC_ANALYSIS
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
    vector<Point3f> axesPoints;
    axesPoints.push_back(Point3f(0, 0, 0));
    axesPoints.push_back(Point3f(length, 0, 0));
    axesPoints.push_back(Point3f(0, length, 0));
    axesPoints.push_back(Point3f(0, 0, length));
    vector<Point2f> imagePoints;
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

    vector<Mat> rvecs, tvecs;
    Ptr<PnPSolver> solver;
    Ptr<PnPRefiner> refiner;

    SolvePnPMethod flg = static_cast<SolvePnPMethod>(flags);
    cvtSolvePnPFlag(flg,useExtrinsicGuess,solver,refiner);

    if (solver.empty())
    {
        CV_Assert(!rvec.empty());
        CV_Assert(!tvec.empty());
        rvecs.push_back(rvec.getMat());
        tvecs.push_back(tvec.getMat());
    }

    int solutions = solvePnPGeneric(opoints, ipoints, cameraMatrix, distCoeffs, rvecs, tvecs, solver, refiner, true);
    if ((solutions == 0) && (flg == SOLVEPNP_ITERATIVE))
    {
        ////this copies the behaviour of SOLVEPNP_ITERATIVE, where if it is not possible to find an initial solution, all-zeros are used for r and t, then this is refined.
        cv::Mat rInit(3,1,CV_64FC1);
        rInit.setTo(0.0);
        cv::Mat tInit(3,1,CV_64FC1);
        tInit.setTo(0,0);
        refiner->refine(opoints,ipoints,cameraMatrix,distCoeffs,rInit,tInit,rvecs,tvecs);
    }


    if (solutions > 0)
    {
        rvecs.resize(1);
        tvecs.resize(1);

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

    PnPRansacCallback(Ptr<PnPSolver> _minSolver, Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), Mat _rvec=Mat(), Mat _tvec=Mat() )
        : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), minSolver(_minSolver),
          rvec(_rvec), tvec(_tvec) {}

    /* Pre: True */
    /* Post: compute _model with given points and return number of found models */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const CV_OVERRIDE
    {
        std::vector<Mat> rs,ts;
        int numSolutions = solvePnPGeneric(_m1, _m2, cameraMatrix, distCoeffs,
                                           rs, ts, minSolver,Ptr<PnPRefiner>(), true);

        if (numSolutions>0)
        {
            Mat _local_model;
            hconcat(rs[0], ts[0], _local_model);
            _local_model.copyTo(_model);
        }

        return numSolutions>0;
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
    const Ptr<PnPSolver> minSolver;
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

    Ptr<PnPSolver> minimalSolver; //used to solve the minimal problem
    Ptr<PnPSolver> inlierSolver;  //after a model has been found, used to solve using model's inliers
    Ptr<PnPRefiner> inlierRefiner; //used to refine solution from initialSolver (optional)

    //handling of flags input
    SolvePnPMethod m = static_cast<SolvePnPMethod>(flags);
    int model_points;
    if (m == SOLVEPNP_P3P)
    {
        minimalSolver = PnPSolverP3PComplete::create();
        model_points = 4;
    }
    else if (m == SOLVEPNP_AP3P)
    {
        minimalSolver = PnPSolverAP3P::create();
        model_points = 4;
    }
    else if (npoints ==4)
    {
        minimalSolver =PnPSolverP3PComplete::create();
        model_points = 4;
    }
    else
    {
        minimalSolver =PnPSolverEPnP3D::create();
        model_points = 5;
    }

    if (m == SOLVEPNP_P3P || m == SOLVEPNP_AP3P)
    {
        inlierSolver = PnPSolverEPnP3D::create();
    }
    else {
        cvtSolvePnPFlag(m,useExtrinsicGuess,inlierSolver,inlierRefiner);
    }

    CV_Assert(!minimalSolver.empty());

    if( model_points == npoints )
    {
        opoints = opoints.reshape(3);
        ipoints = ipoints.reshape(2);
        std::vector<cv::Mat> rVecs, tVecs;

        bool result = (solvePnPGeneric(opoints, ipoints, cameraMatrix, distCoeffs, rVecs, tVecs, minimalSolver, Ptr<PnPRefiner>(), true)>0);

        if(!result)
        {
            if( _inliers.needed() )
                _inliers.release();

            return false;
        }
        else {
            _rvec.assign(rVecs[0]);    // output rotation vector
            _tvec.assign(tVecs[0]);    // output translation vector
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
    cb = makePtr<PnPRansacCallback>(minimalSolver, cameraMatrix, distCoeffs, rvec, tvec);

    double param1 = static_cast<double>(reprojectionError);                // reprojection error
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

    vector<Point3d> opoints_inliers;
    vector<Point2d> ipoints_inliers;
    opoints = opoints.reshape(3);
    ipoints = ipoints.reshape(2);
    opoints.convertTo(opoints_inliers, CV_64F);
    ipoints.convertTo(ipoints_inliers, CV_64F);

    const uchar* mask = _mask_local_inliers.ptr<uchar>();
    int npoints1 = compressElems(&opoints_inliers[0], mask, 1, npoints);
    compressElems(&ipoints_inliers[0], mask, 1, npoints);

    opoints_inliers.resize(static_cast<size_t>(npoints1));
    ipoints_inliers.resize(static_cast<size_t>(npoints1));

    std::vector<cv::Mat> rVecs, tVecs;
    result = solvePnPGeneric(opoints_inliers, ipoints_inliers, cameraMatrix,
                             distCoeffs, rVecs, tVecs, inlierSolver,inlierRefiner);


    if( result <= 0 )
    {
        _rvec.assign(_local_model.col(0));    // output rotation vector
        _tvec.assign(_local_model.col(1));    // output translation vector

        if( _inliers.needed() )
            _inliers.release();

        return false;
    }
    else
    {
        _rvec.assign(rVecs[0]);    // output rotation vector
        _tvec.assign(tVecs[0]);    // output translation vector
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

void sortPosesOnReprojectionError(InputArray _opoints, InputArray _ipoints, InputArray cameraMatrix, InputArray distCoeffs, std::vector<Mat> & rVecs,
                                  std::vector<Mat> & tVecs,  std::vector<double> & rmses)
{

    CV_Assert(tVecs.size()==rVecs.size());
    CV_Assert(tVecs.size()>0);
    CV_Assert(_opoints.depth()==CV_64F);
    CV_Assert(_ipoints.depth()==CV_64F);

    rmses.clear();
    size_t solutions = rVecs.size();

    //sort poses on reprojection error:
    std::vector<Mat> rVecsSorted(solutions),tVecsSorted(solutions);
    std::vector<double> sortedRmses;
    rmses.reserve(solutions);
    int npoints = _opoints.getMat().checkVector(3, CV_64F);
    for (size_t i = 0; i < solutions; i++)
    {
        auto rv = rVecs[i];
        auto tv = tVecs[i];

        cv::Mat projectedPoints;
        projectPoints(_opoints,rv,tv, cameraMatrix, distCoeffs, projectedPoints);
        if (projectedPoints.rows != _ipoints.getMat().rows)
        {
            transpose(projectedPoints,projectedPoints);

        }
        double rmse;
        cv::Mat ipoints = _ipoints.getMat();
        if (_ipoints.channels()!=2)
        {
            ipoints = ipoints.reshape(2);
        }
        rmse = norm(projectedPoints, ipoints, NORM_L2) / sqrt(2*npoints);
        rmses.push_back(rmse);
    }

    std::vector<size_t> y(solutions);
    std::size_t n(0);
    std::generate(std::begin(y), std::end(y), [&]{ return n++; });

    std::sort(  std::begin(y),
                std::end(y),
                [&](size_t i1, size_t i2) { return rmses[i1] < rmses[i2]; } );

    sortedRmses.resize(solutions);
    for (size_t i= 0; i < solutions; i++)
    {
        rVecsSorted[i] = rVecs[y[i]];
        tVecsSorted[i] = tVecs[y[i]];

        sortedRmses[i] = rmses[y[i]];
    }
    rVecs = rVecsSorted;
    tVecs = tVecsSorted;
    rmses = sortedRmses;

}


int solvePnPGeneric( InputArray _opoints, InputArray _ipoints,
                     InputArray _cameraMatrix, InputArray _distCoeffs,
                     OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                     const Ptr<PnPSolver> solver, const Ptr<PnPRefiner> refiner,
                     bool sortOnReprojectionError,
                     OutputArray reprojectionError) {

    CV_INSTRUMENT_REGION();

    //parsing of _opoints, _ipoints, _cameraMatrix and _distCoeffs with format checking
    Mat opoints_ = _opoints.getMat(), ipoints_ = _ipoints.getMat();
    Mat opoints, ipoints, cameraMatrix, distCoeffs, ipointsNormalized;
    opoints_.convertTo(opoints, CV_64F);
    ipoints_.convertTo(ipoints, CV_64F);
    _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

    ipoints.copyTo(ipointsNormalized);
    if (opoints.channels()==1)
    {
        opoints = opoints.reshape(3);
    }

    if (!_distCoeffs.empty())
    {
        _distCoeffs.getMat().convertTo(distCoeffs, CV_64F);
        undistortPoints(ipoints, ipointsNormalized, cameraMatrix, distCoeffs);
    }
    else
    {
        undistortPoints(ipoints, ipointsNormalized, cameraMatrix, noArray());
    }

    //output pose solutions:
    std::vector<Mat> rVecs;
    std::vector<Mat> tVecs;

    //output depths:
    int depthRot = CV_64F;
    int depthTrans = CV_64F;

    if (_rvecs.fixedType() | !_rvecs.empty())
    {
        depthRot = _rvecs.depth();
    }
    if (_tvecs.fixedType() | !_tvecs.empty())
    {
        depthTrans = _tvecs.depth();
    }

    if (!solver.empty())
    {
        rVecs.clear();
        tVecs.clear();
        solver->solveProblem(opoints,ipointsNormalized,rVecs,tVecs,false);
    }
    else
    {
        //empty pnp solver is used, so copy poses from inputs and proceed
        const int n = _rvecs.rows()*_rvecs.cols();
        for (int i = 0; i < n; i++)
        {
            Mat r(3,1,CV_64F);
            Mat t(3,1,CV_64F);

            Mat r_,t_;

            if (_rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR)
            {
                Mat rref = _rvecs.getMat_();

                if (depthRot == CV_32F)
                    r_ = rref.at<Vec3f>(0,i);

                else
                    r_ = rref.at<Vec3d>(0,i);
            }
            else
            {
                r_ = _rvecs.getMatRef(i);
            }
            r_.convertTo(r,CV_64F);


            if (_tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR)
            {
                Mat tref = _tvecs.getMat_();

                if (depthTrans == CV_32F)
                    t_ = tref.at<Vec3f>(0,i);

                else
                    t_ = tref.at<Vec3d>(0,i);
            }
            else
            {
                t_ = _tvecs.getMatRef(i);
            }
            t_.convertTo(t,CV_64F);

            rVecs.push_back(r);
            tVecs.push_back(t);
        }

    }
    _rvecs.clear();
    _tvecs.clear();


    size_t solutions = rVecs.size();
    if (!refiner.empty())
    {
        //perform pose refinement
        std::vector<Mat> rVecs2;
        std::vector<Mat> tVecs2;

        for (size_t i = 0; i < solutions; i++)
        {

            Mat r = rVecs[i];
            Mat t = tVecs[i];
            std::vector<Mat> rvs;
            std::vector<Mat> tvs;
            refiner->refine(opoints,ipoints,cameraMatrix,distCoeffs,r,t,rvs,tvs);
            for (size_t j = 0; j < rvs.size(); j++)
            {
                rVecs2.push_back(rvs[j]);
                tVecs2.push_back(tvs[j]);
            }

        }
        rVecs = rVecs2;
        tVecs = tVecs2;
    }


    //handle sorting of solutions based on reprojection error:
    std::vector<double> rmses;
    if ((solutions>0) & (sortOnReprojectionError | reprojectionError.needed()))
    {
        sortPosesOnReprojectionError(opoints, ipoints, cameraMatrix, distCoeffs,  rVecs, tVecs, rmses);
    }

    //copying results to outputs
    _rvecs.create(static_cast<int>(solutions), 1, CV_MAKETYPE(depthRot, _rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));
    _tvecs.create(static_cast<int>(solutions), 1, CV_MAKETYPE(depthTrans, _tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR ? 3 : 1));

    for (size_t i = 0; i < solutions; i++)
    {
        Mat rvec0, tvec0;
        if (depthRot == CV_64F)
        {rvec0 = rVecs[i];}
        else
        {rVecs[i].convertTo(rvec0, depthRot);}

        //std::cout <<"rvec0 " << rvec0 << std::endl;

        if (depthTrans == CV_64F)
        {tvec0 = tVecs[i];}
        else
        {tVecs[i].convertTo(tvec0, depthTrans);}

        if (_rvecs.fixedType() && _rvecs.kind() == _InputArray::STD_VECTOR)
        {
            Mat rref = _rvecs.getMat_();

            if (_rvecs.depth() == CV_32F)
                rref.at<Vec3f>(0,static_cast<int>(i)) = Vec3f(rvec0.at<float>(0,0), rvec0.at<float>(1,0), rvec0.at<float>(2,0));
            else
                rref.at<Vec3d>(0,static_cast<int>(i)) = Vec3d(rvec0.at<double>(0,0), rvec0.at<double>(1,0), rvec0.at<double>(2,0));
        }
        else
        {
            _rvecs.getMatRef(static_cast<int>(i)) = rvec0;
        }

        if (_tvecs.fixedType() && _tvecs.kind() == _InputArray::STD_VECTOR)
        {

            Mat tref = _tvecs.getMat_();

            if (_tvecs.depth() == CV_32F)
                tref.at<Vec3f>(0,static_cast<int>(i)) = Vec3f(tvec0.at<float>(0,0), tvec0.at<float>(1,0), tvec0.at<float>(2,0));
            else
                tref.at<Vec3d>(0,static_cast<int>(i)) = Vec3d(tvec0.at<double>(0,0), tvec0.at<double>(1,0), tvec0.at<double>(2,0));
        }
        else
        {
            _tvecs.getMatRef(static_cast<int>(i)) = tvec0;
        }
    }

    if (reprojectionError.needed())
    {
        int type;
        if (!reprojectionError.empty())
        {
            type = reprojectionError.type();
        }
        else {
            type = CV_64FC1;
        }
        reprojectionError.create(static_cast<int>(solutions), 1, type);
        CV_CheckType(reprojectionError.type(), type == CV_32FC1 || type == CV_64FC1,
                     "Type of reprojectionError must be CV_32FC1 or CV_64FC1!");

        for (size_t i = 0; i <solutions; i++)
        {
            Mat err = reprojectionError.getMat();
            if (type == CV_32F)
            {
                err.at<float>(static_cast<int>(i)) = static_cast<float>(rmses[i]);
            }
            else
            {
                err.at<double>(static_cast<int>(i)) = rmses[i];
            }
        }
    }

    return static_cast<int>(solutions);
}


void cvtSolvePnPFlag(const SolvePnPMethod & flag, bool useExtrinsicGuess, Ptr<PnPSolver> & solver, Ptr<PnPRefiner> & refiner)
{
    solver = Ptr<PnPSolver>();
    refiner = Ptr<PnPRefiner>();

    if (useExtrinsicGuess)
    {
        refiner = PnPRefinerLM::create();
    }
    else
    {
        switch (flag)
        {

        case SOLVEPNP_ITERATIVE:
        {
            solver = PnPSolverAutoSelect1::create();
            refiner = PnPRefinerLM::create();
            break;
        }
        case SOLVEPNP_P3P:
        {
            solver = PnPSolverP3PComplete::create();
            break;
        }
        case SOLVEPNP_AP3P:
        {
            solver = PnPSolverAP3P::create();
            break;
        }
        case SOLVEPNP_DLS:
        {
            solver = PnPSolverEPnP3D::create();
            break;
        }
        case SOLVEPNP_UPNP:
        {
            solver = PnPSolverEPnP3D::create();
            break;
        }
        case SOLVEPNP_EPNP:
        {
            solver = PnPSolverEPnP3D::create();
            break;
        }

        case SOLVEPNP_IPPE:
        {
            solver = PnPSolverIPPE::create();
            break;
        }
        case SOLVEPNP_IPPE_SQUARE:
        {
            solver = PnPSolverIPPESquare::create();
            break;
        }

        case SOLVEPNP_MAX_COUNT:
        {

            break;
        }

        }
    }
}








void get3DPointsetShape(InputArray _objectPoints, InputOutputArray _singValues)
{
    CV_INSTRUMENT_REGION();
    Mat objectPoints;

    _singValues.create(3,1,CV_64FC1);
    Mat singValues = _singValues.getMat();

    if (_objectPoints.getMat().channels()!=3)
    {
        objectPoints = _objectPoints.getMat().reshape(3);
    }
    else
    {
        objectPoints = _objectPoints.getMat();
    }
    CV_CheckType(objectPoints.type(), objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3,
                 "Type of _objectPoints must be CV_32FC3 or CV_64FC3");

    bool floatDepth = objectPoints.type() == CV_32FC3;
    if (floatDepth)
    {
        objectPoints.convertTo(objectPoints, CV_64F);
    }


    Scalar meanValues = mean(objectPoints);
    int nbPts = objectPoints.checkVector(3, CV_64F);
    Mat objectPointsCentred = objectPoints - meanValues;
    objectPointsCentred = objectPointsCentred.reshape(1, nbPts);

    Mat w, u, vt;
    Mat MM = objectPointsCentred.t() * objectPointsCentred;
    SVDecomp(MM, w, u, vt);
    singValues.at<double>(0) = w.at<double>(0);
    singValues.at<double>(1) = w.at<double>(1);
    singValues.at<double>(2) = w.at<double>(2);
}
}






