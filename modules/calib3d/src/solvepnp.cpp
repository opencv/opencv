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
#include "calib3d_c_api.h"

#include <iostream>

namespace cv
{
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

bool solvePnP( InputArray _opoints, InputArray _ipoints,
               InputArray _cameraMatrix, InputArray _distCoeffs,
               OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess, int flags )
{
    CV_INSTRUMENT_REGION();

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( ( (npoints >= 4) || (npoints == 3 && flags == SOLVEPNP_ITERATIVE && useExtrinsicGuess) )
               && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

    Mat rvec, tvec;
    if( flags != SOLVEPNP_ITERATIVE )
        useExtrinsicGuess = false;

    if( useExtrinsicGuess )
    {
        int rtype = _rvec.type(), ttype = _tvec.type();
        Size rsize = _rvec.size(), tsize = _tvec.size();
        CV_Assert( (rtype == CV_32F || rtype == CV_64F) &&
                   (ttype == CV_32F || ttype == CV_64F) );
        CV_Assert( (rsize == Size(1, 3) || rsize == Size(3, 1)) &&
                   (tsize == Size(1, 3) || tsize == Size(3, 1)) );
    }
    else
    {
        int mtype = CV_64F;
        // use CV_32F if all PnP inputs are CV_32F and outputs are empty
        if (_ipoints.depth() == _cameraMatrix.depth() && _ipoints.depth() == _opoints.depth() &&
            _rvec.empty() && _tvec.empty())
            mtype = _opoints.depth();

        _rvec.create(3, 1, mtype);
        _tvec.create(3, 1, mtype);
    }
    rvec = _rvec.getMat();
    tvec = _tvec.getMat();

    Mat cameraMatrix0 = _cameraMatrix.getMat();
    Mat distCoeffs0 = _distCoeffs.getMat();
    Mat cameraMatrix = Mat_<double>(cameraMatrix0);
    Mat distCoeffs = Mat_<double>(distCoeffs0);
    bool result = false;

    if (flags == SOLVEPNP_EPNP || flags == SOLVEPNP_DLS || flags == SOLVEPNP_UPNP)
    {
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        epnp PnP(cameraMatrix, opoints, undistortedPoints);

        Mat R;
        PnP.compute_pose(R, tvec);
        Rodrigues(R, rvec);
        result = true;
    }
    else if (flags == SOLVEPNP_P3P)
    {
        CV_Assert( npoints == 4);
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        p3p P3Psolver(cameraMatrix);

        Mat R;
        result = P3Psolver.solve(R, tvec, opoints, undistortedPoints);
        if (result)
            Rodrigues(R, rvec);
    }
    else if (flags == SOLVEPNP_AP3P)
    {
        CV_Assert( npoints == 4);
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        ap3p P3Psolver(cameraMatrix);

        Mat R;
        result = P3Psolver.solve(R, tvec, opoints, undistortedPoints);
        if (result)
            Rodrigues(R, rvec);
    }
    else if (flags == SOLVEPNP_ITERATIVE)
    {
        CvMat c_objectPoints = cvMat(opoints), c_imagePoints = cvMat(ipoints);
        CvMat c_cameraMatrix = cvMat(cameraMatrix), c_distCoeffs = cvMat(distCoeffs);
        CvMat c_rvec = cvMat(rvec), c_tvec = cvMat(tvec);
        cvFindExtrinsicCameraParams2(&c_objectPoints, &c_imagePoints, &c_cameraMatrix,
                                     (c_distCoeffs.rows && c_distCoeffs.cols) ? &c_distCoeffs : 0,
                                     &c_rvec, &c_tvec, useExtrinsicGuess );
        result = true;
    }
    /*else if (flags == SOLVEPNP_DLS)
    {
        Mat undistortedPoints;
        undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);

        dls PnP(opoints, undistortedPoints);

        Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        bool result = PnP.compute_pose(R, tvec);
        if (result)
            Rodrigues(R, rvec);
        return result;
    }
    else if (flags == SOLVEPNP_UPNP)
    {
        upnp PnP(cameraMatrix, opoints, ipoints);

        Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        PnP.compute_pose(R, tvec);
        Rodrigues(R, rvec);
        return true;
    }*/
    else
        CV_Error(CV_StsBadArg, "The flags argument must be one of SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, SOLVEPNP_EPNP or SOLVEPNP_DLS");
    return result;
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

        bool correspondence = solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                            rvec, tvec, useExtrinsicGuess, flags );

        Mat _local_model;
        hconcat(rvec, tvec, _local_model);
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

    vector<Point3d> opoints_inliers;
    vector<Point2d> ipoints_inliers;
    opoints = opoints.reshape(3);
    ipoints = ipoints.reshape(2);
    opoints.convertTo(opoints_inliers, CV_64F);
    ipoints.convertTo(ipoints_inliers, CV_64F);

    const uchar* mask = _mask_local_inliers.ptr<uchar>();
    int npoints1 = compressElems(&opoints_inliers[0], mask, 1, npoints);
    compressElems(&ipoints_inliers[0], mask, 1, npoints);

    opoints_inliers.resize(npoints1);
    ipoints_inliers.resize(npoints1);
    result = solvePnP(opoints_inliers, ipoints_inliers, cameraMatrix,
                      distCoeffs, rvec, tvec, useExtrinsicGuess,
                      (flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P) ? SOLVEPNP_EPNP : flags) ? 1 : -1;

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

int solveP3P( InputArray _opoints, InputArray _ipoints,
              InputArray _cameraMatrix, InputArray _distCoeffs,
              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags) {
    CV_INSTRUMENT_REGION();

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints == 3 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );
    CV_Assert( flags == SOLVEPNP_P3P || flags == SOLVEPNP_AP3P );

    Mat cameraMatrix0 = _cameraMatrix.getMat();
    Mat distCoeffs0 = _distCoeffs.getMat();
    Mat cameraMatrix = Mat_<double>(cameraMatrix0);
    Mat distCoeffs = Mat_<double>(distCoeffs0);

    Mat undistortedPoints;
    undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
    std::vector<Mat> Rs, ts;

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

    if (_rvecs.needed()) {
        _rvecs.create(solutions, 1, CV_64F);
    }

    if (_tvecs.needed()) {
        _tvecs.create(solutions, 1, CV_64F);
    }

    for (int i = 0; i < solutions; i++) {
        Mat rvec;
        Rodrigues(Rs[i], rvec);
        _tvecs.getMatRef(i) = ts[i];
        _rvecs.getMatRef(i) = rvec;
    }

    return solutions;
}

}
