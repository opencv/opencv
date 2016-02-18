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
#include "opencv2/calib3d/calib3d_c.h"

#include <iostream>
using namespace cv;

bool cv::solvePnP( InputArray _opoints, InputArray _ipoints,
                  InputArray _cameraMatrix, InputArray _distCoeffs,
                  OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess, int flags )
{
    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints >= 0 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );
    _rvec.create(3, 1, CV_64F);
    _tvec.create(3, 1, CV_64F);
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();

    if (flags == SOLVEPNP_EPNP)
    {
        cv::Mat undistortedPoints;
        cv::undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        epnp PnP(cameraMatrix, opoints, undistortedPoints);

        cv::Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        PnP.compute_pose(R, tvec);
        cv::Rodrigues(R, rvec);
        return true;
    }
    else if (flags == SOLVEPNP_P3P)
    {
        CV_Assert( npoints == 4);
        cv::Mat undistortedPoints;
        cv::undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);
        p3p P3Psolver(cameraMatrix);

        cv::Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        bool result = P3Psolver.solve(R, tvec, opoints, undistortedPoints);
        if (result)
            cv::Rodrigues(R, rvec);
        return result;
    }
    else if (flags == SOLVEPNP_ITERATIVE)
    {
        CvMat c_objectPoints = opoints, c_imagePoints = ipoints;
        CvMat c_cameraMatrix = cameraMatrix, c_distCoeffs = distCoeffs;
        CvMat c_rvec = _rvec.getMat(), c_tvec = _tvec.getMat();
        cvFindExtrinsicCameraParams2(&c_objectPoints, &c_imagePoints, &c_cameraMatrix,
                                     c_distCoeffs.rows*c_distCoeffs.cols ? &c_distCoeffs : 0,
                                     &c_rvec, &c_tvec, useExtrinsicGuess );
        return true;
    }
    else if (flags == SOLVEPNP_DLS)
    {
        cv::Mat undistortedPoints;
        cv::undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs);

        dls PnP(opoints, undistortedPoints);

        cv::Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        bool result = PnP.compute_pose(R, tvec);
        if (result)
            cv::Rodrigues(R, rvec);
        return result;
    }
    else if (flags == SOLVEPNP_UPNP)
    {
        upnp PnP(cameraMatrix, opoints, ipoints);

        cv::Mat R, rvec = _rvec.getMat(), tvec = _tvec.getMat();
        double f = PnP.compute_pose(R, tvec);
        cv::Rodrigues(R, rvec);
        if(cameraMatrix.type() == CV_32F)
            cameraMatrix.at<float>(0,0) = cameraMatrix.at<float>(1,1) = (float)f;
        else
            cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = f;
        return true;
    }
    else
        CV_Error(CV_StsBadArg, "The flags argument must be one of SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, SOLVEPNP_EPNP or SOLVEPNP_DLS");
    return false;
}

class PnPRansacCallback : public PointSetRegistrator::Callback
{

public:

    PnPRansacCallback(Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), int _flags=cv::SOLVEPNP_ITERATIVE,
            bool _useExtrinsicGuess=false, Mat _rvec=Mat(), Mat _tvec=Mat() )
        : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), flags(_flags), useExtrinsicGuess(_useExtrinsicGuess),
          rvec(_rvec), tvec(_tvec) {}

    /* Pre: True */
    /* Post: compute _model with given points an return number of found models */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat opoints = _m1.getMat(), ipoints = _m2.getMat();

<<<<<<< HEAD

        bool correspondence = cv::solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                            rvec, tvec, useExtrinsicGuess, flags );
=======
        struct Parameters
        {
            int iterationsCount;
            float reprojectionError;
            int minInliersCount;
            bool useExtrinsicGuess;
            int flags;
            CameraParameters camera;
        };

        template <typename OpointType, typename IpointType>
        static void pnpTask(const vector<char>& pointsMask, const Mat& objectPoints, const Mat& imagePoints,
                     const Parameters& params, vector<int>& inliers, Mat& rvec, Mat& tvec,
                     const Mat& rvecInit, const Mat& tvecInit, Mutex& resultsMutex)
        {
            Mat modelObjectPoints(1, MIN_POINTS_COUNT, CV_MAKETYPE(DataDepth<OpointType>::value, 3));
            Mat modelImagePoints(1, MIN_POINTS_COUNT, CV_MAKETYPE(DataDepth<IpointType>::value, 2));
            for (int i = 0, colIndex = 0; i < (int)pointsMask.size(); i++)
            {
                if (pointsMask[i])
                {
                    Mat colModelImagePoints = modelImagePoints(Rect(colIndex, 0, 1, 1));
                    imagePoints.col(i).copyTo(colModelImagePoints);
                    Mat colModelObjectPoints = modelObjectPoints(Rect(colIndex, 0, 1, 1));
                    objectPoints.col(i).copyTo(colModelObjectPoints);
                    colIndex = colIndex+1;
                }
            }

            //filter same 3d points, hang in solvePnP
            double eps = 1e-10;
            int num_same_points = 0;
            for (int i = 0; i < MIN_POINTS_COUNT; i++)
                for (int j = i + 1; j < MIN_POINTS_COUNT; j++)
                {
                    if (norm(modelObjectPoints.at<Vec<OpointType,3> >(0, i) - modelObjectPoints.at<Vec<OpointType,3> >(0, j)) < eps)
                        num_same_points++;
                }
            if (num_same_points > 0)
                return;
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd

        Mat _local_model;
        cv::hconcat(rvec, tvec, _local_model);
        _local_model.copyTo(_model);

        return correspondence;
    }

    /* Pre: True */
    /* Post: fill _err with projection errors */
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {

<<<<<<< HEAD
        Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();
=======
            vector<Point_<OpointType> > projected_points;
            projected_points.resize(objectPoints.cols);
            projectPoints(objectPoints, localRvec, localTvec, params.camera.intrinsics, params.camera.distortion, projected_points);
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd

        int i, count = opoints.cols;
        Mat _rvec = model.col(0);
        Mat _tvec = model.col(1);

<<<<<<< HEAD
=======
            vector<int> localInliers;
            for (int i = 0; i < objectPoints.cols; i++)
            {
                //Although p is a 2D point it needs the same type as the object points to enable the norm calculation
                Point_<OpointType> p((OpointType)imagePoints.at<Vec<IpointType,2> >(0, i)[0],
                                     (OpointType)imagePoints.at<Vec<IpointType,2> >(0, i)[1]);
                if ((norm(p - projected_points[i]) < params.reprojectionError)
                    && (rotatedPoints.at<Vec<OpointType,3> >(0, i)[2] > 0)) //hack
                {
                    localInliers.push_back(i);
                }
            }
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd

        Mat projpoints(count, 2, CV_32FC1);
        cv::projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

        const Point2f* ipoints_ptr = ipoints.ptr<Point2f>();
        const Point2f* projpoints_ptr = projpoints.ptr<Point2f>();

        _err.create(count, 1, CV_32FC1);
        float* err = _err.getMat().ptr<float>();

<<<<<<< HEAD
        for ( i = 0; i < count; ++i)
            err[i] = (float)cv::norm( ipoints_ptr[i] - projpoints_ptr[i] );
=======
        static void pnpTask(const vector<char>& pointsMask, const Mat& objectPoints, const Mat& imagePoints,
            const Parameters& params, vector<int>& inliers, Mat& rvec, Mat& tvec,
            const Mat& rvecInit, const Mat& tvecInit, Mutex& resultsMutex)
        {
            CV_Assert(objectPoints.depth() == CV_64F ||  objectPoints.depth() == CV_32F);
            CV_Assert(imagePoints.depth() == CV_64F ||  imagePoints.depth() == CV_32F);
            const bool objectDoublePrecision = objectPoints.depth() == CV_64F;
            const bool imageDoublePrecision = imagePoints.depth() == CV_64F;
            if(objectDoublePrecision)
            {
                if(imageDoublePrecision)
                    pnpTask<double, double>(pointsMask, objectPoints, imagePoints, params, inliers, rvec, tvec, rvecInit, tvecInit, resultsMutex);
                else
                    pnpTask<double, float>(pointsMask, objectPoints, imagePoints, params, inliers, rvec, tvec, rvecInit, tvecInit, resultsMutex);
            }
            else
            {
                if(imageDoublePrecision)
                    pnpTask<float, double>(pointsMask, objectPoints, imagePoints, params, inliers, rvec, tvec, rvecInit, tvecInit, resultsMutex);
                else
                    pnpTask<float, float>(pointsMask, objectPoints, imagePoints, params, inliers, rvec, tvec, rvecInit, tvecInit, resultsMutex);
            }
        }

        class PnPSolver
        {
        public:
            void operator()( const BlockedRange& r ) const
            {
                vector<char> pointsMask(objectPoints.cols, 0);
                memset(&pointsMask[0], 1, MIN_POINTS_COUNT );
                for( int i=r.begin(); i!=r.end(); ++i )
                {
                    generateVar(pointsMask);
                    pnpTask(pointsMask, objectPoints, imagePoints, parameters,
                            inliers, rvec, tvec, initRvec, initTvec, syncMutex);
                    if ((int)inliers.size() >= parameters.minInliersCount)
                    {
#ifdef HAVE_TBB
                        tbb::task::self().cancel_group_execution();
#else
                        break;
#endif
                    }
                }
            }
            PnPSolver(const Mat& _objectPoints, const Mat& _imagePoints, const Parameters& _parameters,
                      Mat& _rvec, Mat& _tvec, vector<int>& _inliers):
            objectPoints(_objectPoints), imagePoints(_imagePoints), parameters(_parameters),
            rvec(_rvec), tvec(_tvec), inliers(_inliers)
            {
                rvec.copyTo(initRvec);
                tvec.copyTo(initTvec);

                generator.state = theRNG().state; //to control it somehow...
            }
        private:
            PnPSolver& operator=(const PnPSolver&);

            const Mat& objectPoints;
            const Mat& imagePoints;
            const Parameters& parameters;
            Mat &rvec, &tvec;
            vector<int>& inliers;
            Mat initRvec, initTvec;
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd

    }


    Mat cameraMatrix;
    Mat distCoeffs;
    int flags;
    bool useExtrinsicGuess;
    Mat rvec;
    Mat tvec;
};

bool cv::solvePnPRansac(InputArray _opoints, InputArray _ipoints,
                        InputArray _cameraMatrix, InputArray _distCoeffs,
                        OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess,
                        int iterationsCount, float reprojectionError, double confidence,
                        OutputArray _inliers, int flags)
{

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();

    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints >= 0 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

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

    Ptr<PointSetRegistrator::Callback> cb; // pointer to callback
    cb = makePtr<PnPRansacCallback>( cameraMatrix, distCoeffs, flags, useExtrinsicGuess, rvec, tvec);

    int model_points = 4;                             // minimum of number of model points
    if( flags == cv::SOLVEPNP_ITERATIVE ) model_points = 6;
    else if( flags == cv::SOLVEPNP_UPNP ) model_points = 6;
    else if( flags == cv::SOLVEPNP_EPNP ) model_points = 5;

    double param1 = reprojectionError;                // reprojection error
    double param2 = confidence;                       // confidence
    int param3 = iterationsCount;                     // number maximum iterations

    cv::Mat _local_model(3, 2, CV_64FC1);
    cv::Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);

    // call Ransac
    int result = createRANSACPointSetRegistrator(cb, model_points, param1, param2, param3)->run(opoints, ipoints, _local_model, _mask_local_inliers);

    if( result <= 0 || _local_model.rows <= 0)
    {
        _rvec.assign(rvec);    // output rotation vector
        _tvec.assign(tvec);    // output translation vector

        if( _inliers.needed() )
            _inliers.release();

        return false;
    }
    else
    {
        _rvec.assign(_local_model.col(0));    // output rotation vector
        _tvec.assign(_local_model.col(1));    // output translation vector
    }

    if(_inliers.needed())
    {
        Mat _local_inliers;
        int count = 0;
        for (int i = 0; i < _mask_local_inliers.rows; ++i)
        {
<<<<<<< HEAD
            if((int)_mask_local_inliers.at<uchar>(i) == 1) // inliers mask
=======
            int i, pointsCount = (int)localInliers.size();
            Mat inlierObjectPoints(1, pointsCount, CV_MAKE_TYPE(opoints.depth(), 3)), inlierImagePoints(1, pointsCount, CV_MAKE_TYPE(ipoints.depth(), 2));
            for (i = 0; i < pointsCount; i++)
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd
            {
                _local_inliers.push_back(count);    // output inliers vector
                count++;
            }
        }
        _local_inliers.copyTo(_inliers);
    }
    return true;
}
