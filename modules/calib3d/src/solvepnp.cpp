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
using namespace cv;

void cv::solvePnP( const Mat& opoints, const Mat& ipoints,
                   const Mat& cameraMatrix, const Mat& distCoeffs,
                   Mat& rvec, Mat& tvec, bool useExtrinsicGuess )
{
    CV_Assert(opoints.isContinuous() && opoints.depth() == CV_32F &&
              ((opoints.rows == 1 && opoints.channels() == 3) ||
               opoints.cols*opoints.channels() == 3) &&
              ipoints.isContinuous() && ipoints.depth() == CV_32F &&
              ((ipoints.rows == 1 && ipoints.channels() == 2) ||
              ipoints.cols*ipoints.channels() == 2));

    rvec.create(3, 1, CV_64F);
    tvec.create(3, 1, CV_64F);
    CvMat _objectPoints = opoints, _imagePoints = ipoints;
    CvMat _cameraMatrix = cameraMatrix, _distCoeffs = distCoeffs;
    CvMat _rvec = rvec, _tvec = tvec;
    cvFindExtrinsicCameraParams2(&_objectPoints, &_imagePoints, &_cameraMatrix,
                                 distCoeffs.data ? &_distCoeffs : 0,
                                 &_rvec, &_tvec, useExtrinsicGuess );
}

namespace cv
{
  namespace pnpransac
  {
    const int MIN_POINTS_COUNT = 4;

    void project3dPoints(const Mat& points, const Mat& rvec, const Mat& tvec, Mat& modif_points)
    {
      modif_points.create(1, points.cols, CV_32FC3);
      Mat R(3, 3, CV_64FC1);
      Rodrigues(rvec, R);
      Mat transformation(3, 4, CV_64F);
      Mat r = transformation.colRange(0, 2);
      R.copyTo(r);
      Mat t = transformation.colRange(3, 4);
      tvec.copyTo(t);
      transform(points, modif_points, transformation);
    }

    class Mutex
    {
    public:
        Mutex() {}
        void lock()
        {
        #ifdef HAVE_TBB
          slock.acquire(resultsMutex);
        #endif
        }

        void unlock()
        {
        #ifdef HAVE_TBB
          slock.release();
        #endif
        }

    private:
    #ifdef HAVE_TBB
      tbb::mutex resultsMutex;
      tbb::mutex::scoped_lock slock;
    #endif
    };

    struct CameraParameters
    {
        void init(Mat _intrinsics, Mat _distCoeffs)
        {
            _intrinsics.copyTo(intrinsics);
            _distCoeffs.copyTo(distortion);
        }

        Mat intrinsics;
        Mat distortion;
    };

    struct Parameters
    {
        int iterationsCount;
        float reprojectionError;
        int minInliersCount;
        bool useExtrinsicGuess;
        CameraParameters camera;
    };

    void pnpTask(const vector<char>& pointsMask, const Mat& objectPoints, const Mat& imagePoints,
                 const Parameters& params, vector<int>& inliers, Mat& rvec, Mat& tvec,
                 const Mat& rvecInit, const Mat& tvecInit, Mutex& resultsMutex)
    {
      Mat modelObjectPoints(1, MIN_POINTS_COUNT, CV_32FC3), modelImagePoints(1, MIN_POINTS_COUNT, CV_32FC2);
      for (size_t i = 0, colIndex = 0; i < pointsMask.size(); i++)
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
          if (norm(modelObjectPoints.at<Vec3f>(0, i) - modelObjectPoints.at<Vec3f>(0, j)) < eps)
            num_same_points++;
        }
      if (num_same_points > 0)
        return;

      Mat localRvec, localTvec;
      rvecInit.copyTo(localRvec);
      tvecInit.copyTo(localTvec);

      solvePnP(modelObjectPoints, modelImagePoints, params.camera.intrinsics, params.camera.distortion, localRvec, localTvec, params.useExtrinsicGuess);

      vector<Point2f> projected_points;
      projected_points.resize(objectPoints.cols);
      projectPoints(objectPoints, localRvec, localTvec, params.camera.intrinsics, params.camera.distortion, projected_points);

      Mat rotatedPoints;
      project3dPoints(objectPoints, localRvec, localTvec, rotatedPoints);

      vector<int> localInliers;
      for (size_t i = 0; i < objectPoints.cols; i++)
      {
        Point2f p(imagePoints.at<Vec2f>(0, i)[0], imagePoints.at<Vec2f>(0, i)[1]);
        if ((norm(p - projected_points[i]) < params.reprojectionError)
            && (rotatedPoints.at<Vec3f>(0, i)[2] > 0)) //hack
        {
          localInliers.push_back(i);
        }
      }

      if (localInliers.size() > inliers.size())
      {
        resultsMutex.lock();

        inliers.clear();
        inliers.resize(localInliers.size());
        memcpy(&inliers[0], &localInliers[0], sizeof(int) * localInliers.size());
        localRvec.copyTo(rvec);
        localTvec.copyTo(tvec);

        resultsMutex.unlock();
      }
    }

    class PnPSolver
    {
    public:
        void operator()( const BlockedRange& r ) const
        {
            vector<char> pointsMask(objectPoints.cols, 0);
            memset(&pointsMask[0], 1, MIN_POINTS_COUNT );
            for( size_t i=r.begin(); i!=r.end(); ++i )
            {
               generateVar(pointsMask);
               pnpTask(pointsMask, objectPoints, imagePoints, parameters,
                       inliers, rvec, tvec, initRvec, initTvec, syncMutex);
               if ((int)inliers.size() > parameters.minInliersCount)
               {
                    #ifdef HAVE_TBB
                      tbb::task::self().cancel_group_execution();
                    #else
                      break;
                     #endif
               }
            }
        }
        PnPSolver(const Mat& objectPoints, const Mat& imagePoints, const Parameters& parameters,
                  Mat& rvec, Mat& tvec, vector<int>& inliers):
                    objectPoints(objectPoints), imagePoints(imagePoints), parameters(parameters),
                    rvec(rvec), tvec(tvec), inliers(inliers)
        {
          rvec.copyTo(initRvec);
          tvec.copyTo(initTvec);
        }
    private:
        const Mat& objectPoints;
        const Mat& imagePoints;
        const Parameters& parameters;
        vector<int>& inliers;
        Mat &rvec, &tvec;
        Mat initRvec, initTvec;

        static RNG generator;
        static Mutex syncMutex;

        void generateVar(vector<char>& mask) const
        {
          size_t size = mask.size();
          for (size_t i = 0; i < size; i++)
          {
                int i1 = generator.uniform(0, size);
                int i2 = generator.uniform(0, size);
                char curr = mask[i1];
                mask[i1] = mask[i2];
                mask[i2] = curr;
          }
        }
    };

    Mutex PnPSolver::syncMutex;
    RNG PnPSolver::generator;

  }
}

void cv::solvePnPRansac(const Mat& opoints, const Mat& ipoints,
                    const Mat& cameraMatrix, const Mat& distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess,
                    int iterationsCount, float reprojectionError, int minInliersCount, vector<int>* inliers)
{

  CV_Assert(opoints.isContinuous());
  CV_Assert(opoints.depth() == CV_32F);
  CV_Assert((opoints.rows == 1 && opoints.channels() == 3) || opoints.cols*opoints.channels() == 3);
  CV_Assert(ipoints.isContinuous());
  CV_Assert(ipoints.depth() == CV_32F);
  CV_Assert((ipoints.rows == 1 && ipoints.channels() == 2) || ipoints.cols*ipoints.channels() == 2);

  rvec.create(3, 1, CV_64FC1);
  tvec.create(3, 1, CV_64FC1);

  Mat objectPoints = opoints.reshape(3, 1), imagePoints = ipoints.reshape(2, 1);

  if (minInliersCount <= 0)
    minInliersCount = objectPoints.cols;
  cv::pnpransac::Parameters params;
  params.iterationsCount = iterationsCount;
  params.minInliersCount = minInliersCount;
  params.reprojectionError = reprojectionError;
  params.useExtrinsicGuess = useExtrinsicGuess;
  params.camera.init(cameraMatrix, distCoeffs);

  vector<int> localInliers;
  Mat localRvec, localTvec;
  rvec.copyTo(localRvec);
  tvec.copyTo(localTvec);

  if (objectPoints.cols >= pnpransac::MIN_POINTS_COUNT)
  {
     parallel_for(BlockedRange(0,iterationsCount), cv::pnpransac::PnPSolver(objectPoints, imagePoints, params,
        localRvec, localTvec, localInliers));
  }

  if (localInliers.size() >= pnpransac::MIN_POINTS_COUNT)
  {
    size_t pointsCount = localInliers.size();
    Mat inlierObjectPoints(1, pointsCount, CV_32FC3), inlierImagePoints(1, pointsCount, CV_32FC2);
    int index;
    for (size_t i = 0; i < localInliers.size(); i++)
    {
      index = localInliers[i];
      Mat colInlierImagePoints = inlierImagePoints(Rect(i, 0, 1, 1));
      imagePoints.col(index).copyTo(colInlierImagePoints);
      Mat colInlierObjectPoints = inlierObjectPoints(Rect(i, 0, 1, 1));
      objectPoints.col(index).copyTo(colInlierObjectPoints);
    }
    solvePnP(inlierObjectPoints, inlierImagePoints, params.camera.intrinsics, params.camera.distortion, localRvec, localTvec, true);
    localRvec.copyTo(rvec);
    localTvec.copyTo(tvec);
    if (inliers)
       *inliers = localInliers;
  }
  else
  {
    tvec.setTo(Scalar(0));
    Mat R = Mat::ones(3, 3, CV_64F);
    Rodrigues(R, rvec);
  }
  return;
}

