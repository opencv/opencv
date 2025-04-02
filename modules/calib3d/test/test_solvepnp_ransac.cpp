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

#include "test_precomp.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace opencv_test { namespace {

//Statistics Helpers
struct ErrorInfo
{
    ErrorInfo(double errT, double errR) : errorTrans(errT), errorRot(errR)
    {
    }

    bool operator<(const ErrorInfo& e) const
    {
        return sqrt(errorTrans*errorTrans + errorRot*errorRot) <
                sqrt(e.errorTrans*e.errorTrans + e.errorRot*e.errorRot);
    }

    double errorTrans;
    double errorRot;
};

//Try to find the translation and rotation thresholds to achieve a predefined percentage of success.
//Since a success is defined by error_trans < trans_thresh && error_rot < rot_thresh
//this just gives an idea of the values to use
static void findThreshold(const std::vector<double>& v_trans, const std::vector<double>& v_rot, double percentage,
                          double& transThresh, double& rotThresh)
{
    if (v_trans.empty() || v_rot.empty() || v_trans.size() != v_rot.size())
    {
        transThresh = -1;
        rotThresh = -1;
        return;
    }

    std::vector<ErrorInfo> error_info;
    error_info.reserve(v_trans.size());
    for (size_t i = 0; i < v_trans.size(); i++)
    {
        error_info.push_back(ErrorInfo(v_trans[i], v_rot[i]));
    }

    std::sort(error_info.begin(), error_info.end());
    size_t idx = static_cast<size_t>(error_info.size() * percentage);
    transThresh = error_info[idx].errorTrans;
    rotThresh = error_info[idx].errorRot;
}

static double getMax(const std::vector<double>& v)
{
    return *std::max_element(v.begin(), v.end());
}

static double getMean(const std::vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

static double getMedian(const std::vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    std::vector<double> v_copy = v;
    size_t size = v_copy.size();

    size_t n = size / 2;
    std::nth_element(v_copy.begin(), v_copy.begin() + n, v_copy.end());
    double val_n = v_copy[n];

    if (size % 2 == 1)
    {
        return val_n;
    } else
    {
        std::nth_element(v_copy.begin(), v_copy.begin() + n - 1, v_copy.end());
        return 0.5 * (val_n + v_copy[n - 1]);
    }
}

static void generatePose(const vector<Point3d>& points, Mat& rvec, Mat& tvec, RNG& rng, int nbTrials=10)
{
    const double minVal = 1.0e-3;
    const double maxVal = 1.0;
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);

    bool validPose = false;
    for (int trial = 0; trial < nbTrials && !validPose; trial++)
    {
        for (int i = 0; i < 3; i++)
        {
            rvec.at<double>(i,0) = rng.uniform(minVal, maxVal);
            tvec.at<double>(i,0) = (i == 2) ? rng.uniform(minVal*10, maxVal) : rng.uniform(-maxVal, maxVal);
        }

        Mat R;
        cv::Rodrigues(rvec, R);
        bool positiveDepth = true;
        for (size_t i = 0; i < points.size() && positiveDepth; i++)
        {
            Matx31d objPts(points[i].x, points[i].y, points[i].z);
            Mat camPts = R*objPts + tvec;
            if (camPts.at<double>(2,0) <= 0)
            {
                positiveDepth = false;
            }
        }
        validPose = positiveDepth;
    }
}

static void generatePose(const vector<Point3f>& points, Mat& rvec, Mat& tvec, RNG& rng, int nbTrials=10)
{
    vector<Point3d> points_double(points.size());

    for (size_t i = 0; i < points.size(); i++)
    {
        points_double[i] = Point3d(points[i].x, points[i].y, points[i].z);
    }

    generatePose(points_double, rvec, tvec, rng, nbTrials);
}

static std::string printMethod(int method)
{
    switch (method) {
    case 0:
        return "SOLVEPNP_ITERATIVE";
    case 1:
        return "SOLVEPNP_EPNP";
    case 2:
        return "SOLVEPNP_P3P";
    case 3:
        return "SOLVEPNP_DLS (remapped to SOLVEPNP_EPNP)";
    case 4:
        return "SOLVEPNP_UPNP (remapped to SOLVEPNP_EPNP)";
    case 5:
        return "SOLVEPNP_AP3P";
    case 6:
        return "SOLVEPNP_IPPE";
    case 7:
        return "SOLVEPNP_IPPE_SQUARE";
    case 8:
        return "SOLVEPNP_SQPNP";
    default:
        return "Unknown value";
    }
}

class CV_solvePnPRansac_Test : public cvtest::BaseTest
{
public:
    CV_solvePnPRansac_Test(bool planar_=false, bool planarTag_=false) : planar(planar_), planarTag(planarTag_)
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-2;
        eps[SOLVEPNP_EPNP] = 1.0e-2;
        eps[SOLVEPNP_P3P] = 1.0e-2;
        eps[SOLVEPNP_AP3P] = 1.0e-2;
        eps[SOLVEPNP_DLS] = 1.0e-2; // DLS is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_UPNP] = 1.0e-2; // UPnP is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_IPPE] = 1.0e-2;
        eps[SOLVEPNP_IPPE_SQUARE] = 1.0e-2;
        eps[SOLVEPNP_SQPNP] = 1.0e-2;

        totalTestsCount = 1000;

        if (planar || planarTag)
        {
            if (planarTag)
            {
                pointsCount = 4;
            }
            else
            {
                pointsCount = 30;
            }
        }
        else
        {
            pointsCount = 500;
        }
    }
    ~CV_solvePnPRansac_Test() {}
protected:
    void generate3DPointCloud(vector<Point3f>& points,
                              Point3f pmin = Point3f(-1, -1, 5),
                              Point3f pmax = Point3f(1, 1, 10))
    {
        RNG& rng = theRNG(); // fix the seed to use "fixed" input 3D points

        for (size_t i = 0; i < points.size(); i++)
        {
            float _x = rng.uniform(pmin.x, pmax.x);
            float _y = rng.uniform(pmin.y, pmax.y);
            float _z = rng.uniform(pmin.z, pmax.z);
            points[i] = Point3f(_x, _y, _z);
        }
    }

    void generatePlanarPointCloud(vector<Point3f>& points,
                                  Point2f pmin = Point2f(-1, -1),
                                  Point2f pmax = Point2f(1, 1))
    {
        RNG& rng = theRNG(); // fix the seed to use "fixed" input 3D points

        if (planarTag)
        {
            const float squareLength_2 = rng.uniform(0.01f, pmax.x) / 2;
            points.clear();
            points.push_back(Point3f(-squareLength_2, squareLength_2, 0));
            points.push_back(Point3f(squareLength_2, squareLength_2, 0));
            points.push_back(Point3f(squareLength_2, -squareLength_2, 0));
            points.push_back(Point3f(-squareLength_2, -squareLength_2, 0));
        }
        else
        {
            Mat rvec_double, tvec_double;
            generatePose(points, rvec_double, tvec_double, rng);

            Mat rvec, tvec, R;
            rvec_double.convertTo(rvec, CV_32F);
            tvec_double.convertTo(tvec, CV_32F);
            cv::Rodrigues(rvec, R);

            for (size_t i = 0; i < points.size(); i++)
            {
                float x = rng.uniform(pmin.x, pmax.x);
                float y = rng.uniform(pmin.y, pmax.y);
                float z = 0;

                Matx31f pt(x, y, z);
                Mat pt_trans = R * pt + tvec;
                points[i] = Point3f(pt_trans.at<float>(0,0), pt_trans.at<float>(1,0), pt_trans.at<float>(2,0));
            }
        }
    }

    void generateCameraMatrix(Mat& cameraMatrix, RNG& rng)
    {
        const double fcMinVal = 1e-3;
        const double fcMaxVal = 100;
        cameraMatrix.create(3, 3, CV_64FC1);
        cameraMatrix.setTo(Scalar(0));
        cameraMatrix.at<double>(0,0) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,1) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(0,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(2,2) = 1;
    }

    void generateDistCoeffs(Mat& distCoeffs, RNG& rng)
    {
        distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
            distCoeffs.at<double>(i,0) = rng.uniform(0.0, 1.0e-6);
    }

    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        Mat rvec, tvec;
        vector<int> inliers;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        //UPnP is mapped to EPnP
        //Uncomment this when UPnP is fixed
//        if (method == SOLVEPNP_UPNP)
//        {
//            intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
//        }
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }

        generatePose(points, trueRvec, trueTvec, rng);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(points.size());
        projectPoints(points, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        size_t numOutliers = 0;
        for (size_t i = 0; i < projectedPoints.size(); i++)
        {
            if (!planarTag && rng.uniform(0., 1.) > 0.95)
            {
                projectedPoints[i] = projectedPoints[rng.uniform(0,(int)points.size()-1)];
                numOutliers++;
            }
        }

        solvePnPRansac(points, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, pointsCount, 0.5f, 0.99, inliers, method);

        bool isTestSuccess = inliers.size() + numOutliers >= points.size();

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        isTestSuccess = isTestSuccess && rvecDiff < eps[method] && tvecDiff < eps[method];
        errorTrans = tvecDiff;
        errorRot = rvecDiff;

        return isTestSuccess;
    }

    virtual void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points, points_dls;
        points.resize(static_cast<size_t>(pointsCount));

        if (planar || planarTag)
        {
            generatePlanarPointCloud(points);
        }
        else
        {
            generate3DPointCloud(points);
        }

        RNG& rng = ts->get_rng();

        for (int mode = 0; mode < 2; mode++)
        {
            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                // SOLVEPNP_IPPE need planar object
                if (!planar && method == SOLVEPNP_IPPE)
                {
                    cout << "mode: " << printMode(mode) << ", method: " << printMethod(method) << " -> "
                         << "Skip for non-planar object" << endl;
                    continue;
                }

                // SOLVEPNP_IPPE_SQUARE need planar tag object
                if (!planarTag && method == SOLVEPNP_IPPE_SQUARE)
                {
                    cout << "mode: " << printMode(mode) << ", method: " << printMethod(method) << " -> "
                         << "Skip for non-planar tag object" << endl;
                    continue;
                }

                //To get the same input for each methods
                RNG rngCopy = rng;
                std::vector<double> vec_errorTrans, vec_errorRot;
                vec_errorTrans.reserve(static_cast<size_t>(totalTestsCount));
                vec_errorRot.reserve(static_cast<size_t>(totalTestsCount));

                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    double errorTrans, errorRot;
                    if (runTest(rngCopy, mode, method, points, errorTrans, errorRot))
                    {
                        successfulTestsCount++;
                    }
                    vec_errorTrans.push_back(errorTrans);
                    vec_errorRot.push_back(errorRot);
                }

                double maxErrorTrans = getMax(vec_errorTrans);
                double maxErrorRot = getMax(vec_errorRot);
                double meanErrorTrans = getMean(vec_errorTrans);
                double meanErrorRot = getMean(vec_errorRot);
                double medianErrorTrans = getMedian(vec_errorTrans);
                double medianErrorRot = getMedian(vec_errorRot);

                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy for %s, failed %d tests from %d, %s, "
                                                "maxErrT: %f, maxErrR: %f, "
                                                "meanErrT: %f, meanErrR: %f, "
                                                "medErrT: %f, medErrR: %f\n",
                               printMethod(method).c_str(), totalTestsCount - successfulTestsCount, totalTestsCount, printMode(mode).c_str(),
                               maxErrorTrans, maxErrorRot, meanErrorTrans, meanErrorRot, medianErrorTrans, medianErrorRot);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
                cout << "mode: " << printMode(mode) << ", method: " << printMethod(method) << " -> "
                     << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
                     << " (maxErrT: " << maxErrorTrans << ", maxErrR: " << maxErrorRot
                     << ", meanErrT: " << meanErrorTrans << ", meanErrR: " << meanErrorRot
                     << ", medErrT: " << medianErrorTrans << ", medErrR: " << medianErrorRot << ")" << endl;
                double transThres, rotThresh;
                findThreshold(vec_errorTrans, vec_errorRot, 0.7, transThres, rotThresh);
                cout << "approximate translation threshold for 0.7: " << transThres
                     << ", approximate rotation threshold for 0.7: " << rotThresh << endl;
            }
            cout << endl;
        }
    }
    std::string printMode(int mode)
    {
        switch (mode) {
        case 0:
            return "no distortion";
        case 1:
        default:
            return "distorsion";
        }
    }
    double eps[SOLVEPNP_MAX_COUNT];
    int totalTestsCount;
    int pointsCount;
    bool planar;
    bool planarTag;
};

class CV_solvePnP_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solvePnP_Test(bool planar_=false, bool planarTag_=false) : CV_solvePnPRansac_Test(planar_, planarTag_)
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-6;
        eps[SOLVEPNP_EPNP] = 1.0e-6;
        eps[SOLVEPNP_P3P] = 2.0e-4;
        eps[SOLVEPNP_AP3P] = 1.0e-4;
        eps[SOLVEPNP_DLS] = 1.0e-6; // DLS is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_UPNP] = 1.0e-6; // UPnP is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_IPPE] = 1.0e-6;
        eps[SOLVEPNP_IPPE_SQUARE] = 1.0e-6;
        eps[SOLVEPNP_SQPNP] = 1.0e-6;

        totalTestsCount = 1000;

        if (planar || planarTag)
        {
            if (planarTag)
            {
                pointsCount = 4;
            }
            else
            {
                pointsCount = 30;
            }
        }
        else
        {
            pointsCount = 500;
        }
    }

    ~CV_solvePnP_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        //Tune thresholds...
        double epsilon_trans[SOLVEPNP_MAX_COUNT];
        memcpy(epsilon_trans, eps, SOLVEPNP_MAX_COUNT * sizeof(*epsilon_trans));

        double epsilon_rot[SOLVEPNP_MAX_COUNT];
        memcpy(epsilon_rot, eps, SOLVEPNP_MAX_COUNT * sizeof(*epsilon_rot));

        if (planar)
        {
            if (mode == 0)
            {
                epsilon_trans[SOLVEPNP_EPNP] = 5.0e-3;
                epsilon_trans[SOLVEPNP_DLS] = 5.0e-3; // DLS is remapped to EPnP, so we use the same threshold
                epsilon_trans[SOLVEPNP_UPNP] = 5.0e-3; // UPnP is remapped to EPnP, so we use the same threshold

                epsilon_rot[SOLVEPNP_EPNP] = 5.0e-3;
                epsilon_rot[SOLVEPNP_DLS] = 5.0e-3; // DLS is remapped to EPnP, so we use the same threshold
                epsilon_rot[SOLVEPNP_UPNP] = 5.0e-3; // UPnP is remapped to EPnP, so we use the same threshold
            }
            else
            {
                epsilon_trans[SOLVEPNP_ITERATIVE] = 1e-4;
                epsilon_trans[SOLVEPNP_EPNP] = 5e-3;
                epsilon_trans[SOLVEPNP_DLS] = 5e-3; // DLS is remapped to EPnP, so we use the same threshold
                epsilon_trans[SOLVEPNP_UPNP] = 5e-3; // UPnP is remapped to EPnP, so we use the same threshold
                epsilon_trans[SOLVEPNP_P3P] = 1e-4;
                epsilon_trans[SOLVEPNP_AP3P] = 1e-4;
                epsilon_trans[SOLVEPNP_IPPE] = 1e-4;
                epsilon_trans[SOLVEPNP_IPPE_SQUARE] = 1e-4;

                epsilon_rot[SOLVEPNP_ITERATIVE] = 1e-4;
                epsilon_rot[SOLVEPNP_EPNP] = 5e-3;
                epsilon_rot[SOLVEPNP_DLS] = 5e-3; // DLS is remapped to EPnP, so we use the same threshold
                epsilon_rot[SOLVEPNP_UPNP] = 5e-3; // UPnP is remapped to EPnP, so we use the same threshold
                epsilon_rot[SOLVEPNP_P3P] = 1e-4;
                epsilon_rot[SOLVEPNP_AP3P] = 1e-4;
                epsilon_rot[SOLVEPNP_IPPE] = 1e-4;
                epsilon_rot[SOLVEPNP_IPPE_SQUARE] = 1e-4;
            }
        }

        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        //UPnP is mapped to EPnP
        //Uncomment this when UPnP is fixed
//        if (method == SOLVEPNP_UPNP)
//        {
//            intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
//        }
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }

        generatePose(points, trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        switch(method)
        {
            case SOLVEPNP_P3P:
            case SOLVEPNP_AP3P:
                opoints = std::vector<Point3f>(points.begin(), points.begin()+4);
                break;
                //UPnP is mapped to EPnP
                //Uncomment this when UPnP is fixed
//            case SOLVEPNP_UPNP:
//                if (points.size() > 50)
//                {
//                    opoints = std::vector<Point3f>(points.begin(), points.begin()+50);
//                }
//                else
//                {
//                    opoints = points;
//                }
//                break;
            default:
                opoints = points;
                break;
        }

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        Mat rvec, tvec;
        bool isEstimateSuccess = solvePnP(opoints, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, method);

        if (!isEstimateSuccess)
        {
            return false;
        }

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        bool isTestSuccess = rvecDiff < epsilon_rot[method] && tvecDiff < epsilon_trans[method];

        errorTrans = tvecDiff;
        errorRot = rvecDiff;

        return isTestSuccess;
    }
};

class CV_solveP3P_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solveP3P_Test()
    {
        eps[SOLVEPNP_P3P] = 2.0e-4;
        eps[SOLVEPNP_AP3P] = 1.0e-4;
        totalTestsCount = 1000;
    }

    ~CV_solveP3P_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        std::vector<Mat> rvecs, tvecs;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }
        generatePose(points, trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        opoints = std::vector<Point3f>(points.begin(), points.begin()+3);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        int num_of_solutions = solveP3P(opoints, projectedPoints, intrinsics, distCoeffs, rvecs, tvecs, method);
        if (num_of_solutions != (int) rvecs.size() || num_of_solutions != (int) tvecs.size() || num_of_solutions == 0)
        {
            return false;
        }

        bool isTestSuccess = false;
        for (size_t i = 0; i < rvecs.size() && !isTestSuccess; i++) {
            double rvecDiff = cvtest::norm(rvecs[i], trueRvec, NORM_L2);
            double tvecDiff = cvtest::norm(tvecs[i], trueTvec, NORM_L2);
            isTestSuccess = rvecDiff < eps[method] && tvecDiff < eps[method];

            errorTrans = std::min(errorTrans, tvecDiff);
            errorRot = std::min(errorRot, rvecDiff);
        }

        return isTestSuccess;
    }

    virtual void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points;
        points.resize(static_cast<size_t>(pointsCount));
        generate3DPointCloud(points);

        const int methodsCount = 2;
        int methods[] = {SOLVEPNP_P3P, SOLVEPNP_AP3P};
        RNG rng = ts->get_rng();

        for (int mode = 0; mode < 2; mode++)
        {
            //To get the same input for each methods
            RNG rngCopy = rng;
            for (int method = 0; method < methodsCount; method++)
            {
                std::vector<double> vec_errorTrans, vec_errorRot;
                vec_errorTrans.reserve(static_cast<size_t>(totalTestsCount));
                vec_errorRot.reserve(static_cast<size_t>(totalTestsCount));

                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    double errorTrans = 0, errorRot = 0;
                    if (runTest(rngCopy, mode, methods[method], points, errorTrans, errorRot))
                    {
                        successfulTestsCount++;
                    }
                    vec_errorTrans.push_back(errorTrans);
                    vec_errorRot.push_back(errorRot);
                }

                double maxErrorTrans = getMax(vec_errorTrans);
                double maxErrorRot = getMax(vec_errorRot);
                double meanErrorTrans = getMean(vec_errorTrans);
                double meanErrorRot = getMean(vec_errorRot);
                double medianErrorTrans = getMedian(vec_errorTrans);
                double medianErrorRot = getMedian(vec_errorRot);

                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy for %s, failed %d tests from %d, %s, "
                                                "maxErrT: %f, maxErrR: %f, "
                                                "meanErrT: %f, meanErrR: %f, "
                                                "medErrT: %f, medErrR: %f\n",
                               printMethod(methods[method]).c_str(), totalTestsCount - successfulTestsCount, totalTestsCount, printMode(mode).c_str(),
                               maxErrorTrans, maxErrorRot, meanErrorTrans, meanErrorRot, medianErrorTrans, medianErrorRot);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
                cout << "mode: " << printMode(mode) << ", method: " << printMethod(methods[method]) << " -> "
                     << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
                     << " (maxErrT: " << maxErrorTrans << ", maxErrR: " << maxErrorRot
                     << ", meanErrT: " << meanErrorTrans << ", meanErrR: " << meanErrorRot
                     << ", medErrT: " << medianErrorTrans << ", medErrR: " << medianErrorRot << ")" << endl;
                double transThres, rotThresh;
                findThreshold(vec_errorTrans, vec_errorRot, 0.7, transThres, rotThresh);
                cout << "approximate translation threshold for 0.7: " << transThres
                     << ", approximate rotation threshold for 0.7: " << rotThresh << endl;
            }
        }
    }
};


TEST(Calib3d_SolveP3P, accuracy) { CV_solveP3P_Test test; test.safe_run();}
TEST(Calib3d_SolvePnPRansac, accuracy) { CV_solvePnPRansac_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnPRansac, accuracy_planar) { CV_solvePnPRansac_Test test(true); test.safe_run(); }
TEST(Calib3d_SolvePnPRansac, accuracy_planar_tag) { CV_solvePnPRansac_Test test(true, true); test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy) { CV_solvePnP_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy_planar) { CV_solvePnP_Test test(true); test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy_planar_tag) { CV_solvePnP_Test test(true, true); test.safe_run(); }

TEST(Calib3d_SolvePnPRansac, concurrency)
{
    int count = 7*13;

    Mat object(1, count, CV_32FC3);
    randu(object, -100, 100);

    Mat camera_mat(3, 3, CV_32FC1);
    randu(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;
    camera_mat.at<float>(2, 2) = 1.f;

    Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    vector<cv::Point2f> image_vec;
    Mat rvec_gold(1, 3, CV_32FC1);
    randu(rvec_gold, 0, 1);
    Mat tvec_gold(1, 3, CV_32FC1);
    randu(tvec_gold, 0, 1);
    projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    Mat image(1, count, CV_32FC2, &image_vec[0]);

    Mat rvec1, rvec2;
    Mat tvec1, tvec2;

    int threads = getNumThreads();
    {
        // limit concurrency to get deterministic result
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec1, tvec1);
    }

    {
        setNumThreads(threads);
        Mat rvec;
        Mat tvec;
        // parallel executions
        for(int i = 0; i < 10; ++i)
        {
            cv::theRNG().state = 20121010;
            solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
        }
    }

    {
        // single thread again
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec2, tvec2);
    }

    double rnorm = cvtest::norm(rvec1, rvec2, NORM_INF);
    double tnorm = cvtest::norm(tvec1, tvec2, NORM_INF);

    EXPECT_LT(rnorm, 1e-6);
    EXPECT_LT(tnorm, 1e-6);
}

TEST(Calib3d_SolvePnPRansac, input_type)
{
    const int numPoints = 10;
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    for (int i = 0; i < numPoints; i+=2)
    {
        points3d.push_back(cv::Point3i(5+i, 3, 2));
        points3d.push_back(cv::Point3i(5+i, 3+i, 2+i));
        points2d.push_back(cv::Point2i(0, i));
        points2d.push_back(cv::Point2i(-i, i));
    }
    Mat R1, t1, R2, t2, R3, t3, R4, t4;

    EXPECT_TRUE(solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R1, t1));

    Mat points3dMat(points3d);
    Mat points2dMat(points2d);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R2, t2));

    points3dMat = points3dMat.reshape(3, 1);
    points2dMat = points2dMat.reshape(2, 1);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R3, t3));

    points3dMat = points3dMat.reshape(1, numPoints);
    points2dMat = points2dMat.reshape(1, numPoints);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R4, t4));

    EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R4, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t4, NORM_INF), 1e-6);
}

TEST(Calib3d_SolvePnPRansac, double_support)
{
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> points2d;
    std::vector<cv::Point3f> points3dF;
    std::vector<cv::Point2f> points2dF;
    for (int i = 0; i < 10 ; i+=2)
    {
        points3d.push_back(cv::Point3d(5+i, 3, 2));
        points3dF.push_back(cv::Point3f(static_cast<float>(5+i), 3, 2));
        points3d.push_back(cv::Point3d(5+i, 3+i, 2+i));
        points3dF.push_back(cv::Point3f(static_cast<float>(5+i), static_cast<float>(3+i), static_cast<float>(2+i)));
        points2d.push_back(cv::Point2d(0, i));
        points2dF.push_back(cv::Point2f(0, static_cast<float>(i)));
        points2d.push_back(cv::Point2d(-i, i));
        points2dF.push_back(cv::Point2f(static_cast<float>(-i), static_cast<float>(i)));
    }
    Mat R, t, RF, tF;
    vector<int> inliers;

    solvePnPRansac(points3dF, points2dF, intrinsics, cv::Mat(), RF, tF, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);
    solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R, t, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);

    EXPECT_LE(cvtest::norm(R, Mat_<double>(RF), NORM_INF), 1e-3);
    EXPECT_LE(cvtest::norm(t, Mat_<double>(tF), NORM_INF), 1e-3);
}

TEST(Calib3d_SolvePnPRansac, bad_input_points_19253)
{
    // with this specific data
    // when computing the final pose using points in the consensus set with SOLVEPNP_ITERATIVE and solvePnP()
    // an exception is thrown from solvePnP because there are 5 non-coplanar 3D points and the DLT algorithm needs at least 6 non-coplanar 3D points
    // with PR #19253 we choose to return true, with the pose estimated from the MSS stage instead of throwing the exception

    float pts2d_[] = {
        -5.38358629e-01f, -5.09638414e-02f,
        -5.07192254e-01f, -2.20743284e-01f,
        -5.43107152e-01f, -4.90474701e-02f,
        -5.54325163e-01f, -1.86715424e-01f,
        -5.59334219e-01f, -4.01909500e-02f,
        -5.43504596e-01f, -4.61776406e-02f
    };
    Mat pts2d(6, 2, CV_32FC1, pts2d_);

    float pts3d_[] = {
        -3.01153604e-02f, -1.55665115e-01f, 4.50000018e-01f,
        4.27827090e-01f, 4.28645730e-01f, 1.08600008e+00f,
        -3.14165242e-02f, -1.52656138e-01f, 4.50000018e-01f,
        -1.46217480e-01f, 5.57961613e-02f, 7.17000008e-01f,
        -4.89348806e-02f, -1.38795510e-01f, 4.47000027e-01f,
        -3.13065052e-02f, -1.52636901e-01f, 4.51000035e-01f
    };
    Mat pts3d(6, 3, CV_32FC1, pts3d_);

    Mat camera_mat = Mat::eye(3, 3, CV_64FC1);
    Mat rvec, tvec;
    vector<int> inliers;

    // solvePnPRansac will return true with 5 inliers, which means the result is from MSS stage.
    bool result = solvePnPRansac(pts3d, pts2d, camera_mat, noArray(), rvec, tvec, false, 100, 4.f / 460.f, 0.99, inliers);
    EXPECT_EQ(inliers.size(), size_t(5));
    EXPECT_TRUE(result);
}

TEST(Calib3d_SolvePnP, input_type)
{
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);
    vector<Point3d> points3d_;
    vector<Point3f> points3dF_;
    //Cube
    const float l = -0.1f;
    //Front face
    points3d_.push_back(Point3d(-l, -l, -l));
    points3dF_.push_back(Point3f(-l, -l, -l));
    points3d_.push_back(Point3d(l, -l, -l));
    points3dF_.push_back(Point3f(l, -l, -l));
    points3d_.push_back(Point3d(l, l, -l));
    points3dF_.push_back(Point3f(l, l, -l));
    points3d_.push_back(Point3d(-l, l, -l));
    points3dF_.push_back(Point3f(-l, l, -l));
    //Back face
    points3d_.push_back(Point3d(-l, -l, l));
    points3dF_.push_back(Point3f(-l, -l, l));
    points3d_.push_back(Point3d(l, -l, l));
    points3dF_.push_back(Point3f(l, -l, l));
    points3d_.push_back(Point3d(l, l, l));
    points3dF_.push_back(Point3f(l, l, l));
    points3d_.push_back(Point3d(-l, l, l));
    points3dF_.push_back(Point3f(-l, l, l));

    Mat trueRvec = (Mat_<double>(3,1) << 0.1, -0.25, 0.467);
    Mat trueTvec = (Mat_<double>(3,1) << -0.21, 0.12, 0.746);

    for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
    {
        vector<Point3d> points3d;
        vector<Point2d> points2d;
        vector<Point3f> points3dF;
        vector<Point2f> points2dF;

        if (method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
        {
            const float tagSize_2 = 0.05f / 2;
            points3d.push_back(Point3d(-tagSize_2,  tagSize_2, 0));
            points3d.push_back(Point3d( tagSize_2,  tagSize_2, 0));
            points3d.push_back(Point3d( tagSize_2, -tagSize_2, 0));
            points3d.push_back(Point3d(-tagSize_2, -tagSize_2, 0));

            points3dF.push_back(Point3f(-tagSize_2,  tagSize_2, 0));
            points3dF.push_back(Point3f( tagSize_2,  tagSize_2, 0));
            points3dF.push_back(Point3f( tagSize_2, -tagSize_2, 0));
            points3dF.push_back(Point3f(-tagSize_2, -tagSize_2, 0));
        }
        else if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P)
        {
            points3d = vector<Point3d>(points3d_.begin(), points3d_.begin()+4);
            points3dF = vector<Point3f>(points3dF_.begin(), points3dF_.begin()+4);
        }
        else
        {
            points3d = points3d_;
            points3dF = points3dF_;
        }

        projectPoints(points3d, trueRvec, trueTvec, intrinsics, noArray(), points2d);
        projectPoints(points3dF, trueRvec, trueTvec, intrinsics, noArray(), points2dF);

        //solvePnP
        {
            Mat R, t, RF, tF;

            solvePnP(points3dF, points2dF, Matx33f(intrinsics), Mat(), RF, tF, false, method);
            solvePnP(points3d, points2d, intrinsics, Mat(), R, t, false, method);

            //By default rvec and tvec must be returned in double precision
            EXPECT_EQ(RF.type(), tF.type());
            EXPECT_EQ(RF.type(), CV_64FC1);

            EXPECT_EQ(R.type(), t.type());
            EXPECT_EQ(R.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
        }
        {
            Mat R1, t1, R2, t2;

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            //By default rvec and tvec must be returned in double precision
            EXPECT_EQ(R1.type(), t1.type());
            EXPECT_EQ(R1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), t2.type());
            EXPECT_EQ(R2.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
        }
        {
            Mat R1(3,1,CV_32FC1), t1(3,1,CV_64FC1);
            Mat R2(3,1,CV_64FC1), t2(3,1,CV_32FC1);

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            //If not null, rvec and tvec must be returned in the same precision
            EXPECT_EQ(R1.type(), CV_32FC1);
            EXPECT_EQ(t1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), CV_64FC1);
            EXPECT_EQ(t2.type(), CV_32FC1);

            EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
        }
        {
            Matx31f R1, t2;
            Matx31d R2, t1;

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            Matx31d R1d(R1(0), R1(1), R1(2));
            Matx31d t2d(t2(0), t2(1), t2(2));

            EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
        }

        //solvePnPGeneric
        {
            vector<Mat> Rs, ts, RFs, tFs;

            int res1 = solvePnPGeneric(points3dF, points2dF, Matx33f(intrinsics), Mat(), RFs, tFs, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2d, intrinsics, Mat(), Rs, ts, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R = Rs.front(), t = ts.front(), RF = RFs.front(), tF = tFs.front();

            //By default rvecs and tvecs must be returned in double precision
            EXPECT_EQ(RF.type(), tF.type());
            EXPECT_EQ(RF.type(), CV_64FC1);

            EXPECT_EQ(R.type(), t.type());
            EXPECT_EQ(R.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
        }
        {
            vector<Mat> R1s, t1s, R2s, t2s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R1 = R1s.front(), t1 = t1s.front(), R2 = R2s.front(), t2 = t2s.front();

            //By default rvecs and tvecs must be returned in double precision
            EXPECT_EQ(R1.type(), t1.type());
            EXPECT_EQ(R1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), t2.type());
            EXPECT_EQ(R2.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
        }
        {
            vector<Mat_<float> > R1s, t2s;
            vector<Mat_<double> > R2s, t1s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R1 = R1s.front(), t1 = t1s.front();
            Mat R2 = R2s.front(), t2 = t2s.front();

            //If not null, rvecs and tvecs must be returned in the same precision
            EXPECT_EQ(R1.type(), CV_32FC1);
            EXPECT_EQ(t1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), CV_64FC1);
            EXPECT_EQ(t2.type(), CV_32FC1);

            EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
        }
        {
            vector<Matx31f> R1s, t2s;
            vector<Matx31d> R2s, t1s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Matx31f R1 = R1s.front(), t2 = t2s.front();
            Matx31d R2 = R2s.front(), t1 = t1s.front();
            Matx31d R1d(R1(0), R1(1), R1(2)), t2d(t2(0), t2(1), t2(2));

            EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
        }

        if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P)
        {
            //solveP3P
            {
                vector<Mat> Rs, ts, RFs, tFs;

                int res1 = solveP3P(points3dF, points2dF, Matx33f(intrinsics), Mat(), RFs, tFs, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2d, intrinsics, Mat(), Rs, ts, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R = Rs.front(), t = ts.front(), RF = RFs.front(), tF = tFs.front();

                //By default rvecs and tvecs must be returned in double precision
                EXPECT_EQ(RF.type(), tF.type());
                EXPECT_EQ(RF.type(), CV_64FC1);

                EXPECT_EQ(R.type(), t.type());
                EXPECT_EQ(R.type(), CV_64FC1);

                EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
            }
            {
                vector<Mat> R1s, t1s, R2s, t2s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R1 = R1s.front(), t1 = t1s.front(), R2 = R2s.front(), t2 = t2s.front();

                //By default rvecs and tvecs must be returned in double precision
                EXPECT_EQ(R1.type(), t1.type());
                EXPECT_EQ(R1.type(), CV_64FC1);

                EXPECT_EQ(R2.type(), t2.type());
                EXPECT_EQ(R2.type(), CV_64FC1);

                EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
            }
            {
                vector<Mat_<float> > R1s, t2s;
                vector<Mat_<double> > R2s, t1s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R1 = R1s.front(), t1 = t1s.front();
                Mat R2 = R2s.front(), t2 = t2s.front();

                //If not null, rvecs and tvecs must be returned in the same precision
                EXPECT_EQ(R1.type(), CV_32FC1);
                EXPECT_EQ(t1.type(), CV_64FC1);

                EXPECT_EQ(R2.type(), CV_64FC1);
                EXPECT_EQ(t2.type(), CV_32FC1);

                EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
            }
            {
                vector<Matx31f> R1s, t2s;
                vector<Matx31d> R2s, t1s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Matx31f R1 = R1s.front(), t2 = t2s.front();
                Matx31d R2 = R2s.front(), t1 = t1s.front();
                Matx31d R1d(R1(0), R1(1), R1(2)), t2d(t2(0), t2(1), t2(2));

                EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
            }
        }
    }
}

TEST(Calib3d_SolvePnP, translation)
{
    Mat cameraIntrinsic = Mat::eye(3,3, CV_32FC1);
    vector<float> crvec;
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    vector<float> ctvec;
    ctvec.push_back(100.f);
    ctvec.push_back(100.f);
    ctvec.push_back(0.f);
    vector<Point3f> p3d;
    p3d.push_back(Point3f(0,0,0));
    p3d.push_back(Point3f(0,0,10));
    p3d.push_back(Point3f(0,10,10));
    p3d.push_back(Point3f(10,10,10));
    p3d.push_back(Point3f(2,5,5));
    p3d.push_back(Point3f(-4,8,6));

    vector<Point2f> p2d;
    projectPoints(p3d, crvec, ctvec, cameraIntrinsic, noArray(), p2d);
    Mat rvec;
    Mat tvec;
    rvec =(Mat_<float>(3,1) << 0, 0, 0);
    tvec = (Mat_<float>(3,1) << 100, 100, 0);

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    rvec =(Mat_<double>(3,1) << 0, 0, 0);
    tvec = (Mat_<double>(3,1) << 100, 100, 0);
    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, false);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));
}

TEST(Calib3d_SolvePnP, iterativeInitialGuess3pts)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
        Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_64FC1);
        EXPECT_EQ(tvec_est.type(), CV_64FC1);
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
        Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_32FC1);
        EXPECT_EQ(tvec_est.type(), CV_32FC1);
    }
}

TEST(Calib3d_SolvePnP, iterativeInitialGuess)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
        Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_64FC1);
        EXPECT_EQ(tvec_est.type(), CV_64FC1);
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));
        p3d.push_back(Point3f(-L, L, L/2));
        p3d.push_back(Point3f(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
        Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_32FC1);
        EXPECT_EQ(tvec_est.type(), CV_32FC1);
    }
}

TEST(Calib3d_SolvePnP, generic)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d_;
        p3d_.push_back(Point3d(-L, L, 0));
        p3d_.push_back(Point3d(L, L, 0));
        p3d_.push_back(Point3d(L, -L, 0));
        p3d_.push_back(Point3d(-L, -L, 0));
        p3d_.push_back(Point3d(-L, L, L/2));
        p3d_.push_back(Point3d(0, 0, -L/2));

        const int ntests = 10;
        for (int numTest = 0; numTest < ntests; numTest++)
        {
            Mat rvec_ground_truth;
            Mat tvec_ground_truth;
            generatePose(p3d_, rvec_ground_truth, tvec_ground_truth, theRNG());

            vector<Point2d> p2d_;
            projectPoints(p3d_, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d_);

            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                vector<Mat> rvecs_est;
                vector<Mat> tvecs_est;

                vector<Point3d> p3d;
                vector<Point2d> p2d;
                if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P ||
                    method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
                {
                    p3d = vector<Point3d>(p3d_.begin(), p3d_.begin()+4);
                    p2d = vector<Point2d>(p2d_.begin(), p2d_.begin()+4);
                }
                else
                {
                    p3d = p3d_;
                    p2d = p2d_;
                }

                vector<double> reprojectionErrors;
                solvePnPGeneric(p3d, p2d, intrinsics, noArray(), rvecs_est, tvecs_est, false, (SolvePnPMethod)method,
                                noArray(), noArray(), reprojectionErrors);

                EXPECT_TRUE(!rvecs_est.empty());
                EXPECT_TRUE(rvecs_est.size() == tvecs_est.size() && tvecs_est.size() == reprojectionErrors.size());

                for (size_t i = 0; i < reprojectionErrors.size()-1; i++)
                {
                    EXPECT_GE(reprojectionErrors[i+1], reprojectionErrors[i]);
                }

                bool isTestSuccess = false;
                for (size_t i = 0; i < rvecs_est.size() && !isTestSuccess; i++) {
                    double rvecDiff = cvtest::norm(rvecs_est[i], rvec_ground_truth, NORM_L2);
                    double tvecDiff = cvtest::norm(tvecs_est[i], tvec_ground_truth, NORM_L2);
                    const double threshold = method == SOLVEPNP_P3P ? 1e-2 : 1e-4;
                    isTestSuccess = rvecDiff < threshold && tvecDiff < threshold;
                }

                EXPECT_TRUE(isTestSuccess);
            }
        }
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3f_;
        p3f_.push_back(Point3f(-L, L, 0));
        p3f_.push_back(Point3f(L, L, 0));
        p3f_.push_back(Point3f(L, -L, 0));
        p3f_.push_back(Point3f(-L, -L, 0));
        p3f_.push_back(Point3f(-L, L, L/2));
        p3f_.push_back(Point3f(0, 0, -L/2));

        const int ntests = 10;
        for (int numTest = 0; numTest < ntests; numTest++)
        {
            Mat rvec_ground_truth;
            Mat tvec_ground_truth;
            generatePose(p3f_, rvec_ground_truth, tvec_ground_truth, theRNG());

            vector<Point2f> p2f_;
            projectPoints(p3f_, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2f_);

            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                vector<Mat> rvecs_est;
                vector<Mat> tvecs_est;

                vector<Point3f> p3f;
                vector<Point2f> p2f;
                if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P ||
                    method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
                {
                    p3f = vector<Point3f>(p3f_.begin(), p3f_.begin()+4);
                    p2f = vector<Point2f>(p2f_.begin(), p2f_.begin()+4);
                }
                else
                {
                    p3f = vector<Point3f>(p3f_.begin(), p3f_.end());
                    p2f = vector<Point2f>(p2f_.begin(), p2f_.end());
                }

                vector<double> reprojectionErrors;
                solvePnPGeneric(p3f, p2f, intrinsics, noArray(), rvecs_est, tvecs_est, false, (SolvePnPMethod)method,
                                noArray(), noArray(), reprojectionErrors);

                EXPECT_TRUE(!rvecs_est.empty());
                EXPECT_TRUE(rvecs_est.size() == tvecs_est.size() && tvecs_est.size() == reprojectionErrors.size());

                for (size_t i = 0; i < reprojectionErrors.size()-1; i++)
                {
                    EXPECT_GE(reprojectionErrors[i+1], reprojectionErrors[i]);
                }

                bool isTestSuccess = false;
                for (size_t i = 0; i < rvecs_est.size() && !isTestSuccess; i++) {
                    double rvecDiff = cvtest::norm(rvecs_est[i], rvec_ground_truth, NORM_L2);
                    double tvecDiff = cvtest::norm(tvecs_est[i], tvec_ground_truth, NORM_L2);
                    const double threshold = method == SOLVEPNP_P3P ? 1e-2 : 1e-4;
                    isTestSuccess = rvecDiff < threshold && tvecDiff < threshold;
                }

                EXPECT_TRUE(isTestSuccess);
            }
        }
    }
}

TEST(Calib3d_SolvePnP, refine3pts)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
            Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
            Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }
}

TEST(Calib3d_SolvePnP, refine)
{
    //double
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    //float
    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));
        p3d.push_back(Point3f(-L, L, L/2));
        p3d.push_back(Point3f(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    //refine after solvePnP
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        //add small Gaussian noise
        RNG& rng = theRNG();
        for (size_t i = 0; i < p2d.size(); i++)
        {
            p2d[i].x += rng.gaussian(5e-2);
            p2d[i].y += rng.gaussian(5e-2);
        }

        Mat rvec_est, tvec_est;
        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, false, SOLVEPNP_EPNP);

        {

            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est (EPnP): " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est (EPnP): " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
        {
            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
        {
            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
    }
}

TEST(Calib3d_SolvePnPRansac, minPoints)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.9072420896651262, 0.09226497171882152, 0.8880772883671504);
    Matx31d true_tvec(7.376333362427632, 8.434449036856979, 13.79801619778456);

    {
        //nb points = 5 --> ransac_kernel_method = SOLVEPNP_EPNP
        Mat keypoints13D = (Mat_<float>(5, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434,
                                                 15.316854, 3.7486348, 12.491116);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        vector<Point3f> objectPoints;
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            objectPoints.push_back(Point3f(keypoints13D.at<float>(i,0), keypoints13D.at<float>(i,1), keypoints13D.at<float>(i,2)));
        }

        Mat rvec = Mat::zeros(1,3,CV_64FC1);
        Mat Tvec = Mat::zeros(1,3,CV_64FC1);
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        Mat rvec2, Tvec2;
        solvePnP(objectPoints, imagesPoints, matK, distCoeff, rvec2, Tvec2, false, SOLVEPNP_EPNP);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(rvec, rvec2, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(Tvec, Tvec2, NORM_INF), 1e-6);
    }
    {
        //nb points = 4 --> ransac_kernel_method = SOLVEPNP_P3P
        Mat keypoints13D = (Mat_<float>(4, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        vector<Point3f> objectPoints;
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            objectPoints.push_back(Point3f(keypoints13D.at<float>(i,0), keypoints13D.at<float>(i,1), keypoints13D.at<float>(i,2)));
        }

        Mat rvec = Mat::zeros(1,3,CV_64FC1);
        Mat Tvec = Mat::zeros(1,3,CV_64FC1);
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        Mat rvec2, Tvec2;
        solvePnP(objectPoints, imagesPoints, matK, distCoeff, rvec2, Tvec2, false, SOLVEPNP_P3P);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(rvec, rvec2, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(Tvec, Tvec2, NORM_INF), 1e-6);
    }
}

TEST(Calib3d_SolvePnPRansac, inputShape)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.9072420896651262, 0.09226497171882152, 0.8880772883671504);
    Matx31d true_tvec(7.376333362427632, 8.434449036856979, 13.79801619778456);

    {
        //Nx3 1-channel
        Mat keypoints13D = (Mat_<float>(6, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434,
                                                 10.00604,  2.8654366, 15.472504,
                                                 -4.6863389, 5.9355154, 13.146358);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //1xN 3-channel
        Mat keypoints13D(1, 6, CV_32FC3);
        keypoints13D.at<Vec3f>(0,0) = Vec3f(12.00604f, -2.8654366f, 18.472504f);
        keypoints13D.at<Vec3f>(0,1) = Vec3f(7.6863389f, 4.9355154f, 11.146358f);
        keypoints13D.at<Vec3f>(0,2) = Vec3f(14.260933f, 2.8320458f, 12.582781f);
        keypoints13D.at<Vec3f>(0,3) = Vec3f(3.4562225f, 8.2668982f, 11.300434f);
        keypoints13D.at<Vec3f>(0,4) = Vec3f(10.00604f,  2.8654366f, 15.472504f);
        keypoints13D.at<Vec3f>(0,5) = Vec3f(-4.6863389f, 5.9355154f, 13.146358f);

        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<Vec2f>(0,i) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //Nx1 3-channel
        Mat keypoints13D(6, 1, CV_32FC3);
        keypoints13D.at<Vec3f>(0,0) = Vec3f(12.00604f, -2.8654366f, 18.472504f);
        keypoints13D.at<Vec3f>(1,0) = Vec3f(7.6863389f, 4.9355154f, 11.146358f);
        keypoints13D.at<Vec3f>(2,0) = Vec3f(14.260933f, 2.8320458f, 12.582781f);
        keypoints13D.at<Vec3f>(3,0) = Vec3f(3.4562225f, 8.2668982f, 11.300434f);
        keypoints13D.at<Vec3f>(4,0) = Vec3f(10.00604f,  2.8654366f, 15.472504f);
        keypoints13D.at<Vec3f>(5,0) = Vec3f(-4.6863389f, 5.9355154f, 13.146358f);

        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<Vec2f>(i,0) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //vector<Point3f>
        vector<Point3f> keypoints13D;
        keypoints13D.push_back(Point3f(12.00604f, -2.8654366f, 18.472504f));
        keypoints13D.push_back(Point3f(7.6863389f, 4.9355154f, 11.146358f));
        keypoints13D.push_back(Point3f(14.260933f, 2.8320458f, 12.582781f));
        keypoints13D.push_back(Point3f(3.4562225f, 8.2668982f, 11.300434f));
        keypoints13D.push_back(Point3f(10.00604f,  2.8654366f, 15.472504f));
        keypoints13D.push_back(Point3f(-4.6863389f, 5.9355154f, 13.146358f));

        vector<Point2f> keypoints22D;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //vector<Point3d>
        vector<Point3d> keypoints13D;
        keypoints13D.push_back(Point3d(12.00604f, -2.8654366f, 18.472504f));
        keypoints13D.push_back(Point3d(7.6863389f, 4.9355154f, 11.146358f));
        keypoints13D.push_back(Point3d(14.260933f, 2.8320458f, 12.582781f));
        keypoints13D.push_back(Point3d(3.4562225f, 8.2668982f, 11.300434f));
        keypoints13D.push_back(Point3d(10.00604f,  2.8654366f, 15.472504f));
        keypoints13D.push_back(Point3d(-4.6863389f, 5.9355154f, 13.146358f));

        vector<Point2d> keypoints22D;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
}

TEST(Calib3d_SolvePnP, inputShape)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.407, 0.092, 0.88);
    Matx31d true_tvec(0.576, -0.43, 1.3798);

    vector<Point3d> objectPoints;
    const double L = 0.5;
    objectPoints.push_back(Point3d(-L, -L,  L));
    objectPoints.push_back(Point3d( L, -L,  L));
    objectPoints.push_back(Point3d( L,  L,  L));
    objectPoints.push_back(Point3d(-L,  L,  L));
    objectPoints.push_back(Point3d(-L, -L, -L));
    objectPoints.push_back(Point3d( L, -L, -L));

    const int methodsCount = 6;
    int methods[] = {SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE};
    for (int method = 0; method < methodsCount; method++)
    {
        if (methods[method] == SOLVEPNP_IPPE_SQUARE)
        {
            objectPoints[0] = Point3d(-L,  L,  0);
            objectPoints[1] = Point3d( L,  L,  0);
            objectPoints[2] = Point3d( L, -L,  0);
            objectPoints[3] = Point3d(-L, -L,  0);
        }

        {
            //Nx3 1-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(4, 3, CV_32FC1);
            }
            else
            {
                keypoints13D = Mat(6, 3, CV_32FC1);
            }

            for (int i = 0; i < keypoints13D.rows; i++)
            {
                keypoints13D.at<float>(i,0) = static_cast<float>(objectPoints[i].x);
                keypoints13D.at<float>(i,1) = static_cast<float>(objectPoints[i].y);
                keypoints13D.at<float>(i,2) = static_cast<float>(objectPoints[i].z);
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<float>(i,0) = imagesPoints[i].x;
                keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //1xN 3-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(1, 4, CV_32FC3);
            }
            else
            {
                keypoints13D = Mat(1, 6, CV_32FC3);
            }

            for (int i = 0; i < keypoints13D.cols; i++)
            {
                keypoints13D.at<Vec3f>(0,i) = Vec3f(static_cast<float>(objectPoints[i].x),
                                                    static_cast<float>(objectPoints[i].y),
                                                    static_cast<float>(objectPoints[i].z));
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<Vec2f>(0,i) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //Nx1 3-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(4, 1, CV_32FC3);
            }
            else
            {
                keypoints13D = Mat(6, 1, CV_32FC3);
            }

            for (int i = 0; i < keypoints13D.rows; i++)
            {
                keypoints13D.at<Vec3f>(i,0) = Vec3f(static_cast<float>(objectPoints[i].x),
                                                    static_cast<float>(objectPoints[i].y),
                                                    static_cast<float>(objectPoints[i].z));
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<Vec2f>(i,0) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //vector<Point3f>
            vector<Point3f> keypoints13D;
            const int nbPts = (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                               methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE) ? 4 : 6;
            for (int i = 0; i < nbPts; i++)
            {
                keypoints13D.push_back(Point3f(static_cast<float>(objectPoints[i].x),
                                               static_cast<float>(objectPoints[i].y),
                                               static_cast<float>(objectPoints[i].z)));
            }

            vector<Point2f> keypoints22D;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //vector<Point3d>
            vector<Point3d> keypoints13D;
            const int nbPts = (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                               methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE) ? 4 : 6;
            for (int i = 0; i < nbPts; i++)
            {
                keypoints13D.push_back(objectPoints[i]);
            }

            vector<Point2d> keypoints22D;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
    }
}

bool hasNan(const cv::Mat& mat)
{
    bool has = false;
    if (mat.type() == CV_32F)
    {
        for(int i = 0; i < static_cast<int>(mat.total()); i++)
            has |= cvIsNaN(mat.at<float>(i)) != 0;
    }
    else if (mat.type() == CV_64F)
    {
        for(int i = 0; i < static_cast<int>(mat.total()); i++)
            has |= cvIsNaN(mat.at<double>(i)) != 0;
    }
    else
    {
        has = true;
        CV_LOG_ERROR(NULL, "check hasNan called with unsupported type!");
    }

    return has;
}

TEST(AP3P, ctheta1p_nan_23607)
{
    // the task is not well defined and may not converge (empty R, t) or should
    // converge to some non-NaN solution
    const std::array<cv::Point2d, 3> cameraPts = {
        cv::Point2d{0.042784865945577621, 0.59844839572906494},
        cv::Point2d{-0.028428621590137482, 0.60354739427566528},
        cv::Point2d{0.0046037044376134872, 0.70674681663513184}
    };
    const std::array<cv::Point3d, 3> modelPts = {
        cv::Point3d{-0.043258000165224075, 0.020459245890378952, -0.0069921980611979961},
        cv::Point3d{-0.045648999512195587, 0.0029820732306689024, 0.0079000638797879219},
        cv::Point3d{-0.043276999145746231, -0.013622495345771313, 0.0080113131552934647}
    };

    std::vector<Mat> R, t;
    solveP3P(modelPts, cameraPts, Mat::eye(3, 3, CV_64F), Mat(), R, t, SOLVEPNP_AP3P);

    EXPECT_EQ(R.size(), 2ul);
    EXPECT_EQ(t.size(), 2ul);

    // Try apply rvec and tvec to get model points from camera points.
    Mat pts = Mat(modelPts).reshape(1, 3);
    Mat expected = Mat(cameraPts).reshape(1, 3);
    for (size_t i = 0; i < R.size(); ++i) {
        EXPECT_TRUE(!hasNan(R[i]));
        EXPECT_TRUE(!hasNan(t[i]));

        Mat transform;
        cv::Rodrigues(R[i], transform);
        Mat res = pts * transform.t();
        for (int j = 0; j < 3; ++j) {
            res.row(j) += t[i].reshape(1, 1);
            res.row(j) /= res.row(j).at<double>(2);
        }
        EXPECT_LE(cvtest::norm(res.colRange(0, 2), expected, NORM_INF), 3.34e-16);
    }
}

}} // namespace
