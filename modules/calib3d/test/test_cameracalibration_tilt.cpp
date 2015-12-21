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

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp>
#include "opencv2/calib3d.hpp"

#define NUM_DIST_COEFF_TILT 14

/**
Some conventions:
- the first camera determines the world coordinate system
- y points down, hence top means minimal y value (negative) and
bottom means maximal y value (positive)
- the field of view plane is tilted around x such that it
intersects the xy-plane in a line with a large (positive)
y-value
- image sensor and object are both modelled in the halfspace
z > 0


**/
class cameraCalibrationTiltTest : public ::testing::Test {

protected:
    cameraCalibrationTiltTest()
        : m_toRadian(acos(-1.0)/180.0)
        , m_toDegree(180.0/acos(-1.0))
    {}
    virtual void SetUp();

protected:
    static const cv::Size m_imageSize;
    static const double m_pixelSize;
    static const double m_circleConfusionPixel;
    static const double m_lensFocalLength;
    static const double m_lensFNumber;
    static const double m_objectDistance;
    static const double m_planeTiltDegree;
    static const double m_pointTargetDist;
    static const int m_pointTargetNum;

    /** image distance coresponding to working distance */
    double m_imageDistance;
    /** image tilt angle corresponding to the tilt of the object plane */
    double m_imageTiltDegree;
    /** center of the field of view, near and far plane */
    std::vector<cv::Vec3d> m_fovCenter;
    /** normal of the field of view, near and far plane */
    std::vector<cv::Vec3d> m_fovNormal;
    /** points on a plane calibration target */
    std::vector<cv::Point3d> m_pointTarget;
    /** rotations for the calibration target */
    std::vector<cv::Vec3d> m_pointTargetRvec;
    /** translations for the calibration target */
    std::vector<cv::Vec3d> m_pointTargetTvec;
    /** camera matrix */
    cv::Matx33d m_cameraMatrix;
    /** distortion coefficients */
    cv::Vec<double, NUM_DIST_COEFF_TILT> m_distortionCoeff;

    /** random generator */
    cv::RNG m_rng;
    /** degree to radian conversion factor */
    const double m_toRadian;
    /** radian to degree conversion factor */
    const double m_toDegree;

    /**
    computes for a given distance of an image or object point
    the distance of the corresponding object or image point
    */
    double opticalMap(double dist) {
        return m_lensFocalLength*dist/(dist - m_lensFocalLength);
    }

    /** magnification of the optical map */
    double magnification(double dist) {
        return m_lensFocalLength/(dist - m_lensFocalLength);
    }

    /**
    Changes given distortion coefficients randomly by adding
    a uniformly distributed random variable in [-max max]
    \param coeff input
    \param max limits for the random variables
    */
    void randomDistortionCoeff(
        cv::Vec<double, NUM_DIST_COEFF_TILT>& coeff,
        const cv::Vec<double, NUM_DIST_COEFF_TILT>& max)
    {
        for (int i = 0; i < coeff.rows; ++i)
            coeff(i) += m_rng.uniform(-max(i), max(i));
    }

    /** numerical jacobian */
    void numericalDerivative(
        cv::Mat& jac,
        double eps,
        const std::vector<cv::Point3d>& obj,
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        const cv::Matx33d& camera,
        const cv::Vec<double, NUM_DIST_COEFF_TILT>& distor);

    /** remove points with projection outside the sensor array */
    void removeInvalidPoints(
        std::vector<cv::Point2d>& imagePoints,
        std::vector<cv::Point3d>& objectPoints);

    /** add uniform distribute noise in [-halfWidthNoise, halfWidthNoise]
    to the image points and remove out of range points */
    void addNoiseRemoveInvalidPoints(
        std::vector<cv::Point2f>& imagePoints,
        std::vector<cv::Point3f>& objectPoints,
        std::vector<cv::Point2f>& noisyImagePoints,
        double halfWidthNoise);
};

/** Number of Pixel of the sensor */
const cv::Size cameraCalibrationTiltTest::m_imageSize(1600, 1200);
/** Size of a pixel in mm */
const double cameraCalibrationTiltTest::m_pixelSize(.005);
/** Diameter of the circle of confusion */
const double cameraCalibrationTiltTest::m_circleConfusionPixel(3);
/** Focal length of the lens */
const double cameraCalibrationTiltTest::m_lensFocalLength(16.4);
/** F-Number */
const double cameraCalibrationTiltTest::m_lensFNumber(8);
/** Working distance */
const double cameraCalibrationTiltTest::m_objectDistance(200);
/** Angle between optical axis and object plane normal */
const double cameraCalibrationTiltTest::m_planeTiltDegree(55);
/** the calibration target are points on a square grid with this side length */
const double cameraCalibrationTiltTest::m_pointTargetDist(5);
/** the calibration target has (2*n + 1) x (2*n + 1) points */
const int cameraCalibrationTiltTest::m_pointTargetNum(15);


void cameraCalibrationTiltTest::SetUp()
{
    m_imageDistance = opticalMap(m_objectDistance);
    m_imageTiltDegree = m_toDegree * atan2(
        m_imageDistance * tan(m_toRadian * m_planeTiltDegree),
        m_objectDistance);
    // half sensor height
    double tmp = .5 * (m_imageSize.height - 1) * m_pixelSize
        * cos(m_toRadian * m_imageTiltDegree);
    // y-Value of tilted sensor
    double yImage[2] = {tmp, -tmp};
    // change in z because of the tilt
    tmp *= sin(m_toRadian * m_imageTiltDegree);
    // z-values of the sensor lower and upper corner
    double zImage[2] = {
        m_imageDistance + tmp,
        m_imageDistance - tmp};
    // circle of confusion
    double circleConfusion = m_circleConfusionPixel*m_pixelSize;
    // aperture of the lense
    double aperture = m_lensFocalLength/m_lensFNumber;
    // near and far factor on the image side
    double nearFarFactorImage[2] = {
        aperture/(aperture - circleConfusion),
        aperture/(aperture + circleConfusion)};
    // on the object side - points that determin the field of
    // view
    std::vector<cv::Vec3d> fovBottomTop(6);
    std::vector<cv::Vec3d>::iterator itFov = fovBottomTop.begin();
    for (size_t iBottomTop = 0; iBottomTop < 2; ++iBottomTop)
    {
        // mapping sensor to field of view
        *itFov = cv::Vec3d(0,yImage[iBottomTop],zImage[iBottomTop]);
        *itFov *= magnification((*itFov)(2));
        ++itFov;
        for (size_t iNearFar = 0; iNearFar < 2; ++iNearFar, ++itFov)
        {
            // scaling to the near and far distance on the
            // image side
            *itFov = cv::Vec3d(0,yImage[iBottomTop],zImage[iBottomTop]) *
                nearFarFactorImage[iNearFar];
            // scaling to the object side
            *itFov *= magnification((*itFov)(2));
        }
    }
    m_fovCenter.resize(3);
    m_fovNormal.resize(3);
    for (size_t i = 0; i < 3; ++i)
    {
        m_fovCenter[i] = .5*(fovBottomTop[i] + fovBottomTop[i+3]);
        m_fovNormal[i] = fovBottomTop[i+3] - fovBottomTop[i];
        m_fovNormal[i] = cv::normalize(m_fovNormal[i]);
        m_fovNormal[i] = cv::Vec3d(
            m_fovNormal[i](0),
            -m_fovNormal[i](2),
            m_fovNormal[i](1));
        // one target position in each plane
        m_pointTargetTvec.push_back(m_fovCenter[i]);
        cv::Vec3d rvec = cv::Vec3d(0,0,1).cross(m_fovNormal[i]);
        rvec = cv::normalize(rvec);
        rvec *= acos(m_fovNormal[i](2));
        m_pointTargetRvec.push_back(rvec);
    }
    // calibration target
    size_t num = 2*m_pointTargetNum + 1;
    m_pointTarget.resize(num*num);
    std::vector<cv::Point3d>::iterator itTarget = m_pointTarget.begin();
    for (int iY = -m_pointTargetNum; iY <= m_pointTargetNum; ++iY)
    {
        for (int iX = -m_pointTargetNum; iX <= m_pointTargetNum; ++iX, ++itTarget)
        {
            *itTarget = cv::Point3d(iX, iY, 0) * m_pointTargetDist;
        }
    }
    // oblique target positions
    // approximate distance to the near and far plane
    double dist = std::max(
        std::abs(m_fovNormal[0].dot(m_fovCenter[0] - m_fovCenter[1])),
        std::abs(m_fovNormal[0].dot(m_fovCenter[0] - m_fovCenter[2])));
    // maximal angle such that target border "reaches" near and far plane
    double maxAngle = atan2(dist, m_pointTargetNum*m_pointTargetDist);
    std::vector<double> angle;
    angle.push_back(-maxAngle);
    angle.push_back(maxAngle);
    cv::Matx33d baseMatrix;
    cv::Rodrigues(m_pointTargetRvec.front(), baseMatrix);
    for (std::vector<double>::const_iterator itAngle = angle.begin(); itAngle != angle.end(); ++itAngle)
    {
        cv::Matx33d rmat;
        for (int i = 0; i < 2; ++i)
        {
            cv::Vec3d rvec(0,0,0);
            rvec(i) = *itAngle;
            cv::Rodrigues(rvec, rmat);
            rmat = baseMatrix*rmat;
            cv::Rodrigues(rmat, rvec);
            m_pointTargetTvec.push_back(m_fovCenter.front());
            m_pointTargetRvec.push_back(rvec);
        }
    }
    // camera matrix
    double cx = .5 * (m_imageSize.width - 1);
    double cy = .5 * (m_imageSize.height - 1);
    double f = m_imageDistance/m_pixelSize;
    m_cameraMatrix = cv::Matx33d(
        f,0,cx,
        0,f,cy,
        0,0,1);
    // distortion coefficients
    m_distortionCoeff = cv::Vec<double, NUM_DIST_COEFF_TILT>::all(0);
    // tauX
    m_distortionCoeff(12) = -m_toRadian*m_imageTiltDegree;

}

void cameraCalibrationTiltTest::numericalDerivative(
    cv::Mat& jac,
    double eps,
    const std::vector<cv::Point3d>& obj,
    const cv::Vec3d& rvec,
    const cv::Vec3d& tvec,
    const cv::Matx33d& camera,
    const cv::Vec<double, NUM_DIST_COEFF_TILT>& distor)
{
    cv::Vec3d r(rvec);
    cv::Vec3d t(tvec);
    cv::Matx33d cm(camera);
    cv::Vec<double, NUM_DIST_COEFF_TILT> dc(distor);
    double* param[10+NUM_DIST_COEFF_TILT] = {
        &r(0), &r(1), &r(2),
        &t(0), &t(1), &t(2),
        &cm(0,0), &cm(1,1), &cm(0,2), &cm(1,2),
        &dc(0), &dc(1), &dc(2), &dc(3), &dc(4), &dc(5), &dc(6),
        &dc(7), &dc(8), &dc(9), &dc(10), &dc(11), &dc(12), &dc(13)};
    std::vector<cv::Point2d> pix0, pix1;
    double invEps = .5/eps;

    for (int col = 0; col < 10+NUM_DIST_COEFF_TILT; ++col)
    {
        double save = *(param[col]);
        *(param[col]) = save + eps;
        cv::projectPoints(obj, r, t, cm, dc, pix0);
        *(param[col]) = save - eps;
        cv::projectPoints(obj, r, t, cm, dc, pix1);
        *(param[col]) = save;

        std::vector<cv::Point2d>::const_iterator it0 = pix0.begin();
        std::vector<cv::Point2d>::const_iterator it1 = pix1.begin();
        int row = 0;
        for (;it0 != pix0.end(); ++it0, ++it1)
        {
            cv::Point2d d = invEps*(*it0 - *it1);
            jac.at<double>(row, col) = d.x;
            ++row;
            jac.at<double>(row, col) = d.y;
            ++row;
        }
    }
}

void cameraCalibrationTiltTest::removeInvalidPoints(
    std::vector<cv::Point2d>& imagePoints,
    std::vector<cv::Point3d>& objectPoints)
{
    // remove object and imgage points out of range
    std::vector<cv::Point2d>::iterator itImg = imagePoints.begin();
    std::vector<cv::Point3d>::iterator itObj = objectPoints.begin();
    while (itImg != imagePoints.end())
    {
        bool ok =
            itImg->x >= 0 &&
            itImg->x <= m_imageSize.width - 1.0 &&
            itImg->y >= 0 &&
            itImg->y <= m_imageSize.height - 1.0;
        if (ok)
        {
            ++itImg;
            ++itObj;
        }
        else
        {
            itImg = imagePoints.erase(itImg);
            itObj = objectPoints.erase(itObj);
        }
    }
}

void cameraCalibrationTiltTest::addNoiseRemoveInvalidPoints(
    std::vector<cv::Point2f>& imagePoints,
    std::vector<cv::Point3f>& objectPoints,
    std::vector<cv::Point2f>& noisyImagePoints,
    double halfWidthNoise)
{
    std::vector<cv::Point2f>::iterator itImg = imagePoints.begin();
    std::vector<cv::Point3f>::iterator itObj = objectPoints.begin();
    noisyImagePoints.clear();
    noisyImagePoints.reserve(imagePoints.size());
    while (itImg != imagePoints.end())
    {
        cv::Point2f pix = *itImg + cv::Point2f(
            (float)m_rng.uniform(-halfWidthNoise, halfWidthNoise),
            (float)m_rng.uniform(-halfWidthNoise, halfWidthNoise));
        bool ok =
            pix.x >= 0 &&
            pix.x <= m_imageSize.width - 1.0 &&
            pix.y >= 0 &&
            pix.y <= m_imageSize.height - 1.0;
        if (ok)
        {
            noisyImagePoints.push_back(pix);
            ++itImg;
            ++itObj;
        }
        else
        {
            itImg = imagePoints.erase(itImg);
            itObj = objectPoints.erase(itObj);
        }
    }
}


TEST_F(cameraCalibrationTiltTest, projectPoints)
{
    std::vector<cv::Point2d> imagePoints;
    std::vector<cv::Point3d> objectPoints = m_pointTarget;
    cv::Vec3d rvec = m_pointTargetRvec.front();
    cv::Vec3d tvec = m_pointTargetTvec.front();

    cv::Vec<double, NUM_DIST_COEFF_TILT> coeffNoiseHalfWidth(
        .1, .1, // k1 k2
        .01, .01, // p1 p2
        .001, .001, .001, .001, // k3 k4 k5 k6
        .001, .001, .001, .001, // s1 s2 s3 s4
        .01, .01); // tauX tauY
    for (size_t numTest = 0; numTest < 10; ++numTest)
    {
        // create random distortion coefficients
        cv::Vec<double, NUM_DIST_COEFF_TILT> distortionCoeff = m_distortionCoeff;
        randomDistortionCoeff(distortionCoeff, coeffNoiseHalfWidth);

        // projection
        cv::projectPoints(
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff,
            imagePoints);

        // remove object and imgage points out of range
        removeInvalidPoints(imagePoints, objectPoints);

        int numPoints = (int)imagePoints.size();
        int numParams = 10 + distortionCoeff.rows;
        cv::Mat jacobian(2*numPoints, numParams, CV_64FC1);

        // projection and jacobian
        cv::projectPoints(
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff,
            imagePoints,
            jacobian);

        // numerical derivatives
        cv::Mat numericJacobian(2*numPoints, numParams, CV_64FC1);
        double eps = 1e-7;
        numericalDerivative(
            numericJacobian,
            eps,
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff);

#if 0
        for (size_t row = 0; row < 2; ++row)
        {
            std::cout << "------ Row = " << row << " ------\n";
            for (size_t i = 0; i < 10+NUM_DIST_COEFF_TILT; ++i)
            {
                std::cout << i
                    << "  jac = " << jacobian.at<double>(row,i)
                    << "  num = " << numericJacobian.at<double>(row,i)
                    << "  rel. diff = " << abs(numericJacobian.at<double>(row,i) - jacobian.at<double>(row,i))/abs(numericJacobian.at<double>(row,i))
                    << "\n";
            }
        }
#endif
        // relative difference for large values (rvec and tvec)
        cv::Mat check = abs(jacobian(cv::Range::all(), cv::Range(0,6)) - numericJacobian(cv::Range::all(), cv::Range(0,6)))/
            (1 + abs(jacobian(cv::Range::all(), cv::Range(0,6))));
        double minVal, maxVal;
        cv::minMaxIdx(check, &minVal, &maxVal);
        EXPECT_LE(maxVal, .01);
        // absolute difference for distortion and camera matrix
        EXPECT_MAT_NEAR(jacobian(cv::Range::all(), cv::Range(6,numParams)), numericJacobian(cv::Range::all(), cv::Range(6,numParams)), 1e-5);
    }
}

TEST_F(cameraCalibrationTiltTest, undistortPoints)
{
    cv::Vec<double, NUM_DIST_COEFF_TILT> coeffNoiseHalfWidth(
        .2, .1, // k1 k2
        .01, .01, // p1 p2
        .01, .01, .01, .01, // k3 k4 k5 k6
        .001, .001, .001, .001, // s1 s2 s3 s4
        .001, .001); // tauX tauY
    double step = 99;
    double toleranceBackProjection = 1e-5;

    for (size_t numTest = 0; numTest < 10; ++numTest)
    {
        cv::Vec<double, NUM_DIST_COEFF_TILT> distortionCoeff = m_distortionCoeff;
        randomDistortionCoeff(distortionCoeff, coeffNoiseHalfWidth);

        // distorted points
        std::vector<cv::Point2d> distorted;
        for (double x = 0; x <= m_imageSize.width-1; x += step)
            for (double y = 0; y <= m_imageSize.height-1; y += step)
                distorted.push_back(cv::Point2d(x,y));
        std::vector<cv::Point2d> normalizedUndistorted;

        // undistort
        cv::undistortPoints(distorted,
            normalizedUndistorted,
            m_cameraMatrix,
            distortionCoeff);

        // copy normalized points to 3D
        std::vector<cv::Point3d> objectPoints;
        for (std::vector<cv::Point2d>::const_iterator itPnt = normalizedUndistorted.begin();
            itPnt != normalizedUndistorted.end(); ++itPnt)
            objectPoints.push_back(cv::Point3d(itPnt->x, itPnt->y, 1));

        // project
        std::vector<cv::Point2d> imagePoints(objectPoints.size());
        cv::projectPoints(objectPoints,
            cv::Vec3d(0,0,0),
            cv::Vec3d(0,0,0),
            m_cameraMatrix,
            distortionCoeff,
            imagePoints);

        EXPECT_MAT_NEAR(distorted, imagePoints, toleranceBackProjection);
    }
}

template <typename INPUT, typename ESTIMATE>
void show(const std::string& name, const INPUT in, const ESTIMATE est)
{
    std::cout << name << " = " << est << " (init = " << in
        << ", diff = " << est-in << ")\n";
}

template <typename INPUT>
void showVec(const std::string& name, const INPUT& in, const cv::Mat& est)
{

    for (size_t i = 0; i < in.channels; ++i)
    {
        std::stringstream ss;
        ss << name << "[" << i << "]";
        show(ss.str(), in(i), est.at<double>(i));
    }
}

/**
For given camera matrix and distortion coefficients
- project point target in different positions onto the sensor
- add pixel noise
- estimate camera modell with noisy measurements
- compare result with initial model parameter

Parameter are differently affected by the noise
*/
TEST_F(cameraCalibrationTiltTest, calibrateCamera)
{
    cv::Vec<double, NUM_DIST_COEFF_TILT> coeffNoiseHalfWidth(
        .2, .1, // k1 k2
        .01, .01, // p1 p2
        0, 0, 0, 0, // k3 k4 k5 k6
        .001, .001, .001, .001, // s1 s2 s3 s4
        .001, .001); // tauX tauY
    double pixelNoiseHalfWidth = .5;
    std::vector<cv::Point3f> pointTarget;
    pointTarget.reserve(m_pointTarget.size());
    for (std::vector<cv::Point3d>::const_iterator it = m_pointTarget.begin(); it != m_pointTarget.end(); ++it)
        pointTarget.push_back(cv::Point3f(
        (float)(it->x),
        (float)(it->y),
        (float)(it->z)));

    for (size_t numTest = 0; numTest < 5; ++numTest)
    {
        // create random distortion coefficients
        cv::Vec<double, NUM_DIST_COEFF_TILT> distortionCoeff = m_distortionCoeff;
        randomDistortionCoeff(distortionCoeff, coeffNoiseHalfWidth);

        // container for calibration data
        std::vector<std::vector<cv::Point3f> > viewsObjectPoints;
        std::vector<std::vector<cv::Point2f> > viewsImagePoints;
        std::vector<std::vector<cv::Point2f> > viewsNoisyImagePoints;

        // simulate calibration data with projectPoints
        std::vector<cv::Vec3d>::const_iterator itRvec = m_pointTargetRvec.begin();
        std::vector<cv::Vec3d>::const_iterator itTvec = m_pointTargetTvec.begin();
        // loop over different views
        for (;itRvec != m_pointTargetRvec.end(); ++ itRvec, ++itTvec)
        {
            std::vector<cv::Point3f> objectPoints(pointTarget);
            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point2f> noisyImagePoints;
            // project calibration target to sensor
            cv::projectPoints(
                objectPoints,
                *itRvec,
                *itTvec,
                m_cameraMatrix,
                distortionCoeff,
                imagePoints);
            // remove invisible points
            addNoiseRemoveInvalidPoints(
                imagePoints,
                objectPoints,
                noisyImagePoints,
                pixelNoiseHalfWidth);
            // add data for view
            viewsNoisyImagePoints.push_back(noisyImagePoints);
            viewsImagePoints.push_back(imagePoints);
            viewsObjectPoints.push_back(objectPoints);
        }

        // Output
        std::vector<cv::Mat> outRvecs, outTvecs;
        cv::Mat outCameraMatrix(3, 3, CV_64F, cv::Scalar::all(1)), outDistCoeff;

        // Stopping criteria
        cv::TermCriteria stop(
            cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
            50000,
            1e-14);
        // modell coice
        int flag =
            cv::CALIB_FIX_ASPECT_RATIO |
            // cv::CALIB_RATIONAL_MODEL |
            cv::CALIB_FIX_K3 |
            // cv::CALIB_FIX_K6 |
            cv::CALIB_THIN_PRISM_MODEL |
            cv::CALIB_TILTED_MODEL;
        // estimate
        double backProjErr = cv::calibrateCamera(
            viewsObjectPoints,
            viewsNoisyImagePoints,
            m_imageSize,
            outCameraMatrix,
            outDistCoeff,
            outRvecs,
            outTvecs,
            flag,
            stop);

        EXPECT_LE(backProjErr, pixelNoiseHalfWidth);

#if 0
        std::cout << "------ estimate ------\n";
        std::cout << "back projection error = " << backProjErr << "\n";
        std::cout << "points per view = {" << viewsObjectPoints.front().size();
        for (size_t i = 1; i < viewsObjectPoints.size(); ++i)
            std::cout << ", " << viewsObjectPoints[i].size();
        std::cout << "}\n";
        show("fx", m_cameraMatrix(0,0), outCameraMatrix.at<double>(0,0));
        show("fy", m_cameraMatrix(1,1), outCameraMatrix.at<double>(1,1));
        show("cx", m_cameraMatrix(0,2), outCameraMatrix.at<double>(0,2));
        show("cy", m_cameraMatrix(1,2), outCameraMatrix.at<double>(1,2));
        showVec("distor", distortionCoeff, outDistCoeff);
#endif
        if (pixelNoiseHalfWidth > 0)
        {
            double tolRvec = pixelNoiseHalfWidth;
            double tolTvec = m_objectDistance * tolRvec;
            // back projection error
            for (size_t i = 0; i < viewsNoisyImagePoints.size(); ++i)
            {
                double dRvec = norm(
                    m_pointTargetRvec[i] -
                    cv::Vec3d(
                    outRvecs[i].at<double>(0),
                    outRvecs[i].at<double>(1),
                    outRvecs[i].at<double>(2)));
                // std::cout << dRvec << "  " << tolRvec << "\n";
                EXPECT_LE(dRvec,
                    tolRvec);
                double dTvec = norm(
                    m_pointTargetTvec[i] -
                    cv::Vec3d(
                    outTvecs[i].at<double>(0),
                    outTvecs[i].at<double>(1),
                    outTvecs[i].at<double>(2)));
                // std::cout << dTvec << "  " << tolTvec << "\n";
                EXPECT_LE(dTvec,
                    tolTvec);

                std::vector<cv::Point2f> backProjection;
                cv::projectPoints(
                    viewsObjectPoints[i],
                    outRvecs[i],
                    outTvecs[i],
                    outCameraMatrix,
                    outDistCoeff,
                    backProjection);
                EXPECT_MAT_NEAR(backProjection, viewsNoisyImagePoints[i], 1.5*pixelNoiseHalfWidth);
                EXPECT_MAT_NEAR(backProjection, viewsImagePoints[i], 1.5*pixelNoiseHalfWidth);
            }
        }
        pixelNoiseHalfWidth *= .25;
    }
}
