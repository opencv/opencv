// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace testUtil
{

cv::RNG rng(/*std::time(0)*/0);

const float sigma = 1.f;
const float pointsMaxX = 500.f;
const float pointsMaxY = 500.f;
const int testRun = 5000;

void generatePoints(cv::Mat points);
void addNoise(cv::Mat points);

cv::Mat generateTransform(const cv::videostab::MotionModel model);

double performTest(const cv::videostab::MotionModel model, int size);

}

void testUtil::generatePoints(cv::Mat points)
{
    CV_Assert(!points.empty());
    for(int i = 0; i < points.cols; ++i)
    {
        points.at<float>(0, i) = rng.uniform(0.f, pointsMaxX);
        points.at<float>(1, i) = rng.uniform(0.f, pointsMaxY);
        points.at<float>(2, i) = 1.f;
    }
}

void testUtil::addNoise(cv::Mat points)
{
    CV_Assert(!points.empty());
    for(int i = 0; i < points.cols; i++)
    {
        points.at<float>(0, i) += static_cast<float>(rng.gaussian(sigma));
        points.at<float>(1, i) += static_cast<float>(rng.gaussian(sigma));

    }
}


cv::Mat testUtil::generateTransform(const cv::videostab::MotionModel model)
{
    /*----------Params----------*/
    const float minAngle = 0.f, maxAngle = static_cast<float>(CV_PI);
    const float minScale = 0.5f, maxScale = 2.f;
    const float maxTranslation = 100.f;
    const float affineCoeff = 3.f;
    /*----------Params----------*/

    cv::Mat transform = cv::Mat::eye(3, 3, CV_32F);

    if(model != cv::videostab::MM_ROTATION)
    {
        transform.at<float>(0,2) = rng.uniform(-maxTranslation, maxTranslation);
        transform.at<float>(1,2) = rng.uniform(-maxTranslation, maxTranslation);
    }

    if(model != cv::videostab::MM_AFFINE)
    {

        if(model != cv::videostab::MM_TRANSLATION_AND_SCALE &&
                model != cv::videostab::MM_TRANSLATION)
        {
            const float angle = rng.uniform(minAngle, maxAngle);

            transform.at<float>(1,1) = transform.at<float>(0,0) = std::cos(angle);
            transform.at<float>(0,1) = std::sin(angle);
            transform.at<float>(1,0) = -transform.at<float>(0,1);

        }

        if(model == cv::videostab::MM_TRANSLATION_AND_SCALE ||
                model == cv::videostab::MM_SIMILARITY)
        {
            const float scale = rng.uniform(minScale, maxScale);

            transform.at<float>(0,0) *= scale;
            transform.at<float>(1,1) *= scale;

        }

    }
    else
    {
        transform.at<float>(0,0) = rng.uniform(-affineCoeff, affineCoeff);
        transform.at<float>(0,1) = rng.uniform(-affineCoeff, affineCoeff);
        transform.at<float>(1,0) = rng.uniform(-affineCoeff, affineCoeff);
        transform.at<float>(1,1) = rng.uniform(-affineCoeff, affineCoeff);
    }

    return transform;
}


double testUtil::performTest(const cv::videostab::MotionModel model, int size)
{
    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> estimator = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(model);

    estimator->setRansacParams(cv::videostab::RansacParams(size, 3.f*testUtil::sigma /*3 sigma rule*/, 0.5f, 0.5f));

    double disparity = 0.;

    for(int attempt = 0; attempt < testUtil::testRun; attempt++)
    {
        const cv::Mat transform = testUtil::generateTransform(model);

        const int pointsNumber = testUtil::rng.uniform(10, 100);

        cv::Mat points(3, pointsNumber, CV_32F);

        testUtil::generatePoints(points);

        cv::Mat transformedPoints = transform * points;

        testUtil::addNoise(transformedPoints);

        const cv::Mat src = points.rowRange(0,2).t();
        const cv::Mat dst = transformedPoints.rowRange(0,2).t();

        bool isOK = false;
        const cv::Mat estTransform = estimator->estimate(src.reshape(2), dst.reshape(2), &isOK);

        CV_Assert(isOK);
        const cv::Mat testPoints = estTransform * points;

        const double norm = cv::norm(testPoints, transformedPoints, cv::NORM_INF);

        disparity = std::max(disparity, norm);
    }

    return disparity;

}

TEST(Regression, MM_TRANSLATION)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_TRANSLATION, 2), 7.f);
}

TEST(Regression, MM_TRANSLATION_AND_SCALE)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_TRANSLATION_AND_SCALE, 3), 7.f);
}

TEST(Regression, MM_ROTATION)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_ROTATION, 2), 7.f);
}

TEST(Regression, MM_RIGID)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_RIGID, 3), 7.f);
}

TEST(Regression, MM_SIMILARITY)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_SIMILARITY, 4), 7.f);
}

TEST(Regression, MM_AFFINE)
{
    EXPECT_LT(testUtil::performTest(cv::videostab::MM_AFFINE, 6), 9.f);
}
