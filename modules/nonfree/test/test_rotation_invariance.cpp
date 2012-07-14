/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";

static
Mat generateHomography(float angle)
{
    float angleRadian = angle * CV_PI / 180.;
    Mat H = Mat::eye(3, 3, CV_32FC1);
    H.at<float>(0,0) = H.at<float>(1,1) = std::cos(angleRadian);
    H.at<float>(0,1) = -std::sin(angleRadian);
    H.at<float>(1,0) =  std::sin(angleRadian);

    return H;
}

static
Mat rotateImage(const Mat& srcImage, float angle, Mat& dstImage, Mat& dstMask)
{
    float diag = std::sqrt(static_cast<float>(srcImage.cols * srcImage.cols + srcImage.rows * srcImage.rows));
    Mat LUShift = Mat::eye(3, 3, CV_32FC1); // left up
    LUShift.at<float>(0,2) = -srcImage.cols/2;
    LUShift.at<float>(1,2) = -srcImage.rows/2;
    Mat RDShift = Mat::eye(3, 3, CV_32FC1); // right down
    RDShift.at<float>(0,2) = diag/2;
    RDShift.at<float>(1,2) = diag/2;
    Size sz(cvRound(diag), cvRound(diag));

    Mat srcMask(srcImage.size(), CV_8UC1, Scalar(255));

    Mat H = RDShift * generateHomography(angle) * LUShift;
    warpPerspective(srcImage, dstImage, H, sz);
    warpPerspective(srcMask, dstMask, H, sz);

    return H;
}

static
float calcIntersectArea(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float c = norm(p0 - p1), sqr_c = c * c;

    float sqr_r0 = r0 * r0;
    float sqr_r1 = r1 * r1;

    if(r0 + r1 <= c)
       return 0;

    float minR = std::min(r0, r1);
    float maxR = std::max(r0, r1);
    if(c + minR <= maxR)
        return CV_PI * minR * minR;

    float cos_halfA0 = (sqr_r0 + sqr_c - sqr_r1) / (2 * r0 * c);
    float cos_halfA1 = (sqr_r1 + sqr_c - sqr_r0) / (2 * r1 * c);

    float A0 = 2 * acos(cos_halfA0);
    float A1 = 2 * acos(cos_halfA1);

    return  0.5 * sqr_r0 * (A0 - sin(A0)) +
            0.5 * sqr_r1 * (A1 - sin(A1));
}

static
float calcIntersectRatio(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float intersectArea = calcIntersectArea(p0, r0, p1, r1);
    float unionArea = CV_PI * (r0 * r0 + r1 * r1) - intersectArea;
    return intersectArea / unionArea;
}

class DetectorRotatationInvarianceTest : public cvtest::BaseTest
{
public:
    DetectorRotatationInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                     float _minInliersRatio,
                                     float _minAngleInliersRatio) :
        featureDetector(_featureDetector), minInliersRatio(_minInliersRatio), minAngleInliersRatio(_minAngleInliersRatio)
    {
        CV_Assert(!featureDetector.empty());
    }

protected:

    void run(int)
    {
        const string imageFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

        // Read test data
        Mat image0 = imread(imageFilename), image1, mask1;

        if(image0.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints0;
        featureDetector->detect(image0, keypoints0);

        CV_Assert(keypoints0.size() > 15);

        const int maxAngle = 360, angleStep = 10;
        for(int angle = 0; angle < maxAngle; angle += angleStep)
        {
            Mat H = rotateImage(image0, angle, image1, mask1);

            vector<KeyPoint> keypoints1;
            featureDetector->detect(image1, keypoints1, mask1);

            vector<Point2f> points0;
            KeyPoint::convert(keypoints0, points0);
            Mat points0t;
            perspectiveTransform(Mat(points0), points0t, H);

            int inliersCount = 0;
            int angleInliersCount = 0;

            for(size_t m0 = 0; m0 < points0t.total(); m0++)
            {
                int nearestPointIndex = -1;
                float maxIntersectRatio = 0.f;
                const float r0 =  0.5f * keypoints0[m0].size;
                for(size_t m1 = 0; m1 < keypoints1.size(); m1++)
                {

                    float r1 = 0.5f * keypoints1[m1].size;
                    float intersectRatio = calcIntersectRatio(points0t.at<Point2f>(m0), r0,
                                                              keypoints1[m1].pt, r1);
                    if(intersectRatio > maxIntersectRatio)
                    {
                        maxIntersectRatio = intersectRatio;
                        nearestPointIndex = m1;
                    }
                }

                if(maxIntersectRatio > 0.5f)
                {
                    inliersCount++;

                    const float maxAngleDiff = 15.f; // grad
					float angle0 = keypoints0[m0].angle;
                    float angle1 = keypoints1[nearestPointIndex].angle;
                    if(angle0 == -1 || angle1 == -1)
                        CV_Error(CV_StsBadArg, "Given FeatureDetector is not rotation invariant, it can not be tested here.\n");
                    CV_Assert(angle0 >= 0.f && angle0 < 360.f);
                    CV_Assert(angle1 >= 0.f && angle1 < 360.f);

                    float rotAngle0 = angle0 + angle;
                    if(rotAngle0 >= 360.f)
                        rotAngle0 -= 360.f;

                    float angleDiff = std::max(rotAngle0, angle1) - std::min(rotAngle0, angle1);
                    angleDiff = std::min(angleDiff, static_cast<float>(360.f - angleDiff));
					CV_Assert(angleDiff >= 0.f);
                    bool isAngleCorrect = angleDiff < maxAngleDiff;

                    if(isAngleCorrect)
                        angleInliersCount++;
                }
            }

            float inliersRatio = static_cast<float>(inliersCount) / keypoints0.size();
            if(inliersRatio < minInliersRatio)
            {
                ts->printf(cvtest::TS::LOG, "Incorrect inliersRatio: curr = %f, min = %f.\n",
                           inliersRatio, minInliersRatio);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            if(inliersCount)
            {
                float angleInliersRatio = static_cast<float>(angleInliersCount) / inliersCount;
                if(angleInliersRatio < minAngleInliersRatio)
                {
                    ts->printf(cvtest::TS::LOG, "Incorrect angleInliersRatio: curr = %f, min = %f.\n",
                               angleInliersRatio, minAngleInliersRatio);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }

//            std::cout << "inliersRatio - " << inliersRatio
//                      << " - angleInliersRatio " << static_cast<float>(angleInliersCount) / inliersCount << std::endl;
        }
        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector;
    float minInliersRatio;
    float minAngleInliersRatio;
};

// Tests registration

TEST(Features2d_RotationInvariance_Detector_SURF, regression)
{
    DetectorRotatationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"), 0.60f, 0.76f);
    test.safe_run();
}

TEST(Features2d_RotationInvariance_Detector_SIFT, regression)
{
    DetectorRotatationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"), 0.76f, 0.76f);
    test.safe_run();
}

