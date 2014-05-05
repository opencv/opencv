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
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

const string IMAGE_TSUKUBA = "/features2d/tsukuba.png";
const string IMAGE_BIKES = "/detectors_descriptors_evaluation/images_datasets/bikes/img1.png";

#define SHOW_DEBUG_LOG 0

static
Mat generateHomography(float angle)
{
    // angle - rotation around Oz in degrees
    float angleRadian = static_cast<float>(angle * CV_PI / 180);
    Mat H = Mat::eye(3, 3, CV_32FC1);
    H.at<float>(0,0) = H.at<float>(1,1) = std::cos(angleRadian);
    H.at<float>(0,1) = -std::sin(angleRadian);
    H.at<float>(1,0) =  std::sin(angleRadian);

    return H;
}

static
Mat rotateImage(const Mat& srcImage, float angle, Mat& dstImage, Mat& dstMask)
{
    // angle - rotation around Oz in degrees
    float diag = std::sqrt(static_cast<float>(srcImage.cols * srcImage.cols + srcImage.rows * srcImage.rows));
    Mat LUShift = Mat::eye(3, 3, CV_32FC1); // left up
    LUShift.at<float>(0,2) = static_cast<float>(-srcImage.cols/2);
    LUShift.at<float>(1,2) = static_cast<float>(-srcImage.rows/2);
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

void rotateKeyPoints(const vector<KeyPoint>& src, const Mat& H, float angle, vector<KeyPoint>& dst)
{
    // suppose that H is rotation given from rotateImage() and angle has value passed to rotateImage()
    vector<Point2f> srcCenters, dstCenters;
    KeyPoint::convert(src, srcCenters);

    perspectiveTransform(srcCenters, dstCenters, H);

    dst = src;
    for(size_t i = 0; i < dst.size(); i++)
    {
        dst[i].pt = dstCenters[i];
        float dstAngle = src[i].angle + angle;
        if(dstAngle >= 360.f)
            dstAngle -= 360.f;
        dst[i].angle = dstAngle;
    }
}

void scaleKeyPoints(const vector<KeyPoint>& src, vector<KeyPoint>& dst, float scale)
{
    dst.resize(src.size());
    for(size_t i = 0; i < src.size(); i++)
        dst[i] = KeyPoint(src[i].pt.x * scale, src[i].pt.y * scale, src[i].size * scale, src[i].angle);
}

static
float calcCirclesIntersectArea(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float c = static_cast<float>(norm(p0 - p1)), sqr_c = c * c;

    float sqr_r0 = r0 * r0;
    float sqr_r1 = r1 * r1;

    if(r0 + r1 <= c)
       return 0;

    float minR = std::min(r0, r1);
    float maxR = std::max(r0, r1);
    if(c + minR <= maxR)
        return static_cast<float>(CV_PI * minR * minR);

    float cos_halfA0 = (sqr_r0 + sqr_c - sqr_r1) / (2 * r0 * c);
    float cos_halfA1 = (sqr_r1 + sqr_c - sqr_r0) / (2 * r1 * c);

    float A0 = 2 * acos(cos_halfA0);
    float A1 = 2 * acos(cos_halfA1);

    return  0.5f * sqr_r0 * (A0 - sin(A0)) +
            0.5f * sqr_r1 * (A1 - sin(A1));
}

static
float calcIntersectRatio(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float intersectArea = calcCirclesIntersectArea(p0, r0, p1, r1);
    float unionArea = static_cast<float>(CV_PI) * (r0 * r0 + r1 * r1) - intersectArea;
    return intersectArea / unionArea;
}

static
void matchKeyPoints(const vector<KeyPoint>& keypoints0, const Mat& H,
                    const vector<KeyPoint>& keypoints1,
                    vector<DMatch>& matches)
{
    vector<Point2f> points0;
    KeyPoint::convert(keypoints0, points0);
    Mat points0t;
    if(H.empty())
        points0t = Mat(points0);
    else
        perspectiveTransform(Mat(points0), points0t, H);

    matches.clear();
    vector<uchar> usedMask(keypoints1.size(), 0);
    for(int i0 = 0; i0 < static_cast<int>(keypoints0.size()); i0++)
    {
        int nearestPointIndex = -1;
        float maxIntersectRatio = 0.f;
        const float r0 =  0.5f * keypoints0[i0].size;
        for(size_t i1 = 0; i1 < keypoints1.size(); i1++)
        {
            if(nearestPointIndex >= 0 && usedMask[i1])
                continue;

            float r1 = 0.5f * keypoints1[i1].size;
            float intersectRatio = calcIntersectRatio(points0t.at<Point2f>(i0), r0,
                                                      keypoints1[i1].pt, r1);
            if(intersectRatio > maxIntersectRatio)
            {
                maxIntersectRatio = intersectRatio;
                nearestPointIndex = static_cast<int>(i1);
            }
        }

        matches.push_back(DMatch(i0, nearestPointIndex, maxIntersectRatio));
        if(nearestPointIndex >= 0)
            usedMask[nearestPointIndex] = 1;
    }
}

static void removeVerySmallKeypoints(vector<KeyPoint>& keypoints)
{
    size_t i, j = 0, n = keypoints.size();
    for( i = 0; i < n; i++ )
    {
        if( (keypoints[i].octave & 128) != 0 )
            ;
        else
            keypoints[j++] = keypoints[i];
    }
    keypoints.resize(j);
}


class DetectorRotationInvarianceTest : public cvtest::BaseTest
{
public:
    DetectorRotationInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                     float _minKeyPointMatchesRatio,
                                     float _minAngleInliersRatio) :
        featureDetector(_featureDetector),
        minKeyPointMatchesRatio(_minKeyPointMatchesRatio),
        minAngleInliersRatio(_minAngleInliersRatio)
    {
        CV_Assert(featureDetector);
    }

protected:

    void run(int)
    {
        const string imageFilename = string(ts->get_data_path()) + IMAGE_TSUKUBA;

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
        removeVerySmallKeypoints(keypoints0);
        if(keypoints0.size() < 15)
            CV_Error(Error::StsAssert, "Detector gives too few points in a test image\n");

        const int maxAngle = 360, angleStep = 15;
        for(int angle = 0; angle < maxAngle; angle += angleStep)
        {
            Mat H = rotateImage(image0, static_cast<float>(angle), image1, mask1);

            vector<KeyPoint> keypoints1;
            featureDetector->detect(image1, keypoints1, mask1);
            removeVerySmallKeypoints(keypoints1);

            vector<DMatch> matches;
            matchKeyPoints(keypoints0, H, keypoints1, matches);

            int angleInliersCount = 0;

            const float minIntersectRatio = 0.5f;
            int keyPointMatchesCount = 0;
            for(size_t m = 0; m < matches.size(); m++)
            {
                if(matches[m].distance < minIntersectRatio)
                    continue;

                keyPointMatchesCount++;

                // Check does this inlier have consistent angles
                const float maxAngleDiff = 15.f; // grad
                float angle0 = keypoints0[matches[m].queryIdx].angle;
                float angle1 = keypoints1[matches[m].trainIdx].angle;
                if(angle0 == -1 || angle1 == -1)
                    CV_Error(Error::StsBadArg, "Given FeatureDetector is not rotation invariant, it can not be tested here.\n");
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

            float keyPointMatchesRatio = static_cast<float>(keyPointMatchesCount) / keypoints0.size();
            if(keyPointMatchesRatio < minKeyPointMatchesRatio)
            {
                ts->printf(cvtest::TS::LOG, "Incorrect keyPointMatchesRatio: curr = %f, min = %f.\n",
                           keyPointMatchesRatio, minKeyPointMatchesRatio);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            if(keyPointMatchesCount)
            {
                float angleInliersRatio = static_cast<float>(angleInliersCount) / keyPointMatchesCount;
                if(angleInliersRatio < minAngleInliersRatio)
                {
                    ts->printf(cvtest::TS::LOG, "Incorrect angleInliersRatio: curr = %f, min = %f.\n",
                               angleInliersRatio, minAngleInliersRatio);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
#if SHOW_DEBUG_LOG
            std::cout << "keyPointMatchesRatio - " << keyPointMatchesRatio
                << " - angleInliersRatio " << static_cast<float>(angleInliersCount) / keyPointMatchesCount << std::endl;
#endif
        }
        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector;
    float minKeyPointMatchesRatio;
    float minAngleInliersRatio;
};

class DescriptorRotationInvarianceTest : public cvtest::BaseTest
{
public:
    DescriptorRotationInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                     const Ptr<DescriptorExtractor>& _descriptorExtractor,
                                     int _normType,
                                     float _minDescInliersRatio) :
        featureDetector(_featureDetector),
        descriptorExtractor(_descriptorExtractor),
        normType(_normType),
        minDescInliersRatio(_minDescInliersRatio)
    {
        CV_Assert(featureDetector);
        CV_Assert(descriptorExtractor);
    }

protected:

    void run(int)
    {
        const string imageFilename = string(ts->get_data_path()) + IMAGE_TSUKUBA;

        // Read test data
        Mat image0 = imread(imageFilename), image1, mask1;
        if(image0.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints0;
        Mat descriptors0;
        featureDetector->detect(image0, keypoints0);
        removeVerySmallKeypoints(keypoints0);
        if(keypoints0.size() < 15)
            CV_Error(Error::StsAssert, "Detector gives too few points in a test image\n");
        descriptorExtractor->compute(image0, keypoints0, descriptors0);

        BFMatcher bfmatcher(normType);

        const float minIntersectRatio = 0.5f;
        const int maxAngle = 360, angleStep = 15;
        for(int angle = 0; angle < maxAngle; angle += angleStep)
        {
            Mat H = rotateImage(image0, static_cast<float>(angle), image1, mask1);

            vector<KeyPoint> keypoints1;
            rotateKeyPoints(keypoints0, H, static_cast<float>(angle), keypoints1);
            Mat descriptors1;
            descriptorExtractor->compute(image1, keypoints1, descriptors1);

            vector<DMatch> descMatches;
            bfmatcher.match(descriptors0, descriptors1, descMatches);

            int descInliersCount = 0;
            for(size_t m = 0; m < descMatches.size(); m++)
            {
                const KeyPoint& transformed_p0 = keypoints1[descMatches[m].queryIdx];
                const KeyPoint& p1 = keypoints1[descMatches[m].trainIdx];
                if(calcIntersectRatio(transformed_p0.pt, 0.5f * transformed_p0.size,
                                      p1.pt, 0.5f * p1.size) >= minIntersectRatio)
                {
                    descInliersCount++;
                }
            }

            float descInliersRatio = static_cast<float>(descInliersCount) / keypoints0.size();
            if(descInliersRatio < minDescInliersRatio)
            {
                ts->printf(cvtest::TS::LOG, "Incorrect descInliersRatio: curr = %f, min = %f.\n",
                           descInliersRatio, minDescInliersRatio);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
#if SHOW_DEBUG_LOG
            std::cout << "descInliersRatio " << static_cast<float>(descInliersCount) / keypoints0.size() << std::endl;
#endif
        }
        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    int normType;
    float minDescInliersRatio;
};


class DetectorScaleInvarianceTest : public cvtest::BaseTest
{
public:
    DetectorScaleInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                float _minKeyPointMatchesRatio,
                                float _minScaleInliersRatio) :
        featureDetector(_featureDetector),
        minKeyPointMatchesRatio(_minKeyPointMatchesRatio),
        minScaleInliersRatio(_minScaleInliersRatio)
    {
        CV_Assert(featureDetector);
    }

protected:

    void run(int)
    {
        const string imageFilename = string(ts->get_data_path()) + IMAGE_BIKES;

        // Read test data
        Mat image0 = imread(imageFilename);
        if(image0.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints0;
        featureDetector->detect(image0, keypoints0);
        removeVerySmallKeypoints(keypoints0);
        if(keypoints0.size() < 15)
            CV_Error(Error::StsAssert, "Detector gives too few points in a test image\n");

        for(int scaleIdx = 1; scaleIdx <= 3; scaleIdx++)
        {
            float scale = 1.f + scaleIdx * 0.5f;
            Mat image1;
            resize(image0, image1, Size(), 1./scale, 1./scale);

            vector<KeyPoint> keypoints1, osiKeypoints1; // osi - original size image
            featureDetector->detect(image1, keypoints1);
            removeVerySmallKeypoints(keypoints1);
            if(keypoints1.size() < 15)
                CV_Error(Error::StsAssert, "Detector gives too few points in a test image\n");

            if(keypoints1.size() > keypoints0.size())
            {
                ts->printf(cvtest::TS::LOG, "Strange behavior of the detector. "
                    "It gives more points count in an image of the smaller size.\n"
                    "original size (%d, %d), keypoints count = %d\n"
                    "reduced size (%d, %d), keypoints count = %d\n",
                    image0.cols, image0.rows, keypoints0.size(),
                    image1.cols, image1.rows, keypoints1.size());
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            scaleKeyPoints(keypoints1, osiKeypoints1, scale);

            vector<DMatch> matches;
            // image1 is query image (it's reduced image0)
            // image0 is train image
            matchKeyPoints(osiKeypoints1, Mat(), keypoints0, matches);

            const float minIntersectRatio = 0.5f;
            int keyPointMatchesCount = 0;
            int scaleInliersCount = 0;

            for(size_t m = 0; m < matches.size(); m++)
            {
                if(matches[m].distance < minIntersectRatio)
                    continue;

                keyPointMatchesCount++;

                // Check does this inlier have consistent sizes
                const float maxSizeDiff = 0.8f;//0.9f; // grad
                float size0 = keypoints0[matches[m].trainIdx].size;
                float size1 = osiKeypoints1[matches[m].queryIdx].size;
                CV_Assert(size0 > 0 && size1 > 0);
                if(std::min(size0, size1) > maxSizeDiff * std::max(size0, size1))
                    scaleInliersCount++;
            }

            float keyPointMatchesRatio = static_cast<float>(keyPointMatchesCount) / keypoints1.size();
            if(keyPointMatchesRatio < minKeyPointMatchesRatio)
            {
                ts->printf(cvtest::TS::LOG, "Incorrect keyPointMatchesRatio: curr = %f, min = %f.\n",
                           keyPointMatchesRatio, minKeyPointMatchesRatio);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            if(keyPointMatchesCount)
            {
                float scaleInliersRatio = static_cast<float>(scaleInliersCount) / keyPointMatchesCount;
                if(scaleInliersRatio < minScaleInliersRatio)
                {
                    ts->printf(cvtest::TS::LOG, "Incorrect scaleInliersRatio: curr = %f, min = %f.\n",
                               scaleInliersRatio, minScaleInliersRatio);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
#if SHOW_DEBUG_LOG
            std::cout << "keyPointMatchesRatio - " << keyPointMatchesRatio
                << " - scaleInliersRatio " << static_cast<float>(scaleInliersCount) / keyPointMatchesCount << std::endl;
#endif
        }
        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector;
    float minKeyPointMatchesRatio;
    float minScaleInliersRatio;
};

class DescriptorScaleInvarianceTest : public cvtest::BaseTest
{
public:
    DescriptorScaleInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                const Ptr<DescriptorExtractor>& _descriptorExtractor,
                                int _normType,
                                float _minDescInliersRatio) :
        featureDetector(_featureDetector),
        descriptorExtractor(_descriptorExtractor),
        normType(_normType),
        minDescInliersRatio(_minDescInliersRatio)
    {
        CV_Assert(featureDetector);
        CV_Assert(descriptorExtractor);
    }

protected:

    void run(int)
    {
        const string imageFilename = string(ts->get_data_path()) + IMAGE_BIKES;

        // Read test data
        Mat image0 = imread(imageFilename);
        if(image0.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints0;
        featureDetector->detect(image0, keypoints0);
        removeVerySmallKeypoints(keypoints0);
        if(keypoints0.size() < 15)
            CV_Error(Error::StsAssert, "Detector gives too few points in a test image\n");
        Mat descriptors0;
        descriptorExtractor->compute(image0, keypoints0, descriptors0);

        BFMatcher bfmatcher(normType);
        for(int scaleIdx = 1; scaleIdx <= 3; scaleIdx++)
        {
            float scale = 1.f + scaleIdx * 0.5f;

            Mat image1;
            resize(image0, image1, Size(), 1./scale, 1./scale);

            vector<KeyPoint> keypoints1;
            scaleKeyPoints(keypoints0, keypoints1, 1.0f/scale);
            Mat descriptors1;
            descriptorExtractor->compute(image1, keypoints1, descriptors1);

            vector<DMatch> descMatches;
            bfmatcher.match(descriptors0, descriptors1, descMatches);

            const float minIntersectRatio = 0.5f;
            int descInliersCount = 0;
            for(size_t m = 0; m < descMatches.size(); m++)
            {
                const KeyPoint& transformed_p0 = keypoints0[descMatches[m].queryIdx];
                const KeyPoint& p1 = keypoints0[descMatches[m].trainIdx];
                if(calcIntersectRatio(transformed_p0.pt, 0.5f * transformed_p0.size,
                                      p1.pt, 0.5f * p1.size) >= minIntersectRatio)
                {
                    descInliersCount++;
                }
            }

            float descInliersRatio = static_cast<float>(descInliersCount) / keypoints0.size();
            if(descInliersRatio < minDescInliersRatio)
            {
                ts->printf(cvtest::TS::LOG, "Incorrect descInliersRatio: curr = %f, min = %f.\n",
                           descInliersRatio, minDescInliersRatio);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
#if SHOW_DEBUG_LOG
            std::cout << "descInliersRatio " << static_cast<float>(descInliersCount) / keypoints0.size() << std::endl;
#endif
        }
        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    int normType;
    float minKeyPointMatchesRatio;
    float minDescInliersRatio;
};

// Tests registration

/*
 * Detector's rotation invariance check
 */
TEST(Features2d_RotationInvariance_Detector_SURF, regression)
{
    DetectorRotationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"),
                                        0.44f,
                                        0.76f);
    test.safe_run();
}

TEST(Features2d_RotationInvariance_Detector_SIFT, DISABLED_regression)
{
    DetectorRotationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"),
                                        0.45f,
                                        0.70f);
    test.safe_run();
}

/*
 * Descriptors's rotation invariance check
 */
TEST(Features2d_RotationInvariance_Descriptor_SURF, regression)
{
    DescriptorRotationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"),
                                          Algorithm::create<DescriptorExtractor>("Feature2D.SURF"),
                                          NORM_L1,
                                          0.83f);
    test.safe_run();
}

TEST(Features2d_RotationInvariance_Descriptor_SIFT, regression)
{
    DescriptorRotationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"),
                                          Algorithm::create<DescriptorExtractor>("Feature2D.SIFT"),
                                          NORM_L1,
                                          0.98f);
    test.safe_run();
}

/*
 * Detector's scale invariance check
 */
TEST(Features2d_ScaleInvariance_Detector_SURF, regression)
{
    DetectorScaleInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"),
                                     0.64f,
                                     0.84f);
    test.safe_run();
}

TEST(Features2d_ScaleInvariance_Detector_SIFT, regression)
{
    DetectorScaleInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"),
                                     0.69f,
                                     0.99f);
    test.safe_run();
}

/*
 * Descriptor's scale invariance check
 */
TEST(Features2d_ScaleInvariance_Descriptor_SURF, regression)
{
    DescriptorScaleInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"),
                                       Algorithm::create<DescriptorExtractor>("Feature2D.SURF"),
                                       NORM_L1,
                                       0.61f);
    test.safe_run();
}

TEST(Features2d_ScaleInvariance_Descriptor_SIFT, regression)
{
    DescriptorScaleInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"),
                                       Algorithm::create<DescriptorExtractor>("Feature2D.SIFT"),
                                       NORM_L1,
                                       0.78f);
    test.safe_run();
}


TEST(Features2d_RotationInvariance2_Detector_SURF, regression)
{
    Mat cross(100, 100, CV_8UC1, Scalar(255));
    line(cross, Point(30, 50), Point(69, 50), Scalar(100), 3);
    line(cross, Point(50, 30), Point(50, 69), Scalar(100), 3);

    SURF surf(8000., 3, 4, true, false);

    vector<KeyPoint> keypoints;

    surf(cross, noArray(), keypoints);

    ASSERT_EQ(keypoints.size(), (vector<KeyPoint>::size_type) 5);
    ASSERT_LT( fabs(keypoints[1].response - keypoints[2].response), 1e-6);
    ASSERT_LT( fabs(keypoints[1].response - keypoints[3].response), 1e-6);
    ASSERT_LT( fabs(keypoints[1].response - keypoints[4].response), 1e-6);
}
