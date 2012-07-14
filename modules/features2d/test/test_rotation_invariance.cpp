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

#define SHOW_DEBUG_LOG 0

static
Mat generateHomography(float angle)
{
	// angle - rotation around Oz in degrees
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
	// angle - rotation around Oz in degrees
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
float calcCirclesIntersectArea(const Point2f& p0, float r0, const Point2f& p1, float r1)
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
    float intersectArea = calcCirclesIntersectArea(p0, r0, p1, r1);
    float unionArea = CV_PI * (r0 * r0 + r1 * r1) - intersectArea;
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
    perspectiveTransform(Mat(points0), points0t, H);

	matches.clear();
	vector<uchar> usedMask(keypoints1.size(), 0);
	for(size_t i0 = 0; i0 < keypoints0.size(); i0++)
	{
		int nearestPointIndex = -1;
        float maxIntersectRatio = -1.f;
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
				nearestPointIndex = i1;
            }
        }

		matches.push_back(DMatch(i0, nearestPointIndex, maxIntersectRatio));
		if(nearestPointIndex >= 0)
			usedMask[nearestPointIndex] = 1;
	}
}

class DetectorRotatationInvarianceTest : public cvtest::BaseTest
{
public:
    DetectorRotatationInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
                                     float _minKeyPointMatchesRatio,
                                     float _minAngleInliersRatio) :
        featureDetector(_featureDetector), 
		minKeyPointMatchesRatio(_minKeyPointMatchesRatio), 
		minAngleInliersRatio(_minAngleInliersRatio)
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

        const int maxAngle = 360, angleStep = 15;
        for(int angle = 0; angle < maxAngle; angle += angleStep)
        {
            Mat H = rotateImage(image0, angle, image1, mask1);

            vector<KeyPoint> keypoints1;
            featureDetector->detect(image1, keypoints1, mask1);

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

class DescriptorRotatationInvarianceTest : public cvtest::BaseTest
{
public:
    DescriptorRotatationInvarianceTest(const Ptr<FeatureDetector>& _featureDetector,
									   const Ptr<DescriptorExtractor>& _descriptorExtractor,
									   int _normType,
									   float _minKeyPointMatchesRatio,
									   float _minDescInliersRatio) :
        featureDetector(_featureDetector), 
		descriptorExtractor(_descriptorExtractor),
		normType(_normType),
		minKeyPointMatchesRatio(_minKeyPointMatchesRatio),
		minDescInliersRatio(_minDescInliersRatio)
    {
        CV_Assert(!featureDetector.empty());
		CV_Assert(!descriptorExtractor.empty());
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
		Mat descriptors0;
        featureDetector->detect(image0, keypoints0);
		descriptorExtractor->compute(image0, keypoints0, descriptors0);

        CV_Assert(keypoints0.size() > 15);
		BFMatcher bfmatcher(normType);

        const int maxAngle = 360, angleStep = 15;
        for(int angle = 0; angle < maxAngle; angle += angleStep)
        {
            Mat H = rotateImage(image0, angle, image1, mask1);

            vector<KeyPoint> keypoints1;
			Mat descriptors1;
            featureDetector->detect(image1, keypoints1, mask1);
			descriptorExtractor->compute(image1, keypoints1, descriptors1);

			vector<DMatch> descMatches;
			bfmatcher.match(descriptors0, descriptors1, descMatches);

			vector<DMatch> keyPointMatches;
			matchKeyPoints(keypoints0, H, keypoints1, keyPointMatches);

			const float minIntersectRatio = 0.5f;
			int keyPointMatchesCount = 0;
			for(size_t m = 0; m < keyPointMatches.size(); m++)
			{
				if(keyPointMatches[m].distance >= minIntersectRatio)
					keyPointMatchesCount++;
			}
			int descInliersCount = 0;
			for(size_t m = 0; m < descMatches.size(); m++)
            {
				int queryIdx = descMatches[m].queryIdx;
				if(keyPointMatches[queryIdx].distance >= minIntersectRatio &&
					descMatches[m].trainIdx == keyPointMatches[queryIdx].trainIdx)
					descInliersCount++;
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
				float descInliersRatio = static_cast<float>(descInliersCount) / keyPointMatchesCount;
				if(descInliersRatio < minDescInliersRatio)
                {
                    ts->printf(cvtest::TS::LOG, "Incorrect descInliersRatio: curr = %f, min = %f.\n",
                               descInliersRatio, minDescInliersRatio);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
#if SHOW_DEBUG_LOG
            std::cout << "keyPointMatchesRatio - " << keyPointMatchesRatio
   				<< " - descInliersRatio " << static_cast<float>(descInliersCount) / keyPointMatchesCount << std::endl;
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

// Detector's rotation invariance check
TEST(Features2d_RotationInvariance_Detector_ORB, regression)
{
    DetectorRotatationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.ORB"),
										  0.45f, 
										  0.75f);
    test.safe_run();
}

// Descriptors's rotation invariance check
TEST(Features2d_RotationInvariance_Descriptor_ORB, regression)
{
    DescriptorRotatationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.ORB"), 
											Algorithm::create<DescriptorExtractor>("Feature2D.ORB"), 
											NORM_HAMMING, 
											0.45f,
											0.53f);
    test.safe_run();
}

// TODO: uncomment test for FREAK when it will work
//TEST(Features2d_RotationInvariance_Descriptor_FREAK, regression)
//{
//    DescriptorRotatationInvarianceTest test(Algorithm::create<FeatureDetector>("Feature2D.ORB"), 
//											Algorithm::create<DescriptorExtractor>("Feature2D.FREAK"), 
//											NORM_HAMMING(?), 
//											0.45f,
//											0.?f);
//    test.safe_run();
//}