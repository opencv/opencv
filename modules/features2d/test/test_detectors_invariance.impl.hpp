// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_invariance_utils.hpp"

namespace opencv_test { namespace {

#define SHOW_DEBUG_LOG 1

typedef tuple<std::string, Ptr<FeatureDetector>, float, float> String_FeatureDetector_Float_Float_t;


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

class DetectorInvariance : public TestWithParam<String_FeatureDetector_Float_Float_t>
{
protected:
    virtual void SetUp() {
        // Read test data
        const std::string filename = cvtest::TS::ptr()->get_data_path() + get<0>(GetParam());
        image0 = imread(filename);
        ASSERT_FALSE(image0.empty()) << "couldn't read input image";

        featureDetector = get<1>(GetParam());
        minKeyPointMatchesRatio = get<2>(GetParam());
        minInliersRatio = get<3>(GetParam());
    }

    Ptr<FeatureDetector> featureDetector;
    float minKeyPointMatchesRatio;
    float minInliersRatio;
    Mat image0;
};

typedef DetectorInvariance DetectorScaleInvariance;
typedef DetectorInvariance DetectorRotationInvariance;

TEST_P(DetectorRotationInvariance, rotation)
{
    Mat image1, mask1;
    const int borderSize = 16;
    Mat mask0(image0.size(), CV_8UC1, Scalar(0));
    mask0(Rect(borderSize, borderSize, mask0.cols - 2*borderSize, mask0.rows - 2*borderSize)).setTo(Scalar(255));

    vector<KeyPoint> keypoints0;
    featureDetector->detect(image0, keypoints0, mask0);
    EXPECT_GE(keypoints0.size(), 15u);

    const int maxAngle = 360, angleStep = 15;
    for(int angle = 0; angle < maxAngle; angle += angleStep)
    {
        Mat H = rotateImage(image0, mask0, static_cast<float>(angle), image1, mask1);

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
            ASSERT_FALSE(angle0 == -1 || angle1 == -1) << "Given FeatureDetector is not rotation invariant, it can not be tested here.";
            ASSERT_GE(angle0, 0.f);
            ASSERT_LT(angle0, 360.f);
            ASSERT_GE(angle1, 0.f);
            ASSERT_LT(angle1, 360.f);

            float rotAngle0 = angle0 + angle;
            if(rotAngle0 >= 360.f)
                rotAngle0 -= 360.f;

            float angleDiff = std::max(rotAngle0, angle1) - std::min(rotAngle0, angle1);
            angleDiff = std::min(angleDiff, static_cast<float>(360.f - angleDiff));
            ASSERT_GE(angleDiff, 0.f);
            bool isAngleCorrect = angleDiff < maxAngleDiff;
            if(isAngleCorrect)
                angleInliersCount++;
        }

        float keyPointMatchesRatio = static_cast<float>(keyPointMatchesCount) / keypoints0.size();
        EXPECT_GE(keyPointMatchesRatio, minKeyPointMatchesRatio) << "angle: " << angle;

        if(keyPointMatchesCount)
        {
            float angleInliersRatio = static_cast<float>(angleInliersCount) / keyPointMatchesCount;
            EXPECT_GE(angleInliersRatio, minInliersRatio) << "angle: " << angle;
        }
#if SHOW_DEBUG_LOG
        std::cout
            << "angle = " << angle
            << ", keypoints = " << keypoints1.size()
            << ", keyPointMatchesRatio = " << keyPointMatchesRatio
            << ", angleInliersRatio = " << (keyPointMatchesCount ? (static_cast<float>(angleInliersCount) / keyPointMatchesCount) : 0)
            << std::endl;
#endif
    }
}

TEST_P(DetectorScaleInvariance, scale)
{
    vector<KeyPoint> keypoints0;
    featureDetector->detect(image0, keypoints0);
    EXPECT_GE(keypoints0.size(),  15u);

    for(int scaleIdx = 1; scaleIdx <= 3; scaleIdx++)
    {
        float scale = 1.f + scaleIdx * 0.5f;
        Mat image1;
        resize(image0, image1, Size(), 1./scale, 1./scale, INTER_LINEAR_EXACT);

        vector<KeyPoint> keypoints1, osiKeypoints1; // osi - original size image
        featureDetector->detect(image1, keypoints1);
        EXPECT_GE(keypoints1.size(), 15u);
        EXPECT_LE(keypoints1.size(), keypoints0.size()) << "Strange behavior of the detector. "
                  "It gives more points count in an image of the smaller size.";

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
            ASSERT_GT(size0, 0);
            ASSERT_GT(size1, 0);
            if(std::min(size0, size1) > maxSizeDiff * std::max(size0, size1))
                scaleInliersCount++;
        }

        float keyPointMatchesRatio = static_cast<float>(keyPointMatchesCount) / keypoints1.size();
        EXPECT_GE(keyPointMatchesRatio, minKeyPointMatchesRatio);

        if(keyPointMatchesCount)
        {
            float scaleInliersRatio = static_cast<float>(scaleInliersCount) / keyPointMatchesCount;
            EXPECT_GE(scaleInliersRatio, minInliersRatio);
        }
#if SHOW_DEBUG_LOG
        std::cout
            << "scale = " << scale
            << ", keyPointMatchesRatio = " << keyPointMatchesRatio
            << ", scaleInliersRatio = " << (keyPointMatchesCount ? static_cast<float>(scaleInliersCount) / keyPointMatchesCount : 0)
            << std::endl;
#endif
    }
}

#undef SHOW_DEBUG_LOG
}} // namespace

namespace std {
using namespace opencv_test;
static inline void PrintTo(const String_FeatureDetector_Float_Float_t& v, std::ostream* os)
{
    *os << "(\"" << get<0>(v)
        << "\", " << get<2>(v)
        << ", " << get<3>(v)
        << ")";
}
} // namespace
