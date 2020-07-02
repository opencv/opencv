// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_invariance_utils.hpp"

namespace opencv_test { namespace {

#define SHOW_DEBUG_LOG 1

typedef tuple<std::string, Ptr<FeatureDetector>, Ptr<DescriptorExtractor>, float>
    String_FeatureDetector_DescriptorExtractor_Float_t;


static
void SetSuitableSIFTOctave(vector<KeyPoint>& keypoints,
                             int firstOctave = -1, int nOctaveLayers = 3, double sigma = 1.6)
{
    for (size_t i = 0; i < keypoints.size(); i++ )
    {
        int octv, layer;
        KeyPoint& kpt = keypoints[i];
        double octv_layer = std::log(kpt.size / sigma) / std::log(2.) - 1;
        octv = cvFloor(octv_layer);
        layer = cvRound( (octv_layer - octv) * nOctaveLayers );
        if (octv < firstOctave)
        {
            octv = firstOctave;
            layer = 0;
        }
        kpt.octave = (layer << 8) | (octv & 255);
    }
}

static
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

class DescriptorInvariance : public TestWithParam<String_FeatureDetector_DescriptorExtractor_Float_t>
{
protected:
    virtual void SetUp() {
        // Read test data
        const std::string filename = cvtest::TS::ptr()->get_data_path() + get<0>(GetParam());
        image0 = imread(filename);
        ASSERT_FALSE(image0.empty()) << "couldn't read input image";

        featureDetector = get<1>(GetParam());
        descriptorExtractor = get<2>(GetParam());
        minInliersRatio = get<3>(GetParam());
    }

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    float minInliersRatio;
    Mat image0;
};

typedef DescriptorInvariance DescriptorScaleInvariance;
typedef DescriptorInvariance DescriptorRotationInvariance;

TEST_P(DescriptorRotationInvariance, rotation)
{
    Mat image1, mask1;
    const int borderSize = 16;
    Mat mask0(image0.size(), CV_8UC1, Scalar(0));
    mask0(Rect(borderSize, borderSize, mask0.cols - 2*borderSize, mask0.rows - 2*borderSize)).setTo(Scalar(255));

    vector<KeyPoint> keypoints0;
    Mat descriptors0;
    featureDetector->detect(image0, keypoints0, mask0);
    std::cout << "Keypoints: " << keypoints0.size() << std::endl;
    EXPECT_GE(keypoints0.size(), 15u);
    descriptorExtractor->compute(image0, keypoints0, descriptors0);

    BFMatcher bfmatcher(descriptorExtractor->defaultNorm());

    const float minIntersectRatio = 0.5f;
    const int maxAngle = 360, angleStep = 15;
    for(int angle = 0; angle < maxAngle; angle += angleStep)
    {
        Mat H = rotateImage(image0, mask0, static_cast<float>(angle), image1, mask1);

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
        EXPECT_GE(descInliersRatio, minInliersRatio);
#if SHOW_DEBUG_LOG
        std::cout
            << "angle = " << angle
            << ", inliers = " << descInliersCount
            << ", descInliersRatio = " << static_cast<float>(descInliersCount) / keypoints0.size()
            << std::endl;
#endif
    }
}


TEST_P(DescriptorScaleInvariance, scale)
{
    vector<KeyPoint> keypoints0;
    featureDetector->detect(image0, keypoints0);
    std::cout << "Keypoints: " << keypoints0.size() << std::endl;
    EXPECT_GE(keypoints0.size(), 15u);
    Mat descriptors0;
    descriptorExtractor->compute(image0, keypoints0, descriptors0);

    BFMatcher bfmatcher(descriptorExtractor->defaultNorm());
    for(int scaleIdx = 1; scaleIdx <= 3; scaleIdx++)
    {
        float scale = 1.f + scaleIdx * 0.5f;

        Mat image1;
        resize(image0, image1, Size(), 1./scale, 1./scale, INTER_LINEAR_EXACT);

        vector<KeyPoint> keypoints1;
        scaleKeyPoints(keypoints0, keypoints1, 1.0f/scale);
        if (featureDetector->getDefaultName() == "Feature2D.SIFT")
        {
            SetSuitableSIFTOctave(keypoints1);
        }
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
        EXPECT_GE(descInliersRatio, minInliersRatio);
#if SHOW_DEBUG_LOG
        std::cout
            << "scale = " << scale
            << ", inliers = " << descInliersCount
            << ", descInliersRatio = " << static_cast<float>(descInliersCount) / keypoints0.size()
            << std::endl;
#endif
    }
}

#undef SHOW_DEBUG_LOG
}} // namespace

namespace std {
using namespace opencv_test;
static inline void PrintTo(const String_FeatureDetector_DescriptorExtractor_Float_t& v, std::ostream* os)
{
    *os << "(\"" << get<0>(v)
        << "\", " << get<3>(v)
        << ")";
}
} // namespace
