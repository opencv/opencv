// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include "opencv2/features/feature_extractor.hpp"
#include "opencv2/features/feature_matcher.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

#include <cstdlib>

namespace opencv_test
{
    namespace
    {

        TEST(Features_FeaturePipeline, TraditionalWrapper)
        {
            Mat img(128, 128, CV_8UC1, Scalar::all(0));
            randu(img, Scalar::all(0), Scalar::all(255));

            Ptr<cv::features::FeatureExtractor> extractor = cv::features::TraditionalFeatureExtractor::create(ORB::create());
            ASSERT_FALSE(extractor.empty());

            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            extractor->extract(img, keypoints, descriptors);

            ASSERT_GE(descriptors.rows, 0);

            Ptr<cv::features::FeatureMatcher> matcher =
                cv::features::TraditionalFeatureMatcher::create(DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING));
            ASSERT_FALSE(matcher.empty());

            std::vector<DMatch> matches;
            matcher->match(descriptors, descriptors, matches);
            if (!descriptors.empty())
            {
                ASSERT_FALSE(matches.empty());
            }
        }

#ifdef HAVE_OPENCV_DNN

        TEST(Features_FeaturePipeline, SuperPointLightGlue)
        {
            const char *superpointModel = std::getenv("OPENCV_TEST_SUPERPOINT_ONNX");
            const char *lightglueModel = std::getenv("OPENCV_TEST_LIGHTGLUE_ONNX");
            const char *imagePath = std::getenv("OPENCV_TEST_IMAGE");

            if (!superpointModel || !lightglueModel || !imagePath)
                return;

            Mat img = imread(imagePath, IMREAD_COLOR);
            if (img.empty())
                return;

            cv::features::SuperPoint::Params spParams;
            spParams.modelPath = String(superpointModel);
            spParams.dnnEngine = dnn::ENGINE_ORT;
            spParams.inputSize = Size(640, 640);

            Ptr<cv::features::FeatureExtractor> extractor = cv::features::SuperPoint::create(spParams);
            ASSERT_FALSE(extractor.empty());

            std::vector<KeyPoint> kpts0, kpts1;
            Mat desc0, desc1;
            extractor->extract(img, kpts0, desc0);
            extractor->extract(img, kpts1, desc1);

            ASSERT_FALSE(kpts0.empty());
            ASSERT_EQ(static_cast<int>(kpts0.size()), desc0.rows);
            ASSERT_EQ(static_cast<int>(kpts1.size()), desc1.rows);

            std::vector<Point2f> pts0, pts1;
            KeyPoint::convert(kpts0, pts0);
            KeyPoint::convert(kpts1, pts1);

            cv::features::LightGlue::Params lgParams;
            lgParams.modelPath = String(lightglueModel);
            lgParams.dnnEngine = dnn::ENGINE_ORT;

            Ptr<cv::features::FeatureMatcher> matcher = cv::features::LightGlue::create(lgParams);
            ASSERT_FALSE(matcher.empty());

            std::vector<DMatch> matches;
            matcher->match(Mat(pts0), desc0, Mat(pts1), desc1, matches, noArray(), img.size(), img.size());

            for (size_t i = 0; i < matches.size(); ++i)
            {
                EXPECT_GE(matches[i].queryIdx, 0);
                EXPECT_LT(matches[i].queryIdx, static_cast<int>(kpts0.size()));
                EXPECT_GE(matches[i].trainIdx, 0);
                EXPECT_LT(matches[i].trainIdx, static_cast<int>(kpts1.size()));
            }
        }

#endif

    }
} // namespace
