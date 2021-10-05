// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(NMS, Accuracy)
{
    //reference results obtained using tf.image.non_max_suppression with iou_threshold=0.5
    std::string dataPath = findDataFile("dnn/nms_reference.yml");
    FileStorage fs(dataPath, FileStorage::READ);

    std::vector<Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> ref_indices;

    fs["boxes"] >> bboxes;
    fs["probs"] >> scores;
    fs["output"] >> ref_indices;

    const float nms_thresh = .5f;
    const float score_thresh = .01f;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, score_thresh, nms_thresh, indices);

    ASSERT_EQ(ref_indices.size(), indices.size());

    std::sort(indices.begin(), indices.end());
    std::sort(ref_indices.begin(), ref_indices.end());

    for(size_t i = 0; i < indices.size(); i++)
        ASSERT_EQ(indices[i], ref_indices[i]);
}

TEST(SoftNMS, Accuracy)
{
    //reference results are obtained using TF v2.7 tf.image.non_max_suppression_with_scores
    std::string dataPath = findDataFile("dnn/soft_nms_reference.yml");
    FileStorage fs(dataPath, FileStorage::READ);

    std::vector<Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> ref_indices;
    std::vector<float> ref_updated_scores;

    fs["boxes"] >> bboxes;
    fs["probs"] >> scores;
    fs["indices"] >> ref_indices;
    fs["updated_scores"] >> ref_updated_scores;

    std::vector<float> updated_scores;
    const float score_thresh = .01f;
    const float nms_thresh = .5f;
    std::vector<int> indices;
    const size_t top_k = 0;
    const float sigma = 1.; // sigma in TF is being multiplied by 2, so 0.5 should be passed there
    cv::dnn::softNMSBoxes(bboxes, scores, updated_scores, score_thresh, nms_thresh, indices, top_k, sigma);

    ASSERT_EQ(ref_indices.size(), indices.size());
    for(size_t i = 0; i < indices.size(); i++)
    {
        ASSERT_EQ(indices[i], ref_indices[i]);
    }

    ASSERT_EQ(ref_updated_scores.size(), updated_scores.size());
    for(size_t i = 0; i < updated_scores.size(); i++)
    {
        EXPECT_NEAR(updated_scores[i], ref_updated_scores[i], 1e-7);
    }
}

}} // namespace
