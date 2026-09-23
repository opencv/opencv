// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// update() builds a tracks-by-detections cost matrix and solves it, so both grow with the object
// count. Mode 0 is IoU only, 1 adds the chi-square gate, 2 adds appearance on top.
typedef TestBaseWithParam< tuple<int, int> > Objects_Mode_MultiTracker;

PERF_TEST_P_(Objects_Mode_MultiTracker, update)
{
    const int nObjects = get<0>(GetParam());
    const int mode = get<1>(GetParam());
    const int nFrames = 20;
    const int featureDim = 128;

    MultiTracker::Params params;
    params.minHits = 1;                                     // every track is live from frame one
    params.gatingThreshold = mode >= 1 ? 9.4877f : 0.0f;
    params.embeddingWeight = mode >= 2 ? 0.4f : 0.0f;

    // One object per grid cell, each walking right a pixel a frame. Every track keeps a detection
    // of its own, so the association runs at full size on every frame.
    const int cols = cvCeil(std::sqrt((double)nObjects));
    std::vector<Rect2d> boxes((size_t)nObjects);
    std::vector<float> scores((size_t)nObjects, 0.9f);
    std::vector<int> classes((size_t)nObjects, 0);

    Mat embeddings;
    if (mode >= 2)
    {
        embeddings.create(nObjects, featureDim, CV_32F);
        RNG rng(0x900D5EED);
        rng.fill(embeddings, RNG::UNIFORM, 0.f, 1.f);
        for (int i = 0; i < nObjects; i++)
        {
            Mat r = embeddings.row(i);
            normalize(r, r, 1.0, 0.0, NORM_L2);
        }
    }

    std::vector<int> ids, outClasses;
    std::vector<Rect2d> outBoxes;

    TEST_CYCLE()
    {
        Ptr<MultiTracker> tracker = MultiTracker::create(params);
        for (int f = 0; f < nFrames; f++)
        {
            for (int i = 0; i < nObjects; i++)
                boxes[(size_t)i] = Rect2d(64.0 * (i % cols) + f, 64.0 * (i / cols), 16.0, 32.0);

            if (mode >= 2)
                tracker->update(boxes, scores, classes, embeddings, ids, outBoxes, outClasses);
            else
                tracker->update(boxes, scores, classes, ids, outBoxes, outClasses);
        }
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Objects_Mode_MultiTracker,
                        testing::Combine(
                            testing::Values(10, 50, 200),
                            testing::Values(0, 1, 2)));

} // namespace opencv_test
