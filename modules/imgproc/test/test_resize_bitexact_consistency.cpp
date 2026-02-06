#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Resize, LinearBitExactConsistencyAcrossChannels)
{
    RNG rng(42);

    int src_w = 4160, src_h = 3120;
    float scale = std::min(640.f / src_w, 640.f / src_h);
    int dst_w = static_cast<int>(std::round(scale * src_w));
    int dst_h = static_cast<int>(std::round(scale * src_h));

    for (int cn = 2; cn <= 3; cn++)
    {
        Mat src(src_h, src_w, CV_8UC(cn));
        rng.fill(src, RNG::UNIFORM, 0, 256);

        Mat directResult;
        resize(src, directResult, Size(dst_w, dst_h), 0, 0, INTER_LINEAR);

        std::vector<Mat> channels;
        split(src, channels);
        std::vector<Mat> resizedChannels(cn);
        for (int c = 0; c < cn; c++)
            resize(channels[c], resizedChannels[c], Size(dst_w, dst_h), 0, 0, INTER_LINEAR);
        Mat splitMergeResult;
        merge(resizedChannels, splitMergeResult);

        Mat diff;
        absdiff(directResult, splitMergeResult, diff);
        double maxDiff;
        minMaxLoc(diff.reshape(1), nullptr, &maxDiff);

        EXPECT_EQ(maxDiff, 0)
            << "Resize INTER_LINEAR inconsistency for cn=" << cn
            << " between direct and split-merge";
    }
}

}} // namespace