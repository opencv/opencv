// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat referenceMatMul(const Mat& A, const Mat& B, bool trans_a, float alpha)
{
    const int M = trans_a ? A.cols : A.rows;
    const int K = trans_a ? A.rows : A.cols;
    const int N = B.cols;
    Mat expected(M, N, CV_32F);

    for (int m = 0; m < M; m++)
    {
        float* dst = expected.ptr<float>(m);
        for (int n = 0; n < N; n++)
        {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
            {
                const float a = trans_a ? A.ptr<float>(k)[m] : A.ptr<float>(m)[k];
                acc += a * B.ptr<float>(k)[n];
            }
            dst[n] = alpha * acc;
        }
    }
    return expected;
}

// Exercises the fastGemmThin path (constant-B MatMul). M covers every register
// block width and the multi-block remainder, N covers full strips and partial
// column tails for any VLEN <= 512.
typedef testing::TestWithParam<tuple<int, int, int, int, float>> DNN_FastGemmThin;

TEST_P(DNN_FastGemmThin, MatMulAccuracy)
{
    int M = get<0>(GetParam()), N = get<1>(GetParam()), K = get<2>(GetParam());
    int trans_a = get<3>(GetParam());
    float alpha = get<4>(GetParam());

    RNG rng(0x5EED);
    Mat B(K, N, CV_32F);            rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
    Mat A(trans_a ? K : M, trans_a ? M : K, CV_32F);
    rng.fill(A, RNG::UNIFORM, -1.f, 1.f);

    LayerParams lp;
    lp.type = "MatMul";
    lp.name = "thin_matmul";
    lp.set("transA", trans_a != 0);
    lp.set("transB", false);
    lp.set("alpha", alpha);
    lp.blobs.push_back(B);

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setInputsNames(std::vector<String>{ "A" });
    net.setInput(A, "A");
    Mat actual = net.forward();
    Mat expected = referenceMatMul(A, B, trans_a != 0, alpha);

    EXPECT_LE(cv::norm(expected, actual, NORM_INF), 2e-5f * std::max(K, 1))
        << "M=" << M << ", N=" << N << ", K=" << K
        << ", trans_a=" << trans_a << ", alpha=" << alpha;
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNN_FastGemmThin, Combine(
    Values(1, 2, 3, 4, 5),
    Values(16, 19, 64, 67, 80, 83),
    Values(1, 3, 4, 7, 16, 64),
    Values(0, 1),
    Values(1.f, -0.5f)
));

}} // namespace
