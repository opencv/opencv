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
    Values(1, 2, 3, 4, 5, 6, 9, 12),
    Values(16, 19, 64, 67, 80, 83),
    Values(1, 3, 4, 7, 16, 64),
    Values(0, 1),
    Values(1.f, -0.5f)
));

static Mat runGemm(const Mat& A, const Mat& B, const Mat& C, bool constB, bool transB, float alpha, float beta)
{
    LayerParams lp;
    lp.type = "Gemm";
    lp.name = "few_rows_gemm";
    lp.set("transB", transB);
    lp.set("alpha", alpha);
    lp.set("beta", beta);
    lp.set("constB", constB);
    lp.set("constC", constB);
    lp.set("have_bias", true);
    lp.set("real_ndims_C", 2);
    if (constB)
    {
        // As the ONNX importer does: constant B and C are both blobs.
        lp.blobs.push_back(B);
        lp.blobs.push_back(C);
    }
    Ptr<Layer> layer = LayerFactory::createLayerInstance(lp.type, lp);
    std::vector<Mat> inputs = constB ? std::vector<Mat>{A} : std::vector<Mat>{A, B, C}, outputs;
    runLayer(layer, inputs, outputs);
    return outputs[0];
}

// Gemm with 1..12 rows: constant B (packed strips) and dynamic B (read in place),
// with column tails around the 16- and 32-column blocks, a bias and alpha/beta.
typedef testing::TestWithParam<tuple<int, int, int, bool, bool>> DNN_FastGemmFewRows;

TEST_P(DNN_FastGemmFewRows, GemmAccuracy)
{
    const int M = get<0>(GetParam()), N = get<1>(GetParam()), K = get<2>(GetParam());
    const bool constB = get<3>(GetParam()), transB = get<4>(GetParam());
    const float alpha = 0.75f, beta = -0.25f;

    RNG& rng = theRNG();
    Mat A(M, K, CV_32F), B(transB ? N : K, transB ? K : N, CV_32F), C(M, N, CV_32F);
    rng.fill(A, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(C, RNG::UNIFORM, -1.f, 1.f);

    Mat actual = runGemm(A, B, C, constB, transB, alpha, beta), expected;
    gemm(A, B, alpha, C, beta, expected, transB ? GEMM_2_T : 0);
    EXPECT_LE(cvtest::norm(expected, actual, NORM_INF), 2e-5 * std::max(K, 1))
        << "M=" << M << ", N=" << N << ", K=" << K << ", constB=" << constB << ", transB=" << transB;
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNN_FastGemmFewRows, Combine(
    Values(1, 2, 3, 5, 7, 10, 12),
    Values(5, 16, 31, 32, 33, 67),
    Values(1, 7, 64, 129),
    testing::Bool(),
    testing::Bool()
));

// MatMul with a constant B takes fastGemmThin with or without MLAS.
static Mat runMatMulConstB(const Mat& A, const Mat& B)
{
    LayerParams lp;
    lp.type = "MatMul";
    lp.name = "few_rows_matmul";
    lp.blobs.push_back(B);
    Ptr<Layer> layer = LayerFactory::createLayerInstance(lp.type, lp);
    std::vector<Mat> inputs = {A}, outputs;
    runLayer(layer, inputs, outputs);
    return outputs[0];
}

// Each output is computed by one task, so the thread count must not change any bit of the result.
// Dynamic B goes through Gemm, constant B through MatMul, so both reach the new code in MLAS builds too.
TEST(DNN_FastGemm, FewRowsSameResultForAnyThreadCount)
{
    const int N = 3072, K = 768;
    const int nthreads = getNumThreads();
    RNG& rng = theRNG();
    for (int M : {1, 2, 5, 9, 12})
    {
        Mat A(M, K, CV_32F), B(K, N, CV_32F), C(M, N, CV_32F);
        rng.fill(A, RNG::UNIFORM, -1.f, 1.f);
        rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
        rng.fill(C, RNG::UNIFORM, -1.f, 1.f);
        for (bool constB : {false, true})
        {
            setNumThreads(1);
            Mat one = constB ? runMatMulConstB(A, B) : runGemm(A, B, C, false, false, 1.f, 1.f);
            setNumThreads(std::max(nthreads, 8));
            Mat many = constB ? runMatMulConstB(A, B) : runGemm(A, B, C, false, false, 1.f, 1.f);
            setNumThreads(nthreads);
            EXPECT_EQ(0, cvtest::norm(one, many, NORM_INF)) << "M=" << M << ", constB=" << constB;
        }
    }
}

// An empty output (N == 0) must not reach the column split.
TEST(DNN_FastGemm, FewRowsEmptyOutput)
{
    const int M = 3, K = 8;
    Mat A(M, K, CV_32F), B(K, 0, CV_32F), C(M, 0, CV_32F);
    theRNG().fill(A, RNG::UNIFORM, -1.f, 1.f);
    Mat Y = runGemm(A, B, C, false, false, 1.f, 1.f);
    EXPECT_EQ(Y.total(), (size_t)0);
}

}} // namespace
