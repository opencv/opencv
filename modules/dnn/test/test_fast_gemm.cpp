// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "../src/layers/cpu_kernels/fast_gemm.hpp"

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

    RNG& rng = theRNG();
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

struct ScopedNumThreads
{
    explicit ScopedNumThreads(int n) : saved(getNumThreads()) { setNumThreads(n); }
    ~ScopedNumThreads() { setNumThreads(saved); }
    const int saved;
};

static Mat forwardWithThreads(LayerParams lp, std::vector<Mat> inputs, int threads)
{
    ScopedNumThreads numThreads(threads);
    Ptr<Layer> layer = LayerFactory::createLayerInstance(lp.type, lp);
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    return outputs[0];
}

// The work is split into more tiles when more threads are available; no output value may change.
TEST(DNN_FastGemm, SameResultForAnyThreadCount)
{
    ScopedNoMlas noMlas;
    RNG& rng = theRNG();
    Mat A(50, 256, CV_32F), B(256, 1024, CV_32F);
    Mat Ab({3, 50, 64}, CV_32F), Bb({3, 64, 200}, CV_32F);
    rng.fill(A, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(Ab, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(Bb, RNG::UNIFORM, -1.f, 1.f);

    LayerParams gemm, gemmPacked, matmul, matmulPacked;
    gemm.type = "Gemm";
    gemm.name = "gemm";
    gemmPacked = gemm;
    gemmPacked.name = "gemm_packed";
    gemmPacked.blobs.push_back(B);
    matmul.type = "MatMul";
    matmul.name = "matmul_batch";
    matmulPacked = matmul;
    matmulPacked.name = "matmul_batch_packed";
    matmulPacked.blobs.push_back(Bb);

    const struct { LayerParams lp; std::vector<Mat> inputs; } cases[] = {
        { gemm, {A, B} }, { gemmPacked, {A} }, { matmul, {Ab, Bb} }, { matmulPacked, {Ab} }
    };
    for (const auto& c : cases)
    {
        Mat single = forwardWithThreads(c.lp, c.inputs, 1);
        Mat multi = forwardWithThreads(c.lp, c.inputs, 8);
        EXPECT_EQ(0., cvtest::norm(single, multi, NORM_INF)) << c.lp.name;
    }
}

// Element (i, k) of A is a[i*a0 + k*a1] and (k, j) of B is b[k*b0 + j*b1]; C0 is read only when beta != 0.
static Mat gemmReference(int M, int N, int K, float alpha, const float* a, int a0, int a1,
                         const float* b, int b0, int b1, float beta, const Mat& C0)
{
    Mat ref(M, N, CV_32F);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)a[(size_t)i * a0 + (size_t)k * a1] * b[(size_t)k * b0 + (size_t)j * b1];
            ref.at<float>(i, j) = (float)(alpha * s + (beta != 0.f ? beta * C0.at<float>(i, j) : 0.));
        }
    return ref;
}

// NaN when beta == 0, so a kernel that reads C anyway fails the comparison.
static Mat initialC(int rows, int cols, float beta, RNG& rng)
{
    Mat C(rows, cols, CV_32F);
    if (beta == 0.f)
        C.setTo(Scalar::all(std::numeric_limits<float>::quiet_NaN()));
    else
        rng.fill(C, RNG::UNIFORM, -1.f, 1.f);
    return C;
}

template<typename F> static Mat runWithThreads(int threads, F f)
{
    ScopedNumThreads numThreads(threads);
    return f();
}

static void checkGemm(const Mat& ref, const Mat& single, const Mat& multi, int K, const std::string& what)
{
    EXPECT_LE(cvtest::norm(ref, multi, NORM_INF), 2e-5 * std::max(K, 1)) << what;
    EXPECT_EQ(0., cvtest::norm(single, multi, NORM_INF)) << what << ": 1 and 8 threads differ";
}

static const float gemmAlpha[] = { 1.f, 0.5f, -1.5f }, gemmBeta[] = { 0.f, 1.f, 0.25f };

// The blocked kernels, single and batched, with every transpose, alpha/beta pair, and row,
// column and K tails. M > 12 keeps the products off the thin path.
typedef testing::TestWithParam<tuple<Vec3i, int, int> > DNN_FastGemmKernels;
TEST_P(DNN_FastGemmKernels, accuracy)
{
    ScopedNoMlas noMlas;
    const Vec3i shapeMNK = get<0>(GetParam());
    const int M = shapeMNK[0], N = shapeMNK[1], K = shapeMNK[2], batch = 3;
    const bool ta = (get<1>(GetParam()) & 1) != 0, tb = (get<1>(GetParam()) & 2) != 0;
    const float alpha = gemmAlpha[get<2>(GetParam())], beta = gemmBeta[get<2>(GetParam())];
    const int a0 = ta ? 1 : K, a1 = ta ? M : 1, b0 = tb ? 1 : N, b1 = tb ? K : 1;
    const int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;

    RNG& rng = theRNG();
    Mat A(batch * ar, ac, CV_32F), B(batch * br, bc, CV_32F);
    rng.fill(A, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
    Mat C0 = initialC(batch * M, N, beta, rng);
    FastGemmOpt opt;
    opt.init();

    std::vector<size_t> aofs(batch), bofs(batch), cofs(batch);
    for (int i = 0; i < batch; i++)
    {
        aofs[i] = (size_t)i * ar * ac;
        bofs[i] = (size_t)i * br * bc;
        cofs[i] = (size_t)i * M * N;
    }

    Mat ref(batch * M, N, CV_32F);
    for (int i = 0; i < batch; i++)
        gemmReference(M, N, K, alpha, A.ptr<float>(i * ar), a0, a1, B.ptr<float>(i * br), b0, b1,
                      beta, C0.rowRange(i * M, (i + 1) * M)).copyTo(ref.rowRange(i * M, (i + 1) * M));
    Mat ref0 = ref.rowRange(0, M);

    auto plain = [&]() {
        Mat C = C0.rowRange(0, M).clone();
        fastGemm(ta, tb, ar, ac, br, bc, alpha, A.ptr<float>(), ac, 1, B.ptr<float>(), bc, 1,
                 beta, C.ptr<float>(), N, opt);
        return C;
    };
    auto batched = [&]() {
        Mat C = C0.clone();
        fastGemmBatch(batch, aofs.data(), bofs.data(), cofs.data(), M, N, K, alpha, A.ptr<float>(), a0, a1,
                      B.ptr<float>(), b0, b1, beta, C.ptr<float>(), N, opt);
        return C;
    };
    checkGemm(ref0, runWithThreads(1, plain), runWithThreads(8, plain), K, "plain");
    checkGemm(ref, runWithThreads(1, batched), runWithThreads(8, batched), K, "batched");
}

INSTANTIATE_TEST_CASE_P(/**/, DNN_FastGemmKernels, Combine(
    Values(Vec3i(13, 1000, 300), Vec3i(50, 520, 257), Vec3i(197, 190, 64), Vec3i(64, 37, 129)),
    Values(0, 1, 2, 3),  // bit 0: A transposed, bit 1: B transposed
    Values(0, 1, 2)      // alpha/beta pair
));

// 1-4 rows times a transposed B, computed directly from B: column tails (N % 4) and K tails
// (K % vector width), with every alpha/beta pair.
typedef testing::TestWithParam<tuple<int, int, int> > DNN_FastGemmFewRows;
TEST_P(DNN_FastGemmFewRows, transposedB)
{
    ScopedNoMlas noMlas;
    const int M = get<0>(GetParam()), N = get<1>(GetParam()), K = get<2>(GetParam()), batch = 2;
    RNG& rng = theRNG();
    Mat A(batch * M, K, CV_32F), B(batch * N, K, CV_32F);
    rng.fill(A, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(B, RNG::UNIFORM, -1.f, 1.f);
    FastGemmOpt opt;
    opt.init();
    std::vector<size_t> aofs = {0, (size_t)M * K}, bofs = {0, (size_t)N * K}, cofs = {0, (size_t)M * N};

    for (int ab = 0; ab < 3; ab++)
    {
        const float alpha = gemmAlpha[ab], beta = gemmBeta[ab];
        Mat C0 = initialC(batch * M, N, beta, rng);
        Mat ref(batch * M, N, CV_32F);
        for (int i = 0; i < batch; i++)
            gemmReference(M, N, K, alpha, A.ptr<float>(i * M), K, 1, B.ptr<float>(i * N), 1, K,
                          beta, C0.rowRange(i * M, (i + 1) * M)).copyTo(ref.rowRange(i * M, (i + 1) * M));
        auto run = [&]() {
            Mat C = C0.clone();
            fastGemmBatch(batch, aofs.data(), bofs.data(), cofs.data(), M, N, K, alpha, A.ptr<float>(), K, 1,
                          B.ptr<float>(), 1, K, beta, C.ptr<float>(), N, opt);
            return C;
        };
        checkGemm(ref, runWithThreads(1, run), runWithThreads(8, run), K, format("alpha=%g beta=%g", alpha, beta));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, DNN_FastGemmFewRows, Combine(
    Values(1, 2, 3, 4),
    Values(1, 5, 67, 700),
    Values(1, 7, 9, 300)
));

}} // namespace
