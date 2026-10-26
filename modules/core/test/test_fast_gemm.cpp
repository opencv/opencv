// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "test_precomp.hpp"
#include "opencv2/core/private/fast_gemm.hpp"

namespace opencv_test { namespace {

// Random matrix whose rows are padded, so that the step is larger than the row width.
static Mat randMat(int rows, int cols, RNG& rng, int pad)
{
    Mat m(rows, cols + pad, CV_32F);
    rng.fill(m, RNG::UNIFORM, -1., 1.);
    return m.colRange(0, cols);
}

static int ld(const Mat& m) { return (int)(m.step / sizeof(float)); }

// alpha * A * B + beta * C in double precision, with element (i, k) of A at a[i*a0 + k*a1]
// (B and C likewise), and the sum of the magnitudes of all terms as the error scale.
static void refGemm(int M, int N, int K, double alpha, const float* a, int a0, int a1,
                    const float* b, int b0, int b1, double beta, const float* c, int c0, int c1,
                    Mat& ref, Mat& scale)
{
    ref.create(M, N, CV_64F);
    scale.create(M, N, CV_64F);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            double s = 0, sa = 0;
            for (int k = 0; k < K; k++)
            {
                double t = (double)a[(size_t)i*a0 + (size_t)k*a1] * b[(size_t)k*b0 + (size_t)j*b1];
                s += t;
                sa += std::abs(t);
            }
            double cij = c ? (double)c[(size_t)i*c0 + (size_t)j*c1] : 0.;
            ref.at<double>(i, j) = alpha*s + beta*cij;
            scale.at<double>(i, j) = std::abs(alpha)*sa + std::abs(beta*cij);
        }
}

static void checkGemm(const float* d, int ldd, const Mat& ref, const Mat& scale, int K)
{
    const double eps = (K + 4) * FLT_EPSILON;
    for (int i = 0; i < ref.rows; i++)
        for (int j = 0; j < ref.cols; j++)
        {
            double r = ref.at<double>(i, j);
            ASSERT_LE(std::abs(d[(size_t)i*ldd + j] - r), eps * scale.at<double>(i, j) + 1e-6)
                << "at (" << i << ", " << j << ")";
        }
}

enum { NO_C, SEPARATE_C, INPLACE_C };

typedef tuple<Vec3i, int, int> GemmParams;  // (M, N, K), GEMM flags, how C is passed
typedef TestWithParam<GemmParams> Core_HalGemm32f;

TEST_P(Core_HalGemm32f, accuracy)
{
    const Vec3i mnk = get<0>(GetParam());
    const int M = mnk[0], N = mnk[1], K = mnk[2], flags = get<1>(GetParam()), cmode = get<2>(GetParam());
    const bool ta = (flags & GEMM_1_T) != 0, tb = (flags & GEMM_2_T) != 0, tc = (flags & GEMM_3_T) != 0;
    RNG& rng = theRNG();

    Mat A = randMat(ta ? K : M, ta ? M : K, rng, 3), B = randMat(tb ? N : K, tb ? K : N, rng, 1);
    Mat D = randMat(M, N, rng, 2), C;
    if (cmode == SEPARATE_C)
        C = randMat(tc ? N : M, tc ? M : N, rng, 5);
    else if (cmode == INPLACE_C)
        C = D;
    const float alpha = 1.5f, beta = cmode == NO_C ? 0.f : 0.75f;

    Mat ref, scale;
    refGemm(M, N, K, alpha, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
            B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1,
            beta, C.empty() ? NULL : C.ptr<float>(), tc ? 1 : ld(C), tc ? ld(C) : 1, ref, scale);

    cv::hal::gemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, alpha,
                     C.empty() ? NULL : C.ptr<float>(), C.empty() ? 0 : C.step, beta,
                     D.ptr<float>(), D.step, A.rows, A.cols, N, flags);
    ASSERT_NO_FATAL_FAILURE(checkGemm(D.ptr<float>(), ld(D), ref, scale, K));
}

// Below 32x32x32 cv::hal::gemm32f keeps its older code; most of the rest goes to fastGemm.
static const Vec3i gemmShapes[] = {
    Vec3i(3, 3, 3), Vec3i(16, 16, 16), Vec3i(32, 32, 32), Vec3i(5, 64, 300), Vec3i(1, 64, 2500),
    Vec3i(64, 1, 256), Vec3i(130, 70, 90), Vec3i(197, 192, 576)
};

INSTANTIATE_TEST_CASE_P(/**/, Core_HalGemm32f, testing::Combine(
    testing::ValuesIn(gemmShapes),
    testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T, (int)(GEMM_1_T | GEMM_2_T)),
    testing::Values((int)NO_C, (int)SEPARATE_C, (int)INPLACE_C)));

INSTANTIATE_TEST_CASE_P(TransposedC, Core_HalGemm32f, testing::Combine(
    testing::ValuesIn(gemmShapes),
    testing::Values((int)GEMM_3_T, (int)(GEMM_1_T | GEMM_3_T), (int)(GEMM_2_T | GEMM_3_T),
                    (int)(GEMM_1_T | GEMM_2_T | GEMM_3_T)),
    testing::Values((int)SEPARATE_C)));

TEST(Core_FastGemm32f, size_threshold)
{
    RNG& rng = theRNG();
    // (M, N, K) and whether fastGemm takes the product: small or few-row ones stay with the older code
    const struct { int M, N, K; bool taken; } cases[] = {
        {4, 4, 4, false}, {64, 64, 64, true}, {1, 64, 2500, false}, {1, 1024, 1024, true},
        {256, 256, 4, false}, {256, 1, 256, true}
    };
    for (const auto& c : cases)
    {
        Mat A = randMat(c.M, c.K, rng, 0), B = randMat(c.K, c.N, rng, 0), D(c.M, c.N, CV_32F);
        bool taken = cv::hal::fastGemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, 1.f, NULL, 0, 0.f,
                                          D.ptr<float>(), D.step, c.M, c.K, c.N, 0);
        EXPECT_EQ(c.taken, taken) << c.M << "x" << c.N << "x" << c.K;
    }
}

TEST(Core_FastGemm32f, overlapping_output)
{
    RNG& rng = theRNG();
    Mat A = randMat(64, 64, rng, 0), B = randMat(64, 64, rng, 0);
    EXPECT_FALSE(cv::hal::fastGemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, 1.f, NULL, 0, 0.f,
                                      A.ptr<float>(), A.step, 64, 64, 64, 0));
}

// The OpenCV-only entry points used by dnn: operands given by two strides each, packed B,
// batches and the few-row paths, in multi- and single-threaded mode.
class Core_FastGemm : public testing::TestWithParam<bool>
{
protected:
    void SetUp() CV_OVERRIDE
    {
        opt.init();
        opt.multi_thread = GetParam();
    }
    cv::FastGemmOpt opt;
};

TEST_P(Core_FastGemm, strides)
{
    RNG& rng = theRNG();
    const int M = 45, N = 70, K = 93;
    for (int t = 0; t < 4; t++)
    {
        const bool ta = (t & 1) != 0, tb = (t & 2) != 0;
        Mat A = randMat(ta ? K : M, ta ? M : K, rng, 2), B = randMat(tb ? N : K, tb ? K : N, rng, 3);
        Mat C = randMat(M, N, rng, 1), ref, scale;
        refGemm(M, N, K, 0.5, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
                B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 2., C.ptr<float>(), ld(C), 1, ref, scale);
        // the transposes are given both as flags and as swapped strides
        cv::hal::fastGemm(ta, tb, A.rows, A.cols, B.rows, B.cols, 0.5f, A.ptr<float>(), ld(A), 1,
                          B.ptr<float>(), ld(B), 1, 2.f, C.ptr<float>(), ld(C), opt);
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K)) << "trans " << ta << tb;

        C = randMat(M, N, rng, 1);
        refGemm(M, N, K, 0.5, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
                B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 0., NULL, 0, 0, ref, scale);
        cv::hal::fastGemm(false, false, M, K, K, N, 0.5f, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
                          B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 0.f, C.ptr<float>(), ld(C), opt);
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K)) << "strides " << ta << tb;
    }
}

TEST_P(Core_FastGemm, packedB)
{
    RNG& rng = theRNG();
    const int M = 50, N = 330, K = 140;
    for (int t = 0; t < 4; t++)
    {
        const bool ta = (t & 1) != 0, tb = (t & 2) != 0;
        Mat A = randMat(ta ? K : M, ta ? M : K, rng, 0), B = randMat(tb ? N : K, tb ? K : N, rng, 4);
        std::vector<float> packed(cv::hal::fastGemmPackBSize(N, K, opt));
        cv::hal::fastGemmPackB(tb, B.cols, B.rows, B.ptr<float>(), ld(B), packed.data(), opt);

        Mat C = randMat(M, N, rng, 0), ref, scale;
        refGemm(M, N, K, 1., A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
                B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 1., C.ptr<float>(), ld(C), 1, ref, scale);
        cv::hal::fastGemm(ta, M, N, K, 1.f, A.ptr<float>(), ld(A), packed.data(), 1.f, C.ptr<float>(), ld(C), opt);
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K)) << "trans " << ta << tb;
    }
}

TEST_P(Core_FastGemm, batch)
{
    RNG& rng = theRNG();
    const int batch = 3, K = 64;
    // M = 2 with a transposed B takes the few-row path; M = 40 the blocked kernels
    for (int M : {2, 40})
    {
        const int N = 96;
        Mat A = randMat(batch * M, K, rng, 0), B = randMat(batch * N, K, rng, 0), C(batch * M, N, CV_32F);
        std::vector<size_t> a_ofs(batch), b_ofs(batch), c_ofs(batch);
        for (int i = 0; i < batch; i++)
        {
            a_ofs[i] = (size_t)i * M * K;
            b_ofs[i] = (size_t)i * N * K;
            c_ofs[i] = (size_t)i * M * N;
        }
        cv::hal::fastGemmBatch(batch, a_ofs.data(), b_ofs.data(), c_ofs.data(), M, N, K, 1.f,
                               A.ptr<float>(), K, 1, B.ptr<float>(), 1, K, 0.f, C.ptr<float>(), N, opt);
        for (int i = 0; i < batch; i++)
        {
            Mat ref, scale;
            refGemm(M, N, K, 1., A.ptr<float>(i * M), K, 1, B.ptr<float>(i * N), 1, K, 0., NULL, 0, 0, ref, scale);
            ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(i * M), N, ref, scale, K)) << "M " << M << ", batch " << i;
        }
    }

    // packed B, one packed matrix per batch item
    const int M = 33, N = 100;
    Mat A = randMat(batch * M, K, rng, 0), B = randMat(batch * K, N, rng, 0), C(batch * M, N, CV_32F);
    const size_t packed_size = cv::hal::fastGemmPackBSize(N, K, opt);
    std::vector<float> packed(packed_size * batch);
    std::vector<size_t> a_ofs(batch), p_ofs(batch), c_ofs(batch);
    for (int i = 0; i < batch; i++)
    {
        cv::hal::fastGemmPackB(false, N, K, B.ptr<float>(i * K), N, packed.data() + i * packed_size, opt);
        a_ofs[i] = (size_t)i * M * K;
        p_ofs[i] = i * packed_size;
        c_ofs[i] = (size_t)i * M * N;
    }
    cv::hal::fastGemmBatch(batch, a_ofs.data(), p_ofs.data(), c_ofs.data(), M, N, K, 1.f,
                           A.ptr<float>(), K, 1, packed.data(), 0.f, C.ptr<float>(), N, opt);
    for (int i = 0; i < batch; i++)
    {
        Mat ref, scale;
        refGemm(M, N, K, 1., A.ptr<float>(i * M), K, 1, B.ptr<float>(i * K), N, 1, 0., NULL, 0, 0, ref, scale);
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(i * M), N, ref, scale, K)) << "packed, batch " << i;
    }
}

TEST_P(Core_FastGemm, thin)
{
    RNG& rng = theRNG();
    const int M = 3, N = 75, K = 200;
    ASSERT_TRUE(cv::hal::fastGemmThinEligible(M, N, K));
    Mat A = randMat(M, K, rng, 0), B = randMat(K, N, rng, 0), C = randMat(M, N, rng, 0), ref, scale;
    std::vector<float> packed(cv::hal::fastGemmThinPackBSize(N, K));
    cv::hal::fastGemmThinPackB(N, K, B.ptr<float>(), ld(B), 1, packed.data());
    refGemm(M, N, K, 2., A.ptr<float>(), ld(A), 1, B.ptr<float>(), ld(B), 1, 0.5, C.ptr<float>(), ld(C), 1, ref, scale);
    cv::hal::fastGemmThin(M, N, K, 2.f, A.ptr<float>(), ld(A), 1, packed.data(), 0.5f, C.ptr<float>(), ld(C),
                          opt.multi_thread);
    ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K));
}

INSTANTIATE_TEST_CASE_P(/**/, Core_FastGemm, testing::Values(true, false));

}} // namespace
