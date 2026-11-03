// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "test_precomp.hpp"
#include "opencv2/core/private/fast_gemm.hpp"

namespace opencv_test { namespace {

static Mat randMat(int rows, int cols, RNG& rng, int pad)
{
    Mat m(rows, cols + pad, CV_32F);
    rng.fill(m, RNG::UNIFORM, -1., 1.);
    return m.colRange(0, cols);
}

static int ld(const Mat& m) { return (int)(m.step / sizeof(float)); }

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

typedef tuple<Vec3i, int, int> GemmParams;
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

    cv::hal::gemm(CV_32FC1, A.data, A.step, B.data, B.step, alpha,
                  C.data, C.empty() ? 0 : C.step, beta,
                  D.data, D.step, A.rows, A.cols, N, flags);
    ASSERT_NO_FATAL_FAILURE(checkGemm(D.ptr<float>(), ld(D), ref, scale, K));
}

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

TEST(Core_HalGemm, unsupported_type)
{
    Mat A(4, 4, CV_32SC1, Scalar(1)), B(4, 4, CV_32SC1, Scalar(1)), D(4, 4, CV_32SC1);
    EXPECT_THROW(cv::hal::gemm(CV_32SC1, A.data, A.step, B.data, B.step, 1., NULL, 0, 0.,
                               D.data, D.step, 4, 4, 4, 0), cv::Exception);
}

// (M, N, K), whether fastGemm32f takes the call
typedef TestWithParam<tuple<Vec3i, bool> > Core_FastGemm32f_SizeThreshold;

TEST_P(Core_FastGemm32f_SizeThreshold, taken)
{
    const Vec3i mnk = get<0>(GetParam());
    const int M = mnk[0], N = mnk[1], K = mnk[2];
    RNG& rng = theRNG();
    Mat A = randMat(M, K, rng, 0), B = randMat(K, N, rng, 0), D(M, N, CV_32F);
    bool taken = cv::hal::fastGemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, 1.f, NULL, 0, 0.f,
                                      D.ptr<float>(), D.step, M, K, N, 0);
    EXPECT_EQ(get<1>(GetParam()), taken);
}

INSTANTIATE_TEST_CASE_P(/**/, Core_FastGemm32f_SizeThreshold, testing::Values(
    make_tuple(Vec3i(4, 4, 4), false), make_tuple(Vec3i(64, 64, 64), true),
    make_tuple(Vec3i(1, 64, 2500), false), make_tuple(Vec3i(1, 1024, 1024), true),
    make_tuple(Vec3i(256, 256, 4), false), make_tuple(Vec3i(256, 1, 256), true)));

TEST(Core_FastGemm32f, overlapping_output)
{
    RNG& rng = theRNG();
    Mat A = randMat(64, 64, rng, 0), B = randMat(64, 64, rng, 0);
    EXPECT_FALSE(cv::hal::fastGemm32f(A.ptr<float>(), A.step, B.ptr<float>(), B.step, 1.f, NULL, 0, 0.f,
                                      A.ptr<float>(), A.step, 64, 64, 64, 0));
}

static cv::FastGemmOpt fastGemmOpt(bool multi_thread)
{
    cv::FastGemmOpt opt;
    opt.init();
    opt.multi_thread = multi_thread;
    return opt;
}

// multi_thread, transposed A, transposed B
typedef TestWithParam<tuple<bool, bool, bool> > Core_FastGemmTrans;

TEST_P(Core_FastGemmTrans, strides)
{
    cv::FastGemmOpt opt = fastGemmOpt(get<0>(GetParam()));
    const bool ta = get<1>(GetParam()), tb = get<2>(GetParam());
    RNG& rng = theRNG();
    const int M = 45, N = 70, K = 93;
    Mat A = randMat(ta ? K : M, ta ? M : K, rng, 2), B = randMat(tb ? N : K, tb ? K : N, rng, 3);
    Mat C = randMat(M, N, rng, 1), ref, scale;
    refGemm(M, N, K, 0.5, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
            B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 2., C.ptr<float>(), ld(C), 1, ref, scale);
    cv::hal::fastGemm(ta, tb, A.rows, A.cols, B.rows, B.cols, 0.5f, A.ptr<float>(), ld(A), 1,
                      B.ptr<float>(), ld(B), 1, 2.f, C.ptr<float>(), ld(C), opt);
    ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K)) << "transpose flags";

    C = randMat(M, N, rng, 1);
    refGemm(M, N, K, 0.5, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
            B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 0., NULL, 0, 0, ref, scale);
    cv::hal::fastGemm(false, false, M, K, K, N, 0.5f, A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
                      B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 0.f, C.ptr<float>(), ld(C), opt);
    ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K)) << "explicit strides";
}

TEST_P(Core_FastGemmTrans, packedB)
{
    cv::FastGemmOpt opt = fastGemmOpt(get<0>(GetParam()));
    const bool ta = get<1>(GetParam()), tb = get<2>(GetParam());
    RNG& rng = theRNG();
    const int M = 50, N = 330, K = 140;
    Mat A = randMat(ta ? K : M, ta ? M : K, rng, 0), B = randMat(tb ? N : K, tb ? K : N, rng, 4);
    std::vector<float> packed(cv::hal::fastGemmPackBSize(N, K, opt));
    cv::hal::fastGemmPackB(tb, B.cols, B.rows, B.ptr<float>(), ld(B), packed.data(), opt);

    Mat C = randMat(M, N, rng, 0), ref, scale;
    refGemm(M, N, K, 1., A.ptr<float>(), ta ? 1 : ld(A), ta ? ld(A) : 1,
            B.ptr<float>(), tb ? 1 : ld(B), tb ? ld(B) : 1, 1., C.ptr<float>(), ld(C), 1, ref, scale);
    cv::hal::fastGemm(ta, M, N, K, 1.f, A.ptr<float>(), ld(A), packed.data(), 1.f, C.ptr<float>(), ld(C), opt);
    ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K));
}

INSTANTIATE_TEST_CASE_P(/**/, Core_FastGemmTrans, testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()));

// multi_thread, M
typedef TestWithParam<tuple<bool, int> > Core_FastGemmBatch;

TEST_P(Core_FastGemmBatch, transposedB)
{
    cv::FastGemmOpt opt = fastGemmOpt(get<0>(GetParam()));
    const int M = get<1>(GetParam()), N = 96, K = 64, batch = 3;
    RNG& rng = theRNG();
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
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(i * M), N, ref, scale, K)) << "batch " << i;
    }
}

TEST_P(Core_FastGemmBatch, packedB)
{
    cv::FastGemmOpt opt = fastGemmOpt(get<0>(GetParam()));
    const int M = get<1>(GetParam()), N = 100, K = 64, batch = 3;
    RNG& rng = theRNG();
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
        ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(i * M), N, ref, scale, K)) << "batch " << i;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Core_FastGemmBatch, testing::Combine(testing::Bool(), testing::Values(2, 33, 40)));

// multi_thread
typedef TestWithParam<bool> Core_FastGemm;

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
                          GetParam());
    ASSERT_NO_FATAL_FAILURE(checkGemm(C.ptr<float>(), ld(C), ref, scale, K));
}

INSTANTIATE_TEST_CASE_P(/**/, Core_FastGemm, testing::Bool());

}} // namespace
