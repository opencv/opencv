// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/hal/hal.hpp"

namespace opencv_test { namespace {

struct GemmShape { int m, n, k; };

// Edge sizes for every path: small M (1-32 rows), the 8x12 blocks and the SME tiles.
// The last two fit in one SME tile but have a long k, so the task split cannot reach its target.
static const GemmShape gemmShapes[] = {
    { 1, 1, 1 }, { 3, 5, 7 }, { 1, 40, 33 }, { 2, 70, 37 }, { 8, 12, 64 }, { 9, 13, 65 },
    { 17, 25, 3 }, { 24, 50, 40 }, { 32, 33, 300 }, { 50, 100, 200 }, { 145, 73, 130 },
    { 200, 150, 1300 }, { 300, 290, 1700 }, { 12, 20, 9000 }, { 30, 30, 5000 }
};

static const float alphaBeta[][2] = { { 1.f, 0.f }, { 0.5f, 1.f }, { -1.5f, 0.25f } };

// A rows x cols view inside a wider matrix, so the row step is not dense.
static Mat randomMat(int rows, int cols, RNG& rng)
{
    Mat big(rows, cols + 3, CV_32F);
    rng.fill(big, RNG::UNIFORM, -1., 1.);
    return big.colRange(0, cols);
}

// Compares got with alpha*op(A)*op(B) + beta*C0 in double, allowing the usual k-scaled fp32 error.
static void checkGemm(const Mat& A, const Mat& B, const Mat& C0, const Mat& got,
                      float alpha, float beta, bool tA, bool tB)
{
    Mat a = tA ? Mat(A.t()) : A.clone(), b = tB ? Mat(B.t()) : B.clone();
    const int m = got.rows, n = got.cols, k = a.cols;
    ASSERT_EQ(m, a.rows);
    ASSERT_EQ(k, b.rows);
    ASSERT_EQ(n, b.cols);
    std::vector<double> sum(n), absSum(n);
    for (int i = 0; i < m; i++)
    {
        std::fill(sum.begin(), sum.end(), 0.);
        std::fill(absSum.begin(), absSum.end(), 0.);
        const float* ai = a.ptr<float>(i);
        for (int p = 0; p < k; p++)
        {
            const float* bp = b.ptr<float>(p);
            for (int j = 0; j < n; j++)
            {
                const double t = (double)ai[p] * bp[j];
                sum[j] += t;
                absSum[j] += std::abs(t);
            }
        }
        for (int j = 0; j < n; j++)
        {
            const double c = beta == 0.f ? 0. : (double)beta * C0.at<float>(i, j);
            const double ref = alpha * sum[j] + c;
            const double bound = 2. * (k + 2) * FLT_EPSILON * (std::abs(alpha) * absSum[j] + std::abs(c)) + 1e-30;
            ASSERT_LE(std::abs(got.at<float>(i, j) - ref), bound)
                << "at (" << i << ", " << j << ") m=" << m << " n=" << n << " k=" << k
                << " tA=" << tA << " tB=" << tB << " alpha=" << alpha << " beta=" << beta;
        }
    }
}

TEST(Core_GemmEngine, batch_one_accuracy)
{
    RNG& rng = theRNG();
    const size_t zero = 0;
    for (const GemmShape& s : gemmShapes)
        for (int t = 0; t < 4; t++)
            for (const auto& ab : alphaBeta)
            {
                const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
                Mat A = tA ? randomMat(s.k, s.m, rng) : randomMat(s.m, s.k, rng);
                Mat B = tB ? randomMat(s.n, s.k, rng) : randomMat(s.k, s.n, rng);
                Mat C = randomMat(s.m, s.n, rng), C0 = C.clone();
                const int flags = (tA ? CV_HAL_GEMM_1_T : 0) | (tB ? CV_HAL_GEMM_2_T : 0);
                hal::gemm32fBatch(1, A.ptr<float>(), &zero, A.step, B.ptr<float>(), &zero, B.step,
                                  C.ptr<float>(), &zero, C.step, s.m, s.n, s.k, ab[0], ab[1], flags);
                checkGemm(A, B, C0, C, ab[0], ab[1], tA, tB);
            }
}

TEST(Core_GemmEngine, batch_broadcast_accuracy)
{
    RNG& rng = theRNG();
    const int batch = 3, n = 29, k = 75;
    for (int t = 0; t < 8; t++)
    {
        const int m = t < 4 ? 37 : 5;
        const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
        const int aRows = tA ? k : m, aCols = tA ? m : k;
        SCOPED_TRACE(cv::format("m=%d", m));
        Mat A = randomMat(batch * aRows, aCols, rng);
        Mat B = tB ? randomMat(n, k, rng) : randomMat(k, n, rng);
        Mat C = randomMat(batch * m, n, rng), C0 = C.clone();
        std::vector<size_t> aOfs(batch), bOfs(batch, 0), cOfs(batch);
        for (int i = 0; i < batch; i++)
        {
            aOfs[i] = (size_t)i * aRows * (A.step / sizeof(float));
            cOfs[i] = (size_t)i * m * (C.step / sizeof(float));
        }
        const int flags = (tA ? CV_HAL_GEMM_1_T : 0) | (tB ? CV_HAL_GEMM_2_T : 0);
        hal::gemm32fBatch(batch, A.ptr<float>(), aOfs.data(), A.step, B.ptr<float>(), bOfs.data(), B.step,
                          C.ptr<float>(), cOfs.data(), C.step, m, n, k, 0.75f, 0.5f, flags);
        for (int i = 0; i < batch; i++)
            checkGemm(A.rowRange(i * aRows, (i + 1) * aRows), B, C0.rowRange(i * m, (i + 1) * m),
                      C.rowRange(i * m, (i + 1) * m), 0.75f, 0.5f, tA, tB);
    }
}

TEST(Core_GemmEngine, packed_accuracy)
{
    RNG& rng = theRNG();
    for (const GemmShape& s : gemmShapes)
        for (int t = 0; t < 4; t++)
            for (const auto& ab : alphaBeta)
            {
                const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
                Mat A = tA ? randomMat(s.k, s.m, rng) : randomMat(s.m, s.k, rng);
                Mat B = tB ? randomMat(s.n, s.k, rng) : randomMat(s.k, s.n, rng);
                Mat C = randomMat(s.m, s.n, rng), C0 = C.clone();
                const int flagsB = tB ? CV_HAL_GEMM_2_T : 0;
                std::vector<uchar> packed(hal::gemm32fPackBSize(s.n, s.k, flagsB));
                hal::gemm32fPackB(B.ptr<float>(), B.step, s.n, s.k, flagsB, packed.data());
                hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C.ptr<float>(), C.step,
                                   s.m, s.n, s.k, ab[0], ab[1], tA ? CV_HAL_GEMM_1_T : 0);
                checkGemm(A, B, C0, C, ab[0], ab[1], tA, tB);
            }
}

TEST(Core_GemmEngine, packed_header_mismatch)
{
    RNG& rng = theRNG();
    const int m = 4, n = 20, k = 30;
    Mat A = randomMat(m, k, rng), B = randomMat(k, n, rng), C(m, n + 1, CV_32F);
    std::vector<uchar> packed(hal::gemm32fPackBSize(n, k, 0));
    hal::gemm32fPackB(B.ptr<float>(), B.step, n, k, 0, packed.data());
    EXPECT_THROW(hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C.ptr<float>(), C.step,
                                    m, n + 1, k, 1.f, 0.f, 0), cv::Exception);
    std::vector<uchar> garbage(packed.size(), 0);
    EXPECT_THROW(hal::gemm32fPacked(A.ptr<float>(), A.step, garbage.data(), C.ptr<float>(), C.step,
                                    m, n, k, 1.f, 0.f, 0), cv::Exception);
}

TEST(Core_GemmEngine, beta_zero_ignores_nan)
{
    RNG& rng = theRNG();
    const int m = 30, n = 40, k = 50;
    const size_t zero = 0;
    Mat A = randomMat(m, k, rng), B = randomMat(k, n, rng);
    Mat C(m, n, CV_32F, Scalar::all(std::numeric_limits<float>::quiet_NaN()));
    hal::gemm32fBatch(1, A.ptr<float>(), &zero, A.step, B.ptr<float>(), &zero, B.step,
                      C.ptr<float>(), &zero, C.step, m, n, k, 1.f, 0.f, 0);
    EXPECT_TRUE(cv::checkRange(C));

    C.setTo(Scalar::all(std::numeric_limits<float>::quiet_NaN()));
    std::vector<uchar> packed(hal::gemm32fPackBSize(n, k, 0));
    hal::gemm32fPackB(B.ptr<float>(), B.step, n, k, 0, packed.data());
    hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C.ptr<float>(), C.step, m, n, k, 1.f, 0.f, 0);
    EXPECT_TRUE(cv::checkRange(C));
}

TEST(Core_GemmEngine, k_zero_scales_c)
{
    RNG& rng = theRNG();
    Mat C = randomMat(6, 4, rng), C0 = C.clone();
    const size_t abOfs[2] = { 0, 0 }, cOfs[2] = { 0, 3 * (C.step / sizeof(float)) };
    const float dummy = 0.f;
    hal::gemm32fBatch(2, &dummy, abOfs, sizeof(float), &dummy, abOfs, sizeof(float),
                      C.ptr<float>(), cOfs, C.step, 3, 4, 0, 1.f, 0.5f, 0);
    EXPECT_EQ(0., cvtest::norm(C, Mat(C0 * 0.5), NORM_INF));
}

TEST(Core_GemmEngine, single_thread_is_bit_exact)
{
    RNG& rng = theRNG();
    const size_t zero = 0;
    for (const GemmShape& s : { GemmShape{ 145, 73, 130 }, GemmShape{ 200, 150, 1300 } })
    {
        Mat A = randomMat(s.m, s.k, rng), B = randomMat(s.k, s.n, rng);
        Mat C1 = randomMat(s.m, s.n, rng), C2 = C1.clone();
        hal::gemm32fBatch(1, A.ptr<float>(), &zero, A.step, B.ptr<float>(), &zero, B.step,
                          C1.ptr<float>(), &zero, C1.step, s.m, s.n, s.k, 1.f, 1.f, 0);
        hal::gemm32fBatch(1, A.ptr<float>(), &zero, A.step, B.ptr<float>(), &zero, B.step,
                          C2.ptr<float>(), &zero, C2.step, s.m, s.n, s.k, 1.f, 1.f, CV_HAL_GEMM_SINGLE_THREAD);
        EXPECT_EQ(0., cvtest::norm(C1, C2, NORM_INF));

        std::vector<uchar> packed(hal::gemm32fPackBSize(s.n, s.k, 0));
        hal::gemm32fPackB(B.ptr<float>(), B.step, s.n, s.k, 0, packed.data());
        C1.setTo(0); C2.setTo(0);
        hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C1.ptr<float>(), C1.step, s.m, s.n, s.k, 1.f, 0.f, 0);
        hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C2.ptr<float>(), C2.step, s.m, s.n, s.k, 1.f, 0.f,
                           CV_HAL_GEMM_SINGLE_THREAD);
        EXPECT_EQ(0., cvtest::norm(C1, C2, NORM_INF));
    }
}

TEST(Core_GemmEngine, cv_gemm_accuracy)
{
    RNG& rng = theRNG();
    for (const GemmShape& s : gemmShapes)
        for (int t = 0; t < 8; t++)
            for (const auto& ab : alphaBeta)
            {
                const bool tA = (t & 1) != 0, tB = (t & 2) != 0, tC = (t & 4) != 0;
                Mat A = tA ? randomMat(s.k, s.m, rng) : randomMat(s.m, s.k, rng);
                Mat B = tB ? randomMat(s.n, s.k, rng) : randomMat(s.k, s.n, rng);
                Mat C = tC ? randomMat(s.n, s.m, rng) : randomMat(s.m, s.n, rng);
                const int flags = (tA ? GEMM_1_T : 0) | (tB ? GEMM_2_T : 0) | (tC ? GEMM_3_T : 0);
                Mat D;
                cv::gemm(A, B, ab[0], C, ab[1], D, flags);
                checkGemm(A, B, tC ? Mat(C.t()) : C, D, ab[0], ab[1], tA, tB);
            }
}

// FastCV and ArmPL take every cv::gemm shape themselves.
static bool vendorGemmHal()
{
    const std::string info = getBuildInformation();
    const size_t pos = info.find("Custom HAL:");
    if (pos == std::string::npos)
        return false;
    const std::string line = info.substr(pos, info.find('\n', pos) - pos);
    return line.find("fastcv") != std::string::npos || line.find("ARMPL") != std::string::npos;
}

// Shapes below the LAPACK threshold keep the generic path, which sums in double.
TEST(Core_GemmEngine, cv_gemm_keeps_double_accumulation_below_lapack_threshold)
{
    if (vendorGemmHal())
        throw SkipTestException("FastCV or ArmPL implements cv_hal_gemm32f");
    RNG& rng = theRNG();
    // One shape per branch of shouldUseLapackGemm().
    static const GemmShape shapes[] = {
        { 200, 3, 100 }, { 4, 64, 300 }, { 12, 100, 200 }, { 32, 32, 32 }, { 40, 100, 8 }, { 16, 40, 40 }
    };
    for (const GemmShape& s : shapes)
        for (int t = 0; t < 4; t++)
            for (const auto& ab : alphaBeta)
            {
                // LAPACK takes transposed products up to 32x32x32.
                if (t != 0 && s.m <= 32 && s.n <= 32 && s.k <= 32)
                    continue;
                const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
                Mat A = tA ? randomMat(s.k, s.m, rng) : randomMat(s.m, s.k, rng);
                Mat B = tB ? randomMat(s.n, s.k, rng) : randomMat(s.k, s.n, rng);
                Mat C = randomMat(s.m, s.n, rng);
                Mat D;
                cv::gemm(A, B, ab[0], C, ab[1], D, (tA ? GEMM_1_T : 0) | (tB ? GEMM_2_T : 0));

                Mat a = tA ? Mat(A.t()) : A.clone(), b = tB ? Mat(B.t()) : B.clone();
                Mat ref(s.m, s.n, CV_32F);
                for (int i = 0; i < s.m; i++)
                    for (int j = 0; j < s.n; j++)
                    {
                        double sum = 0.;
                        for (int p = 0; p < s.k; p++)
                            sum += (double)a.at<float>(i, p) * b.at<float>(p, j);
                        const double c = ab[1] == 0.f ? 0. : (double)ab[1] * C.at<float>(i, j);
                        ref.at<float>(i, j) = (float)(ab[0] * sum + c);
                    }
                // A different order of the double sums can move a result by one ulp.
                double maxUlps = 0.;
                EXPECT_TRUE(cvtest::cmpUlps(D, ref, 1, &maxUlps, NULL))
                    << "m=" << s.m << " n=" << s.n << " k=" << s.k << " tA=" << tA << " tB=" << tB
                    << " alpha=" << ab[0] << " beta=" << ab[1] << " max ulps=" << maxUlps;
            }
}

}} // namespace
