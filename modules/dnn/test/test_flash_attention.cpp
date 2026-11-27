// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Plain attention in double: Y[b, i, h*Dv + :] = softmax(Q K^T / scale) V, with K given transposed.
static Mat referenceSDPA(const Mat& Q, const Mat& KT, const Mat& V, double scale)
{
    const int B = Q.size[0], H = Q.size[1], Sq = Q.size[2], D = Q.size[3];
    const int Skv = V.size[2], Dv = V.size[3];
    Mat Y(std::vector<int>{B, Sq, H * Dv}, CV_32F);
    std::vector<double> s(Skv);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < Sq; i++)
            {
                const float* q = Q.ptr<float>(b, h) + (size_t)i * D;
                const float* kt = KT.ptr<float>(b, h);
                const float* v = V.ptr<float>(b, h);
                double mx = -DBL_MAX, sum = 0;
                for (int j = 0; j < Skv; j++)
                {
                    double d = 0;
                    for (int c = 0; c < D; c++)
                        d += (double)q[c] * kt[(size_t)c * Skv + j];
                    s[j] = d / scale;
                    mx = std::max(mx, s[j]);
                }
                for (int j = 0; j < Skv; j++)
                    sum += (s[j] = std::exp(s[j] - mx));
                float* y = Y.ptr<float>(b, i) + h * Dv;
                for (int c = 0; c < Dv; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < Skv; j++)
                        acc += s[j] * v[(size_t)j * Dv + c];
                    y[c] = (float)(acc / sum);
                }
            }
    return Y;
}

// B, H, Sq, Skv, D, Dv, input range: several Q and K/V blocks (256), Dv != D and large scores.
typedef testing::TestWithParam<tuple<int, int, int, int, int, int, float> > Test_SDPA_Accuracy;
TEST_P(Test_SDPA_Accuracy, Accuracy)
{
    const int B = get<0>(GetParam()), H = get<1>(GetParam()), Sq = get<2>(GetParam()), Skv = get<3>(GetParam());
    const int D = get<4>(GetParam()), Dv = get<5>(GetParam());
    const float range = get<6>(GetParam());
    Mat Q(std::vector<int>{B, H, Sq, D}, CV_32F), KT(std::vector<int>{B, H, D, Skv}, CV_32F);
    Mat V(std::vector<int>{B, H, Skv, Dv}, CV_32F);
    RNG& rng = theRNG();
    rng.fill(Q, RNG::UNIFORM, -range, range);
    rng.fill(KT, RNG::UNIFORM, -range, range);
    rng.fill(V, RNG::UNIFORM, -1.f, 1.f);
    const double scale = std::sqrt((double)D);

    LayerParams lp;
    lp.type = "SDPA";
    lp.name = "sdpa";
    lp.set("scale", (float)scale);
    Net net;
    const int id = net.addLayer(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0);
    net.connect(0, 1, id, 1);
    net.connect(0, 2, id, 2);
    net.setInputsNames({"q", "kt", "v"});
    net.setInput(Q, "q");
    net.setInput(KT, "kt");
    net.setInput(V, "v");
    Mat out = net.forward();

    Mat ref = referenceSDPA(Q, KT, V, scale);
    ASSERT_EQ(ref.total(), out.total());
    EXPECT_LE(cvtest::norm(ref.reshape(1, 1), out.reshape(1, 1), NORM_INF), 2e-5);
}
INSTANTIATE_TEST_CASE_P(/**/, Test_SDPA_Accuracy, testing::Values(
    make_tuple(1, 2, 7, 5, 16, 16, 1.f),
    make_tuple(2, 3, 50, 50, 64, 64, 1.f),
    make_tuple(1, 2, 300, 600, 48, 40, 1.f),
    make_tuple(2, 2, 257, 513, 32, 24, 1.f),
    make_tuple(1, 2, 64, 700, 64, 32, 6.f)));

}} // namespace
