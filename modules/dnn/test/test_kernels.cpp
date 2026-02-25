// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include "../src/layers/cpu_kernels/fast_gemm.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;


class pagedAttentionQKTest : public ::testing::Test {
protected:
    const int S = 8; // 8 pages
    const int B = 1;
    const int D = 16;
    const int T_q = 1; // 4
    const int Nq = 8;
    const int Nk = 4;
    int T_s, T;
    Mat K;
    std::vector<Mat> K_pages;
    FastGemmOpt opt;
    std::vector<size_t> A_offsets, Q_offsets, K_offsets;


    virtual void SetUp() {
        opt.init();
        T_s = fastGemmNC(opt) * 4;
        T = S * T_s;
        K = Mat({B, Nk, D, S * T_s}, CV_32F);
        for (int s = 0; s < S; s++) {
            Mat K_page({B, Nk, D, T_s}, CV_32F);
            randu(K_page, -1.f, 1.f);
            Range ranges[] = {Range::all(), Range::all(), Range::all(), Range(s * T_s, (s + 1) * T_s)};
            K_page.copyTo(K(ranges));

            std::vector<float> packed;
            fastGemmPackB(K_page, packed, false, opt);
            Mat packedK({B, Nk, D * T_s}, CV_32F);
            std::memcpy(packedK.data, packed.data(), packed.size() * sizeof(float));

            K_pages.push_back(packedK);
        }

        const size_t mat_size_a = T * T_q;
        const size_t mat_size_k = T * D;
        const size_t mat_size_q = D * T_q;

        size_t cur_offset_a = 0, cur_offset_q = 0;
        int group_size = Nq / Nk;
        for (int b = 0; b < B; b++) {
            for (int nq = 0; nq < Nq; nq++) {
                A_offsets.push_back(cur_offset_a);
                Q_offsets.push_back(cur_offset_q);

                int nk = nq / group_size;
                size_t k_off = (b * Nk + nk) * mat_size_k;
                K_offsets.push_back(k_off);

                cur_offset_a += mat_size_a;
                cur_offset_q += mat_size_q;
            }
        }
    }

    void validate(Mat &A, Mat &Q, float eps = 1e-3) {
        Mat A_ref({B, Nq, T_q, T}, CV_32F, Scalar(0.f));

        fastGemmBatch(A_offsets.size(),
                    Q_offsets, K_offsets, A_offsets,
                    T_q, T, D, 1.f, Q, D, 1,
                    K, T, 1, 0.f, A_ref, T, opt);

        EXPECT_LE(cvtest::norm(A, A_ref, NORM_INF), eps);
    }

};


TEST_F(pagedAttentionQKTest, pagedAttnQKGemm_Q4D)
{
    Mat Q({B, Nq, T_q, D}, CV_32F);
    randu(Q, -1.f, 1.f);

    Mat A({B, Nq, T_q, T}, CV_32F, Scalar(0.f));

    pagedAttnQKGemm(Q, K_pages, A, T_q, Nq, Nk, T_s, D, opt);

    validate(A, Q);
}

TEST_F(pagedAttentionQKTest, pagedAttnQKGemm_Q3D)
{
    Mat Q({B, T_q, Nq * D}, CV_32F);
    randu(Q, -1.f, 1.f);

    Mat A({B, Nq, T_q, T}, CV_32F, Scalar(0.f));

    pagedAttnQKGemm(Q, K_pages, A, T_q, Nq, Nk, T_s, D, opt);

    validate(A, Q);
}


class pagedAttentionAVTest : public ::testing::Test {
protected:
    const int S = 8; // 8 pages
    const int B = 1;
    const int D = 16;
    const int T_a = 1; // 4
    const int Nq = 8;
    const int Nkv = 4;
    int T_s;
    Mat V;
    std::vector<Mat> V_pages;
    FastGemmOpt opt;
    std::vector<size_t> V_offsets, A_offsets, Out_offsets;

    virtual void SetUp() {
        opt.init();
        T_s = fastGemmKC(opt) * 7;
        V = Mat({B, Nkv, S * T_s, D}, CV_32F, Scalar(0.f));
        for (int s = 0; s < S; s++) {
            Mat V_page({B, Nkv, T_s, D}, CV_32F);
            randu(V_page, -1.f, 1.f);
            Range ranges[] = {Range::all(), Range::all(), Range(s * T_s, (s + 1) * T_s), Range::all()};
            V_page.copyTo(V(ranges));

            std::vector<float> packed;
            fastGemmPackB(V_page, packed, false, opt);
            int packed_size = fastGemmPackBSize(D, T_s, opt);
            Mat packedV({B, Nkv, packed_size}, CV_32F);
            std::memcpy(packedV.data, packed.data(), packed.size() * sizeof(float));

            V_pages.push_back(packedV);
        }

        const int T = S * T_s;
        const size_t mat_size_k = (size_t)T * D;
        const size_t mat_size_a = (size_t)T_a * T;
        const size_t mat_size_out = (size_t)T_a * D;

        size_t cur_offset_a = 0, cur_offset_out = 0;
        int group_size = Nq / Nkv;

        for (int b = 0; b < B; b++) {
            for (int nq = 0; nq < Nq; nq++) {
                A_offsets.push_back(cur_offset_a);
                Out_offsets.push_back(cur_offset_out);

                int nv = nq / group_size;
                size_t v_off = (b * Nkv + nv) * mat_size_k;
                V_offsets.push_back(v_off);

                cur_offset_a += mat_size_a;
                cur_offset_out += mat_size_out;
            }
        }
    }

    void validate(Mat &Out, Mat &A, float eps = 1e-3) {
        Mat Out_ref(Out.size, CV_32F, Scalar(0.f));
        fastGemmBatch(Out_offsets.size(),
                      A_offsets, V_offsets, Out_offsets,
                      T_a, D, S * T_s, 1.f, A, S * T_s, 1,
                      V, D, 1, 0.f, Out_ref, D, opt);

        EXPECT_LE(cvtest::norm(Out, Out_ref, NORM_INF), eps);
    }
};


TEST_F(pagedAttentionAVTest, pagedAttnAVGemm_O4D)
{
    Mat A({B, Nq, T_a, S * T_s}, CV_32F, Scalar(0.f));
    randu(A, -1.f, 1.f);

    Mat Out({B, Nq, T_a, D}, CV_32F, Scalar(0.f));

    pagedAttnAVGemm(
        A, V_pages, Out,
        T_a, Nq, Nkv, T_s, D,
        opt
    );

    validate(Out, A);
}


TEST_F(pagedAttentionAVTest, pagedAttnAVGemm_O3D)
{
    Mat A({B, Nq, T_a, S * T_s}, CV_32F, Scalar(0.f));
    randu(A, -1.f, 1.f);

    Mat Out({B, T_a, Nq * D}, CV_32F, Scalar(0.f));

    pagedAttnAVGemm(
        A, V_pages, Out,
        T_a, Nq, Nkv, T_s, D,
        opt
    );

    validate(Out, A);
}

}} // namespace opencv_test