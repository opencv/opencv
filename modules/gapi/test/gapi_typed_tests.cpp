// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "common/gapi_tests_common.hpp"

namespace custom
{
G_TYPED_KERNEL(GKernelForGArrayGMatOut, <cv::GArray<cv::GMat>(cv::GMat)>,
               "custom.test.kernelForGArrayGMatOut")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc&)
    {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(GCPUKernelForGArrayGMatOut, custom::GKernelForGArrayGMatOut)
{
    static void run(const cv::Mat &src, std::vector<cv::Mat> &out)
    {
        out[0] = src.clone();
    }
};

G_TYPED_KERNEL(GSizeOfVectorGMat, <cv::GOpaque<size_t>(cv::GArray<cv::GMat>)>,
               "custom.test.sizeOfVectorGMat")
{
    static cv::GOpaqueDesc outMeta(const cv::GArrayDesc&)
    {
        return cv::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL(GCPUSizeOfVectorGMat, custom::GSizeOfVectorGMat)
{
    static void run(const std::vector<cv::Mat> &src, size_t &out)
    {
        out = src.size();
    }
};
}

namespace opencv_test
{

TEST(GAPI_Typed, UnaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat         (sz, CV_8UC3),
        out_mat_untyped(sz, CV_8UC3),
        out_mat_typed1 (sz, CV_8UC3),
        out_mat_typed2 (sz, CV_8UC3),
        out_mat_cv     (sz, CV_8UC3);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in;
        cv::GMat out = cv::gapi::RGB2YUV(in);
        return cv::GComputation(in, out);
    });
    cvtU.apply(in_mat, out_mat_untyped);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat)> cvtT(cv::gapi::RGB2YUV);
    auto cvtTComp = cvtT.compile(cv::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1);
    cvtTComp(in_mat, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    cv::cvtColor(in_mat, out_mat_cv, cv::COLOR_RGB2YUV);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_untyped, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed2, NORM_INF));
}

TEST(GAPI_Typed, BinaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1),
        out_mat_cv     (sz, CV_8UC1);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in1, in2;
        cv::GMat out = cv::gapi::add(in1, in2);
        return cv::GComputation({in1, in2}, {out});
    });
    std::vector<cv::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<cv::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat, cv::GMat)> cvtT([](cv::GMat m1, cv::GMat m2)
    {
        return m1+m2;
    });
    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat1),
                               cv::descr_of(in_mat2));

    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);
    cvtTC(in_mat1, in_mat2, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    cv::add(in_mat1, in_mat2, out_mat_cv);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_untyped, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed2, NORM_INF));
}

TEST(GAPI_Typed, MultipleOuts)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat        (sz, CV_8UC1),
        out_mat_unt1  (sz, CV_8UC1),
        out_mat_unt2  (sz, CV_8UC1),
        out_mat_typed1(sz, CV_8UC1),
        out_mat_typed2(sz, CV_8UC1),
        out_mat_comp1 (sz, CV_8UC1),
        out_mat_comp2 (sz, CV_8UC1),
        out_mat_cv1   (sz, CV_8UC1),
        out_mat_cv2   (sz, CV_8UC1);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in;
        cv::GMat out1 = in * 2.f;
        cv::GMat out2 = in * 4.f;
        return cv::GComputation({in}, {out1, out2});
    });
    std::vector<cv::Mat> u_ins  = {in_mat};
    std::vector<cv::Mat> u_outs = {out_mat_unt1, out_mat_unt2};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<std::tuple<cv::GMat, cv::GMat> (cv::GMat)> cvtT([](cv::GMat in)
    {
        return std::make_tuple(in*2.f, in*4.f);
    });
    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1, out_mat_typed2);
    cvtTC(in_mat, out_mat_comp1, out_mat_comp2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_mat_cv1 = in_mat * 2.f;
    out_mat_cv2 = in_mat * 4.f;

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_unt1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_unt2, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_typed2, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_comp1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_comp2, NORM_INF));
}

TEST(GAPI_Typed, GArrayGMatOut)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat in_mat(sz, CV_8UC3);
    std::vector<cv::Mat> out_vec_mat_untyped(1),
                         out_vec_mat_typed1 (1),
                         out_vec_mat_typed2 (1),
                         out_vec_mat_cv     (1);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    auto customKernel = cv::gapi::kernels<custom::GCPUKernelForGArrayGMatOut>();
    auto absExactCompare = AbsExact().to_compare_f();

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cptU([]()
    {
        cv::GMat in;
        cv::GArray<cv::GMat> out = custom::GKernelForGArrayGMatOut::on(in);
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    });
    cptU.apply(cv::gin(in_mat), cv::gout(out_vec_mat_untyped), cv::compile_args(customKernel));

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GArray<cv::GMat> (cv::GMat)> cptT(custom::GKernelForGArrayGMatOut::on);
    auto cplT = cptT.compile(cv::descr_of(in_mat), cv::compile_args(customKernel));

    cptT.apply(in_mat, out_vec_mat_typed1, cv::compile_args(customKernel));
    cplT(in_mat, out_vec_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_vec_mat_cv[0] = in_mat.clone();

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_untyped[0]));
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_typed1 [0]));
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_typed2 [0]));
}

TEST(GAPI_Typed, GArrayGMatIn)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    size_t vectorSize = 5;

    cv::Mat in_mat (sz, CV_8UC3);
    size_t out_size_t_untyped, out_size_t_typed1, out_size_t_typed2, out_size_t_cv;

    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    std::vector<cv::Mat> in_vec(vectorSize);
    for (size_t i = 0; i < vectorSize; i++)
        in_vec[i] = in_mat.clone();

    auto customKernel = cv::gapi::kernels<custom::GCPUSizeOfVectorGMat>();

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cptU([]()
    {
        cv::GArray<cv::GMat> in;
        cv::GOpaque<size_t> out = custom::GSizeOfVectorGMat::on(in);
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    });
    cptU.apply(cv::gin(in_vec), cv::gout(out_size_t_untyped), cv::compile_args(customKernel));

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GOpaque<size_t> (cv::GArray<cv::GMat>)> cptT(custom::GSizeOfVectorGMat::on);
    auto cplT = cptT.compile(cv::descr_of(in_vec), cv::compile_args(customKernel));

    cptT.apply(in_vec, out_size_t_typed1, cv::compile_args(customKernel));
    cplT(in_vec, out_size_t_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_size_t_cv = in_vec.size();

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(out_size_t_cv      == vectorSize);
    EXPECT_TRUE(out_size_t_untyped == vectorSize);
    EXPECT_TRUE(out_size_t_typed1  == vectorSize);
    EXPECT_TRUE(out_size_t_typed2  == vectorSize);
}
} // opencv_test
