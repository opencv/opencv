// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

namespace opencv_test
{
    typedef ::testing::Types<cv::GMat, cv::GMatP, cv::GFrame,
                             cv::GScalar, cv::GOpaque<int>,
                             cv::GArray<int>> VectorProtoTypes;

    template<typename T> struct DynamicGraphProtoArgs: public ::testing::Test { using Type = T; };

    TYPED_TEST_CASE(DynamicGraphProtoArgs, VectorProtoTypes);

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoInputArgs)
    {
        using T = typename TestFixture::Type;
        auto ins = GIn();
        T in;
        EXPECT_NO_THROW(ins += GIn(in));
    }

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoOutputArgs)
    {
        using T = typename TestFixture::Type;
        auto outs = GOut();
        T out;
        EXPECT_NO_THROW(outs += GOut(out));
    }

    typedef ::testing::Types<cv::Mat,
#if !defined(GAPI_STANDALONE)
                             cv::UMat,
#endif // !defined(GAPI_STANDALONE)
                             cv::Scalar,
                             cv::detail::VectorRef,
                             cv::detail::OpaqueRef> VectorRunTypes;

    template<typename T> struct DynamicGraphRunArgs: public ::testing::Test { using Type = T; };

    TYPED_TEST_CASE(DynamicGraphRunArgs, VectorRunTypes);

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgs)
    {
        auto in_vector = cv::gin();

        using T = typename TestFixture::Type;
        T in;
        EXPECT_NO_THROW(in_vector += cv::gin(in));
    }

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgsP)
    {
        auto out_vector = cv::gout();

        using T = typename TestFixture::Type;
        T out;
        EXPECT_NO_THROW(out_vector += cv::gout(out));
    }

    TEST(DynamicGraph, ProtoInputArgsExecute)
    {
        cv::GComputation cc([]() {
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out = cv::gapi::copy(in1 + in2);

            return cv::GComputation(std::move(ins), GOut(out));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat)));
    }

    TEST(DynamicGraph, ProtoOutputArgsExecute)
    {
        cv::GComputation cc([]() {
            cv::GMat in;
            cv::GMat out1 = cv::gapi::copy(in);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::copy(in);
            outs += GOut(out2);

            return cv::GComputation(cv::GIn(in), std::move(outs));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat1;
        cv::Mat out_mat2;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat1)));
    }

    TEST(DynamicGraph, ProtoOutputInputArgsExecute)
    {
        cv::GComputation cc([]() {
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::copy(in1 + in2);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::copy(in1 + in2);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat1, out_mat2;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat1, out_mat2)));
    }

    TEST(DynamicGraph, ProtoArgsExecute)
    {
        cv::GComputation cc([]() {
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::copy(in1 + in2);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::copy(in1 + in2);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat1, out_mat2;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat1, out_mat2)));
    }

    TEST(DynamicGraph, ProtoOutputInputArgsAccuracy)
    {
        cv::Size szOut(4, 4);
        cv::GComputation cc([&](){
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::resize(in1, szOut);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        // G-API test code
        cv::Mat in_mat1( 8,  8, CV_8UC3);
        cv::Mat in_mat2(16, 16, CV_8UC3);
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

        auto in_vector = cv::gin();
        in_vector += cv::gin(in_mat1);
        in_vector += cv::gin(in_mat2);

        cv::Mat out_mat1, out_mat2;
        auto out_vector = cv::gout();
        out_vector += cv::gout(out_mat1);
        out_vector += cv::gout(out_mat2);

        cc.apply(std::move(in_vector), std::move(out_vector));

        // OCV ref code
        cv::Mat cv_out_mat1, cv_out_mat2;
        cv::resize(in_mat1, cv_out_mat1, szOut);
        cv::resize(in_mat2, cv_out_mat2, szOut);

        EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
    }

    TEST(DynamicGraph, Streaming)
    {
        cv::GComputation cc([&](){
            cv::Size szOut(4, 4);

            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::resize(in1, szOut);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        EXPECT_NO_THROW(cc.compileStreaming(cv::compile_args(cv::gapi::core::cpu::kernels())));
    }

    TEST(DynamicGraph, StreamingAccuracy)
    {
        cv::Size szOut(4, 4);
        cv::GComputation cc([&](){
//! [GIOProtoArgs += usage]
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::resize(in1, szOut);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::resize(in2, szOut);
            outs += GOut(out2);
//! [GIOProtoArgs += usage]
            return cv::GComputation(std::move(ins), std::move(outs));
        });

        // G-API test code
        cv::Mat in_mat1( 8,  8, CV_8UC3);
        cv::Mat in_mat2(16, 16, CV_8UC3);
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

//! [GRunArgs += usage]
        auto in_vector = cv::gin();
        in_vector += cv::gin(in_mat1);
        in_vector += cv::gin(in_mat2);
//! [GRunArgs += usage]

//! [GRunArgsP += usage]
        cv::Mat out_mat1, out_mat2;
        auto out_vector = cv::gout();
        out_vector += cv::gout(out_mat1);
        out_vector += cv::gout(out_mat2);
//! [GRunArgsP += usage]

        auto stream = cc.compileStreaming(cv::compile_args(cv::gapi::core::cpu::kernels()));
        stream.setSource(std::move(in_vector));

        stream.start();
        stream.pull(std::move(out_vector));
        stream.stop();

        // OCV ref code
        cv::Mat cv_out_mat1, cv_out_mat2;
        cv::resize(in_mat1, cv_out_mat1, szOut);
        cv::resize(in_mat2, cv_out_mat2, szOut);

        EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
    }
} // namespace opencv_test
