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
    TEST(DynamicGraph, AddProtoInputArgs)
    {
        auto ins = GIn();

        cv::GMat in;
        EXPECT_NO_THROW(ins += GIn(in));

        cv::GScalar inScalar;
        EXPECT_NO_THROW(ins += GIn(inScalar));

        cv::GMatP inGMatP;
        EXPECT_NO_THROW(ins += GIn(inGMatP));

        cv::GFrame inGFrame;
        EXPECT_NO_THROW(ins += GIn(inGFrame));

        cv::GArray<int> inGArray;
        EXPECT_NO_THROW(ins += GIn(inGArray));

        cv::GOpaque<int> inGOpaque;
        EXPECT_NO_THROW(ins += GIn(inGOpaque));
    }

    TEST(DynamicGraph, AddProtoOutputArgs)
    {
        auto outs = GOut();

        cv::GMat out;
        EXPECT_NO_THROW(outs += GOut(out));

        cv::GScalar outScalar;
        EXPECT_NO_THROW(outs += GOut(outScalar));

        cv::GMatP outGMatP;
        EXPECT_NO_THROW(outs += GOut(outGMatP));

        cv::GFrame outGFrame;
        EXPECT_NO_THROW(outs += GOut(outGFrame));

        cv::GArray<int> outGArray;
        EXPECT_NO_THROW(outs += GOut(outGArray));

        cv::GOpaque<int> outGOpaque;
        EXPECT_NO_THROW(outs += GOut(outGOpaque));
    }

    TEST(DynamicGraph, AddRunArgs)
    {
        auto in_vector = cv::gin();

        cv::Mat in_mat;
        EXPECT_NO_THROW(in_vector += cv::gin(in_mat));

#if !defined(GAPI_STANDALONE)
        cv::UMat in_umat;
        EXPECT_NO_THROW(in_vector += cv::gin(in_umat));
#endif // !defined(GAPI_STANDALONE)

        cv::Scalar in_scalar {1, 2, 3};
        EXPECT_NO_THROW(in_vector += cv::gin(in_scalar));

        cv::detail::VectorRef writer;
        EXPECT_NO_THROW(in_vector += cv::gin(writer));

        cv::detail::OpaqueRef opaque;
        EXPECT_NO_THROW(in_vector += cv::gin(opaque));
    }

    TEST(DynamicGraph, AddRunArgsP)
    {
        auto out_vector = cv::gout();

        cv::Mat out_mat;
        EXPECT_NO_THROW(out_vector += cv::gout(out_mat));

#if !defined(GAPI_STANDALONE)
        cv::UMat out_umat;
        EXPECT_NO_THROW(out_vector += cv::gout(out_umat));
#endif // !defined(GAPI_STANDALONE)

        cv::Scalar out_scalar;
        EXPECT_NO_THROW(out_vector += cv::gout(out_scalar));

        cv::detail::VectorRef writer;
        EXPECT_NO_THROW(out_vector += cv::gout(writer));

        cv::detail::OpaqueRef opaque;
        EXPECT_NO_THROW(out_vector += cv::gout(opaque));
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
