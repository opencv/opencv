// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"

#include <stdexcept>
#include <ade/util/iota_range.hpp>
#include "logger.hpp"

#include <opencv2/gapi/plaidml/core.hpp>
#include <opencv2/gapi/plaidml/plaidml.hpp>

namespace opencv_test
{

#ifdef HAVE_PLAIDML

inline cv::gapi::plaidml::config getConfig()
{
    auto read_var_from_env = [](const char* env)
    {
        const char* raw = std::getenv(env);
        if (!raw)
        {
            cv::util::throw_error(std::runtime_error(std::string(env) + " is't set"));
        }

        return std::string(raw);
    };

    auto dev_id = read_var_from_env("PLAIDML_DEVICE");
    auto trg_id = read_var_from_env("PLAIDML_TARGET");

    return cv::gapi::plaidml::config{std::move(dev_id),
                                     std::move(trg_id)};
}

TEST(GAPI_PlaidML_Pipelines, SimpleArithmetic)
{
    cv::Size size(1920, 1080);
    int type = CV_8UC1;

    cv::Mat in_mat1(size, type);
    cv::Mat in_mat2(size, type);

    // NB: What about overflow ? PlaidML doesn't handle it
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(127));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(127));

    cv::Mat out_mat(size, type, cv::Scalar::all(0));
    cv::Mat ref_mat(size, type, cv::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    cv::GMat in1, in2;
    auto out = in1 + in2;

    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out));
    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat),
               cv::compile_args(getConfig(),
                                cv::gapi::use_only{cv::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    cv::add(in_mat1, in_mat2, ref_mat, cv::noArray(), type);

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

// FIXME PlaidML cpu backend does't support bitwise operations
TEST(GAPI_PlaidML_Pipelines, DISABLED_ComplexArithmetic)
{
    cv::Size size(1920, 1080);
    int type = CV_8UC1;

    cv::Mat in_mat1(size, type);
    cv::Mat in_mat2(size, type);

    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat out_mat(size, type, cv::Scalar::all(0));
    cv::Mat ref_mat(size, type, cv::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    cv::GMat in1, in2;
    auto out = in1 | (in2 ^ (in1 & (in2 + (in1 - in2))));

    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out));
    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat),
               cv::compile_args(getConfig(),
                                cv::gapi::use_only{cv::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    cv::subtract(in_mat1,  in_mat2, ref_mat, cv::noArray(), type);
    cv::add(in_mat2, ref_mat, ref_mat, cv::noArray(), type);
    cv::bitwise_and(in_mat1, ref_mat, ref_mat);
    cv::bitwise_xor(in_mat2, ref_mat, ref_mat);
    cv::bitwise_or(in_mat1, ref_mat, ref_mat);

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

TEST(GAPI_PlaidML_Pipelines, TwoInputOperations)
{
    cv::Size size(1920, 1080);
    int type = CV_8UC1;

    constexpr int kNumInputs = 4;
    std::vector<cv::Mat> in_mat(kNumInputs, cv::Mat(size, type));
    for (int i = 0; i < kNumInputs; ++i)
    {
        cv::randu(in_mat[i], cv::Scalar::all(0), cv::Scalar::all(60));
    }

    cv::Mat out_mat(size, type, cv::Scalar::all(0));
    cv::Mat ref_mat(size, type, cv::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    cv::GMat in[4];
    auto out = (in[3] - in[0]) + (in[2] - in[1]);

    cv::GComputation comp(cv::GIn(in[0], in[1], in[2], in[3]), cv::GOut(out));

    // FIXME Doesn't work just apply(in_mat, out_mat, ...)
    comp.apply(cv::gin(in_mat[0], in_mat[1], in_mat[2], in_mat[3]), cv::gout(out_mat),
               cv::compile_args(getConfig(),
                                cv::gapi::use_only{cv::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    cv::subtract(in_mat[3], in_mat[0],  ref_mat, cv::noArray(), type);
    cv::add(ref_mat, in_mat[2], ref_mat, cv::noArray(), type);
    cv::subtract(ref_mat, in_mat[1], ref_mat, cv::noArray(), type);

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

TEST(GAPI_PlaidML_Pipelines, TwoOutputOperations)
{
    cv::Size size(1920, 1080);
    int type = CV_8UC1;

    constexpr int kNumInputs = 4;
    std::vector<cv::Mat> in_mat(kNumInputs, cv::Mat(size, type));
    for (int i = 0; i < kNumInputs; ++i)
    {
        cv::randu(in_mat[i], cv::Scalar::all(0), cv::Scalar::all(60));
    }

    std::vector<cv::Mat> out_mat(kNumInputs, cv::Mat(size, type, cv::Scalar::all(0)));
    std::vector<cv::Mat> ref_mat(kNumInputs, cv::Mat(size, type, cv::Scalar::all(0)));

    ////////////////////////////// G-API //////////////////////////////////////
    cv::GMat in[4], out[2];
    out[0] = in[0] + in[3];
    out[1] = in[1] + in[2];

    cv::GComputation comp(cv::GIn(in[0], in[1], in[2], in[3]), cv::GOut(out[0], out[1]));

    // FIXME Doesn't work just apply(in_mat, out_mat, ...)
    comp.apply(cv::gin(in_mat[0], in_mat[1], in_mat[2], in_mat[3]),
               cv::gout(out_mat[0], out_mat[1]),
               cv::compile_args(getConfig(),
                                cv::gapi::use_only{cv::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    cv::add(in_mat[0], in_mat[3], ref_mat[0], cv::noArray(), type);
    cv::add(in_mat[1], in_mat[2], ref_mat[1], cv::noArray(), type);

    EXPECT_EQ(0, cv::norm(out_mat[0], ref_mat[0]));
    EXPECT_EQ(0, cv::norm(out_mat[1], ref_mat[1]));
}

#else // HAVE_PLAIDML

TEST(GAPI_PlaidML_Pipelines, ThrowIfPlaidMLNotFound)
{
    ASSERT_ANY_THROW(cv::gapi::core::plaidml::kernels());
}

#endif // HAVE_PLAIDML

} // namespace opencv_test
