// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "api/gcomputation_priv.hpp"

namespace opencv_test
{

TEST(GMetaArg, Traits_Is_Positive)
{
    using namespace cv::detail;

    static_assert(is_meta_descr<cv::GScalarDesc>::value,
                  "GScalarDesc is a meta description type");

    static_assert(is_meta_descr<cv::GMatDesc>::value,
                  "GMatDesc is a meta description type");
}

TEST(GMetaArg, Traits_Is_Negative)
{
    using namespace cv::detail;

    static_assert(!is_meta_descr<cv::GCompileArgs>::value,
                  "GCompileArgs is NOT a meta description type");

    static_assert(!is_meta_descr<int>::value,
                  "int is NOT a meta description type");

    static_assert(!is_meta_descr<std::string>::value,
                  "str::string is NOT a meta description type");
}

TEST(GMetaArg, Traits_Are_EntireList_Positive)
{
    using namespace cv::detail;

    static_assert(are_meta_descrs<cv::GScalarDesc>::value,
                  "GScalarDesc is a meta description type");

    static_assert(are_meta_descrs<cv::GMatDesc>::value,
                  "GMatDesc is a meta description type");

    static_assert(are_meta_descrs<cv::GMatDesc, cv::GScalarDesc>::value,
                  "Both GMatDesc and GScalarDesc are meta types");
}

TEST(GMetaArg, Traits_Are_EntireList_Negative)
{
    using namespace cv::detail;

    static_assert(!are_meta_descrs<cv::GCompileArgs>::value,
                  "GCompileArgs is NOT among meta types");

    static_assert(!are_meta_descrs<int, std::string>::value,
                  "Both int and std::string is NOT among meta types");

    static_assert(!are_meta_descrs<cv::GMatDesc, cv::GScalarDesc, int>::value,
                  "List of type is not valid for meta as there\'s int");

    static_assert(!are_meta_descrs<cv::GMatDesc, cv::GScalarDesc, cv::GCompileArgs>::value,
                  "List of type is not valid for meta as there\'s GCompileArgs");
}

TEST(GMetaArg, Traits_Are_ButLast_Positive)
{
    using namespace cv::detail;

    static_assert(are_meta_descrs_but_last<cv::GScalarDesc, int>::value,
                  "List is valid (int is omitted)");

    static_assert(are_meta_descrs_but_last<cv::GMatDesc, cv::GScalarDesc, cv::GCompileArgs>::value,
                  "List is valid (GCompileArgs are omitted)");
}

TEST(GMetaArg, Traits_Are_ButLast_Negative)
{
    using namespace cv::detail;

    static_assert(!are_meta_descrs_but_last<int, std::string>::value,
                  "Both int is NOT among meta types (std::string is omitted)");

    static_assert(!are_meta_descrs_but_last<cv::GMatDesc, cv::GScalarDesc, int, int>::value,
                  "List of type is not valid for meta as there\'s two ints");

    static_assert(!are_meta_descrs_but_last<cv::GMatDesc, cv::GScalarDesc, cv::GCompileArgs, float>::value,
                  "List of type is not valid for meta as there\'s GCompileArgs");
}

TEST(GMetaArg, Can_Get_Metas_From_Input_Run_Args)
{
    cv::Mat m(3, 3, CV_8UC3);
    cv::Scalar s;
    std::vector<int> v;

    GMatDesc m_desc;
    GMetaArgs meta_args = descr_of(cv::gin(m, s, v));

    EXPECT_EQ(meta_args.size(), 3u);
    EXPECT_NO_THROW(m_desc = util::get<cv::GMatDesc>(meta_args[0]));
    EXPECT_NO_THROW(util::get<cv::GScalarDesc>(meta_args[1]));
    EXPECT_NO_THROW(util::get<cv::GArrayDesc>(meta_args[2]));

    EXPECT_EQ(CV_8U, m_desc.depth);
    EXPECT_EQ(3, m_desc.chan);
    EXPECT_EQ(cv::gapi::own::Size(3, 3), m_desc.size);
}

TEST(GMetaArg, Can_Get_Metas_From_Output_Run_Args)
{
    cv::Mat m(3, 3, CV_8UC3);
    cv::Scalar s;
    std::vector<int> v;

    GMatDesc m_desc;
    GRunArgsP out_run_args = cv::gout(m, s, v);
    GMetaArg m_meta = descr_of(out_run_args[0]);
    GMetaArg s_meta = descr_of(out_run_args[1]);
    GMetaArg v_meta = descr_of(out_run_args[2]);

    EXPECT_NO_THROW(m_desc = util::get<cv::GMatDesc>(m_meta));
    EXPECT_NO_THROW(util::get<cv::GScalarDesc>(s_meta));
    EXPECT_NO_THROW(util::get<cv::GArrayDesc>(v_meta));

    EXPECT_EQ(CV_8U, m_desc.depth);
    EXPECT_EQ(3, m_desc.chan);
    EXPECT_EQ(cv::Size(3, 3), m_desc.size);
}

} // namespace opencv_test
