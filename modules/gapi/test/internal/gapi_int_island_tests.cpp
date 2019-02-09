// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gcompiled_priv.hpp"

namespace opencv_test
{

////////////////////////////////////////////////////////////////////////////////
// Tests on a plain graph
//
// (in) -> Blur1 -> (tmp0) -> Blur2 -> (tmp1) -> Blur3 -> (tmp2) -> Blur4 -> (out)
//
namespace
{
    struct PlainIslandsFixture
    {
        cv::GMat in;
        cv::GMat tmp[3];
        cv::GMat out;

        PlainIslandsFixture()
        {
            tmp[0] = cv::gapi::boxFilter(in,     -1, cv::Size(3,3));
            tmp[1] = cv::gapi::boxFilter(tmp[0], -1, cv::Size(3,3));
            tmp[2] = cv::gapi::boxFilter(tmp[1], -1, cv::Size(3,3));
            out    = cv::gapi::boxFilter(tmp[2], -1, cv::Size(3,3));
        }
    };

    struct Islands: public ::testing::Test, public PlainIslandsFixture {};

    using GIntArray = GArray<int>;

    G_TYPED_KERNEL(CreateMatWithDiag, <GMat(GIntArray)>, "test.array.create_mat_with_diag")
    {
        static GMatDesc outMeta(const GArrayDesc&) { return cv::GMatDesc{CV_32S, 1,{3, 3}}; }
    };

    GAPI_OCV_KERNEL(CreateMatWithDiagImpl, CreateMatWithDiag)
    {
        static void run(const std::vector<int> &in, cv::Mat& out)
        {
            auto size = static_cast<int>(in.size());
            out = Mat::zeros(size, size, CV_32SC1);
            for(int i = 0; i < out.rows; i++)
            {
                auto* row = out.ptr<int>(i);
                row[i] = in[i];
            }
        }
    };

    G_TYPED_KERNEL(Mat2Array, <GIntArray(GMat)>, "test.array.mat2array")
    {
        static GArrayDesc outMeta(const GMatDesc&) { return empty_array_desc(); }
    };

    GAPI_OCV_KERNEL(Mat2ArrayImpl, Mat2Array)
    {
        static void run(const cv::Mat& in, std::vector<int> &out)
        {
            GAPI_Assert(in.depth() == CV_32S && in.isContinuous());
            out.reserve(in.cols * in.rows);
            out.assign((int*)in.datastart, (int*)in.dataend);
        }
    };
}

TEST_F(Islands, SmokeTest)
{
    // (in) -> Blur1 -> (tmp0) -> Blur2 -> (tmp1) -> Blur3 -> (tmp2) -> Blur4 -> (out)
    //                         :        "test"             :
    //                         :<------------------------->:
    cv::gapi::island("test", cv::GIn(tmp[0]), cv::GOut(tmp[2]));
    auto cc = cv::GComputation(in, out).compile(cv::GMatDesc{CV_8U,1,{640,480}});

    const auto &gm = cc.priv().model();
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);

    // tmp1 and tmp3 is not a part of any island
    EXPECT_FALSE(gm.metadata(tmp0_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp2_nh).contains<cv::gimpl::Island>());

    // tmp2 is part of "test" island
    EXPECT_TRUE(gm.metadata(tmp1_nh).contains<cv::gimpl::Island>());
    EXPECT_EQ("test", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);
}

TEST_F(Islands, TwoIslands)
{
    // (in) -> Blur1 -> (tmp0) -> Blur2 -> (tmp1) -> Blur3 -> (tmp2) -> Blur4 -> (out)
    //       :  "test1"                     :  : "test2"                          :
    //       :<---------------------------->:  :<--------------------------------->
    EXPECT_NO_THROW(cv::gapi::island("test1", cv::GIn(in),     cv::GOut(tmp[1])));
    EXPECT_NO_THROW(cv::gapi::island("test2", cv::GIn(tmp[1]), cv::GOut(out)));

    auto cc = cv::GComputation(in, out).compile(cv::GMatDesc{CV_8U,1,{640,480}});
    const auto &gm = cc.priv().model();
    const auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in);
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto out_nh  = cv::gimpl::GModel::dataNodeOf(gm, out);

    // Only tmp0 and tmp2 should be listed in islands.
    EXPECT_TRUE (gm.metadata(tmp0_nh).contains<cv::gimpl::Island>());
    EXPECT_TRUE (gm.metadata(tmp2_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in_nh)  .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp1_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh) .contains<cv::gimpl::Island>());

    EXPECT_EQ("test1", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("test2", gm.metadata(tmp2_nh).get<cv::gimpl::Island>().island);
}

// FIXME: Disabled since currently merge procedure merges two into one
// successfully
TEST_F(Islands, DISABLED_Two_Islands_With_Same_Name_Should_Fail)
{
    // (in) -> Blur1 -> (tmp0) -> Blur2 -> (tmp1) -> Blur3 -> (tmp2) -> Blur4 -> (out)
    //       :  "test1"                     :  : "test1"                          :
    //       :<---------------------------->:  :<--------------------------------->

    EXPECT_NO_THROW(cv::gapi::island("test1", cv::GIn(in),     cv::GOut(tmp[1])));
    EXPECT_NO_THROW(cv::gapi::island("test1", cv::GIn(tmp[1]), cv::GOut(out)));

    EXPECT_ANY_THROW(cv::GComputation(in, out).compile(cv::GMatDesc{CV_8U,1,{640,480}}));
}


// (in) -> Blur1 -> (tmp0) -> Blur2 -> (tmp1) -> Blur3 -> (tmp2) -> Blur4 -> (out)
//       :          "test1":            :              :
//       :<----------------:----------->:              :
//                         :                           :
//                         :        "test2"            :
//                         :<------------------------->:
TEST_F(Islands, OverlappingIslands1)
{
    EXPECT_NO_THROW (cv::gapi::island("test1", cv::GIn(in),     cv::GOut(tmp[1])));
    EXPECT_ANY_THROW(cv::gapi::island("test2", cv::GIn(tmp[0]), cv::GOut(tmp[2])));
}

TEST_F(Islands, OverlappingIslands2)
{
    EXPECT_NO_THROW (cv::gapi::island("test2", cv::GIn(tmp[0]), cv::GOut(tmp[2])));
    EXPECT_ANY_THROW(cv::gapi::island("test1", cv::GIn(in),     cv::GOut(tmp[1])));
}

////////////////////////////////////////////////////////////////////////////////
// Tests on a complex graph
//
// (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
//                             ^                         ^
// (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
//                   :
//                   `------------> Median -> (tmp3) --> Blur -------> (out1)
//
namespace
{
    struct ComplexIslandsFixture
    {
        cv::GMat    in[2];
        cv::GMat    tmp[4];
        cv::GScalar scl;
        cv::GMat    out[2];

        ComplexIslandsFixture()
        {
            tmp[0] = cv::gapi::bitwise_not(in[0]);
            tmp[1] = cv::gapi::boxFilter(in[1], -1, cv::Size(3,3));
            tmp[2] = tmp[0] + tmp[1]; // FIXME: handle tmp[2] = tmp[0]+tmp[2] typo
            scl    = cv::gapi::sum(tmp[1]);
            tmp[3] = cv::gapi::medianBlur(tmp[1], 3);
            out[0] = tmp[2] + scl;
            out[1] = cv::gapi::boxFilter(tmp[3], -1, cv::Size(3,3));
        }
    };

    struct ComplexIslands: public ::testing::Test, public ComplexIslandsFixture {};
} // namespace

TEST_F(ComplexIslands, SmokeTest)
{
    //       isl0                                          #internal1
    //       ...........................                   ........
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //       :............ ........^...:                   :.^....:
    //                   ...       :                         :
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //                   :                                     isl1
    //                   :           ..............................
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)
    //                               :............................:

    cv::gapi::island("isl0", cv::GIn(in[0], tmp[1]),  cv::GOut(tmp[2]));
    cv::gapi::island("isl1", cv::GIn(tmp[1]), cv::GOut(out[1]));
    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                 cv::GMatDesc{CV_8U,1,{640,480}});
    const auto &gm = cc.priv().model();
    const auto in0_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    const auto in1_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[3]);
    const auto scl_nh  = cv::gimpl::GModel::dataNodeOf(gm, scl);
    const auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out[0]);
    const auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out[1]);

    // tmp0, tmp3 are in islands, others are not
    EXPECT_TRUE(gm.metadata(tmp0_nh) .contains<cv::gimpl::Island>()); // isl0
    EXPECT_TRUE(gm.metadata(tmp3_nh) .contains<cv::gimpl::Island>()); // isl1
    EXPECT_FALSE(gm.metadata(in0_nh) .contains<cv::gimpl::Island>()); // (input is never fused)
    EXPECT_FALSE(gm.metadata(in1_nh) .contains<cv::gimpl::Island>()); // (input is never fused)
    EXPECT_TRUE (gm.metadata(tmp1_nh).contains<cv::gimpl::Island>()); // <internal island>
    EXPECT_FALSE(gm.metadata(tmp2_nh).contains<cv::gimpl::Island>()); // #not fused as cycle-causing#
    EXPECT_FALSE(gm.metadata(scl_nh) .contains<cv::gimpl::Island>()); // #not fused as cycle-causing#
    EXPECT_FALSE(gm.metadata(out0_nh).contains<cv::gimpl::Island>()); // (output is never fused)
    EXPECT_FALSE(gm.metadata(out1_nh).contains<cv::gimpl::Island>()); // (output is never fused)

    EXPECT_EQ("isl0", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl1", gm.metadata(tmp3_nh).get<cv::gimpl::Island>().island);

    EXPECT_NE("isl0", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);
    EXPECT_NE("isl1", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);

    // FIXME: Add a test with same graph for Fusion and check GIslandModel
}

TEST_F(ComplexIslands, DistinictIslandsWithSameName)
{
    //       isl0
    //       ...........................
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //       :............ ........^...:                     ^
    //                   ...       :                         :
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //                   :                                     isl0
    //                   :           ..............................
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)
    //                               :............................:

    cv::gapi::island("isl0", cv::GIn(in[0], tmp[1]),  cv::GOut(tmp[2]));
    cv::gapi::island("isl0", cv::GIn(tmp[1]), cv::GOut(out[1]));

    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]));

    EXPECT_ANY_THROW(cc.compile(cv::GMatDesc{CV_8U,1,{640,480}},
                                cv::GMatDesc{CV_8U,1,{640,480}}));
}

TEST_F(ComplexIslands, FullGraph)
{
    cv::gapi::island("isl0",   cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]));
    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                 cv::GMatDesc{CV_8U,1,{640,480}});
    const auto &gm = cc.priv().model();
    std::vector<ade::NodeHandle> handles_inside = {
        cv::gimpl::GModel::dataNodeOf(gm, tmp[0]),
        cv::gimpl::GModel::dataNodeOf(gm, tmp[1]),
        cv::gimpl::GModel::dataNodeOf(gm, tmp[2]),
        cv::gimpl::GModel::dataNodeOf(gm, tmp[3]),
        cv::gimpl::GModel::dataNodeOf(gm, scl),
    };
    std::vector<ade::NodeHandle> handles_outside = {
        cv::gimpl::GModel::dataNodeOf(gm, in[0]),
        cv::gimpl::GModel::dataNodeOf(gm, in[1]),
        cv::gimpl::GModel::dataNodeOf(gm, out[0]),
        cv::gimpl::GModel::dataNodeOf(gm, out[1]),
    };

    for (auto nh_inside : handles_inside)
    {
        EXPECT_EQ("isl0", gm.metadata(nh_inside).get<cv::gimpl::Island>().island);
    }
    for (auto nh_outside : handles_outside)
    {
        EXPECT_FALSE(gm.metadata(nh_outside).contains<cv::gimpl::Island>());
    }
}

TEST_F(ComplexIslands, ViaScalar)
{
    //
    //        .........................................#internal0.
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //        :....................^.........................^...:
    //                             :                         :
    //        .....................:.........(isl0).         :
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //        :..........:.........................:
    //                   :
    //                   :            ..................#internal1.
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)
    //                                :...........................:

    cv::gapi::island("isl0",   cv::GIn(in[1]), cv::GOut(scl));
    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                 cv::GMatDesc{CV_8U,1,{640,480}});
    const auto &gm = cc.priv().model();

    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[3]);

    EXPECT_NE("isl0", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island); // <internal>
    EXPECT_EQ("isl0", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island); // isl0
    EXPECT_NE("isl0", gm.metadata(tmp2_nh).get<cv::gimpl::Island>().island); // <internal>
    EXPECT_NE("isl0", gm.metadata(tmp3_nh).get<cv::gimpl::Island>().island); // <internal>

    std::vector<ade::NodeHandle> handles_outside = {
        cv::gimpl::GModel::dataNodeOf(gm, in[0]),
        cv::gimpl::GModel::dataNodeOf(gm, in[1]),
        cv::gimpl::GModel::dataNodeOf(gm, scl),
        cv::gimpl::GModel::dataNodeOf(gm, out[0]),
        cv::gimpl::GModel::dataNodeOf(gm, out[1]),
    };
    for (auto nh_outside : handles_outside)
    {
        EXPECT_FALSE(gm.metadata(nh_outside).contains<cv::gimpl::Island>());
    }
}

TEST_F(ComplexIslands, BorderDataIsland)
{
    //       .................................(isl0)..
    //       :                                       :
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //       :                     ^                 :       ^
    //       :                     :                 :       :
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //       :...........:...........................:
    //                :  :  :
    //                :  :  :.........................................(isl1)..
    //                :  `------------> Median -> (tmp3) --> Blur -------> (out1)
    //                :                                                      :
    //                :......................................................:

    cv::gapi::island("isl0", cv::GIn(in[0],  in[1]), cv::GOut(tmp[2], scl));
    cv::gapi::island("isl1", cv::GIn(tmp[1]),        cv::GOut(out[1]));

    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                 cv::GMatDesc{CV_8U,1,{640,480}});
    const auto &gm = cc.priv().model();
    const auto in0_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    const auto in1_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[3]);
    const auto scl_nh  = cv::gimpl::GModel::dataNodeOf(gm, scl);
    const auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out[0]);
    const auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out[1]);

    // Check handles inside isl0
    EXPECT_EQ("isl0", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl0", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);
    // ^^^ Important - tmp1 is assigned to isl0, not isl1

    // Check handles inside isl1
    EXPECT_EQ("isl1", gm.metadata(tmp3_nh).get<cv::gimpl::Island>().island);

    // Check outside handles
    EXPECT_FALSE(gm.metadata(in0_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp2_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(scl_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out0_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out1_nh).contains<cv::gimpl::Island>());
}


TEST_F(ComplexIslands, IncompleteSpec)
{
    //       isl0
    //       ...........................
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //       :...........xxx.......^...:                     ^
    //                             :                         :
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //                   :
    //                   :
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)
    //

    // tmp1 is missing in the below spec
    EXPECT_ANY_THROW(cv::gapi::island("isl0", cv::GIn(in[0]),  cv::GOut(tmp[2])));

    // empty range
    EXPECT_ANY_THROW(cv::gapi::island("isl1", cv::GIn(tmp[2]),  cv::GOut(tmp[2])));
}

TEST_F(ComplexIslands, InputOperationFromDifferentIslands)
{
    //       isl1
    //       ...........................                   ........
    // (in0)--> Not  -> (tmp0) --> Add :--------> (tmp2)-->: AddC : -------> (out0)
    //       :......................^..:                   :  ^   :
    //       isl0                   :                      :  :   :
    //       .......................:.......................  :   :
    // (in1) :-> Blur -> (tmp1) ----'--> Sum ----> (scl0) -----   :
    //       :....................................................:
    //       isl0        :
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)
    //

    cv::gapi::island("isl0", cv::GIn(in[1], tmp[2]), cv::GOut(out[0]));
    cv::gapi::island("isl1", cv::GIn(in[0], tmp[1]), cv::GOut(tmp[2]));
    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                cv::GMatDesc{CV_8U,1,{640,480}});

    const auto &gm = cc.priv().model();
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);

    EXPECT_EQ("isl1", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl0", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);
    EXPECT_FALSE(gm.metadata(tmp2_nh).contains<cv::gimpl::Island>());
}

TEST_F(ComplexIslands, NoWayBetweenNodes)
{
    // (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)
    //                             ^                         ^
    // (in1) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //                   :
    //                   `------------> Median -> (tmp3) --> Blur -------> (out1)

    EXPECT_ANY_THROW(cv::gapi::island("isl0", cv::GIn(in[1]), cv::GOut(tmp[0])));
}

TEST_F(ComplexIslands, IslandsContainUnusedPart)
{
    // Unused part of the graph
    // x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
    // x                                                                               x
    // x(in0) -> Not  -> (tmp0) --> Add ---------> (tmp2)---> AddC ---------> (out0)   x
    // x                             ^                         ^                       x
    // x x x x x x x x x x x x x x x | x x                     |                       x
    //                               |   x                     |                       x
    //          ......               |   x                     |                       x
    // (in1) -> :Blur:----------> (tmp1) x-----> Sum ------> (scl0)                    x
    //          ......    :              x x x x x x x x x x x x x x x x x x x x x x x x
    //          isl0
    //                    :
    //                    `------------> Median -> (tmp3) --> Blur -------> (out1)

    cv::gapi::island("isl0", cv::GIn(in[1]), cv::GOut(scl));
    auto cc = cv::GComputation(cv::GIn(in[1]), cv::GOut(out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}});

    const auto &gm = cc.priv().model();
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);

    //The output 0 is not specified in the graph
    //means that there will not be a node scl, so that  tmp1 will not assign to the island
    // FIXME Check that blur assigned to island using the function producerOf
    // After merge islands fusion
    EXPECT_FALSE(gm.metadata(tmp1_nh) .contains<cv::gimpl::Island>());
}

TEST_F(ComplexIslands, FullGraphInTwoIslands)
{
    //       isl0
    //          ..................................................
    // (in0) -> :Not -> (tmp0) --> Add ---------> (tmp2) --> AddC: -------> (out0)
    //          ...................^....                     ^   :
    //          ...............    |   :                     :   :
    // (in1) -> :Blur-> (tmp1):----'-->:Sum ----> (scl0) ----'   :
    //          ........ |    :        ...........................
    //          isl1   : |    :............................................
    //                 : `------------> Median -> (tmp3) --> Blur ------->:(out1)
    //                 ....................................................

    cv::gapi::island("isl0", cv::GIn(in[0], tmp[1]), cv::GOut(out[0]));
    cv::gapi::island("isl1", cv::GIn(in[1]), cv::GOut(out[1]));
    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                cv::GMatDesc{CV_8U,1,{640,480}});

    const auto &gm = cc.priv().model();
    const auto in0_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    const auto in1_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[3]);
    const auto scl_nh  = cv::gimpl::GModel::dataNodeOf(gm, scl);
    const auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out[0]);
    const auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out[1]);

    // Check handles inside isl0
    EXPECT_EQ("isl0", gm.metadata(tmp0_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl0", gm.metadata(tmp2_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl0", gm.metadata(scl_nh).get<cv::gimpl::Island>().island);

    // Check handles inside isl1
    EXPECT_EQ("isl1", gm.metadata(tmp1_nh).get<cv::gimpl::Island>().island);
    EXPECT_EQ("isl1", gm.metadata(tmp3_nh).get<cv::gimpl::Island>().island);

    // Check outside handles
    EXPECT_FALSE(gm.metadata(in0_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out0_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out1_nh).contains<cv::gimpl::Island>());
}

TEST_F(ComplexIslands, OnlyOperationsAssignedToIslands)
{
    cv::gapi::island("isl0", cv::GIn(in[1]), cv::GOut(tmp[1]));
    cv::gapi::island("isl1", cv::GIn(tmp[1]), cv::GOut(scl));
    cv::gapi::island("isl2", cv::GIn(scl, tmp[2]), cv::GOut(out[0]));
    cv::gapi::island("isl3", cv::GIn(in[0]), cv::GOut(tmp[0]));
    cv::gapi::island("isl4", cv::GIn(tmp[0], tmp[1]), cv::GOut(tmp[2]));
    cv::gapi::island("isl5", cv::GIn(tmp[1]), cv::GOut(tmp[3]));
    cv::gapi::island("isl6", cv::GIn(tmp[3]), cv::GOut(out[1]));

    auto cc = cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out[0], out[1]))
        .compile(cv::GMatDesc{CV_8U,1,{640,480}},
                cv::GMatDesc{CV_8U,1,{640,480}});

    const auto &gm = cc.priv().model();
    //FIXME: Check that operation handles are really assigned to isl0..isl6
    const auto in0_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    const auto in1_nh  = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    const auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[0]);
    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[1]);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[2]);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp[3]);
    const auto scl_nh  = cv::gimpl::GModel::dataNodeOf(gm, scl);
    const auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out[0]);
    const auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out[1]);

    EXPECT_FALSE(gm.metadata(in0_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp0_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp1_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp2_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp3_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(scl_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out0_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out1_nh).contains<cv::gimpl::Island>());
}

namespace
{
    struct IslandStructureWithGArray
    {
        GIntArray in, out;
        GMat tmp;

        IslandStructureWithGArray()
        {
            tmp = CreateMatWithDiag::on(in);
            out = Mat2Array::on(tmp);
        }
    };

    struct IslandsWithGArray: public ::testing::Test, public IslandStructureWithGArray {};
} // namespace

TEST_F(IslandsWithGArray, IslandWithGArrayAsInput)
{
    cv::gapi::island("isl0", cv::GIn(in), cv::GOut(tmp));

    const auto pkg = cv::gapi::kernels<CreateMatWithDiagImpl, Mat2ArrayImpl>();
    auto cc = cv::GComputation(cv::GIn(in), GOut(out)).compile(cv::empty_array_desc(), cv::compile_args(pkg));
    const auto &gm = cc.priv().model();

    const auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in.strip());
    const auto out_nh  = cv::gimpl::GModel::dataNodeOf(gm, out.strip());
    const auto tmp_nh  = cv::gimpl::GModel::dataNodeOf(gm, tmp);
    GAPI_Assert(tmp_nh->inNodes().size() == 1);
    const auto create_diag_mat_nh = tmp_nh->inNodes().front();

    EXPECT_EQ("isl0", gm.metadata(create_diag_mat_nh).get<cv::gimpl::Island>().island);
    EXPECT_FALSE(gm.metadata(in_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh) .contains<cv::gimpl::Island>());
}

TEST_F(IslandsWithGArray, IslandWithGArrayAsOutput)
{
    cv::gapi::island("isl0", cv::GIn(tmp), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<CreateMatWithDiagImpl, Mat2ArrayImpl>();
    auto cc = cv::GComputation(cv::GIn(in), GOut(out)).compile(cv::empty_array_desc(), cv::compile_args(pkg));
    const auto &gm = cc.priv().model();

    const auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in.strip());
    const auto out_nh  = cv::gimpl::GModel::dataNodeOf(gm, out.strip());
    const auto tmp_nh  = cv::gimpl::GModel::dataNodeOf(gm, tmp);
    GAPI_Assert(tmp_nh->inNodes().size() == 1);
    const auto mat2array_nh = out_nh->inNodes().front();

    EXPECT_EQ("isl0", gm.metadata(mat2array_nh).get<cv::gimpl::Island>().island);
    EXPECT_FALSE(gm.metadata(in_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh) .contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh) .contains<cv::gimpl::Island>());
}
////////////////////////////////////////////////////////////////////////////////
// Wrong input tests on island name
//
namespace
{
    struct CheckName : public TestWithParam<std::tuple<bool, const char*> >,
                       public PlainIslandsFixture
    {
        void assignIsland(const std::string &s)
        {
            cv::gapi::island(s, cv::GIn(tmp[0]), cv::GOut(tmp[2]));
        };
    };
    TEST_P(CheckName, Test)
    {
        bool correct = false;
        const char *name = "";
        std::tie(correct, name) = GetParam();
        if (correct) EXPECT_NO_THROW(assignIsland(name));
        else EXPECT_ANY_THROW(assignIsland(name));
    }
} // namespace
INSTANTIATE_TEST_CASE_P(IslandName, CheckName,
                        Values(std::make_tuple(true,  "name"),
                               std::make_tuple(true,  " name "),
                               std::make_tuple(true,  " n a m e "),
                               std::make_tuple(true,  " 123 $$ %%"),
                               std::make_tuple(true,  ".: -"),
                               std::make_tuple(false, ""),
                               std::make_tuple(false, " "),
                               std::make_tuple(false, " \t "),
                               std::make_tuple(false, "  \t \t   ")));

// FIXME: add <internal> test on unrollExpr() use for islands

} // opencv_test
