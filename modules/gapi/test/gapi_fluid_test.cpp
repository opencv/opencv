// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/fluid/gfluidbuffer.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

 // FIXME: move these tests with priv() to internal suite
#include "backends/fluid/gfluidbuffer_priv.hpp"

#include "gapi_fluid_test_kernels.hpp"
#include "logger.hpp"

namespace opencv_test
{

using namespace cv::gapi_test_kernels;

namespace
{
    void WriteFunction(uint8_t* row, int nr, int w) {
        for (int i = 0; i < w; i++)
            row[i] = static_cast<uint8_t>(nr+i);
    }
    void ReadFunction1x1(const uint8_t* row, int w) {
        for (int i = 0; i < w; i++)
            std::cout << std::setw(4) << static_cast<int>(row[i]) << " ";
        std::cout << "\n";
    }
    void ReadFunction3x3(const uint8_t* rows[3], int w) {
        for (int i = 0; i < 3; i++) {
            for (int j = -1; j < w+1; j++) {
                std::cout << std::setw(4) << static_cast<int>(rows[i][j]) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

TEST(FluidBuffer, InputTest)
{
    const cv::Size buffer_size = {8,8};
    cv::Mat in_mat = cv::Mat::eye(buffer_size, CV_8U);

    cv::gapi::fluid::Buffer buffer(in_mat, true);
    cv::gapi::fluid::View  view = buffer.mkView(0, false);
    view.priv().allocate(1, {});
    view.priv().reset(1);
    int this_y = 0;

    while (this_y < buffer_size.height)
    {
        view.priv().prepareToRead();
        const uint8_t* rrow = view.InLine<uint8_t>(0);
        ReadFunction1x1(rrow, buffer_size.width);
        view.priv().readDone(1,1);

        cv::Mat from_buffer(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow));
        EXPECT_EQ(0, cvtest::norm(in_mat.row(this_y), from_buffer, NORM_INF));

        this_y++;
    }
}

TEST(FluidBuffer, CircularTest)
{
    const cv::Size buffer_size = {8,16};

    cv::gapi::fluid::Buffer buffer(cv::GMatDesc{CV_8U,1,buffer_size}, 3, 1, 0, 1,
        util::make_optional(cv::gapi::fluid::Border{cv::BORDER_CONSTANT, cv::Scalar(255)}));
    cv::gapi::fluid::View view = buffer.mkView(1, {});
    view.priv().reset(3);
    view.priv().allocate(3, {});
    buffer.debug(std::cout);

    const auto whole_line_is = [](const uint8_t *line, int len, int value)
    {
        return std::all_of(line, line+len, [&](const uint8_t v){return v == value;});
    };

    // Store all read/written data in separate Mats to compare with
    cv::Mat written_data(buffer_size, CV_8U);

    // Simulate write/read process
    int num_reads = 0, num_writes = 0;
    while (num_reads < buffer_size.height)
    {
        if (num_writes < buffer_size.height)
        {
            uint8_t* wrow = buffer.OutLine<uint8_t>();
            WriteFunction(wrow, num_writes, buffer_size.width);
            buffer.priv().writeDone();

            cv::Mat(1, buffer_size.width, CV_8U, wrow)
                .copyTo(written_data.row(num_writes));
            num_writes++;
        }
        buffer.debug(std::cout);

        if (view.ready())
        {
            view.priv().prepareToRead();
            const uint8_t* rrow[3] = {
                view.InLine<uint8_t>(-1),
                view.InLine<uint8_t>( 0),
                view.InLine<uint8_t>( 1),
            };
            ReadFunction3x3(rrow, buffer_size.width);
            view.priv().readDone(1,3);
            buffer.debug(std::cout);

            // Check borders right here
            EXPECT_EQ(255u, rrow[0][-1]);
            EXPECT_EQ(255u, rrow[0][buffer_size.width]);
            if (num_reads == 0)
            {
                EXPECT_TRUE(whole_line_is(rrow[0]-1, buffer_size.width+2, 255u));
            }
            if (num_reads == buffer_size.height-1)
            {
                EXPECT_TRUE(whole_line_is(rrow[2]-1, buffer_size.width+2, 255u));
            }

            // Check window (without borders)
            if (num_reads > 0 && num_reads < buffer_size.height-1)
            {
                // +1 everywhere since num_writes was just incremented above
                cv::Mat written_lastLine2 = written_data.row(num_writes - (2+1));
                cv::Mat written_lastLine1 = written_data.row(num_writes - (1+1));
                cv::Mat written_lastLine0 = written_data.row(num_writes - (0+1));

                cv::Mat read_prevLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[0]));
                cv::Mat read_thisLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[1]));
                cv::Mat read_nextLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[2]));

                EXPECT_EQ(0, cvtest::norm(written_lastLine2, read_prevLine, NORM_INF));
                EXPECT_EQ(0, cvtest::norm(written_lastLine1, read_thisLine, NORM_INF));
                EXPECT_EQ(0, cvtest::norm(written_lastLine0, read_nextLine, NORM_INF));
            }
            num_reads++;
        }
    }
}

TEST(FluidBuffer, OutputTest)
{
    const cv::Size buffer_size = {8,16};
    cv::Mat out_mat = cv::Mat(buffer_size, CV_8U);

    cv::gapi::fluid::Buffer buffer(out_mat, false);
    int num_writes = 0;
    while (num_writes < buffer_size.height)
    {
        uint8_t* wrow = buffer.OutLine<uint8_t>();
        WriteFunction(wrow, num_writes, buffer_size.width);
        buffer.priv().writeDone();
        num_writes++;
    }

    GAPI_LOG_INFO(NULL, "\n" << out_mat);

    // Validity check
    for (int r = 0; r < buffer_size.height; r++)
    {
        for (int c = 0; c < buffer_size.width; c++)
        {
            EXPECT_EQ(r+c, out_mat.at<uint8_t>(r, c));
        }
    }
}

TEST(Fluid, AddC_WithScalar)
{
    cv::GMat in;
    cv::GScalar s;

    cv::GComputation c(cv::GIn(in, s), cv::GOut(TAddScalar::on(in, s)));
    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    cv::Scalar in_s(100);

    auto cc = c.compile(cv::descr_of(in_mat), cv::descr_of(in_s), cv::compile_args(fluidTestPackage));

    cc(cv::gin(in_mat, in_s), cv::gout(out_mat));
    ref_mat = in_mat + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Scalar_In_Middle_Graph)
{
    cv::GMat in;
    cv::GScalar s;

    cv::GComputation c(cv::GIn(in, s), cv::GOut(TAddScalar::on(TAddCSimple::on(in, 5), s)));
    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    cv::Scalar in_s(100);

    auto cc = c.compile(cv::descr_of(in_mat), cv::descr_of(in_s), cv::compile_args(fluidTestPackage));

    cc(cv::gin(in_mat, in_s), cv::gout(out_mat));
    ref_mat = (in_mat + 5) + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Add_Scalar_To_Mat)
{
    cv::GMat in;
    cv::GScalar s;

    cv::GComputation c(cv::GIn(s, in), cv::GOut(TAddScalarToMat::on(s, in)));
    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    cv::Scalar in_s(100);

    auto cc = c.compile(cv::descr_of(in_s), cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));

    cc(cv::gin(in_s, in_mat), cv::gout(out_mat));
    ref_mat = in_mat + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Sum_2_Mats_And_Scalar)
{
    cv::GMat a, b;
    cv::GScalar s;

    cv::GComputation c(cv::GIn(a, s, b), cv::GOut(TSum2MatsAndScalar::on(a, s, b)));
    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat;
    cv::Scalar in_s(100);

    auto cc = c.compile(cv::descr_of(in_mat1), cv::descr_of(in_s), cv::descr_of(in_mat2), cv::compile_args(fluidTestPackage));

    cc(cv::gin(in_mat1, in_s, in_mat2), cv::gout(out_mat));
    ref_mat = in_mat1 + in_mat2 + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, EqualizeHist)
{
    cv::GMat in, out;
    cv::GComputation c(cv::GIn(in), cv::GOut(TEqualizeHist::on(in, TCalcHist::on(in))));

    cv::Mat in_mat(320, 480, CV_8UC1),
            out_mat(320, 480, CV_8UC1),
            ref_mat(320, 480, CV_8UC1);

    cv::randu(in_mat, 200, 240);

    auto cc = c.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));

    cc(cv::gin(in_mat), cv::gout(out_mat));

    cv::equalizeHist(in_mat, ref_mat);

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Split3)
{
    cv::GMat bgr;
    cv::GMat r,g,b;
    std::tie(b,g,r) = cv::gapi::split3(bgr);
    auto rr = TAddSimple::on(r, TId::on(b));
    auto rrr = TAddSimple::on(TId::on(rr), g);
    cv::GComputation c(bgr, TId::on(rrr));

    cv::Size sz(5120, 5120);
    cv::Mat eye_1 = cv::Mat::eye(sz, CV_8UC1);
    std::vector<cv::Mat> eyes = {eye_1, eye_1, eye_1};
    cv::Mat in_mat;
    cv::merge(eyes, in_mat);
    cv::Mat out_mat(sz, CV_8UC1);

    // G-API
    auto cc = c.compile(cv::descr_of(in_mat),
                        cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    // OCV
    std::vector<cv::Mat> chans;
    cv::split(in_mat, chans);

    // Compare
    EXPECT_EQ(0, cvtest::norm(out_mat, Mat(chans[2]*3), NORM_INF));
}

TEST(Fluid, ScratchTest)
{
    cv::GMat in;
    cv::GMat out = TPlusRow0::on(TPlusRow0::on(in));
    cv::GComputation c(in, out);

    cv::Size sz(8, 8);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat(sz, CV_8UC1);

    // OpenCV (reference)
    cv::Mat ref;
    {
        cv::Mat first_row = cv::Mat::zeros(1, sz.width, CV_8U);
        cv::Mat remaining = cv::repeat(in_mat.row(0), sz.height-1, 1);
        cv::Mat operand;
        cv::vconcat(first_row, 2*remaining, operand);
        ref = in_mat + operand;
    }
    GAPI_LOG_INFO(NULL, "\n" << ref);

    // G-API
    auto cc = c.compile(cv::descr_of(in_mat),
                        cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);
    GAPI_LOG_INFO(NULL, "\n" << out_mat);
    EXPECT_EQ(0, cvtest::norm(ref, out_mat, NORM_INF));

    cc(in_mat, out_mat);
    GAPI_LOG_INFO(NULL, "\n" << out_mat);
    EXPECT_EQ(0, cvtest::norm(ref, out_mat, NORM_INF));
}

TEST(Fluid, MultipleOutRowsTest)
{
    cv::GMat in;
    cv::GMat out = TAddCSimple::on(TAddCSimple::on(in, 1), 2);
    cv::GComputation c(in, out);

    cv::Size sz(4, 4);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat(sz, CV_8UC1);

    auto cc = c.compile(cv::descr_of(in_mat),
                        cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    std::cout << out_mat << std::endl;

    cv::Mat ocv_ref = in_mat + 1 + 2;
    EXPECT_EQ(0, cvtest::norm(ocv_ref, out_mat, NORM_INF));
}


TEST(Fluid, LPIWindow)
{
    cv::GMat in;
    cv::GMat r,g,b;
    std::tie(r,g,b) = cv::gapi::split3(in);
    cv::GMat rr = TId7x7::on(r);
    cv::GMat tmp = TAddSimple::on(rr, g);
    cv::GMat out = TAddSimple::on(tmp, b);

    cv::GComputation c(in, out);

    cv::Size sz(8, 8);

    cv::Mat eye_1 = cv::Mat::eye(sz, CV_8UC1);
    std::vector<cv::Mat> eyes = {eye_1, eye_1, eye_1};
    cv::Mat in_mat;
    cv::merge(eyes, in_mat);

    cv::Mat out_mat(sz, CV_8U);
    auto cc = c.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    //std::cout << out_mat << std::endl;

    // OpenCV reference
    cv::Mat ocv_ref = eyes[0]+eyes[1]+eyes[2];

    EXPECT_EQ(0, cvtest::norm(ocv_ref, out_mat, NORM_INF));
}

TEST(Fluid, MultipleReaders_SameLatency)
{
    //  in -> AddC -> a -> AddC -> b -> Add -> out
    //                '--> AddC -> c -'
    //
    // b and c have the same skew

    cv::GMat in;
    cv::GMat a = TAddCSimple::on(in, 1); // FIXME - align naming (G, non-G)
    cv::GMat b = TAddCSimple::on(a,  2);
    cv::GMat c = TAddCSimple::on(a,  3);
    cv::GMat out = TAddSimple::on(b, c);
    cv::GComputation comp(in, out);

    const auto sz = cv::Size(32, 32);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);
    cv::Mat out_mat_ocv (sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat_gapi);

    // Check with OpenCV
    cv::Mat tmp = in_mat + 1;
    out_mat_ocv = (tmp+2) + (tmp+3);
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(Fluid, MultipleReaders_DifferentLatency)
{
    //  in1 -> AddC -> a -> AddC -------------> b -> Add -> out
    //                 '--------------> Add --> c -'
    //                 '--> Id7x7-> d -'
    //
    // b and c have different skew (due to latency introduced by Id7x7)
    // a is ready by multiple views with different latency.

    cv::GMat in;
    cv::GMat a   = TAddCSimple::on(in, 1); // FIXME - align naming (G, non-G)
    cv::GMat b   = TAddCSimple::on(a,  2);
    cv::GMat d   = TId7x7::on(a);
    cv::GMat c   = TAddSimple::on(a, d);
    cv::GMat out = TAddSimple::on(b, c);
    cv::GComputation comp(in, out);

    const auto sz = cv::Size(32, 32);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(in_mat, out_mat_gapi);

    // Check with OpenCV
    cv::Mat ocv_a = in_mat + 1;
    cv::Mat ocv_b = ocv_a + 2;
    cv::Mat ocv_d = ocv_a;
    cv::Mat ocv_c = ocv_a + ocv_d;
    cv::Mat out_mat_ocv = ocv_b + ocv_c;
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(Fluid, MultipleOutputs)
{
    // in -> AddC -> a -> AddC ------------------> out1
    //               `--> Id7x7  --> b --> AddC -> out2

    cv::GMat in;
    cv::GMat a    = TAddCSimple::on(in, 1);
    cv::GMat b    = TId7x7::on(a);
    cv::GMat out1 = TAddCSimple::on(a, 2);
    cv::GMat out2 = TAddCSimple::on(b, 7);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out1, out2));

    const auto sz = cv::Size(32, 32);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi1(sz, CV_8UC1), out_mat_gapi2(sz, CV_8UC1);
    cv::Mat out_mat_ocv1(sz, CV_8UC1), out_mat_ocv2(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(cv::gin(in_mat), cv::gout(out_mat_gapi1, out_mat_gapi2));

    // Check with OpenCV
    out_mat_ocv1 = in_mat + 1 + 2;
    out_mat_ocv2 = in_mat + 1 + 7;
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi1, out_mat_ocv1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi2, out_mat_ocv2, NORM_INF));
}

TEST(Fluid, EmptyOutputMatTest)
{
    cv::GMat in;
    cv::GMat out = TAddCSimple::on(in, 2);
    cv::GComputation c(in, out);

    cv::Mat in_mat = cv::Mat::eye(cv::Size(32, 24), CV_8UC1);
    cv::Mat out_mat;

    auto cc = c.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));

    cc(in_mat,    out_mat);
    EXPECT_EQ(CV_8UC1, out_mat.type());
    EXPECT_EQ(32, out_mat.cols);
    EXPECT_EQ(24, out_mat.rows);
    EXPECT_TRUE(out_mat.ptr() != nullptr);
}

struct LPISequenceTest : public TestWithParam<int>{};
TEST_P(LPISequenceTest, LPISequenceTest)
{
    // in -> AddC -> a -> Blur (2lpi) -> out

    int kernelSize = GetParam();
    cv::GMat in;
    cv::GMat a = TAddCSimple::on(in, 1);
    auto blur = kernelSize == 3 ? &TBlur3x3_2lpi::on : &TBlur5x5_2lpi::on;
    cv::GMat out = blur(a, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    const auto sz = cv::Size(8, 10);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);
    cv::Mat out_mat_ocv(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(cv::gin(in_mat), cv::gout(out_mat_gapi));

    // Check with OpenCV
    cv::blur(in_mat + 1, out_mat_ocv, {kernelSize,kernelSize}, {-1,-1}, cv::BORDER_CONSTANT);
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, LPISequenceTest,
                        Values(3, 5));

struct InputImageBorderTest : public TestWithParam <std::tuple<int, int>> {};
TEST_P(InputImageBorderTest, InputImageBorderTest)
{
    cv::Size sz_in = { 320, 240 };

    int ks         = 0;
    int borderType = 0;
    std::tie(ks, borderType) = GetParam();
    cv::Mat in_mat1(sz_in, CV_8UC1);
    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat1, mean, stddev);

    cv::Size kernelSize = {ks, ks};
    cv::Point anchor = {-1, -1};
    cv::Scalar borderValue(0);

    auto gblur = ks == 3 ? &TBlur3x3::on : &TBlur5x5::on;

    GMat in;
    auto out = gblur(in, borderType, borderValue);

    Mat out_mat_gapi = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out));
    auto cc = c.compile(descr_of(in_mat1), cv::compile_args(fluidTestPackage));
    cc(gin(in_mat1), gout(out_mat_gapi));

    cv::Mat out_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    cv::blur(in_mat1, out_mat_ocv, kernelSize, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, InputImageBorderTest,
                        Combine(Values(3, 5),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)));

struct SequenceOfBlursTest : public TestWithParam <std::tuple<int>> {};
TEST_P(SequenceOfBlursTest, Test)
{
    cv::Size sz_in = { 320, 240 };

    int borderType = 0;;
    std::tie(borderType) = GetParam();
    cv::Mat in_mat(sz_in, CV_8UC1);
    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat, mean, stddev);

    cv::Point anchor = {-1, -1};
    cv::Scalar borderValue(0);

    GMat in;
    auto mid = TBlur3x3::on(in,  borderType, borderValue);
    auto out = TBlur5x5::on(mid, borderType, borderValue);

    Mat out_mat_gapi = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out));
    auto cc = c.compile(descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi));

    cv::Mat mid_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    cv::blur(in_mat, mid_mat_ocv, {3,3}, anchor, borderType);
    cv::blur(mid_mat_ocv, out_mat_ocv, {5,5}, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, SequenceOfBlursTest,
                               Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101));

struct TwoBlursTest : public TestWithParam <std::tuple<int, int, int, int, int, int, bool>> {};
TEST_P(TwoBlursTest, Test)
{
    cv::Size sz_in = { 320, 240 };

    int kernelSize1 = 0, kernelSize2 = 0;
    int borderType1 = -1, borderType2 = -1;
    cv::Scalar borderValue1{}, borderValue2{};
    bool readFromInput = false;
    std::tie(kernelSize1, borderType1, borderValue1, kernelSize2, borderType2, borderValue2, readFromInput) = GetParam();
    cv::Mat in_mat(sz_in, CV_8UC1);
    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat, mean, stddev);

    cv::Point anchor = {-1, -1};

    auto blur1 = kernelSize1 == 3 ? &TBlur3x3::on : TBlur5x5::on;
    auto blur2 = kernelSize2 == 3 ? &TBlur3x3::on : TBlur5x5::on;

    GMat in, out1, out2;
    if (readFromInput)
    {
        out1 = blur1(in, borderType1, borderValue1);
        out2 = blur2(in, borderType2, borderValue2);
    }
    else
    {
        auto mid = TAddCSimple::on(in, 0);
        out1 = blur1(mid, borderType1, borderValue1);
        out2 = blur2(mid, borderType2, borderValue2);
    }

    Mat out_mat_gapi1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat_gapi2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out1, out2));
    auto cc = c.compile(descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi1, out_mat_gapi2));

    cv::Mat out_mat_ocv1 = Mat::zeros(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = Mat::zeros(sz_in, CV_8UC1);
    cv::blur(in_mat, out_mat_ocv1, {kernelSize1, kernelSize1}, anchor, borderType1);
    cv::blur(in_mat, out_mat_ocv2, {kernelSize2, kernelSize2}, anchor, borderType2);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat_gapi1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat_gapi2, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, TwoBlursTest,
                               Combine(Values(3, 5),
                                       Values(cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT_101),
                                       Values(0),
                                       Values(3, 5),
                                       Values(cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT_101),
                                       Values(0),
                                       testing::Bool())); // Read from input directly or place a copy node at start

struct TwoReadersTest : public TestWithParam <std::tuple<int, int, int, bool>> {};
TEST_P(TwoReadersTest, Test)
{
    cv::Size sz_in = { 320, 240 };

    int kernelSize = 0;
    int borderType = -1;
    cv::Scalar borderValue;
    bool readFromInput = false;
    std::tie(kernelSize, borderType, borderValue, readFromInput) = GetParam();
    cv::Mat in_mat(sz_in, CV_8UC1);
    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat, mean, stddev);

    cv::Point anchor = {-1, -1};

    auto blur = kernelSize == 3 ? &TBlur3x3::on : TBlur5x5::on;

    GMat in, out1, out2;
    if (readFromInput)
    {
        out1 = TAddCSimple::on(in, 0);
        out2 = blur(in, borderType, borderValue);
    }
    else
    {
        auto mid = TAddCSimple::on(in, 0);
        out1 = TAddCSimple::on(mid, 0);
        out2 = blur(mid, borderType, borderValue);
    }

    Mat out_mat_gapi1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat_gapi2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out1, out2));
    auto cc = c.compile(descr_of(in_mat), cv::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi1, out_mat_gapi2));

    cv::Mat out_mat_ocv1 = Mat::zeros(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = Mat::zeros(sz_in, CV_8UC1);
    out_mat_ocv1 = in_mat;
    cv::blur(in_mat, out_mat_ocv2, {kernelSize, kernelSize}, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat_gapi1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat_gapi2, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, TwoReadersTest,
                               Combine(Values(3, 5),
                                       Values(cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT_101),
                                       Values(0),
                                       testing::Bool())); // Read from input directly or place a copy node at start

TEST(FluidTwoIslands, SanityTest)
{
    cv::Size sz_in{8,8};

    GMat in1, in2;
    auto out1 = TAddScalar::on(in1, {0});
    auto out2 = TAddScalar::on(in2, {0});

    cv::Mat in_mat1(sz_in, CV_8UC1);
    cv::Mat in_mat2(sz_in, CV_8UC1);
    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    Mat out_mat1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in1, in2), GOut(out1, out2));
    EXPECT_NO_THROW(c.apply(gin(in_mat1, in_mat2), gout(out_mat1, out_mat2), cv::compile_args(fluidTestPackage)));
    EXPECT_EQ(0, cvtest::norm(in_mat1, out_mat1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(in_mat2, out_mat2, NORM_INF));
}

struct NV12RoiTest : public TestWithParam <std::pair<cv::Size, cv::Rect>> {};
TEST_P(NV12RoiTest, Test)
{
    cv::Size y_sz;
    cv::Rect roi;
    std::tie(y_sz, roi) = GetParam();

    cv::Size uv_sz(y_sz.width / 2, y_sz.height / 2);
    cv::Size in_sz(y_sz.width, y_sz.height*3/2);

    cv::Mat in_mat = cv::Mat(in_sz, CV_8UC1);

    cv::Scalar mean   = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);

    cv::Mat y_mat  = cv::Mat(y_sz, CV_8UC1, in_mat.data);
    cv::Mat uv_mat = cv::Mat(uv_sz, CV_8UC2, in_mat.data + in_mat.step1() * y_sz.height);
    cv::Mat out_mat, out_mat_ocv;

    cv::GMat y, uv;
    auto rgb = cv::gapi::NV12toRGB(y, uv);
    cv::GComputation c(cv::GIn(y, uv), cv::GOut(rgb));

    c.apply(cv::gin(y_mat, uv_mat), cv::gout(out_mat), cv::compile_args(fluidTestPackage, cv::GFluidOutputRois{{roi}}));

    cv::cvtColor(in_mat, out_mat_ocv, cv::COLOR_YUV2RGB_NV12);

    EXPECT_EQ(0, cvtest::norm(out_mat(roi), out_mat_ocv(roi), NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, NV12RoiTest,
                        Values(std::make_pair(cv::Size{8, 8}, cv::Rect{0, 0, 8, 2})
                              ,std::make_pair(cv::Size{8, 8}, cv::Rect{0, 2, 8, 2})
                              ,std::make_pair(cv::Size{8, 8}, cv::Rect{0, 4, 8, 2})
                              ,std::make_pair(cv::Size{8, 8}, cv::Rect{0, 6, 8, 2})
                              ,std::make_pair(cv::Size{1920, 1080}, cv::Rect{0,   0, 1920, 270})
                              ,std::make_pair(cv::Size{1920, 1080}, cv::Rect{0, 270, 1920, 270})
                              ,std::make_pair(cv::Size{1920, 1080}, cv::Rect{0, 540, 1920, 270})
                              ,std::make_pair(cv::Size{1920, 1080}, cv::Rect{0, 710, 1920, 270})
                              ));

TEST(Fluid, UnusedNodeOutputCompileTest)
{
    cv::GMat in;
    cv::GMat a, b, c, d;
    std::tie(a, b, c, d) = cv::gapi::split4(in);
    cv::GMat out = cv::gapi::merge3(a, b, c);

    cv::Mat in_mat(cv::Size(8, 8), CV_8UC4);
    cv::Mat out_mat(cv::Size(8, 8), CV_8UC3);

    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    ASSERT_NO_THROW(comp.apply(cv::gin(in_mat), cv::gout(out_mat),
        cv::compile_args(cv::gapi::core::fluid::kernels())));
}

TEST(Fluid, UnusedNodeOutputReshapeTest)
{
    const auto test_size = cv::Size(8, 8);

    const auto get_compile_args = [] () {
        return cv::compile_args(
            cv::gapi::combine(
                cv::gapi::core::fluid::kernels(),
                cv::gapi::imgproc::fluid::kernels()
            )
        );
    };

    cv::GMat in;
    cv::GMat a, b, c, d;
    std::tie(a, b, c, d) = cv::gapi::split4(in);
    cv::GMat out = cv::gapi::resize(cv::gapi::merge3(a, b, c), test_size, 0.0, 0.0,
        cv::INTER_LINEAR);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    cv::Mat in_mat(test_size, CV_8UC4);
    cv::Mat out_mat(test_size, CV_8UC3);

    cv::GCompiled compiled;
    ASSERT_NO_THROW(compiled = comp.compile(descr_of(in_mat), get_compile_args()));

    in_mat = cv::Mat(test_size * 2, CV_8UC4);
    ASSERT_TRUE(compiled.canReshape());
    ASSERT_NO_THROW(compiled.reshape(descr_of(gin(in_mat)), get_compile_args()));
    ASSERT_NO_THROW(compiled(in_mat, out_mat));
}

TEST(Fluid, InvalidROIs)
{
    cv::GMat in;
    cv::GMat out = cv::gapi::add(in, in);

    cv::Mat in_mat(cv::Size(8, 8), CV_8UC3);
    cv::Mat out_mat = in_mat.clone();
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(100));

    std::vector<cv::Rect> invalid_rois =
    {
        cv::Rect(1, 0, 0, 0),
        cv::Rect(0, 1, 0, 0),
        cv::Rect(0, 0, 1, 0),
        cv::Rect(0, 0, 0, 1),
        cv::Rect(0, 0, out_mat.cols, 0),
        cv::Rect(0, 0, 0, out_mat.rows),
        cv::Rect(0, out_mat.rows, out_mat.cols, out_mat.rows),
        cv::Rect(out_mat.cols, 0, out_mat.cols, out_mat.rows),
    };

    const auto compile_args = [] (cv::Rect roi) {
        return cv::compile_args(cv::gapi::core::fluid::kernels(), GFluidOutputRois{{roi}});
    };

    for (const auto& roi : invalid_rois)
    {
        cv::GComputation comp(cv::GIn(in), cv::GOut(out));
        EXPECT_THROW(comp.apply(cv::gin(in_mat), cv::gout(out_mat), compile_args(roi)),
            std::exception);
    }
}


namespace
{
#if defined(__linux__)
uint64_t currMemoryConsumption()
{
    // check self-state via /proc information
    constexpr const char stat_file_path[] = "/proc/self/statm";
    std::ifstream proc_stat(stat_file_path);
    if (!proc_stat.is_open() || !proc_stat.good())
    {
        CV_LOG_WARNING(NULL, "Failed to open stat file: " << stat_file_path);
        return static_cast<uint64_t>(0);
    }
    std::string stat_line;
    std::getline(proc_stat, stat_line);
    uint64_t unused, data_and_stack;
    std::istringstream(stat_line) >> unused >> unused >> unused >> unused >> unused
                                  >> data_and_stack;
    CV_Assert(data_and_stack != 0);
    return data_and_stack;
}
#else
// FIXME: implement this part (at least for Windows?), right now it's enough to check Linux only
uint64_t currMemoryConsumption() { return static_cast<uint64_t>(0); }
#endif
}  // anonymous namespace

TEST(Fluid, MemoryConsumptionDoesNotGrowOnReshape)
{
    cv::GMat in;
    cv::GMat a, b, c;
    std::tie(a, b, c) = cv::gapi::split3(in);
    cv::GMat merged = cv::gapi::merge4(a, b, c, a);
    cv::GMat d, e, f, g;
    std::tie(d, e, f, g) = cv::gapi::split4(merged);
    cv::GMat out = cv::gapi::merge3(d, e, f);

    cv::Mat in_mat(cv::Size(8, 8), CV_8UC3);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::Mat out_mat;

    const auto compile_args = [] () {
        return cv::compile_args(cv::gapi::core::fluid::kernels());
    };

    cv::GCompiled compiled = cv::GComputation(cv::GIn(in), cv::GOut(out)).compile(
        cv::descr_of(in_mat), compile_args());
    ASSERT_TRUE(compiled.canReshape());

    const auto mem_before = currMemoryConsumption();
    for (int _ = 0; _ < 1000; ++_) compiled.reshape(cv::descr_of(cv::gin(in_mat)), compile_args());
    const auto mem_after = currMemoryConsumption();

    ASSERT_GE(mem_before, mem_after);
}

} // namespace opencv_test
