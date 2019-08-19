// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


// FIXME: move out from Common

#include "../test_precomp.hpp"
#include <opencv2/gapi/cpu/core.hpp>

#include <ade/util/algorithm.hpp>

namespace opencv_test
{
namespace
{
    G_TYPED_KERNEL(GCompoundDoubleAddC, <GMat(GMat, GScalar)>, "org.opencv.test.compound_double_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundDoubleAddCImpl, GCompoundDoubleAddC)
    {
        static GMat expand(cv::GMat in, cv::GScalar s)
        {
            return cv::gapi::addC(cv::gapi::addC(in, s), s);
        }
    };

    G_TYPED_KERNEL(GCompoundAddC, <GMat(GMat, GScalar)>, "org.opencv.test.compound_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddCImpl, GCompoundAddC)
    {
        static GMat expand(cv::GMat in, cv::GScalar s)
        {
            return cv::gapi::addC(in, s);
        }
    };

    using GMat3 = std::tuple<GMat,GMat,GMat>;
    using GMat2 = std::tuple<GMat,GMat>;

    G_TYPED_KERNEL_M(GCompoundMergeWithSplit, <GMat3(GMat, GMat, GMat)>, "org.opencv.test.compound_merge_split")
    {
        static std::tuple<GMatDesc,GMatDesc,GMatDesc> outMeta(GMatDesc a, GMatDesc b, GMatDesc c)
        {
            return std::make_tuple(a, b, c);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundMergeWithSplitImpl, GCompoundMergeWithSplit)
    {
        static GMat3 expand(cv::GMat a, cv::GMat b, cv::GMat c)
        {
            return cv::gapi::split3(cv::gapi::merge3(a, b, c));
        }
    };

    G_TYPED_KERNEL(GCompoundAddWithAddC, <GMat(GMat, GMat, GScalar)>, "org.opencv.test.compound_add_with_addc")
    {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GScalarDesc)
        {
            return in;
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddWithAddCImpl, GCompoundAddWithAddC)
    {
        static GMat expand(cv::GMat in1, cv::GMat in2, cv::GScalar s)
        {
            return cv::gapi::addC(cv::gapi::add(in1, in2), s);
        }
    };

    G_TYPED_KERNEL_M(GCompoundSplitWithAdd, <GMat2(GMat)>, "org.opencv.test.compound_split_with_add")
    {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in)
        {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundSplitWithAddImpl, GCompoundSplitWithAdd)
    {
        static GMat2 expand(cv::GMat in)
        {
            cv::GMat a, b, c;
            std::tie(a, b, c) = cv::gapi::split3(in);
            return std::make_tuple(cv::gapi::add(a, b), c);
        }
    };

    G_TYPED_KERNEL_M(GCompoundParallelAddC, <GMat2(GMat, GScalar)>, "org.opencv.test.compound_parallel_addc")
    {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in, GScalarDesc)
        {
            return std::make_tuple(in, in);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundParallelAddCImpl, GCompoundParallelAddC)
    {
        static GMat2 expand(cv::GMat in, cv::GScalar s)
        {
            return std::make_tuple(cv::gapi::addC(in, s), cv::gapi::addC(in, s));
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddImpl, cv::gapi::core::GAdd)
    {
        static GMat expand(cv::GMat in1, cv::GMat in2, int)
        {
            return cv::gapi::sub(cv::gapi::sub(in1, in2), in2);
        }
    };

    G_TYPED_KERNEL(GCompoundAddWithAddCWithDoubleAddC, <GMat(GMat, GMat, GScalar)>, "org.opencv.test.compound_add_with_addC_with_double_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GScalarDesc)
        {
            return in;
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddWithAddCWithDoubleAddCImpl, GCompoundAddWithAddCWithDoubleAddC)
    {
        static GMat expand(cv::GMat in1, cv::GMat in2, cv::GScalar s)
        {
            return GCompoundDoubleAddC::on(GCompoundAddWithAddC::on(in1, in2, s), s);
        }
    };

    using GDoubleArray = cv::GArray<double>;
    G_TYPED_KERNEL(GNegateArray, <GDoubleArray(GDoubleArray)>, "org.opencv.test.negate_array")
    {
        static GArrayDesc outMeta(const GArrayDesc&) { return empty_array_desc(); }
    };

    GAPI_OCV_KERNEL(GNegateArrayImpl, GNegateArray)
    {
        static void run(const std::vector<double>& in, std::vector<double>& out)
        {
            ade::util::transform(in, std::back_inserter(out), std::negate<double>());
        }
    };

    G_TYPED_KERNEL(GMaxInArray, <GScalar(GDoubleArray)>, "org.opencv.test.max_in_array")
    {
        static GScalarDesc outMeta(const GArrayDesc&) { return empty_scalar_desc(); }
    };

    GAPI_OCV_KERNEL(GMaxInArrayImpl, GMaxInArray)
    {
        static void run(const std::vector<double>& in, cv::Scalar& out)
        {
            out = *std::max_element(in.begin(), in.end());
        }
    };

    G_TYPED_KERNEL(GCompoundMaxInArray, <GScalar(GDoubleArray)>, "org.opencv.test.compound_max_in_array")
    {
        static GScalarDesc outMeta(const GArrayDesc&) { return empty_scalar_desc(); }
    };

    GAPI_COMPOUND_KERNEL(GCompoundMaxInArrayImpl, GCompoundMaxInArray)
    {
        static GScalar expand(GDoubleArray in)
        {
            return GMaxInArray::on(in);
        }
    };

    G_TYPED_KERNEL(GCompoundNegateArray, <GDoubleArray(GDoubleArray)>, "org.opencv.test.compound_negate_array")
    {
        static GArrayDesc outMeta(const GArrayDesc&) { return empty_array_desc(); }
    };

    GAPI_COMPOUND_KERNEL(GCompoundNegateArrayImpl, GCompoundNegateArray)
    {
        static GDoubleArray expand(GDoubleArray in)
        {
            return GNegateArray::on(in);
        }
    };

    G_TYPED_KERNEL(SetDiagKernel, <GMat(GMat, GDoubleArray)>, "org.opencv.test.empty_kernel")
    {
        static GMatDesc outMeta(GMatDesc in, GArrayDesc) { return in; }
    };

    void setDiag(cv::Mat& in, const std::vector<double>& diag)
    {
        GAPI_Assert(in.rows == static_cast<int>(diag.size()));
        GAPI_Assert(in.cols == static_cast<int>(diag.size()));
        for (int i = 0; i < in.rows; ++i)
        {
            in.at<uchar>(i, i) = static_cast<uchar>(diag[i]);
        }
    }

    GAPI_OCV_KERNEL(SetDiagKernelImpl, SetDiagKernel)
    {
        static void run(const cv::Mat& in, const std::vector<double>& v, cv::Mat& out)
        {
            in.copyTo(out);
            setDiag(out, v);
        }
    };

    G_TYPED_KERNEL(GCompoundGMatGArrayGMat, <GMat(GMat, GDoubleArray, GMat)>, "org.opencv.test.compound_gmat_garray_gmat")
    {
        static GMatDesc outMeta(GMatDesc in, GArrayDesc, GMatDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundGMatGArrayGMatImpl, GCompoundGMatGArrayGMat)
    {
        static GMat expand(GMat a, GDoubleArray b, GMat c)
        {
            return SetDiagKernel::on(cv::gapi::add(a, c), b);
        }
    };

} // namespace

// FIXME avoid cv::combine that use custom and default kernels together
TEST(GCompoundKernel, ReplaceDefaultKernel)
{
    cv::GMat in1, in2;
    auto out = cv::gapi::add(in1, in2);
    const auto custom_pkg = cv::gapi::kernels<GCompoundAddImpl>();
    const auto full_pkg   = cv::gapi::combine(cv::gapi::core::cpu::kernels(), custom_pkg);
    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out));
    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat(3, 3, CV_8UC1);

    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 - in_mat2 - in_mat2;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, DoubleAddC)
{
    cv::GMat in1, in2;
    cv::GScalar s;
    auto add_res   = cv::gapi::add(in1, in2);
    auto super     = GCompoundDoubleAddC::on(add_res, s);
    auto out       = cv::gapi::addC(super, s);

    const auto custom_pkg = cv::gapi::kernels<GCompoundDoubleAddCImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2, s), cv::GOut(out));

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat1, in_mat2, scalar), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, AddC)
{
    cv::GMat in1, in2;
    cv::GScalar s;
    auto add_res   = cv::gapi::add(in1, in2);
    auto super     = GCompoundAddC::on(add_res, s);
    auto out       = cv::gapi::addC(super, s);

    const auto custom_pkg = cv::gapi::kernels<GCompoundAddCImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2, s), cv::GOut(out));

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat1, in_mat2, scalar), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, MergeWithSplit)
{
    cv::GMat in, a1, b1, c1,
        a2, b2, c2;

    std::tie(a1, b1, c1) = cv::gapi::split3(in);
    std::tie(a2, b2, c2) = GCompoundMergeWithSplit::on(a1, b1, c1);
    auto out = cv::gapi::merge3(a2, b2, c2);

    const auto custom_pkg = cv::gapi::kernels<GCompoundMergeWithSplitImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC3), out_mat, ref_mat;
    comp.apply(cv::gin(in_mat), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, AddWithAddC)
{
    cv::GMat in1, in2;
    cv::GScalar s;
    auto out = GCompoundAddWithAddC::on(in1, in2, s);

    const auto custom_pkg = cv::gapi::kernels<GCompoundAddWithAddCImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2, s), cv::GOut(out));

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat1, in_mat2, scalar), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, SplitWithAdd)
{
    cv::GMat in, out1, out2;
    std::tie(out1, out2) = GCompoundSplitWithAdd::on(in);

    const auto custom_pkg = cv::gapi::kernels<GCompoundSplitWithAddImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in), cv::GOut(out1, out2));

    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC3),
        out_mat1(3, 3, CV_8UC1),
        out_mat2(3, 3, CV_8UC1),
        ref_mat1(3, 3, CV_8UC1),
        ref_mat2(3, 3, CV_8UC1);

    comp.apply(cv::gin(in_mat), cv::gout(out_mat1, out_mat2), cv::compile_args(full_pkg));

    std::vector<cv::Mat> channels(3);
    cv::split(in_mat, channels);

    ref_mat1 = channels[0] + channels[1];
    ref_mat2 = channels[2];

    EXPECT_EQ(0, cv::countNonZero(out_mat1 != ref_mat1));
    EXPECT_EQ(0, cv::countNonZero(out_mat2 != ref_mat2));
}

TEST(GCompoundKernel, ParallelAddC)
{
    cv::GMat in1, out1, out2;
    cv::GScalar in2;
    std::tie(out1, out2) = GCompoundParallelAddC::on(in1, in2);

    const auto custom_pkg = cv::gapi::kernels<GCompoundParallelAddCImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out1, out2));

    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC1),
        out_mat1(3, 3, CV_8UC1),
        out_mat2(3, 3, CV_8UC1),
        ref_mat1(3, 3, CV_8UC1),
        ref_mat2(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat, scalar), cv::gout(out_mat1, out_mat2), cv::compile_args(full_pkg));

    ref_mat1 = in_mat + scalar;
    ref_mat2 = in_mat + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat1 != ref_mat1));
    EXPECT_EQ(0, cv::countNonZero(out_mat2 != ref_mat2));
}

TEST(GCompoundKernel, GCompundKernelAndDefaultUseOneData)
{
    cv::GMat in1, in2;
    cv::GScalar s;
    auto out = cv::gapi::add(GCompoundAddWithAddC::on(in1, in2, s), cv::gapi::addC(in2, s));

    const auto custom_pkg = cv::gapi::kernels<GCompoundAddWithAddCImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2, s), cv::GOut(out));

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat1, in_mat2, scalar), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + in_mat2 + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, CompoundExpandedToCompound)
{
    cv::GMat in1, in2;
    cv::GScalar s;
    auto out = GCompoundAddWithAddCWithDoubleAddC::on(in1, in2, s);

    const auto custom_pkg = cv::gapi::kernels<GCompoundAddWithAddCWithDoubleAddCImpl,
                                              GCompoundAddWithAddCImpl,
                                              GCompoundDoubleAddCImpl>();

    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in1, in2, s), cv::GOut(out));

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat(3, 3, CV_8UC1);

    cv::Scalar scalar = 2;

    comp.apply(cv::gin(in_mat1, in_mat2, scalar), cv::gout(out_mat), cv::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar + scalar;

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(GCompoundKernel, MaxInArray)
{
    GDoubleArray in;
    auto out = GCompoundMaxInArray::on(in);
    const auto custom_pkg = cv::gapi::kernels<GCompoundMaxInArrayImpl, GMaxInArrayImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    std::vector<double> v = { 1, 5, -2, 3, 10, 2};
    cv::Scalar out_scl;
    cv::Scalar ref_scl(*std::max_element(v.begin(), v.end()));

    comp.apply(cv::gin(v), cv::gout(out_scl), cv::compile_args(full_pkg));

    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GCompoundKernel, NegateArray)
{
    GDoubleArray in;
    GDoubleArray out = GCompoundNegateArray::on(in);
    const auto custom_pkg = cv::gapi::kernels<GCompoundNegateArrayImpl, GNegateArrayImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    std::vector<double> in_v = {1, 5, -2, -10, 3};
    std::vector<double> out_v;
    std::vector<double> ref_v;
    ade::util::transform(in_v, std::back_inserter(ref_v), std::negate<double>());

    comp.apply(cv::gin(in_v), cv::gout(out_v), cv::compile_args(full_pkg));

    EXPECT_EQ(out_v, ref_v);
}

TEST(GCompoundKernel, RightGArrayHandle)
{
    cv::GMat in[2];
    GDoubleArray a;
    cv::GMat out = GCompoundGMatGArrayGMat::on(in[0], a, in[1]);
    const auto custom_pkg = cv::gapi::kernels<GCompoundGMatGArrayGMatImpl, SetDiagKernelImpl>();
    const auto full_pkg   = cv::gapi::combine(custom_pkg, cv::gapi::core::cpu::kernels());
    cv::GComputation comp(cv::GIn(in[0], a, in[1]), cv::GOut(out));
    std::vector<double> in_v(3, 1.0);
    cv::Mat in_mat1 = cv::Mat::eye(cv::Size(3, 3), CV_8UC1),
            in_mat2 = cv::Mat::eye(cv::Size(3, 3), CV_8UC1),
            out_mat;
    cv::Mat ref_mat= in_mat1 + in_mat2;
    setDiag(ref_mat, in_v);

    comp.apply(cv::gin(in_mat1, in_v, in_mat2), cv::gout(out_mat), cv::compile_args(full_pkg));

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));

}
} // opencv_test
