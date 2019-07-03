// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"

#include "gapi_fluid_test_kernels.hpp"

namespace opencv_test
{

namespace {
    cv::Mat randomMat(cv::Size img_sz, int type = CV_8UC1, cv::Scalar mean   = cv::Scalar(127.0f), cv::Scalar stddev = cv::Scalar(40.f)){
        cv::Mat mat(img_sz, type);
        cv::randn(mat, mean, stddev);
        return mat;
    }

    cv::GFluidParallelOutputRois asGFluidParallelOutputRois(const std::vector<cv::Rect>& rois){
        cv::GFluidParallelOutputRois parallel_rois;
        for (auto const& roi : rois) {
            parallel_rois.parallel_rois.emplace_back(GFluidOutputRois{{to_own(roi)}});
        }
        return parallel_rois;
    }

    void adjust_empty_roi(cv::Rect& roi, cv::Size size){
        if (roi.empty()) roi = cv::Rect{{0,0}, size};
    }
}
using namespace cv::gapi_test_kernels;

//As GTest can not simultaneously parameterize test with both types and values - lets use type-erasure and virtual interfaces
//to use different computation pipelines
struct ComputationPair {
    virtual void run_with_gapi(const cv::Mat& in_mat, cv::GFluidParallelOutputRois const& parallel_rois, cv::Mat& out_mat) = 0;
    virtual void run_with_ocv (const cv::Mat& in_mat, const std::vector<cv::Rect>& rois,                 cv::Mat& out_mat) = 0;

    virtual std::string name() const { return {}; }

    virtual ~ComputationPair ()  = default;

    friend std::ostream& operator<<(std::ostream& o, ComputationPair const* cp){
        std::string custom_name = cp->name();
        return o << (custom_name.empty() ? typeid(cp).name() : custom_name );
    }
};

struct Blur3x3CP  : ComputationPair{
    static constexpr int borderType = BORDER_REPLICATE;
    static constexpr int kernelSize = 3;

    std::string name() const override { return "Blur3x3"; }
    void run_with_gapi(const cv::Mat& in_mat, cv::GFluidParallelOutputRois const& parallel_rois, cv::Mat& out_mat_gapi) override {
        cv::GMat in;
        cv::GMat out = TBlur3x3::on(in, borderType, {});
        cv::GComputation c(cv::GIn(in), cv::GOut(out));

        // Run G-API
        auto cc = c.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage, parallel_rois));
        cc(cv::gin(in_mat), cv::gout(out_mat_gapi));
    }

    void run_with_ocv(const cv::Mat& in_mat, const std::vector<cv::Rect>& rois, cv::Mat& out_mat_ocv) override {
        cv::Point anchor = {-1, -1};
        // Check with OpenCV
        for (auto roi : rois) {
            adjust_empty_roi(roi, in_mat.size());
            cv::blur(in_mat(roi), out_mat_ocv(roi), {kernelSize, kernelSize}, anchor, borderType);
        }
    }
};

struct AddCCP : ComputationPair{
    std::string name() const override { return "AddC"; }
    void run_with_gapi(const cv::Mat& in_mat, cv::GFluidParallelOutputRois const& parallel_rois, cv::Mat& out_mat_gapi) override {
        cv::GMat in;
        cv::GMat out = TAddCSimple::on(in, 1);
        cv::GComputation c(cv::GIn(in), cv::GOut(out));

        // Run G-API
        auto cc = c.compile(cv::descr_of(in_mat), cv::compile_args(fluidTestPackage, parallel_rois));
        cc(cv::gin(in_mat), cv::gout(out_mat_gapi));
    }

    void run_with_ocv(const cv::Mat& in_mat, const std::vector<cv::Rect>& rois, cv::Mat& out_mat_ocv) override {
        // Check with OpenCV
        for (auto roi : rois) {
            adjust_empty_roi(roi, in_mat.size());
            out_mat_ocv(roi) = in_mat(roi) + 1u;
        }
    }
};

template<BorderTypes _borderType>
struct SequenceOfBlursCP : ComputationPair{
    BorderTypes borderType = _borderType;

    std::string name() const override { return "SequenceOfBlurs, border type: " + std::to_string(static_cast<int>(borderType)); }
    void run_with_gapi(const cv::Mat& in_mat, cv::GFluidParallelOutputRois const& parallel_rois, cv::Mat& out_mat) override {
        cv::Scalar borderValue(0);

        GMat in;
        auto mid = TBlur3x3::on(in,  borderType, borderValue);
        auto out = TBlur5x5::on(mid, borderType, borderValue);

        GComputation c(GIn(in), GOut(out));
        auto cc = c.compile(descr_of(in_mat), cv::compile_args(fluidTestPackage, parallel_rois));
        cc(cv::gin(in_mat), cv::gout(out_mat));
    }
    void run_with_ocv (const cv::Mat& in_mat, const std::vector<cv::Rect>& rois,                 cv::Mat& out_mat) override {
        cv::Mat mid_mat_ocv = Mat::zeros(in_mat.size(), in_mat.type());
        cv::Point anchor = {-1, -1};

        for (auto roi : rois) {
            adjust_empty_roi(roi, in_mat.size());
            cv::blur(in_mat, mid_mat_ocv, {3,3}, anchor, borderType);
            cv::blur(mid_mat_ocv(roi), out_mat(roi), {5,5}, anchor, borderType);
        }
    }
};

struct TiledComputation : public TestWithParam <std::tuple<ComputationPair*, cv::Size, std::vector<cv::Rect>>> {};
TEST_P(TiledComputation, Test)
{
    ComputationPair*        cp;
    cv::Size                img_sz;
    std::vector<cv::Rect>   rois ;
    auto                    mat_type  =  CV_8UC1;

    std::tie(cp, img_sz, rois) = GetParam();

    cv::Mat in_mat       =      randomMat(img_sz, mat_type);
    cv::Mat out_mat_gapi = cv::Mat::zeros(img_sz, mat_type);
    cv::Mat out_mat_ocv  = cv::Mat::zeros(img_sz, mat_type);

    cp->run_with_gapi(in_mat, asGFluidParallelOutputRois(rois),  out_mat_gapi);
    cp->run_with_ocv (in_mat, rois,                              out_mat_ocv);

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv))
            << "in_mat : \n"      << in_mat << std::endl
            << "diff matrix :\n " << (out_mat_gapi != out_mat_ocv) << std::endl
            << "out_mat_gapi: \n" << out_mat_gapi << std::endl
            << "out_mat_ocv:  \n" << out_mat_ocv << std::endl;
}
namespace {
    //this is ugly but other variants (like using shared_ptr) are IMHO even more ugly :)
    template<typename T, typename... Arg>
    T* addr_of_static(Arg... arg) {
        static T obj(std::forward<Arg>(arg)...);
        return &obj;
    }
}

auto single_arg_computations = [](){
    return Values(  addr_of_static<Blur3x3CP>(),
                    addr_of_static<AddCCP>(),
                    addr_of_static<SequenceOfBlursCP<BORDER_CONSTANT>>(),
                    addr_of_static<SequenceOfBlursCP<BORDER_REPLICATE>>(),
                    addr_of_static<SequenceOfBlursCP<BORDER_REFLECT_101>>()
            );

};

INSTANTIATE_TEST_CASE_P(FluidTiledSerial8x10, TiledComputation,
                        Combine(
                            single_arg_computations(),
                            Values(cv::Size(8, 10)),
                            Values(std::vector<cv::Rect>{cv::Rect{}},
                                   std::vector<cv::Rect>{cv::Rect{0,0,8,5}, cv::Rect{0,5,8,5}},
                                   std::vector<cv::Rect>{cv::Rect{0,1,8,3}, cv::Rect{0,4,8,3}},
                                   std::vector<cv::Rect>{cv::Rect{0,2,8,3}, cv::Rect{0,5,8,2}},
                                   std::vector<cv::Rect>{cv::Rect{0,3,8,4}, cv::Rect{0,9,8,1}}))
);

INSTANTIATE_TEST_CASE_P(FluidTiledSerial20x15, TiledComputation,
                        Combine(
                            single_arg_computations(),
                            Values(cv::Size(20, 15)),
                            Values(std::vector<cv::Rect>{cv::Rect{}},
                                   std::vector<cv::Rect>{cv::Rect{{0,0},cv::Size{20,7}},
                                                         cv::Rect{{0,7},cv::Size{20,8}}}))
);

INSTANTIATE_TEST_CASE_P(FluidTiledSerial320x240, TiledComputation,
                        Combine(
                            single_arg_computations(),
                            Values(cv::Size(320, 240)),
                            Values(std::vector<cv::Rect>{cv::Rect{{0,0},   cv::Size{320,120}},
                                                         cv::Rect{{0,120}, cv::Size{320,120}}},

                                   std::vector<cv::Rect>{cv::Rect{{0,0},   cv::Size{320,120}},
                                                         cv::Rect{{0,120}, cv::Size{320,120}}},

                                   std::vector<cv::Rect>{cv::Rect{{0,0},  cv::Size{320,60}},
                                                         cv::Rect{{0,60}, cv::Size{320,60}},
                                                         cv::Rect{{0,120},cv::Size{320,120}}}))
);

//FIXME: add multiple outputs tests
} // namespace opencv_test
