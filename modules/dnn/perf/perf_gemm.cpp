// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <numeric>

namespace opencv_test {

struct GemmParam_t {
    std::vector<int> a_shape;
    std::vector<int> b_shape;
    std::vector<int> c_shape;
    bool trans_a;
    bool trans_b;

    GemmParam_t(std::vector<int> a_shape_, std::vector<int> b_shape_, std::vector<int> c_shape_ = {}, bool trans_a_ = false, bool trans_b_ = false)
        : a_shape(a_shape_), b_shape(b_shape_), c_shape(c_shape_), trans_a(trans_a_), trans_b(trans_b_) {}
};

// TODO: Dsiable most of the test cases except vision transformers to save time
static const GemmParam_t test_gemm_configs[] = {
    // vision transformers cases
    { {  768,  768 }, {  768,  768 }, {  768 } },
    { { 1024, 1024 }, { 1024, 1024 }, { 1024 } },
    { {   50,  768 }, {  768, 2304 } },
    { {  197,  768 }, {  768, 2304 } },
    { {   50, 1024 }, { 1024, 3072 } },
    { {  197, 1024 }, { 1024, 3072 } },

// these cases are commented to save testing time
/*
    // square mat
    { {   64,   64 }, {   64,   64 } },
    { {  128,  128 }, {  128,  128 } },
    { {  256,  256 }, {  256,  256 } },
    { {  512,  512 }, {  512,  512 } },
    { { 1024, 1024 }, { 1024, 1024 } },
    { { 4096, 4096 }, { 4096, 4096 } },

    // retangular mat
    { {  256,  256 }, {  256, 1024 } },
    { {  256, 1024 }, { 1024,  256 } },
    { {  256, 1024 }, { 1024, 1024 } },
    { { 1024, 1024 }, { 1024,  256 } },
    { { 1024,  256 }, {  256, 1024 } },
    { { 1024,  256 }, {  256,  256 } },

    // with C
    { {  256,  256 }, {  256,  256 }, {  256 } },
    { {  256,  256 }, {  256, 1024 }, { 1024 } },
    { {  256, 1024 }, { 1024,  256 }, {  256 } },
    { {  256, 1024 }, { 1024, 1024 }, { 1024 } },
    { { 1024, 1024 }, { 1024,  256 }, {  256 } },
    { { 1024,  256 }, {  256, 1024 }, { 1024 } },
    { { 1024,  256 }, {  256,  256 }, {  256 } },

    // with C and trans_b
    { {  256,  256 }, {  256,  256 }, {  256 } , false, true},
    { {  256, 1024 }, {  256, 1024 }, {  256 } , false, true},
    { {  256, 1024 }, { 1024, 1024 }, { 1024 } , false, true},
    { { 1024, 1024 }, { 1024, 1024 }, { 1024 } , false, true},
    { { 1024,  256 }, { 1024,  256 }, { 1024 } , false, true},
    { { 1024,  256 }, {  256,  256 }, {  256 } , false, true},

    // with C and trans_b and trans_a
    { {  256,  256 }, {  256,  256 }, {  256 } , true, true},
    { { 1024,  256 }, {  256, 1024 }, {  256 } , true, true},
    { {  256, 1024 }, { 1024,  256 }, { 1024 } , true, true},
    { { 1024, 1024 }, { 1024, 1024 }, { 1024 } , true, true},
*/
};

static const GemmParam_t test_matmul_configs[] = {
    // vision transformer cases
    { {12, 197, 197}, {12, 197, 64} },
    { {12, 197, 64 }, {12, 64, 197} },
    { {12, 50, 64}, {12, 64, 50} },
    { {12, 50, 50}, {12, 50, 64} },
    { {16, 197, 197}, {16, 197, 64} },
    { {16, 197, 64 }, {16, 64, 197} },
    { {16, 50, 64}, {16, 64, 50} },
    { {16, 50, 50}, {16, 50, 64} },
};

struct GemmParamId
{
    enum {
        GEMM_0 = 0,
        GEMM_LAST = sizeof(test_gemm_configs) / sizeof(test_gemm_configs[0])
    };
    int val_;
    GemmParamId(int val = 0) : val_(val) {}
    operator int() const { return val_; }
    static ::testing::internal::ParamGenerator<GemmParamId> all()
    {
        enum { NUM = (int)GEMM_LAST };
        GemmParamId v_[NUM]; for (int i = 0; i < NUM; ++i) { v_[i] = GemmParamId(i); } // reduce generated code size
        return ::testing::ValuesIn(v_, v_ + NUM);
    }
};

struct MatMulParamId {
    enum {
        MATMUL_0 = 0,
        MATMUL_LAST = sizeof(test_matmul_configs) / sizeof(test_matmul_configs[0])
    };
    int val_;
    MatMulParamId(int val = 0) : val_(val) {}
    operator int() const { return val_; }
    static ::testing::internal::ParamGenerator<MatMulParamId> all() {
        enum { NUM = (int)MATMUL_LAST };
        MatMulParamId v_[NUM]; for (int i = 0; i < NUM; i++) { v_[i] = MatMulParamId(i); }
        return ::testing::ValuesIn(v_, v_ + NUM);
    }
};

static inline void PrintTo(const GemmParamId& v, std::ostream* os)
{
    CV_Assert((int)v >= 0); CV_Assert((int)v < GemmParamId::GEMM_LAST);
    const GemmParam_t& p = test_gemm_configs[(int)v];

    auto print_shape = [os](const std::vector<int>& shape, const std::string tag) {
        if (shape.empty()) {
            return ;
        }

        *os << tag << "=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == shape.size() - 1) {
                *os << shape[i] << "]";
                break;
            }
            *os << shape[i] << ", ";
        }
    };

    print_shape(p.a_shape, "A");
    print_shape(p.b_shape, ", B");
    print_shape(p.c_shape, ", C");
    *os << ", trans_a=" << p.trans_a << ", trans_b=" << p.trans_b;
}

typedef tuple<GemmParamId, tuple<Backend, Target> > GemmTestParam_t;
typedef TestBaseWithParam<GemmTestParam_t> Gemm;

PERF_TEST_P_(Gemm, gemm)
{
    int test_id = (int)get<0>(GetParam());
    ASSERT_GE(test_id, 0); ASSERT_LT(test_id, GemmParamId::GEMM_LAST);
    const GemmParam_t& params = test_gemm_configs[test_id];
    auto a_shape = params.a_shape;
    auto b_shape = params.b_shape;
    auto c_shape = params.c_shape;
    auto trans_a = params.trans_a;
    auto trans_b = params.trans_b;
    float alpha = 1.f;
    float beta = 1.f;

    Backend backend_id = get<0>(get<1>(GetParam()));
    Target target_id = get<1>(get<1>(GetParam()));

    bool have_bias = c_shape.empty() ? false : true;

    Mat A(static_cast<int>(a_shape.size()), a_shape.data(), CV_32F);
    randu(A, -1.0f, 1.0f);
    Mat B(static_cast<int>(b_shape.size()), b_shape.data(), CV_32F);
    randu(B, -1.0f, 1.0f);

    LayerParams lp;
    lp.type = "Gemm";
    lp.name = "testLayer";
    lp.set("transA", trans_a);
    lp.set("transB", trans_b);
    lp.set("alpha", alpha);
    lp.set("beta", beta);
    lp.set("real_ndims_C", static_cast<int>(c_shape.size()));

    lp.set("constB", true);
    lp.blobs.push_back(B);
    if (have_bias) {
        Mat C(static_cast<int>(c_shape.size()), c_shape.data(), CV_32F);
        randu(C, -1.0f, 1.0f);
        lp.set("have_bias", true);
        lp.set("constC", true);
        lp.blobs.push_back(C);
    }

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(backend_id);
    net.setPreferableTarget(target_id);

    // warmup
    {
        net.setInput(A);
        Mat out = net.forward();
    }

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Gemm, innerproduct)
{
    int test_id = (int)get<0>(GetParam());
    ASSERT_GE(test_id, 0); ASSERT_LT(test_id, GemmParamId::GEMM_LAST);
    const GemmParam_t& params = test_gemm_configs[test_id];
    auto a_shape = params.a_shape;
    auto b_shape = params.b_shape;
    auto c_shape = params.c_shape;
    auto trans_a = params.trans_a;
    auto trans_b = params.trans_b;

    Backend backend_id = get<0>(get<1>(GetParam()));
    Target target_id = get<1>(get<1>(GetParam()));

    bool have_bias = c_shape.empty() ? false : true;

    Mat A(static_cast<int>(a_shape.size()), a_shape.data(), CV_32F);
    randu(A, -1.0f, 1.0f);
    Mat B(static_cast<int>(b_shape.size()), b_shape.data(), CV_32F);
    randu(B, -1.0f, 1.0f);

    LayerParams lp;
    lp.type = "InnerProduct";
    lp.name = "testLayer";
    if (trans_a) {
        cv::transpose(A, A);
    }
    if (!trans_b) {
        cv::transpose(B, B);
    }
    lp.blobs.push_back(B);
    lp.set("num_output", B.size[0]);
    if (have_bias) {
        Mat C(static_cast<int>(c_shape.size()), c_shape.data(), CV_32F);
        randu(C, -1.0f, 1.0f);
        lp.blobs.push_back(C);
        lp.set("bias_term", true);
    } else {
        lp.set("bias_term", false);
    }

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(backend_id);
    net.setPreferableTarget(target_id);

    // warmup
    {
        std::vector<std::string> input_names(1);
        input_names[0] = "A";
        net.setInputsNames(input_names);
        net.setInput(A, input_names[0]);
        Mat out = net.forward();
    }

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

static inline void PrintTo(const MatMulParamId& v, std::ostream* os)
{
    CV_Assert((int)v >= 0); CV_Assert((int)v < MatMulParamId::MATMUL_LAST);
    const GemmParam_t& p = test_matmul_configs[(int)v];

    auto print_shape = [os](const std::vector<int>& shape, const std::string tag) {
        if (shape.empty()) {
            return ;
        }

        *os << tag << "=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == shape.size() - 1) {
                *os << shape[i] << "]";
                break;
            }
            *os << shape[i] << ", ";
        }
    };

    print_shape(p.a_shape, "A");
    print_shape(p.b_shape, ", B");
    print_shape(p.c_shape, ", C");
    *os << ", trans_a=" << p.trans_a << ", trans_b=" << p.trans_b;
}

using MatMulTestParam_t = tuple<MatMulParamId, tuple<Backend, Target>>;
using MatMul = TestBaseWithParam<MatMulTestParam_t>;

PERF_TEST_P_(MatMul, matmul)
{
    int test_id = (int)get<0>(GetParam());
    ASSERT_GE(test_id, 0); ASSERT_LT(test_id, MatMulParamId::MATMUL_LAST);
    const GemmParam_t& params = test_matmul_configs[test_id];
    auto a_shape = params.a_shape;
    auto b_shape = params.b_shape;
    auto trans_a = params.trans_a;
    auto trans_b = params.trans_b;
    float alpha = 1.f;
    float beta = 1.f;

    Backend backend_id = get<0>(get<1>(GetParam()));
    Target target_id = get<1>(get<1>(GetParam()));

    Mat A(a_shape, CV_32F);
    randu(A, -1.0f, 1.0f);
    Mat B(b_shape, CV_32F);
    randu(B, -1.0f, 1.0f);

    LayerParams lp;
    lp.type = "MatMul";
    lp.name = "testLayer";
    lp.set("transA", trans_a);
    lp.set("transB", trans_b);
    lp.set("alpha", alpha);
    lp.set("beta", beta);
    lp.blobs.push_back(B);

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(backend_id);
    net.setPreferableTarget(target_id);

    // warmup
    {
        std::vector<std::string> input_names{"A"};
        net.setInputsNames(input_names);
        net.setInput(A, input_names[0]);
        Mat out = net.forward();
    }

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(MatMul, innerproduct)
{
    int test_id = (int)get<0>(GetParam());
    ASSERT_GE(test_id, 0); ASSERT_LT(test_id, MatMulParamId::MATMUL_LAST);
    const GemmParam_t& params = test_matmul_configs[test_id];
    auto a_shape = params.a_shape;
    auto b_shape = params.b_shape;

    Backend backend_id = get<0>(get<1>(GetParam()));
    Target target_id = get<1>(get<1>(GetParam()));

    Mat A(a_shape, CV_32F);
    randu(A, -1.0f, 1.0f);
    Mat B(b_shape, CV_32F);
    randu(B, -1.0f, 1.0f);

    LayerParams lp;
    lp.type = "InnerProduct";
    lp.name = "testLayer";
    lp.set("axis", (int)(a_shape.size() - 1));
    lp.set("bias_term", false);

    // pre-transpose
    std::vector<int> order(b_shape.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order.back(), order[b_shape.size() - 2]);
    Mat B_transposed;
    transposeND(B, order, B_transposed);
    lp.blobs.push_back(B_transposed);
    lp.set("num_output", int(B_transposed.total(0, b_shape.size() - 1)));
    lp.set("is_matmul", true);

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(backend_id);
    net.setPreferableTarget(target_id);

    // warmup
    {
        std::vector<std::string> input_names{"A"};
        net.setInputsNames(input_names);
        net.setInput(A, input_names[0]);
        Mat out = net.forward();
    }

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Gemm, Combine(
    GemmParamId::all(),
    dnnBackendsAndTargets(false, false)  // defined in ../test/test_common.hpp
));

INSTANTIATE_TEST_CASE_P(/**/, MatMul, Combine(
    MatMulParamId::all(),
    dnnBackendsAndTargets(false, false)  // defined in ../test/test_common.hpp
));

} // namespace
