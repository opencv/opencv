#include "../test_precomp.hpp"

#include "backends/common/serialization.hpp"

namespace opencv_test {

struct S11N_Basic: public ::testing::Test {
    template<typename T> void put(T &&t) {
        cv::gimpl::s11n::ByteMemoryOutStream os;
        os << t;
        m_buffer = os.data();
    }

    template<typename T> T get() {
        // FIXME: This stream API needs a fix-up
        cv::gimpl::s11n::ByteMemoryInStream is(m_buffer);
        T t{};
        is >> t;
        return t;
    }

private:
    std::vector<char> m_buffer;
};

TEST_F(S11N_Basic, Test_int_pos) {
    int x = 42;
    put(x);
    EXPECT_EQ(x, get<int>());
}

TEST_F(S11N_Basic, Test_int_neg) {
    int x = -42;
    put(x);
    EXPECT_EQ(x, get<int>());
}

TEST_F(S11N_Basic, Test_fp32) {
    float x = 3.14f;
    put(x);
    EXPECT_EQ(x, get<float>());
}

TEST_F(S11N_Basic, Test_fp64) {
    double x = 3.14;
    put(x);
    EXPECT_EQ(x, get<double>());
}

TEST_F(S11N_Basic, Test_vector_int) {
    std::vector<int> v = {1,2,3};
    put(v);
    EXPECT_EQ(v, get<std::vector<int> >());
}

TEST_F(S11N_Basic, Test_vector_cvSize) {
    std::vector<cv::Size> v = {
        cv::Size(640, 480),
        cv::Size(1280, 1024),
    };
    put(v);
    EXPECT_EQ(v, get<std::vector<cv::Size> >());
}

TEST_F(S11N_Basic, Test_vector_string) {
    std::vector<std::string> v = {
        "hello",
        "world",
        "ok!"
    };
    put(v);
    EXPECT_EQ(v, get<std::vector<std::string> >());
}

TEST_F(S11N_Basic, Test_vector_empty) {
    std::vector<char> v;
    put(v);
    EXPECT_EQ(v, get<std::vector<char> >());
}

TEST_F(S11N_Basic, Test_variant) {
    using S = std::string;
    using V = cv::util::variant<int,S>;
    V v1{42}, v2{S{"hey"}};

    put(v1);
    EXPECT_EQ(v1, get<V>());

    put(v2);
    EXPECT_EQ(v2, get<V>());
}

TEST_F(S11N_Basic, Test_GArg_int) {
    const int x = 42;
    cv::GArg gs(x);
    put(gs);

    cv::GArg gd = get<cv::GArg>();
    EXPECT_EQ(cv::detail::ArgKind::OPAQUE_VAL, gd.kind);
    EXPECT_EQ(cv::detail::OpaqueKind::CV_INT, gd.opaque_kind);
    EXPECT_EQ(x, gs.get<int>());
}

TEST_F(S11N_Basic, Test_GArg_Point) {
    const cv::Point pt{1,2};
    cv::GArg gs(pt);
    put(gs);

    cv::GArg gd = get<cv::GArg>();
    EXPECT_EQ(cv::detail::ArgKind::OPAQUE_VAL, gd.kind);
    EXPECT_EQ(cv::detail::OpaqueKind::CV_POINT, gd.opaque_kind);
    EXPECT_EQ(pt, gs.get<cv::Point>());
}

TEST_F(S11N_Basic, Test_Mat_full) {
    auto mat = cv::Mat::eye(cv::Size(64,64), CV_8UC3);
    put(mat);
    EXPECT_EQ(0, cv::norm(mat, get<cv::Mat>(), cv::NORM_INF));
}

TEST_F(S11N_Basic, Test_Mat_view) {
    auto mat  = cv::Mat::eye(cv::Size(320,240), CV_8UC3);
    auto view = mat(cv::Rect(10,15,123,70));
    put(view);
    EXPECT_EQ(0, cv::norm(view, get<cv::Mat>(), cv::NORM_INF));
}

TEST_F(S11N_Basic, Test_MatDesc) {
    cv::GMatDesc v = { CV_8U, 1, {320,240} };
    put(v);
    EXPECT_EQ(v, get<cv::GMatDesc>());
}

TEST_F(S11N_Basic, Test_MetaArg_MatDesc) {
    cv::GMatDesc desc = { CV_8U, 1,{ 320,240 } };
    auto v = cv::GMetaArg{ desc };
    put(v);
    cv::GMetaArg out_v = get<cv::GMetaArg>();
    cv::GMatDesc out_desc = cv::util::get<cv::GMatDesc>(out_v);
    EXPECT_EQ(desc, out_desc);
}

TEST_F(S11N_Basic, Test_MetaArg_Monostate) {
    GMetaArg v;
    put(v);
    cv::GMetaArg out_v = get<cv::GMetaArg>();
    if (!util::holds_alternative<util::monostate>(out_v))
    {
        GTEST_FAIL();
    }
}

TEST_F(S11N_Basic, Test_RunArg_Mat) {
    cv::Mat mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC3);
    auto v = cv::GRunArg{ mat };
    put(v);
    cv::GRunArg out_v = get<cv::GRunArg>();
    cv::Mat out_mat = cv::util::get<cv::Mat>(out_v);
    EXPECT_EQ(0, cv::norm(mat, out_mat, cv::NORM_INF));
}

TEST_F(S11N_Basic, Test_RunArg_Scalar) {
    cv::Scalar scalar = cv::Scalar(128, 33, 53);
    auto v = cv::GRunArg{ scalar };
    put(v);
    cv::GRunArg out_v = get<cv::GRunArg>();
    cv::Scalar out_scalar = cv::util::get<cv::Scalar>(out_v);
    EXPECT_EQ(scalar, out_scalar);
}

} // namespace opencv_test
