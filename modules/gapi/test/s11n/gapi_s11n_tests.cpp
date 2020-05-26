#include "../test_precomp.hpp"

#include "serialization.hpp"

namespace opencv_test {

struct S11N_Basic: public ::testing::Test {
    template<typename T> void put(T &&t) {
        cv::gimpl::s11n::SerializationStream os;
        os << t;

        // FIXME: This stream API needs a fix-up
        m_buffer.resize(os.getSize());
        std::copy_n(os.getData(), os.getSize(), m_buffer.begin());
    }

    template<typename T> T get() {
        // FIXME: This stream API needs a fix-up
        cv::gimpl::s11n::DeSerializationStream is(m_buffer.data(), m_buffer.size());

        T t{};
        is >> t;
        return t;
    }

private:
    std::vector<char> m_buffer;
};

TEST_F(S11N_Basic, Test_uint32) {
    uint32_t x = 42;
    put(x);

    EXPECT_EQ(x, get<uint32_t>());
}

TEST_F(S11N_Basic, Test_int) {
    int x = 42;
    put(x);

    EXPECT_EQ(x, get<int>());
}

TEST_F(S11N_Basic, Test_fp32) {
    static_assert(sizeof(float) == 4, "Expect float to have 32 bits");
    float x = 3.14f;
    put(x);

    EXPECT_EQ(x, get<float>());
}

TEST_F(S11N_Basic, Test_fp64) {
    static_assert(sizeof(double) == 8, "Expect double to have 64 bits");
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


} // namespace opencv_test
