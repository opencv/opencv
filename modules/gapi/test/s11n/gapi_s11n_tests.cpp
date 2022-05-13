#include "../test_precomp.hpp"

#include "backends/common/serialization.hpp"
#include <opencv2/gapi/rmat.hpp>
#include <opencv2/gapi/media.hpp>
#include <../src/backends/common/gbackend.hpp> // asView

namespace {
struct EmptyCustomType { };

struct SimpleCustomType {
    bool val;
    bool operator==(const SimpleCustomType& other) const {
        return val == other.val;
    }
};

struct SimpleCustomType2 {
    int id;
    bool operator==(const SimpleCustomType2& other) const {
        return id == other.id;
    }
};

struct MyCustomType {
    int val;
    std::string name;
    std::vector<float> vec;
    std::map<int, uint64_t> mmap;
    bool operator==(const MyCustomType& other) const {
        return val == other.val && name == other.name &&
                vec == other.vec && mmap == other.mmap;
    }
};

struct MyCustomTypeNoS11N {
    char sym;
    int id;
    std::string name;

    bool operator==(const MyCustomTypeNoS11N& other) const {
        return sym == other.sym && id == other.id &&
                name == other.name;
    }
};
} // anonymous namespace

namespace cv {
namespace gapi {
namespace s11n {
namespace detail {
template<> struct S11N<EmptyCustomType> {
    static void serialize(IOStream &, const EmptyCustomType &) { }
    static EmptyCustomType deserialize(IIStream &) { return EmptyCustomType { }; }
};

template<> struct S11N<SimpleCustomType> {
    static void serialize(IOStream &os, const SimpleCustomType &p) {
        os << p.val;
    }
    static SimpleCustomType deserialize(IIStream &is) {
        SimpleCustomType p;
        is >> p.val;
        return p;
    }
};

template<> struct S11N<SimpleCustomType2> {
    static void serialize(IOStream &os, const SimpleCustomType2 &p) {
        os << p.id;
    }
    static SimpleCustomType2 deserialize(IIStream &is) {
        SimpleCustomType2 p;
        is >> p.id;
        return p;
    }
};

template<> struct S11N<MyCustomType> {
    static void serialize(IOStream &os, const MyCustomType &p) {
        os << p.val << p.name << p.vec << p.mmap;
    }
    static MyCustomType deserialize(IIStream &is) {
        MyCustomType p;
        is >> p.val >> p.name >> p.vec >> p.mmap;
        return p;
    }
};
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv


namespace cv {
namespace detail {
template<> struct CompileArgTag<EmptyCustomType> {
    static const char* tag() {
        return "org.opencv.test.empty_custom_type";
    }
};

template<> struct CompileArgTag<SimpleCustomType> {
    static const char* tag() {
        return "org.opencv.test.simple_custom_type";
    }
};

template<> struct CompileArgTag<SimpleCustomType2> {
    static const char* tag() {
        return "org.opencv.test.simple_custom_type_2";
    }
};

template<> struct CompileArgTag<MyCustomType> {
    static const char* tag() {
        return "org.opencv.test.my_custom_type";
    }
};

template<> struct CompileArgTag<MyCustomTypeNoS11N> {
    static const char* tag() {
        return "org.opencv.test.my_custom_type_no_s11n";
    }
};
} // namespace detail
} // namespace cv

namespace {
class MyRMatAdapter : public cv::RMat::IAdapter {
    cv::Mat m_mat;
    int m_value;
    std::string m_str;
public:
    MyRMatAdapter() = default;
    MyRMatAdapter(cv::Mat m, int value, const std::string& str)
        : m_mat(m), m_value(value), m_str(str)
    {}
    virtual cv::RMat::View access(cv::RMat::Access) override {
        return cv::gimpl::asView(m_mat);
    }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
    virtual void serialize(cv::gapi::s11n::IOStream& os) override {
        os << m_value << m_str;
    }
    virtual void deserialize(cv::gapi::s11n::IIStream& is) override {
        is >> m_value >> m_str;
    }
    int getVal() { return m_value; }
    std::string getStr() { return m_str; }
};

class MyMediaFrameAdapter : public cv::MediaFrame::IAdapter {
    cv::Mat m_mat;
    int m_value;
    std::string m_str;
public:
    MyMediaFrameAdapter() = default;
    MyMediaFrameAdapter(cv::Mat m, int value, const std::string& str)
        : m_mat(m), m_value(value), m_str(str)
    {}
    virtual cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        return cv::MediaFrame::View({m_mat.data}, {m_mat.step});
    }
    virtual cv::GFrameDesc meta() const override { return {cv::MediaFormat::BGR, m_mat.size()}; }
    virtual void serialize(cv::gapi::s11n::IOStream& os) override {
        os << m_value << m_str;
    }
    virtual void deserialize(cv::gapi::s11n::IIStream& is) override {
        is >> m_value >> m_str;
    }
    int getVal() { return m_value; }
    std::string getStr() { return m_str; }
};
}

namespace opencv_test {

struct S11N_Basic: public ::testing::Test {
    template<typename T> void put(T &&t) {
        cv::gapi::s11n::ByteMemoryOutStream os;
        os << t;
        m_buffer = os.data();
    }

    template<typename T> T get() {
        // FIXME: This stream API needs a fix-up
        cv::gapi::s11n::ByteMemoryInStream is(m_buffer);
        T t{};
        is >> t;
        return t;
    }

private:
    std::vector<char> m_buffer;
};

namespace
{
    template<typename T>
    bool operator==(const cv::detail::VectorRef& a, const cv::detail::VectorRef& b)
    {
        return a.rref<T>() == b.rref<T>();
    }

    template<typename T>
    bool operator==(const cv::detail::OpaqueRef& a, const cv::detail::OpaqueRef& b)
    {
        return a.rref<T>() == b.rref<T>();
    }
}

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

TEST_F(S11N_Basic, Test_uint64) {
    uint64_t x = 2147483647374;
    put(x);
    EXPECT_EQ(x, get<uint64_t>());
}

TEST_F(S11N_Basic, Test_int32_pos) {
    int32_t x = 2147483647;
    put(x);
    EXPECT_EQ(x, get<int32_t>());
}

TEST_F(S11N_Basic, Test_int32_neg) {
    int32_t x = -2147483646;
    put(x);
    EXPECT_EQ(x, get<int32_t>());
}

TEST_F(S11N_Basic, Test_vector_bool) {
    std::vector<bool> v = {false, true, false};
    put(v);
    EXPECT_EQ(v, get<std::vector<bool>>());
}

TEST_F(S11N_Basic, Test_map_string2string) {
    using T = std::map<std::string, std::string>;
    T v;
    v["gapi"] = "cool";
    v["42"] = "answer";
    v["hi"] = "hello there";
    put(v);
    EXPECT_EQ(v, get<T>());
}

TEST_F(S11N_Basic, Test_map_int2int) {
    using T = std::map<int, int32_t>;
    T v;
    v[1] = 23;
    v[-100] = 0;
    v[435346] = -12346;
    put(v);
    EXPECT_EQ(v, get<T>());
}

TEST_F(S11N_Basic, Test_map_float2cvsize) {
    using T = std::map<float, cv::Size>;
    T v;
    v[0.4f] = cv::Size(4, 5);
    v[234.43f] = cv::Size(3421, 321);
    v[2223.f] = cv::Size(1920, 1080);
    put(v);
    EXPECT_EQ(v, get<T>());
}

TEST_F(S11N_Basic, Test_map_uint642cvmat) {
    using T = std::map<uint64_t, cv::Mat>;
    T v;
    v[21304805324] = cv::Mat(3, 3, CV_8UC1, cv::Scalar::all(3));
    v[4353245222] = cv::Mat(5, 5, CV_8UC3, cv::Scalar::all(7));
    v[0] = cv::Mat(10, 10, CV_32FC2, cv::Scalar::all(-128.f));
    put(v);
    auto out_v = get<T>();
    for (const auto& el : out_v) {
        EXPECT_NE(v.end(), v.find(el.first));
        EXPECT_EQ(0, cv::norm(el.second, v[el.first]));
    }
}

TEST_F(S11N_Basic, Test_vector_int) {
    std::vector<int> v = {1,2,3};
    put(v);
    EXPECT_EQ(v, get<std::vector<int>>());
}

TEST_F(S11N_Basic, Test_vector_cvSize) {
    std::vector<cv::Size> v = {
        cv::Size(640, 480),
        cv::Size(1280, 1024),
    };
    put(v);
    EXPECT_EQ(v, get<std::vector<cv::Size>>());
}

TEST_F(S11N_Basic, Test_vector_string) {
    std::vector<std::string> v = {
        "hello",
        "world",
        "ok!"
    };
    put(v);
    EXPECT_EQ(v, get<std::vector<std::string>>());
}

TEST_F(S11N_Basic, Test_vector_empty) {
    std::vector<char> v;
    put(v);
    EXPECT_EQ(v, get<std::vector<char>>());
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

TEST_F(S11N_Basic, Test_MatDescND) {
    cv::GMatDesc v = { CV_8U, {1,1,224,224} };
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

TEST_F(S11N_Basic, Test_MetaArgs_MatDesc) {
    cv::GMatDesc desc1 = { CV_8U, 1,{ 320,240 } };
    cv::GMatDesc desc2 = { CV_8U, 1,{ 640,480 } };
    GMetaArgs v;
    v.resize(2);
    v[0] = cv::GMetaArg{ desc1 };
    v[1] = cv::GMetaArg{ desc2 };
    put(v);
    cv::GMetaArgs out_v = get<cv::GMetaArgs>();
    cv::GMatDesc out_desc1 = cv::util::get<cv::GMatDesc>(out_v[0]);
    cv::GMatDesc out_desc2 = cv::util::get<cv::GMatDesc>(out_v[1]);
    EXPECT_EQ(desc1, out_desc1);
    EXPECT_EQ(desc2, out_desc2);
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

TEST_F(S11N_Basic, Test_RunArgs_Mat) {
    cv::Mat mat1 = cv::Mat::eye(cv::Size(64, 64), CV_8UC3);
    cv::Mat mat2 = cv::Mat::eye(cv::Size(128, 128), CV_8UC3);
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ mat1 };
    v[1] = cv::GRunArg{ mat2 };
    put(v);
    cv::GRunArgs out_v = get<cv::GRunArgs>();
    cv::Mat out_mat1 = cv::util::get<cv::Mat>(out_v[0]);
    cv::Mat out_mat2 = cv::util::get<cv::Mat>(out_v[1]);
    EXPECT_EQ(0, cv::norm(mat1, out_mat1, cv::NORM_INF));
    EXPECT_EQ(0, cv::norm(mat2, out_mat2, cv::NORM_INF));
}

TEST_F(S11N_Basic, Test_RunArg_Scalar) {
    cv::Scalar scalar = cv::Scalar(128, 33, 53);
    auto v = cv::GRunArg{ scalar };
    put(v);
    cv::GRunArg out_v = get<cv::GRunArg>();
    cv::Scalar out_scalar = cv::util::get<cv::Scalar>(out_v);
    EXPECT_EQ(scalar, out_scalar);
}

TEST_F(S11N_Basic, Test_RunArgs_Scalar) {
    cv::Scalar scalar1 = cv::Scalar(128, 33, 53);
    cv::Scalar scalar2 = cv::Scalar(64, 15, 23);
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ scalar1 };
    v[1] = cv::GRunArg{ scalar2 };
    put(v);
    cv::GRunArgs out_v = get<cv::GRunArgs>();
    cv::Scalar out_scalar1 = cv::util::get<cv::Scalar>(out_v[0]);
    cv::Scalar out_scalar2 = cv::util::get<cv::Scalar>(out_v[1]);
    EXPECT_EQ(scalar1, out_scalar1);
    EXPECT_EQ(scalar2, out_scalar2);
}

TEST_F(S11N_Basic, Test_RunArg_Opaque) {
    auto op = cv::detail::OpaqueRef(42);
    auto v = cv::GRunArg{ op };
    put(v);
    cv::GRunArg out_v = get<cv::GRunArg>();
    cv::detail::OpaqueRef out_op = cv::util::get<cv::detail::OpaqueRef>(out_v);
    EXPECT_TRUE(operator==<int>(op, out_op));
}

TEST_F(S11N_Basic, Test_RunArgs_Opaque) {
    cv::detail::OpaqueRef op1 = cv::detail::OpaqueRef(cv::Point(1, 2));
    cv::detail::OpaqueRef op2 = cv::detail::OpaqueRef(cv::Size(12, 21));
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ op1 };
    v[1] = cv::GRunArg{ op2 };
    put(v);
    cv::GRunArgs out_v = get<cv::GRunArgs>();
    cv::detail::OpaqueRef out_op1 = cv::util::get<cv::detail::OpaqueRef>(out_v[0]);
    cv::detail::OpaqueRef out_op2 = cv::util::get<cv::detail::OpaqueRef>(out_v[1]);
    EXPECT_TRUE(operator==<cv::Point>(op1, out_op1));
    EXPECT_TRUE(operator==<cv::Size>(op2, out_op2));
}

TEST_F(S11N_Basic, Test_RunArg_Array) {
    auto op = cv::detail::VectorRef(std::vector<cv::Mat>{cv::Mat::eye(3, 3, CV_8UC1), cv::Mat::zeros(5, 5, CV_8UC3)});

    auto v = cv::GRunArg{ op };
    put(v);
    cv::GRunArg out_v = get<cv::GRunArg>();
    cv::detail::VectorRef out_op = cv::util::get<cv::detail::VectorRef>(out_v);
    auto vec1 = op.rref<cv::Mat>();
    auto vec2 = out_op.rref<cv::Mat>();
    EXPECT_EQ(0, cv::norm(vec1[0], vec2[0], cv::NORM_INF));
    EXPECT_EQ(0, cv::norm(vec1[1], vec2[1], cv::NORM_INF));
}

TEST_F(S11N_Basic, Test_RunArgs_Array) {
    auto vec_sc = std::vector<cv::Scalar>{cv::Scalar(11), cv::Scalar(31)};
    auto vec_d = std::vector<double>{0.4, 1.0, 123.55, 22.08};
    cv::detail::VectorRef op1 = cv::detail::VectorRef(vec_sc);
    cv::detail::VectorRef op2 = cv::detail::VectorRef(vec_d);
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ op1 };
    v[1] = cv::GRunArg{ op2 };
    put(v);
    cv::GRunArgs out_v = get<cv::GRunArgs>();
    cv::detail::VectorRef out_op1 = cv::util::get<cv::detail::VectorRef>(out_v[0]);
    cv::detail::VectorRef out_op2 = cv::util::get<cv::detail::VectorRef>(out_v[1]);
    EXPECT_TRUE(operator==<cv::Scalar>(op1, out_op1));
    EXPECT_TRUE(operator==<double>(op2, out_op2));
}

TEST_F(S11N_Basic, Test_RunArgs_MatScalar) {
    cv::Mat mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC3);
    cv::Scalar scalar = cv::Scalar(128, 33, 53);
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ mat };
    v[1] = cv::GRunArg{ scalar };
    put(v);
    cv::GRunArgs out_v = get<cv::GRunArgs>();
    unsigned int i = 0;
    for (auto it : out_v)
    {
        using T = cv::GRunArg;
        switch (it.index())
        {
        case T::index_of<cv::Mat>() :
        {
            cv::Mat out_mat = cv::util::get<cv::Mat>(out_v[i]);
            EXPECT_EQ(0, cv::norm(mat, out_mat, cv::NORM_INF));
        } break;
        case T::index_of<cv::Scalar>() :
        {
            cv::Scalar out_scalar = cv::util::get<cv::Scalar>(out_v[i]);
            EXPECT_EQ(scalar, out_scalar);
        } break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
        i++;
    }
}

TEST_F(S11N_Basic, Test_Bind_RunArgs_MatScalar) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    cv::Scalar scalar = cv::Scalar(128, 33, 53);
    GRunArgs v;
    v.resize(2);
    v[0] = cv::GRunArg{ mat };
    v[1] = cv::GRunArg{ scalar };
    GRunArgsP output = cv::gapi::bind(v);
    for (auto it : output)
    {
        using T = cv::GRunArgP;
        switch (it.index())
        {
        case T::index_of<cv::Mat*>() :
        {
            cv::Mat* out_mat = cv::util::get<cv::Mat*>(it);
            EXPECT_EQ(mat.size(), out_mat->size());
        } break;
        case T::index_of<cv::Scalar*>() :
        {
            cv::Scalar* out_scalar = cv::util::get<cv::Scalar*>(it);
            EXPECT_EQ(out_scalar->val[0], scalar.val[0]);
            EXPECT_EQ(out_scalar->val[1], scalar.val[1]);
            EXPECT_EQ(out_scalar->val[2], scalar.val[2]);
        } break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
}

TEST_F(S11N_Basic, Test_Vector_Of_Strings) {
    std::vector<std::string> vs{"hello", "world", "42"};

    const std::vector<char> ser = cv::gapi::serialize(vs);
    auto des = cv::gapi::deserialize<std::vector<std::string>>(ser);
    EXPECT_EQ("hello", des[0]);
    EXPECT_EQ("world", des[1]);
    EXPECT_EQ("42", des[2]);
}

TEST_F(S11N_Basic, Test_RunArg) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    auto v = cv::GRunArgs{ cv::GRunArg{ mat } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs>(sargsin);
    cv::Mat out_mat = cv::util::get<cv::Mat>(out[0]);

    EXPECT_EQ(0, cv::norm(mat, out_mat));
}

TEST_F(S11N_Basic, Test_RunArg_RMat) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    cv::RMat rmat = cv::make_rmat<MyRMatAdapter>(mat, 42, "It actually works");
    auto v = cv::GRunArgs{ cv::GRunArg{ rmat } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyRMatAdapter>(sargsin);
    cv::RMat out_mat = cv::util::get<cv::RMat>(out[0]);
    auto adapter = out_mat.get<MyRMatAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());
}

TEST_F(S11N_Basic, Test_RunArg_RMat_Scalar_Mat) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    cv::RMat rmat = cv::make_rmat<MyRMatAdapter>(mat, 42, "It actually works");
    cv::Scalar sc(111);
    auto v = cv::GRunArgs{ cv::GRunArg{ rmat }, cv::GRunArg{ sc }, cv::GRunArg{ mat } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyRMatAdapter>(sargsin);
    cv::RMat out_rmat = cv::util::get<cv::RMat>(out[0]);
    auto adapter = out_rmat.get<MyRMatAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());

    cv::Scalar out_sc = cv::util::get<cv::Scalar>(out[1]);
    EXPECT_EQ(sc, out_sc);

    cv::Mat out_mat = cv::util::get<cv::Mat>(out[2]);
    EXPECT_EQ(0, cv::norm(mat, out_mat));
}

TEST_F(S11N_Basic, Test_RunArg_MediaFrame) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    auto frame = cv::MediaFrame::Create<MyMediaFrameAdapter>(mat, 42, "It actually works");
    auto v = cv::GRunArgs{ cv::GRunArg{ frame } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyMediaFrameAdapter>(sargsin);
    cv::MediaFrame out_mat = cv::util::get<cv::MediaFrame>(out[0]);
    auto adapter = out_mat.get<MyMediaFrameAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());
}

TEST_F(S11N_Basic, Test_RunArg_MediaFrame_Scalar_Mat) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    auto frame = cv::MediaFrame::Create<MyMediaFrameAdapter>(mat, 42, "It actually works");
    cv::Scalar sc(111);
    auto v = cv::GRunArgs{ cv::GRunArg{ frame }, cv::GRunArg{ sc }, cv::GRunArg{ mat } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyMediaFrameAdapter>(sargsin);
    cv::MediaFrame out_frame = cv::util::get<cv::MediaFrame>(out[0]);
    auto adapter = out_frame.get<MyMediaFrameAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());

    cv::Scalar out_sc = cv::util::get<cv::Scalar>(out[1]);
    EXPECT_EQ(sc, out_sc);

    cv::Mat out_mat = cv::util::get<cv::Mat>(out[2]);
    EXPECT_EQ(0, cv::norm(mat, out_mat));
}

TEST_F(S11N_Basic, Test_RunArg_MediaFrame_RMat) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    cv::Mat mat2 = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);

    auto frame = cv::MediaFrame::Create<MyMediaFrameAdapter>(mat, 42, "It actually works");
    auto rmat = cv::make_rmat<MyRMatAdapter>(mat2, 24, "Hello there");

    auto v = cv::GRunArgs{ cv::GRunArg{ frame }, cv::GRunArg{ rmat } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyMediaFrameAdapter, MyRMatAdapter>(sargsin);

    cv::MediaFrame out_frame = cv::util::get<cv::MediaFrame>(out[0]);
    cv::RMat out_rmat = cv::util::get<cv::RMat>(out[1]);

    auto adapter = out_frame.get<MyMediaFrameAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());

    auto adapter2 = out_rmat.get<MyRMatAdapter>();
    EXPECT_EQ(24, adapter2->getVal());
    EXPECT_EQ("Hello there", adapter2->getStr());
}

TEST_F(S11N_Basic, Test_RunArg_RMat_MediaFrame) {
    cv::Mat mat = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);
    cv::Mat mat2 = cv::Mat::eye(cv::Size(128, 64), CV_8UC3);

    auto frame = cv::MediaFrame::Create<MyMediaFrameAdapter>(mat, 42, "It actually works");
    auto rmat = cv::make_rmat<MyRMatAdapter>(mat2, 24, "Hello there");

    auto v = cv::GRunArgs{ cv::GRunArg{ rmat }, cv::GRunArg{ frame } };

    const std::vector<char> sargsin = cv::gapi::serialize(v);
    cv::GRunArgs out = cv::gapi::deserialize<cv::GRunArgs, MyMediaFrameAdapter, MyRMatAdapter>(sargsin);

    cv::RMat out_rmat = cv::util::get<cv::RMat>(out[0]);
    cv::MediaFrame out_frame = cv::util::get<cv::MediaFrame>(out[1]);

    auto adapter = out_frame.get<MyMediaFrameAdapter>();
    EXPECT_EQ(42, adapter->getVal());
    EXPECT_EQ("It actually works", adapter->getStr());

    auto adapter2 = out_rmat.get<MyRMatAdapter>();
    EXPECT_EQ(24, adapter2->getVal());
    EXPECT_EQ("Hello there", adapter2->getStr());
}

namespace {
    template <cv::detail::OpaqueKind K, typename T>
    bool verifyOpaqueKind(T&& in) {
        auto inObjs = cv::gin(in);
        auto in_o_ref = cv::util::get<cv::detail::OpaqueRef>(inObjs[0]);
        return K == in_o_ref.getKind();
    }

    template <cv::detail::OpaqueKind K, typename T>
    bool verifyArrayKind(T&& in) {
        auto inObjs = cv::gin(in);
        auto in_o_ref = cv::util::get<cv::detail::VectorRef>(inObjs[0]);
        return K == in_o_ref.getKind();
    }
}

TEST_F(S11N_Basic, Test_Gin_GOpaque) {
    int i; float f; double d;
    std::uint64_t ui; bool b;
    std::string s;
    cv::Rect r; cv::Size sz;
    cv::Point p;
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_INT>(i));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_FLOAT>(f));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_DOUBLE>(d));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_UINT64>(ui));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_BOOL>(b));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_STRING>(s));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_RECT>(r));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_SIZE>(sz));
    EXPECT_TRUE(verifyOpaqueKind<cv::detail::OpaqueKind::CV_POINT>(p));
}

TEST_F(S11N_Basic, Test_Gin_GArray) {
    std::vector<int> i; std::vector<float> f; std::vector<double> d;
    std::vector<std::uint64_t> ui; std::vector<bool> b;
    std::vector<std::string> s;
    std::vector<cv::Rect> r; std::vector<cv::Size> sz;
    std::vector<cv::Point> p;
    std::vector<cv::Mat> mat;
    std::vector<cv::Scalar> sc;
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_INT>(i));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_FLOAT>(f));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_DOUBLE>(d));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_UINT64>(ui));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_BOOL>(b));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_STRING>(s));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_RECT>(r));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_SIZE>(sz));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_POINT>(p));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_MAT>(mat));
    EXPECT_TRUE(verifyArrayKind<cv::detail::OpaqueKind::CV_SCALAR>(sc));
}

TEST_F(S11N_Basic, Test_Custom_Type) {
    MyCustomType var{1324, "Hello", {1920, 1080, 720}, {{1, 2937459432}, {42, 253245432}}};
    cv::gapi::s11n::ByteMemoryOutStream os;
    cv::gapi::s11n::detail::S11N<MyCustomType>::serialize(os, var);
    cv::gapi::s11n::ByteMemoryInStream is(os.data());
    MyCustomType new_var = cv::gapi::s11n::detail::S11N<MyCustomType>::deserialize(is);
    EXPECT_EQ(var, new_var);
}

TEST_F(S11N_Basic, Test_CompileArg) {
    MyCustomType customVar{1248, "World", {1280, 720, 640, 480}, {{5, 32434142342}, {7, 34242432}}};

    std::vector<char> sArgs = cv::gapi::serialize(cv::compile_args(customVar));

    GCompileArgs dArgs = cv::gapi::deserialize<GCompileArgs, MyCustomType>(sArgs);

    MyCustomType dCustomVar = cv::gapi::getCompileArg<MyCustomType>(dArgs).value();
    EXPECT_EQ(customVar, dCustomVar);
}

TEST_F(S11N_Basic, Test_CompileArg_Without_UserCallback) {
    SimpleCustomType   customVar1 { false };
    MyCustomTypeNoS11N customVar2 { 'z', 189, "Name" };
    MyCustomType       customVar3 { 1248, "World", {1280, 720, 640, 480},
                                    {{5, 32434142342}, {7, 34242432}} };

    EXPECT_NO_THROW(cv::gapi::serialize(cv::compile_args(customVar1, customVar2, customVar3)));

    std::vector<char> sArgs = cv::gapi::serialize(
        cv::compile_args(customVar1, customVar2, customVar3));

    GCompileArgs dArgs = cv::gapi::deserialize<GCompileArgs,
                                               SimpleCustomType,
                                               MyCustomType>(sArgs);

    SimpleCustomType dCustomVar1 = cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value();
    MyCustomType     dCustomVar3 = cv::gapi::getCompileArg<MyCustomType>(dArgs).value();

    EXPECT_EQ(customVar1, dCustomVar1);
    EXPECT_EQ(customVar3, dCustomVar3);
}

TEST_F(S11N_Basic, Test_Deserialize_Only_Requested_CompileArgs) {
    MyCustomType     myCustomVar { 1248, "World", {1280, 720, 640, 480},
                                   {{5, 32434142342}, {7, 34242432}} };
    SimpleCustomType simpleCustomVar { false };

    std::vector<char> sArgs = cv::gapi::serialize(cv::compile_args(myCustomVar, simpleCustomVar));

    GCompileArgs dArgs = cv::gapi::deserialize<GCompileArgs, MyCustomType>(sArgs);
    EXPECT_EQ(1u, dArgs.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgs).value());

    dArgs.clear();
    dArgs = cv::gapi::deserialize<GCompileArgs, SimpleCustomType>(sArgs);
    EXPECT_EQ(1u, dArgs.size());
    EXPECT_EQ(simpleCustomVar, cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value());

    dArgs.clear();
    dArgs = cv::gapi::deserialize<GCompileArgs, SimpleCustomType2>(sArgs);
    EXPECT_EQ(0u, dArgs.size());

    dArgs.clear();
    dArgs = cv::gapi::deserialize<GCompileArgs, MyCustomType, SimpleCustomType>(sArgs);
    EXPECT_EQ(2u, dArgs.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgs).value());
    EXPECT_EQ(simpleCustomVar, cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value());

    SimpleCustomType2 simpleCustomVar2 { 5 };
    std::vector<char> sArgs2 = cv::gapi::serialize(
        cv::compile_args(myCustomVar, simpleCustomVar, simpleCustomVar2));
    GCompileArgs dArgs2 = cv::gapi::deserialize<GCompileArgs,
                                                MyCustomType,
                                                SimpleCustomType2>(sArgs2);
    EXPECT_EQ(2u, dArgs2.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgs2).value());
    EXPECT_EQ(simpleCustomVar2, cv::gapi::getCompileArg<SimpleCustomType2>(dArgs2).value());
}

TEST_F(S11N_Basic, Test_Deserialize_CompileArgs_RandomOrder) {
    SimpleCustomType  simpleCustomVar { false };
    SimpleCustomType2 simpleCustomVar2 { 5 };

    std::vector<char> sArgs = cv::gapi::serialize(
        cv::compile_args(simpleCustomVar, simpleCustomVar2));
    GCompileArgs dArgs = cv::gapi::deserialize<GCompileArgs,
                                               SimpleCustomType2,
                                               SimpleCustomType>(sArgs);

    EXPECT_EQ(simpleCustomVar, cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value());
    EXPECT_EQ(simpleCustomVar2, cv::gapi::getCompileArg<SimpleCustomType2>(dArgs).value());
}

TEST_F(S11N_Basic, Test_CompileArgs_With_EmptyCompileArg) {
    MyCustomType      myCustomVar { 1248, "World", {1280, 720, 640, 480},
                                    {{5, 32434142342}, {7, 34242432}} };
    SimpleCustomType  simpleCustomVar { false };
    EmptyCustomType   emptyCustomVar {  };

    //----{ emptyCustomVar, myCustomVar }----
    std::vector<char> sArgs1 = cv::gapi::serialize(cv::compile_args(emptyCustomVar, myCustomVar));
    GCompileArgs dArgsEmptyVar1 = cv::gapi::deserialize<GCompileArgs, EmptyCustomType>(sArgs1);
    GCompileArgs dArgsMyVar1 = cv::gapi::deserialize<GCompileArgs, MyCustomType>(sArgs1);
    GCompileArgs dArgsEmptyAndMyVars1 = cv::gapi::deserialize<GCompileArgs,
                                                              EmptyCustomType,
                                                              MyCustomType>(sArgs1);
    EXPECT_EQ(1u, dArgsEmptyVar1.size());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsEmptyVar1).has_value());
    EXPECT_EQ(1u, dArgsMyVar1.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsMyVar1).value());
    EXPECT_EQ(2u, dArgsEmptyAndMyVars1.size());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsEmptyAndMyVars1).has_value());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsEmptyAndMyVars1).value());

    //----{ myCustomVar, emptyCustomVar }----
    std::vector<char> sArgs2 = cv::gapi::serialize(cv::compile_args(myCustomVar, emptyCustomVar));
    GCompileArgs dArgsMyVar2 = cv::gapi::deserialize<GCompileArgs, MyCustomType>(sArgs2);
    GCompileArgs dArgsEmptyVar2 = cv::gapi::deserialize<GCompileArgs, EmptyCustomType>(sArgs2);
    GCompileArgs dArgsMyAndEmptyVars2 = cv::gapi::deserialize<GCompileArgs,
                                                              MyCustomType,
                                                              EmptyCustomType>(sArgs2);
    EXPECT_EQ(1u, dArgsMyVar2.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsMyVar2).value());
    EXPECT_EQ(1u, dArgsEmptyVar2.size());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsEmptyVar2).has_value());
    EXPECT_EQ(2u, dArgsMyAndEmptyVars2.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsMyAndEmptyVars2).value());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsMyAndEmptyVars2).has_value());

    //----{ myCustomVar, emptyCustomVar, simpleCustomVar }----
    std::vector<char> sArgs3 = cv::gapi::serialize(
        cv::compile_args(myCustomVar, emptyCustomVar, simpleCustomVar));
    GCompileArgs dArgsMyVar3 = cv::gapi::deserialize<GCompileArgs, MyCustomType>(sArgs3);
    GCompileArgs dArgsEmptyVar3 = cv::gapi::deserialize<GCompileArgs, EmptyCustomType>(sArgs3);
    GCompileArgs dArgsSimpleVar3 = cv::gapi::deserialize<GCompileArgs, SimpleCustomType>(sArgs3);
    GCompileArgs dArgsMyAndSimpleVars3 = cv::gapi::deserialize<GCompileArgs,
                                                               MyCustomType,
                                                               SimpleCustomType>(sArgs3);
    GCompileArgs dArgs3 = cv::gapi::deserialize<GCompileArgs,
                                                MyCustomType,
                                                EmptyCustomType,
                                                SimpleCustomType>(sArgs3);
    EXPECT_EQ(1u, dArgsMyVar3.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsMyVar3).value());
    EXPECT_EQ(1u, dArgsEmptyVar3.size());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsEmptyVar3).has_value());
    EXPECT_EQ(1u, dArgsSimpleVar3.size());
    EXPECT_EQ(simpleCustomVar, cv::gapi::getCompileArg<SimpleCustomType>(dArgsSimpleVar3).value());
    EXPECT_EQ(2u, dArgsMyAndSimpleVars3.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgsMyAndSimpleVars3).value());
    EXPECT_EQ(simpleCustomVar,
              cv::gapi::getCompileArg<SimpleCustomType>(dArgsMyAndSimpleVars3).value());
    EXPECT_EQ(3u, dArgs3.size());
    EXPECT_EQ(myCustomVar, cv::gapi::getCompileArg<MyCustomType>(dArgs3).value());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgs3).has_value());
    EXPECT_EQ(simpleCustomVar, cv::gapi::getCompileArg<SimpleCustomType>(dArgs3).value());

    //----{ emptyCustomVar }----
    std::vector<char> sArgs4 = cv::gapi::serialize(cv::compile_args(emptyCustomVar));
    GCompileArgs dArgsEmptyVar4 = cv::gapi::deserialize<GCompileArgs, EmptyCustomType>(sArgs4);
    EXPECT_EQ(1u, dArgsEmptyVar4.size());
    EXPECT_TRUE(cv::gapi::getCompileArg<EmptyCustomType>(dArgsEmptyVar4).has_value());
}

} // namespace opencv_test
