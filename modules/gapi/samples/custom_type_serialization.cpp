#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/gcommon.hpp>

struct SimpleCustomType {
    bool val;
    bool operator==(const SimpleCustomType& other) const {
        return val == other.val;
    }
};

struct SimpleCustomType2 {
    int val;
    std::string name;
    std::vector<float> vec;
    std::map<int, uint64_t> mmap;
    bool operator==(const SimpleCustomType2& other) const {
        return val == other.val && name == other.name &&
               vec == other.vec && mmap == other.mmap;
    }
};

namespace cv {
namespace gapi {
namespace s11n {
namespace detail {
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
        os << p.val << p.name << p.vec << p.mmap;
    }
    static SimpleCustomType2 deserialize(IIStream &is) {
        SimpleCustomType2 p;
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
} // namespace detail
} // namespace cv

int main(int argc, char *argv[])
{
    SimpleCustomType  customVar1 { false };
    SimpleCustomType2 customVar2 { 1248, "World", {1280, 720, 640, 480},
                                   { {5, 32434142342}, {7, 34242432} } };

// ! [bind usage]
    std::vector<char> sArgs = cv::gapi::serialize(
        cv::compile_args(customVar1, customVar2));

    cv::GCompileArgs dArgs = cv::gapi::deserialize<cv::GCompileArgs,
                                                   SimpleCustomType,
                                                   SimpleCustomType2>(sArgs);
// ! [bind usage]

    SimpleCustomType  dCustomVar1 = cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value();
    SimpleCustomType2 dCustomVar2 = cv::gapi::getCompileArg<SimpleCustomType2>(dArgs).value();
}
