#ifndef __OPENCV_TS_PERF_HPP__
#define __OPENCV_TS_PERF_HPP__

#include "opencv2/core/core.hpp"
#include "ts_gtest.h"

#if defined(ANDROID) && defined(USE_ANDROID_LOGGING)
#include <android/log.h>

#define PERF_TESTS_LOG_TAG "OpenCV_perf"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#else
#define LOGD(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGI(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGW(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGE(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#endif


namespace perf
{

/*****************************************************************************************\
*                Predefined typical frame sizes and typical test parameters               *
\*****************************************************************************************/
const cv::Size szQVGA = cv::Size(320, 240);
const cv::Size szVGA = cv::Size(640, 480);
const cv::Size szSVGA = cv::Size(800, 600);
const cv::Size szXGA = cv::Size(1024, 768);
const cv::Size szSXGA = cv::Size(1280, 1024);

const cv::Size sznHD = cv::Size(640, 360);
const cv::Size szqHD = cv::Size(960, 540);
const cv::Size sz720p = cv::Size(1280, 720);
const cv::Size sz1080p = cv::Size(1920, 1080);

const cv::Size szODD = cv::Size(127, 61);

#define SZ_ALL_VGA ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA)
#define SZ_ALL_GA  ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA, ::perf::szXGA, ::perf::szSXGA)
#define SZ_ALL_HD  ::testing::Values(::perf::sznHD, ::perf::szqHD, ::perf::sz720p, ::perf::sz1080p)
#define SZ_ALL  ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA, ::perf::szXGA, ::perf::szSXGA, ::perf::sznHD, ::perf::szqHD, ::perf::sz720p, ::perf::sz1080p)
#define SZ_TYPICAL  ::testing::Values(::perf::szVGA, ::perf::szqHD, ::perf::sz720p, ::perf::szODD)


#define TYPICAL_MAT_SIZES ::perf::szVGA, ::perf::sz720p, ::perf::sz1080p, ::perf::szODD
#define TYPICAL_MAT_TYPES CV_8UC1, CV_8UC4, CV_32FC1
#define TYPICAL_MATS testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( TYPICAL_MAT_TYPES ) )
#define TYPICAL_MATS_C1 testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( CV_8UC1, CV_32FC1 ) )
#define TYPICAL_MATS_C4 testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( CV_8UC4 ) )


/*****************************************************************************************\
*                MatType - printable wrapper over integer 'type' of Mat                   *
\*****************************************************************************************/
class MatType
{
public:
    MatType(int val=0) : _type(val) {}
    operator int() const {return _type;}

private:
    int _type;
};

/*****************************************************************************************\
*     CV_ENUM and CV_FLAGS - macro to create printable wrappers for defines and enums     *
\*****************************************************************************************/

#define CV_ENUM(class_name, ...) \
class CV_EXPORTS class_name {\
public:\
  class_name(int val = 0) : _val(val) {}\
  operator int() const {return _val;}\
  void PrintTo(std::ostream* os) const {\
    const int vals[] = {__VA_ARGS__};\
    const char* svals = #__VA_ARGS__;\
    for(int i = 0, pos = 0; i < (int)(sizeof(vals)/sizeof(int)); ++i){\
      while(isspace(svals[pos]) || svals[pos] == ',') ++pos;\
      int start = pos;\
      while(!(isspace(svals[pos]) || svals[pos] == ',' || svals[pos] == 0)) ++pos;\
      if (_val == vals[i]) {\
        *os << std::string(svals + start, svals + pos);\
        return;\
      }\
    }\
    *os << "UNKNOWN";\
  }\
private: int _val;\
};\
inline void PrintTo(const class_name& t, std::ostream* os) { t.PrintTo(os); }

#define CV_FLAGS(class_name, ...) \
class CV_EXPORTS class_name {\
public:\
  class_name(int val = 0) : _val(val) {}\
  operator int() const {return _val;}\
  void PrintTo(std::ostream* os) const {\
    const int vals[] = {__VA_ARGS__};\
    const char* svals = #__VA_ARGS__;\
    int value = _val;\
    bool first = true;\
    for(int i = 0, pos = 0; i < (int)(sizeof(vals)/sizeof(int)); ++i){\
      while(isspace(svals[pos]) || svals[pos] == ',') ++pos;\
      int start = pos;\
      while(!(isspace(svals[pos]) || svals[pos] == ',' || svals[pos] == 0)) ++pos;\
      if ((value & vals[i]) == vals[i]) {\
        value &= ~vals[i]; \
        if (first) first = false; else *os << "|"; \
        *os << std::string(svals + start, svals + pos);\
        if (!value) return;\
      }\
    }\
    if (first) *os << "UNKNOWN";\
  }\
private: int _val;\
};\
inline void PrintTo(const class_name& t, std::ostream* os) { t.PrintTo(os); }

CV_ENUM(MatDepth, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_USRTYPE1)

/*****************************************************************************************\
*                 Regression control utility for performance testing                      *
\*****************************************************************************************/
class CV_EXPORTS Regression
{
public:
    static Regression& add(const std::string& name, cv::InputArray array, double eps = DBL_EPSILON);
    static void Init(const std::string& testSuitName, const std::string& ext = ".xml");

    Regression& operator() (const std::string& name, cv::InputArray array, double eps = DBL_EPSILON);

private:
    static Regression& instance();
    Regression();
    ~Regression();

    Regression(const Regression&);
    Regression& operator=(const Regression&);

    cv::RNG regRNG;//own random numbers generator to make collection and verification work identical
    std::string storageInPath;
    std::string storageOutPath;
    cv::FileStorage storageIn;
    cv::FileStorage storageOut;
    cv::FileNode rootIn;
    std::string currentTestNodeName;
    cv::FileStorage& write();

    static std::string getCurrentTestNodeName();
    static bool isVector(cv::InputArray a);
    static double getElem(cv::Mat& m, int x, int y, int cn = 0);

    void init(const std::string& testSuitName, const std::string& ext);
    void write(cv::InputArray array);
    void write(cv::Mat m);
    void verify(cv::FileNode node, cv::InputArray array, double eps);
    void verify(cv::FileNode node, cv::Mat actual, double eps, std::string argname);
};

#define SANITY_CHECK(array, ...) ::perf::Regression::add(#array, array , ## __VA_ARGS__)


/*****************************************************************************************\
*                            Container for performance metrics                            *
\*****************************************************************************************/
typedef struct CV_EXPORTS performance_metrics
{
    size_t bytesIn;
    size_t bytesOut;
    unsigned int samples;
    unsigned int outliers;
    double gmean;
    double gstddev;//stddev for log(time)
    double mean;
    double stddev;
    double median;
    double min;
    double frequency;
    int terminationReason;

    enum
    {
        TERM_ITERATIONS = 0,
        TERM_TIME = 1,
        TERM_INTERRUPT = 2,
        TERM_EXCEPTION = 3,
        TERM_UNKNOWN = -1
    };

    performance_metrics();
} performance_metrics;


/*****************************************************************************************\
*                           Base fixture for performance tests                            *
\*****************************************************************************************/
class CV_EXPORTS TestBase: public ::testing::Test
{
public:
    TestBase();

    static void Init(int argc, const char* const argv[]);
    static std::string getDataPath(const std::string& relativePath);

protected:
    virtual void PerfTestBody() = 0;

    virtual void SetUp();
    virtual void TearDown();

    void startTimer();
    void stopTimer();
    bool next();

    //_declareHelper declare;

    enum
    {
        WARMUP_READ,
        WARMUP_WRITE,
        WARMUP_RNG,
        WARMUP_NONE
    };

    void reportMetrics(bool toJUnitXML = false);
    static void warmup(cv::InputOutputArray a, int wtype = WARMUP_READ);

    performance_metrics& calcMetrics();
    void RunPerfTestBody();
private:
    typedef std::vector<std::pair<int, cv::Size> > SizeVector;
    typedef std::vector<int64> TimeVector;

    SizeVector inputData;
    SizeVector outputData;
    unsigned int getTotalInputSize() const;
    unsigned int getTotalOutputSize() const;

    TimeVector times;
    int64 lastTime;
    int64 totalTime;
    int64 timeLimit;
    static int64 timeLimitDefault;

    unsigned int nIters;
    unsigned int currentIter;

    performance_metrics metrics;
    void validateMetrics();

    static int64 _timeadjustment;
    static int64 _calibrate();

    static void warmup(cv::Mat m, int wtype);
    static int getSizeInBytes(cv::InputArray a);
    static cv::Size getSize(cv::InputArray a);
    static void declareArray(SizeVector& sizes, cv::InputOutputArray a, int wtype = 0);

    class CV_EXPORTS _declareHelper
    {
    public:
        _declareHelper& in(cv::InputOutputArray a1, int wtype = WARMUP_READ);
        _declareHelper& in(cv::InputOutputArray a1, cv::InputOutputArray a2, int wtype = WARMUP_READ);
        _declareHelper& in(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, int wtype = WARMUP_READ);
        _declareHelper& in(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, cv::InputOutputArray a4, int wtype = WARMUP_READ);

        _declareHelper& out(cv::InputOutputArray a1, int wtype = WARMUP_WRITE);
        _declareHelper& out(cv::InputOutputArray a1, cv::InputOutputArray a2, int wtype = WARMUP_WRITE);
        _declareHelper& out(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, int wtype = WARMUP_WRITE);
        _declareHelper& out(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, cv::InputOutputArray a4, int wtype = WARMUP_WRITE);

        _declareHelper& iterations(int n);
        _declareHelper& time(double timeLimitSecs);
    private:
        TestBase* test;
        _declareHelper(TestBase* t);
        _declareHelper(const _declareHelper&);
        _declareHelper& operator=(const _declareHelper&);
        friend class TestBase;
    };
    friend class _declareHelper;

public:
    _declareHelper declare;
};

template<typename T> class TestBaseWithParam: public TestBase, public ::testing::WithParamInterface<T> {};

typedef std::tr1::tuple<cv::Size, MatType> Size_MatType_t;
typedef TestBaseWithParam<Size_MatType_t> Size_MatType;

/*****************************************************************************************\
*                              Print functions for googletest                             *
\*****************************************************************************************/
CV_EXPORTS void PrintTo(const MatType& t, std::ostream* os);

} //namespace perf

namespace cv
{

CV_EXPORTS void PrintTo(const Size& sz, ::std::ostream* os);

} //namespace cv


/*****************************************************************************************\
*                        Macro definitions for performance tests                          *
\*****************************************************************************************/
#define PERF_PROXY_NAMESPACE_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_perf_namespace_proxy

// Defines a performance test.
//
// The first parameter is the name of the test case, and the second
// parameter is the name of the test within the test case.
//
// The user should put his test code between braces after using this
// macro.  Example:
//
//   PERF_TEST(FooTest, InitializesCorrectly) {
//     Foo foo;
//     EXPECT_TRUE(foo.StatusIsOK());
//   }
#define PERF_TEST(test_case_name, test_name)\
    namespace PERF_PROXY_NAMESPACE_NAME_(test_case_name, test_name) {\
     class TestBase {/*compile error for this class means that you are trying to use perf::TestBase as a fixture*/};\
     class test_case_name : public ::perf::TestBase {\
      public:\
       test_case_name() {}\
      protected:\
       virtual void PerfTestBody();\
     };\
     TEST_F(test_case_name, test_name){ RunPerfTestBody(); }\
    }\
    void PERF_PROXY_NAMESPACE_NAME_(test_case_name, test_name)::test_case_name::PerfTestBody()

// Defines a performance test that uses a test fixture.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// A test fixture class must be declared earlier.  The user should put
// his test code between braces after using this macro.  Example:
//
//   class FooTest : public ::perf::TestBase {
//    protected:
//     virtual void SetUp() { TestBase::SetUp(); b_.AddElement(3); }
//
//     Foo a_;
//     Foo b_;
//   };
//
//   PERF_TEST_F(FooTest, InitializesCorrectly) {
//     EXPECT_TRUE(a_.StatusIsOK());
//   }
//
//   PERF_TEST_F(FooTest, ReturnsElementCountCorrectly) {
//     EXPECT_EQ(0, a_.size());
//     EXPECT_EQ(1, b_.size());
//   }
#define PERF_TEST_F(fixture, testname) \
    namespace PERF_PROXY_NAMESPACE_NAME_(fixture, testname) {\
     class TestBase {/*compile error for this class means that you are trying to use perf::TestBase as a fixture*/};\
     class fixture : public ::fixture {\
      public:\
       fixture() {}\
      protected:\
       virtual void PerfTestBody();\
     };\
     TEST_F(fixture, testname){ RunPerfTestBody(); }\
    }\
    void PERF_PROXY_NAMESPACE_NAME_(fixture, testname)::fixture::PerfTestBody()

// Defines a parametrized performance test.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// The user should put his test code between braces after using this
// macro.  Example:
//
//   typedef ::perf::TestBaseWithParam<cv::Size> FooTest;
//
//   PERF_TEST_P(FooTest, DoTestingRight, ::testing::Values(::perf::szVGA, ::perf::sz720p) {
//     cv::Mat b(GetParam(), CV_8U, cv::Scalar(10));
//     cv::Mat a(GetParam(), CV_8U, cv::Scalar(20));
//     cv::Mat c(GetParam(), CV_8U, cv::Scalar(0));
//
//     declare.in(a, b).out(c).time(0.5);
//
//     SIMPLE_TEST_CYCLE() cv::add(a, b, c);
//
//     SANITY_CHECK(c);
//   }
#define PERF_TEST_P(fixture, name, params)  \
    class fixture##_##name : public ::fixture {\
     public:\
      fixture##_##name() {}\
     protected:\
      virtual void PerfTestBody();\
    };\
    TEST_P(fixture##_##name, name /*perf*/){ RunPerfTestBody(); }\
    INSTANTIATE_TEST_CASE_P(/*none*/, fixture##_##name, params);\
    void fixture##_##name::PerfTestBody()


#define CV_PERF_TEST_MAIN(testsuitname) \
int main(int argc, char **argv)\
{\
    ::perf::Regression::Init(#testsuitname);\
    ::perf::TestBase::Init(argc, argv);\
    ::testing::InitGoogleTest(&argc, argv);\
    return RUN_ALL_TESTS();\
}

#define TEST_CYCLE(n) for(declare.iterations(n); startTimer(), next(); stopTimer())
#define SIMPLE_TEST_CYCLE() for(; startTimer(), next(); stopTimer())

//flags
namespace perf
{
//GTEST_DECLARE_int32_(allowed_outliers);
} //namespace perf

#endif //__OPENCV_TS_PERF_HPP__
