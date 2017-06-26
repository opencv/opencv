#ifndef OPENCV_TS_HPP
#define OPENCV_TS_HPP

#ifndef __OPENCV_TESTS
#define __OPENCV_TESTS 1
#endif

#include "opencv2/opencv_modules.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/core/utility.hpp"

#include "opencv2/core/utils/trace.hpp"

#include <stdarg.h> // for va_list

#include "cvconfig.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <limits>
#include <numeric>

#ifdef WINRT
    #pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
#endif

#ifdef _MSC_VER
#pragma warning( disable: 4127 ) // conditional expression is constant
#pragma warning( disable: 4503 ) // decorated name length exceeded, name was truncated
#endif

#define GTEST_DONT_DEFINE_FAIL      0
#define GTEST_DONT_DEFINE_SUCCEED   0
#define GTEST_DONT_DEFINE_ASSERT_EQ 0
#define GTEST_DONT_DEFINE_ASSERT_NE 0
#define GTEST_DONT_DEFINE_ASSERT_LE 0
#define GTEST_DONT_DEFINE_ASSERT_LT 0
#define GTEST_DONT_DEFINE_ASSERT_GE 0
#define GTEST_DONT_DEFINE_ASSERT_GT 0
#define GTEST_DONT_DEFINE_TEST      0

#include "opencv2/ts/ts_gtest.h"
#include "opencv2/ts/ts_ext.hpp"

#ifndef GTEST_USES_SIMPLE_RE
#  define GTEST_USES_SIMPLE_RE 0
#endif
#ifndef GTEST_USES_POSIX_RE
#  define GTEST_USES_POSIX_RE 0
#endif

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >
#define GET_PARAM(k) std::tr1::get< k >(GetParam())

namespace cvtest
{

using std::vector;
using std::string;
using namespace cv;
using testing::Values;
using testing::Combine;


class SkipTestException: public cv::Exception
{
public:
    int dummy; // workaround for MacOSX Xcode 7.3 bug (don't make class "empty")
    SkipTestException() : dummy(0) {}
    SkipTestException(const cv::String& message) : dummy(0) { this->msg = message; }
};

class CV_EXPORTS TS;

CV_EXPORTS int64 readSeed(const char* str);

CV_EXPORTS void randUni( RNG& rng, Mat& a, const Scalar& param1, const Scalar& param2 );

inline unsigned randInt( RNG& rng )
{
    return (unsigned)rng;
}

inline  double randReal( RNG& rng )
{
    return (double)rng;
}


CV_EXPORTS const char* getTypeName( int type );
CV_EXPORTS int typeByName( const char* type_name );

CV_EXPORTS string vec2str(const string& sep, const int* v, size_t nelems);

inline int clipInt( int val, int min_val, int max_val )
{
    if( val < min_val )
        val = min_val;
    if( val > max_val )
        val = max_val;
    return val;
}

CV_EXPORTS double getMinVal(int depth);
CV_EXPORTS double getMaxVal(int depth);

CV_EXPORTS Size randomSize(RNG& rng, double maxSizeLog);
CV_EXPORTS void randomSize(RNG& rng, int minDims, int maxDims, double maxSizeLog, vector<int>& sz);
CV_EXPORTS int randomType(RNG& rng, int typeMask, int minChannels, int maxChannels);
CV_EXPORTS Mat randomMat(RNG& rng, Size size, int type, double minVal, double maxVal, bool useRoi);
CV_EXPORTS Mat randomMat(RNG& rng, const vector<int>& size, int type, double minVal, double maxVal, bool useRoi);
CV_EXPORTS void add(const Mat& a, double alpha, const Mat& b, double beta,
                      Scalar gamma, Mat& c, int ctype, bool calcAbs=false);
CV_EXPORTS void multiply(const Mat& a, const Mat& b, Mat& c, double alpha=1);
CV_EXPORTS void divide(const Mat& a, const Mat& b, Mat& c, double alpha=1);

CV_EXPORTS void convert(const Mat& src, cv::OutputArray dst, int dtype, double alpha=1, double beta=0);
CV_EXPORTS void copy(const Mat& src, Mat& dst, const Mat& mask=Mat(), bool invertMask=false);
CV_EXPORTS void set(Mat& dst, const Scalar& gamma, const Mat& mask=Mat());

// working with multi-channel arrays
CV_EXPORTS void extract( const Mat& a, Mat& plane, int coi );
CV_EXPORTS void insert( const Mat& plane, Mat& a, int coi );

// checks that the array does not have NaNs and/or Infs and all the elements are
// within [min_val,max_val). idx is the index of the first "bad" element.
CV_EXPORTS int check( const Mat& data, double min_val, double max_val, vector<int>* idx );

// modifies values that are close to zero
CV_EXPORTS void  patchZeros( Mat& mat, double level );

CV_EXPORTS void transpose(const Mat& src, Mat& dst);
CV_EXPORTS void erode(const Mat& src, Mat& dst, const Mat& _kernel, Point anchor=Point(-1,-1),
                      int borderType=0, const Scalar& borderValue=Scalar());
CV_EXPORTS void dilate(const Mat& src, Mat& dst, const Mat& _kernel, Point anchor=Point(-1,-1),
                       int borderType=0, const Scalar& borderValue=Scalar());
CV_EXPORTS void filter2D(const Mat& src, Mat& dst, int ddepth, const Mat& kernel,
                         Point anchor, double delta, int borderType,
                         const Scalar& borderValue=Scalar());
CV_EXPORTS void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right,
                               int borderType, const Scalar& borderValue=Scalar());
CV_EXPORTS Mat calcSobelKernel2D( int dx, int dy, int apertureSize, int origin=0 );
CV_EXPORTS Mat calcLaplaceKernel2D( int aperture_size );

CV_EXPORTS void initUndistortMap( const Mat& a, const Mat& k, Size sz, Mat& mapx, Mat& mapy );

CV_EXPORTS void minMaxLoc(const Mat& src, double* minval, double* maxval,
                          vector<int>* minloc, vector<int>* maxloc, const Mat& mask=Mat());
CV_EXPORTS double norm(InputArray src, int normType, InputArray mask=noArray());
CV_EXPORTS double norm(InputArray src1, InputArray src2, int normType, InputArray mask=noArray());
CV_EXPORTS Scalar mean(const Mat& src, const Mat& mask=Mat());
CV_EXPORTS double PSNR(InputArray src1, InputArray src2);

CV_EXPORTS bool cmpUlps(const Mat& data, const Mat& refdata, int expMaxDiff, double* realMaxDiff, vector<int>* idx);

// compares two arrays. max_diff is the maximum actual difference,
// success_err_level is maximum allowed difference, idx is the index of the first
// element for which difference is >success_err_level
// (or index of element with the maximum difference)
CV_EXPORTS int cmpEps( const Mat& data, const Mat& refdata, double* max_diff,
                       double success_err_level, vector<int>* idx,
                       bool element_wise_relative_error );

// a wrapper for the previous function. in case of error prints the message to log file.
CV_EXPORTS int cmpEps2( TS* ts, const Mat& data, const Mat& refdata, double success_err_level,
                        bool element_wise_relative_error, const char* desc );

CV_EXPORTS int cmpEps2_64f( TS* ts, const double* val, const double* refval, int len,
                        double eps, const char* param_name );

CV_EXPORTS void logicOp(const Mat& src1, const Mat& src2, Mat& dst, char c);
CV_EXPORTS void logicOp(const Mat& src, const Scalar& s, Mat& dst, char c);
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
CV_EXPORTS void min(const Mat& src, double s, Mat& dst);
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
CV_EXPORTS void max(const Mat& src, double s, Mat& dst);

CV_EXPORTS void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop);
CV_EXPORTS void compare(const Mat& src, double s, Mat& dst, int cmpop);
CV_EXPORTS void gemm(const Mat& src1, const Mat& src2, double alpha,
                     const Mat& src3, double beta, Mat& dst, int flags);
CV_EXPORTS void transform( const Mat& src, Mat& dst, const Mat& transmat, const Mat& shift );
CV_EXPORTS double crossCorr(const Mat& src1, const Mat& src2);
CV_EXPORTS void threshold( const Mat& src, Mat& dst, double thresh, double maxval, int thresh_type );
CV_EXPORTS void minMaxIdx( InputArray _img, double* minVal, double* maxVal,
                    Point* minLoc, Point* maxLoc, InputArray _mask );

struct CV_EXPORTS MatInfo
{
    MatInfo(const Mat& _m) : m(&_m) {}
    const Mat* m;
};

CV_EXPORTS std::ostream& operator << (std::ostream& out, const MatInfo& m);

struct CV_EXPORTS MatComparator
{
public:
    MatComparator(double maxdiff, int context);

    ::testing::AssertionResult operator()(const char* expr1, const char* expr2,
                                          const Mat& m1, const Mat& m2);

    double maxdiff;
    double realmaxdiff;
    vector<int> loc0;
    int context;
};



class BaseTest;
class TS;

class CV_EXPORTS BaseTest
{
public:
    // constructor(s) and destructor
    BaseTest();
    virtual ~BaseTest();

    // the main procedure of the test
    virtual void run( int start_from );

    // the wrapper for run that cares of exceptions
    virtual void safe_run( int start_from=0 );

    const string& get_name() const { return name; }

    // returns true if and only if the different test cases do not depend on each other
    // (so that test system could get right to a problematic test case)
    virtual bool can_do_fast_forward();

    // deallocates all the memory.
    // called by init() (before initialization) and by the destructor
    virtual void clear();

protected:
    int test_case_count; // the total number of test cases

    // read test params
    virtual int read_params( CvFileStorage* fs );

    // returns the number of tests or -1 if it is unknown a-priori
    virtual int get_test_case_count();

    // prepares data for the next test case. rng seed is updated by the function
    virtual int prepare_test_case( int test_case_idx );

    // checks if the test output is valid and accurate
    virtual int validate_test_results( int test_case_idx );

    // calls the tested function. the method is called from run_test_case()
    virtual void run_func(); // runs tested func(s)

    // updates progress bar
    virtual int update_progress( int progress, int test_case_idx, int count, double dt );

    // finds test parameter
    const CvFileNode* find_param( CvFileStorage* fs, const char* param_name );

    // name of the test (it is possible to locate a test by its name)
    string name;

    // pointer to the system that includes the test
    TS* ts;
};


/*****************************************************************************************\
*                               Information about a failed test                           *
\*****************************************************************************************/

struct TestInfo
{
    TestInfo();

    // pointer to the test
    BaseTest* test;

    // failure code (TS::FAIL_*)
    int code;

    // seed value right before the data for the failed test case is prepared.
    uint64 rng_seed;

    // seed value right before running the test
    uint64 rng_seed0;

    // index of test case, can be then passed to BaseTest::proceed_to_test_case()
    int test_case_idx;
};

/*****************************************************************************************\
*                                 Base Class for test system                              *
\*****************************************************************************************/

// common parameters:
struct CV_EXPORTS TSParams
{
    TSParams();

    // RNG seed, passed to and updated by every test executed.
    uint64 rng_seed;

    // whether to use IPP, MKL etc. or not
    bool use_optimized;

    // extensivity of the tests, scale factor for test_case_count
    double test_case_count_scale;
};


class CV_EXPORTS TS
{
public:
    // constructor(s) and destructor
    TS();
    virtual ~TS();

    enum
    {
        NUL=0,
        SUMMARY_IDX=0,
        SUMMARY=1 << SUMMARY_IDX,
        LOG_IDX=1,
        LOG=1 << LOG_IDX,
        CSV_IDX=2,
        CSV=1 << CSV_IDX,
        CONSOLE_IDX=3,
        CONSOLE=1 << CONSOLE_IDX,
        MAX_IDX=4
    };

    static TS* ptr();

    // initialize test system before running the first test
    virtual void init( const string& modulename );

    // low-level printing functions that are used by individual tests and by the system itself
    virtual void printf( int streams, const char* fmt, ... );
    virtual void vprintf( int streams, const char* fmt, va_list arglist );

    // updates the context: current test, test case, rng state
    virtual void update_context( BaseTest* test, int test_case_idx, bool update_ts_context );

    const TestInfo* get_current_test_info() { return &current_test_info; }

    // sets information about a failed test
    virtual void set_failed_test_info( int fail_code );

    virtual void set_gtest_status();

    // test error codes
    enum FailureCode
    {
        // everything is Ok
        OK=0,

        // generic error: stub value to be used
        // temporarily if the error's cause is unknown
        FAIL_GENERIC=-1,

        // the test is missing some essential data to proceed further
        FAIL_MISSING_TEST_DATA=-2,

        // the tested function raised an error via cxcore error handler
        FAIL_ERROR_IN_CALLED_FUNC=-3,

        // an exception has been raised;
        // for memory and arithmetic exception
        // there are two specialized codes (see below...)
        FAIL_EXCEPTION=-4,

        // a memory exception
        // (access violation, access to missed page, stack overflow etc.)
        FAIL_MEMORY_EXCEPTION=-5,

        // arithmetic exception (overflow, division by zero etc.)
        FAIL_ARITHM_EXCEPTION=-6,

        // the tested function corrupted memory (no exception have been raised)
        FAIL_MEMORY_CORRUPTION_BEGIN=-7,
        FAIL_MEMORY_CORRUPTION_END=-8,

        // the tested function (or test ifself) do not deallocate some memory
        FAIL_MEMORY_LEAK=-9,

        // the tested function returned invalid object, e.g. matrix, containing NaNs,
        // structure with NULL or out-of-range fields (while it should not)
        FAIL_INVALID_OUTPUT=-10,

        // the tested function returned valid object, but it does not match to
        // the original (or produced by the test) object
        FAIL_MISMATCH=-11,

        // the tested function returned valid object (a single number or numerical array),
        // but it differs too much from the original (or produced by the test) object
        FAIL_BAD_ACCURACY=-12,

        // the tested function hung. Sometimes, can be determined by unexpectedly long
        // processing time (in this case there should be possibility to interrupt such a function
        FAIL_HANG=-13,

        // unexpected response on passing bad arguments to the tested function
        // (the function crashed, proceed successfully (while it should not), or returned
        // error code that is different from what is expected)
        FAIL_BAD_ARG_CHECK=-14,

        // the test data (in whole or for the particular test case) is invalid
        FAIL_INVALID_TEST_DATA=-15,

        // the test has been skipped because it is not in the selected subset of the tests to run,
        // because it has been run already within the same run with the same parameters, or because
        // of some other reason and this is not considered as an error.
        // Normally TS::run() (or overridden method in the derived class) takes care of what
        // needs to be run, so this code should not occur.
        SKIPPED=1
    };

    // get file storage
    CvFileStorage* get_file_storage();

    // get RNG to generate random input data for a test
    RNG& get_rng() { return rng; }

    // returns the current error code
    TS::FailureCode get_err_code() { return TS::FailureCode(current_test_info.code); }

    // returns the test extensivity scale
    double get_test_case_count_scale() { return params.test_case_count_scale; }

    const string& get_data_path() const { return data_path; }

    // returns textual description of failure code
    static string str_from_code( const TS::FailureCode code );

    std::vector<std::string> data_search_path;
    std::vector<std::string> data_search_subdir;
protected:

    // these are allocated within a test to try keep them valid in case of stack corruption
    RNG rng;

    // information about the current test
    TestInfo current_test_info;

    // the path to data files used by tests
    string data_path;

    TSParams params;
    std::string output_buf[MAX_IDX];
};


/*****************************************************************************************\
*            Subclass of BaseTest for testing functions that process dense arrays           *
\*****************************************************************************************/

class CV_EXPORTS ArrayTest : public BaseTest
{
public:
    // constructor(s) and destructor
    ArrayTest();
    virtual ~ArrayTest();

    virtual void clear();

protected:

    virtual int read_params( CvFileStorage* fs );
    virtual int prepare_test_case( int test_case_idx );
    virtual int validate_test_results( int test_case_idx );

    virtual void prepare_to_validation( int test_case_idx );
    virtual void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    virtual void fill_array( int test_case_idx, int i, int j, Mat& arr );
    virtual void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    virtual double get_success_error_level( int test_case_idx, int i, int j );

    bool cvmat_allowed;
    bool iplimage_allowed;
    bool optional_mask;
    bool element_wise_relative_error;

    int min_log_array_size;
    int max_log_array_size;

    enum { INPUT, INPUT_OUTPUT, OUTPUT, REF_INPUT_OUTPUT, REF_OUTPUT, TEMP, MASK, MAX_ARR };

    vector<vector<void*> > test_array;
    vector<vector<Mat> > test_mat;
    float buf[4];
};


class CV_EXPORTS BadArgTest : public BaseTest
{
public:
    // constructor(s) and destructor
    BadArgTest();
    virtual ~BadArgTest();

protected:
    virtual int run_test_case( int expected_code, const string& descr );
    virtual void run_func(void) = 0;
    int test_case_idx;

    template<class F>
    int run_test_case( int expected_code, const string& _descr, F f)
    {
        int errcount = 0;
        bool thrown = false;
        const char* descr = _descr.c_str() ? _descr.c_str() : "";

        try
        {
            f();
        }
        catch(const cv::Exception& e)
        {
            thrown = true;
            if( e.code != expected_code )
            {
                ts->printf(TS::LOG, "%s (test case #%d): the error code %d is different from the expected %d\n",
                    descr, test_case_idx, e.code, expected_code);
                errcount = 1;
            }
        }
        catch(...)
        {
            thrown = true;
            ts->printf(TS::LOG, "%s  (test case #%d): unknown exception was thrown (the function has likely crashed)\n",
                       descr, test_case_idx);
            errcount = 1;
        }
        if(!thrown)
        {
            ts->printf(TS::LOG, "%s  (test case #%d): no expected exception was thrown\n",
                       descr, test_case_idx);
            errcount = 1;
        }
        test_case_idx++;

        return errcount;
    }
};

extern uint64 param_seed;

struct CV_EXPORTS DefaultRngAuto
{
    const uint64 old_state;

    DefaultRngAuto() : old_state(cv::theRNG().state) { cv::theRNG().state = cvtest::param_seed; }
    ~DefaultRngAuto() { cv::theRNG().state = old_state; }

    DefaultRngAuto& operator=(const DefaultRngAuto&);
};


// test images generation functions
CV_EXPORTS void fillGradient(Mat& img, int delta = 5);
CV_EXPORTS void smoothBorder(Mat& img, const Scalar& color, int delta = 3);

CV_EXPORTS void printVersionInfo(bool useStdOut = true);


// Utility functions

CV_EXPORTS void addDataSearchPath(const std::string& path);
CV_EXPORTS void addDataSearchSubDirectory(const std::string& subdir);

/*! @brief Try to find requested data file

  Search directories:

  0. TS::data_search_path (search sub-directories are not used)
  1. OPENCV_TEST_DATA_PATH environment variable
  2. One of these:
     a. OpenCV testdata based on build location: "./" + "share/OpenCV/testdata"
     b. OpenCV testdata at install location: CMAKE_INSTALL_PREFIX + "share/OpenCV/testdata"

  Search sub-directories:

  - addDataSearchSubDirectory()
  - modulename from TS::init()

 */
CV_EXPORTS std::string findDataFile(const std::string& relative_path, bool required = true);


#ifndef __CV_TEST_EXEC_ARGS
#if defined(_MSC_VER) && (_MSC_VER <= 1400)
#define __CV_TEST_EXEC_ARGS(...)    \
    while (++argc >= (--argc,-1)) {__VA_ARGS__; break;} /*this ugly construction is needed for VS 2005*/
#else
#define __CV_TEST_EXEC_ARGS(...)    \
    __VA_ARGS__;
#endif
#endif

#ifdef HAVE_OPENCL
namespace ocl {
void dumpOpenCLDevice();
}
#define TEST_DUMP_OCL_INFO cvtest::ocl::dumpOpenCLDevice();
#else
#define TEST_DUMP_OCL_INFO
#endif

void parseCustomOptions(int argc, char **argv);

#define CV_TEST_INIT0_NOOP (void)0

#define CV_TEST_MAIN(resourcesubdir, ...) CV_TEST_MAIN_EX(resourcesubdir, NOOP, __VA_ARGS__)

#define CV_TEST_MAIN_EX(resourcesubdir, INIT0, ...) \
int main(int argc, char **argv) \
{ \
    CV_TRACE_FUNCTION(); \
    { CV_TRACE_REGION("INIT"); \
    using namespace cvtest; \
    TS* ts = TS::ptr(); \
    ts->init(resourcesubdir); \
    __CV_TEST_EXEC_ARGS(CV_TEST_INIT0_ ## INIT0) \
    ::testing::InitGoogleTest(&argc, argv); \
    cvtest::printVersionInfo(); \
    TEST_DUMP_OCL_INFO \
    __CV_TEST_EXEC_ARGS(__VA_ARGS__) \
    parseCustomOptions(argc, argv); \
    } \
    return RUN_ALL_TESTS(); \
}

// This usually only makes sense in perf tests with several implementations,
// some of which are not available.
#define CV_TEST_FAIL_NO_IMPL() do { \
    ::testing::Test::RecordProperty("custom_status", "noimpl"); \
    FAIL() << "No equivalent implementation."; \
} while (0)

} //namespace cvtest

#include "opencv2/ts/ts_perf.hpp"

namespace cvtest {
using perf::MatDepth;
}

#ifdef WINRT
#ifndef __FSTREAM_EMULATED__
#define __FSTREAM_EMULATED__
#include <stdlib.h>
#include <fstream>
#include <sstream>

#undef ifstream
#undef ofstream
#define ifstream ifstream_emulated
#define ofstream ofstream_emulated

namespace std {

class ifstream : public stringstream
{
    FILE* f;
public:
    ifstream(const char* filename, ios_base::openmode mode = ios_base::in)
        : f(NULL)
    {
        string modeStr("r");
        printf("Open file (read): %s\n", filename);
        if (mode & ios_base::binary)
            modeStr += "b";
        f = fopen(filename, modeStr.c_str());

        if (f == NULL)
        {
            printf("Can't open file: %s\n", filename);
            return;
        }
        fseek(f, 0, SEEK_END);
        size_t sz = ftell(f);
        if (sz > 0)
        {
            char* buf = (char*) malloc(sz);
            fseek(f, 0, SEEK_SET);
            if (fread(buf, 1, sz, f) == sz)
            {
                this->str(std::string(buf, sz));
            }
            free(buf);
        }
    }

    ~ifstream() { close(); }
    bool is_open() const { return f != NULL; }
    void close()
    {
        if (f)
            fclose(f);
        f = NULL;
        this->str("");
    }
};

class ofstream : public stringstream
{
    FILE* f;
public:
    ofstream(const char* filename, ios_base::openmode mode = ios_base::out)
    : f(NULL)
    {
        open(filename, mode);
    }
    ~ofstream() { close(); }
    void open(const char* filename, ios_base::openmode mode = ios_base::out)
    {
        string modeStr("w+");
        if (mode & ios_base::trunc)
            modeStr = "w";
        if (mode & ios_base::binary)
            modeStr += "b";
        f = fopen(filename, modeStr.c_str());
        printf("Open file (write): %s\n", filename);
        if (f == NULL)
        {
            printf("Can't open file (write): %s\n", filename);
            return;
        }
    }
    bool is_open() const { return f != NULL; }
    void close()
    {
        if (f)
        {
            fwrite(reinterpret_cast<const char *>(this->str().c_str()), this->str().size(), 1, f);
            fclose(f);
        }
        f = NULL;
        this->str("");
    }
};

} // namespace std
#endif // __FSTREAM_EMULATED__
#endif // WINRT

#endif // OPENCV_TS_HPP
