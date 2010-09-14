/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __CXTS_H__
#define __CXTS_H__

#include "cxcore.h"
#include "cxmisc.h"
#include <assert.h>
#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <string>

#if _MSC_VER >= 1200
#pragma warning( disable: 4710 )
#endif

#define CV_TS_VERSION "CxTest 0.1"

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

// Helper class for growing vector (to avoid dependency from STL)
template < typename T > class CvTestVec
{
public:
    CvTestVec() { _max_size = _size = 0; _buf = 0; }
    ~CvTestVec() { delete[] _buf; }
    T& operator []( int i ) { assert( (unsigned)i < (unsigned)_size ); return _buf[i]; }
    T at( int i ) { assert( (unsigned)i < (unsigned)_size ); return _buf[i]; }
    T pop() { assert( _size > 0 ); return _buf[--_size]; }
    void push( const T& elem )
    {
        if( _size >= _max_size )
        {
            int i, _new_size = _max_size < 16 ? 16 : _max_size*3/2;
            T* temp = new T[_new_size];
            for( i = 0; i < _size; i++ )
                temp[i] = _buf[i];
            delete[] _buf;
            _max_size = _new_size;
            _buf = temp;
        }
        _buf[_size++] = elem;
    }

    int size() { return _size; }
    T* data() { return _buf; }
    void clear() { _size = 0; }

protected:
    T* _buf;
    int _size, _max_size;
};

/*****************************************************************************************\
*                                    Base class for tests                                 *
\*****************************************************************************************/

class CvTest;
class CvTS;

class CV_EXPORTS CvTest
{
public:
    // constructor(s) and destructor
    CvTest( const char* test_name, const char* test_funcs, const char* test_descr = "" );
    virtual ~CvTest();

    virtual int init( CvTS* system );

    // writes default parameters to file storage
    virtual int write_defaults(CvTS* ts);

    // the main procedure of the test
    virtual void run( int start_from );

    // the wrapper for run that cares of exceptions
    virtual void safe_run( int start_from );

    const char* get_name() const { return name; }
    const char* get_func_list() const { return tested_functions; }
    const char* get_description() const { return description; }
    const char* get_group_name( char* buffer ) const;
    CvTest* get_next() { return next; }
    static CvTest* get_first_test();
    static const char* get_parent_name( const char* name, char* buffer );

    // returns true if and only if the different test cases do not depend on each other
    // (so that test system could get right to a problematic test case)
    virtual bool can_do_fast_forward();

    // deallocates all the memory.
    // called by init() (before initialization) and by the destructor
    virtual void clear();

    // returns the testing modes supported by the particular test
    int get_support_testing_modes();

    enum { TIMING_EXTRA_PARAMS=5 };

protected:
    static CvTest* first;
    static CvTest* last;
    static int test_count;
    CvTest* next;

    const char** default_timing_param_names; // the names of timing parameters to write
    const CvFileNode* timing_param_names; // and the read param names
    const CvFileNode** timing_param_current; // the current tuple of timing parameters
    const CvFileNode** timing_param_seqs; // the array of parameter sequences
    int* timing_param_idxs; // the array of indices
    int timing_param_count; // the number of parameters in the tuple
    int support_testing_modes;

    int test_case_count; // the total number of test cases

    // called from write_defaults
    virtual int write_default_params(CvFileStorage* fs);

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

    // prints results of timing test
    virtual void print_time( int test_case_idx, double time_usecs, double time_cpu_clocks );

    // updates progress bar
    virtual int update_progress( int progress, int test_case_idx, int count, double dt );

    // finds test parameter
    const CvFileNode* find_param( CvFileStorage* fs, const char* param_name );

    // writes parameters
    void write_param( CvFileStorage* fs, const char* paramname, int val );
    void write_param( CvFileStorage* fs, const char* paramname, double val );
    void write_param( CvFileStorage* fs, const char* paramname, const char* val );
    void write_string_list( CvFileStorage* fs, const char* paramname, const char** val, int count=-1 );
    void write_int_list( CvFileStorage* fs, const char* paramname, const int* val,
                         int count, int stop_value=INT_MIN );
    void write_real_list( CvFileStorage* fs, const char* paramname, const double* val,
                          int count, double stop_value=DBL_MIN );
    void start_write_param( CvFileStorage* fs );

    // returns the specified parameter from the current parameter tuple
    const CvFileNode* find_timing_param( const char* paramname );

    // gets the next tuple of timing parameters
    int get_next_timing_param_tuple();

    // name of the test (it is possible to locate a test by its name)
    const char* name;

    // comma-separated list of functions that are invoked
    // (and, thus, tested explicitly or implicitly) by the test
    // methods of classes can be grouped using {}.
    // a few examples:
    //    "cvCanny, cvAdd, cvSub, cvMul"
    //    "CvImage::{Create, CopyOf}, cvMatMulAdd, CvCalibFilter::{PushFrame, SetCameraCount}"
    const char* tested_functions;

    // description of the test
    const char* description;

    // pointer to the system that includes the test
    CvTS* ts;

    int hdr_state;
};


/*****************************************************************************************\
*                               Information about a failed test                           *
\*****************************************************************************************/

typedef struct CvTestInfo
{
    // pointer to the test
    CvTest* test;

    // failure code (CV_FAIL*)
    int code;

    // seed value right before the data for the failed test case is prepared.
    uint64 rng_seed;
    
    // seed value right before running the test
    uint64 rng_seed0;

    // index of test case, can be then passed to CvTest::proceed_to_test_case()
    int test_case_idx;

    // index of the corrupted or leaked block
    int alloc_index;

    // index of the first block in the group
    // (used to adjust alloc_index when some test/test cases are skipped).
    int base_alloc_index;
}
CvTestInfo;

/*****************************************************************************************\
*                                 Base Class for test system                              *
\*****************************************************************************************/

class CvTestMemoryManager;

typedef CvTestVec<int> CvTestIntVec;
typedef CvTestVec<void*> CvTestPtrVec;
typedef CvTestVec<CvTestInfo> CvTestInfoVec;

class CV_EXPORTS CvTS
{
public:

    // constructor(s) and destructor
    CvTS();
    virtual ~CvTS();

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

    // low-level printing functions that are used by individual tests and by the system itself
    virtual void printf( int streams, const char* fmt, ... );
    virtual void vprintf( int streams, const char* fmt, va_list arglist );

    // runs the tests (the whole set or some selected tests)
    virtual int run( int argc, char** argv, const char** blacklist=0 );

    // updates the context: current test, test case, rng state
    virtual void update_context( CvTest* test, int test_case_idx, bool update_ts_context );

    const CvTestInfo* get_current_test_info() { return &current_test_info; }

    // sets information about a failed test
    virtual void set_failed_test_info( int fail_code, int alloc_index = -1 );

    // types of tests
    enum
    {
        CORRECTNESS_CHECK_MODE = 1,
        TIMING_MODE = 2
    };

    // the modes of timing tests:
    enum { AVG_TIME = 1, MIN_TIME = 2 };

    // test error codes
    enum
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

        // unexpected responce on passing bad arguments to the tested function
        // (the function crashed, proceed succesfully (while it should not), or returned
        // error code that is different from what is expected)
        FAIL_BAD_ARG_CHECK=-14,

        // the test data (in whole or for the particular test case) is invalid
        FAIL_INVALID_TEST_DATA=-15,

        // the test has been skipped because it is not in the selected subset of the tests to run,
        // because it has been run already within the same run with the same parameters, or because
        // of some other reason and this is not considered as an error.
        // Normally CvTS::run() (or overrided method in the derived class) takes care of what
        // needs to be run, so this code should not occur.
        SKIPPED=1
    };

    // get file storage
    CvFileStorage* get_file_storage() { return fs; }

    // get RNG to generate random input data for a test
    CvRNG* get_rng() { return &rng; }

    // returns the current error code
    int get_err_code() { return current_test_info.code; }

    // retrieves the first registered test
    CvTest* get_first_test() { return CvTest::get_first_test(); }

    // retrieves one of global options of the test system
    int is_debug_mode() { return params.debug_mode; }

    // returns the current testing mode
    int get_testing_mode()  { return params.test_mode; }

    // returns the current timing mode
    int get_timing_mode() { return params.timing_mode; }

    // returns the test extensivity scale
    double get_test_case_count_scale() { return params.test_case_count_scale; }

    int find_written_param( CvTest* test, const char* paramname,
                            int valtype, const void* val );

    const char* get_data_path() { return params.data_path ? params.data_path : ""; }

protected:
    // deallocates memory buffers and closes all the streams;
    // called by init() and from destructor. It does not remove any tests!!!
    virtual void clear();

    // retrieves information about the test libraries (names, versions, build dates etc.)
    virtual const char* get_libs_info( const char** loaded_ipp_modules );

    // returns textual description of failure code
    virtual const char* str_from_code( int code );

    // prints header of summary of test suite run.
    // It goes before the results of individual tests and contains information about tested libraries
    // (as reported by get_libs_info()), information about test environment (CPU, test machine name),
    // date and time etc.
    virtual void print_summary_header( int streams );

    // prints tailer of summary of test suite run.
    // it goes after the results of individual tests and contains the number of
    // failed tests, total running time, exit code (whether the system has been crashed,
    // interrupted by the user etc.), names of files with additional information etc.
    virtual void print_summary_tailer( int streams );

    // reads common parameters of the test system; called from init()
    virtual int read_params( CvFileStorage* fs );
    
    // checks, whether the test needs to be run (1) or not (0); called from run()
    virtual int filter( CvTest* test, const char** blacklist=0 );

    // makes base name of output files
    virtual void make_output_stream_base_name( const char* config_name );

    // forms default test configuration file that can be
    // customized further
    virtual void write_default_params( CvFileStorage* fs );

    // enables/disables the specific output stream[s]
    virtual void enable_output_streams( int streams, int flag );

    // sets memory and exception handlers
    virtual void set_handlers( bool on );

    // changes the path to test data files
    virtual void set_data_path( const char* data_path );

    // prints the information about command-line parameters
    virtual void print_help();
    
    // changes the text color in console
    virtual void set_color(int color);

    // a sequence of tests to run
    CvTestPtrVec* selected_tests;

    // a sequence of written test params
    CvTestPtrVec* written_params;

    // a sequence of failed tests
    CvTestInfoVec* failed_tests;

    // base name for output streams
    char* ostrm_base_name;
    const char* ostrm_suffixes[MAX_IDX];

    // parameters that can be read from file storage
    CvFileStorage* fs;

    enum { CHOOSE_TESTS = 0, CHOOSE_FUNCTIONS = 1 };

    // common parameters:
    struct
    {
        // if non-zero, the tests are run in unprotected mode to debug possible crashes,
        // otherwise the system tries to catch the exceptions and continue with other tests
        int debug_mode;

        // if non-zero, the header is not print
        bool skip_header;

        // if non-zero, the system includes only failed tests into summary
        bool print_only_failed;

        // rerun failed tests in debug mode
        bool rerun_failed;

        // if non-zero, the failed tests are rerun immediately
        bool rerun_immediately;

        // choose_tests or choose_functions;
        int  test_filter_mode;

        // correctness or performance [or bad-arg, stress etc.]
        int  test_mode;

        // timing mode
        int  timing_mode;

        // pattern for choosing tests
        const char* test_filter_pattern;

        // RNG seed, passed to and updated by every test executed.
        uint64 rng_seed;

        // relative or absolute path of directory containing subfolders with test data
        const char* resource_path;

        // whether to use IPP, MKL etc. or not
        int use_optimized;

        // extensivity of the tests, scale factor for test_case_count
        double test_case_count_scale;

        // the path to data files used by tests
        char* data_path;
        
        // whether the output to console should be colored
        int color_terminal;
    }
    params;

    // these are allocated within a test to try keep them valid in case of stack corruption
    CvRNG rng;

    // test system start time
    time_t start_time;

    // test system version (=CV_TS_VERSION by default)
    const char* version;

    // name of config file
    const char* config_name;

    // information about the current test
    CvTestInfo current_test_info;

    // memory manager used to detect memory corruptions and leaks
    CvTestMemoryManager* memory_manager;

    // output streams
    struct StreamInfo
    {
        FILE* f;
        //const char* filename;
        int default_handle; // for stderr
        int enable;
    };

    StreamInfo output_streams[MAX_IDX];
    int ostream_testname_mask;
    std::string logbuf;
};


/*****************************************************************************************\
*            Subclass of CvTest for testing functions that process dense arrays           *
\*****************************************************************************************/

class CV_EXPORTS CvArrTest : public CvTest
{
public:
    // constructor(s) and destructor
    CvArrTest( const char* test_name, const char* test_funcs, const char* test_descr = "" );
    virtual ~CvArrTest();

    virtual int write_default_params( CvFileStorage* fs );
    virtual void clear();

protected:

    virtual int read_params( CvFileStorage* fs );
    virtual int prepare_test_case( int test_case_idx );
    virtual int validate_test_results( int test_case_idx );

    virtual void prepare_to_validation( int test_case_idx );
    virtual void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    virtual void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                        CvSize** whole_sizes, bool *are_images );
    virtual void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    virtual void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    virtual double get_success_error_level( int test_case_idx, int i, int j );
    virtual void print_time( int test_case_idx, double time_usecs, double time_cpu_clocks );
    virtual void print_timing_params( int test_case_idx, char* ptr, int params_left=TIMING_EXTRA_PARAMS );

    bool cvmat_allowed;
    bool iplimage_allowed;
    bool optional_mask;
    bool element_wise_relative_error;

    int min_log_array_size;
    int max_log_array_size;

    int max_arr; // = MAX_ARR by default, the number of different types of arrays
    int max_hdr; // size of header buffer
    enum { INPUT, INPUT_OUTPUT, OUTPUT, REF_INPUT_OUTPUT, REF_OUTPUT, TEMP, MASK, MAX_ARR };

    const CvSize* size_list;
    const CvSize* whole_size_list;
    const int* depth_list;
    const int* cn_list;

    CvTestPtrVec* test_array;
    CvMat* test_mat[MAX_ARR];
    CvMat* hdr;
    float buf[4];
};


class CV_EXPORTS CvBadArgTest : public CvTest
{
public:
    // constructor(s) and destructor
    CvBadArgTest( const char* test_name, const char* test_funcs, const char* test_descr = "" );
    virtual ~CvBadArgTest();

protected:
    virtual int run_test_case( int expected_code, const char* descr );
    virtual void run_func(void) = 0;
    int test_case_idx;
    int progress;
    double t, freq;   

    template<class F>
    int run_test_case( int expected_code, const char* descr, F f)
    {
        double new_t = (double)cv::getTickCount(), dt;
        if( test_case_idx < 0 )
        {
            test_case_idx = 0;
            progress = 0;
            dt = 0;
        }
        else
        {
            dt = (new_t - t)/(freq*1000);
            t = new_t;
        }
        progress = update_progress(progress, test_case_idx, 0, dt);
        
        int errcount = 0;
        bool thrown = false;
        if(!descr)
            descr = "";

        try
        {
            f();
        }
        catch(const cv::Exception& e)
        {
            thrown = true;
            if( e.code != expected_code )
            {
                ts->printf(CvTS::LOG, "%s (test case #%d): the error code %d is different from the expected %d\n",
                    descr, test_case_idx, e.code, expected_code);
                errcount = 1;
            }
        }
        catch(...)
        {
            thrown = true;
            ts->printf(CvTS::LOG, "%s  (test case #%d): unknown exception was thrown (the function has likely crashed)\n",
                       descr, test_case_idx);
            errcount = 1;
        }
        if(!thrown)
        {
            ts->printf(CvTS::LOG, "%s  (test case #%d): no expected exception was thrown\n",
                       descr, test_case_idx);
            errcount = 1;
        }
        test_case_idx++;
        
        return errcount;
    }
};

/****************************************************************************************\
*                                 Utility Functions                                      *
\****************************************************************************************/

CV_EXPORTS const char* cvTsGetTypeName( int type );
CV_EXPORTS int cvTsTypeByName( const char* type_name );

inline  int cvTsClipInt( int val, int min_val, int max_val )
{
    if( val < min_val )
        val = min_val;
    if( val > max_val )
        val = max_val;
    return val;
}

// return min & max values for given type, e.g. for CV_8S ~  -128 and 127, respectively.
CV_EXPORTS double cvTsMinVal( int type );
CV_EXPORTS double cvTsMaxVal( int type );

// returns c-norm of the array
CV_EXPORTS double cvTsMaxVal( const CvMat* arr );

inline CvMat* cvTsGetMat( const CvMat* arr, CvMat* stub, int* coi=0 )
{
    return cvGetMat( arr, stub, coi );
}

// fills array with random numbers
CV_EXPORTS void cvTsRandUni( CvRNG* rng, CvMat* a, CvScalar param1, CvScalar param2 );

inline  unsigned cvTsRandInt( CvRNG* rng )
{
    uint64 temp = *rng;
    temp = (uint64)(unsigned)temp*1554115554 + (temp >> 32);
    *rng = temp;
    return (unsigned)temp;
}

inline  double cvTsRandReal( CvRNG* rng )
{
    return cvTsRandInt( rng ) * 2.3283064365386962890625e-10 /* 2^-32 */;
}

// fills c with zeros
CV_EXPORTS void cvTsZero( CvMat* c, const CvMat* mask=0 );

// initializes scaled identity matrix
CV_EXPORTS void cvTsSetIdentity( CvMat* c, CvScalar diag_value );

// copies a to b (whole matrix or only the selected region)
CV_EXPORTS void cvTsCopy( const CvMat* a, CvMat* b, const CvMat* mask=0 );

// converts one array to another
CV_EXPORTS void  cvTsConvert( const CvMat* src, CvMat* dst );

// working with multi-channel arrays
CV_EXPORTS void cvTsExtract( const CvMat* a, CvMat* plane, int coi );
CV_EXPORTS void cvTsInsert( const CvMat* plane, CvMat* a, int coi );

// c = alpha*a + beta*b + gamma
CV_EXPORTS void cvTsAdd( const CvMat* a, CvScalar alpha, const CvMat* b, CvScalar beta,
                    CvScalar gamma, CvMat* c, int calc_abs );

// c = a*b*alpha
CV_EXPORTS void cvTsMul( const CvMat* _a, const CvMat* _b, CvScalar alpha, CvMat* _c );

// c = a*alpha/b
CV_EXPORTS void cvTsDiv( const CvMat* _a, const CvMat* _b, CvScalar alpha, CvMat* _c );

enum { CV_TS_MIN = 0, CV_TS_MAX = 1 };

// min/max
CV_EXPORTS void cvTsMinMax( const CvMat* _a, const CvMat* _b, CvMat* _c, int op_type );
CV_EXPORTS void cvTsMinMaxS( const CvMat* _a, double scalar, CvMat* _c, int op_type );

// checks that the array does not have NaNs and/or Infs and all the elements are
// within [min_val,max_val). idx is the index of the first "bad" element.
CV_EXPORTS int cvTsCheck( const CvMat* data, double min_val, double max_val, CvPoint* idx );

// compares two arrays. max_diff is the maximum actual difference,
// success_err_level is maximum allowed difference, idx is the index of the first
// element for which difference is >success_err_level
// (or index of element with the maximum difference)
CV_EXPORTS int cvTsCmpEps( const CvMat* data, const CvMat* etalon, double* max_diff,
                      double success_err_level, CvPoint* idx,
                      bool element_wise_relative_error );

// a wrapper for the previous function. in case of error prints the message to log file.
CV_EXPORTS int cvTsCmpEps2( CvTS* ts, const CvArr* _a, const CvArr* _b, double success_err_level,
                            bool element_wise_relative_error, const char* desc );

CV_EXPORTS int cvTsCmpEps2_64f( CvTS* ts, const double* val, const double* ref_val, int len,
                                double eps, const char* param_name );

// compares two arrays. the result is 8s image that takes values -1, 0, 1
CV_EXPORTS void cvTsCmp( const CvMat* a, const CvMat* b, CvMat* result, int cmp_op );

// compares array and a scalar.
CV_EXPORTS void cvTsCmpS( const CvMat* a, double fval, CvMat* result, int cmp_op );

// retrieves C, L1 or L2 norm of array or its region
CV_EXPORTS double cvTsNorm( const CvMat* _arr, const CvMat* _mask, int norm_type, int coi );

// retrieves mean, standard deviation and the number of nonzero mask pixels
CV_EXPORTS int cvTsMeanStdDevNonZero( const CvMat* _arr, const CvMat* _mask,
                           CvScalar* _mean, CvScalar* _stddev, int coi );

// retrieves global extremums and their positions
CV_EXPORTS void cvTsMinMaxLoc( const CvMat* _arr, const CvMat* _mask,
                    double* _minval, double* _maxval,
                    CvPoint* _minidx, CvPoint* _maxidx, int coi );

enum { CV_TS_LOGIC_AND = 0, CV_TS_LOGIC_OR = 1, CV_TS_LOGIC_XOR = 2, CV_TS_LOGIC_NOT = 3 };

CV_EXPORTS void cvTsLogic( const CvMat* a, const CvMat* b, CvMat* c, int logic_op );
CV_EXPORTS void cvTsLogicS( const CvMat* a, CvScalar s, CvMat* c, int logic_op );

enum { CV_TS_GEMM_A_T = 1, CV_TS_GEMM_B_T = 2, CV_TS_GEMM_C_T = 4 };

CV_EXPORTS void cvTsGEMM( const CvMat* a, const CvMat* b, double alpha,
                     const CvMat* c, double beta, CvMat* d, int flags );

CV_EXPORTS void cvTsConvolve2D( const CvMat* a, CvMat* b, const CvMat* kernel, CvPoint anchor );
// op_type == CV_TS_MIN/CV_TS_MAX
CV_EXPORTS void cvTsMinMaxFilter( const CvMat* a, CvMat* b,
                                  const IplConvKernel* element, int op_type );

enum { CV_TS_BORDER_REPLICATE=0, CV_TS_BORDER_REFLECT=1, CV_TS_BORDER_FILL=2 };

CV_EXPORTS void cvTsPrepareToFilter( const CvMat* a, CvMat* b, CvPoint ofs,
                                     int border_mode = CV_TS_BORDER_REPLICATE,
                                     CvScalar fill_val=cvScalarAll(0));

CV_EXPORTS double cvTsCrossCorr( const CvMat* a, const CvMat* b );

CV_EXPORTS CvMat* cvTsSelect( const CvMat* a, CvMat* header, CvRect rect );

CV_EXPORTS CvMat* cvTsTranspose( const CvMat* a, CvMat* b );
CV_EXPORTS void cvTsFlip( const CvMat* a, CvMat* b, int flip_type );

CV_EXPORTS void cvTsTransform( const CvMat* a, CvMat* b, const CvMat* transmat, const CvMat* shift );

// modifies values that are close to zero
CV_EXPORTS void  cvTsPatchZeros( CvMat* mat, double level );

#endif/*__CXTS_H__*/

