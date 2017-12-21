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

#include "precomp.hpp"
#include "opencv2/core/core_c.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#if defined _WIN32
#include <io.h>

#include <windows.h>
#undef small
#undef min
#undef max
#undef abs

#ifdef _MSC_VER
#include <eh.h>
#endif

#else
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#endif

// isDirectory
#if defined _WIN32 || defined WINCE
# include <windows.h>
#else
# include <dirent.h>
# include <sys/stat.h>
#endif


#include "opencv_tests_config.hpp"

namespace cvtest
{

uint64 param_seed = 0x12345678; // real value is passed via parseCustomOptions function

static std::string path_join(const std::string& prefix, const std::string& subpath)
{
    CV_Assert(subpath.empty() || subpath[0] != '/');
    if (prefix.empty())
        return subpath;
    bool skipSlash = prefix.size() > 0 ? (prefix[prefix.size()-1] == '/' || prefix[prefix.size()-1] == '\\') : false;
    std::string path = prefix + (skipSlash ? "" : "/") + subpath;
    return path;
}



/*****************************************************************************************\
*                                Exception and memory handlers                            *
\*****************************************************************************************/

// a few platform-dependent declarations

#if defined _WIN32
#ifdef _MSC_VER
static void SEHTranslator( unsigned int /*u*/, EXCEPTION_POINTERS* pExp )
{
    TS::FailureCode code = TS::FAIL_EXCEPTION;
    switch( pExp->ExceptionRecord->ExceptionCode )
    {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case EXCEPTION_FLT_STACK_CHECK:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_IN_PAGE_ERROR:
        code = TS::FAIL_MEMORY_EXCEPTION;
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INEXACT_RESULT:
    case EXCEPTION_FLT_INVALID_OPERATION:
    case EXCEPTION_FLT_OVERFLOW:
    case EXCEPTION_FLT_UNDERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_INT_OVERFLOW:
        code = TS::FAIL_ARITHM_EXCEPTION;
        break;
    case EXCEPTION_BREAKPOINT:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_INVALID_DISPOSITION:
    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_SINGLE_STEP:
        code = TS::FAIL_EXCEPTION;
    }
    throw code;
}
#endif

#else

static const int tsSigId[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, -1 };

static jmp_buf tsJmpMark;

static void signalHandler( int sig_code )
{
    TS::FailureCode code = TS::FAIL_EXCEPTION;
    switch( sig_code )
    {
    case SIGFPE:
        code = TS::FAIL_ARITHM_EXCEPTION;
        break;
    case SIGSEGV:
    case SIGBUS:
        code = TS::FAIL_ARITHM_EXCEPTION;
        break;
    case SIGILL:
        code = TS::FAIL_EXCEPTION;
    }

    longjmp( tsJmpMark, (int)code );
}

#endif


// reads 16-digit hexadecimal number (i.e. 64-bit integer)
int64 readSeed( const char* str )
{
    int64 val = 0;
    if( str && strlen(str) == 16 )
    {
        for( int i = 0; str[i]; i++ )
        {
            int c = tolower(str[i]);
            if( !isxdigit(c) )
                return 0;
            val = val * 16 +
            (str[i] < 'a' ? str[i] - '0' : str[i] - 'a' + 10);
        }
    }
    return val;
}


/*****************************************************************************************\
*                                    Base Class for Tests                                 *
\*****************************************************************************************/

BaseTest::BaseTest()
{
    ts = TS::ptr();
    test_case_count = -1;
}

BaseTest::~BaseTest()
{
    clear();
}

void BaseTest::clear()
{
}


const CvFileNode* BaseTest::find_param( CvFileStorage* fs, const char* param_name )
{
    CvFileNode* node = cvGetFileNodeByName(fs, 0, get_name().c_str());
    return node ? cvGetFileNodeByName( fs, node, param_name ) : 0;
}


int BaseTest::read_params( CvFileStorage* )
{
    return 0;
}


bool BaseTest::can_do_fast_forward()
{
    return true;
}


void BaseTest::safe_run( int start_from )
{
    CV_TRACE_FUNCTION();
    read_params( ts->get_file_storage() );
    ts->update_context( 0, -1, true );
    ts->update_context( this, -1, true );

    if( !::testing::GTEST_FLAG(catch_exceptions) )
        run( start_from );
    else
    {
        try
        {
        #if !defined _WIN32
        int _code = setjmp( tsJmpMark );
        if( !_code )
            run( start_from );
        else
            throw TS::FailureCode(_code);
        #else
            run( start_from );
        #endif
        }
        catch (const cv::Exception& exc)
        {
            const char* errorStr = cvErrorStr(exc.code);
            char buf[1 << 16];

            sprintf( buf, "OpenCV Error:\n\t%s (%s) in %s, file %s, line %d",
                    errorStr, exc.err.c_str(), exc.func.size() > 0 ?
                    exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line );
            ts->printf(TS::LOG, "%s\n", buf);

            ts->set_failed_test_info( TS::FAIL_ERROR_IN_CALLED_FUNC );
        }
        catch (const TS::FailureCode& fc)
        {
            std::string errorStr = TS::str_from_code(fc);
            ts->printf(TS::LOG, "General failure:\n\t%s (%d)\n", errorStr.c_str(), fc);

            ts->set_failed_test_info( fc );
        }
        catch (...)
        {
            ts->printf(TS::LOG, "Unknown failure\n");

            ts->set_failed_test_info( TS::FAIL_EXCEPTION );
        }
    }

    ts->set_gtest_status();
}


void BaseTest::run( int start_from )
{
    int test_case_idx, count = get_test_case_count();
    int64 t_start = cvGetTickCount();
    double freq = cv::getTickFrequency();
    bool ff = can_do_fast_forward();
    int progress = 0, code;
    int64 t1 = t_start;

    for( test_case_idx = ff && start_from >= 0 ? start_from : 0;
         count < 0 || test_case_idx < count; test_case_idx++ )
    {
        ts->update_context( this, test_case_idx, ff );
        progress = update_progress( progress, test_case_idx, count, (double)(t1 - t_start)/(freq*1000) );

        code = prepare_test_case( test_case_idx );
        if( code < 0 || ts->get_err_code() < 0 )
            return;

        if( code == 0 )
            continue;

        run_func();

        if( ts->get_err_code() < 0 )
            return;

        if( validate_test_results( test_case_idx ) < 0 || ts->get_err_code() < 0 )
            return;
    }
}


void BaseTest::run_func(void)
{
    assert(0);
}


int BaseTest::get_test_case_count(void)
{
    return test_case_count;
}


int BaseTest::prepare_test_case( int )
{
    return 0;
}


int BaseTest::validate_test_results( int )
{
    return 0;
}


int BaseTest::update_progress( int progress, int test_case_idx, int count, double dt )
{
    int width = 60 - (int)get_name().size();
    if( count > 0 )
    {
        int t = cvRound( ((double)test_case_idx * width)/count );
        if( t > progress )
        {
            ts->printf( TS::CONSOLE, "." );
            progress = t;
        }
    }
    else if( cvRound(dt) > progress )
    {
        ts->printf( TS::CONSOLE, "." );
        progress = cvRound(dt);
    }

    return progress;
}


BadArgTest::BadArgTest()
{
    test_case_idx   = -1;
    // oldErrorCbk     = 0;
    // oldErrorCbkData = 0;
}

BadArgTest::~BadArgTest(void)
{
}

int BadArgTest::run_test_case( int expected_code, const string& _descr )
{
    int errcount = 0;
    bool thrown = false;
    const char* descr = _descr.c_str() ? _descr.c_str() : "";

    try
    {
        run_func();
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

/*****************************************************************************************\
*                                 Base Class for Test System                              *
\*****************************************************************************************/

/******************************** Constructors/Destructors ******************************/

TSParams::TSParams()
{
    rng_seed = (uint64)-1;
    use_optimized = true;
    test_case_count_scale = 1;
}


TestInfo::TestInfo()
{
    test = 0;
    code = 0;
    rng_seed = rng_seed0 = 0;
    test_case_idx = -1;
}


TS::TS()
{
} // ctor


TS::~TS()
{
} // dtor


string TS::str_from_code( const TS::FailureCode code )
{
    switch( code )
    {
    case OK:                           return "Ok";
    case FAIL_GENERIC:                 return "Generic/Unknown";
    case FAIL_MISSING_TEST_DATA:       return "No test data";
    case FAIL_INVALID_TEST_DATA:       return "Invalid test data";
    case FAIL_ERROR_IN_CALLED_FUNC:    return "cvError invoked";
    case FAIL_EXCEPTION:               return "Hardware/OS exception";
    case FAIL_MEMORY_EXCEPTION:        return "Invalid memory access";
    case FAIL_ARITHM_EXCEPTION:        return "Arithmetic exception";
    case FAIL_MEMORY_CORRUPTION_BEGIN: return "Corrupted memblock (beginning)";
    case FAIL_MEMORY_CORRUPTION_END:   return "Corrupted memblock (end)";
    case FAIL_MEMORY_LEAK:             return "Memory leak";
    case FAIL_INVALID_OUTPUT:          return "Invalid function output";
    case FAIL_MISMATCH:                return "Unexpected output";
    case FAIL_BAD_ACCURACY:            return "Bad accuracy";
    case FAIL_HANG:                    return "Infinite loop(?)";
    case FAIL_BAD_ARG_CHECK:           return "Incorrect handling of bad arguments";
    default:
            ;
    }
    return "Generic/Unknown";
}

static int tsErrorCallback( int status, const char* func_name, const char* err_msg, const char* file_name, int line, TS* ts )
{
    ts->printf(TS::LOG, "OpenCV Error:\n\t%s (%s) in %s, file %s, line %d\n", cvErrorStr(status), err_msg, func_name[0] != 0 ? func_name : "unknown function", file_name, line);
    return 0;
}

/************************************** Running tests **********************************/

void TS::init( const string& modulename )
{
    data_search_subdir.push_back(modulename);
#ifndef WINRT
    char* datapath_dir = getenv("OPENCV_TEST_DATA_PATH");
#else
    char* datapath_dir = OPENCV_TEST_DATA_PATH;
#endif

    if( datapath_dir )
    {
        data_path = path_join(path_join(datapath_dir, modulename), "");
    }

    cv::redirectError((cv::ErrorCallback)tsErrorCallback, this);

    if( ::testing::GTEST_FLAG(catch_exceptions) )
    {
#if defined _WIN32
#ifdef _MSC_VER
        _set_se_translator( SEHTranslator );
#endif
#else
        for( int i = 0; tsSigId[i] >= 0; i++ )
            signal( tsSigId[i], signalHandler );
#endif
    }
    else
    {
#if defined _WIN32
#ifdef _MSC_VER
        _set_se_translator( 0 );
#endif
#else
        for( int i = 0; tsSigId[i] >= 0; i++ )
            signal( tsSigId[i], SIG_DFL );
#endif
    }

    if( params.use_optimized == 0 )
        cv::setUseOptimized(false);

    rng = RNG(params.rng_seed);
}


void TS::set_gtest_status()
{
    TS::FailureCode code = get_err_code();
    if( code >= 0 )
        return SUCCEED();

    char seedstr[32];
    sprintf(seedstr, "%08x%08x", (unsigned)(current_test_info.rng_seed>>32),
                                (unsigned)(current_test_info.rng_seed));

    string logs = "";
    if( !output_buf[SUMMARY_IDX].empty() )
        logs += "\n-----------------------------------\n\tSUM: " + output_buf[SUMMARY_IDX];
    if( !output_buf[LOG_IDX].empty() )
        logs += "\n-----------------------------------\n\tLOG:\n" + output_buf[LOG_IDX];
    if( !output_buf[CONSOLE_IDX].empty() )
        logs += "\n-----------------------------------\n\tCONSOLE: " + output_buf[CONSOLE_IDX];
    logs += "\n-----------------------------------\n";

    FAIL() << "\n\tfailure reason: " << str_from_code(code) <<
        "\n\ttest case #" << current_test_info.test_case_idx <<
        "\n\tseed: " << seedstr << logs;
}


CvFileStorage* TS::get_file_storage() { return 0; }

void TS::update_context( BaseTest* test, int test_case_idx, bool update_ts_context )
{
    if( current_test_info.test != test )
    {
        for( int i = 0; i <= CONSOLE_IDX; i++ )
            output_buf[i] = string();
        rng = RNG(params.rng_seed);
        current_test_info.rng_seed0 = current_test_info.rng_seed = rng.state;
    }

    current_test_info.test = test;
    current_test_info.test_case_idx = test_case_idx;
    current_test_info.code = 0;
    cvSetErrStatus( CV_StsOk );
    if( update_ts_context )
        current_test_info.rng_seed = rng.state;
}


void TS::set_failed_test_info( int fail_code )
{
    if( current_test_info.code >= 0 )
        current_test_info.code = TS::FailureCode(fail_code);
}

#if defined _MSC_VER && _MSC_VER < 1400
#undef vsnprintf
#define vsnprintf _vsnprintf
#endif

void TS::vprintf( int streams, const char* fmt, va_list l )
{
    char str[1 << 14];
    vsnprintf( str, sizeof(str)-1, fmt, l );

    for( int i = 0; i < MAX_IDX; i++ )
        if( (streams & (1 << i)) )
        {
            output_buf[i] += std::string(str);
            // in the new GTest-based framework we do not use
            // any output files (except for the automatically generated xml report).
            // if a test fails, all the buffers are printed, so we do not want to duplicate the information and
            // thus only add the new information to a single buffer and return from the function.
            break;
        }
}


void TS::printf( int streams, const char* fmt, ... )
{
    if( streams )
    {
        va_list l;
        va_start( l, fmt );
        vprintf( streams, fmt, l );
        va_end( l );
    }
}


static TS ts;
TS* TS::ptr() { return &ts; }

void fillGradient(Mat& img, int delta)
{
    const int ch = img.channels();
    CV_Assert(!img.empty() && img.depth() == CV_8U && ch <= 4);

    int n = 255 / delta;
    int r, c, i;
    for(r=0; r<img.rows; r++)
    {
        int kR = r % (2*n);
        int valR = (kR<=n) ? delta*kR : delta*(2*n-kR);
        for(c=0; c<img.cols; c++)
        {
            int kC = c % (2*n);
            int valC = (kC<=n) ? delta*kC : delta*(2*n-kC);
            uchar vals[] = {uchar(valR), uchar(valC), uchar(200*r/img.rows), uchar(255)};
            uchar *p = img.ptr(r, c);
            for(i=0; i<ch; i++) p[i] = vals[i];
        }
    }
}

void smoothBorder(Mat& img, const Scalar& color, int delta)
{
    const int ch = img.channels();
    CV_Assert(!img.empty() && img.depth() == CV_8U && ch <= 4);

    Scalar s;
    uchar *p = NULL;
    int n = 100/delta;
    int nR = std::min(n, (img.rows+1)/2), nC = std::min(n, (img.cols+1)/2);

    int r, c, i;
    for(r=0; r<nR; r++)
    {
        double k1 = r*delta/100., k2 = 1-k1;
        for(c=0; c<img.cols; c++)
        {
            p = img.ptr(r, c);
            for(i=0; i<ch; i++) s[i] = p[i];
            s = s * k1 + color * k2;
            for(i=0; i<ch; i++) p[i] = uchar(s[i]);
        }
        for(c=0; c<img.cols; c++)
        {
            p = img.ptr(img.rows-r-1, c);
            for(i=0; i<ch; i++) s[i] = p[i];
            s = s * k1 + color * k2;
            for(i=0; i<ch; i++) p[i] = uchar(s[i]);
        }
    }

    for(r=0; r<img.rows; r++)
    {
        for(c=0; c<nC; c++)
        {
            double k1 = c*delta/100., k2 = 1-k1;
            p = img.ptr(r, c);
            for(i=0; i<ch; i++) s[i] = p[i];
            s = s * k1 + color * k2;
            for(i=0; i<ch; i++) p[i] = uchar(s[i]);
        }
        for(c=0; c<n; c++)
        {
            double k1 = c*delta/100., k2 = 1-k1;
            p = img.ptr(r, img.cols-c-1);
            for(i=0; i<ch; i++) s[i] = p[i];
            s = s * k1 + color * k2;
            for(i=0; i<ch; i++) p[i] = uchar(s[i]);
        }
    }
}


bool test_ipp_check = false;

void checkIppStatus()
{
    if (test_ipp_check)
    {
        int status = cv::ipp::getIppStatus();
        EXPECT_LE(0, status) << cv::ipp::getIppErrorLocation().c_str();
    }
}

bool skipUnstableTests = false;

void parseCustomOptions(int argc, char **argv)
{
    const char * const command_line_keys =
        "{ ipp test_ipp_check |false    |check whether IPP works without failures }"
        "{ test_seed          |809564   |seed for random numbers generator }"
        "{ skip_unstable      |false    |skip unstable tests }"
        "{ h   help           |false    |print help info                          }";

    cv::CommandLineParser parser(argc, argv, command_line_keys);
    if (parser.get<bool>("help"))
    {
        std::cout << "\nAvailable options besides google test option: \n";
        parser.printMessage();
    }

    test_ipp_check = parser.get<bool>("test_ipp_check");
    if (!test_ipp_check)
#ifndef WINRT
        test_ipp_check = getenv("OPENCV_IPP_CHECK") != NULL;
#else
        test_ipp_check = false;
#endif

    param_seed = parser.get<unsigned int>("test_seed");

    skipUnstableTests = parser.get<bool>("skip_unstable");
}


static bool isDirectory(const std::string& path)
{
#if defined _WIN32 || defined WINCE
    WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#ifdef WINRT
    wchar_t wpath[MAX_PATH];
    size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
    CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
    BOOL status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
    BOOL status = ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
    DWORD attributes = all_attrs.dwFileAttributes;
    return status && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    struct stat s;
    if (0 != stat(path.c_str(), &s))
        return false;
    return S_ISDIR(s.st_mode);
#endif
}

void addDataSearchPath(const std::string& path)
{
    if (isDirectory(path))
        TS::ptr()->data_search_path.push_back(path);
}
void addDataSearchSubDirectory(const std::string& subdir)
{
    TS::ptr()->data_search_subdir.push_back(subdir);
}

std::string findDataFile(const std::string& relative_path, bool required)
{
#define TEST_TRY_FILE_WITH_PREFIX(prefix) \
{ \
    std::string path = path_join(prefix, relative_path); \
    /*printf("Trying %s\n", path.c_str());*/ \
    FILE* f = fopen(path.c_str(), "rb"); \
    if(f) { \
       fclose(f); \
       return path; \
    } \
}

    const std::vector<std::string>& search_path = TS::ptr()->data_search_path;
    for(size_t i = search_path.size(); i > 0; i--)
    {
        const std::string& prefix = search_path[i - 1];
        TEST_TRY_FILE_WITH_PREFIX(prefix);
    }

    const std::vector<std::string>& search_subdir = TS::ptr()->data_search_subdir;

#ifndef WINRT
    char* datapath_dir = getenv("OPENCV_TEST_DATA_PATH");
#else
    char* datapath_dir = OPENCV_TEST_DATA_PATH;
#endif

    std::string datapath;
    if (datapath_dir)
    {
        datapath = datapath_dir;
        //CV_Assert(isDirectory(datapath) && "OPENCV_TEST_DATA_PATH is specified but it doesn't exist");
        if (isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const std::string& subdir = search_subdir[i - 1];
                std::string prefix = path_join(datapath, subdir);
                TEST_TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#ifdef OPENCV_TEST_DATA_INSTALL_PATH
    datapath = path_join("./", OPENCV_TEST_DATA_INSTALL_PATH);
    if (isDirectory(datapath))
    {
        for(size_t i = search_subdir.size(); i > 0; i--)
        {
            const std::string& subdir = search_subdir[i - 1];
            std::string prefix = path_join(datapath, subdir);
            TEST_TRY_FILE_WITH_PREFIX(prefix);
        }
    }
#ifdef OPENCV_INSTALL_PREFIX
    else
    {
        datapath = path_join(OPENCV_INSTALL_PREFIX, OPENCV_TEST_DATA_INSTALL_PATH);
        if (isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const std::string& subdir = search_subdir[i - 1];
                std::string prefix = path_join(datapath, subdir);
                TEST_TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#endif
#endif
    if (required)
        CV_ErrorNoReturn(cv::Error::StsError, cv::format("OpenCV tests: Can't find required data file: %s", relative_path.c_str()));
    throw SkipTestException(cv::format("OpenCV tests: Can't find data file: %s", relative_path.c_str()));
}


} //namespace cvtest

/* End of file. */
