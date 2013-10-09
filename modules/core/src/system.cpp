/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif

#if defined WIN32 || defined _WIN32 || defined WINCE
#ifndef _WIN32_WINNT           // This is needed for the declaration of TryEnterCriticalSection in winbase.h with Visual Studio 2005 (and older?)
  #define _WIN32_WINNT 0x0400  // http://msdn.microsoft.com/en-us/library/ms686857(VS.85).aspx
#endif
#include <windows.h>
#if (_WIN32_WINNT >= 0x0602)
  #include <synchapi.h>
#endif
#undef small
#undef min
#undef max
#undef abs
#include <tchar.h>
#if defined _MSC_VER
  #if _MSC_VER >= 1400
    #include <intrin.h>
  #elif defined _M_IX86
    static void __cpuid(int* cpuid_data, int)
    {
        __asm
        {
            push ebx
            push edi
            mov edi, cpuid_data
            mov eax, 1
            cpuid
            mov [edi], eax
            mov [edi + 4], ebx
            mov [edi + 8], ecx
            mov [edi + 12], edx
            pop edi
            pop ebx
        }
    }
  #endif
#endif

#ifdef HAVE_WINRT
#include <wrl/client.h>

std::wstring GetTempPathWinRT()
{
    return std::wstring(Windows::Storage::ApplicationData::Current->TemporaryFolder->Path->Data());
}

std::wstring GetTempFileNameWinRT(std::wstring prefix)
{
    wchar_t guidStr[40];
    GUID g;
    CoCreateGuid(&g);
    wchar_t* mask = L"%08x_%04x_%04x_%02x%02x_%02x%02x%02x%02x%02x%02x";
    swprintf(&guidStr[0], sizeof(guidStr)/sizeof(wchar_t), mask,
             g.Data1, g.Data2, g.Data3, UINT(g.Data4[0]), UINT(g.Data4[1]),
             UINT(g.Data4[2]), UINT(g.Data4[3]), UINT(g.Data4[4]),
             UINT(g.Data4[5]), UINT(g.Data4[6]), UINT(g.Data4[7]));

    return prefix + std::wstring(guidStr);
}

#endif
#else
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

#if defined __MACH__ && defined __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

#endif

#ifdef _OPENMP
#include "omp.h"
#endif

#include <stdarg.h>

#if defined __linux__ || defined __APPLE__
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#if defined ANDROID
#include <sys/sysconf.h>
#else
#include <sys/sysctl.h>
#endif
#endif

#ifdef ANDROID
# include <android/log.h>
#endif

namespace cv
{

Exception::Exception() { code = 0; line = 0; }

Exception::Exception(int _code, const String& _err, const String& _func, const String& _file, int _line)
: code(_code), err(_err), func(_func), file(_file), line(_line)
{
    formatMessage();
}

Exception::~Exception() throw() {}

/*!
 \return the error description and the context as a text string.
 */
const char* Exception::what() const throw() { return msg.c_str(); }

void Exception::formatMessage()
{
    if( func.size() > 0 )
        msg = format("%s:%d: error: (%d) %s in function %s\n", file.c_str(), line, code, err.c_str(), func.c_str());
    else
        msg = format("%s:%d: error: (%d) %s\n", file.c_str(), line, code, err.c_str());
}

struct HWFeatures
{
    enum { MAX_FEATURE = CV_HARDWARE_MAX_FEATURE };

    HWFeatures(void)
     {
        memset( have, 0, sizeof(have) );
        x86_family = 0;
    }

    static HWFeatures initialize(void)
    {
        HWFeatures f;
        int cpuid_data[4] = { 0, 0, 0, 0 };

    #if defined _MSC_VER && (defined _M_IX86 || defined _M_X64)
        __cpuid(cpuid_data, 1);
    #elif defined __GNUC__ && (defined __i386__ || defined __x86_64__)
        #ifdef __x86_64__
        asm __volatile__
        (
         "movl $1, %%eax\n\t"
         "cpuid\n\t"
         :[eax]"=a"(cpuid_data[0]),[ebx]"=b"(cpuid_data[1]),[ecx]"=c"(cpuid_data[2]),[edx]"=d"(cpuid_data[3])
         :
         : "cc"
        );
        #else
        asm volatile
        (
         "pushl %%ebx\n\t"
         "movl $1,%%eax\n\t"
         "cpuid\n\t"
         "popl %%ebx\n\t"
         : "=a"(cpuid_data[0]), "=c"(cpuid_data[2]), "=d"(cpuid_data[3])
         :
         : "cc"
        );
        #endif
    #endif

        f.x86_family = (cpuid_data[0] >> 8) & 15;
        if( f.x86_family >= 6 )
        {
            f.have[CV_CPU_MMX]    = (cpuid_data[3] & (1 << 23)) != 0;
            f.have[CV_CPU_SSE]    = (cpuid_data[3] & (1<<25)) != 0;
            f.have[CV_CPU_SSE2]   = (cpuid_data[3] & (1<<26)) != 0;
            f.have[CV_CPU_SSE3]   = (cpuid_data[2] & (1<<0)) != 0;
            f.have[CV_CPU_SSSE3]  = (cpuid_data[2] & (1<<9)) != 0;
            f.have[CV_CPU_SSE4_1] = (cpuid_data[2] & (1<<19)) != 0;
            f.have[CV_CPU_SSE4_2] = (cpuid_data[2] & (1<<20)) != 0;
            f.have[CV_CPU_POPCNT] = (cpuid_data[2] & (1<<23)) != 0;
            f.have[CV_CPU_AVX]    = (((cpuid_data[2] & (1<<28)) != 0)&&((cpuid_data[2] & (1<<27)) != 0));//OS uses XSAVE_XRSTORE and CPU support AVX
        }

        return f;
    }

    int x86_family;
    bool have[MAX_FEATURE+1];
};

static HWFeatures  featuresEnabled = HWFeatures::initialize(), featuresDisabled = HWFeatures();
static HWFeatures* currentFeatures = &featuresEnabled;

bool checkHardwareSupport(int feature)
{
    CV_DbgAssert( 0 <= feature && feature <= CV_HARDWARE_MAX_FEATURE );
    return currentFeatures->have[feature];
}


volatile bool useOptimizedFlag = true;
#ifdef HAVE_IPP
struct IPPInitializer
{
    IPPInitializer(void) { ippStaticInit(); }
};

IPPInitializer ippInitializer;
#endif

volatile bool USE_SSE2 = featuresEnabled.have[CV_CPU_SSE2];
volatile bool USE_SSE4_2 = featuresEnabled.have[CV_CPU_SSE4_2];
volatile bool USE_AVX = featuresEnabled.have[CV_CPU_AVX];

void setUseOptimized( bool flag )
{
    useOptimizedFlag = flag;
    currentFeatures = flag ? &featuresEnabled : &featuresDisabled;
    USE_SSE2 = currentFeatures->have[CV_CPU_SSE2];
}

bool useOptimized(void)
{
    return useOptimizedFlag;
}

int64 getTickCount(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER counter;
    QueryPerformanceCounter( &counter );
    return (int64)counter.QuadPart;
#elif defined __linux || defined __linux__
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (int64)tp.tv_sec*1000000000 + tp.tv_nsec;
#elif defined __MACH__ && defined __APPLE__
    return (int64)mach_absolute_time();
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday( &tv, &tz );
    return (int64)tv.tv_sec*1000000 + tv.tv_usec;
#endif
}

double getTickFrequency(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double)freq.QuadPart;
#elif defined __linux || defined __linux__
    return 1e9;
#elif defined __MACH__ && defined __APPLE__
    static double freq = 0;
    if( freq == 0 )
    {
        mach_timebase_info_data_t sTimebaseInfo;
        mach_timebase_info(&sTimebaseInfo);
        freq = sTimebaseInfo.denom*1e9/sTimebaseInfo.numer;
    }
    return freq;
#else
    return 1e6;
#endif
}

#if defined __GNUC__ && (defined __i386__ || defined __x86_64__ || defined __ppc__)
#if defined(__i386__)

int64 getCPUTickCount(void)
{
    int64 x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}
#elif defined(__x86_64__)

int64 getCPUTickCount(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return (int64)lo | ((int64)hi << 32);
}

#elif defined(__ppc__)

int64 getCPUTickCount(void)
{
    int64 result = 0;
    unsigned upper, lower, tmp;
    __asm__ volatile(
                     "0:                  \n"
                     "\tmftbu   %0           \n"
                     "\tmftb    %1           \n"
                     "\tmftbu   %2           \n"
                     "\tcmpw    %2,%0        \n"
                     "\tbne     0b         \n"
                     : "=r"(upper),"=r"(lower),"=r"(tmp)
                     );
    return lower | ((int64)upper << 32);
}

#else

#error "RDTSC not defined"

#endif

#elif defined _MSC_VER && defined WIN32 && defined _M_IX86

int64 getCPUTickCount(void)
{
    __asm _emit 0x0f;
    __asm _emit 0x31;
}

#else

#ifdef HAVE_IPP
int64 getCPUTickCount(void)
{
    return ippGetCpuClocks();
}
#else
int64 getCPUTickCount(void)
{
    return getTickCount();
}
#endif

#endif

const String& getBuildInformation()
{
    static String build_info =
#include "version_string.inc"
    ;
    return build_info;
}

String format( const char* fmt, ... )
{
    char buf[1024];

    va_list va;
    va_start(va, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, va);
    va_end(va);

    if (len >= (int)sizeof(buf))
    {
        String s(len, '\0');
        va_start(va, fmt);
        len = vsnprintf((char*)s.c_str(), len + 1, fmt, va);
        va_end(va);
        return s;
    }

    return String(buf, len);
}

String tempfile( const char* suffix )
{
#ifdef HAVE_WINRT
    std::wstring temp_dir = L"";
    const wchar_t* opencv_temp_dir = _wgetenv(L"OPENCV_TEMP_PATH");
    if (opencv_temp_dir)
        temp_dir = std::wstring(opencv_temp_dir);
#else
    const char *temp_dir = getenv("OPENCV_TEMP_PATH");
    String fname;
#endif

#if defined WIN32 || defined _WIN32
#ifdef HAVE_WINRT
    RoInitialize(RO_INIT_MULTITHREADED);
    std::wstring temp_dir2;
    if (temp_dir.empty())
        temp_dir = GetTempPathWinRT();

    std::wstring temp_file;
    temp_file = GetTempFileNameWinRT(L"ocv");
    if (temp_file.empty())
        return std::string();

    temp_file = temp_dir + std::wstring(L"\\") + temp_file;
    DeleteFileW(temp_file.c_str());

    char aname[MAX_PATH];
    size_t copied = wcstombs(aname, temp_file.c_str(), MAX_PATH);
    CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
    fname = std::string(aname);
    RoUninitialize();
#else
    char temp_dir2[MAX_PATH] = { 0 };
    char temp_file[MAX_PATH] = { 0 };

    if (temp_dir == 0 || temp_dir[0] == 0)
    {
        ::GetTempPathA(sizeof(temp_dir2), temp_dir2);
        temp_dir = temp_dir2;
    }
    if(0 == ::GetTempFileNameA(temp_dir, "ocv", 0, temp_file))
        return String();

    DeleteFileA(temp_file);

    fname = temp_file;
#endif
# else
#  ifdef ANDROID
    //char defaultTemplate[] = "/mnt/sdcard/__opencv_temp.XXXXXX";
    char defaultTemplate[] = "/data/local/tmp/__opencv_temp.XXXXXX";
#  else
    char defaultTemplate[] = "/tmp/__opencv_temp.XXXXXX";
#  endif

    if (temp_dir == 0 || temp_dir[0] == 0)
        fname = defaultTemplate;
    else
    {
        fname = temp_dir;
        char ech = fname[fname.size() - 1];
        if(ech != '/' && ech != '\\')
            fname = fname + "/";
        fname = fname + "__opencv_temp.XXXXXX";
    }

    const int fd = mkstemp((char*)fname.c_str());
    if (fd == -1) return String();

    close(fd);
    remove(fname.c_str());
# endif

    if (suffix)
    {
        if (suffix[0] != '.')
            return fname + "." + suffix;
        else
            return fname + suffix;
    }
    return fname;
}

static CvErrorCallback customErrorCallback = 0;
static void* customErrorCallbackData = 0;
static bool breakOnError = false;

bool setBreakOnError(bool value)
{
    bool prevVal = breakOnError;
    breakOnError = value;
    return prevVal;
}

void error( const Exception& exc )
{
    if (customErrorCallback != 0)
        customErrorCallback(exc.code, exc.func.c_str(), exc.err.c_str(),
                            exc.file.c_str(), exc.line, customErrorCallbackData);
    else
    {
        const char* errorStr = cvErrorStr(exc.code);
        char buf[1 << 16];

        sprintf( buf, "OpenCV Error: %s (%s) in %s, file %s, line %d",
            errorStr, exc.err.c_str(), exc.func.size() > 0 ?
            exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line );
        fprintf( stderr, "%s\n", buf );
        fflush( stderr );
#  ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "cv::error()", "%s", buf);
#  endif
    }

    if(breakOnError)
    {
        static volatile int* p = 0;
        *p = 0;
    }

    throw exc;
}

void error(int _code, const String& _err, const char* _func, const char* _file, int _line)
{
    error(cv::Exception(_code, _err, _func, _file, _line));
}

CvErrorCallback
redirectError( CvErrorCallback errCallback, void* userdata, void** prevUserdata)
{
    if( prevUserdata )
        *prevUserdata = customErrorCallbackData;

    CvErrorCallback prevCallback = customErrorCallback;

    customErrorCallback     = errCallback;
    customErrorCallbackData = userdata;

    return prevCallback;
}

}

CV_IMPL int cvCheckHardwareSupport(int feature)
{
    CV_DbgAssert( 0 <= feature && feature <= CV_HARDWARE_MAX_FEATURE );
    return cv::currentFeatures->have[feature];
}

CV_IMPL int cvUseOptimized( int flag )
{
    int prevMode = cv::useOptimizedFlag;
    cv::setUseOptimized( flag != 0 );
    return prevMode;
}

CV_IMPL int64  cvGetTickCount(void)
{
    return cv::getTickCount();
}

CV_IMPL double cvGetTickFrequency(void)
{
    return cv::getTickFrequency()*1e-6;
}

CV_IMPL CvErrorCallback
cvRedirectError( CvErrorCallback errCallback, void* userdata, void** prevUserdata)
{
    return cv::redirectError(errCallback, userdata, prevUserdata);
}

CV_IMPL int cvNulDevReport( int, const char*, const char*,
                            const char*, int, void* )
{
    return 0;
}

CV_IMPL int cvStdErrReport( int, const char*, const char*,
                            const char*, int, void* )
{
    return 0;
}

CV_IMPL int cvGuiBoxReport( int, const char*, const char*,
                            const char*, int, void* )
{
    return 0;
}

CV_IMPL int cvGetErrInfo( const char**, const char**, const char**, int* )
{
    return 0;
}


CV_IMPL const char* cvErrorStr( int status )
{
    static char buf[256];

    switch (status)
    {
    case CV_StsOk :                  return "No Error";
    case CV_StsBackTrace :           return "Backtrace";
    case CV_StsError :               return "Unspecified error";
    case CV_StsInternal :            return "Internal error";
    case CV_StsNoMem :               return "Insufficient memory";
    case CV_StsBadArg :              return "Bad argument";
    case CV_StsNoConv :              return "Iterations do not converge";
    case CV_StsAutoTrace :           return "Autotrace call";
    case CV_StsBadSize :             return "Incorrect size of input array";
    case CV_StsNullPtr :             return "Null pointer";
    case CV_StsDivByZero :           return "Division by zero occured";
    case CV_BadStep :                return "Image step is wrong";
    case CV_StsInplaceNotSupported : return "Inplace operation is not supported";
    case CV_StsObjectNotFound :      return "Requested object was not found";
    case CV_BadDepth :               return "Input image depth is not supported by function";
    case CV_StsUnmatchedFormats :    return "Formats of input arguments do not match";
    case CV_StsUnmatchedSizes :      return "Sizes of input arguments do not match";
    case CV_StsOutOfRange :          return "One of arguments\' values is out of range";
    case CV_StsUnsupportedFormat :   return "Unsupported format or combination of formats";
    case CV_BadCOI :                 return "Input COI is not supported";
    case CV_BadNumChannels :         return "Bad number of channels";
    case CV_StsBadFlag :             return "Bad flag (parameter or structure field)";
    case CV_StsBadPoint :            return "Bad parameter of type CvPoint";
    case CV_StsBadMask :             return "Bad type of mask argument";
    case CV_StsParseError :          return "Parsing error";
    case CV_StsNotImplemented :      return "The function/feature is not implemented";
    case CV_StsBadMemBlock :         return "Memory block has been corrupted";
    case CV_StsAssert :              return "Assertion failed";
    case CV_GpuNotSupported :        return "No GPU support";
    case CV_GpuApiCallError :        return "Gpu API call";
    case CV_OpenGlNotSupported :     return "No OpenGL support";
    case CV_OpenGlApiCallError :     return "OpenGL API call";
    };

    sprintf(buf, "Unknown %s code %d", status >= 0 ? "status":"error", status);
    return buf;
}

CV_IMPL int cvGetErrMode(void)
{
    return 0;
}

CV_IMPL int cvSetErrMode(int)
{
    return 0;
}

CV_IMPL int cvGetErrStatus(void)
{
    return 0;
}

CV_IMPL void cvSetErrStatus(int)
{
}


CV_IMPL void cvError( int code, const char* func_name,
                      const char* err_msg,
                      const char* file_name, int line )
{
    cv::error(cv::Exception(code, err_msg, func_name, file_name, line));
}

/* function, which converts int to int */
CV_IMPL int
cvErrorFromIppStatus( int status )
{
    switch (status)
    {
    case CV_BADSIZE_ERR:               return CV_StsBadSize;
    case CV_BADMEMBLOCK_ERR:           return CV_StsBadMemBlock;
    case CV_NULLPTR_ERR:               return CV_StsNullPtr;
    case CV_DIV_BY_ZERO_ERR:           return CV_StsDivByZero;
    case CV_BADSTEP_ERR:               return CV_BadStep;
    case CV_OUTOFMEM_ERR:              return CV_StsNoMem;
    case CV_BADARG_ERR:                return CV_StsBadArg;
    case CV_NOTDEFINED_ERR:            return CV_StsError;
    case CV_INPLACE_NOT_SUPPORTED_ERR: return CV_StsInplaceNotSupported;
    case CV_NOTFOUND_ERR:              return CV_StsObjectNotFound;
    case CV_BADCONVERGENCE_ERR:        return CV_StsNoConv;
    case CV_BADDEPTH_ERR:              return CV_BadDepth;
    case CV_UNMATCHED_FORMATS_ERR:     return CV_StsUnmatchedFormats;
    case CV_UNSUPPORTED_COI_ERR:       return CV_BadCOI;
    case CV_UNSUPPORTED_CHANNELS_ERR:  return CV_BadNumChannels;
    case CV_BADFLAG_ERR:               return CV_StsBadFlag;
    case CV_BADRANGE_ERR:              return CV_StsBadArg;
    case CV_BADCOEF_ERR:               return CV_StsBadArg;
    case CV_BADFACTOR_ERR:             return CV_StsBadArg;
    case CV_BADPOINT_ERR:              return CV_StsBadPoint;

    default:
      return CV_StsError;
    }
}


#if defined CVAPI_EXPORTS && defined WIN32 && !defined WINCE
#ifdef HAVE_WINRT
    #pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
#endif

BOOL WINAPI DllMain( HINSTANCE, DWORD  fdwReason, LPVOID );

BOOL WINAPI DllMain( HINSTANCE, DWORD  fdwReason, LPVOID )
{
    if( fdwReason == DLL_THREAD_DETACH || fdwReason == DLL_PROCESS_DETACH )
    {
        cv::deleteThreadAllocData();
        cv::deleteThreadRNGData();
    }
    return TRUE;
}
#endif

namespace cv
{

#if defined WIN32 || defined _WIN32 || defined WINCE

struct Mutex::Impl
{
    Impl()
    {
#if (_WIN32_WINNT >= 0x0600)
        ::InitializeCriticalSectionEx(&cs, 1000, 0);
#else
        ::InitializeCriticalSection(&cs);
#endif
        refcount = 1;
    }
    ~Impl() { DeleteCriticalSection(&cs); }

    void lock() { EnterCriticalSection(&cs); }
    bool trylock() { return TryEnterCriticalSection(&cs) != 0; }
    void unlock() { LeaveCriticalSection(&cs); }

    CRITICAL_SECTION cs;
    int refcount;
};

#elif defined __APPLE__

#include <libkern/OSAtomic.h>

struct Mutex::Impl
{
    Impl() { sl = OS_SPINLOCK_INIT; refcount = 1; }
    ~Impl() {}

    void lock() { OSSpinLockLock(&sl); }
    bool trylock() { return OSSpinLockTry(&sl); }
    void unlock() { OSSpinLockUnlock(&sl); }

    OSSpinLock sl;
    int refcount;
};

#elif defined __linux__ && !defined ANDROID

struct Mutex::Impl
{
    Impl() { pthread_spin_init(&sl, 0); refcount = 1; }
    ~Impl() { pthread_spin_destroy(&sl); }

    void lock() { pthread_spin_lock(&sl); }
    bool trylock() { return pthread_spin_trylock(&sl) == 0; }
    void unlock() { pthread_spin_unlock(&sl); }

    pthread_spinlock_t sl;
    int refcount;
};

#else

struct Mutex::Impl
{
    Impl() { pthread_mutex_init(&sl, 0); refcount = 1; }
    ~Impl() { pthread_mutex_destroy(&sl); }

    void lock() { pthread_mutex_lock(&sl); }
    bool trylock() { return pthread_mutex_trylock(&sl) == 0; }
    void unlock() { pthread_mutex_unlock(&sl); }

    pthread_mutex_t sl;
    int refcount;
};

#endif

Mutex::Mutex()
{
    impl = new Mutex::Impl;
}

Mutex::~Mutex()
{
    if( CV_XADD(&impl->refcount, -1) == 1 )
        delete impl;
    impl = 0;
}

Mutex::Mutex(const Mutex& m)
{
    impl = m.impl;
    CV_XADD(&impl->refcount, 1);
}

Mutex& Mutex::operator = (const Mutex& m)
{
    CV_XADD(&m.impl->refcount, 1);
    if( CV_XADD(&impl->refcount, -1) == 1 )
        delete impl;
    impl = m.impl;
    return *this;
}

void Mutex::lock() { impl->lock(); }
void Mutex::unlock() { impl->unlock(); }
bool Mutex::trylock() { return impl->trylock(); }

}

//////////////////////////////// thread-local storage ////////////////////////////////

namespace cv
{

#ifdef WIN32

#ifdef HAVE_WINRT
    // using C++11 thread attribute for local thread data
    static __declspec( thread ) TLSData* g_tlsdata = NULL;

    static void deleteThreadRNGData()
    {
        if (g_tlsdata)
            delete g_tlsdata;
    }

    TLSData* TLSData::get()
    {
        if (!g_tlsdata)
        {
            g_tlsdata = new TLSData;
        }
        return g_tlsdata;
    }
#else
#ifdef WINCE
#	define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
    static DWORD tlsKey = TLS_OUT_OF_INDEXES;

    void deleteThreadData()
    {
        if( tlsKey != TLS_OUT_OF_INDEXES )
            delete (TLSData*)TlsGetValue( tlsKey );
    }

    TLSData* TLSData::get()
    {
        if( tlsKey == TLS_OUT_OF_INDEXES )
        {
            tlsRNGKey = TlsAlloc();
            CV_Assert(tlsRNGKey != TLS_OUT_OF_INDEXES);
        }
        TLSData* d = (TLSData*)TlsGetValue( tlsKey );
        if( !d )
        {
            d = new TLSData;
            TlsSetValue( tlsRNGKey, d );
        }
        return d;
    }
#endif //HAVE_WINRT
#else
    static pthread_key_t tlsKey = 0;
    static pthread_once_t tlsKeyOnce = PTHREAD_ONCE_INIT;

    static void deleteTLSData(void* data)
    {
        delete (TLSData*)data;
    }

    static void makeKey()
    {
        int errcode = pthread_key_create(&tlsKey, deleteTLSData);
        CV_Assert(errcode == 0);
    }

    TLSData* TLSData::get()
    {
        pthread_once(&tlsKeyOnce, makeKey);
        TLSData* d = (TLSData*)pthread_getspecific(tlsKey);
        if( !d )
        {
            d = new TLSData;
            pthread_setspecific(tlsKey, d);
        }
        return d;
    }
#endif
}


//////////////////// simple MD5 implementation from Wikipedia /////////////////////

namespace cv
{

static void to_bytes(unsigned val, uchar *bytes)
{
    bytes[0] = (uchar) val;
    bytes[1] = (uchar) (val >> 8);
    bytes[2] = (uchar) (val >> 16);
    bytes[3] = (uchar) (val >> 24);
}

bool md5sum(const uchar* initial_msg, size_t initial_len, uchar* digest, const uchar* prev)
{
    // Constants are the integer part of the sines of integers (in radians) * 2^32.
    static const unsigned k[64] =
    {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };

    // r specifies the per-round shift amounts
    static const unsigned r[] =
    {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
    };

    // leftrotate function definition
    #define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

    //Pre-processing:
    //append "1" bit to message
    //append "0" bits until message length in bits ≡ 448 (mod 512)
    //append length mod (2^64) to message
    size_t new_len = ((initial_len + 1)/(512/8))*(512/8) + (448/8);

    AutoBuffer<uchar> msgbuf(new_len + 8);
    uchar *msg = msgbuf;
    size_t offset, i;

    uint32_t w[16];

    // Initialize variables - simple count in nibbles:
    unsigned h0 = 0x67452301;
    unsigned h1 = 0xefcdab89;
    unsigned h2 = 0x98badcfe;
    unsigned h3 = 0x10325476;

    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
    for (offset = initial_len + 1; offset < new_len; offset++)
        msg[offset] = 0; // append "0" bits

    // append the len in bits at the end of the buffer.
    to_bytes(initial_len*8, msg + new_len);
    // initial_len>>29 == initial_len*8>>32, but avoids overflow.
    to_bytes(initial_len>>29, msg + new_len + 4);

    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    for( offset=0; offset < new_len; offset += (512/8) )
    {
        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        for( i = 0; i < 16; i++ )
        {
            uchar* p = msg + offset + i*4;
            w[i] = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
        }

        // Initialize hash value for this chunk:
        unsigned a = h0, b = h1, c = h2, d = h3, f, g;

        // Main loop:
        for( i = 0; i < 64; i++ )
        {
            if( i < 16 )
            {
                f = (b & c) | ((~b) & d);
                g = i;
            }
            else if( i < 32 )
            {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            }
            else if( i < 48 )
            {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;
            }
            else
            {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }

            unsigned temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;
        }

        // Add this chunk's hash to result so far:
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }

    uchar digest0[16];
    if( !digest )
        digest = digest0;

    to_bytes(h0, digest);
    to_bytes(h1, digest + 4);
    to_bytes(h2, digest + 8);
    to_bytes(h3, digest + 12);

    if( !prev )
        return true;

    return memcmp(digest, prev, 16) == 0;
}

}

/* End of file. */
