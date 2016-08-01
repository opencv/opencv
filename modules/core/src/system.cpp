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
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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
#include <iostream>

namespace cv {

static Mutex* __initialization_mutex = NULL;
Mutex& getInitializationMutex()
{
    if (__initialization_mutex == NULL)
        __initialization_mutex = new Mutex();
    return *__initialization_mutex;
}
// force initialization (single-threaded environment)
Mutex* __initialization_mutex_initializer = &getInitializationMutex();

} // namespace cv

#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif

#if defined ANDROID || defined __linux__ || defined __FreeBSD__
#  include <unistd.h>
#  include <fcntl.h>
#  include <elf.h>
#if defined ANDROID || defined __linux__
#  include <linux/auxvec.h>
#endif
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
    static void __cpuidex(int* cpuid_data, int, int)
    {
        __asm
        {
            push edi
            mov edi, cpuid_data
            mov eax, 7
            mov ecx, 0
            cpuid
            mov [edi], eax
            mov [edi + 4], ebx
            mov [edi + 8], ecx
            mov [edi + 12], edx
            pop edi
        }
    }
  #endif
#endif

#ifdef WINRT
#include <wrl/client.h>
#ifndef __cplusplus_winrt
#include <windows.storage.h>
#pragma comment(lib, "runtimeobject.lib")
#endif

std::wstring GetTempPathWinRT()
{
#ifdef __cplusplus_winrt
    return std::wstring(Windows::Storage::ApplicationData::Current->TemporaryFolder->Path->Data());
#else
    Microsoft::WRL::ComPtr<ABI::Windows::Storage::IApplicationDataStatics> appdataFactory;
    Microsoft::WRL::ComPtr<ABI::Windows::Storage::IApplicationData> appdataRef;
    Microsoft::WRL::ComPtr<ABI::Windows::Storage::IStorageFolder> storagefolderRef;
    Microsoft::WRL::ComPtr<ABI::Windows::Storage::IStorageItem> storageitemRef;
    HSTRING str;
    HSTRING_HEADER hstrHead;
    std::wstring wstr;
    if (FAILED(WindowsCreateStringReference(RuntimeClass_Windows_Storage_ApplicationData,
                                            (UINT32)wcslen(RuntimeClass_Windows_Storage_ApplicationData), &hstrHead, &str)))
        return wstr;
    if (FAILED(RoGetActivationFactory(str, IID_PPV_ARGS(appdataFactory.ReleaseAndGetAddressOf()))))
        return wstr;
    if (FAILED(appdataFactory->get_Current(appdataRef.ReleaseAndGetAddressOf())))
        return wstr;
    if (FAILED(appdataRef->get_TemporaryFolder(storagefolderRef.ReleaseAndGetAddressOf())))
        return wstr;
    if (FAILED(storagefolderRef.As(&storageitemRef)))
        return wstr;
    str = NULL;
    if (FAILED(storageitemRef->get_Path(&str)))
        return wstr;
    wstr = WindowsGetStringRawBuffer(str, NULL);
    WindowsDeleteString(str);
    return wstr;
#endif
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

    return prefix.append(std::wstring(guidStr));
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

#if defined __linux__ || defined __APPLE__ || defined __EMSCRIPTEN__
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#if defined ANDROID
#include <sys/sysconf.h>
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
            f.have[CV_CPU_FMA3]  = (cpuid_data[2] & (1<<12)) != 0;
            f.have[CV_CPU_SSE4_1] = (cpuid_data[2] & (1<<19)) != 0;
            f.have[CV_CPU_SSE4_2] = (cpuid_data[2] & (1<<20)) != 0;
            f.have[CV_CPU_POPCNT] = (cpuid_data[2] & (1<<23)) != 0;
            f.have[CV_CPU_AVX]    = (((cpuid_data[2] & (1<<28)) != 0)&&((cpuid_data[2] & (1<<27)) != 0));//OS uses XSAVE_XRSTORE and CPU support AVX
            f.have[CV_CPU_FP16]   = (cpuid_data[2] & (1<<29)) != 0;

            // make the second call to the cpuid command in order to get
            // information about extended features like AVX2
        #if defined _MSC_VER && (defined _M_IX86 || defined _M_X64)
            __cpuidex(cpuid_data, 7, 0);
        #elif defined __GNUC__ && (defined __i386__ || defined __x86_64__)
            #ifdef __x86_64__
            asm __volatile__
            (
             "movl $7, %%eax\n\t"
             "movl $0, %%ecx\n\t"
             "cpuid\n\t"
             :[eax]"=a"(cpuid_data[0]),[ebx]"=b"(cpuid_data[1]),[ecx]"=c"(cpuid_data[2]),[edx]"=d"(cpuid_data[3])
             :
             : "cc"
            );
            #else
            asm volatile
            (
             "pushl %%ebx\n\t"
             "movl $7,%%eax\n\t"
             "movl $0,%%ecx\n\t"
             "cpuid\n\t"
             "movl %%ebx, %0\n\t"
             "popl %%ebx\n\t"
             : "=r"(cpuid_data[1]), "=c"(cpuid_data[2])
             :
             : "cc"
            );
            #endif
        #endif
            f.have[CV_CPU_AVX2]   = (cpuid_data[1] & (1<<5)) != 0;

            f.have[CV_CPU_AVX_512F]       = (cpuid_data[1] & (1<<16)) != 0;
            f.have[CV_CPU_AVX_512DQ]      = (cpuid_data[1] & (1<<17)) != 0;
            f.have[CV_CPU_AVX_512IFMA512] = (cpuid_data[1] & (1<<21)) != 0;
            f.have[CV_CPU_AVX_512PF]      = (cpuid_data[1] & (1<<26)) != 0;
            f.have[CV_CPU_AVX_512ER]      = (cpuid_data[1] & (1<<27)) != 0;
            f.have[CV_CPU_AVX_512CD]      = (cpuid_data[1] & (1<<28)) != 0;
            f.have[CV_CPU_AVX_512BW]      = (cpuid_data[1] & (1<<30)) != 0;
            f.have[CV_CPU_AVX_512VL]      = (cpuid_data[1] & (1<<31)) != 0;
            f.have[CV_CPU_AVX_512VBMI]    = (cpuid_data[2] &  (1<<1)) != 0;
        }

    #if defined ANDROID || defined __linux__
    #ifdef __aarch64__
        f.have[CV_CPU_NEON] = true;
        f.have[CV_CPU_FP16] = true;
    #elif defined __arm__
        int cpufile = open("/proc/self/auxv", O_RDONLY);

        if (cpufile >= 0)
        {
            Elf32_auxv_t auxv;
            const size_t size_auxv_t = sizeof(auxv);

            while ((size_t)read(cpufile, &auxv, size_auxv_t) == size_auxv_t)
            {
                if (auxv.a_type == AT_HWCAP)
                {
                    f.have[CV_CPU_NEON] = (auxv.a_un.a_val & 4096) != 0;
                    f.have[CV_CPU_FP16] = (auxv.a_un.a_val & 2) != 0;
                    break;
                }
            }

            close(cpufile);
        }
    #endif
    #elif (defined __clang__ || defined __APPLE__)
    #if (defined __ARM_NEON__ || (defined __ARM_NEON && defined __aarch64__))
        f.have[CV_CPU_NEON] = true;
    #endif
    #if (defined __ARM_FP  && (((__ARM_FP & 0x2) != 0) && defined __ARM_NEON__))
        f.have[CV_CPU_FP16] = true;
    #endif
    #endif

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

void setUseOptimized( bool flag )
{
    useOptimizedFlag = flag;
    currentFeatures = flag ? &featuresEnabled : &featuresDisabled;

    ipp::setUseIPP(flag);
#ifdef HAVE_OPENCL
    ocl::setUseOpenCL(flag);
#endif
#ifdef HAVE_TEGRA_OPTIMIZATION
    ::tegra::setUseTegra(flag);
#endif
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

//#ifdef HAVE_IPP
//int64 getCPUTickCount(void)
//{
//    return ippGetCpuClocks();
//}
//#else
int64 getCPUTickCount(void)
{
    return getTickCount();
}
//#endif

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
    AutoBuffer<char, 1024> buf;

    for ( ; ; )
    {
        va_list va;
        va_start(va, fmt);
        int bsize = static_cast<int>(buf.size()),
                len = vsnprintf((char *)buf, bsize, fmt, va);
        va_end(va);

        if (len < 0 || len >= bsize)
        {
            buf.resize(std::max(bsize << 1, len + 1));
            continue;
        }
        return String((char *)buf, len);
    }
}

String tempfile( const char* suffix )
{
    String fname;
#ifndef WINRT
    const char *temp_dir = getenv("OPENCV_TEMP_PATH");
#endif

#if defined WIN32 || defined _WIN32
#ifdef WINRT
    RoInitialize(RO_INIT_MULTITHREADED);
    std::wstring temp_dir = GetTempPathWinRT();

    std::wstring temp_file = GetTempFileNameWinRT(L"ocv");
    if (temp_file.empty())
        return String();

    temp_file = temp_dir.append(std::wstring(L"\\")).append(temp_file);
    DeleteFileW(temp_file.c_str());

    char aname[MAX_PATH];
    size_t copied = wcstombs(aname, temp_file.c_str(), MAX_PATH);
    CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
    fname = String(aname);
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
    case CV_GpuNotSupported :        return "No CUDA support";
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

namespace cv {
bool __termination = false;
}

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

#else

struct Mutex::Impl
{
    Impl()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&mt, &attr);
        pthread_mutexattr_destroy(&attr);

        refcount = 1;
    }
    ~Impl() { pthread_mutex_destroy(&mt); }

    void lock() { pthread_mutex_lock(&mt); }
    bool trylock() { return pthread_mutex_trylock(&mt) == 0; }
    void unlock() { pthread_mutex_unlock(&mt); }

    pthread_mutex_t mt;
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


//////////////////////////////// thread-local storage ////////////////////////////////

#ifdef WIN32
#ifdef _MSC_VER
#pragma warning(disable:4505) // unreferenced local function has been removed
#endif
#ifndef TLS_OUT_OF_INDEXES
#define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
#endif

// TLS platform abstraction layer
class TlsAbstraction
{
public:
    TlsAbstraction();
    ~TlsAbstraction();
    void* GetData() const;
    void  SetData(void *pData);

private:
#ifdef WIN32
#ifndef WINRT
    DWORD tlsKey;
#endif
#else // WIN32
    pthread_key_t  tlsKey;
#endif
};

#ifdef WIN32
#ifdef WINRT
static __declspec( thread ) void* tlsData = NULL; // using C++11 thread attribute for local thread data
TlsAbstraction::TlsAbstraction() {}
TlsAbstraction::~TlsAbstraction() {}
void* TlsAbstraction::GetData() const
{
    return tlsData;
}
void  TlsAbstraction::SetData(void *pData)
{
    tlsData = pData;
}
#else //WINRT
TlsAbstraction::TlsAbstraction()
{
    tlsKey = TlsAlloc();
    CV_Assert(tlsKey != TLS_OUT_OF_INDEXES);
}
TlsAbstraction::~TlsAbstraction()
{
    TlsFree(tlsKey);
}
void* TlsAbstraction::GetData() const
{
    return TlsGetValue(tlsKey);
}
void  TlsAbstraction::SetData(void *pData)
{
    CV_Assert(TlsSetValue(tlsKey, pData) == TRUE);
}
#endif
#else // WIN32
TlsAbstraction::TlsAbstraction()
{
    CV_Assert(pthread_key_create(&tlsKey, NULL) == 0);
}
TlsAbstraction::~TlsAbstraction()
{
    CV_Assert(pthread_key_delete(tlsKey) == 0);
}
void* TlsAbstraction::GetData() const
{
    return pthread_getspecific(tlsKey);
}
void  TlsAbstraction::SetData(void *pData)
{
    CV_Assert(pthread_setspecific(tlsKey, pData) == 0);
}
#endif

// Per-thread data structure
struct ThreadData
{
    ThreadData()
    {
        idx = 0;
        slots.reserve(32);
    }

    std::vector<void*> slots; // Data array for a thread
    size_t idx;               // Thread index in TLS storage. This is not OS thread ID!
};

// Main TLS storage class
class TlsStorage
{
public:
    TlsStorage()
    {
        tlsSlots.reserve(32);
        threads.reserve(32);
    }
    ~TlsStorage()
    {
        for(size_t i = 0; i < threads.size(); i++)
        {
            if(threads[i])
            {
                /* Current architecture doesn't allow proper global objects relase, so this check can cause crashes

                // Check if all slots were properly cleared
                for(size_t j = 0; j < threads[i]->slots.size(); j++)
                {
                    CV_Assert(threads[i]->slots[j] == 0);
                }
                */
                delete threads[i];
            }
        }
        threads.clear();
    }

    void releaseThread()
    {
        AutoLock guard(mtxGlobalAccess);
        ThreadData *pTD = (ThreadData*)tls.GetData();
        for(size_t i = 0; i < threads.size(); i++)
        {
            if(pTD == threads[i])
            {
                threads[i] = 0;
                break;
            }
        }
        tls.SetData(0);
        delete pTD;
    }

    // Reserve TLS storage index
    size_t reserveSlot()
    {
        AutoLock guard(mtxGlobalAccess);

        // Find unused slots
        for(size_t slot = 0; slot < tlsSlots.size(); slot++)
        {
            if(!tlsSlots[slot])
            {
                tlsSlots[slot] = 1;
                return slot;
            }
        }

        // Create new slot
        tlsSlots.push_back(1);
        return (tlsSlots.size()-1);
    }

    // Release TLS storage index and pass assosiated data to caller
    void releaseSlot(size_t slotIdx, std::vector<void*> &dataVec)
    {
        AutoLock guard(mtxGlobalAccess);
        CV_Assert(tlsSlots.size() > slotIdx);

        for(size_t i = 0; i < threads.size(); i++)
        {
            if(threads[i])
            {
                std::vector<void*>& thread_slots = threads[i]->slots;
                if (thread_slots.size() > slotIdx && thread_slots[slotIdx])
                {
                    dataVec.push_back(thread_slots[slotIdx]);
                    threads[i]->slots[slotIdx] = 0;
                }
            }
        }

        tlsSlots[slotIdx] = 0;
    }

    // Get data by TLS storage index
    void* getData(size_t slotIdx) const
    {
        CV_Assert(tlsSlots.size() > slotIdx);

        ThreadData* threadData = (ThreadData*)tls.GetData();
        if(threadData && threadData->slots.size() > slotIdx)
            return threadData->slots[slotIdx];

        return NULL;
    }

    // Gather data from threads by TLS storage index
    void gather(size_t slotIdx, std::vector<void*> &dataVec)
    {
        AutoLock guard(mtxGlobalAccess);
        CV_Assert(tlsSlots.size() > slotIdx);

        for(size_t i = 0; i < threads.size(); i++)
        {
            if(threads[i])
            {
                std::vector<void*>& thread_slots = threads[i]->slots;
                if (thread_slots.size() > slotIdx && thread_slots[slotIdx])
                    dataVec.push_back(thread_slots[slotIdx]);
            }
        }
    }

    // Set data to storage index
    void setData(size_t slotIdx, void* pData)
    {
        CV_Assert(tlsSlots.size() > slotIdx && pData != NULL);

        ThreadData* threadData = (ThreadData*)tls.GetData();
        if(!threadData)
        {
            threadData = new ThreadData;
            tls.SetData((void*)threadData);
            {
                AutoLock guard(mtxGlobalAccess);
                threadData->idx = threads.size();
                threads.push_back(threadData);
            }
        }

        if(slotIdx >= threadData->slots.size())
        {
            AutoLock guard(mtxGlobalAccess);
            while(slotIdx >= threadData->slots.size())
                threadData->slots.push_back(NULL);
        }
        threadData->slots[slotIdx] = pData;
    }

private:
    TlsAbstraction tls; // TLS abstraction layer instance

    Mutex  mtxGlobalAccess;           // Shared objects operation guard
    std::vector<int> tlsSlots;        // TLS keys state
    std::vector<ThreadData*> threads; // Array for all allocated data. Thread data pointers are placed here to allow data cleanup
};

// Create global TLS storage object
static TlsStorage &getTlsStorage()
{
    CV_SINGLETON_LAZY_INIT_REF(TlsStorage, new TlsStorage())
}

TLSDataContainer::TLSDataContainer()
{
    key_ = (int)getTlsStorage().reserveSlot(); // Reserve key from TLS storage
}

TLSDataContainer::~TLSDataContainer()
{
    CV_Assert(key_ == -1); // Key must be released in child object
}

void TLSDataContainer::gatherData(std::vector<void*> &data) const
{
    getTlsStorage().gather(key_, data);
}

void TLSDataContainer::release()
{
    std::vector<void*> data;
    data.reserve(32);
    getTlsStorage().releaseSlot(key_, data); // Release key and get stored data for proper destruction
    for(size_t i = 0; i < data.size(); i++)  // Delete all assosiated data
        deleteDataInstance(data[i]);
    key_ = -1;
}

void* TLSDataContainer::getData() const
{
    void* pData = getTlsStorage().getData(key_); // Check if data was already allocated
    if(!pData)
    {
        // Create new data instance and save it to TLS storage
        pData = createDataInstance();
        getTlsStorage().setData(key_, pData);
    }
    return pData;
}

TLSData<CoreTLSData>& getCoreTlsData()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<CoreTLSData>, new TLSData<CoreTLSData>())
}

#if defined CVAPI_EXPORTS && defined WIN32 && !defined WINCE
#ifdef WINRT
    #pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
#endif

extern "C"
BOOL WINAPI DllMain(HINSTANCE, DWORD fdwReason, LPVOID lpReserved);

extern "C"
BOOL WINAPI DllMain(HINSTANCE, DWORD fdwReason, LPVOID lpReserved)
{
    if (fdwReason == DLL_THREAD_DETACH || fdwReason == DLL_PROCESS_DETACH)
    {
        if (lpReserved != NULL) // called after ExitProcess() call
        {
            cv::__termination = true;
        }
        else
        {
            // Not allowed to free resources if lpReserved is non-null
            // http://msdn.microsoft.com/en-us/library/windows/desktop/ms682583.aspx
            cv::deleteThreadAllocData();
            cv::getTlsStorage().releaseThread();
        }
    }
    return TRUE;
}
#endif

#ifdef CV_COLLECT_IMPL_DATA
ImplCollector& getImplData()
{
    CV_SINGLETON_LAZY_INIT_REF(ImplCollector, new ImplCollector())
}

void setImpl(int flags)
{
    cv::AutoLock lock(getImplData().mutex);

    getImplData().implFlags = flags;
    getImplData().implCode.clear();
    getImplData().implFun.clear();
}

void addImpl(int flag, const char* func)
{
    cv::AutoLock lock(getImplData().mutex);

    getImplData().implFlags |= flag;
    if(func) // use lazy collection if name was not specified
    {
        size_t index = getImplData().implCode.size();
        if(!index || (getImplData().implCode[index-1] != flag || getImplData().implFun[index-1].compare(func))) // avoid duplicates
        {
            getImplData().implCode.push_back(flag);
            getImplData().implFun.push_back(func);
        }
    }
}

int getImpl(std::vector<int> &impl, std::vector<String> &funName)
{
    cv::AutoLock lock(getImplData().mutex);

    impl    = getImplData().implCode;
    funName = getImplData().implFun;
    return getImplData().implFlags; // return actual flags for lazy collection
}

bool useCollection()
{
    return getImplData().useCollection;
}

void setUseCollection(bool flag)
{
    cv::AutoLock lock(getImplData().mutex);

    getImplData().useCollection = flag;
}
#endif

namespace ipp
{

struct IPPInitSingelton
{
public:
    IPPInitSingelton()
    {
        useIPP      = true;
        ippStatus   = 0;
        funcname    = NULL;
        filename    = NULL;
        linen       = 0;
        ippFeatures = 0;

#ifdef HAVE_IPP
        const char* pIppEnv = getenv("OPENCV_IPP");
        cv::String env = pIppEnv;
        if(env.size())
        {
            if(env == "disabled")
            {
                std::cerr << "WARNING: IPP was disabled by OPENCV_IPP environment variable" << std::endl;
                useIPP = false;
            }
#if IPP_VERSION_X100 >= 900
            else if(env == "sse")
                ippFeatures = ippCPUID_SSE;
            else if(env == "sse2")
                ippFeatures = ippCPUID_SSE2;
            else if(env == "sse3")
                ippFeatures = ippCPUID_SSE3;
            else if(env == "ssse3")
                ippFeatures = ippCPUID_SSSE3;
            else if(env == "sse41")
                ippFeatures = ippCPUID_SSE41;
            else if(env == "sse42")
                ippFeatures = ippCPUID_SSE42;
            else if(env == "avx")
                ippFeatures = ippCPUID_AVX;
            else if(env == "avx2")
                ippFeatures = ippCPUID_AVX2;
#endif
            else
                std::cerr << "ERROR: Improper value of OPENCV_IPP: " << env.c_str() << std::endl;
        }

        IPP_INITIALIZER(ippFeatures)
#endif
    }

    bool useIPP;

    int         ippStatus; // 0 - all is ok, -1 - IPP functions failed
    const char *funcname;
    const char *filename;
    int         linen;
    int         ippFeatures;
};

static IPPInitSingelton& getIPPSingelton()
{
    CV_SINGLETON_LAZY_INIT_REF(IPPInitSingelton, new IPPInitSingelton())
}

int getIppFeatures()
{
#ifdef HAVE_IPP
    return getIPPSingelton().ippFeatures;
#else
    return 0;
#endif
}

void setIppStatus(int status, const char * const _funcname, const char * const _filename, int _line)
{
    getIPPSingelton().ippStatus = status;
    getIPPSingelton().funcname = _funcname;
    getIPPSingelton().filename = _filename;
    getIPPSingelton().linen = _line;
}

int getIppStatus()
{
    return getIPPSingelton().ippStatus;
}

String getIppErrorLocation()
{
    return format("%s:%d %s", getIPPSingelton().filename ? getIPPSingelton().filename : "", getIPPSingelton().linen, getIPPSingelton().funcname ? getIPPSingelton().funcname : "");
}

bool useIPP()
{
#ifdef HAVE_IPP
    CoreTLSData* data = getCoreTlsData().get();
    if(data->useIPP < 0)
    {
        data->useIPP = getIPPSingelton().useIPP;
    }
    return (data->useIPP > 0);
#else
    return false;
#endif
}

void setUseIPP(bool flag)
{
    CoreTLSData* data = getCoreTlsData().get();
#ifdef HAVE_IPP
    data->useIPP = flag;
#else
    (void)flag;
    data->useIPP = false;
#endif
}

} // namespace ipp

} // namespace cv

#ifdef HAVE_TEGRA_OPTIMIZATION

namespace tegra {

bool useTegra()
{
    cv::CoreTLSData* data = cv::getCoreTlsData().get();

    if (data->useTegra < 0)
    {
        const char* pTegraEnv = getenv("OPENCV_TEGRA");
        if (pTegraEnv && (cv::String(pTegraEnv) == "disabled"))
            data->useTegra = false;
        else
            data->useTegra = true;
    }

    return (data->useTegra > 0);
}

void setUseTegra(bool flag)
{
    cv::CoreTLSData* data = cv::getCoreTlsData().get();
    data->useTegra = flag;
}

} // namespace tegra

#endif

/* End of file. */
