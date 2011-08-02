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

#if defined WIN32 || defined _WIN32 || defined WINCE
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
#include <sys/sysctl.h>
#endif

namespace cv
{

Exception::Exception() { code = 0; line = 0; }

Exception::Exception(int _code, const string& _err, const string& _func, const string& _file, int _line)
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
            f.have[CV_CPU_AVX]    = (cpuid_data[2] & (1<<28)) != 0;
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

#ifdef HAVE_IPP
volatile bool useOptimizedFlag = true;

struct IPPInitializer
{
    IPPInitializer(void) { ippStaticInit(); }
};

IPPInitializer ippInitializer;
#else
volatile bool useOptimizedFlag = false;
#endif

volatile bool USE_SSE2 = false;

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


static int numThreads = 0;
static int numProcs   = 0;

int getNumThreads(void)
{
    if( !numProcs )
        setNumThreads(0);
    return numThreads;
}

void setNumThreads( int
#ifdef _OPENMP
                             threads
#endif
                  )
{
    if( !numProcs )
    {
#ifdef _OPENMP
        numProcs = omp_get_num_procs();
#else
        numProcs = 1;
#endif
    }

#ifdef _OPENMP
    if( threads <= 0 )
        threads = numProcs;
    else
        threads = MIN( threads, numProcs );

    numThreads = threads;
#else
    numThreads = 1;
#endif
}


int getThreadNum(void)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

int getNumberOfCPUs(void)
{
#if defined WIN32 || defined _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    
    return (int)sysinfo.dwNumberOfProcessors;
#elif defined __linux__
    return (int)sysconf( _SC_NPROCESSORS_ONLN );
#elif defined __APPLE__
    int numCPU=0;
    int mib[4];
    size_t len = sizeof(numCPU); 
    
    /* set the mib for hw.ncpu */
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;
    
    /* get the number of CPUs from the system */
    sysctl(mib, 2, &numCPU, &len, NULL, 0);
    
    if( numCPU < 1 ) 
    {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );
        
        if( numCPU < 1 )
            numCPU = 1;
    }
    
    return (int)numCPU;
#else
    return 1;
#endif
}

string format( const char* fmt, ... )
{
    char buf[1 << 16];
    va_list args;
    va_start( args, fmt );
    vsprintf( buf, fmt, args );
    return string(buf);
}

string tempfile( const char* suffix )
{
    char buf[L_tmpnam];
    char* name = 0;
#if ANDROID
    strcpy(buf, "/sdcard/__opencv_temp_XXXXXX");
    name = mktemp(buf);
#else
    name = tmpnam(buf);
#endif
    if (*name == '\\')
        ++name;
    string n(name);
    if (suffix != 0)
        n += (n[n.size()-1] == '.' && suffix[0] == '.' ? suffix + 1 : suffix);
    return n;
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
    }

    if(breakOnError)
    {
        static volatile int* p = 0;
        *p = 0;
    }

    throw exc;
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

/*CV_IMPL int
cvGuiBoxReport( int code, const char *func_name, const char *err_msg,
                const char *file, int line, void* )
{
#if (!defined WIN32 && !defined _WIN32) || defined WINCE
    return cvStdErrReport( code, func_name, err_msg, file, line, 0 );
#else
    if( code != CV_StsBackTrace && code != CV_StsAutoTrace )
    {
        size_t msg_len = strlen(err_msg ? err_msg : "") + 1024;
        char* message = (char*)alloca(msg_len);
        char title[100];

        wsprintf( message, "%s (%s)\nin function %s, %s(%d)\n\n"
                  "Press \"Abort\" to terminate application.\n"
                  "Press \"Retry\" to debug (if the app is running under debugger).\n"
                  "Press \"Ignore\" to continue (this is not safe).\n",
                  cvErrorStr(code), err_msg ? err_msg : "no description",
                  func_name, file, line );

        wsprintf( title, "OpenCV GUI Error Handler" );

        int answer = MessageBox( NULL, message, title, MB_ICONERROR|MB_ABORTRETRYIGNORE|MB_SYSTEMMODAL );

        if( answer == IDRETRY )
        {
            CV_DBG_BREAK();
        }
        return answer != IDIGNORE;
    }
    return 0;
#endif
}*/

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

CV_IMPL void cvSetNumThreads(int nt)
{
    cv::setNumThreads(nt);
}

CV_IMPL int cvGetNumThreads()
{
    return cv::getNumThreads();
}

CV_IMPL int cvGetThreadNum()
{
    return cv::getThreadNum();
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
    case CV_GpuNotSupported : return "No GPU support";
    case CV_GpuApiCallError : return "Gpu Api call";
    case CV_GpuNppCallError : return "Npp Api call";
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

static CvModuleInfo cxcore_info = { 0, "cxcore", CV_VERSION, 0 };

CvModuleInfo* CvModule::first = 0, *CvModule::last = 0;

CvModule::CvModule( CvModuleInfo* _info )
{
    cvRegisterModule( _info );
    info = last;
}

CvModule::~CvModule(void)
{
    if( info )
    {
        CvModuleInfo* p = first;
        for( ; p != 0 && p->next != info; p = p->next )
            ;

        if( p )
            p->next = info->next;

        if( first == info )
            first = info->next;

        if( last == info )
            last = p;

        free( info );
        info = 0;
    }
}

CV_IMPL int
cvRegisterModule( const CvModuleInfo* module )
{
    CV_Assert( module != 0 && module->name != 0 && module->version != 0 );

    size_t name_len = strlen(module->name);
    size_t version_len = strlen(module->version);

    CvModuleInfo* module_copy = (CvModuleInfo*)malloc( sizeof(*module_copy) +
                                name_len + 1 + version_len + 1 );

    *module_copy = *module;
    module_copy->name = (char*)(module_copy + 1);
    module_copy->version = (char*)(module_copy + 1) + name_len + 1;

    memcpy( (void*)module_copy->name, module->name, name_len + 1 );
    memcpy( (void*)module_copy->version, module->version, version_len + 1 );
    module_copy->next = 0;

    if( CvModule::first == 0 )
        CvModule::first = module_copy;
    else
        CvModule::last->next = module_copy;

    CvModule::last = module_copy;

    return 0;
}

CvModule cxcore_module( &cxcore_info );

CV_IMPL void
cvGetModuleInfo( const char* name, const char **version, const char **plugin_list )
{
    static char joint_verinfo[1024]   = "";
    static char plugin_list_buf[1024] = "";

    if( version )
        *version = 0;

    if( plugin_list )
        *plugin_list = 0;

    CvModuleInfo* module;

    if( version )
    {
        if( name )
        {
            size_t i, name_len = strlen(name);

            for( module = CvModule::first; module != 0; module = module->next )
            {
                if( strlen(module->name) == name_len )
                {
                    for( i = 0; i < name_len; i++ )
                    {
                        int c0 = toupper(module->name[i]), c1 = toupper(name[i]);
                        if( c0 != c1 )
                            break;
                    }
                    if( i == name_len )
                        break;
                }
            }
            if( !module )
                CV_Error( CV_StsObjectNotFound, "The module is not found" );

            *version = module->version;
        }
        else
        {
            char* ptr = joint_verinfo;

            for( module = CvModule::first; module != 0; module = module->next )
            {
                sprintf( ptr, "%s: %s%s", module->name, module->version, module->next ? ", " : "" );
                ptr += strlen(ptr);
            }

            *version = joint_verinfo;
        }
    }

    if( plugin_list )
        *plugin_list = plugin_list_buf;
}

#if defined BUILD_SHARED_LIBS && defined CVAPI_EXPORTS && defined WIN32 && !defined WINCE
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

/* End of file. */
