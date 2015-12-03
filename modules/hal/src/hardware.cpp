#include "precomp.hpp"

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h>
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
#endif

#if defined ANDROID || defined __linux__
#  include <unistd.h>
#  include <fcntl.h>
#  include <elf.h>
#  include <linux/auxvec.h>
#endif

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
    #else
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
                    break;
                }
            }

            close(cpufile);
        }
    #endif
    #elif (defined __clang__ || defined __APPLE__) && (defined __ARM_NEON__ || (defined __ARM_NEON && defined __aarch64__))
        f.have[CV_CPU_NEON] = true;
    #endif

        return f;
    }

    int x86_family;
    bool have[MAX_FEATURE+1];
};

static HWFeatures  featuresEnabled = HWFeatures::initialize(), featuresDisabled = HWFeatures();
static HWFeatures* currentFeatures = &featuresEnabled;
volatile bool useOptimizedFlag = true;

namespace cv { namespace hal {

bool checkHardwareSupport(int feature)
{
//    CV_DbgAssert( 0 <= feature && feature <= CV_HARDWARE_MAX_FEATURE );
    return currentFeatures->have[feature];
}

void setUseOptimized( bool flag )
{
    useOptimizedFlag = flag;
    currentFeatures = flag ? &featuresEnabled : &featuresDisabled;
}

bool useOptimized(void)
{
    return useOptimizedFlag;
}

}}
