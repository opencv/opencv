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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/trace.private.hpp>

#if defined _WIN32 || defined WINCE
    #include <windows.h>
    #undef small
    #undef min
    #undef max
    #undef abs
#endif

#if defined __linux__ || defined __APPLE__ || defined __GLIBC__ \
    || defined __HAIKU__
    #include <unistd.h>
    #include <stdio.h>
    #include <sys/types.h>
    #if defined __ANDROID__
        #include <sys/sysconf.h>
    #elif defined __APPLE__
        #include <sys/sysctl.h>
    #endif
#endif

#ifdef _OPENMP
    #define HAVE_OPENMP
#endif

#ifdef __APPLE__
    #define HAVE_GCD
#endif

#if defined _MSC_VER && _MSC_VER >= 1600
    #define HAVE_CONCURRENCY
#endif

/* IMPORTANT: always use the same order of defines
   - HAVE_TBB         - 3rdparty library, should be explicitly enabled
   - HAVE_HPX         - 3rdparty library, should be explicitly enabled
   - HAVE_OPENMP      - integrated to compiler, should be explicitly enabled
   - HAVE_GCD         - system wide, used automatically        (APPLE only)
   - WINRT            - system wide, used automatically        (Windows RT only)
   - HAVE_CONCURRENCY - part of runtime, used automatically    (Windows only - MSVS 10, MSVS 11)
   - HAVE_PTHREADS_PF - pthreads if available
*/

#if defined HAVE_TBB
    #include "tbb/tbb.h"
    #include "tbb/task.h"
    #include "tbb/tbb_stddef.h"
    #if TBB_INTERFACE_VERSION >= 8000
        #include "tbb/task_arena.h"
    #endif
    #undef min
    #undef max
#elif defined HAVE_HPX
    #include <hpx/parallel/algorithms/for_loop.hpp>
    #include <hpx/parallel/execution.hpp>
    //
    #include <hpx/hpx_start.hpp>
    #include <hpx/hpx_suspend.hpp>
    #include <hpx/include/apply.hpp>
    #include <hpx/util/yield_while.hpp>
    #include <hpx/include/threadmanager.hpp>

#elif defined HAVE_OPENMP
    #include <omp.h>
#elif defined HAVE_GCD
    #include <dispatch/dispatch.h>
    #include <pthread.h>
#elif defined WINRT && _MSC_VER < 1900
    #include <ppltasks.h>
#elif defined HAVE_CONCURRENCY
    #include <ppl.h>
#endif


#if defined HAVE_TBB
#  define CV_PARALLEL_FRAMEWORK "tbb"
#elif defined HAVE_HPX
#  define CV_PARALLEL_FRAMEWORK "hpx"
#elif defined HAVE_OPENMP
#  define CV_PARALLEL_FRAMEWORK "openmp"
#elif defined HAVE_GCD
#  define CV_PARALLEL_FRAMEWORK "gcd"
#elif defined WINRT
#  define CV_PARALLEL_FRAMEWORK "winrt-concurrency"
#elif defined HAVE_CONCURRENCY
#  define CV_PARALLEL_FRAMEWORK "ms-concurrency"
#elif defined HAVE_PTHREADS_PF
#  define CV_PARALLEL_FRAMEWORK "pthreads"
#endif

#include "parallel_impl.hpp"


#ifndef CV__EXCEPTION_PTR
#  if defined(__ANDROID__) && defined(ATOMIC_INT_LOCK_FREE) && ATOMIC_INT_LOCK_FREE < 2
#    define CV__EXCEPTION_PTR 0  // Not supported, details: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58938
#  elif defined(CV_CXX11)
#    define CV__EXCEPTION_PTR 1
#  elif defined(_MSC_VER)
#    define CV__EXCEPTION_PTR (_MSC_VER >= 1600)
#  elif defined(__clang__)
#    define CV__EXCEPTION_PTR 0  // C++11 only (see above)
#  elif defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define CV__EXCEPTION_PTR (__GXX_EXPERIMENTAL_CXX0X__ > 0)
#  endif
#endif
#ifndef CV__EXCEPTION_PTR
#  define CV__EXCEPTION_PTR 0
#elif CV__EXCEPTION_PTR
#  include <exception>  // std::exception_ptr
#endif



using namespace cv;

namespace cv
{
    ParallelLoopBody::~ParallelLoopBody() {}
}

namespace
{
#ifdef CV_PARALLEL_FRAMEWORK
#ifdef ENABLE_INSTRUMENTATION
    static void SyncNodes(cv::instr::InstrNode *pNode)
    {
        std::vector<cv::instr::NodeDataTls*> data;
        pNode->m_payload.m_tls.gather(data);

        uint64 ticksMax = 0;
        int    threads  = 0;
        for(size_t i = 0; i < data.size(); i++)
        {
            if(data[i] && data[i]->m_ticksTotal)
            {
                ticksMax = MAX(ticksMax, data[i]->m_ticksTotal);
                pNode->m_payload.m_ticksTotal -= data[i]->m_ticksTotal;
                data[i]->m_ticksTotal = 0;
                threads++;
            }
        }
        pNode->m_payload.m_ticksTotal += ticksMax;
        pNode->m_payload.m_threads = MAX(pNode->m_payload.m_threads, threads);

        for(size_t i = 0; i < pNode->m_childs.size(); i++)
            SyncNodes(pNode->m_childs[i]);
    }
#endif

    class ParallelLoopBodyWrapperContext
    {
    public:
        ParallelLoopBodyWrapperContext(const cv::ParallelLoopBody& _body, const cv::Range& _r, double _nstripes) :
            is_rng_used(false), hasException(false)
        {

            body = &_body;
            wholeRange = _r;
            double len = wholeRange.end - wholeRange.start;
            nstripes = cvRound(_nstripes <= 0 ? len : MIN(MAX(_nstripes, 1.), len));

            // propagate main thread state
            rng = cv::theRNG();

#ifdef OPENCV_TRACE
            traceRootRegion = CV_TRACE_NS::details::getCurrentRegion();
            traceRootContext = CV_TRACE_NS::details::getTraceManager().tls.get();
#endif

#ifdef ENABLE_INSTRUMENTATION
            pThreadRoot = cv::instr::getInstrumentTLSStruct().pCurrentNode;
#endif
        }
        void finalize()
        {
#ifdef ENABLE_INSTRUMENTATION
            for(size_t i = 0; i < pThreadRoot->m_childs.size(); i++)
                SyncNodes(pThreadRoot->m_childs[i]);
#endif
            if (is_rng_used)
            {
                // Some parallel backends execute nested jobs in the main thread,
                // so we need to restore initial RNG state here.
                cv::theRNG() = rng;
                // We can't properly update RNG state based on RNG usage in worker threads,
                // so lets just change main thread RNG state to the next value.
                // Note: this behaviour is not equal to single-threaded mode.
                cv::theRNG().next();
            }
#ifdef OPENCV_TRACE
            if (traceRootRegion)
                CV_TRACE_NS::details::parallelForFinalize(*traceRootRegion);
#endif

            if (hasException)
            {
#if CV__EXCEPTION_PTR
                std::rethrow_exception(pException);
#else
                CV_Error(Error::StsError, "Exception in parallel_for() body: " + exception_message);
#endif
            }
        }
        ~ParallelLoopBodyWrapperContext() {}

        const cv::ParallelLoopBody* body;
        cv::Range wholeRange;
        int nstripes;
        cv::RNG rng;
        mutable bool is_rng_used;
#ifdef OPENCV_TRACE
        CV_TRACE_NS::details::Region* traceRootRegion;
        CV_TRACE_NS::details::TraceManagerThreadLocal* traceRootContext;
#endif
#ifdef ENABLE_INSTRUMENTATION
        cv::instr::InstrNode *pThreadRoot;
#endif
        bool hasException;
#if CV__EXCEPTION_PTR
        std::exception_ptr pException;
#else
        cv::String exception_message;
#endif
#if CV__EXCEPTION_PTR
        void recordException()
#else
        void recordException(const cv::String& msg)
#endif
        {
            if (!hasException)
            {
                cv::AutoLock lock(cv::getInitializationMutex());
                if (!hasException)
                {
                    hasException = true;
#if CV__EXCEPTION_PTR
                    pException = std::current_exception();
#else
                    exception_message = msg;
#endif
                }
            }
        }
    private:
        ParallelLoopBodyWrapperContext(const ParallelLoopBodyWrapperContext&); // disabled
        ParallelLoopBodyWrapperContext& operator=(const ParallelLoopBodyWrapperContext&); // disabled
    };

    class ParallelLoopBodyWrapper : public cv::ParallelLoopBody
    {
    public:
        ParallelLoopBodyWrapper(ParallelLoopBodyWrapperContext& ctx_) :
            ctx(ctx_)
        {
        }
        ~ParallelLoopBodyWrapper()
        {
        }
        void operator()(const cv::Range& sr) const CV_OVERRIDE
        {
#ifdef OPENCV_TRACE
            // TODO CV_TRACE_NS::details::setCurrentRegion(rootRegion);
            if (ctx.traceRootRegion && ctx.traceRootContext)
                CV_TRACE_NS::details::parallelForSetRootRegion(*ctx.traceRootRegion, *ctx.traceRootContext);
            CV__TRACE_OPENCV_FUNCTION_NAME("parallel_for_body");
            if (ctx.traceRootRegion)
                CV_TRACE_NS::details::parallelForAttachNestedRegion(*ctx.traceRootRegion);
#endif

#ifdef ENABLE_INSTRUMENTATION
            {
                cv::instr::InstrTLSStruct *pInstrTLS = &cv::instr::getInstrumentTLSStruct();
                pInstrTLS->pCurrentNode = ctx.pThreadRoot; // Initialize TLS node for thread
            }
            CV_INSTRUMENT_REGION();
#endif

            // propagate main thread state
            cv::theRNG() = ctx.rng;

            cv::Range r;
            cv::Range wholeRange = ctx.wholeRange;
            int nstripes = ctx.nstripes;
            r.start = (int)(wholeRange.start +
                            ((uint64)sr.start*(wholeRange.end - wholeRange.start) + nstripes/2)/nstripes);
            r.end = sr.end >= nstripes ? wholeRange.end : (int)(wholeRange.start +
                            ((uint64)sr.end*(wholeRange.end - wholeRange.start) + nstripes/2)/nstripes);

#ifdef OPENCV_TRACE
            CV_TRACE_ARG_VALUE(range_start, "range.start", (int64)r.start);
            CV_TRACE_ARG_VALUE(range_end, "range.end", (int64)r.end);
#endif

            try
            {
                (*ctx.body)(r);
            }
#if CV__EXCEPTION_PTR
            catch (...)
            {
                ctx.recordException();
            }
#else
            catch (const cv::Exception& e)
            {
                ctx.recordException(e.what());
            }
            catch (const std::exception& e)
            {
                ctx.recordException(e.what());
            }
            catch (...)
            {
                ctx.recordException("Unknown exception");
            }
#endif

            if (!ctx.is_rng_used && !(cv::theRNG() == ctx.rng))
                ctx.is_rng_used = true;
        }
        cv::Range stripeRange() const { return cv::Range(0, ctx.nstripes); }

    protected:
        ParallelLoopBodyWrapperContext& ctx;
    };

#if defined HAVE_TBB
    class ProxyLoopBody : public ParallelLoopBodyWrapper
    {
    public:
        ProxyLoopBody(ParallelLoopBodyWrapperContext& ctx_)
        : ParallelLoopBodyWrapper(ctx_)
        {}

        void operator ()(const tbb::blocked_range<int>& range) const
        {
            this->ParallelLoopBodyWrapper::operator()(cv::Range(range.begin(), range.end()));
        }

        void operator ()() const  // run parallel job
        {
            cv::Range range = this->stripeRange();
            tbb::parallel_for(tbb::blocked_range<int>(range.start, range.end), *this);
        }
    };
#elif defined HAVE_HPX
    class ProxyLoopBody : public ParallelLoopBodyWrapper
    {
    public:
        ProxyLoopBody(ParallelLoopBodyWrapperContext& ctx_)
                : ParallelLoopBodyWrapper(ctx_)
        {}

        void operator ()() const  // run parallel job
        {
            cv::Range stripeRange = this->stripeRange();
            hpx::parallel::for_loop(
                    hpx::parallel::execution::par,
                    stripeRange.start, stripeRange.end,
                    [&](const int &i) { ;
                        this->ParallelLoopBodyWrapper::operator()(
                                cv::Range(i, i + 1));
                    });
        }
    };
#elif defined HAVE_OPENMP
    typedef ParallelLoopBodyWrapper ProxyLoopBody;
#elif defined HAVE_GCD
    typedef ParallelLoopBodyWrapper ProxyLoopBody;
    static void block_function(void* context, size_t index)
    {
        ProxyLoopBody* ptr_body = static_cast<ProxyLoopBody*>(context);
        (*ptr_body)(cv::Range((int)index, (int)index + 1));
    }
#elif defined WINRT || defined HAVE_CONCURRENCY
    class ProxyLoopBody : public ParallelLoopBodyWrapper
    {
    public:
        ProxyLoopBody(ParallelLoopBodyWrapperContext& ctx)
        : ParallelLoopBodyWrapper(ctx)
        {}

        void operator ()(int i) const
        {
            this->ParallelLoopBodyWrapper::operator()(cv::Range(i, i + 1));
        }
    };
#else
    typedef ParallelLoopBodyWrapper ProxyLoopBody;
#endif

static int numThreads = -1;

#if defined HAVE_TBB
    #if TBB_INTERFACE_VERSION >= 8000
        static tbb::task_arena tbbArena(tbb::task_arena::automatic);
    #else
        static tbb::task_scheduler_init tbbScheduler(tbb::task_scheduler_init::deferred);
    #endif
#elif defined HAVE_HPX
// nothing for HPX
#elif defined HAVE_OPENMP
static inline int _initMaxThreads()
{
    int maxThreads = omp_get_max_threads();
    if (!utils::getConfigurationParameterBool("OPENCV_FOR_OPENMP_DYNAMIC_DISABLE", false))
    {
        omp_set_dynamic(maxThreads);
    }
    return numThreads;
}
static int numThreadsMax = _initMaxThreads();
#elif defined HAVE_GCD
// nothing for GCD
#elif defined WINRT
// nothing for WINRT
#elif defined HAVE_CONCURRENCY

class SchedPtr
{
    Concurrency::Scheduler* sched_;
public:
    Concurrency::Scheduler* operator->() { return sched_; }
    operator Concurrency::Scheduler*() { return sched_; }

    void operator=(Concurrency::Scheduler* sched)
    {
        if (sched_) sched_->Release();
        sched_ = sched;
    }

    SchedPtr() : sched_(0) {}
    ~SchedPtr() {}
};
static SchedPtr pplScheduler;

#endif

#endif // CV_PARALLEL_FRAMEWORK

} //namespace

/* ================================   parallel_for_  ================================ */

#ifdef CV_PARALLEL_FRAMEWORK
static void parallel_for_impl(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes); // forward declaration
#endif

void cv::parallel_for_(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
#ifdef OPENCV_TRACE
    CV__TRACE_OPENCV_FUNCTION_NAME_("parallel_for", 0);
    CV_TRACE_ARG_VALUE(range_start, "range.start", (int64)range.start);
    CV_TRACE_ARG_VALUE(range_end, "range.end", (int64)range.end);
    CV_TRACE_ARG_VALUE(nstripes, "nstripes", (int64)nstripes);
#endif

    CV_INSTRUMENT_REGION_MT_FORK();
    if (range.empty())
        return;

#ifdef CV_PARALLEL_FRAMEWORK
    static volatile int flagNestedParallelFor = 0;
    bool isNotNestedRegion = flagNestedParallelFor == 0;
    if (isNotNestedRegion)
      isNotNestedRegion = CV_XADD(&flagNestedParallelFor, 1) == 0;
    if (isNotNestedRegion)
    {
        try
        {
            parallel_for_impl(range, body, nstripes);
            flagNestedParallelFor = 0;
        }
        catch (...)
        {
            flagNestedParallelFor = 0;
            throw;
        }
    }
    else // nested parallel_for_() calls are not parallelized
#endif // CV_PARALLEL_FRAMEWORK
    {
        CV_UNUSED(nstripes);
        body(range);
    }
}

#ifdef CV_PARALLEL_FRAMEWORK
static void parallel_for_impl(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
    if ((numThreads < 0 || numThreads > 1) && range.end - range.start > 1)
    {
        ParallelLoopBodyWrapperContext ctx(body, range, nstripes);
        ProxyLoopBody pbody(ctx);
        cv::Range stripeRange = pbody.stripeRange();
        if( stripeRange.end - stripeRange.start == 1 )
        {
            body(range);
            return;
        }

#if defined HAVE_TBB

#if TBB_INTERFACE_VERSION >= 8000
        tbbArena.execute(pbody);
#else
        pbody();
#endif

#elif defined HAVE_HPX
        pbody();

#elif defined HAVE_OPENMP

        #pragma omp parallel for schedule(dynamic) num_threads(numThreads > 0 ? numThreads : numThreadsMax)
        for (int i = stripeRange.start; i < stripeRange.end; ++i)
            pbody(Range(i, i + 1));

#elif defined HAVE_GCD

        dispatch_queue_t concurrent_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        dispatch_apply_f(stripeRange.end - stripeRange.start, concurrent_queue, &pbody, block_function);

#elif defined WINRT

        Concurrency::parallel_for(stripeRange.start, stripeRange.end, pbody);

#elif defined HAVE_CONCURRENCY

        if(!pplScheduler || pplScheduler->Id() == Concurrency::CurrentScheduler::Id())
        {
            Concurrency::parallel_for(stripeRange.start, stripeRange.end, pbody);
        }
        else
        {
            pplScheduler->Attach();
            Concurrency::parallel_for(stripeRange.start, stripeRange.end, pbody);
            Concurrency::CurrentScheduler::Detach();
        }

#elif defined HAVE_PTHREADS_PF

        parallel_for_pthreads(pbody.stripeRange(), pbody, pbody.stripeRange().size());

#else

#error You have hacked and compiling with unsupported parallel framework

#endif

        ctx.finalize();  // propagate exceptions if exists
    }
    else
    {
        body(range);
    }
}
#endif // CV_PARALLEL_FRAMEWORK


int cv::getNumThreads(void)
{
#ifdef CV_PARALLEL_FRAMEWORK

    if(numThreads == 0)
        return 1;

#endif

#if defined HAVE_TBB

#if TBB_INTERFACE_VERSION >= 9100
    return tbbArena.max_concurrency();
#elif TBB_INTERFACE_VERSION >= 8000
    return numThreads > 0
        ? numThreads
        : tbb::task_scheduler_init::default_num_threads();
#else
    return tbbScheduler.is_active()
           ? numThreads
           : tbb::task_scheduler_init::default_num_threads();
#endif

#elif defined HAVE_HPX
    return numThreads;

#elif defined HAVE_OPENMP

    return numThreads > 0
           ? numThreads
           : numThreadsMax;


#elif defined HAVE_GCD

    return cv::getNumberOfCPUs(); // the GCD thread pool limit

#elif defined WINRT

    return 0;

#elif defined HAVE_CONCURRENCY

    return 1 + (pplScheduler == 0
        ? Concurrency::CurrentScheduler::Get()->GetNumberOfVirtualProcessors()
        : pplScheduler->GetNumberOfVirtualProcessors());

#elif defined HAVE_PTHREADS_PF

        return parallel_pthreads_get_threads_num();

#else

    return 1;

#endif
}

namespace cv {
unsigned defaultNumberOfThreads()
{
#ifdef __ANDROID__
    // many modern phones/tables have 4-core CPUs. Let's use no more
    // than 2 threads by default not to overheat the devices
    const unsigned int default_number_of_threads = 2;
#else
    const unsigned int default_number_of_threads = (unsigned int)std::max(1, cv::getNumberOfCPUs());
#endif

    unsigned result = default_number_of_threads;

    static int config_num_threads = (int)utils::getConfigurationParameterSizeT("OPENCV_FOR_THREADS_NUM", 0);

    if (config_num_threads)
    {
        result = (unsigned)std::max(1, config_num_threads);
        //do we need upper limit of threads number?
    }
    return result;
}
}

void cv::setNumThreads( int threads_ )
{
    CV_UNUSED(threads_);
#ifdef CV_PARALLEL_FRAMEWORK
    int threads = (threads_ < 0) ? defaultNumberOfThreads() : (unsigned)threads_;
    numThreads = threads;
#endif

#ifdef HAVE_TBB

#if TBB_INTERFACE_VERSION >= 8000
    if(tbbArena.is_active()) tbbArena.terminate();
    if(threads > 0) tbbArena.initialize(threads);
#else
    if(tbbScheduler.is_active()) tbbScheduler.terminate();
    if(threads > 0) tbbScheduler.initialize(threads);
#endif

#elif defined HAVE_HPX
    return; // nothing needed as numThreads is used

#elif defined HAVE_OPENMP

    return; // nothing needed as num_threads clause is used in #pragma omp parallel for

#elif defined HAVE_GCD

    // unsupported
    // there is only private dispatch_queue_set_width() and only for desktop

#elif defined WINRT

    return;

#elif defined HAVE_CONCURRENCY

    if (threads <= 0)
    {
        pplScheduler = 0;
    }
    else if (threads == 1)
    {
        // Concurrency always uses >=2 threads, so we just disable it if 1 thread is requested
        numThreads = 0;
    }
    else if (pplScheduler == 0 || 1 + pplScheduler->GetNumberOfVirtualProcessors() != (unsigned int)threads)
    {
        pplScheduler = Concurrency::Scheduler::Create(Concurrency::SchedulerPolicy(2,
                       Concurrency::MinConcurrency, threads-1,
                       Concurrency::MaxConcurrency, threads-1));
    }

#elif defined HAVE_PTHREADS_PF

    parallel_pthreads_set_threads_num(threads);

#endif
}


int cv::getThreadNum(void)
{
#if defined HAVE_TBB
    #if TBB_INTERFACE_VERSION >= 9100
        return tbb::this_task_arena::current_thread_index();
    #elif TBB_INTERFACE_VERSION >= 8000
        return tbb::task_arena::current_thread_index();
    #else
        return 0;
    #endif
#elif defined HAVE_HPX
        return (int)(hpx::get_num_worker_threads());
#elif defined HAVE_OPENMP
    return omp_get_thread_num();
#elif defined HAVE_GCD
    return (int)(size_t)(void*)pthread_self(); // no zero-based indexing
#elif defined WINRT
    return 0;
#elif defined HAVE_CONCURRENCY
    return std::max(0, (int)Concurrency::Context::VirtualProcessorId()); // zero for master thread, unique number for others but not necessary 1,2,3,...
#elif defined HAVE_PTHREADS_PF
    return (int)(size_t)(void*)pthread_self(); // no zero-based indexing
#else
    return 0;
#endif
}

#ifdef __ANDROID__
static inline int getNumberOfCPUsImpl()
{
   FILE* cpuPossible = fopen("/sys/devices/system/cpu/possible", "r");
   if(!cpuPossible)
       return 1;

   char buf[2000]; //big enough for 1000 CPUs in worst possible configuration
   char* pbuf = fgets(buf, sizeof(buf), cpuPossible);
   fclose(cpuPossible);
   if(!pbuf)
      return 1;

   //parse string of form "0-1,3,5-7,10,13-15"
   int cpusAvailable = 0;

   while(*pbuf)
   {
      const char* pos = pbuf;
      bool range = false;
      while(*pbuf && *pbuf != ',')
      {
          if(*pbuf == '-') range = true;
          ++pbuf;
      }
      if(*pbuf) *pbuf++ = 0;
      if(!range)
        ++cpusAvailable;
      else
      {
          int rstart = 0, rend = 0;
          sscanf(pos, "%d-%d", &rstart, &rend);
          cpusAvailable += rend - rstart + 1;
      }

   }
   return cpusAvailable ? cpusAvailable : 1;
}
#endif

int cv::getNumberOfCPUs(void)
{
#if defined _WIN32
    SYSTEM_INFO sysinfo;
#if (defined(_M_ARM) || defined(_M_X64) || defined(WINRT)) && _WIN32_WINNT >= 0x501
    GetNativeSystemInfo( &sysinfo );
#else
    GetSystemInfo( &sysinfo );
#endif

    return (int)sysinfo.dwNumberOfProcessors;
#elif defined __ANDROID__
    static int ncpus = getNumberOfCPUsImpl();
    return ncpus;
#elif defined __linux__ || defined __GLIBC__ || defined __HAIKU__
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

const char* cv::currentParallelFramework() {
#ifdef CV_PARALLEL_FRAMEWORK
    return CV_PARALLEL_FRAMEWORK;
#else
    return NULL;
#endif
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
