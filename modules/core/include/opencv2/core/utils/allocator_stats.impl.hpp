// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP
#define OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP

#include "./allocator_stats.hpp"

#ifdef CV_CXX11
#include <atomic>
#endif

namespace cv { namespace utils {

#ifdef CV__ALLOCATOR_STATS_LOG
namespace {
#endif

class AllocatorStatistics : public AllocatorStatisticsInterface
{
protected:
#ifdef CV_CXX11
    std::atomic<long long> curr, total, total_allocs, peak;
#else
    volatile long long curr, total, total_allocs, peak;  // overflow is possible, CV_XADD operates with 'int' only
#endif

public:
    AllocatorStatistics()
#ifndef CV_CXX11
        : curr(0), total(0), total_allocs(0), peak(0)
#endif
    {}
    ~AllocatorStatistics() CV_OVERRIDE {}

    // AllocatorStatisticsInterface

#ifdef CV_CXX11
    uint64_t getCurrentUsage() const CV_OVERRIDE { return (uint64_t)curr.load(); }
    uint64_t getTotalUsage() const CV_OVERRIDE { return (uint64_t)total.load(); }
    uint64_t getNumberOfAllocations() const CV_OVERRIDE { return (uint64_t)total_allocs.load(); }
    uint64_t getPeakUsage() const CV_OVERRIDE { return (uint64_t)peak.load(); }

    /** set peak usage = current usage */
    void resetPeakUsage() CV_OVERRIDE { peak.store(curr.load()); }

    // Controller interface
    void onAllocate(size_t sz)
    {
#ifdef CV__ALLOCATOR_STATS_LOG
        CV__ALLOCATOR_STATS_LOG(cv::format("allocate: %lld (curr=%lld)", (long long int)sz, (long long int)curr.load()));
#endif

        long long new_curr = curr.fetch_add((long long)sz) + (long long)sz;

        // peak = std::max((uint64_t)peak, new_curr);
        auto prev_peak = peak.load();
        while (prev_peak < new_curr)
        {
            if (peak.compare_exchange_weak(prev_peak, new_curr))
                break;
        }
        // end of peak = max(...)

        total += (long long)sz;
        total_allocs++;
    }
    void onFree(size_t sz)
    {
#ifdef CV__ALLOCATOR_STATS_LOG
        CV__ALLOCATOR_STATS_LOG(cv::format("free: %lld (curr=%lld)", (long long int)sz, (long long int)curr.load()));
#endif
        curr -= (long long)sz;
    }

#else
    uint64_t getCurrentUsage() const CV_OVERRIDE { return (uint64_t)curr; }
    uint64_t getTotalUsage() const CV_OVERRIDE { return (uint64_t)total; }
    uint64_t getNumberOfAllocations() const CV_OVERRIDE { return (uint64_t)total_allocs; }
    uint64_t getPeakUsage() const CV_OVERRIDE { return (uint64_t)peak; }

    void resetPeakUsage() CV_OVERRIDE { peak = curr; }

    // Controller interface
    void onAllocate(size_t sz)
    {
#ifdef CV__ALLOCATOR_STATS_LOG
        CV__ALLOCATOR_STATS_LOG(cv::format("allocate: %lld (curr=%lld)", (long long int)sz, (long long int)curr));
#endif

        uint64_t new_curr = (uint64_t)CV_XADD(&curr, (uint64_t)sz) + sz;

        peak = std::max((uint64_t)peak, new_curr);  // non-thread safe

        //CV_XADD(&total, (uint64_t)sz);  // overflow with int, non-reliable...
        total += sz;

        CV_XADD(&total_allocs, (uint64_t)1);
    }
    void onFree(size_t sz)
    {
#ifdef CV__ALLOCATOR_STATS_LOG
        CV__ALLOCATOR_STATS_LOG(cv::format("free: %lld (curr=%lld)", (long long int)sz, (long long int)curr));
#endif
        CV_XADD(&curr, (uint64_t)-sz);
    }
#endif
};

#ifdef CV__ALLOCATOR_STATS_LOG
} // namespace
#endif

}} // namespace

#endif // OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP
