// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP
#define OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP

#include "./allocator_stats.hpp"

//#define OPENCV_DISABLE_ALLOCATOR_STATS

#include <atomic>

#ifndef OPENCV_ALLOCATOR_STATS_COUNTER_TYPE
#if defined(__GNUC__) && (\
        (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4) || \
        (defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4) && !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8)) \
    )
#define OPENCV_ALLOCATOR_STATS_COUNTER_TYPE int
#endif
#endif

#ifndef OPENCV_ALLOCATOR_STATS_COUNTER_TYPE
#define OPENCV_ALLOCATOR_STATS_COUNTER_TYPE long long
#endif

namespace cv { namespace utils {

#ifdef CV__ALLOCATOR_STATS_LOG
namespace {
#endif

class AllocatorStatistics : public AllocatorStatisticsInterface
{
#ifdef OPENCV_DISABLE_ALLOCATOR_STATS

public:
    AllocatorStatistics() {}
    ~AllocatorStatistics() CV_OVERRIDE {}

    uint64_t getCurrentUsage() const CV_OVERRIDE { return 0; }
    uint64_t getTotalUsage() const CV_OVERRIDE { return 0; }
    uint64_t getNumberOfAllocations() const CV_OVERRIDE { return 0; }
    uint64_t getPeakUsage() const CV_OVERRIDE { return 0; }

    /** set peak usage = current usage */
    void resetPeakUsage() CV_OVERRIDE {};

    void onAllocate(size_t /*sz*/) {}
    void onFree(size_t /*sz*/) {}

#else

protected:
    typedef OPENCV_ALLOCATOR_STATS_COUNTER_TYPE counter_t;
    std::atomic<counter_t> curr, total, total_allocs, peak;
public:
    AllocatorStatistics() {}
    ~AllocatorStatistics() CV_OVERRIDE {}

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

        counter_t new_curr = curr.fetch_add((counter_t)sz) + (counter_t)sz;

        // peak = std::max((uint64_t)peak, new_curr);
        auto prev_peak = peak.load();
        while (prev_peak < new_curr)
        {
            if (peak.compare_exchange_weak(prev_peak, new_curr))
                break;
        }
        // end of peak = max(...)

        total += (counter_t)sz;
        total_allocs++;
    }
    void onFree(size_t sz)
    {
#ifdef CV__ALLOCATOR_STATS_LOG
        CV__ALLOCATOR_STATS_LOG(cv::format("free: %lld (curr=%lld)", (long long int)sz, (long long int)curr.load()));
#endif
        curr -= (counter_t)sz;
    }
#endif // OPENCV_DISABLE_ALLOCATOR_STATS
};

#ifdef CV__ALLOCATOR_STATS_LOG
} // namespace
#endif

}} // namespace

#endif // OPENCV_CORE_ALLOCATOR_STATS_IMPL_HPP
