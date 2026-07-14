// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_ALLOCATOR_STATS_HPP
#define OPENCV_CORE_ALLOCATOR_STATS_HPP

#include "../cvdef.h"

namespace cv { namespace utils {

class AllocatorStatisticsInterface
{
protected:
    AllocatorStatisticsInterface() {}
    virtual ~AllocatorStatisticsInterface() {}
public:
    virtual uint64_t getCurrentUsage() const = 0;
    virtual uint64_t getTotalUsage() const = 0;
    virtual uint64_t getNumberOfAllocations() const = 0;
    virtual uint64_t getPeakUsage() const = 0;

    /** set peak usage = current usage */
    virtual void resetPeakUsage() = 0;
};

}} // namespace

#endif // OPENCV_CORE_ALLOCATOR_STATS_HPP
