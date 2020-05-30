// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACE_PRIVATE_HPP
#define OPENCV_TRACE_PRIVATE_HPP

#ifdef OPENCV_TRACE

#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/tls.hpp>

#include "trace.hpp"

//! @cond IGNORED

#include <deque>
#include <ostream>

#define INTEL_ITTNOTIFY_API_PRIVATE 1
#ifdef OPENCV_WITH_ITT
#include "ittnotify.h"
#endif

#ifndef DEBUG_ONLY
#ifdef _DEBUG
#define DEBUG_ONLY(...) __VA_ARGS__
#else
#define DEBUG_ONLY(...) (void)0
#endif
#endif

#ifndef DEBUG_ONLY_
#ifdef _DEBUG
#define DEBUG_ONLY_(...) __VA_ARGS__
#else
#define DEBUG_ONLY_(...)
#endif
#endif


namespace cv {
namespace utils {
namespace trace {
namespace details {

#define CV__TRACE_OPENCV_FUNCTION_NAME_(name, flags) \
    CV__TRACE_DEFINE_LOCATION_FN(name, flags); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));


enum RegionFlag {
    REGION_FLAG__NEED_STACK_POP = (1 << 0),
    REGION_FLAG__ACTIVE = (1 << 1),

    ENUM_REGION_FLAG_IMPL_FORCE_INT = INT_MAX
};


class TraceMessage;

class TraceStorage {
public:
    TraceStorage() {}
    virtual ~TraceStorage() {};

    virtual bool put(const TraceMessage& msg) const = 0;
};

struct RegionStatistics
{
    int currentSkippedRegions;

    int64 duration;
#ifdef HAVE_IPP
    int64 durationImplIPP;
#endif
#ifdef HAVE_OPENCL
    int64 durationImplOpenCL;
#endif
#ifdef HAVE_OPENVX
    int64 durationImplOpenVX;
#endif

    RegionStatistics() :
        currentSkippedRegions(0),
        duration(0)
#ifdef HAVE_IPP
        ,durationImplIPP(0)
#endif
#ifdef HAVE_OPENCL
        ,durationImplOpenCL(0)
#endif
#ifdef HAVE_OPENVX
        ,durationImplOpenVX(0)
#endif
    {}

    void grab(RegionStatistics& result)
    {
        result.currentSkippedRegions = currentSkippedRegions; currentSkippedRegions = 0;
        result.duration = duration; duration = 0;
#ifdef HAVE_IPP
        result.durationImplIPP = durationImplIPP; durationImplIPP = 0;
#endif
#ifdef HAVE_OPENCL
        result.durationImplOpenCL = durationImplOpenCL; durationImplOpenCL = 0;
#endif
#ifdef HAVE_OPENVX
        result.durationImplOpenVX = durationImplOpenVX; durationImplOpenVX = 0;
#endif
    }

    void append(RegionStatistics& stat)
    {
        currentSkippedRegions += stat.currentSkippedRegions;
        duration += stat.duration;
#ifdef HAVE_IPP
        durationImplIPP += stat.durationImplIPP;
#endif
#ifdef HAVE_OPENCL
        durationImplOpenCL += stat.durationImplOpenCL;
#endif
#ifdef HAVE_OPENVX
        durationImplOpenVX += stat.durationImplOpenVX;
#endif
    }

    void multiply(const float c)
    {
        duration = (int64)(duration * c);
#ifdef HAVE_IPP
        durationImplIPP = (int64)(durationImplIPP * c);
#endif
#ifdef HAVE_OPENCL
        durationImplOpenCL = (int64)(durationImplOpenCL * c);
#endif
#ifdef HAVE_OPENVX
        durationImplOpenVX = (int64)(durationImplOpenVX * c);
#endif
    }
};

static inline
std::ostream& operator<<(std::ostream& out, const RegionStatistics& stat)
{
    out << "skip=" << stat.currentSkippedRegions
        << " duration=" << stat.duration
#ifdef HAVE_IPP
        << " durationImplIPP=" << stat.durationImplIPP
#endif
#ifdef HAVE_OPENCL
        << " durationImplOpenCL=" << stat.durationImplOpenCL
#endif
#ifdef HAVE_OPENVX
        << " durationImplOpenVX=" << stat.durationImplOpenVX
#endif
    ;
    return out;
}

struct RegionStatisticsStatus
{
    int _skipDepth;
#ifdef HAVE_IPP
    int ignoreDepthImplIPP;
#endif
#ifdef HAVE_OPENCL
    int ignoreDepthImplOpenCL;
#endif
#ifdef HAVE_OPENVX
    int ignoreDepthImplOpenVX;
#endif

    RegionStatisticsStatus() { reset(); }

    void reset()
    {
        _skipDepth = -1;
#ifdef HAVE_IPP
        ignoreDepthImplIPP = 0;
#endif
#ifdef HAVE_OPENCL
        ignoreDepthImplOpenCL = 0;
#endif
#ifdef HAVE_OPENVX
        ignoreDepthImplOpenVX = 0;
#endif
    }

    void propagateFrom(const RegionStatisticsStatus& src)
    {
        _skipDepth = -1;
        if (src._skipDepth >= 0)
            enableSkipMode(0);
#ifdef HAVE_IPP
        ignoreDepthImplIPP = src.ignoreDepthImplIPP ? 1 : 0;
#endif
#ifdef HAVE_OPENCL
        ignoreDepthImplOpenCL = src.ignoreDepthImplOpenCL ? 1 : 0;
#endif
#ifdef HAVE_OPENVX
        ignoreDepthImplOpenVX = src.ignoreDepthImplOpenVX ? 1 : 0;
#endif
    }

    void enableSkipMode(int depth);
    void checkResetSkipMode(int leaveDepth);
};

static inline
std::ostream& operator<<(std::ostream& out, const RegionStatisticsStatus& s)
{
    out << "ignore={";
    if (s._skipDepth >= 0)
        out << " SKIP=" << s._skipDepth;
#ifdef HAVE_IPP
    if (s.ignoreDepthImplIPP)
        out << " IPP=" << s.ignoreDepthImplIPP;
#endif
#ifdef HAVE_OPENCL
    if (s.ignoreDepthImplOpenCL)
        out << " OpenCL=" << s.ignoreDepthImplOpenCL;
#endif
#ifdef HAVE_OPENVX
    if (s.ignoreDepthImplOpenVX)
        out << " OpenVX=" << s.ignoreDepthImplOpenVX;
#endif
    out << "}";
    return out;
}

//! TraceManager for local thread
struct TraceManagerThreadLocal
{
    const int threadID;
    int region_counter;

    size_t totalSkippedEvents;

    Region* currentActiveRegion;

    struct StackEntry
    {
        Region* region;
        const Region::LocationStaticStorage* location;
        int64 beginTimestamp;
        StackEntry(Region* region_, const Region::LocationStaticStorage* location_, int64 beginTimestamp_) :
            region(region_), location(location_), beginTimestamp(beginTimestamp_)
        {}
        StackEntry() : region(NULL), location(NULL), beginTimestamp(-1) {}
    };
    std::deque<StackEntry> stack;

    int regionDepth;                   // functions only (no named regions)
    int regionDepthOpenCV;             // functions from OpenCV library

    RegionStatistics stat;
    RegionStatisticsStatus stat_status;

    StackEntry dummy_stack_top;        // parallel_for root region
    RegionStatistics parallel_for_stat;
    RegionStatisticsStatus parallel_for_stat_status;
    size_t parallel_for_stack_size;


    mutable cv::Ptr<TraceStorage> storage;

    TraceManagerThreadLocal() :
        threadID(cv::utils::getThreadID()),
        region_counter(0), totalSkippedEvents(0),
        currentActiveRegion(NULL),
        regionDepth(0),
        regionDepthOpenCV(0),
        parallel_for_stack_size(0)
    {
    }

    ~TraceManagerThreadLocal();

    TraceStorage* getStorage() const;

    void recordLocation(const Region::LocationStaticStorage& location);
    void recordRegionEnter(const Region& region);
    void recordRegionLeave(const Region& region, const RegionStatistics& result);
    void recordRegionArg(const Region& region, const TraceArg& arg, const char& value);

    inline void stackPush(Region* region, const Region::LocationStaticStorage* location, int64 beginTimestamp)
    {
        stack.push_back(StackEntry(region, location, beginTimestamp));
    }
    inline Region* stackTopRegion() const
    {
        if (stack.empty())
            return dummy_stack_top.region;
        return stack.back().region;
    }
    inline const Region::LocationStaticStorage* stackTopLocation() const
    {
        if (stack.empty())
            return dummy_stack_top.location;
        return stack.back().location;
    }
    inline int64 stackTopBeginTimestamp() const
    {
        if (stack.empty())
            return dummy_stack_top.beginTimestamp;
        return stack.back().beginTimestamp;
    }
    inline void stackPop()
    {
        CV_DbgAssert(!stack.empty());
        stack.pop_back();
    }
    void dumpStack(std::ostream& out, bool onlyFunctions) const;

    inline Region* getCurrentActiveRegion()
    {
        return currentActiveRegion;
    }

    inline int getCurrentDepth() const { return (int)stack.size(); }
};

class CV_EXPORTS TraceManager
{
public:
    TraceManager();
    ~TraceManager();

    static bool isActivated();

    Mutex mutexCreate;
    Mutex mutexCount;

    TLSDataAccumulator<TraceManagerThreadLocal> tls;

    cv::Ptr<TraceStorage> trace_storage;
private:
    // disable copying
    TraceManager(const TraceManager&);
    TraceManager& operator=(const TraceManager&);
};

CV_EXPORTS TraceManager& getTraceManager();
inline Region* getCurrentActiveRegion() { return getTraceManager().tls.get()->getCurrentActiveRegion(); }
inline Region* getCurrentRegion() { return getTraceManager().tls.get()->stackTopRegion(); }

void parallelForSetRootRegion(const Region& rootRegion, const TraceManagerThreadLocal& root_ctx);
void parallelForAttachNestedRegion(const Region& rootRegion);
void parallelForFinalize(const Region& rootRegion);







struct Region::LocationExtraData
{
    int global_location_id; // 0 - region is disabled
#ifdef OPENCV_WITH_ITT
    // Special fields for ITT
    __itt_string_handle* volatile ittHandle_name;
    __itt_string_handle* volatile ittHandle_filename;
#endif
    LocationExtraData(const LocationStaticStorage& location);

    static Region::LocationExtraData* init(const Region::LocationStaticStorage& location);
};

class Region::Impl
{
public:
    const LocationStaticStorage& location;

    Region& region;
    Region* const parentRegion;

    const int threadID;
    const int global_region_id;

    const int64 beginTimestamp;
    int64 endTimestamp;

    int directChildrenCount;

    enum OptimizationPath {
        CODE_PATH_PLAIN = 0,
        CODE_PATH_IPP,
        CODE_PATH_OPENCL,
        CODE_PATH_OPENVX
    };

#ifdef OPENCV_WITH_ITT
    bool itt_id_registered;
    __itt_id itt_id;
#endif

    Impl(TraceManagerThreadLocal& ctx, Region* parentRegion_, Region& region_, const LocationStaticStorage& location_, int64 beginTimestamp_);

    void enterRegion(TraceManagerThreadLocal& ctx);
    void leaveRegion(TraceManagerThreadLocal& ctx);

    void registerRegion(TraceManagerThreadLocal& ctx);

    void release();
protected:
    ~Impl();
};



}}}} // namespace

//! @endcond

#endif

#endif // OPENCV_TRACE_PRIVATE_HPP
