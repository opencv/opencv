// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: one cv:: call on an accelerator.
//
// 1. Describe every argument (where its memory lives) and pick the device: the innermost
//    ExecutionScope, otherwise the device of the first argument that lives on one.
// 2. Plan each argument: used in place, uploaded into a temporary, created on the device,
//    or produced in a temporary and copied to where the caller keeps it.
// 3. QUERY the backend. If it declines, outputs are still created on the device (so
//    where data ends up does not depend on which operations a backend implements) and
//    the call runs on the CPU through the coherence protocol.
// 4. Acquire every buffer for the device, make the stream wait for conflicting work,
//    EXECUTE, record one fence for all arguments, copy temporaries back.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace accel { namespace detail {

using namespace impl;

namespace {

enum Plan
{
    PLAN_UNDECIDED = -1,    //!< given, not planned yet (planArgs decides)
    PLAN_ABSENT = 0,        //!< optional argument that was not given
    PLAN_IN_PLACE = 1,      //!< the device works on the argument's own memory
    PLAN_UPLOAD = 2,        //!< input copied into a temporary on the device
    PLAN_CREATE = 3,        //!< output (re)allocated on the device
    PLAN_COPY_BACK = 4      //!< output produced in a temporary, then copied to the caller's array
};

struct Arg
{
    Arg() : isOutput(false), type(-1), fullOverwrite(true), isUMat(false), isHost(false),
            onDevice(false), mapped(false), exists(false), explicitAlloc(NULL), plan(PLAN_UNDECIDED), mode(0)
    {
        memset(&desc, 0, sizeof(desc));
        memset(&viewAlloc, 0, sizeof(viewAlloc));
    }
    bool isOutput;
    _OutputArray arr;           //!< outputs, a copy: the caller's may be a temporary (inputs are kept as UMat / Mat below)
    MatShape shape;
    int type;
    bool fullOverwrite;
    bool isUMat;
    bool isHost;                //!< plain host memory (Mat, or UMat from the host allocator)
    UMat um;
    Mat mat;
    cvAccelMemoryDesc desc;
    bool onDevice;              //!< memory from an accelerator backend
    bool mapped;                //!< a Mat view of accelerator memory (UMat::getMat): the CPU has that memory open
    bool exists;                //!< a UMat output that already has the shape and type the call produces
    const MatAllocator* explicitAlloc;  //!< a UMat output's own allocator: where it is created, whichever device runs the call
    int plan;
    UMat temp;
    int mode;
    cvAccelAllocation viewAlloc;
};

bool syncDebugMode()
{
    static bool v = utils::getConfigurationParameterBool("OPENCV_ACCEL_SYNC", false);
    return v;
}

bool sameShape(const UMat& m, const MatShape& s)
{
    if (m.dims != s.dims)
        return false;
    for (int i = 0; i < s.dims; i++)
        if (m.size.p[i] != s.p[i])
            return false;
    return true;
}

//! bytes from the first to the last byte touched by a strided view
size_t extentBytes(int dims, const int* sizes, const size_t* steps, int type)
{
    size_t e = CV_ELEM_SIZE(type);
    for (int i = 0; i < dims; i++)
    {
        if (sizes[i] <= 0)
            return 0;
        e += (size_t)(sizes[i] - 1) * steps[i];
    }
    return e;
}

/** Where the bytes of an argument are: offsets in one physical memory (@p key is its
    PhysicalBacking; a Mat view of accelerator memory included), offsets in a buffer of an
    allocator OpenCV does not know (@p key is its UMatData), or host addresses (@p key NULL). */
struct Span
{
    Span() : key(NULL), start(0), size(0) {}
    const void* key;
    uint64_t start, size;
};

//! the bytes @p m covers; host memory conservatively as its whole buffer
bool spanOfMat(const Mat& m, Span& s)
{
    if (m.empty())
        return false;
    const BufferState* v = m.u ? m.u->accelState : NULL;
    if (v)
    {
        // a mapping of a view: its Mat starts at the view's first byte (getHostView)
        s.key = v->backing;
        s.start = v->start + (uint64_t)(m.data - m.datastart);
        s.size = extentBytes(m.dims, m.size.p, m.step.p, m.type());
    }
    else
    {
        s.key = NULL;
        s.start = (uint64_t)(uintptr_t)m.datastart;
        s.size = (uint64_t)(m.dataend - m.datastart);
    }
    return true;
}

bool spanOfUMat(const UMat& m, bool host, Span& s)
{
    if (!m.u)
        return false;
    if (const BufferState* v = m.u->accelState)
    {
        s.key = v->backing;
        s.start = v->start + m.offset;
        s.size = extentBytes(m.dims, m.size.p, m.step.p, m.type());
    }
    else if (host && m.u->data)
    {
        s.key = NULL;
        s.start = (uint64_t)(uintptr_t)m.u->data;
        s.size = m.u->size;
    }
    else
    {
        s.key = m.u;
        s.start = m.offset;
        s.size = extentBytes(m.dims, m.size.p, m.step.p, m.type());
    }
    return true;
}

/** The argument @p a, whose Mat (when it is one) is @p m, shares bytes with @p b: in one
    buffer, in two views of one physical memory (aliases, planes), or through a Mat that maps
    accelerator memory and a UMat of the same memory. */
bool overlaps(const Arg& a, const Mat& m, const Arg& b)
{
    Span sa, sb;
    const bool ka = a.isUMat ? spanOfUMat(a.um, a.isHost, sa) : spanOfMat(m, sa);
    const bool kb = b.isUMat ? spanOfUMat(b.um, b.isHost, sb) : spanOfMat(b.mat, sb);
    return ka && kb && sa.key == sb.key && rangesOverlap(sa.start, sa.size, sb.start, sb.size);
}

/** Bytes the memory of @p m needs moved before the device uses it: in place (@p inPlace), a
    host copy newer than the device memory is uploaded first; uploaded elsewhere, a stale host
    copy is downloaded first. */
uint64_t pendingTransferBytes(const UMat& m, bool inPlace)
{
    const BufferState* v = m.u ? m.u->accelState : NULL;
    if (!v)
        return 0;
    const uint32_t c = v->imp->alloc.desc.host_access;
    if (c != CV_ACCEL_HOST_ACCESS_COPY && c != CV_ACCEL_HOST_ACCESS_UNCACHED)
        return 0;   // one physical copy: nothing to move
    std::lock_guard<std::mutex> lk(v->backing->mutex);
    const bool hostNewer = m.u->deviceCopyObsolete() && !m.u->hostCopyObsolete();
    const bool hostStale = m.u->hostCopyObsolete() && !m.u->deviceCopyObsolete();
    return (inPlace ? hostNewer : hostStale) ? v->size : 0;
}

//! the image the argument is a plane of, for the backend (cvAccelView::image)
void setImage(cvAccelView& v, const BufferState* bv)
{
    if (bv && bv->image)
    {
        v.image = &bv->image->desc;
        v.plane = (uint32_t)bv->plane;
    }
}

//! the bytes of @p v lie inside the range [start, start + size) of its allocation that the argument may use
bool viewFitsRange(const cvAccelView& v, uint64_t start, uint64_t size)
{
    if (v.offset < start)
        return false;
    cvAccelAllocation range;
    memset(&range, 0, sizeof(range));
    range.size = size;
    cvAccelView rel = v;
    rel.alloc = &range;
    rel.offset = v.offset - start;
    return viewFitsAllocation(rel);
}

//! the CPU can read and write the memory (a mapping or a host copy): data may move through it
bool cpuReachable(const cvAccelMemoryDesc& d)
{
    return d.host_access != CV_ACCEL_HOST_ACCESS_NONE && !(d.flags & CV_ACCEL_MEMF_PROTECTED) && d.layout_modifier == 0;
}

//! the device allocator @p a allocates on, NULL for other allocators
const DeviceAllocator* deviceAllocatorOf(const MatAllocator* a)
{
    return a && isDeviceAllocator(a) ? static_cast<const DeviceAllocator*>(a) : NULL;
}

//! an output without memory of the right shape gets it on (b, device): its own allocator, if any, allocates there
bool createdOn(const Arg& a, Backend* b, int device)
{
    if (!a.explicitAlloc)
        return true;
    const DeviceAllocator* da = deviceAllocatorOf(a.explicitAlloc);
    return da && da->backend() == b && da->device() == device;
}

bool coversAllocation(const UMat& m)
{
    return m.u && m.offset == 0 && m.isContinuous() && m.total() * m.elemSize() == m.u->size;
}

void fillView(cvAccelView& v, const cvAccelAllocation* alloc, size_t offset, int type,
              int dims, const int* sizes, const size_t* steps)
{
    memset(&v, 0, sizeof(v));
    v.alloc = alloc;
    v.offset = offset;
    v.type = type;
    v.dims = dims;
    for (int i = 0; i < dims && i < CV_ACCEL_MAX_DIMS; i++)
    {
        v.size[i] = sizes[i];
        v.step[i] = steps[i];
    }
}

//! view of a continuous array that does not exist yet (QUERY of a planned temporary)
//! what the backend will report for memory it has not allocated yet (see describe_allocation)
cvAccelMemoryDesc plannedDesc(Backend* b, int device, uint32_t kind, uint64_t size, uint32_t flags)
{
    cvAccelMemoryDesc d;
    memset(&d, 0, sizeof(d));
    if (!b->fn().describe_allocation || b->fn().describe_allocation(device, size, kind, flags, &d) != CV_ACCEL_OK)
    {
        memset(&d, 0, sizeof(d));   // the backend cannot tell: only what core knows for sure
        d.kind = kind;
        d.flags = flags;
    }
    d.backend_id = (uint32_t)b->id();
    d.device_index = (uint32_t)device;
    return d;
}

//! a continuous view of a buffer the call will allocate (an output or a temporary)
void fillPlannedView(cvAccelView& v, cvAccelAllocation& alloc, const MatShape& shape, int type,
                     const cvAccelMemoryDesc& desc)
{
    size_t steps[CV_ACCEL_MAX_DIMS];
    size_t total = CV_ELEM_SIZE(type);
    for (int i = shape.dims - 1; i >= 0; i--)
    {
        steps[i] = total;
        total *= (size_t)shape.p[i];
    }
    memset(&alloc, 0, sizeof(alloc));
    alloc.size = total;
    alloc.desc = desc;
    fillView(v, &alloc, 0, type, shape.dims, shape.p, steps);
}

size_t planBytes(const MatShape& shape, int type)
{
    size_t total = CV_ELEM_SIZE(type);
    for (int i = 0; i < shape.dims; i++)
        total *= (size_t)shape.p[i];
    return total;
}

void fillHostView(cvAccelView& v, cvAccelAllocation& alloc, const Mat& m)
{
    memset(&alloc, 0, sizeof(alloc));
    alloc.handle = (uint64_t)(uintptr_t)m.datastart;
    alloc.size = (uint64_t)(m.dataend - m.datastart);
    alloc.desc = hostMemoryDesc();
    fillView(v, &alloc, (size_t)(m.data - m.datastart), m.type(), m.dims, m.size.p, m.step.p);
}

/** A Mat argument: host memory, or a view of accelerator memory that a UMat mapped for the
    CPU. The view keeps its memory's description: it counts as being on that device, and a
    call that takes it runs on the CPU (the device must not touch memory the CPU has open). */
void describeMat(Arg& a, const Mat& m)
{
    if (m.u && m.u->accelState && queryMemoryDesc(m.u, a.desc))
    {
        a.mapped = true;
        a.onDevice = a.desc.backend_id != 0;
        a.isHost = false;
        return;
    }
    a.isHost = true;
    a.desc = hostMemoryDesc();
}

void describeUMat(Arg& a)
{
    if (!a.um.u)
        return;
    if (queryMemoryDesc(a.um.u, a.desc))
    {
        a.onDevice = a.desc.backend_id != 0;
        a.isHost = !a.onDevice && a.desc.kind == CV_ACCEL_MEM_HOST;
    }
    else
    {
        // an allocator that does not describe itself: treated as foreign memory
        memset(&a.desc, 0, sizeof(a.desc));
        a.onDevice = false;
        a.isHost = false;
    }
}

} // namespace

struct OpCall::Impl
{
    Impl(const char* name_, const char* module_, uint32_t version_)
        : name(name_), module(module_), version(version_), unsupported(false), protectedIn(false) {}
    const char* name;
    const char* module;
    uint32_t version;
    bool unsupported;
    bool protectedIn;   //!< an input holds protected content: so do the outputs and temporaries
    std::vector<Arg> args;

    RunResult run(Invoker fn, void* userdata);
private:
    /** Plans the arguments for @p device and asks the backend (QUERY; skipped when @p fn is
        NULL). @p cost: bytes the plan copies. @p protectedRefused: the plan would put results of
        protected content into ordinary memory. */
    cvAccelStatus evaluate(Backend* b, int device, Invoker fn, void* userdata, std::vector<cvAccelView>& views,
                           cvAccelCall& call, const void*& table, uint64_t& cost, bool& protectedRefused);
    bool planArgs(Backend* b, int device);
    void createOutput(Arg& a, Backend* b, int device);
};

OpCall::OpCall(const char* opName, const char* module, uint32_t moduleVersion)
    : p(new Impl(opName, module, moduleVersion))
{}

OpCall::~OpCall()
{
    delete p;
}

void OpCall::input(InputArray arr)
{
    Arg a;
    const _InputArray::KindFlag k = arr.kind();
    if (k == _InputArray::NONE || arr.empty())
        a.plan = PLAN_ABSENT;
    else if (k == _InputArray::UMAT)
    {
        a.isUMat = true;
        a.um = arr.getUMat();
        describeUMat(a);
        a.type = a.um.type();
        a.shape = a.um.shape();
    }
    else if (k == _InputArray::MAT)
    {
        a.mat = arr.getMat();
        describeMat(a, a.mat);
        a.type = a.mat.type();
        a.shape = a.mat.shape();
    }
    else
        p->unsupported = true;  // vectors, Matx, ...: the CPU path handles them
    p->args.push_back(a);
}

void OpCall::output(OutputArray arr, const MatShape& shape, int type, bool fullOverwrite)
{
    Arg a;
    a.isOutput = true;
    a.arr = arr;
    a.shape = shape;
    a.type = CV_MAT_TYPE(type);
    a.fullOverwrite = fullOverwrite;
    const _InputArray::KindFlag k = arr.kind();
    if (k == _InputArray::UMAT)
    {
        a.isUMat = true;
        a.um = arr.getUMatRef();
        describeUMat(a);
        a.exists = !a.um.empty() && sameShape(a.um, shape) && a.um.type() == a.type;
        a.explicitAlloc = arr.getUMatRef().allocator;   // explicit placement wins
    }
    else if (k == _InputArray::MAT)
        describeMat(a, arr.getMatRef());
    else
        p->unsupported = true;
    p->args.push_back(a);
}

RunResult OpCall::run(Invoker fn, void* userdata)
{
    return p->run(fn, userdata);
}

bool OpCall::Impl::planArgs(Backend* b, int device)
{
    for (size_t i = 0; i < args.size(); i++)
    {
        Arg& a = args[i];
        if (!a.isOutput)
        {
            if (a.plan == PLAN_ABSENT)
                continue;
            if ((a.onDevice || a.isHost) && b->canAccess(device, a.desc))
                a.plan = PLAN_IN_PLACE;
            else
                a.plan = PLAN_UPLOAD;
            continue;
        }
        bool exists = false;
        Mat hostOut;    // the caller's Mat, when it already has the right shape and type
        if (a.isUMat)
            exists = a.exists;
        else
        {
            hostOut = a.arr.getMat();
            exists = !hostOut.empty() && hostOut.shape() == a.shape && hostOut.type() == a.type;
            if (!exists)
                hostOut.release();
        }
        if (a.isUMat && !exists)
            a.plan = createdOn(a, b, device) ? PLAN_CREATE : PLAN_COPY_BACK;  // made elsewhere by its own allocator: copied there
        else if ((a.onDevice || a.isHost) && b->canAccess(device, a.desc))
            a.plan = PLAN_IN_PLACE;     // device memory, or host memory the device addresses (an iGPU, an NPU)
        else
            a.plan = PLAN_COPY_BACK;    // memory the device cannot write
        // an output that overlaps an input is produced in a temporary
        if (a.plan == PLAN_IN_PLACE)
        {
            for (size_t j = 0; j < args.size(); j++)
            {
                const Arg& in = args[j];
                if (!in.isOutput && in.plan == PLAN_IN_PLACE && overlaps(a, hostOut, in))
                    a.plan = PLAN_COPY_BACK;
            }
            if (a.plan == PLAN_IN_PLACE && !a.isUMat)
                a.mat = hostOut;    // empty when the Mat is created later, after QUERY
        }
    }
    return true;
}

void OpCall::Impl::createOutput(Arg& a, Backend* b, int device)
{
    UMat& d = a.arr.getUMatRef();
    if (a.explicitAlloc)
    {
        a.arr.create(a.shape, a.type);  // its own allocator, on this device (see createdOn)
        a.um = d;
        describeUMat(a);
        return;
    }
    MatAllocator* saved = d.allocator;
    d.allocator = b->allocator(device, (int)b->deviceInfo(device).preferred_kind, protectedIn);
    try
    {
        a.arr.create(a.shape, a.type);
    }
    catch (...)
    {
        d.allocator = saved;
        throw;
    }
    d.allocator = saved;   // placement belongs to the data, not to the header
    a.um = d;
    describeUMat(a);
}

cvAccelStatus OpCall::Impl::evaluate(Backend* b, int device, Invoker fn, void* userdata, std::vector<cvAccelView>& views,
                                     cvAccelCall& call, const void*& table, uint64_t& cost, bool& protectedRefused)
{
    planArgs(b, device);
    const uint32_t plannedKind = b->deviceInfo(device).preferred_kind;   // what outputs and temporaries get
    // Protected content (secure video) never reaches the CPU, and nothing computed from it may
    // land in ordinary memory: outputs and temporaries are protected too.
    protectedIn = false;
    protectedRefused = false;
    for (size_t i = 0; i < args.size(); i++)
        if (!args[i].isOutput && args[i].plan != PLAN_ABSENT && (args[i].desc.flags & CV_ACCEL_MEMF_PROTECTED))
            protectedIn = true;
    const uint32_t plannedFlags = protectedIn ? CV_ACCEL_MEMF_PROTECTED : 0;
    // A move between two devices goes through the CPU (UMat::copyTo), and copies are linear:
    // memory the CPU cannot read (protected, non-linear, without host access) is used where it
    // is, and only linear memory takes a copy back. A plan that needs more is not this device's
    // to run: the next candidate is tried.
    bool impossible = false;
    for (size_t i = 0; i < args.size(); i++)
    {
        const Arg& a = args[i];
        if (a.plan == PLAN_UPLOAD && a.onDevice && !cpuReachable(a.desc))
            impossible = true;
        if (a.plan == PLAN_COPY_BACK && a.isUMat)
        {
            const bool here = a.onDevice && a.desc.backend_id == (uint32_t)b->id() && (int)a.desc.device_index == device;
            if ((a.onDevice && a.desc.layout_modifier != 0) ||
                (!here && !cpuReachable(plannedDesc(b, device, plannedKind, planBytes(a.shape, a.type), plannedFlags))))
                impossible = true;  // the temporary goes through the CPU, into memory with the same layout
        }
    }
    // bytes the plan moves: temporaries filled or copied back, and what the arguments' own
    // memory needs first (a newer host copy uploaded, a stale one downloaded before an upload)
    cost = 0;
    for (size_t i = 0; i < args.size(); i++)
    {
        const Arg& a = args[i];
        if (protectedIn && a.isOutput && (a.plan == PLAN_IN_PLACE || a.plan == PLAN_COPY_BACK) &&
            !(a.desc.flags & CV_ACCEL_MEMF_PROTECTED))
            protectedRefused = true;
        if (protectedIn && a.isOutput && a.plan == PLAN_CREATE && a.explicitAlloc &&
            !(deviceAllocatorOf(a.explicitAlloc) && deviceAllocatorOf(a.explicitAlloc)->isProtected()))
            protectedRefused = true;    // its own allocator gives ordinary memory
        if (a.plan == PLAN_UPLOAD || a.plan == PLAN_COPY_BACK)
            cost += planBytes(a.shape, a.type) * (a.plan == PLAN_COPY_BACK && !a.fullOverwrite ? 2 : 1);
        if (a.isUMat && a.onDevice && (a.plan == PLAN_IN_PLACE || a.plan == PLAN_UPLOAD))
            cost += pendingTransferBytes(a.um, a.plan == PLAN_IN_PLACE);
    }
    if (protectedRefused || impossible || !fn)
        return CV_ACCEL_NOT_IMPLEMENTED;

    table = b->opTable(module, version);
    for (size_t i = 0; i < args.size(); i++)
    {
        Arg& a = args[i];
        cvAccelView& v = views[i];
        memset(&v, 0, sizeof(v));
        switch (a.plan)
        {
        case PLAN_ABSENT:
            break;
        case PLAN_IN_PLACE:
            if (a.isUMat)
            {
                // device memory: a range of its import, described without handles
                const BufferState* bv = a.um.u->accelState;
                memset(&a.viewAlloc, 0, sizeof(a.viewAlloc));
                a.viewAlloc.size = bv ? bv->imp->alloc.size : a.um.u->size;
                a.viewAlloc.desc = a.desc;
                fillView(v, &a.viewAlloc, (bv ? bv->start : 0) + a.um.offset, a.um.type(), a.um.dims,
                         a.um.size.p, a.um.step.p);
                setImage(v, bv);
            }
            else if (!a.mat.empty())
            {
                fillHostView(v, a.viewAlloc, a.mat);
                a.viewAlloc.handle = 0;
            }
            else
            {
                fillPlannedView(v, a.viewAlloc, a.shape, a.type, hostMemoryDesc());  // a Mat output made after QUERY
            }
            break;
        default:
            fillPlannedView(v, a.viewAlloc, a.shape, a.type,
                            plannedDesc(b, device, plannedKind, planBytes(a.shape, a.type), plannedFlags));
            break;
        }
    }
    memset(&call, 0, sizeof(call));
    call.struct_size = (uint32_t)sizeof(call);
    call.mode = CV_ACCEL_QUERY;
    call.device = device;
    call.op_name = name;
    return table ? fn(table, &call, views.data(), userdata) : CV_ACCEL_NOT_IMPLEMENTED;
}

RunResult OpCall::Impl::run(Invoker fn, void* userdata)
{
    if (unsupported)
        return NOT_HANDLED;

    // 1. candidate devices: the scope's, or every device that holds an argument (in argument order)
    const ScopeEntry* scope = currentScope();
    StreamPtr stream;
    bool anyOnDevice = false, shadowOnDevice = false, mapped = false;
    std::vector<std::pair<int, int> > candidates;
    for (size_t i = 0; i < args.size(); i++)
    {
        const Arg& a = args[i];
        mapped = mapped || a.mapped;
        if (a.isOutput && !a.exists)
        {
            // an output that its own allocator puts on a device: that device is where the result goes
            if (const DeviceAllocator* da = deviceAllocatorOf(a.explicitAlloc))
            {
                anyOnDevice = true;
                const std::pair<int, int> c(da->backend()->id(), da->device());
                if (!scope && std::find(candidates.begin(), candidates.end(), c) == candidates.end())
                    candidates.push_back(c);
            }
        }
        if (!a.onDevice)
            continue;
        anyOnDevice = true;
        if (!a.mapped && (a.desc.host_access == CV_ACCEL_HOST_ACCESS_COPY || a.desc.host_access == CV_ACCEL_HOST_ACCESS_UNCACHED))
            shadowOnDevice = true;
        const std::pair<int, int> c((int)a.desc.backend_id, (int)a.desc.device_index);
        if (!scope && std::find(candidates.begin(), candidates.end(), c) == candidates.end())
            candidates.push_back(c);
    }
    if (scope)
    {
        if (scope->backendId == 0)
            return anyOnDevice ? CPU_ONLY : NOT_HANDLED;   // a host scope runs everything on the CPU
        candidates.assign(1, std::make_pair(scope->backendId, scope->device));
        stream = scope->stream;
    }
    if (candidates.empty())
        return NOT_HANDLED;

    // 2. plan and query every candidate (both free of side effects); the device that accepts and
    //    moves the fewest bytes wins, the earlier argument's device on a tie. A Mat view of
    //    accelerator memory among the arguments leaves no candidate: the CPU has that memory
    //    open, and its bytes may be those of a UMat argument under another name.
    std::vector<cvAccelView> views(args.size());
    cvAccelCall call;
    const void* table = NULL;
    int best = -1;
    uint64_t bestCost = 0;
    bool protectedRefused = false;
    for (size_t c = 0; c < candidates.size() && !mapped; c++)
    {
        Backend* cb = Registry::getInstance().get(candidates[c].first);
        if (!cb || cb->isLost(candidates[c].second))
            continue;
        uint64_t cost = 0;
        bool refused = false;
        const cvAccelStatus st = evaluate(cb, candidates[c].second, fn, userdata, views, call, table, cost, refused);
        if (refused)
            protectedRefused = true;
        else if (st == CV_ACCEL_OK && (best < 0 || cost < bestCost))
        {
            best = (int)c;
            bestCost = cost;
        }
        else if (st != CV_ACCEL_OK && st != CV_ACCEL_NOT_IMPLEMENTED)
            cb->check(st, name, candidates[c].second);
    }

    if (best < 0)
    {
        if (protectedRefused)
            CV_Error_(Error::StsError, ("accel: %s: the result of an operation on protected content can only go to "
                                        "protected memory (pass an empty output to get some)", name));
        // nothing on a device and no scope: the call continues as before (OpenCL, CPU). A device
        // scope decides even when its device declines: the CPU runs the call, counted as a
        // fallback, with the outputs on the scope's device and OpenCL kept away from them.
        if (!anyOnDevice && !scope)
            return NOT_HANDLED;
        counters().cpuFallbacks++;
        Backend* fb = Registry::getInstance().get(candidates[0].first);
        if (!fb || fb->isLost(candidates[0].second))
            return CPU_ONLY;
        const int fdevice = candidates[0].second;
        uint64_t cost = 0;
        bool refused = false;
        evaluate(fb, fdevice, NULL, NULL, views, call, table, cost, refused);  // plans only: where outputs belong
        if (protectedIn)
            CV_Error_(Error::StsNotImplemented, ("accel(%s): %s is not implemented for these arguments, and the "
                                                 "CPU cannot read protected content", fb->name().c_str(), name));
        CV_LOG_DEBUG(NULL, "accel(" << fb->name() << "): " << name << " runs on the CPU ("
                     << (mapped ? "a Mat view of accelerator memory is an argument" : "not implemented for these arguments") << ")");
        if (shadowOnDevice)
            onImplicitTransfer("runs on the CPU, so device data is copied to the host", name);
        // results go where they would have gone if the backend had run the operation
        for (size_t i = 0; i < args.size(); i++)
            if (args[i].isOutput && args[i].plan == PLAN_CREATE)
                createOutput(args[i], fb, fdevice);
        return CPU_ONLY;
    }

    Backend* b = Registry::getInstance().get(candidates[best].first);
    const int device = candidates[best].second;
    if (candidates.size() > 1)
    {
        uint64_t cost = 0;
        bool refused = false;
        const cvAccelStatus status = evaluate(b, device, fn, userdata, views, call, table, cost, refused);   // the winner's plan
        CV_Assert(status == CV_ACCEL_OK && !refused);
    }
    const uint32_t plannedKind = b->deviceInfo(device).preferred_kind;   // what outputs and temporaries get
    cvAccelStatus status = CV_ACCEL_OK;

    // 4. prepare: outputs, temporaries and uploads (these may touch the CPU side, so they
    //    happen before the submission window)
    if (!stream)
        stream = b->defaultStream(device);
    const MatAllocator* deviceAllocator = b->allocator(device, (int)plannedKind, protectedIn);
    bool needSync = syncDebugMode();
    bool borrowed = false;      // the call uses memory that goes back to its owner: no time limit on the wait
    for (size_t i = 0; i < args.size(); i++)
    {
        Arg& a = args[i];
        if (a.plan == PLAN_CREATE)
            createOutput(a, b, device);
        else if (a.plan == PLAN_UPLOAD || a.plan == PLAN_COPY_BACK)
        {
            if (!scope && (a.plan == PLAN_UPLOAD || a.isUMat))
                onImplicitTransfer(a.plan == PLAN_UPLOAD ? "an input is copied to the device"
                                                         : "the result is copied out of the device", name);
            a.temp.allocator = const_cast<MatAllocator*>(deviceAllocator);
            a.temp.create(a.shape, a.type);
            if (a.plan == PLAN_UPLOAD)
            {
                if (a.isUMat)
                    a.um.copyTo(a.temp);
                else
                    a.mat.copyTo(a.temp);
            }
            else if (!a.fullOverwrite)
            {
                // the whole temporary is copied back: what the operation leaves alone must
                // already hold the output's current content
                if (a.isUMat)
                {
                    if (a.exists)
                        a.um.copyTo(a.temp);    // one that does not exist yet has no content to keep
                }
                else
                {
                    Mat current = a.arr.getMat();
                    if (!current.empty() && current.type() == a.type && current.shape() == a.shape)
                        current.copyTo(a.temp);
                }
            }
        }
        else if (a.plan == PLAN_IN_PLACE && !a.onDevice)
        {
            // plain host memory used in place by a device that can address it
            if (a.isOutput && !a.isUMat)
            {
                a.arr.create(a.shape, a.type);     // the caller's Mat (nothing to do when it fits)
                a.mat = a.arr.getMat();
            }
            else if (a.isUMat)
                a.mat = a.um.getMat(a.isOutput ? ACCESS_WRITE : ACCESS_READ);  // host allocator: no copy
            needSync = borrowed = true;     // finish before the caller can touch that memory again
        }
    }

    // 5. submit: acquire, wait, execute, record
    FenceRef done;
    {
        // no new device work while the CPU holds memory it may only touch on an idle device
        SubmitWindow window(b->idleGate(device));
        if (window.blocked())
        {
            counters().cpuFallbacks++;
            CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): " << name << " runs on the CPU (the CPU holds device-idle memory)");
            return CPU_ONLY;
        }

        // every buffer the work touches, acquired together (the same buffer twice counts once)
        DeviceAcquisition acq;
        for (size_t i = 0; i < args.size(); i++)
        {
            Arg& a = args[i];
            const UMat& target = (a.plan == PLAN_UPLOAD || a.plan == PLAN_COPY_BACK) ? a.temp : a.um;
            if (a.plan == PLAN_ABSENT || !target.u || !target.u->accelState)
                continue;
            a.mode = a.isOutput ? (DEV_WRITE | (a.fullOverwrite && coversAllocation(target) ? DEV_DISCARD : 0)) : DEV_READ;
            acq.add(target.u, a.mode);
            // User memory: the owner may touch it once we return. Memory shared through implicit
            // fences (a dma-buf) is not: its other users wait for the fence of this work.
            const PhysicalBacking* bk = target.u->accelState->backing;
            if (bk->borrowed && !(bk->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC))
                needSync = borrowed = true;
        }
        std::vector<FenceRef> waits;
        if (!acq.acquire(stream, waits))
        {
            // a Mat view of one of the buffers is alive: the CPU path is the only safe one
            counters().cpuFallbacks++;
            CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): " << name << " runs on the CPU (a Mat view of an argument is alive)");
            return CPU_ONLY;
        }

        bool queued = false;    // the backend may have queued work
        try
        {
            for (size_t i = 0; i < waits.size(); i++)
                streamWait(stream, waits[i]);

            for (size_t i = 0; i < args.size(); i++)
            {
                Arg& a = args[i];
                cvAccelView& v = views[i];
                const UMat& target = (a.plan == PLAN_UPLOAD || a.plan == PLAN_COPY_BACK) ? a.temp : a.um;
                if (a.plan == PLAN_ABSENT)
                    continue;
                if (target.u && target.u->accelState)
                {
                    // a range of the import of its memory: never past the bytes the UMatData views
                    const BufferState* bv = target.u->accelState;
                    a.viewAlloc = bv->imp->alloc;
                    fillView(v, &a.viewAlloc, bv->start + target.offset, target.type(), target.dims,
                             target.size.p, target.step.p);
                    setImage(v, bv);
                    if (a.viewAlloc.desc.layout_modifier != 0)
                    {
                        // a tiled or compressed layout: steps say nothing about where bytes lie, so the
                        // argument is its whole view, as large as its producer said
                        if (target.offset != 0 || (target.u->size != bv->size) ||
                            (bv->image && (target.rows != bv->image->desc.planes[bv->plane].rows ||
                                           target.cols != bv->image->desc.planes[bv->plane].cols)))
                            CV_Error_(Error::StsNotImplemented, ("accel: %s: argument %d is a part of memory with a "
                                                                 "non-linear layout", name, (int)i));
                        if (!rangeInside(0, a.viewAlloc.size, bv->start, bv->size))
                            CV_Error_(Error::StsInternal, ("accel: %s: argument %d does not fit its allocation", name, (int)i));
                        continue;
                    }
                    if (!viewFitsRange(v, bv->start, bv->size))
                        CV_Error_(Error::StsInternal, ("accel: %s: argument %d does not fit its buffer", name, (int)i));
                }
                else
                {
                    CV_Assert(a.isHost && !a.mat.empty());
                    fillHostView(v, a.viewAlloc, a.mat);
                }
                if (!viewFitsAllocation(v))
                    CV_Error_(Error::StsInternal, ("accel: %s: argument %d does not fit its allocation", name, (int)i));
            }

            call.mode = CV_ACCEL_EXECUTE;
            call.stream = stream->handle;
            {
                StreamCall one(stream);
                status = fn(table, &call, views.data(), userdata);
            }
            if (status == CV_ACCEL_OUT_OF_MEMORY)
            {
                // nothing was queued (backend rule 9): free what finished work left behind, retry once
                collectRetired(true);
                StreamCall one(stream);
                status = fn(table, &call, views.data(), userdata);
            }
            if (status == CV_ACCEL_NOT_IMPLEMENTED)
            {
                // allowed: nothing was queued
                acq.abandon();
                counters().cpuFallbacks++;
                CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): " << name << " runs on the CPU (declined at EXECUTE)");
                return CPU_ONLY;
            }
            queued = status != CV_ACCEL_OUT_OF_MEMORY;
            b->check(status, name, device);
            done = recordFence(stream);
        }
        catch (...)
        {
            if (!queued)
            {
                acq.abandon();
                throw;
            }
            // part of the work may be queued: the buffers wait for it, and what it writes is
            // invalid until it is written whole
            char why[160];
            acq.fail(fenceAfterFailure(stream), currentExceptionMessage(why, sizeof(why)));
            throw;
        }
        acq.record(done);
    }
    counters().deviceOps++;

    // 6. results that the device could not write in place
    for (size_t i = 0; i < args.size(); i++)
    {
        Arg& a = args[i];
        if (a.isOutput && a.plan == PLAN_COPY_BACK)
            a.temp.copyTo(a.arr);
    }
    if (needSync)
        done.wait(!borrowed);
    return HANDLED;
}

}}} // namespace cv::accel::detail
