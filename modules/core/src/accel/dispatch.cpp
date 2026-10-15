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
    PLAN_ABSENT = 0,        //!< optional argument that was not given
    PLAN_IN_PLACE = 1,      //!< the device works on the argument's own memory
    PLAN_UPLOAD = 2,        //!< input copied into a temporary on the device
    PLAN_CREATE = 3,        //!< output (re)allocated on the device
    PLAN_COPY_BACK = 4      //!< output produced in a temporary, then copied to the caller's array
};

struct Arg
{
    Arg() : isOutput(false), arr(0), type(-1), fullOverwrite(true), isUMat(false), isHost(false),
            onDevice(false), plan(PLAN_ABSENT), mode(0), acquired(0)
    {
        memset(&desc, 0, sizeof(desc));
        memset(&viewAlloc, 0, sizeof(viewAlloc));
    }
    bool isOutput;
    const _OutputArray* arr;    //!< outputs (inputs are kept as UMat / Mat below)
    MatShape shape;
    int type;
    bool fullOverwrite;
    bool isUMat;
    bool isHost;                //!< plain host memory (Mat, or UMat from the host allocator)
    UMat um;
    Mat mat;
    cvAccelMemoryDesc desc;
    bool onDevice;              //!< memory from an accelerator backend
    int plan;
    UMat temp;
    int mode;
    UMatData* acquired;
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

bool overlaps(const UMat& a, const UMat& b)
{
    if (!a.u || a.u != b.u)
        return false;
    const size_t ea = extentBytes(a.dims, a.size.p, a.step.p, a.type());
    const size_t eb = extentBytes(b.dims, b.size.p, b.step.p, b.type());
    return a.offset < b.offset + eb && b.offset < a.offset + ea;
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

void fillHostView(cvAccelView& v, cvAccelAllocation& alloc, const Mat& m)
{
    memset(&alloc, 0, sizeof(alloc));
    alloc.handle = (uint64_t)(uintptr_t)m.datastart;
    alloc.size = (uint64_t)(m.dataend - m.datastart);
    alloc.desc = hostMemoryDesc();
    fillView(v, &alloc, (size_t)(m.data - m.datastart), m.type(), m.dims, m.size.p, m.step.p);
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
        : name(name_), module(module_), version(version_), unsupported(false) {}
    const char* name;
    const char* module;
    uint32_t version;
    bool unsupported;
    std::vector<Arg> args;

    RunResult run(Invoker fn, void* userdata);
private:
    bool planArgs(Backend* b, int device);
    void createOutput(Arg& a, Backend* b, int device);
    void releaseAcquired();
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
        a.isHost = true;
        a.desc = hostMemoryDesc();
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
    a.arr = &arr;
    a.shape = shape;
    a.type = CV_MAT_TYPE(type);
    a.fullOverwrite = fullOverwrite;
    const _InputArray::KindFlag k = arr.kind();
    if (k == _InputArray::UMAT)
    {
        a.isUMat = true;
        a.um = arr.getUMatRef();
        describeUMat(a);
    }
    else if (k == _InputArray::MAT)
        a.isHost = true;
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
        const bool exists = a.isUMat && !a.um.empty() && sameShape(a.um, a.shape) && a.um.type() == a.type;
        if (a.isUMat && !exists)
            a.plan = PLAN_CREATE;
        else if (exists && a.onDevice && b->canAccess(device, a.desc))
            a.plan = PLAN_IN_PLACE;
        else
            a.plan = PLAN_COPY_BACK;    // a Mat, or a UMat on memory the device cannot write
        // an output that overlaps an input is produced in a temporary
        if (a.plan == PLAN_IN_PLACE)
        {
            for (size_t j = 0; j < args.size(); j++)
                if (!args[j].isOutput && args[j].plan == PLAN_IN_PLACE && args[j].isUMat && overlaps(a.um, args[j].um))
                    a.plan = PLAN_COPY_BACK;
        }
    }
    return true;
}

void OpCall::Impl::createOutput(Arg& a, Backend* b, int device)
{
    UMat& d = a.arr->getUMatRef();
    MatAllocator* saved = d.allocator;
    d.allocator = b->allocator(device, (int)b->deviceInfo(device).preferred_kind);
    try
    {
        a.arr->create(a.shape, a.type);
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

void OpCall::Impl::releaseAcquired()
{
    for (size_t i = 0; i < args.size(); i++)
    {
        if (args[i].acquired)
        {
            abandonDeviceUse(args[i].acquired);
            args[i].acquired = 0;
        }
    }
}

RunResult OpCall::Impl::run(Invoker fn, void* userdata)
{
    if (unsupported)
        return NOT_HANDLED;

    // 1. device
    const ScopeEntry* scope = currentScope();
    int backendId = 0, device = 0;
    StreamPtr stream;
    bool anyOnDevice = false, shadowOnDevice = false;
    for (size_t i = 0; i < args.size(); i++)
    {
        const Arg& a = args[i];
        if (!a.onDevice)
            continue;
        if (!anyOnDevice && !scope)
        {
            backendId = (int)a.desc.backend_id;
            device = (int)a.desc.device_index;
        }
        anyOnDevice = true;
        if (a.desc.host_access == CV_ACCEL_HOST_ACCESS_COPY || a.desc.host_access == CV_ACCEL_HOST_ACCESS_UNCACHED)
            shadowOnDevice = true;
    }
    if (scope)
    {
        backendId = scope->backendId;
        device = scope->device;
        stream = scope->stream;
    }
    if (backendId == 0)
        return anyOnDevice ? CPU_ONLY : NOT_HANDLED;   // a host scope runs everything on the CPU
    Backend* b = Registry::getInstance().get(backendId);
    if (!b || b->isLost())
        return anyOnDevice ? CPU_ONLY : NOT_HANDLED;

    // 2. plan
    planArgs(b, device);
    cvAccelMemoryDesc deviceDesc;
    memset(&deviceDesc, 0, sizeof(deviceDesc));
    deviceDesc.backend_id = (uint32_t)b->id();
    deviceDesc.device_index = (uint32_t)device;
    deviceDesc.kind = b->deviceInfo(device).preferred_kind;

    // 3. query
    const void* table = b->opTable(module, version);
    std::vector<cvAccelView> views(args.size());
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
                memset(&a.viewAlloc, 0, sizeof(a.viewAlloc));
                a.viewAlloc.size = a.um.u->size;
                a.viewAlloc.desc = a.desc;
                fillView(v, &a.viewAlloc, a.um.offset, a.um.type(), a.um.dims, a.um.size.p, a.um.step.p);
            }
            else
            {
                fillHostView(v, a.viewAlloc, a.mat);
                a.viewAlloc.handle = 0;
            }
            break;
        default:
            fillPlannedView(v, a.viewAlloc, a.shape, a.type, deviceDesc);
            break;
        }
    }
    cvAccelCall call;
    memset(&call, 0, sizeof(call));
    call.struct_size = (uint32_t)sizeof(call);
    call.mode = CV_ACCEL_QUERY;
    call.device = device;
    call.op_name = name;
    cvAccelStatus status = table ? fn(table, &call, views.data(), userdata) : CV_ACCEL_NOT_IMPLEMENTED;
    if (status == CV_ACCEL_NOT_IMPLEMENTED)
    {
        if (!anyOnDevice)
            return NOT_HANDLED;
        counters().cpuFallbacks++;
        CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): " << name << " runs on the CPU (not implemented for these arguments)");
        if (shadowOnDevice)
            onImplicitTransfer("runs on the CPU, so device data is copied to the host", name);
        // results go where they would have gone if the backend had run the operation
        for (size_t i = 0; i < args.size(); i++)
            if (args[i].isOutput && args[i].plan == PLAN_CREATE)
                createOutput(args[i], b, device);
        return CPU_ONLY;
    }
    b->check(status, name);

    // 4. prepare: outputs, temporaries and uploads (these may touch the CPU side, so they
    //    happen before the submission window)
    if (!stream)
        stream = b->defaultStream(device);
    const MatAllocator* deviceAllocator = b->allocator(device, (int)deviceDesc.kind);
    bool needSync = syncDebugMode();
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
        }
        else if (a.plan == PLAN_IN_PLACE && !a.onDevice)
        {
            // plain host memory used in place by a device that can address it
            if (a.isUMat)
                a.mat = a.um.getMat(ACCESS_READ);   // host allocator: no copy
            needSync = true;    // finish before the caller can touch that memory again
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

        std::vector<FenceRef> waits;
        for (size_t i = 0; i < args.size(); i++)
        {
            Arg& a = args[i];
            const UMat& target = (a.plan == PLAN_UPLOAD || a.plan == PLAN_COPY_BACK) ? a.temp : a.um;
            if (a.plan == PLAN_ABSENT || !target.u || !target.u->accelState)
                continue;
            a.mode = a.isOutput ? (DEV_WRITE | (a.fullOverwrite && coversAllocation(target) ? DEV_DISCARD : 0)) : DEV_READ;
            if (!acquireForDevice(target.u, stream, a.mode, waits))
            {
                // a Mat view of this buffer is alive: the CPU path is the only safe one
                releaseAcquired();
                counters().cpuFallbacks++;
                CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): " << name << " runs on the CPU (a Mat view of an argument is alive)");
                return CPU_ONLY;
            }
            a.acquired = target.u;
            if (target.u->accelState->imported)
                needSync = true;    // user memory: the owner may touch it once we return
        }

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
                    a.viewAlloc = target.u->accelState->alloc;
                    fillView(v, &a.viewAlloc, target.offset, target.type(), target.dims, target.size.p, target.step.p);
                }
                else
                {
                    CV_Assert(!a.isOutput && a.isHost);
                    fillHostView(v, a.viewAlloc, a.mat);
                }
                if (!viewFitsAllocation(v))
                    CV_Error_(Error::StsInternal, ("accel: %s: argument %d does not fit its allocation", name, (int)i));
            }

            call.mode = CV_ACCEL_EXECUTE;
            call.stream = stream->handle;
            status = fn(table, &call, views.data(), userdata);
            if (status == CV_ACCEL_OUT_OF_MEMORY)
            {
                b->collectDeferred(true);
                status = fn(table, &call, views.data(), userdata);
            }
            if (status == CV_ACCEL_NOT_IMPLEMENTED)
            {
                // allowed: nothing was enqueued
                releaseAcquired();
                counters().cpuFallbacks++;
                return CPU_ONLY;
            }
            b->check(status, name);
            done = recordFence(stream);
        }
        catch (...)
        {
            releaseAcquired();
            throw;
        }

        for (size_t i = 0; i < args.size(); i++)
        {
            Arg& a = args[i];
            if (a.acquired)
            {
                recordDeviceUse(a.acquired, done, a.mode);
                a.acquired = 0;
            }
        }
    }
    counters().deviceOps++;

    // 6. results that the device could not write in place
    for (size_t i = 0; i < args.size(); i++)
    {
        Arg& a = args[i];
        if (a.isOutput && a.plan == PLAN_COPY_BACK)
            a.temp.copyTo(*a.arr);
    }
    if (needSync)
        done.wait();
    return HANDLED;
}

}}} // namespace cv::accel::detail
