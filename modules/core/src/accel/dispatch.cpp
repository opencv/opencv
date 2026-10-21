// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: running a cv:: call on a device (OpCall).
//
// Where a call runs is decided by fixed rules, never by how many bytes would move:
// the device of this thread's ExecutionScope, otherwise the device that holds the first UMat
// input. Inputs elsewhere are copied to that device; outputs are created on it, or, when they
// cannot move (a Mat, a region of a larger UMat, memory with its own allocator), get the result
// copied in. Memory shared with the CPU may be computed on by other devices: the preferred
// devices are offered the call first, in their order, through imports of the same bytes.

#include "../precomp.hpp"
#include "../umatrix.hpp"
#include "accel_impl.hpp"

#include <algorithm>

namespace cv { namespace accel { namespace detail {

namespace {

enum Plan
{
    PLAN_USE = 0,       // the argument is in the device's memory already
    PLAN_UPLOAD = 1,    // an input copied into a temporary on the device
    PLAN_CREATE = 2,    // an output UMat released and created again on the device
    PLAN_TEMP = 3       // an output computed in a temporary and copied to where it lives
};

inline const UMat& umatOf(const _InputArray& arr)
{
    return *(const UMat*)arr.getObj();
}

inline bool onDevice(const UMat& m, Device::Impl* dev)
{
    BufferState* st = stateOf(m.u);
    return st != 0 && st->dev == dev;
}

//! memory the CPU reaches only through a copy: device memory not shared with the CPU, OpenCL buffers, ...
bool copyOnlyMemory(const UMat& m)
{
    if (!m.u)
        return false;
    BufferState* st = stateOf(m.u);
    if (st)
        return !st->dev->shared;
    return m.u->currAllocator != Mat::getDefaultAllocator();
}

//! an output UMat that may be released and created again on another device: the whole buffer,
//! held by this header only, with no Mat view, no allocator of its own and no fixed size or type
bool movable(const UMat& m, const _OutputArray& arr)
{
    if (arr.fixedSize() || arr.fixedType() || m.allocator != 0)
        return false;
    const UMatData* u = m.u;
    if (!u)
        return true;
    return u->urefcount == 1 && u->refcount == 0 && m.offset == 0 && u->originalUMatData == 0 &&
           !(u->flags & UMatData::USER_ALLOCATED) && m.isContinuous() &&
           u->size == m.total() * m.elemSize();
}

//! creates @p m on @p dev without changing m.allocator; false when the memory landed elsewhere
bool createOn(UMat& m, Device::Impl* dev, const MatShape& shape, int type)
{
    MatAllocator* prev = m.allocator;
    m.allocator = dev->allocator;
    try
    {
        m.create(shape, type);
    }
    catch (...)
    {
        m.allocator = prev;
        throw;
    }
    m.allocator = prev;
    return onDevice(m, dev);    // UMat::create falls back to the default allocator when the device is full
}

//! headers like @p umats over the imports of their bytes on @p dev; false when one cannot be shared
bool aliasArgs(const std::vector<UMat>& umats, Device::Impl* dev, std::vector<UMat>& out)
{
    out.assign(umats.size(), UMat());
    for (size_t i = 0; i < umats.size(); i++)
    {
        UMatData* a = aliasOn(umats[i].u, dev);
        if (!a)
            return false;
        UMat h(umats[i]);                   // same size, steps and offset ...
        CV_XADD(&h.u->urefcount, -1);       // ... (umats[i] keeps its buffer alive)
        h.u = a;                            // over the alias
        CV_XADD(&a->urefcount, 1);
        out[i] = h;
    }
    return true;
}

} // namespace

bool OpCall::runImpl(const char* module, const std::function<bool(Ops*, std::vector<UMat>&, Stream&)>& fn)
{
    // 1. the device
    bool scoped = false;
    Device::Impl* dev = scopeDevice(scoped);
    if (!scoped)
    {
        for (size_t i = 0; i < args_.size() && !dev; i++)
        {
            const Arg& a = args_[i];
            if (!a.isOutput && a.in->isUMat())
            {
                BufferState* st = stateOf(umatOf(*a.in).u);
                if (st)
                    dev = st->dev;
            }
        }
    }
    if (!dev)
        return false;

    // 2. the plan, without side effects
    const size_t n = args_.size();
    std::vector<int> plan(n, PLAN_USE);
    bool mapped = false;            // a device argument has a Mat view: the CPU has to run the call
    bool copies = false;            // running on the device copies between host and device memory
    bool fallbackCopies = !dev->shared;   // the CPU would copy: outputs are created on the device
    for (size_t i = 0; i < n; i++)
    {
        const Arg& a = args_[i];
        const int kind = a.in->kind();
        if (kind != _InputArray::MAT && kind != _InputArray::UMAT)
            return false;
        if (kind == _InputArray::UMAT && copyOnlyMemory(umatOf(*a.in)))
            fallbackCopies = true;
        if (!a.isOutput)
        {
            if (a.in->empty())
                return false;
            if (kind == _InputArray::UMAT)
            {
                const UMat& m = umatOf(*a.in);
                if (onDevice(m, dev))
                {
                    mapped = mapped || m.u->refcount > 0;
                    continue;
                }
                copies = copies || !dev->shared || copyOnlyMemory(m);
            }
            else
                copies = copies || !dev->shared;
            plan[i] = PLAN_UPLOAD;
            continue;
        }
        if (kind == _InputArray::UMAT)
        {
            UMat& m = a.out->getUMatRef();
            bool aliases = false;
            for (size_t j = 0; j < n; j++)
                if (!args_[j].isOutput && args_[j].in->isUMat() && m.u && umatOf(*args_[j].in).u == m.u)
                    aliases = true;
            if (!aliases && onDevice(m, dev) && m.type() == a.type && m.shape() == a.shape)
            {
                mapped = mapped || m.u->refcount > 0;
                continue;
            }
            if (!aliases && movable(m, *a.out))
            {
                plan[i] = PLAN_CREATE;
                continue;
            }
            plan[i] = PLAN_TEMP;
            if (!onDevice(m, dev))
                copies = copies || !dev->shared || copyOnlyMemory(m);
        }
        else
        {
            plan[i] = PLAN_TEMP;
            copies = copies || !dev->shared;
        }
    }

    // outputs follow the device even when the CPU runs the call, so a pipeline stays there
    auto fallback = [&]() -> bool {
        if (fallbackCopies && getTransferPolicy() == TRANSFER_FORBID)
            CV_Error_(Error::StsError, ("accel: %s does not run on %s, and running it on the CPU would copy "
                                        "device data (TRANSFER_FORBID)", name_, dev->name.c_str()));
        for (size_t i = 0; i < n; i++)
        {
            if (plan[i] == PLAN_CREATE)
            {
                UMat& m = args_[i].out->getUMatRef();
                if (!onDevice(m, dev))
                {
                    m.release();
                    createOn(m, dev, args_[i].shape, args_[i].type);
                }
            }
        }
        countFallback();
        return false;
    };

    if (mapped)
        return fallback();
    if (copies && getTransferPolicy() == TRANSFER_FORBID)
        CV_Error_(Error::StsError, ("accel: %s on %s would copy data between host and device memory "
                                    "(TRANSFER_FORBID)", name_, dev->name.c_str()));

    // the devices that may run it: for memory shared with the CPU, the preferred devices first
    // (they reach it through an alias, the data stays where it is), then the device holding it
    std::vector<Device::Impl*> candidates;
    if (!scoped && dev->shared)
        candidates = preferredDevices();
    if (std::find(candidates.begin(), candidates.end(), dev) == candidates.end())
        candidates.push_back(dev);
    bool anyOps = false;
    for (size_t c = 0; c < candidates.size() && !anyOps; c++)
        anyOps = candidates[c]->backend->ops(module) != 0;
    if (!anyOps)
        return fallback();

    // 3. every argument in the memory of that device
    std::vector<UMat> umats(n);
    for (size_t i = 0; i < n; i++)
    {
        const Arg& a = args_[i];
        switch (plan[i])
        {
        case PLAN_USE:
            umats[i] = a.isOutput ? a.out->getUMatRef() : umatOf(*a.in);
            break;
        case PLAN_UPLOAD:
            if (!createOn(umats[i], dev, a.shape, a.type))
                return fallback();
            a.in->copyTo(umats[i]);
            break;
        case PLAN_CREATE:
        {
            UMat& m = a.out->getUMatRef();
            m.release();
            if (!createOn(m, dev, a.shape, a.type))
                return fallback();
            umats[i] = m;
            break;
        }
        case PLAN_TEMP:
            if (!createOn(umats[i], dev, a.shape, a.type))
                return fallback();
            break;
        }
    }

    // 4. offer the call to each candidate; one that declines has queued nothing
    for (size_t c = 0; c < candidates.size(); c++)
    {
        Device::Impl* cand = candidates[c];
        Ops* ops = cand->backend->ops(module);
        if (!ops)
            continue;
        std::vector<UMat> args;
        if (cand == dev)
            args = umats;
        else if (!aliasArgs(umats, cand, args))
            continue;

        const Ptr<Stream>& s = currentStream(cand);
        for (size_t i = 0; i < n; i++)
        {
            UMatData* od = owningData(args[i].u);
            UMatDataAutoLock lock(od);
            orderOnStream(stateOf(od), s, args_[i].isOutput);
        }
        bool done = false;
        try
        {
            done = fn(ops, args, *s);
        }
        catch (...)
        {
            // work may have been queued before the error: later uses of the outputs wait for it
            try
            {
                Ptr<Event> e = s->record();
                for (size_t i = 0; i < n; i++)
                {
                    if (args_[i].isOutput)
                    {
                        UMatData* od = owningData(args[i].u);
                        UMatDataAutoLock lock(od);
                        recordUse(stateOf(od), s, e, true);
                    }
                }
            }
            catch (...) {}
            throw;
        }
        if (!done)
            continue;

        Ptr<Event> e = s->record();
        for (size_t i = 0; i < n; i++)
        {
            UMatData* od = owningData(args[i].u);
            UMatDataAutoLock lock(od);
            BufferState* st = stateOf(od);
            recordUse(st, s, e, args_[i].isOutput);
            if (args_[i].isOutput)
            {
                st->initialized = true;
                if (!st->dev->shared)
                {
                    od->markHostCopyObsolete(true);
                    od->markDeviceCopyObsolete(false);
                }
            }
        }
        countDeviceOp();

        // 5. results whose destination is elsewhere
        for (size_t i = 0; i < n; i++)
            if (plan[i] == PLAN_TEMP)
                umats[i].copyTo(*args_[i].out);
        return true;
    }
    return fallback();
}

OpCall::OpCall(const char* name) : name_(name ? name : "a call") {}

OpCall::~OpCall() {}

OpCall& OpCall::input(InputArray arr)
{
    Arg a;
    a.isOutput = false;
    a.in = &arr;
    a.out = 0;
    a.type = arr.empty() ? -1 : arr.type();
    if (!arr.empty())
        a.shape = arr.shape();
    args_.push_back(a);
    return *this;
}

OpCall& OpCall::output(OutputArray arr, const MatShape& shape, int type)
{
    Arg a;
    a.isOutput = true;
    a.in = &arr;
    a.out = &arr;
    a.shape = shape;
    a.type = type;
    args_.push_back(a);
    return *this;
}

}}} // namespace cv::accel::detail
