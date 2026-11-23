// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Imgproc operations of the HIP backend. Each runs the first kernel of its rules (opencv_hip_tuning/
// next to the library these operations are built into) that can run on the device; nothing is
// timed here.

#include "accel/hip/hip.hpp"

#ifdef HAVE_HIP

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "kernels.hpp"
#include "rules.hpp"

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cv { namespace accel { namespace hip_impl {

namespace {

// OPENCV_LOG_LEVEL=INFO logs where the rules came from and each distinct choice, once
bool tracing()
{
    return utils::logging::getLogLevel() >= utils::logging::LOG_LEVEL_INFO;
}

void trace(const std::string& message)
{
    static std::mutex mtx;
    static std::set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!seen.insert(message).second)
            return;
    }
    CV_LOG_INFO(NULL, "accel(hip): " << message);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// rules

//! the directory of the library these operations are built into (opencv_core), with a trailing slash, or empty
std::string libraryDir()
{
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&libraryDir), &info) || !info.dli_fname)
        return std::string();
    const std::string p = info.dli_fname;
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash + 1);
}

rules::RuleFile loadRules(const std::string& path)
{
    std::vector<std::string> warnings;
    rules::RuleFile r = rules::read(path, warnings);
    for (size_t i = 0; i < warnings.size(); i++)
        CV_LOG_WARNING(NULL, "accel(hip): " << warnings[i]);
    if (tracing())
        trace("kernel rules from " + path);
    return r;
}

// the shipped imgproc.yaml in opencv_hip_tuning/ next to the library;
// OPENCV_ACCEL_HIP_RULES=<file> reads that file instead
const rules::RuleFile& ruleFile()
{
    static const rules::RuleFile file = []() {
        const std::string path = utils::getConfigurationParameterString("OPENCV_ACCEL_HIP_RULES");
        return loadRules(!path.empty() ? path : libraryDir() + "opencv_hip_tuning/imgproc.yaml");
    }();
    return file;
}

struct DeviceFacts
{
    std::string name;       //!< hipDeviceProp_t::name
    std::string arch;       //!< "gfx1100", without the feature suffix of gcnArchName
    size_t ldsPerBlock;
    int runtime;            //!< hipRuntimeGetVersion
};

const DeviceFacts& deviceFacts(int device)
{
    static std::mutex mtx;
    static std::map<int, DeviceFacts> facts;
    std::lock_guard<std::mutex> lock(mtx);
    std::map<int, DeviceFacts>::iterator it = facts.find(device);
    if (it == facts.end())
    {
        hipDeviceProp_t p;
        hipCheck(hipGetDeviceProperties(&p, device), "hipGetDeviceProperties");
        DeviceFacts f;
        f.name = p.name;
        f.arch = p.gcnArchName;
        f.arch = f.arch.substr(0, f.arch.find(':'));
        f.ldsPerBlock = p.sharedMemPerBlock;
        hipCheck(hipRuntimeGetVersion(&f.runtime), "hipRuntimeGetVersion");
        it = facts.insert(std::make_pair(device, f)).first;
    }
    return it->second;
}

// the rules of the device: its arch's section of the YAML, else "default"; NULL when there are none
const rules::ArchRules* rulesFor(int device)
{
    static std::mutex mtx;
    static std::map<int, std::pair<bool, rules::ArchRules> > merged;
    const DeviceFacts& dev = deviceFacts(device);
    std::lock_guard<std::mutex> lock(mtx);
    std::map<int, std::pair<bool, rules::ArchRules> >::iterator it = merged.find(device);
    if (it != merged.end())
        return it->second.first ? &it->second.second : 0;

    std::vector<std::string> warnings;
    bool found = false;
    const rules::ArchRules r = rules::effective(ruleFile(), rules::RuleFile(), dev.arch, dev.name, dev.runtime,
                                                std::string(), warnings, found);
    for (size_t i = 0; i < warnings.size(); i++)
        CV_LOG_WARNING(NULL, "accel(hip): " << warnings[i]);
    it = merged.insert(std::make_pair(device, std::make_pair(found, r))).first;
    return found ? &it->second.second : 0;
}

const rules::OpRules* opRules(const rules::ArchRules* arch, const char* op)
{
    if (!arch)
        return 0;
    std::map<std::string, rules::OpRules>::const_iterator it = arch->ops.find(op);
    return it == arch->ops.end() ? 0 : &it->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// choosing a kernel

// why @p kernel cannot run on @p device (the current one) with @p blockThreads, or NULL; hard limits only
const char* kernelIneligible(const void* kernel, int blockThreads, int device)
{
    static std::mutex mtx;
    static std::map<std::tuple<const void*, int, int>, const char*> verdicts;
    const size_t lds = deviceFacts(device).ldsPerBlock;
    std::lock_guard<std::mutex> lock(mtx);
    const std::tuple<const void*, int, int> key(kernel, blockThreads, device);
    std::map<std::tuple<const void*, int, int>, const char*>::iterator it = verdicts.find(key);
    if (it != verdicts.end())
        return it->second;

    const char* why = 0;
    hipFuncAttributes a;
    int blocks = 0;
    if (hipFuncGetAttributes(&a, kernel) != hipSuccess)
        why = "it is not built for this architecture";
    else if (a.maxThreadsPerBlock < blockThreads)
        why = "its block is larger than it was built for";
    else if (a.sharedSizeBytes > lds)
        why = "its LDS does not fit";
    else if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockThreads, 0) != hipSuccess)
        why = "its occupancy cannot be read";
    else if (blocks < 1)
        why = "none of its blocks fits on a compute unit";
    if (why)
        (void)hipGetLastError();
    verdicts[key] = why;
    return why;
}

struct Candidate
{
    std::string name;
    const void* kernel;             //!< NULL when the kernel does not apply to the call
    const char* notApplicable;
};

// the first candidate that applies and is eligible, or -1; @p reasons says why others were skipped
int choose(const std::vector<Candidate>& c, int blockThreads, int device, std::string& reasons)
{
    for (size_t i = 0; i < c.size(); i++)
    {
        const char* why = c[i].kernel ? kernelIneligible(c[i].kernel, blockThreads, device) : c[i].notApplicable;
        if (!why)
            return (int)i;
        reasons += (reasons.empty() ? "" : "; ") + c[i].name + ": " + why;
    }
    return -1;
}

std::string traceTail(const rules::ArchRules* arch, const rules::OpRules* op, const rules::Choice* choice,
                      const std::string& reasons)
{
    std::string s = " [rules: " + (arch ? arch->section : std::string("none"));
    if (op && choice != &op->general)
        s += ", workload rule";
    s += "]";
    return reasons.empty() ? s : s + " (" + reasons + ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////

// mirrors ColumnSum<ushort, uchar> in box_filter.simd.hpp
kernels::BoxScale boxScale(Size ksize, bool normalize)
{
    kernels::BoxScale b = { 1.f, 0u, 0u };
    const int area = ksize.width * ksize.height;
    if (!normalize || area == 1)
        return b;
    b.scale = (float)(1. / area);
    if (area <= 256)
    {
        const double scalef = (double)(1 << 23) / area;
        int divScale = cvFloor(scalef), divDelta = area / 2;
        if (scalef - divScale < 0.5)
            divDelta++;
        else
            divScale++;
        b.divScale = (unsigned)divScale;
        b.divDelta = (unsigned)divDelta;
    }
    return b;
}

class HipImgprocOps CV_FINAL : public ImgprocOps
{
public:
    bool boxFilter(const UMat& src, UMat& dst, Size ksize, Point anchor, bool normalize,
                   int borderType, Stream& stream) CV_OVERRIDE
    {
        const int cn = src.channels();
        if (src.dims != 2 || src.depth() != CV_8U || (cn != 1 && cn != 3 && cn != 4) ||
            dst.type() != src.type() || src.rows >= kernels::IMGPROC_MAX_ROWS)
            return false;
        // invalid anchors are reported by the CPU path
        if (anchor.x < 0 || anchor.x >= ksize.width || anchor.y < 0 || anchor.y >= ksize.height)
            return false;
        // the window reads pixels that other threads write; UMat::handle() needs no Mat view alive
        if (src.u == dst.u || src.u->refcount != 0 || dst.u->refcount != 0)
            return false;
        const kernels::BoxScale scale = boxScale(ksize, normalize);
        const bool fixed = scale.divScale != 0;
        // BORDER_WRAP and anything else the kernels do not implement
        const void* sliding = kernels::boxSlidingKernelFor(cn, borderType, fixed);
        if (!sliding)
            return false;

        const HipStreamRef s = hipStreamOf(stream);
        HipDeviceGuard g(s.device);
        const DeviceFacts& dev = deviceFacts(s.device);
        const rules::ArchRules* arch = rulesFor(s.device);
        const rules::OpRules* op = opRules(arch, "boxFilter");
        const rules::Choice* choice = op ? &op->choose({ cn, ksize.width == ksize.height ? ksize.width : 0 }) : 0;

        const void* lds = ksize.width == ksize.height ? kernels::boxLdsKernelFor(cn, ksize.width, borderType, fixed) : 0;
        std::vector<Candidate> c;
        for (size_t i = 0; choice && i < choice->kernels.size(); i++)
        {
            if (choice->kernels[i] == "lds_tile")
                c.push_back({ choice->kernels[i], lds, "built for square 3x3 and 5x5 only" });
            else if (choice->kernels[i] == "sliding_column")
                c.push_back({ choice->kernels[i], sliding, 0 });
        }

        std::string reasons;
        const int k = choose(c, kernels::BOX_BLOCK_THREADS, s.device, reasons);
        if (tracing())
            trace(format("boxFilter %dx%d C%d on %s: ", ksize.width, ksize.height, cn, dev.arch.c_str()) +
                  (k < 0 ? std::string("CPU") : c[k].name) + traceTail(arch, op, choice, reasons));
        if (k < 0)
            return false;

        const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
        unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
        if (lds && c[k].kernel == lds)
            hipCheck(kernels::boxFilterLds(lds, sp, src.step[0], dp, dst.step[0], src.rows, src.cols,
                                           anchor.x, anchor.y, scale, s.stream), "boxFilter");
        else
            hipCheck(kernels::boxFilterSliding(sliding, sp, src.step[0], dp, dst.step[0], src.rows, src.cols,
                                               ksize.width, ksize.height, anchor.x, anchor.y, scale, s.stream),
                     "boxFilter");
        stream.retain({ src, dst });
        return true;
    }

    bool cvtTwoPlaneYUVtoBGR(const UMat& src, UMat& dst, int dcn, bool swapBlue, int uIdx,
                             Stream& stream) CV_OVERRIDE
    {
        if (src.dims != 2 || src.type() != CV_8UC1 || (dcn != 3 && dcn != 4) ||
            dst.type() != CV_MAKETYPE(CV_8U, dcn))
            return false;
        if (dst.rows % 2 != 0 || dst.cols % 2 != 0 || src.cols != dst.cols ||
            src.rows != dst.rows * 3 / 2 || dst.rows >= kernels::IMGPROC_MAX_ROWS)
            return false;
        // UMat::handle() needs no Mat view alive
        if (src.u == dst.u || src.u->refcount != 0 || dst.u->refcount != 0)
            return false;

        const HipStreamRef s = hipStreamOf(stream);
        HipDeviceGuard g(s.device);
        const DeviceFacts& dev = deviceFacts(s.device);
        const rules::ArchRules* arch = rulesFor(s.device);
        const rules::OpRules* op = opRules(arch, "nv12ToBgr");
        const rules::Choice* choice = op ? &op->choose({ dcn, 0 }) : 0;

        std::vector<Candidate> c;
        for (size_t i = 0; choice && i < choice->kernels.size(); i++)
            if (choice->kernels[i] == "chroma_share")
                c.push_back({ choice->kernels[i], kernels::nv12ToBgrKernelFor(dcn, swapBlue, uIdx), 0 });

        // 64x4 when the rules give no block
        const Size block = choice && !choice->block.empty() ? choice->block : Size(64, 4);
        std::string reasons;
        const int k = choose(c, block.area(), s.device, reasons);
        if (tracing())
            trace(format("cvtColor NV12 to %d channels on %s: ", dcn, dev.arch.c_str()) +
                  (k < 0 ? std::string("CPU") : format("%s %dx%d", c[k].name.c_str(), block.width, block.height)) +
                  traceTail(arch, op, choice, reasons));
        if (k < 0)
            return false;

        hipCheck(kernels::nv12ToBgr(c[k].kernel, (const unsigned char*)src.handle(ACCESS_READ) + src.offset,
                                    src.step[0], (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset,
                                    dst.step[0], dst.rows, dst.cols, block.width, block.height, s.stream),
                 "cvtColor");
        stream.retain({ src, dst });
        return true;
    }
};

} // namespace

ImgprocOps* imgprocOps()
{
    static HipImgprocOps ops;
    return &ops;
}

}}} // namespace cv::accel::hip_impl

#endif // HAVE_HIP
