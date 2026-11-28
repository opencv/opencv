// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP
#define OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP

// Non-CPU HAL, GPU: the kernels and operations of the CUDA and HIP backends, one folder per module laid
// out as the CPU HAL's (include/, and src/ with a file per family that declares its own kernels). Core
// compiles this file twice on its runtime (src/accel/gpu/gpu_runtime.hpp, included first): from
// src/cuda/accel_kernels.cu and src/hip/accel_kernels.hip with CV_ACCEL_GPU_KERNELS for the kernels,
// and from the end of src/accel/gpu/gpu_backend.impl.hpp for the operations, on its streams and errors.

#include <stddef.h>

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

const int BLOCK_X = 32, BLOCK_Y = 8;

/** @brief How a tunable kernel runs. At run time: @p blockX x @p blockY threads per block, and @p loopY patches
each thread covers down the image. At compile time, one of the patch shapes the build has (patchCompiled()):
@p px x @p py output pixels per thread. */
struct Launch
{
    int blockX = BLOCK_X, blockY = BLOCK_Y, loopY = 1;
    int px = 1, py = 1;
};

//! whether the tunable kernels are compiled for patches of @p px x @p py pixels
bool patchCompiled(int px, int py);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#ifdef CV_ACCEL_GPU_KERNELS

// the patch shapes the tunable kernels are compiled for, PX x PY output pixels per thread: each adds an instance
// per depth and channel count, so the list stays short (reuse along x, along y, and 2x2)
#define CV_ACCEL_GPU_PATCHES(X) X(1, 1) X(2, 1) X(4, 1) X(1, 2) X(1, 4) X(2, 2)

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

// gridDim.y is at most 65535: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

static inline dim3 grid(int rows, int cols, int bx, int by)
{
    const unsigned gy = (unsigned)((rows + by - 1) / by);
    return dim3((unsigned)((cols + bx - 1) / bx), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

//! the grid of @p l over @p rows x @p cols output pixels
static inline dim3 grid(int rows, int cols, const Launch& l)
{
    return grid(rows, cols, l.blockX * l.px, l.blockY * l.py * l.loopY);
}

/** @brief The instance @p Kernel::get<PX, PY>() for the patch of @p l, NULL when the build has none; @p Kernel
names a kernel template with its other arguments fixed. */
template<typename Kernel>
const void* patchKernel(const Launch& l)
{
#define CV_ACCEL_GPU_PATCH_CASE(PX, PY) if (l.px == PX && l.py == PY) return Kernel::template get<PX, PY>();
    CV_ACCEL_GPU_PATCHES(CV_ACCEL_GPU_PATCH_CASE)
#undef CV_ACCEL_GPU_PATCH_CASE
    return 0;
}

// LaunchKernel returns the status of this launch, where GetLastError() after <<<>>> would
// return the oldest error left on the thread by any earlier call

} // namespace

bool patchCompiled(int px, int py)
{
#define CV_ACCEL_GPU_PATCH_CASE(PX, PY) if (px == PX && py == PY) return true;
    CV_ACCEL_GPU_PATCHES(CV_ACCEL_GPU_PATCH_CASE)
#undef CV_ACCEL_GPU_PATCH_CASE
    return false;
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operations

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

// the kernels that take a Launch, by the names a tuning gives them
const char* const TUNABLE_KERNELS[] = { "blurRows", "blurCols", "warpAffine" };

std::string trimmed(const std::string& s)
{
    const size_t b = s.find_first_not_of(" \t"), e = s.find_last_not_of(" \t");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

std::vector<std::string> splitOn(const std::string& s, char sep)
{
    std::vector<std::string> parts;
    size_t b = 0;
    for (size_t e; (e = s.find(sep, b)) != std::string::npos; b = e + 1)
        parts.push_back(trimmed(s.substr(b, e - b)));
    parts.push_back(trimmed(s.substr(b)));
    return parts;
}

//! a positive decimal number, -1 for anything else
int positive(const std::string& s)
{
    if (s.empty() || s.size() > 9 || s.find_first_not_of("0123456789") != std::string::npos)
        return -1;
    const int v = atoi(s.c_str());
    return v > 0 ? v : -1;
}

//! "8UC3" or "CV_8UC3" as CV_8UC3, up to 4 channels; -1 for anything else
int typeFromString(const std::string& s)
{
    const std::string name = s.compare(0, 3, "CV_") == 0 ? s : "CV_" + s;
    for (int depth = 0; depth < CV_DEPTH_CURR_MAX; depth++)
        for (int cn = 1; cn <= 4; cn++)
            if (name == typeToString(CV_MAKETYPE(depth, cn)))
                return CV_MAKETYPE(depth, cn);
    return -1;
}

bool tunable(const std::string& kernel)
{
    for (size_t i = 0; i < sizeof(TUNABLE_KERNELS) / sizeof(TUNABLE_KERNELS[0]); i++)
        if (kernel == TUNABLE_KERNELS[i])
            return true;
    return false;
}

/* The launches of the tunable kernels: Launch(), or what OPENCV_ACCEL_<RUNTIME>_TUNING or Backend::setTuning() gives
   as "kernel[@type]:block=WxH,loopy=N,px=N,py=N;...", where a type's entry overrides its kernel's. */
class KernelTuning
{
public:
    static KernelTuning& get()
    {
        static KernelTuning* t = new KernelTuning();    // leaked: operations may run while the process exits
        return *t;
    }

    kernels::Launch launch(const char* kernel, int type)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_)
            apply(utils::getConfigurationParameterString(CV_GPU_TUNING_ENV));
        if (launches_.empty())
            return kernels::Launch();
        std::map<Key, kernels::Launch>::const_iterator i = launches_.find(Key(kernel, type));
        if (i == launches_.end())
            i = launches_.find(Key(kernel, -1));
        return i == launches_.end() ? kernels::Launch() : i->second;
    }

    //! replaces the tuning with @p spec, or with the environment's when it is empty; a bad spec changes nothing
    void set(const String& spec)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        apply(spec.empty() ? utils::getConfigurationParameterString(CV_GPU_TUNING_ENV) : spec);
    }

private:
    typedef std::pair<std::string, int> Key;    // kernel, and type or -1 for every type
    typedef std::vector<std::pair<std::string, std::string> > Settings;

    void apply(const std::string& spec)
    {
        std::map<Key, Settings> entries;
        const std::vector<std::string> parts = splitOn(spec, ';');
        for (size_t i = 0; i < parts.size(); i++)
            if (!parts[i].empty())
                parseEntry(parts[i], entries);
        std::map<Key, kernels::Launch> launches;
        for (std::map<Key, Settings>::const_iterator e = entries.begin(); e != entries.end(); ++e)
        {
            kernels::Launch l;
            std::map<Key, Settings>::const_iterator all = entries.find(Key(e->first.first, -1));
            if (e->first.second >= 0 && all != entries.end())
                applySettings(all->second, l);
            applySettings(e->second, l);
            if (l.blockX * l.blockY > 1024)
                fail(e->first.first, "a block has at most 1024 threads");
            if (!kernels::patchCompiled(l.px, l.py))
                fail(e->first.first, cv::format("no %dx%d patch is compiled", l.px, l.py));
            launches[e->first] = l;
        }
        launches_.swap(launches);
        loaded_ = true;
    }

    static void parseEntry(const std::string& entry, std::map<Key, Settings>& entries)
    {
        const size_t colon = entry.find(':');
        if (colon == std::string::npos)
            fail(entry, "expected kernel:key=value,...");
        const std::string name = trimmed(entry.substr(0, colon));
        const size_t at = name.find('@');
        const std::string kernel = name.substr(0, at);
        int type = -1;
        if (at != std::string::npos && (type = typeFromString(name.substr(at + 1))) < 0)
            fail(entry, "unknown type after @");
        if (!tunable(kernel))
            fail(entry, "unknown kernel; the tunable ones are blurRows, blurCols and warpAffine");
        Settings& s = entries[Key(kernel, type)];
        const std::vector<std::string> kvs = splitOn(entry.substr(colon + 1), ',');
        for (size_t i = 0; i < kvs.size(); i++)
        {
            const size_t eq = kvs[i].find('=');
            if (eq == std::string::npos)
                fail(entry, "expected key=value");
            s.push_back(std::make_pair(trimmed(kvs[i].substr(0, eq)), trimmed(kvs[i].substr(eq + 1))));
            kernels::Launch probe;
            applySetting(entry, s.back(), probe);   // reports a bad setting at its entry
        }
    }

    static void applySettings(const Settings& s, kernels::Launch& l)
    {
        for (size_t i = 0; i < s.size(); i++)
            applySetting(std::string(), s[i], l);
    }

    static void applySetting(const std::string& entry, const std::pair<std::string, std::string>& kv,
                             kernels::Launch& l)
    {
        const std::string& key = kv.first;
        const std::string& value = kv.second;
        if (key == "block")
        {
            const size_t x = value.find('x');
            l.blockX = x == std::string::npos ? -1 : positive(value.substr(0, x));
            l.blockY = x == std::string::npos ? -1 : positive(value.substr(x + 1));
            if (l.blockX < 0 || l.blockY < 0)
                fail(entry, "block takes WxH");
            return;
        }
        int* field = key == "loopy" ? &l.loopY : key == "px" ? &l.px : key == "py" ? &l.py : 0;
        if (!field)
            fail(entry, "unknown key '" + key + "'; the keys are block, loopy, px and py");
        if ((*field = positive(value)) < 0)
            fail(entry, key + " takes a positive number");
    }

    CV_NORETURN static void fail(const std::string& where, const std::string& what)
    {
        CV_Error_(Error::StsBadArg, ("accel(" CV_GPU_RUNTIME ") tuning '%s': %s", where.c_str(), what.c_str()));
    }

    std::mutex mutex_;
    bool loaded_ = false;
    std::map<Key, kernels::Launch> launches_;
};

//! the launch of the tunable @p kernel for an image of @p type
kernels::Launch tunedLaunch(const char* kernel, int type)
{
    return KernelTuning::get().launch(kernel, type);
}

void setKernelTuning(const String& spec)
{
    KernelTuning::get().set(spec);
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#include "core/src/precomp.hpp"
#include "imgproc/src/precomp.hpp"

#ifdef CV_ACCEL_GPU_KERNELS
#undef CV_ACCEL_GPU_PATCHES
#endif

#endif // OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP
