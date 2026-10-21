Non-CPU HAL: accelerators behind cv:: functions {#non_cpu_hal}
===============================================

UMat data can live on an accelerator: a discrete GPU, an integrated GPU or an NPU. When a cv::
function gets such a UMat, it runs on that device if the device's backend implements the
function, and on the CPU otherwise. Either way its output is created on the device, so a
pipeline stays where its data is. User code does not change: it is the T-API, extended beyond
OpenCL.

This is the first version. It keeps the model small on purpose: two kinds of device memory,
one rule for where a call runs, and a C++ interface for backends.

Using it
--------

```cpp
#include <opencv2/core/accel.hpp>

cv::accel::Device gpu = cv::accel::Device::find("cuda:0");   // empty when there is none

cv::UMat src;
src.allocator = gpu.allocator();      // new data of src goes to the GPU
frame.copyTo(src);                    // upload
src.allocator = nullptr;

cv::UMat flipped, rotated;
cv::flip(src, flipped, 1);            // runs on the GPU, flipped is created there
cv::transpose(flipped, rotated);      // runs on the GPU
cv::Mat result = rotated.getMat(cv::ACCESS_READ);   // waits for the GPU, then downloads
```

`cv::accel::ExecutionScope scope(gpu);` makes the calls of this thread run on the device, and
places new UMats there. `cv::accel::getTransferStats()` counts device calls, CPU fallbacks and
the bytes copied; `cv::accel::setTransferPolicy(cv::accel::TRANSFER_FORBID)` (or
`OPENCV_ACCEL_TRANSFER_POLICY=forbid`) turns every copy a call would make on its own into an
exception, to find them.

Memory
------

OpenCV knows two kinds of memory, host and device, as the T-API does. Each device decides how
the CPU reaches its memory:

| | copy (discrete GPUs) | shared (NPUs, integrated GPUs, Jetson) |
|---|---|---|
| allocation | device memory | memory the CPU can address |
| `UMat::getMat()` | waits for device work, downloads into a host copy when it is stale | waits for device work, refreshes the CPU cache, returns the same bytes |
| releasing the last Mat view | uploads what the CPU wrote, queued on the device's stream | flushes the CPU cache when the CPU wrote |
| a call that falls back to the CPU | one copy down per input, one up per output | no copy |

Cache maintenance for non-coherent memory is the backend's business
(`Backend::beginCpuAccess`, `Backend::endCpuAccess`); core only calls it at the right moments.

A read stays a read: `getMat(ACCESS_READ)` on device memory does not mark the device copy
stale, so a CPU fallback reads its inputs without uploading them back. (Other UMats keep the
T-API behaviour, where every view is read-write.) A buffer that nothing has written yet is not
downloaded at all.

Every device buffer remembers the last device work that writes it and, per stream, the last
work that reads it. The CPU waits for those before it reads or writes; memory is freed only
once the work using it has finished (released buffers wait in a list until then).
`UMat::handle()` waits too, so the caller's own kernels see finished data.

The state lives in `UMatData::allocatorContext` and the T-API flags of `UMatData`, so `UMat` and
`UMatData` are unchanged.

Where a call runs
-----------------

A function has one hook line before its OpenCL branch. The hook picks the device with a fixed
rule, never by counting the bytes each option would move:

1. the device of this thread's `ExecutionScope`, if there is one;
2. otherwise the device that holds the first UMat input.

If no input is in device memory, the hook does nothing and the function runs as before.

Inputs elsewhere (a Mat, an OpenCL UMat, another device) are copied to that device. Outputs are
created on it: an empty UMat, or a whole UMat that only this header holds (no region, no Mat
view, no allocator of its own, no fixed size or type), is released and created again there.
Every other output (a Mat, a region of a larger UMat, user memory) gets the result copied in.
An output that overlaps an input is computed in a temporary first.

When the device does not run the call (the backend lacks the operation or declines this
particular one, or the CPU holds a Mat view of an argument), the outputs are still created on
the device and the function goes on to its OpenCL and CPU code. The OpenCL code declines device
memory it does not own (#30259), so the CPU runs it, through `getMat()`.

Transfers and waiting
---------------------

* Device work is queued on the device's stream and runs asynchronously.
* Downloads are synchronous: the data can be used as soon as the call returns.
* Uploads from OpenCV's own host copy (when the last Mat view goes away) are queued
  asynchronously; the host copy is not touched until they complete.
* Uploads from user memory (a Mat argument, `Mat::copyTo(umat)`) return only once the backend
  has read the memory, as the T-API's do.
* Nothing is moved "later, when needed": a write by the CPU is uploaded when its view is
  released.

A device error surfaces where the CPU next waits for that work, for example in `getMat()`.

Writing a backend
-----------------

`opencv2/core/accel_backend.hpp` is a source-level C++ interface: backends are compiled against
the headers of the OpenCV version they run with, inside OpenCV or as plugins built with it.
They use `UMat` and `cv::Exception` like any other OpenCV code.

```cpp
class MyBackend : public cv::accel::Backend
{
public:
    cv::String name() const override { return "mynpu"; }
    int deviceCount() const override;
    cv::String deviceDescription(int device) const override;
    bool sharesHostMemory(int device) const override { return true; }

    void* allocate(int device, size_t size) override;      // NULL when out of memory
    void deallocate(int device, void* ptr) override;       // called once the work using it is done
    void beginCpuAccess(int device, void* ptr, size_t size, bool write) override;   // cache invalidate
    void endCpuAccess(int device, void* ptr, size_t size, bool written) override;   // cache clean
    void copy(int device, cv::accel::CopyKind kind, void* dst, size_t dstStep,
              const void* src, size_t srcStep, size_t widthBytes, size_t rows,
              cv::accel::Stream& stream, bool hostStable) override;
    cv::Ptr<cv::accel::Stream> createStream(int device) override;
    cv::accel::Ops* ops(const cv::String& module) override;   // &myCoreOps for "core"
};

class MyCoreOps : public cv::accel::CoreOps
{
public:
    bool flip(const cv::UMat& src, cv::UMat& dst, int flipCode, cv::accel::Stream& s) override
    {
        if (src.depth() != CV_8U)
            return false;                   // not this one: OpenCV runs it on the CPU
        launchFlip(cv::accel::devicePtr(src), src.step[0], cv::accel::devicePtr(dst), dst.step[0], ...);
        return true;
    }
};

cv::accel::registerBackend(cv::makePtr<MyBackend>());
```

The rules:

* An operation returns false before it queues anything when the device cannot run that call.
  Errors are thrown.
* Arguments are already in the device's memory, ordered on the stream, and outputs have their
  final size and type and never overlap inputs.
* `copy()` with `COPY_HOST_TO_DEVICE` and `hostStable` false copies from user memory, which may
  change as soon as it returns. In every other case OpenCV keeps the host memory unchanged until
  the stream reaches the copy.
* Methods are called from any thread.

The CUDA backend (`src/accel/backend_cuda.cpp`) registers itself when OpenCV is built with CUDA
and a device is present (`OPENCV_ACCEL_CUDA=0` disables it). Discrete GPUs get memory from the
stream-ordered pool and pinned host copies; integrated GPUs from Xavier on get pinned memory
mapped into the GPU, which is shared. It implements `flip` and `transpose`
(`src/cuda/accel_kernels.cu`).

Adding an operation
-------------------

1. Add a virtual method, returning false by default, to the module's interface (`CoreOps` for
   core).
2. Add the hook to the function, before `CV_OCL_RUN`:

```cpp
static bool accel_flip(InputArray src, OutputArray dst, int code)
{
    const Size sz = src.size();
    accel::detail::OpCall call("flip");
    call.input(src).output(dst, MatShape({sz.height, sz.width}), src.type());
    return call.run<accel::CoreOps>([&](accel::CoreOps& ops, std::vector<UMat>& a, accel::Stream& s) {
        return ops.flip(a[0], a[1], code, s);
    });
}
...
CV_ACCEL_RUN(_src.isUMat() || _dst.isUMat(), accel_flip(_src, _dst, flip_mode))
```

3. Implement it in the backends that can.

Not in this version
-------------------

These are planned on top of the model above; none of them needs a new kind of memory:

* wrapping and exporting dma-bufs (`wrapDmaBuf`, `exportDmaBuf`), so a camera frame reaches a
  2D engine and an NPU without copies;
* an order of preferred devices for shared memory that several devices can reach;
* loading backends as plugins (one `extern "C"` function returning the `Backend`, with a
  version check, as the DNN plugins do);
* streams other than the device's default one;
* functions without a hook create their outputs on the device only inside an `ExecutionScope`;
* the HIP backend.

The earlier, larger design (protected memory, multi-plane images, page-protected views,
several host-access classes) is kept on the `non-cpu-hal` branch.

Tests
-----

`modules/core/test/test_accel.cpp` registers an emulated backend with three devices: discrete
memory at addresses the CPU cannot use, coherent shared memory, and non-coherent shared memory
whose CPU "cache" makes a missing cache call visible as stale data. Its streams run work on a
thread, after an optional delay, so a missing wait shows up as wrong data, and work on freed
memory is counted. The `Core_Accel_CUDA` tests run when a CUDA device is present.
