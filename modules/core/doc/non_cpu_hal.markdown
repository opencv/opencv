Non-CPU HAL: accelerators behind cv:: functions {#non_cpu_hal}
===============================================

UMat data can be placed on an accelerator, such as a discrete GPU, an integrated GPU or an NPU.
A cv:: function with an accelerator hook (cv::flip and cv::transpose in this version) runs on the
device of its input UMat if the device's backend implements it, and on the CPU otherwise. Either
way its output is created on the device, so a pipeline of such functions stays where its data
is. Functions without a hook run on the CPU and create regular UMats for their outputs.

This first version is deliberately small: explicit placement, computation that follows its input,
one stream per device and CPU thread, and backends built into OpenCV.

Using it
--------

```cpp
#include <opencv2/core/accel.hpp>

cv::accel::Device gpu = cv::accel::Device::find("cuda:0");   // empty when there is none

cv::UMat src = frame.upload(gpu);     // copies the Mat to the GPU
cv::UMat flipped, rotated;
cv::flip(src, flipped, 1);            // runs on the GPU, flipped is created there
cv::transpose(flipped, rotated);      // runs on the GPU
cv::Mat result = rotated.getMat(cv::ACCESS_READ);   // waits for the GPU, then downloads
```

Placement is always explicit:

* `UMat::createOn(device, shape, type)` allocates a UMat on a device, or keeps its data when it is
  already there with that shape and type. Data on another device is released, as data of another
  size would be.
* `Mat::upload(device)` copies a Mat to a device. It returns once the Mat has been read, so the
  Mat can be changed or released right after.
* `UMat::upload(device)` returns the UMat itself when its data is already on that device, and a
  copy otherwise. A copy between two devices happens only when it is asked for this way.
* An empty `Device` stands for the CPU: `upload(cv::accel::Device())` brings data back into a
  regular UMat.

`cv::accel::getDevice(umat)` tells where a UMat's data is. `Device::enumerate()` lists the
devices, `Device::find("cuda")` finds one by name.

Where a call runs
-----------------

A function with an accelerator hook runs on the device that holds its input UMat. There is no
device preference and no automatic placement: data moves between devices only through `upload()`.

The output is created on that device with `createOn()`, unless it is a region of a larger UMat or
a fixed-size array that lives elsewhere: such an output stays where it is and the CPU writes it.
A region or fixed-size array of another size or type is left to the CPU code, which reports the
mismatch as it does for any UMat.
If the device's backend does not implement the operation, or declines this particular call, the
function goes on to its OpenCL and CPU code. The OpenCL code declines memory it does not own, so
the CPU runs the call through `UMat::getMat()`, and the output is still on the device.

Streams and threads
-------------------

Device work is queued on a stream and the call returns before it has finished. Each CPU thread
has its own stream on each device (`Device::threadStream()`), created on first use, so the work
of one thread runs in order.

The CPU waits for the work of its own thread where it needs the data: `UMat::getMat()`, copies
to host memory, and `cv::accel::synchronize(device)`. Work queued by different threads is not
ordered against each other. A thread that hands device data to another thread calls
`cv::accel::synchronize(device)` first, as with the per-thread queues of OpenCL.

The buffers an operation uses stay alive until its work has finished, even when the caller
releases its UMats at once. A stream keeps a reference to them (`Stream::retain()`), as the
OpenCL backend does for the UMats of a running kernel, and drops it the next time the thread
waits for the stream or queues more work after that work is done. When a thread exits, its
streams finish their work and release what they kept. An allocation that runs out of device
memory waits for the calling thread's stream and tries again; buffers kept by other threads'
streams are not reclaimed there.

A device error surfaces where the CPU next waits for that work, for example in `getMat()` or
`synchronize()`.

Memory
------

Each UMatData keeps its device in `UMatData::allocatorContext`; the allocator is one per backend.

Memory the CPU cannot address (a discrete GPU's) gets a host copy on the first CPU access.
`getMat()` downloads into it when it is stale, and releasing the last Mat view uploads what the
CPU wrote. A read stays a read: `getMat(ACCESS_READ)` does not mark the device copy stale, so a
CPU fallback reads its inputs without uploading them back. A buffer that nothing has written yet
is not downloaded.

Memory the CPU addresses (an integrated GPU's, when the backend says so with
`Backend::hostAddress()`) is used in place: `getMat()` waits for the thread's device work and
returns the device's own bytes.

Writing a backend
-----------------

A backend implements `cv::accel::Backend` (`opencv2/core/private/accel_backend.hpp`, a header
that is not installed) and is built into OpenCV behind its own build flag. It provides device memory, copies, streams and events, and one
operations interface per module:

```cpp
class MyBackend : public cv::accel::Backend
{
public:
    cv::String name() const override { return "mynpu"; }
    int deviceCount() const override;
    cv::String deviceDescription(int device) const override;
    void* allocate(int device, size_t size) override;           // NULL when out of memory
    void deallocate(int device, void* handle) override;         // once the work using it is done
    void copy(int device, cv::accel::CopyKind kind, cv::accel::DeviceRef dst, size_t dstStep,
              cv::accel::DeviceRef src, size_t srcStep, size_t widthBytes, size_t rows,
              cv::accel::Stream& stream) override;
    cv::Ptr<cv::accel::Stream> createStream(int device) override;
    cv::accel::Ops* ops(const char* module) override;           // &myCoreOps for "core"
};

class MyCoreOps : public cv::accel::CoreOps
{
public:
    bool flip(const cv::UMat& src, cv::UMat& dst, int flipCode, cv::accel::Stream& stream) override
    {
        if (src.depth() != CV_8U || src.u->refcount || dst.u->refcount)
            return false;                   // not this one: the CPU runs it
        const void* s = (const uchar*)src.handle(cv::ACCESS_READ) + src.offset;
        void* d = (uchar*)dst.handle(cv::ACCESS_WRITE) + dst.offset;
        launchFlip(s, src.step[0], d, dst.step[0], ..., stream);
        stream.retain({ src, dst });        // alive until the flip has run
        return true;
    }
};
```

The rules:

* An operation returns false, before it queues anything, when it cannot run that call. One case
  every operation declines is a Mat view of an argument being alive (`u->refcount`), because
  `UMat::handle()` needs the CPU to be done with the data.
* `UMat::handle()` gives the device memory and does the T-API bookkeeping: `ACCESS_WRITE` marks
  the host copy stale. Errors are thrown.
* The output may be the input itself. Each operation handles that as suits it, for example an
  in-place flip that swaps mirrored elements, or declines it.
* `copy()` queues on the stream after the work before it. Copies between host and device memory
  have finished when it returns; device-to-device copies may still run.
* Methods are called from any thread.

A built-in backend registers itself in `registry.cpp`; tests register theirs with
`cv::accel::registerBackend()`.

### Backends in OpenCV

* CUDA (`src/accel/backend_cuda.cpp`), built in when OpenCV is built with CUDA. Discrete GPUs use
  device memory from a stream-ordered pool of the backend's own; integrated GPUs from Xavier on
  use pinned host memory mapped into the GPU, which the CPU uses in place. It implements `flip`
  and `transpose`, in place too. Its kernels and runtime calls (`src/accel/gpu/`) are written
  once for the CUDA and HIP runtimes.

Adding an operation
-------------------

1. Add a virtual method, returning false by default, to the module's interface (`CoreOps` for
   core).
2. Add the hook to the function, before `CV_OCL_RUN`:

```cpp
static bool accel_flip(InputArray _src, OutputArray _dst, int flipCode)
{
    if (!_src.isUMat() || !_dst.isUMat() || _src.empty())
        return false;

    const UMat src = _src.getUMat();
    accel::Device device = accel::getDevice(src);
    if (device.empty() || !accelOutputUsable(_dst, device, src.size(), src.type()))
        return false;

    UMat& dst = _dst.getUMatRef();
    dst.createOn(device, src.shape(), src.type());

    accel::CoreOps* ops = device.ops<accel::CoreOps>("core");
    if (!ops)
        return false;   // CPU fallback; the output is already on the device

    return ops->flip(src, dst, flipCode, device.threadStream());
}
...
if (accel_flip(_src, _dst, flip_mode))
    return;
```

3. Implement it in the backends that can.

Not in this version
-------------------

These are left for later, once the basic model has been tried on real backends and pipelines:
user-selected streams and execution scopes, dma-buf import and export, memory kinds chosen per
buffer, pinned-memory budgets and allocation caches, automatic placement across devices, and
backends loaded as plugins.

Tests
-----

`modules/core/test/test_accel.cpp` registers an emulated backend with three devices: memory at
addresses the CPU cannot use, memory the CPU addresses, and a second device of the first kind for
transfers between devices. Its streams run work on a thread of their own, after an optional
delay, so a missing wait shows up as wrong data, and work on freed memory or on another device's
memory is counted. The `Core_Accel_CUDA` tests run when a CUDA device is present, and
`perf_accel.cpp` measures flip and transpose there.
