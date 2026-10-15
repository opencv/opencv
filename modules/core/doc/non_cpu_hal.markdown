# Non-CPU HAL: accelerators behind cv:: functions {#non_cpu_hal}

@tableofcontents

This page describes how `cv::` functions run on accelerators (discrete GPUs, integrated
GPUs, NPUs, DSPs, 2D engines) through one interface, and how UMat data that lives on such
a device is kept correct when the CPU, the device, or several devices use it. The branch
`non-cpu-hal` contains a working core of this design: the memory layer and its coherence
protocol, the dispatcher, the backend C interface, an emulator backend for tests, the CUDA
memory primitives, and hooks in `cv::flip` and `cv::transpose`.

The design starts from one observation: once an accelerator shares memory with the CPU, or
once work runs asynchronously, "which copy is current" stops being enough. A buffer also
needs to know who is still using it, whether the CPU caches see what the device wrote, and
whether the CPU may touch it at all right now. Today each OpenCV allocator answers these
questions on its own, in slightly different ways. This design answers them once, in core.

## What does not work today

Checked against 5.x at e13b74e9a7 (15 October 2026) with PR #30259 applied.

- **The dispatch key is the allocator's identity.** #30259 declines CUDA buffers in the
  OpenCL path with `currAllocator == cuda::getCudaAllocator()`, after
  `currAllocator != getOpenCLAllocator()` was rejected in review because it cancels OpenCL
  for user allocators. Every new device API would need another identity check.
- **The coherence state machine is re-implemented by every allocator.** The OpenCL
  allocator (zero-copy maps, `CL_MEM_USE_HOST_PTR`, SVM, deferred cleanup), the CUDA
  allocator from #29451 and the HIP allocator from #29372 each have their own version, and
  they differ in details (the HIP one frees its host copy, the CUDA one keeps it).
- **Every CPU fallback writes back.** `UMat::getMat()` widens every request to
  `ACCESS_RW`, so a read-only use of device data costs a download and an upload.
- **There is no notion of shared memory.** CUDA and HIP buffers are always `cudaMalloc` /
  `hipMalloc` plus a host copy, even on Jetson or on an APU where the CPU could use the
  same bytes. `isLargeBar` and `integrated` are queried and not used.
- **Transfers stage through hidden copies.** Host copies come from `fastMalloc`, which is
  pageable, so every `cudaMemcpy` is staged by the driver through its own pinned buffer.
- **Nothing tracks asynchronous use outside OpenCL.** OpenCL kernels keep their arguments
  alive until a completion callback, but there is no general notion of "this buffer is
  still being read by stream B". Work on a second thread's queue is not ordered at all.
- **Per-function device code bypasses the state.** #29372 adds `#ifdef HAVE_HIP` blocks to
  `setTo`, `copyTo` and `convertTo` that read `u->handle + offset` directly: if the host copy
  is newer, the kernel reads stale data, and an ROI write leaves a buffer that is half
  current on each side.
- **Placement can change silently.** `UMat::create` falls back to another allocator when
  allocation fails, and whether an output ends up on the device depends on which
  functions happen to have a device implementation.

## Principles

1. **Key on what the memory is, not on who allocated it.** An allocator describes each
   allocation (`MatAllocator::getMemoryDesc`); code asks the description.
2. **One coherence protocol, in core.** Backends provide primitives (allocate, copy,
   begin/end CPU access, fences); they never change UMat state.
3. **Correct first, then zero-copy.** The protocol decides when to wait, flush, invalidate
   or copy. When memory is shared, the copy steps disappear and the rest stays.
4. **Where data lives does not depend on op coverage.** A cv:: call on device data produces
   its output on that device whether the backend implemented the call or not.
5. **No hidden copies.** Every transfer is visible in `cv::accel::TransferStats`, and
   `TRANSFER_FORBID` turns implicit ones into errors.
6. **A stable C boundary.** Backends can ship inside OpenCV or as plugins, written in C or
   C++, against a versioned table where entries are only ever appended.

## Architecture

```
  cv::flip(src, dst, 1)
      |  CV_ACCEL_RUN(...)            one line per function, before CV_OCL_RUN
      v
  OpCall (core)                       describe args -> pick device -> QUERY -> plan
      |                               -> acquire buffers -> EXECUTE -> record fence
      |-------------------------------------------.
      v                                           v
  coherence protocol (core)               op tables per module ("core", "imgproc", ...)
  UMatData flags + BufferState                    |
      |                                           |
  ======================== llapi/accel_hal.h (C, versioned) ========================
      |                                           |
  backend: memory, streams, fences        backend: kernels
  (CUDA, HIP, Level Zero, Vulkan, RKNPU, QNN, a 2D engine, the test emulator, ...)
```

The layers below the dashed line are the backend. Everything above it is shared:
`UMat`, the allocator, the protocol and the dispatcher are the same for every device.

If no device buffer exists in the process, the hook costs one relaxed atomic load.

## Memory model

### The memory descriptor

Every allocation is described by `cvAccelMemoryDesc` (see `llapi/accel_hal.h`):

| field | meaning |
|---|---|
| `backend_id`, `device_index` | which accelerator owns it; 0 means host or a legacy allocator |
| `kind` | `HOST`, `HOST_PINNED`, `DEVICE`, `SHARED` (one physical copy for CPU and device), `MANAGED` (migrating) |
| `host_access` | how the CPU may touch it (below) |
| `handle_kind` | what `UMatData::handle` holds: host pointer, device pointer, `cl_mem`, `VkBuffer`, dma-buf fd, opaque |
| `flags` | imported, read-only, protected |
| `layout_modifier` | 0 for linear memory; otherwise tiled or compressed, with no linear CPU view |
| `coherence_granule` | the cache-line or atom size that cache operations work on |

The allocator reports it through `MatAllocator::getMemoryDesc`. The default returns
"unknown", which keeps the behaviour from before the method existed; that is what makes
the OpenCL guard from #30259 general without breaking user allocators:

```cpp
// ocl.cpp, Kernel::set and isCLBuffer
static bool isNonOpenCLMemory(const UMatData* u)
{
    cvAccelMemoryDesc d;
    if (!u || !u->currAllocator || !u->currAllocator->getMemoryDesc(u, &d))
        return false;       // user allocators: unchanged
    if (d.backend_id != 0)
        return true;        // accelerator memory
    return d.handle_kind is a device pointer, VkBuffer, dma-buf or opaque handle;
}
```

The CUDA allocator of #29451 reports a device pointer, the OpenCL allocator a `cl_mem`, the
host allocator a host pointer. No allocator pointers are compared.

### Host-access classes

The class decides what core does around every CPU access. It is reported per allocation,
at run time, because the same family of chips can be coherent on one part and not on the
next (Qualcomm marks its DSP banks coherent from SM8550 on but not on SM8250; Jetson pinned
memory is uncached before Xavier).

| class | examples | what core does around CPU access |
|---|---|---|
| `NONE` | protected video buffers, tiled or compressed layouts | refuses; a CPU fallback is an error |
| `COPY` | discrete VRAM, `cudaMalloc` memory on Jetson | keeps a pinned host copy; transfers only when stale |
| `COHERENT` | Apple shared storage, MI300A, Jetson Orin pinned memory, Intel LLC-shared iGPU | waits for fences |
| `NONCOHERENT` | dma-heap buffers used by Rockchip, AMD XDNA or Qualcomm NPUs, non-coherent Vulkan types | waits for fences, then `begin_cpu_access` (invalidate) / `end_cpu_access` (clean) |
| `UNCACHED` | write-combined or uncached mappings, resizable-BAR VRAM | like `COPY`: CPU kernels never read uncached memory directly |
| `DEVICE_IDLE` | CUDA managed memory without concurrent access (Jetson before Thor, Windows) | drains the device and holds back new device work while the CPU holds it |

One-way I/O coherence (the device snoops CPU caches, as on Jetson Xavier and Orin) counts
as `COHERENT`, because a backend has to guarantee that a completed fence makes its writes
visible to the CPU (backend rule 2 below).

## The coherence protocol

The code is `modules/core/src/accel/buffer_protocol.cpp`.

### State

Which copy is current stays in `UMatData::flags`, as for every allocator:
`HOST_COPY_OBSOLETE` (the CPU view is stale) and `DEVICE_COPY_OBSOLETE` (the device copy is
stale). Both set means the content is uninitialized, and no transfer is ever needed for it.
For memory with a single physical copy, `HOST_COPY_OBSOLETE` means "the CPU has to wait and
invalidate before reading".

Everything else lives in `BufferState`, created by the device allocator and reached
through the new `UMatData::accelState` pointer (NULL for host and OpenCL memory):

- the allocation and its description, and the pinned host copy for `COPY` memory;
- `writeFence`, the last device write, and `readFences`, the device reads since then (one
  per stream; a newer fence on an in-order stream covers the older one);
- `pendingDeviceOps`, operations between acquire and record;
- `hostMapped` and `hostAccess`, the current CPU mapping and the union of its access flags;
- `busy`, set while a CPU-side transition waits outside of the buffer lock;
- `poisoned`, set when the device work that produced the content failed.

### CPU access: `UMat::getMat`, `copyTo` into a Mat, `UMat::handle`

1. Wait until no other transition runs and no device operation is between acquire and record.
2. Wait on the CPU for conflicting device work: reading waits for the last write, writing
   also waits for every read (write-after-read). The lock is released during the wait.
3. Per class: for `COPY` memory, download into the host copy only if it is stale and the
   content was ever written; for memory with a CPU view, call `begin_cpu_access`, which
   invalidates CPU caches on non-coherent memory.
4. When the last Mat view goes away: for `COPY` memory a write marks the device copy stale
   (it is uploaded when the device needs it, not now); for memory with a CPU view
   `end_cpu_access` cleans CPU caches, so no dirty line can later be evicted over data the
   device wrote.

`ACCESS_READ` is honoured: a read-only view never writes back. `ACCESS_WRITE` is never
taken as "the old content may be discarded", because `OutputArray::getMat()` uses it for
partial writes (masked copies, ROIs).

### Device access: inside `OpCall`

1. **Acquire** each argument for the operation's stream. If a Mat view is alive, the device
   may read when no view was requested for writing, and may never write: a conflicting
   operation runs on the CPU instead of racing with the views.
2. If the host copy is newer and the operation reads the buffer or writes only part of it,
   upload the host copy first, as a device write that waits for everyone still using the
   old device content.
3. Collect the fences the operation must wait for: the last write if it came from another
   stream, and for writes also the reads from other streams. Same backend: a device-side
   wait. Different backends: a sync_file when both support it, otherwise a CPU wait.
4. **Record** one fence for the operation and store it as the new write fence of the
   outputs and as a read fence of the inputs.

### Transitions at a glance

| event | before | after | transfer |
|---|---|---|---|
| allocate | | both obsolete (uninitialized) | none |
| CPU read, host copy stale | device current | both current | 1 download (`COPY` only) |
| CPU read, host copy current | | unchanged | none |
| CPU write ends | | device obsolete (`COPY`) / caches cleaned (direct) | none now |
| device read, device copy stale | host current | both current | 1 upload |
| device write of the whole buffer | anything | host obsolete | none |
| device write of an ROI, device copy stale | host current | host obsolete | 1 upload first |
| release with work in flight | | freed when all fences complete | none |

### How shared memory is kept safe

This is the part that matters for NPUs and integrated GPUs, where the CPU and the device
see the same DRAM:

1. **Ordering.** Neither side reads before the other finished writing, and neither writes
   while the other still reads. Shared memory removes copies, not waits.
2. **Exclusivity.** A live writable Mat view keeps the device away; a device write waits
   for every reader; CPU and device never write the same buffer at the same time. This
   matches the strictest rule found in practice (Arm's import extension: host and device
   must not touch any part of a buffer concurrently, even non-overlapping parts).
3. **Cache maintenance.** Non-coherent memory is invalidated after the device wrote and
   cleaned before the device may touch it again. Backends may widen the range to the whole
   buffer (dma-buf has no range) or to atoms (Vulkan `nonCoherentAtomSize`).
4. **No shared cache lines.** Non-coherent allocations start and end on
   `coherence_granule` boundaries, so cache maintenance on one buffer never touches bytes of
   another (on HIP, unaligned registrations lose writes: "one of the writes will win").
5. **Device-idle memory.** On parts where touching managed memory during any kernel faults,
   the CPU announces itself on the device's gate, waits until no submission is in
   progress, drains the device, and while it holds the memory every new operation on that
   device runs on the CPU. A handshake on two counters avoids lock-order deadlocks.
6. **Bounds.** Before every EXECUTE call, core checks that each view lies inside its
   allocation (overflow-safe). On SoCs where the NPU has no IOMMU, a wrong stride is not an
   error but memory corruption.
7. **Lifetime.** A released buffer is freed only when its fences complete, so no DMA engine
   writes into memory that was reused. Imported memory goes back to its owner only after the
   device is done.
8. **Protected memory.** Data can be uploaded into it and processed on the device, but it
   never reaches the CPU: `getMat`, downloads and CPU fallbacks throw.
9. **Imported dma-bufs.** Buffers from cameras, decoders or displays may carry implicit
   fences. The backend waits for them before use and attaches its own fence after writing
   (`DMA_BUF_IOCTL_EXPORT_SYNC_FILE` / `IMPORT_SYNC_FILE`), so other drivers stay ordered.

### Discrete GPUs without extra copies

- The host copy of device memory is pinned, from a size-keyed cache
  (`OPENCV_ACCEL_PINNED_CACHE_MB`), so every transfer is one DMA and pinning never happens
  per buffer (pinning is slow and stalls the device's streams).
- A read-only CPU use costs one download and no write-back.
- A CPU write is uploaded once, when the device next needs the data. Three CPU fallbacks in
  a row cost one download, not three round trips.
- Fresh outputs are never downloaded before the CPU overwrites them.
- `UMat::clone()` and `copyTo` between buffers of one device are device-to-device copies;
  ROI transfers are 2D or n-D copies, never packed.
- User host memory is not pinned on the hot path. `Mat::getUMat()` under a device scope
  allocates device memory and uploads lazily; zero-copy use of user memory is explicit
  (`wrapExternalMemory`), or automatic when a device can address host memory (HMM/ATS,
  unified memory) and a Mat is passed straight to an operation.

## Dispatch

### The hook

A function gets one line before its OpenCL branch, plus a small function that maps its
arguments to an op-table entry:

```cpp
static int accel_flip(InputArray src, OutputArray dst, int mode)
{
    accel::detail::OpCall call("core.flip", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(src);
    call.output(dst, src.shape(), src.type());
    return call.run([mode](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        const cvAccelOpsCore_v0* ops = (const cvAccelOpsCore_v0*)t;
        return CV_ACCEL_OP_PRESENT(cvAccelOpsCore_v0, ops, flip) ? ops->flip(c, &v[0], &v[1], mode)
                                                                 : CV_ACCEL_NOT_IMPLEMENTED;
    });
}

void flip(InputArray _src, OutputArray _dst, int flip_mode)
{
    ...
    CV_ACCEL_RUN(_src.isUMat() || _dst.isUMat(), accel_flip(_src, _dst, flip_mode))
    CV_OCL_RUN(_dst.isUMat(), ocl_flip(_src, _dst, flip_mode))
    ...
}
```

Operations are typed C functions in per-module tables, like the CPU HAL's `cv_hal_*`
entries, so the compiler checks every argument. Each call is made twice: first in `QUERY`
mode, with no handles and no side effects, and only if the backend accepts, in `EXECUTE`
mode. Nothing is allocated, uploaded or waited for before a backend has said yes.

### Choosing the device

The innermost `cv::accel::ExecutionScope` of the calling thread wins. Otherwise the call
runs on the device of the first argument that lives on one ("compute follows data"). For
each argument the dispatcher plans one of four things: use its memory in place (the device
can access it), upload it into a temporary, create the output on the device, or produce the
output in a temporary and copy it to where the caller keeps it (a Mat, a UMat on another
device, or an output that overlaps an input).

A pre-allocated output that lives elsewhere is never moved: the result is copied into it.

### When the backend declines

The outputs are still created on the device, then the call runs on the CPU through the
protocol, and the OpenCL path is skipped for the rest of the call (it cannot use the memory
and would only allocate outputs in the wrong place). So adding an operation to a backend
changes speed, never where data ends up.

### Implicit transfers

PyTorch and oneDNN reject arguments on mixed devices; the T-API has always fallen back to
the CPU transparently. This design keeps the T-API behaviour by default, because no backend
will ever implement every cv:: function. Three things keep it honest:

- a CPU fallback on shared memory costs no copy at all, only CPU time;
- `TransferStats` counts device operations, fallbacks, transfers and bytes;
- `cv::accel::setTransferPolicy(TRANSFER_FORBID)` (or `OPENCV_ACCEL_TRANSFER_POLICY=forbid`)
  turns every implicit copy into an exception naming the function. A fallback on shared
  memory stays allowed because it copies nothing.

## Streams, fences and asynchronous work

Operations are enqueued on the calling thread's default stream of the device, or on the
stream of an `ExecutionScope`, and return without waiting. OpenCV waits where the CPU needs
the data (`getMat`, downloads, releasing user memory). Fences are reference counted, carry
the id of their stream (a destroyed stream whose handle is reused is never confused with a
live one), and are created on the stream's device.

For the user's own kernels, `cv::accel::DeviceAccess(umat, stream, flags)` makes the device
copy current, makes the stream wait for conflicting OpenCV work, and records the stream's
position when it goes away, so later OpenCV calls wait for the user's work.

## The backend interface

`modules/core/include/opencv2/core/llapi/accel_hal.h`, following the pattern of the
parallel-backend plugins (`llapi.h`): an `OpenCV_API_Header` with ABI and API versions, then
entry points. Plugins export `opencv_core_accel_plugin_init_v0` and are loaded from
`OPENCV_ACCEL_PLUGINS`. Built-in backends register the same table.

A backend must follow seven rules, listed in the header, because core relies on them:

1. Report the true host-access class; never copy behind core's back.
2. A completed fence makes device writes visible to the CPU (system-scope release).
3. Non-coherent allocations are aligned and padded to `coherence_granule`.
4. Cache operations may widen the range and must retry interrupted system calls.
5. Imported dma-bufs: wait for implicit fences, attach your own after writing.
6. Without `CV_ACCEL_CAP_ASYNC`, work runs during the call and fences are already complete.
7. Imported host memory is never freed by the backend; overlapping imports are refused by core.

Exceptions never cross the boundary (`noexcept` entry points, status codes, a per-thread
error message). A plugin built against an older table simply has fewer entries.

## NPUs and DNN

Most NPUs run compiled graphs, not single image operations. The memory layer is shared;
graph offload is a separate client of it. A typical Rockchip or Qualcomm pipeline:

1. a camera frame arrives as a dma-buf and is wrapped into a UMat without a copy;
2. resize and colour conversion run on the 2D engine or the GPU (operation tables),
   writing into a dma-buf the NPU can import;
3. the DNN engine hands the UMat to the NPU backend, which imports the fd
   (`rknn_create_mem_from_fd`, `QnnMem_register`) and runs the graph;
4. post-processing on the CPU calls `getMat`, which waits for the NPU's fence and
   invalidates the CPU caches.

No pixel is copied, and every hand-over is ordered by a fence. The graph-level interface
(a "dnn" table: compile a subgraph, bind UMats, run) is future work; #29807 already moves
the DNN CUDA backend to UMat buffers, which is the precondition.

## Corner cases

| situation | handling | where |
|---|---|---|
| ROI write on a buffer whose host copy is newer | the rest is uploaded before the device writes | `acquireForDevice` |
| output covers the whole allocation | old content is not uploaded | `DEV_DISCARD` |
| fresh allocation | never transferred | both flags obsolete |
| read-only Mat view | no write-back; writing through it is undefined | `getHostView` |
| `OutputArray::getMat` (ACCESS_WRITE) on a partial output | never treated as discard | `beginHostAccessLocked` |
| a read view is upgraded to write | waits for device readers first | `upgradeHostAccessLocked` |
| live writable Mat view during a device call | the call runs on the CPU | `acquireForDevice` |
| last view released while another thread opens one | refcount re-checked under the buffer lock | `endHostView` |
| CPU access while device work is queued | waits for the conflicting fences | `waitDeviceWork` |
| work on two streams of one device | device-side waits | `streamWait` |
| work on two backends | sync_file if both can, else CPU wait | `streamWait` |
| release while device work is queued | freed after all fences | `Backend::deferRelease` |
| user memory (imported, `Mat::getUMat`) | release and calls on it wait for the device | `DeviceAllocator::deallocate` |
| non-coherent memory | invalidate / clean around every CPU access | `beginCpuAccess` |
| two buffers in one cache line | forbidden for non-coherent allocations | backend rule 3 |
| dma-buf implicit fences | bridged by the backend | backend rule 5 |
| managed memory without concurrent access | device gate | `IdleGate`, `SubmitWindow` |
| uncached or write-combined memory | CPU kernels work on a cached copy | `usesShadow` |
| protected memory, tiled layouts | CPU access and CPU fallback throw | `checkHostAccessAllowed` |
| read-only imported memory | device writes refused, CPU write views throw | `acquireForDevice` |
| pageable memory in a DMA | host copies are pinned; user Mats are not pinned implicitly | `ensureShadow` |
| overlapping or misaligned host imports | refused, the caller copies | `Backend::reserveHostImport` |
| two devices of one backend | peer access if the backend allows, else a staged copy | `can_access` |
| two backends | staged copy, counted as a transfer | `UMat::copyTo` |
| legacy and user allocators | describe themselves or keep their old behaviour | `getMemoryDesc` |
| OpenCL path on accelerator data | skipped for the rest of the call | `OpenCLSuppressor` |
| output placement | independent of what the backend implements | `OpCall::run` |
| pre-allocated output on another device | result copied into it, never moved | `PLAN_COPY_BACK` |
| in-place call, overlapping ROIs | temporary output and a device copy | `planArgs` |
| Mat output, device inputs | runs on the device, one download | `PLAN_COPY_BACK` |
| Mat input, device that can address host memory | used in place, call completes before returning | `PLAN_IN_PLACE` |
| out of device memory | deferred frees collected, one retry, then an error | `DeviceAllocator::allocate` |
| allocation on a device fails | no silent fallback to host memory | `UMat::create` |
| asynchronous device error | buffer poisoned, readers get the original error | `waitDeviceWork` |
| device lost | backend marked lost, its calls fall back | `Backend::markLost` |
| a view outside its allocation | rejected before every EXECUTE | `viewFitsAllocation` |
| exception inside a backend | cannot cross the C boundary | `CV_ACCEL_NOEXCEPT` |
| plugin built against an older table | missing entries read as NULL | `CV_ACCEL_OP_PRESENT` |
| stream destroyed and its handle reused | streams carry a unique id | `StreamHolder::uid` |
| CUDA "current device" is user state | backends save and restore it | `DeviceGuard` |
| 64-bit device addresses on a 32-bit host | handles are `uint64_t` | `cvAccelAllocation` |
| synchronous backends (RKNN) | fences are already complete | backend rule 6 |
| process exit | registry and allocators are never destroyed | `Registry::getInstance` |

Known gaps, deliberately left for later: validity is tracked per buffer, not per region
(two tensors in one DNN arena share a fence record, which serializes but stays correct);
`fork()` after a device was used is not handled (backends should refuse in the child);
a UMat released on one thread while a Mat view is released on another is undefined, as it
is today for every allocator.

## How this relates to existing work

- **#29451 (CUDA allocator for UMat, merged).** Its state machine becomes the CUDA
  backend's memory primitives (`backend_cuda.cpp`); `cuda::getCudaAllocator()` can return
  the accelerator allocator once #29807 has settled. Until then both exist, and the old one
  describes itself so that OpenCL declines its buffers.
- **#30259 (explicit transfers, OpenCL guards).** This branch is based on it. Its explicit
  cross-allocator `copyTo` is kept, and its `isCudaBacked()` becomes the descriptor check
  above, which covers HIP and every future backend and keeps user allocators working.
- **#29372 (HIP allocator).** The allocator maps onto the same primitives; the HIP blocks in
  `setTo`, `copyTo` and `convertTo` become entries of the HIP backend's op tables, reached
  through the hook, which also fixes the stale-data and partial-write problems above.
- **#29807 (DNN CUDA on UMat).** Gets streams, fences and deferred release from this layer
  instead of synchronous copies.
- **#29334 (GPU HAL, WIP).** Not used as a base: it puts a C++ class across the plugin
  boundary, stores a second backend pointer next to the allocator, and has no coherence or
  fence model.
- **OpenCL T-API.** Unchanged and still taken for OpenCL buffers. It could later become a
  backend that also consumes foreign memory through `cl_khr_external_memory`.

## What the branch contains

| part | state |
|---|---|
| `llapi/accel_hal.h`: C interface, rules for backends | complete for version 0 |
| `accel.hpp`: Device, Stream, ExecutionScope, describe, policy, stats, external memory, DeviceAccess | complete |
| coherence protocol, generic device allocator, deferred release, device gate, import registry | complete |
| dispatcher (`OpCall`), hook macro, OpenCL suppression | complete |
| hooks | `cv::flip`, `cv::transpose` |
| emulator backend: discrete, coherent, non-coherent, protected, managed | complete, for tests |
| CUDA backend | memory, streams, fences; no kernels yet |
| plugin loading | explicit list (`OPENCV_ACCEL_PLUGINS`) |
| HIP, Level Zero, Vulkan, NPU backends, imgproc table, DLPack | not started |

Build and run the protocol tests:

```
make opencv_test_core
./bin/opencv_test_core --gtest_filter='Core_Accel*'
```

The emulator makes every rule testable without hardware: device memory of its discrete
device is stored scrambled (a missing download shows as garbage), its non-coherent device
keeps a separate CPU view that only the cache operations synchronize (a missing invalidate
or clean shows as stale data), and its managed device counts device work that overlaps CPU
access. `OPENCV_ACCEL_EMULATOR_DELAY_US` delays every queued task to expose missing waits.

## Migration plan

1. This branch: memory layer, protocol, dispatcher, interface, emulator, CUDA memory.
2. CUDA kernels for a first set (flip, transpose, threshold, boxFilter, resize, cvtColor);
   `cuda::getCudaAllocator()` switches to the accelerator allocator; HIP backend from
   #29372.
3. The imgproc op table and more hooks; plugin packaging; dma-buf import and export (V4L2,
   GStreamer); Level Zero and Vulkan backends; DLPack with its stream protocol.
4. DNN graph offload on the same memory layer for NPU backends (RKNPU, QNN, OpenVINO NPU,
   XDNA).

## Open questions

1. Should debug builds default to `TRANSFER_WARN`?
2. Is a consumer list at allocation worth it (VPI declares which backends will use an image,
   and allocates memory they can all share)?
3. Should `UMat::handle()` on accelerator data be deprecated in favour of `DeviceAccess`,
   since it cannot know which stream the caller will use?
4. Python: `cv2.accel` bindings and `__dlpack__` export with the stream argument.
