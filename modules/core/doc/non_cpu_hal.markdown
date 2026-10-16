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
   its output on that device whether the backend implemented the call or not. A UMat argument
   on a device registers itself for the duration of the call (its `_InputArray`), and every
   UMat the call creates without an allocator of its own goes to the device those arguments
   agree on; a UMat that already held device data keeps its device when it is reallocated.
   This covers functions without a hook, `cv::resize` and `cv::repeat` as much as `cv::add`.
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
  PhysicalBacking + views (UMatData)              |
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

### Physical memory, imports and views

The same bytes can reach OpenCV in many ways: two planes of one decoded frame, one dma-buf
imported by a GPU backend and by an NPU backend, a UMat's own memory wrapped again through its
device pointer, a frame wrapped while the previous wrap of it is still being released. All of
them have to share one record of who uses the memory, or two records will disagree about it.
So the state is kept per piece of physical memory (`PhysicalBacking`, `src/accel/backing.cpp`),
not per `UMatData`:

- **Identity.** Core identifies host memory by its addresses and dma-bufs by their inode
  (`fstat`), so imports through different backends meet. Other memory is identified by the
  backend that reaches it (`cvAccelAllocation::backing_id`, the base of the allocation a
  device pointer points into, with its size from `cuMemGetAddressRange`). Backings live in one
  process-wide registry; a wrap that overlaps a backing without lying inside it is refused.
- **Imports.** Every backend, device, layout and access that reaches a backing has one import
  of all of it. A `UMatData` is a view: a range of one import. Views of memory that the CPU
  addresses directly may overlap freely; views with a host copy (`COPY`, `UNCACHED`) never do,
  so that no byte has two host copies (a wrap of bytes such a view covers becomes a UMat over
  that view).
- **Ordering.** Fences, device operations between acquire and record, and `DeviceAccess`
  holds are per backing. Device work on one plane waits for conflicting work on another plane
  of the same buffer, as the implicit fences of a dma-buf do.
- **Failures.** Bytes that failed device work may have written are recorded as ranges of the
  backing: a failure in the chroma plane does not make the luma plane unreadable, and an alias
  of the chroma plane sees the failure.
- **Lifetime.** A backing lives while views refer to it. When the last one goes, memory that
  belongs to someone else is drained right there, without a time limit; core's own memory is
  freed once its fences complete. Until the backing is gone it stays in the registry, and a
  wrap of the same memory revives it, fences included, instead of creating a second state for
  bytes the first one still orders. Lookups wait while a backing is being freed.

### Images in external memory

`wrapExternalImage` returns one UMat per plane, and the planes stay one image. Every plane's
view refers to an image record (`cvAccelImageDesc`) that operations get with each plane they
are passed (`cvAccelView::image`, `plane`): the image id, the DRM fourcc, every plane's offset,
size, pitch and layout modifier, which planes share a buffer, and the colour encoding (YCbCr
matrix, range, primaries, transfer function, chroma siting). A backend can tell from the views
alone that two arguments are the luma and chroma planes of one NV12 frame, and convert it
correctly. `describePlane()` gives the same to the caller.

Before anything is imported, core checks the planes: every size with overflow-safe arithmetic,
no overlap between planes of one buffer, and, for the fourccs it knows (NV12/21, NV16/61,
NV24/42, P010/012/016/210, I420/YV12/YU16/YV16/YU24, YUYV and its orderings, the 8-bit RGB and
grey formats), the plane count, types and subsampled sizes the format requires. Planes may have
layout modifiers of their own. The size of a tiled or compressed plane cannot be computed from
its pitch and rows, so it must be given (`ExternalPlane::size`), and such a plane is only ever
passed whole: its steps say nothing about where bytes lie.

## The coherence protocol

The code is `modules/core/src/accel/buffer_protocol.cpp`.

### State

Which copy is current stays in `UMatData::flags`, as for every allocator:
`HOST_COPY_OBSOLETE` (the CPU view is stale) and `DEVICE_COPY_OBSOLETE` (the device copy is
stale). Both set means the content is uninitialized, and no transfer is ever needed for it.
For memory with a single physical copy, `HOST_COPY_OBSOLETE` means "the CPU has to wait and
invalidate before reading".

The view (`BufferState`, reached through `UMatData::accelState`, NULL for host and OpenCL
memory) holds what belongs to one range: its import and range, its host copy for `COPY`
memory, the current Mat mapping, and the image it is a plane of. Everything that is about the
memory itself lives in its `PhysicalBacking`, under one lock:

- `writeFence`, the last device write and the ranges it covered, and `readFences`, the device
  reads since then (one per stream; a newer fence on an in-order stream covers the older one);
- `pendingReaders` and `pendingWriters`, device operations between acquire and record, and the
  `DeviceAccess` objects that hold the memory;
- `busy`, set while a CPU-side transition waits outside of the lock;
- the CPU access to the memory: how many views have a Mat mapping, and the nested
  `begin_cpu_access` levels open on every import that has a CPU view;
- the poisoned ranges: bytes that failed device work may have written (asynchronously, or after
  queuing part of its work); readers of those bytes get the original error until they are
  written whole.

### CPU access: `UMat::getMat`, `copyTo` into a Mat, `UMat::handle`

1. Wait until no other transition runs and no device operation is between acquire and record.
2. Wait on the CPU for conflicting device work: reading waits for the last write, writing
   also waits for every read (write-after-read). The lock is released during the wait.
3. Per class: for `COPY` memory, download into the host copy only if it is stale and the
   content was ever written; for memory with a CPU view, open the backing for the CPU.
4. When the last Mat view goes away: for `COPY` memory a write marks the device copy stale
   (it is uploaded when the device needs it, not now); for memory with a CPU view, once the
   last view of the whole backing is gone, every level is ended, which cleans CPU caches, so no
   dirty line can later be evicted over data the device wrote.

The CPU opens a backing once for all of its views: `begin_cpu_access` on every import that has
a CPU view, so each backend does the cache maintenance its device needs. Views that come later
join what is open; a view that needs more (a write where only reads are open, or a read view
opened for writing too) nests one more level, which is ended, innermost first and with the
access it began with, when the backing closes. A write is always opened together with a read.
This matters for dma-bufs, whose cache syncs cover the whole buffer: a sync started for the
chroma plane while the luma plane has unflushed writes would drop them.

`ACCESS_READ` is honoured: a read-only view never writes back. `ACCESS_WRITE` is never
taken as "the old content may be discarded", because `OutputArray::getMat()` uses it for
partial writes (masked copies, ROIs).

### Device access: inside `OpCall`

1. **Acquire** each argument for the operation's stream, all views of one backing together. If
   a Mat view of the memory is alive, the device may read when nothing was opened for writing,
   and may never write: a conflicting operation runs on the CPU instead of racing with the
   views. A CPU access whose `end_cpu_access` failed counts as still open.
2. If the host copy is newer and the operation reads the buffer or writes only part of it,
   upload the host copy first, as a device write that waits for everyone still using the
   old device content.
3. Collect the fences the operation must wait for: the last write if it came from another
   stream, and for writes also the reads from other streams. A device-side wait when the
   stream's device takes the fence (its own, or any device of a backend with
   `CV_ACCEL_CAP_CROSS_DEVICE_FENCES`), a sync_file between backends that support it, a CPU
   wait otherwise.
4. **Record** one fence for the operation and store it as the new write fence of the backings
   it wrote and as a read fence of the ones it read.

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
   device runs on the CPU. A handshake on two counters avoids lock-order deadlocks. The
   gate is on from enumeration (`CV_ACCEL_CAP_DEVICE_IDLE_MEMORY`), and a `DeviceAccess`
   counts as a submission for as long as it exists, since the caller's own work may run at
   any time.
6. **Bounds.** Before every EXECUTE call, core checks that each view lies inside its
   allocation and inside the range of it that its UMat views (overflow-safe). Extents of
   wrapped arrays and image planes are computed with checked arithmetic, and backends check
   imports against the allocation they point into (`cuMemGetAddressRange` for CUDA). On SoCs
   where the NPU has no IOMMU, a wrong stride is not an error but memory corruption.
7. **Lifetime.** Released memory is freed only when its fences complete, so no DMA engine
   writes into memory that was reused, and memory that belongs to someone else goes back only
   after the device is done. After a failure the device's progress may be unknown: the work
   left no fence, and `device_synchronize` failed as well. Then nothing the work may touch is
   freed, and borrowed memory waits until a synchronization succeeds or the backend reports the
   device lost, which by backend rule 8 means stopped. Marking a device lost does not stop its
   DMA.
8. **Protected memory.** Data can be uploaded into it and processed on the device, but it
   never reaches the CPU: `getMat`, downloads and CPU fallbacks throw. Whatever is computed
   from it stays protected: outputs and temporaries of the call are allocated protected
   (`CV_ACCEL_MEMF_PROTECTED` in `allocate`), and an ordinary output, a Mat output or a
   device copy into ordinary memory is refused. Protection is attested by the backends that
   import the memory, and all of them must agree: `ExternalMemory::protectedContent` only makes
   a wrap fail when no backend attests it.
9. **Imported dma-bufs.** Buffers from cameras, decoders or displays may carry implicit
   fences, and core treats every dma-buf as if it did. Core brackets every device use of them,
   its own operations and copies as well as the caller's work under a `DeviceAccess`, with
   `begin_device_access` (the stream waits for the other drivers' last write, and a write also
   for their reads) and `end_device_access` (OpenCV's write becomes the exclusive fence,
   OpenCV's read one more shared fence, so a camera writing the next frame waits for both).
   CPU access waits for them as `DMA_BUF_IOCTL_SYNC` does. A backend without these two entries
   cannot import such memory. When a fence cannot be attached, the call that queued the work
   waits for it before returning; otherwise calls on dma-bufs do not wait, since the fences
   order the other users.
10. **Read-only memory.** A write through a CPU view of read-only memory faults. Its host copy
   is core's own pages, mapped read-only except while a download fills them. Memory the CPU
   addresses directly is shown through the backend's read-only mapping when the backend
   attests one (`CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY`, a dma-buf mapped `PROT_READ`), and
   through a copy in protected pages otherwise. When pages cannot be protected, or protected
   again after a download, the CPU access is refused: nothing falls back to writable memory.
   With `OPENCV_ACCEL_VALIDATE`, every view opened with `ACCESS_READ`, of any memory, is mapped
   read-only the same way, so a write through one faults at once instead of being found later.
11. **CPU access that would not end.** A failed `end_cpu_access` leaves the CPU access open,
   after a read as after a write: no device work may use the memory, and device-idle memory
   keeps its device gate, until a retry of the end succeeds. Core retries before the next use.

### Discrete GPUs without extra copies

- The host copy of device memory is pinned, from a size-keyed cache
  (`OPENCV_ACCEL_PINNED_CACHE_MB`), so every transfer is one DMA and pinning never happens
  per buffer (pinning is slow and stalls the device's streams). Only pinned memory that the
  CPU reaches through its caches is used: where the backend reports it uncached (Jetson
  before Xavier) or write-combined, host copies stay in pageable memory.
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
mode. Nothing is allocated, uploaded or waited for before a backend has said yes. Outputs
and temporaries that do not exist yet are described by the backend itself
(`describe_allocation`), so the answer covers the memory the call will really get.

### Choosing the device

The innermost `cv::accel::ExecutionScope` of the calling thread wins. Otherwise every device
that holds an argument is a candidate: the dispatcher plans the call for each, asks its
backend (QUERY, free of side effects), and picks the device that accepts and copies the fewest
bytes, the earlier argument's device on a tie ("compute follows data", weighed by coverage and
transfer cost). For
each argument the dispatcher plans one of four things: use its memory in place (the device
can access it), upload it into a temporary, create the output on the device, or produce the
output in a temporary and copy it to where the caller keeps it (a Mat, a UMat on another
device, or an output that overlaps an input).

Bytes moved count everything the plan implies: temporaries filled and copied back, and a host
copy that must be uploaded before the device can use an argument in place (or downloaded before
it is copied elsewhere).

A pre-allocated output that lives elsewhere is never moved: the result is copied into it.

### When the backend declines

The outputs are still created on the device, then the call runs on the CPU through the
protocol, and the OpenCL path is skipped for the rest of the call (it cannot use the memory
and would only allocate outputs in the wrong place). So adding an operation to a backend
changes speed, never where data ends up. The same holds inside a device `ExecutionScope` when
no argument is on a device yet: the scope decides, the CPU runs the call, it counts as a
fallback, and the outputs go to the scope's device.

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
position when it goes away, so later OpenCV calls wait for the user's work. Its `handle()` is
the address kernels use, which for mapped host memory need not be the CPU address
(`cvAccelAllocation::device_address`), and `offset()` says where the UMat starts from there.

A stream shared by threads (one explicit `Stream` in several `ExecutionScope`s) is used by one
thread at a time on devices whose streams are not thread-safe.

## The backend interface

`modules/core/include/opencv2/core/llapi/accel_hal.h`, following the pattern of the
parallel-backend plugins (`llapi.h`): an `OpenCV_API_Header` with ABI and API versions, then
entry points. Plugins export `opencv_core_accel_plugin_init_v0` and are loaded from
`OPENCV_ACCEL_PLUGINS`. Built-in backends register the same table.

A backend must follow eleven rules, listed in the header, because core relies on them:

1. Report the true host-access class; never copy behind core's back.
2. A completed fence makes device writes visible to the CPU (system-scope release).
3. Non-coherent allocations are aligned and padded to `coherence_granule`.
4. Cache operations may widen the range and must retry interrupted system calls. CPU access
   nests: every successful `begin_cpu_access` gets its own `end_cpu_access`, innermost first,
   and a failed end counts as still open.
5. Memory with implicit fences (every dma-buf) needs `begin_device_access` and
   `end_device_access`; core refuses imports through backends without them.
6. Without `CV_ACCEL_CAP_ASYNC`, work runs during the call and fences are already complete.
7. Imports are borrowed: the backend never frees the memory, and never closes or keeps the
   handle it was given (it duplicates or references what it needs to keep, and gives that up in
   `deallocate`). Overlapping host imports are refused by core.
8. `CV_ACCEL_DEVICE_LOST` means the device has stopped: nothing of it runs or touches memory.
   `device_synchronize` returns OK once earlier work has finished, failed or not.
9. `NOT_IMPLEMENTED` and `OUT_OF_MEMORY` queue nothing; after other errors core assumes part of
   the work was queued.
10. Read-only memory is mapped read-only for the CPU where the platform allows it, and the
   backend says so (`CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY`).
11. Streams of a device without `CV_ACCEL_CAP_STREAMS_THREAD_SAFE` are used by one thread at a
   time, and `stream_wait_fence` gets fences of other devices only with
   `CV_ACCEL_CAP_CROSS_DEVICE_FENCES`.

Exceptions never cross the boundary (`noexcept` entry points, status codes, a per-thread
error message). A plugin built against an older table simply has fewer entries, and structs
with a `struct_size` too short for their version-0 fields are refused before anything is
written.

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
| ROI write on a buffer whose host copy is newer | the rest is uploaded before the device writes | `DeviceAcquisition::acquire` |
| output covers the whole view | old content is not uploaded | `DEV_DISCARD` |
| fresh allocation | never transferred | both flags obsolete |
| read-only Mat view | no write-back; under `OPENCV_ACCEL_VALIDATE` a write through it faults | `getHostView` |
| `OutputArray::getMat` (ACCESS_WRITE) on a partial output | never treated as discard | `beginHostAccessLocked` |
| a read view is upgraded to write | waits for device readers first; a nested CPU access level, ended on its own | `upgradeHostAccessLocked` |
| live writable Mat view during a device call | the call runs on the CPU | `DeviceAcquisition::acquire` |
| last view released while another thread opens one | refcount re-checked under the backing lock | `endHostView` |
| CPU access while device work is queued | waits for the conflicting fences | `waitDeviceWork` |
| work on two streams of one device | device-side waits | `streamWait` |
| fence of another device of the same backend | device-side wait only with `CV_ACCEL_CAP_CROSS_DEVICE_FENCES`, a CPU wait otherwise | `streamWait` |
| work on two backends | sync_file if both can, else CPU wait | `streamWait` |
| one explicit stream used by several threads | one thread at a time on devices whose streams are not thread-safe | `StreamCall` |
| release while device work is queued | freed after all fences; the backing stays findable until then | `releaseBacking`, `collectRetired` |
| a wrap of memory that is being released | revives its backing, fences included; waits while it is being freed | `findBacking` |
| user memory (imported, `Mat::getUMat`) | release and calls on it wait for the device; CPU writes in a host copy reach it first | `DeviceAllocator::deallocate` |
| failed work that left no fence, and a failed device sync | nothing it may touch is freed; borrowed memory waits until the device syncs or stops | `FenceRef::unknown` |
| `fence_retain` fails | the copy tracks the work by device synchronization, never holds an unreferenced fence | `FenceRef` |
| non-coherent memory | invalidate / clean around every CPU access, for all views of the memory at once | `openCpuLevel` |
| `end_cpu_access` fails | the CPU access stays open: devices stay away (and device-idle memory keeps its gate) until a retry succeeds | `retryFailedCpuEndsLocked` |
| two buffers in one cache line | forbidden for non-coherent allocations | backend rule 3 |
| dma-buf implicit fences | write and read fences both ways; backends without the entries are refused | backend rule 5 |
| dma-buf fence that cannot be attached | the call waits for the work before returning | `endExternal` |
| managed memory without concurrent access | device gate | `IdleGate`, `SubmitWindow` |
| uncached or write-combined memory | CPU kernels work on a cached copy | `usesShadow` |
| protected memory, tiled layouts | CPU access and CPU fallback throw | `checkHostAccessAllowed` |
| read-only imported memory | device writes refused; CPU views fault on writes, or are refused when pages cannot be protected | `beginHostAccessLocked`, `ensureShadow` |
| pageable memory in a DMA | host copies are pinned; user Mats are not pinned implicitly | `ensureShadow` |
| host memory sharing pages with an import | refused, the caller copies (pinning works on whole pages) | `Backend::reserveHostImport` |
| multi-plane images (NV12, P010), one fd or one per plane | one UMat per plane; operations see the image, its format, layouts and colours; planes are checked against the format | `wrapExternalImage`, `cvAccelView::image` |
| planes of one dma-buf, or one dma-buf imported by several backends | one backing: one ordering, one CPU access, one record of failures | `PhysicalBacking` |
| tiled or compressed layouts (DRM modifiers), per plane | no CPU view; size given by the caller; passed whole; refused by backends that do not report the layout back | `ExternalPlane::layoutModifier` |
| the same memory wrapped twice, or a UMat's own memory wrapped | the same backing; a UMat over the view that holds a host copy of those bytes; a range that only partly overlaps a backing is refused | `findBacking`, `cvAccelAllocation::backing_id` |
| a range past the end of the allocation it starts in, or one that overflows | refused | `importChecked`, backend `import_memory` |
| misaligned host imports (device alignment; cache-line granule for non-coherent memory) | refused, the caller copies | `importChecked` |
| two devices of one backend | peer access where it is enabled (CUDA: `OPENCV_ACCEL_CUDA_PEER_ACCESS`, at enumeration), else a staged copy | `can_access` |
| two backends | staged copy, counted as a transfer | `UMat::copyTo` |
| a copy between overlapping views of one memory | through a temporary (none when the bytes are the same) | `DeviceAllocator::copy` |
| legacy and user allocators | describe themselves or keep their old behaviour | `getMemoryDesc` |
| OpenCL path on accelerator data | skipped for the rest of the call | `OpenCLSuppressor` |
| output placement, with or without a hook | the device of the call's device arguments; a reallocation keeps its device | `OpCall::run`, `placementFor` |
| a device scope whose device declines a call with host arguments | the CPU runs it, counted, outputs on the scope's device | `OpCall::run` |
| pre-allocated output on another device | result copied into it, never moved | `PLAN_COPY_BACK` |
| in-place call, overlapping ROIs or aliases | temporary output and a device copy | `planArgs` |
| Mat output, device inputs | runs on the device, one download | `PLAN_COPY_BACK` |
| Mat input, device that can address host memory | used in place, call completes before returning | `PLAN_IN_PLACE` |
| out of device memory | retired memory collected, one retry, then an error | `DeviceAllocator::allocateMemory` |
| allocation on a device fails | no silent fallback to host memory | `UMat::create` |
| asynchronous device error | the bytes the work wrote are poisoned, readers get the original error | `waitDeviceWork` |
| call fails after queuing part of its work | its buffers wait for the queued part (a fence, or a device sync); what it writes is poisoned until written whole | `DeviceAcquisition::fail` |
| device lost | that device marked lost (other devices keep working), its calls fall back, its allocations are leaked rather than freed | `Backend::markStopped` |
| device error while a UMat is destroyed | logged, never thrown; host-side state is still freed | `releaseView` |
| device that never finishes | optional limit on CPU waits (`OPENCV_ACCEL_WAIT_TIMEOUT_MS`), then the device is lost and what it may touch is kept; waits that guard memory going back to its owner have no limit | `FenceRef::wait` |
| a view outside its allocation or its UMat's range | rejected before every EXECUTE | `viewFitsAllocation`, `viewFitsRange` |
| exception inside a backend | cannot cross the C boundary | `CV_ACCEL_NOEXCEPT` |
| plugin built against an older table | missing entries read as NULL | `CV_ACCEL_OP_PRESENT` |
| `struct_size` too short for version-0 fields | refused, nothing written | `CV_ACCEL_HAS_FIELD` |
| stream destroyed and its handle reused | streams carry a unique id | `StreamHolder::uid` |
| CUDA "current device" is user state | primary contexts pushed and popped; a call fails when the GPU cannot be made current | `DeviceGuard` |
| mapped host memory whose device address differs from its CPU address | `DeviceAccess::handle()` and `UMat::handle()` give the device address | `cvAccelAllocation::device_address` |
| 64-bit device addresses on a 32-bit host | handles are `uint64_t` | `cvAccelAllocation` |
| synchronous backends (RKNN) | fences are already complete | backend rule 6 |
| process exit | registries and allocators are never destroyed | `Registry::getInstance` |

Known gaps, deliberately left for later: ordering is per backing, not per range, so device
work on one plane of a frame waits for conflicting work on another plane, and two tensors in
one DNN arena are serialized (correct, not always fast); host copies are per view, so one
memory imported with host copies through two backends is refused, and so is one memory that
two backends reach with different host-access classes; device-idle memory cannot be imported
by a second device; `fork()` after a device was used is not handled (backends should refuse in
the child); a UMat released on one thread while a Mat view is released on another is
undefined, as it is today for every allocator.

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
| `llapi/accel_hal.h`: C interface (API version 3), rules for backends | complete for version 0 |
| `accel.hpp`: Device, Stream, ExecutionScope, describe, policy, stats, external memory and images, DeviceAccess | complete |
| physical memory (backings, imports, views, registry, retirement), coherence protocol, generic device allocator, device gate | complete |
| dispatcher (`OpCall`), hook macro, OpenCL suppression, placement of new UMats by call | complete |
| hooks | `cv::flip`, `cv::transpose` |
| emulator backend: seven devices (discrete, coherent, non-coherent, protected, managed, failing, peer), dma-bufs with read and write fences, failure injection, protocol checks | complete, for tests |
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
or clean shows as stale data), its managed device counts device work that overlaps CPU
access, a sixth device fails on request at chosen points (fences, calls, fence recording,
device synchronization, `fence_retain`, device loss), and a seventh is a peer of the first
whose fences do not cross devices. Its dma-bufs are memfds with an exclusive write fence and
shared read fences, and read-only imports get a `PROT_READ` mapping. It counts protocol
violations: unbalanced or mismatched CPU access levels, a non-thread-safe stream used by two
threads, a fence of another device waited for on the device. `OPENCV_ACCEL_EMULATOR_DELAY_US`
delays every queued task to expose missing waits.

## Evidence so far

Measured on an Intel Core i9-14900KS with an NVIDIA GeForce RTX 5090 (CUDA 12.0, driver 570),
GCC 13.3, Release builds:

- `opencv_test_core --gtest_filter='Core_Accel*'`: the 73 emulator tests pass, also repeated,
  in separate processes and with every device task delayed (`OPENCV_ACCEL_EMULATOR_DELAY_US`).
  They cover the protocol, failures at every point (including work that cannot be proven
  stopped and refused fence references), device loss, dma-buf read and write fences, a
  dma-buf shared by two devices, planes of one buffer, images and their formats, aliases and
  memory revived during its release, read-only and protected memory failing closed, failed
  CPU access ends, nested CPU access levels, placement without hooks, device choice, shared
  streams, cross-device fences, and short `struct_size`s. The stream and fence tests were
  checked to fail with their protection removed.
- The eight `Core_Accel_CUDA` tests pass on the RTX 5090: transfer accounting of a host round
  trip, device-to-device copies, placement after CPU fallbacks, backing identity and range
  checks through `cuMemGetAddressRange`, the device address of mapped pinned memory used by a
  kernel, ordering against a kernel the caller launches on the legacy default stream, and
  copy-back of Mat-backed UMats. The whole `opencv_test_core` passes with CUDA (25,336 tests)
  and without (24,865), and the whole tree builds.
- ThreadSanitizer (`-fsanitize=thread`, pthreads backend): no report from the accelerator code
  over repeated runs and with delayed device tasks. The one report left is in the pthreads
  `ThreadPool` (`parallel_impl.cpp`), which a plain `cv::reduce` test gives as well.
- Cost, 4x4 flip: 0.14 us on a Mat, 0.7 us on a host UMat with a backend registered, 2.8 to
  3.1 us dispatched to the emulator (8.6 to 8.8 us including the wait for its worker thread).
- CUDA, 4K BGR frame (24.9 MB): upload 14 to 15 GB/s (GpuMat::upload 16); `getMat` into the
  pinned host copy 23.5 to 25 GB/s, against 7.1 GB/s into pageable memory (GpuMat::download
  7.1); device copy 740 to 800 GB/s. A chain flip, flip, add, `getMat` on GPU data without CUDA
  kernels moves the frame across the bus once.

Not covered yet: NPU hardware (the emulator's non-coherent and device-idle devices stand in),
HIP, multiple GPUs (peer access, contexts on other devices), and performance of real kernels.

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
