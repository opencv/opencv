GPU Module Introduction
=======================

.. highlight:: cpp

General Information
-------------------

The OpenCV GPU module is a set of classes and functions to utilize GPU computational capabilities. It is implemented using NVIDIA* CUDA* Runtime API and supports only NVIDIA GPUs. The OpenCV GPU module includes utility functions, low-level vision primitives, and high-level algorithms. The utility functions and low-level primitives provide a powerful infrastructure for developing fast vision algorithms taking advantage of GPU whereas the high-level functionality includes some state-of-the-art algorithms (such as stereo correspondence, face and people detectors, and others) ready to be used by the application developers.

The GPU module is designed as a host-level API. This means that if you have pre-compiled OpenCV GPU binaries, you are not required to have the CUDA Toolkit installed or write any extra code to make use of the GPU.

The GPU module depends on the CUDA Toolkit and NVIDIA Performance Primitives library (NPP). Make sure you have the latest versions of this software installed. You can download two libraries for all supported platforms from the NVIDIA site. To compile the OpenCV GPU module, you need a compiler compatible with the CUDA Runtime Toolkit.

The OpenCV GPU module is designed for ease of use and does not require any knowledge of CUDA. Though, such a knowledge will certainly be useful to handle non-trivial cases or achieve the highest performance. It is helpful to understand the cost of various operations, what the GPU does, what the preferred data formats are, and so on. The GPU module is an effective instrument for quick implementation of GPU-accelerated computer vision algorithms. However, if your algorithm involves many simple operations, then, for the best possible performance, you may still need to write your own kernels to avoid extra write and read operations on the intermediate results.

To enable CUDA support, configure OpenCV using ``CMake`` with ``WITH_CUDA=ON`` . When the flag is set and if CUDA is installed, the full-featured OpenCV GPU module is built. Otherwise, the module is still built but at runtime all functions from the module throw
:ocv:func:`Exception` with ``CV_GpuNotSupported`` error code, except for
:ocv:func:`gpu::getCudaEnabledDeviceCount()`. The latter function returns zero GPU count in this case. Building OpenCV without CUDA support does not perform device code compilation, so it does not require the CUDA Toolkit installed. Therefore, using the
:ocv:func:`gpu::getCudaEnabledDeviceCount()` function, you can implement a high-level algorithm that will detect GPU presence at runtime and choose an appropriate implementation (CPU or GPU) accordingly.

Compilation for Different NVIDIA* Platforms
-------------------------------------------

NVIDIA* compiler enables generating binary code (cubin and fatbin) and intermediate code (PTX). Binary code often implies a specific GPU architecture and generation, so the compatibility with other GPUs is not guaranteed. PTX is targeted for a virtual platform that is defined entirely by the set of capabilities or features. Depending on the selected virtual platform, some of the instructions are emulated or disabled, even if the real hardware supports all the features.

At the first call, the PTX code is compiled to binary code for the particular GPU using a JIT compiler. When the target GPU has a compute capability (CC) lower than the PTX code, JIT fails.
By default, the OpenCV GPU module includes:

*
    Binaries for compute capabilities 1.3 and 2.0 (controlled by ``CUDA_ARCH_BIN``     in ``CMake``)

*
    PTX code for compute capabilities 1.1 and 1.3 (controlled by ``CUDA_ARCH_PTX``     in ``CMake``)

This means that for devices with CC 1.3 and 2.0 binary images are ready to run. For all newer platforms, the PTX code for 1.3 is JIT'ed to a binary image. For devices with CC 1.1 and 1.2, the PTX for 1.1 is JIT'ed. For devices with CC 1.0, no code is available and the functions throw
:ocv:func:`Exception`. For platforms where JIT compilation is performed first, the run is slow.

On a GPU with CC 1.0, you can still compile the GPU module and most of the functions will run flawlessly. To achieve this, add "1.0" to the list of binaries, for example, ``CUDA_ARCH_BIN="1.0 1.3 2.0"`` . The functions that cannot be run on CC 1.0 GPUs throw an exception.

You can always determine at runtime whether the OpenCV GPU-built binaries (or PTX code) are compatible with your GPU. The function
:ocv:func:`gpu::DeviceInfo::isCompatible` returns the compatibility status (true/false).

Threading and Multi-threading
------------------------------

The OpenCV GPU module follows the CUDA Runtime API conventions regarding the multi-threaded programming. This means that for the first API call a CUDA context is created implicitly, attached to the current CPU thread and then is used as the "current" context of the thread. All further operations, such as a memory allocation, GPU code compilation, are associated with the context and the thread. Since any other thread is not attached to the context, memory (and other resources) allocated in the first thread cannot be accessed by another thread. Instead, for this other thread CUDA creates another context associated with it. In short, by default, different threads do not share resources. But you can remove this limitation by using the CUDA Driver API (version 3.1 or later). You can retrieve context reference for one thread, attach it to another thread, and make it "current" for that thread. As a result, the threads can share memory and other resources. It is also possible to create a context explicitly before calling any GPU code and attach it to all the threads you want to share the resources with.

It is also possible to create the context explicitly using the CUDA Driver API, attach, and set the "current" context for all necessary threads. The CUDA Runtime API (and OpenCV functions, respectively) picks it up.

Utilizing Multiple GPUs
-----------------------

In the current version, each of the OpenCV GPU algorithms can use only a single GPU. So, to utilize multiple GPUs, you have to manually distribute the work between GPUs. Consider the following ways of utilizing multiple GPUs:

*
    If you use only synchronous functions, create several CPU threads (one per each GPU). From within each thread, create a CUDA context for the corresponding GPU using
    :ocv:func:`gpu::setDevice()`     or Driver API. Each of the threads will use the associated GPU.

*
    If you use asynchronous functions, you can use the Driver API to create several CUDA contexts associated with different GPUs but attached to one CPU thread. Within the thread you can switch from one GPU to another by making the corresponding context "current". With non-blocking GPU calls, managing algorithm is clear.

While developing algorithms for multiple GPUs, note a data passing overhead. For primitive functions and small images, it can be significant, which may eliminate all the advantages of having multiple GPUs. But for high-level algorithms, consider using multi-GPU acceleration. For example, the Stereo Block Matching algorithm has been successfully parallelized using the following algorithm:


 1.   Split each image of the stereo pair into two horizontal overlapping stripes.


 2.   Process each pair of stripes (from the left and right images) on a separate Fermi* GPU.


 3.   Merge the results into a single disparity map.

With this algorithm, a dual GPU gave a 180
%
performance increase comparing to the single Fermi GPU. For a source code example, see
https://code.ros.org/svn/opencv/trunk/opencv/examples/gpu/.

