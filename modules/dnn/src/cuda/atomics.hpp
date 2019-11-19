// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_ATOMICS_HPP
#define OPENCV_DNN_SRC_CUDA_ATOMICS_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#else
inline __device__ void atomicAdd(__half* address, __half val) {
    unsigned int* address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;

        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        __half tmpres = hsum + val;
        hsum = __half_raw(tmpres);

        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif

#endif /* OPENCV_DNN_SRC_CUDA_ATOMICS_HPP */
