// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// CUDA-backed MatAllocator for UMat (Phase P1): synchronous host<->device transfers.

#include "precomp.hpp"

namespace cv { namespace cuda {

#ifndef HAVE_CUDA

MatAllocator* getCudaAllocator()
{
    return NULL;
}

#else

class CudaUMatAllocator CV_FINAL : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type, void* data0,
                       size_t* step, AccessFlag /*flags*/, UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        const size_t elemSize = CV_ELEM_SIZE(type);
        size_t total = elemSize;
        for (int i = 0; i < dims; i++)
            total *= (size_t)sizes[i];

        if (step)
        {
            step[dims - 1] = elemSize;
            for (int i = dims - 2; i >= 0; i--)
                step[i] = step[i + 1] * (size_t)sizes[i + 1];
        }

        UMatData* u = new UMatData(this);
        u->size = total;

        if (total > 0)
            cudaSafeCall(cudaMalloc(&u->handle, total));

        if (data0)
        {
            u->data = u->origdata = static_cast<uchar*>(data0);
            u->flags |= UMatData::USER_ALLOCATED;
            u->markDeviceCopyObsolete(true);
        }
        else
        {
            u->markHostCopyObsolete(true);
        }
        return u;
    }

    bool allocate(UMatData* u, AccessFlag /*accessFlags*/, UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        if (!u)
            return false;
        if (!u->handle && u->size > 0)
        {
            cudaSafeCall(cudaMalloc(&u->handle, u->size));
            if (u->data)
                u->markDeviceCopyObsolete(true);
            else
                u->markHostCopyObsolete(true);
        }
        return true;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (u->handle)
            cudaSafeCall(cudaFree(u->handle));
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED))
            fastFree(u->data);
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (!u->data && u->size > 0)
            u->data = u->origdata = static_cast<uchar*>(fastMalloc(u->size));
        if (u->hostCopyObsolete() && u->handle && u->data)
        {
            cudaSafeCall(cudaMemcpy(u->data, u->handle, u->size, cudaMemcpyDeviceToHost));
            u->markHostCopyObsolete(false);
        }
        if (!!(accessFlags & ACCESS_WRITE))
            u->markDeviceCopyObsolete(true);
    }

    void unmap(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (u->deviceCopyObsolete() && u->handle && u->data)
        {
            cudaSafeCall(cudaMemcpy(u->handle, u->data, u->size, cudaMemcpyHostToDevice));
            u->markDeviceCopyObsolete(false);
        }
    }

    void download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE
    {
        if (!u || !u->handle || !dstptr)
            return;
        copyPlanes((uchar*)u->handle, srcofs, srcstep, (uchar*)dstptr, 0, dststep,
                   dims, sz, cudaMemcpyDeviceToHost);
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u || !srcptr)
            return;
        if (!u->handle && u->size > 0)
            cudaSafeCall(cudaMalloc(&u->handle, u->size));
        copyPlanes((uchar*)srcptr, NULL, srcstep, (uchar*)u->handle, dstofs, dststep,
                   dims, sz, cudaMemcpyHostToDevice);
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
    }

    // Same-world D2D only; UMat::copyTo calls copy() solely when both share this allocator.
    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool /*sync*/) const CV_OVERRIDE
    {
        if (!src || !dst || !src->handle || !dst->handle)
            return;
        copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                   dims, sz, cudaMemcpyDeviceToDevice);
        dst->markHostCopyObsolete(true);
        dst->markDeviceCopyObsolete(false);
    }

private:
    // sz/steps follow the MatAllocator convention: the last dim carries element bytes.
    static void copyPlanes(uchar* srcbase, const size_t srcofs[], const size_t srcstep[],
                           uchar* dstbase, const size_t dstofs[], const size_t dststep[],
                           int dims, const size_t sz[], cudaMemcpyKind kind)
    {
        int isz[CV_MAX_DIM];
        uchar* srcptr = srcbase;
        uchar* dstptr = dstbase;
        for (int i = 0; i < dims; i++)
        {
            CV_Assert(sz[i] <= (size_t)INT_MAX);
            if (sz[i] == 0)
                return;
            if (srcofs)
                srcptr += srcofs[i] * (i <= dims - 2 ? srcstep[i] : 1);
            if (dstofs)
                dstptr += dstofs[i] * (i <= dims - 2 ? dststep[i] : 1);
            isz[i] = (int)sz[i];
        }

        Mat src(dims, isz, CV_8U, srcptr, srcstep);
        Mat dst(dims, isz, CV_8U, dstptr, dststep);

        const Mat* arrays[] = { &src, &dst };
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs, 2);
        const size_t planesz = it.size;
        for (size_t j = 0; j < it.nplanes; j++, ++it)
            cudaSafeCall(cudaMemcpy(ptrs[1], ptrs[0], planesz, kind));
    }
};

MatAllocator* getCudaAllocator()
{
    CV_SINGLETON_LAZY_INIT(CudaUMatAllocator, new CudaUMatAllocator())
}

#endif

}} // namespace cv { namespace cuda {
