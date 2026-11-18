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
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--)
        {
            if (step)
            {
                if (data0 && step[i] != CV_AUTOSTEP)
                {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                }
                else
                {
                    step[i] = total;
                }
            }
            total *= (size_t)sizes[i];
        }

        void* handle = NULL;
        if (total > 0)
            cudaSafeCall(cudaMalloc(&handle, total));

        UMatData* u = new UMatData(this);
        u->handle = handle;
        u->size = total;

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
        CV_Assert(!u->handle || u->currAllocator == this);
        if (!u->handle)
        {
            void* handle = NULL;
            if (u->size > 0)
                cudaSafeCall(cudaMalloc(&handle, u->size));
            u->handle = handle;
            if (u->currAllocator != this)
            {
                u->prevAllocator = u->currAllocator;
                u->currAllocator = this;
            }
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
        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        if (u->handle)
        {
            cudaSafeCall(cudaFree(u->handle));
            u->handle = NULL;
        }
        if (u->prevAllocator && u->prevAllocator != this)
        {
            u->markDeviceCopyObsolete(true);
            if (u->data && u->data != u->origdata && !(u->flags & UMatData::USER_ALLOCATED))
                fastFree(u->data);
            u->data = u->origdata;
            const MatAllocator* prev = u->prevAllocator;
            u->currAllocator = prev;
            u->prevAllocator = NULL;
            prev->deallocate(u);
            return;
        }
        if (u->origdata && !(u->flags & UMatData::USER_ALLOCATED))
        {
            fastFree(u->origdata);
            u->origdata = NULL;
            u->data = NULL;
        }
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u)
            return;
        if (!u->data && u->size > 0)
        {
            uchar* p = static_cast<uchar*>(fastMalloc(u->size));
            u->data = p;
            if (!u->prevAllocator || u->prevAllocator == this)
                u->origdata = p;
        }
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
        if (!u || !dstptr)
            return;
        if (u->deviceCopyObsolete() && u->data)
        {
            copyPlanes(u->data, srcofs, srcstep, (uchar*)dstptr, NULL, dststep,
                       dims, sz, cudaMemcpyHostToHost, true);
            return;
        }
        if (!u->handle)
            return;
        copyPlanes((uchar*)u->handle, srcofs, srcstep, (uchar*)dstptr, NULL, dststep,
                   dims, sz, cudaMemcpyDeviceToHost, false);
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u || !srcptr)
            return;
        CV_Assert(!u->handle || u->currAllocator == this);
        if (!u->handle && u->size > 0)
        {
            void* handle = NULL;
            cudaSafeCall(cudaMalloc(&handle, u->size));
            u->handle = handle;
            if (u->currAllocator != this)
            {
                u->prevAllocator = u->currAllocator;
                u->currAllocator = this;
            }
        }
        if (u->deviceCopyObsolete() && u->data && !isFullOverwrite(u, dims, sz, dstofs, dststep))
            unmap(u);
        copyPlanes((uchar*)srcptr, NULL, srcstep, (uchar*)u->handle, dstofs, dststep,
                   dims, sz, cudaMemcpyHostToDevice, false);
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
        if (src->deviceCopyObsolete() && src->data)
            unmap(src);
        if (dst->deviceCopyObsolete() && dst->data && !isFullOverwrite(dst, dims, sz, dstofs, dststep))
            unmap(dst);
        copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                   dims, sz, cudaMemcpyDeviceToDevice, false);
        dst->markHostCopyObsolete(true);
        dst->markDeviceCopyObsolete(false);
    }

private:
    static bool isFullOverwrite(const UMatData* u, int dims, const size_t sz[],
                                const size_t dstofs[], const size_t dststep[])
    {
        size_t expectedStep = 1;
        for (int i = dims - 1; i >= 0; i--)
        {
            if (sz[i] == 0)
                return false;
            if (dstofs && dstofs[i] != 0)
                return false;
            if (i <= dims - 2 && dststep && dststep[i] != expectedStep)
                return false;
            expectedStep *= sz[i];
        }
        return expectedStep == u->size;
    }

    // sz/steps follow the MatAllocator convention: the last dim carries element bytes.
    static void copyPlanes(uchar* srcbase, const size_t srcofs[], const size_t srcstep[],
                           uchar* dstbase, const size_t dstofs[], const size_t dststep[],
                           int dims, const size_t sz[], cudaMemcpyKind kind, bool hostToHost)
    {
        int isz[CV_MAX_DIM];
        uchar* srcptr = srcbase;
        uchar* dstptr = dstbase;
        for (int i = 0; i < dims; i++)
        {
            CV_Assert(sz[i] <= (size_t)INT_MAX);
            if (sz[i] == 0)
                return;
            if (srcofs && srcofs[i] != 0)
            {
                CV_Assert(i > dims - 2 || srcstep != NULL);
                srcptr += srcofs[i] * (i <= dims - 2 ? srcstep[i] : 1);
            }
            if (dstofs && dstofs[i] != 0)
            {
                CV_Assert(i > dims - 2 || dststep != NULL);
                dstptr += dstofs[i] * (i <= dims - 2 ? dststep[i] : 1);
            }
            isz[i] = (int)sz[i];
        }

        if (dims == 2)
        {
            const size_t height = sz[0];
            const size_t widthBytes = sz[1];
            const size_t spitch = srcstep ? srcstep[0] : widthBytes;
            const size_t dpitch = dststep ? dststep[0] : widthBytes;
            if (spitch >= widthBytes && dpitch >= widthBytes)
            {
                if (hostToHost)
                {
                    if (spitch == widthBytes && dpitch == widthBytes)
                        std::memcpy(dstptr, srcptr, height * widthBytes);
                    else
                        for (size_t r = 0; r < height; r++)
                            std::memcpy(dstptr + r * dpitch, srcptr + r * spitch, widthBytes);
                }
                else
                {
                    cudaSafeCall(cudaMemcpy2D(dstptr, dpitch, srcptr, spitch, widthBytes, height, kind));
                }
                return;
            }
        }

        Mat src(dims, isz, CV_8U, srcptr, srcstep);
        Mat dst(dims, isz, CV_8U, dstptr, dststep);

        const Mat* arrays[] = { &src, &dst };
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs, 2);
        const size_t planesz = it.size;
        for (size_t j = 0; j < it.nplanes; j++, ++it)
        {
            if (hostToHost)
                std::memcpy(ptrs[1], ptrs[0], planesz);
            else
                cudaSafeCall(cudaMemcpy(ptrs[1], ptrs[0], planesz, kind));
        }
    }
};

MatAllocator* getCudaAllocator()
{
    CV_SINGLETON_LAZY_INIT(CudaUMatAllocator, new CudaUMatAllocator())
}

#endif

}} // namespace cv { namespace cuda {
