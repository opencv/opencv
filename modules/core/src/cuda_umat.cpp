// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// CUDA-backed MatAllocator for UMat (Phase P1): synchronous host<->device transfers.

#include "precomp.hpp"

#ifdef HAVE_CUDA
#include "umatrix.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <limits>
#endif

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
        CV_Assert(dims >= 0);
        CV_Assert(dims == 0 || sizes != NULL);
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--)
        {
            CV_Assert(sizes[i] >= 0);
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
            const size_t dimSize = (size_t)sizes[i];
            if (dimSize > 0 && total > std::numeric_limits<size_t>::max() / dimSize)
                CV_Error(Error::StsOutOfRange, "Total allocation size overflows size_t");
            total *= dimSize;
        }

        void* handle = NULL;
        if (total > 0)
            cudaSafeCall(cudaMalloc(&handle, total));

        UMatData* u = NULL;
        try
        {
            u = new UMatData(this);
        }
        catch (...)
        {
            if (handle)
                (void)cudaFree(handle);
            throw;
        }
        u->handle = handle;
        u->size = total;
        u->flags |= UMatData::COPY_ON_MAP;

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
        UMatDataAutoLock lock(u);
        if (u->currAllocator != this)
            return false;
        u->flags |= UMatData::COPY_ON_MAP;
        if (!u->handle && u->size > 0)
        {
            void* handle = NULL;
            cudaSafeCall(cudaMalloc(&handle, u->size));
            u->handle = handle;
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
        CV_Assert(u->currAllocator == this);
        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        if (u->handle)
        {
            cudaError_t err = cudaFree(u->handle);
            if (err != cudaSuccess)
            {
                (void)cudaGetLastError();
                CV_LOG_WARNING(NULL, "CudaUMatAllocator: cudaFree failed: " << cudaGetErrorString(err));
            }
            u->handle = NULL;
        }
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED))
        {
            fastFree(u->data);
            u->data = NULL;
            u->origdata = NULL;
        }
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u)
            return;
        UMatDataAutoLock lock(u);
        CV_Assert(u->currAllocator == this);
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
        bool shouldDeallocate = false;
        {
            UMatDataAutoLock lock(u);
            CV_Assert(u->currAllocator == this);
            if (u->refcount == 0)
            {
                shouldDeallocate = (u->urefcount == 0);
                if (!shouldDeallocate && u->deviceCopyObsolete() && u->handle && u->data)
                {
                    cudaSafeCall(cudaMemcpy(u->handle, u->data, u->size, cudaMemcpyHostToDevice));
                    u->markDeviceCopyObsolete(false);
                }
            }
        }
        if (shouldDeallocate)
            deallocate(u);
    }

    void download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE
    {
        if (!u || !dstptr || hasEmptyDim(dims, sz))
            return;
        UMatDataAutoLock lock(u);
        CV_Assert(u->currAllocator == this);
        if (u->deviceCopyObsolete() && u->data)
        {
            MatAllocator::download(u, dstptr, dims, sz, srcofs, srcstep, dststep);
            return;
        }
        if (!u->handle)
            return;
        copyPlanes((uchar*)u->handle, srcofs, srcstep, (uchar*)dstptr, NULL, dststep,
                   dims, sz, cudaMemcpyDeviceToHost);
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u || !srcptr || hasEmptyDim(dims, sz))
            return;
        UMatDataAutoLock lock(u);
        CV_Assert(u->currAllocator == this);
        if (u->refcount > 0 && u->data)
        {
            // the host copy that Mat views use; unmap() uploads it
            MatAllocator::upload(u, srcptr, dims, sz, dstofs, dststep, srcstep);
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
            return;
        }
        if (!u->handle && u->size > 0)
            allocate(u, ACCESS_RW, USAGE_DEFAULT);
        if (!u->handle)
            return;
        if (u->deviceCopyObsolete() && u->data && !isFullOverwrite(u, dims, sz, dstofs, dststep))
            cudaSafeCall(cudaMemcpy(u->handle, u->data, u->size, cudaMemcpyHostToDevice));
        copyPlanes((uchar*)srcptr, NULL, srcstep, (uchar*)u->handle, dstofs, dststep,
                   dims, sz, cudaMemcpyHostToDevice);
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
    }

    // Same-world copy; UMat::copyTo calls copy() solely when both share this allocator.
    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE
    {
        if (!src || !dst || hasEmptyDim(dims, sz))
            return;
        CV_Assert(src->currAllocator == this && dst->currAllocator == this);
        {
            UMatDataAutoLock lock(src, dst);
            if (src != dst && src->refcount == 0 && dst->refcount == 0 && src->handle && dst->handle &&
                !src->deviceCopyObsolete() && !dst->deviceCopyObsolete())
            {
                copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                           dims, sz, cudaMemcpyDeviceToDevice);
                dst->markHostCopyObsolete(true);
                if (sync)
                    cudaSafeCall(cudaStreamSynchronize(0));
                return;
            }
        }
        // with a Mat view alive, a newer host copy, or within one buffer: through one host buffer,
        // which also keeps overlapping regions safe
        size_t tmpstep[CV_MAX_DIM];
        size_t total = sz[dims - 1];
        tmpstep[dims - 1] = 1;
        for (int k = dims - 2; k >= 0; k--)
        {
            tmpstep[k] = total;
            total *= sz[k];
        }
        AutoBuffer<uchar> tmp(total);
        download(src, tmp.data(), dims, sz, srcofs, srcstep, tmpstep);
        upload(dst, tmp.data(), dims, sz, dstofs, dststep, tmpstep);
    }

private:
    static bool hasEmptyDim(int dims, const size_t sz[])
    {
        if (dims <= 0)
            return true;
        CV_Assert(dims <= CV_MAX_DIM && sz != NULL);
        for (int i = 0; i < dims; i++)
            if (sz[i] == 0)
                return true;
        return false;
    }

    static bool isFullOverwrite(const UMatData* u, int dims, const size_t sz[],
                                const size_t dstofs[], const size_t dststep[])
    {
        if (dims <= 0)
            return false;
        size_t expectedStep = 1;
        for (int i = dims - 1; i >= 0; i--)
        {
            if (sz[i] == 0)
                return false;
            if (dstofs && dstofs[i] != 0)
                return false;
            if (i <= dims - 2 && dststep && dststep[i] != expectedStep)
                return false;
            if (expectedStep > u->size / sz[i])
                return false;
            expectedStep *= sz[i];
        }
        return expectedStep == u->size;
    }

    // sz/steps follow the MatAllocator convention: the last dim carries element bytes.
    static void copyPlanes(uchar* srcbase, const size_t srcofs[], const size_t srcstep[],
                           uchar* dstbase, const size_t dstofs[], const size_t dststep[],
                           int dims, const size_t sz[], cudaMemcpyKind kind)
    {
        if (hasEmptyDim(dims, sz))
            return;
        uchar* srcptr = srcbase;
        uchar* dstptr = dstbase;
        for (int i = 0; i < dims; i++)
        {
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
        }

        if (dims == 1)
        {
            cudaSafeCall(cudaMemcpy(dstptr, srcptr, sz[0], kind));
            return;
        }

        if (dims == 2)
        {
            const size_t height = sz[0];
            const size_t widthBytes = sz[1];
            const size_t spitch = srcstep ? srcstep[0] : widthBytes;
            const size_t dpitch = dststep ? dststep[0] : widthBytes;
            if (height == 1 || (spitch == widthBytes && dpitch == widthBytes))
            {
                if (widthBytes > std::numeric_limits<size_t>::max() / height)
                    CV_Error(Error::StsOutOfRange, "2-D copy size overflows size_t");
                cudaSafeCall(cudaMemcpy(dstptr, srcptr, widthBytes * height, kind));
                return;
            }
            if (spitch >= widthBytes && dpitch >= widthBytes &&
                spitch <= (size_t)INT_MAX && dpitch <= (size_t)INT_MAX)
            {
                cudaSafeCall(cudaMemcpy2D(dstptr, dpitch, srcptr, spitch, widthBytes, height, kind));
                return;
            }
        }

        int isz[CV_MAX_DIM];
        for (int i = 0; i < dims; i++)
        {
            CV_Assert(sz[i] <= (size_t)INT_MAX);
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
