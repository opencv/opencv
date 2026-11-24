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
#include <cstdint>
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
            size_t srcStart = 0, srcEnd = 0, dstStart = 0, dstEnd = 0;
            size_t packedStep[CV_MAX_DIM] = { 0 };
            size_t packedBytes = 0;
            computeRegionSpan(dims, sz, srcofs, srcstep, srcStart, srcEnd, packedStep, packedBytes);
            size_t dummyBytes = 0;
            computeRegionSpan(dims, sz, NULL, dststep, dstStart, dstEnd, NULL, dummyBytes);
            const size_t* effSrcStep = srcstep ? srcstep : packedStep;
            const size_t* effDstStep = dststep ? dststep : packedStep;
            if (hostSpansOverlap(u->data, srcStart, srcEnd, static_cast<const uchar*>(dstptr), dstStart, dstEnd))
            {
                AutoBuffer<uchar> tmp(packedBytes);
                MatAllocator::download(u, tmp.data(), dims, sz, srcofs, effSrcStep, packedStep);
                UMatData tmpU(NULL);
                tmpU.data = tmp.data();
                MatAllocator::download(&tmpU, dstptr, dims, sz, NULL, packedStep, effDstStep);
                return;
            }
            MatAllocator::download(u, dstptr, dims, sz, srcofs, effSrcStep, effDstStep);
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
            size_t srcStart = 0, srcEnd = 0, dstStart = 0, dstEnd = 0;
            size_t packedStep[CV_MAX_DIM] = { 0 };
            size_t packedBytes = 0;
            computeRegionSpan(dims, sz, NULL, srcstep, srcStart, srcEnd, packedStep, packedBytes);
            size_t dummyBytes = 0;
            computeRegionSpan(dims, sz, dstofs, dststep, dstStart, dstEnd, NULL, dummyBytes);
            const size_t* effSrcStep = srcstep ? srcstep : packedStep;
            const size_t* effDstStep = dststep ? dststep : packedStep;
            if (hostSpansOverlap(static_cast<const uchar*>(srcptr), srcStart, srcEnd, u->data, dstStart, dstEnd))
            {
                AutoBuffer<uchar> tmp(packedBytes);
                UMatData tmpU(NULL);
                tmpU.data = const_cast<uchar*>(static_cast<const uchar*>(srcptr));
                MatAllocator::download(&tmpU, tmp.data(), dims, sz, NULL, effSrcStep, packedStep);
                MatAllocator::upload(u, tmp.data(), dims, sz, dstofs, effDstStep, packedStep);
            }
            else
            {
                MatAllocator::upload(u, srcptr, dims, sz, dstofs, effDstStep, effSrcStep);
            }
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
        UMatDataAutoLock lock(src, dst);
        CV_Assert(src->currAllocator == this && dst->currAllocator == this);
        const bool srcHost = src->deviceCopyObsolete() && src->data;
        const bool dstHostView = dst->refcount > 0 && dst->data;

        if (dstHostView)
        {
            if (srcHost)
            {
                size_t srcStart = 0, srcEnd = 0, dstStart = 0, dstEnd = 0;
                size_t packedStep[CV_MAX_DIM] = { 0 };
                size_t packedBytes = 0;
                computeRegionSpan(dims, sz, srcofs, srcstep, srcStart, srcEnd, packedStep, packedBytes);
                size_t dummyBytes = 0;
                computeRegionSpan(dims, sz, dstofs, dststep, dstStart, dstEnd, NULL, dummyBytes);
                const size_t* effSrcStep = srcstep ? srcstep : packedStep;
                const size_t* effDstStep = dststep ? dststep : packedStep;
                if (hostSpansOverlap(src->data, srcStart, srcEnd, dst->data, dstStart, dstEnd))
                {
                    AutoBuffer<uchar> tmp(packedBytes);
                    MatAllocator::download(src, tmp.data(), dims, sz, srcofs, effSrcStep, packedStep);
                    MatAllocator::upload(dst, tmp.data(), dims, sz, dstofs, effDstStep, packedStep);
                }
                else
                {
                    MatAllocator::copy(src, dst, dims, sz, srcofs, effSrcStep, dstofs, effDstStep, false);
                }
            }
            else
            {
                if (!src->handle)
                    return;
                copyPlanes((uchar*)src->handle, srcofs, srcstep, dst->data, dstofs, dststep,
                           dims, sz, cudaMemcpyDeviceToHost);
                if (sync)
                    cudaSafeCall(cudaStreamSynchronize(0));
            }
            dst->markHostCopyObsolete(false);
            dst->markDeviceCopyObsolete(true);
            return;
        }

        if (!srcHost && !src->handle)
            return;
        if (!dst->handle && dst->size > 0)
            allocate(dst, ACCESS_RW, USAGE_DEFAULT);
        if (!dst->handle)
            return;

        if (dst->deviceCopyObsolete() && dst->data && !isFullOverwrite(dst, dims, sz, dstofs, dststep))
            cudaSafeCall(cudaMemcpy(dst->handle, dst->data, dst->size, cudaMemcpyHostToDevice));

        if (srcHost)
        {
            copyPlanes(src->data, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                       dims, sz, cudaMemcpyHostToDevice);
        }
        else if (src == dst)
        {
            size_t srcStart = 0, srcEnd = 0, dstStart = 0, dstEnd = 0;
            size_t packedStep[CV_MAX_DIM] = { 0 };
            size_t packedBytes = 0;
            computeRegionSpan(dims, sz, srcofs, srcstep, srcStart, srcEnd, packedStep, packedBytes);
            size_t dummyBytes = 0;
            computeRegionSpan(dims, sz, dstofs, dststep, dstStart, dstEnd, NULL, dummyBytes);
            if (srcStart < dstEnd && dstStart < srcEnd)
            {
                void* tmpDev = NULL;
                cudaSafeCall(cudaMalloc(&tmpDev, packedBytes));
                try
                {
                    copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)tmpDev, NULL, packedStep,
                               dims, sz, cudaMemcpyDeviceToDevice);
                    copyPlanes((uchar*)tmpDev, NULL, packedStep, (uchar*)dst->handle, dstofs, dststep,
                               dims, sz, cudaMemcpyDeviceToDevice);
                }
                catch (...)
                {
                    (void)cudaFree(tmpDev);
                    throw;
                }
                cudaError_t err = cudaFree(tmpDev);
                if (err != cudaSuccess)
                {
                    (void)cudaGetLastError();
                    CV_LOG_WARNING(NULL, "CudaUMatAllocator: cudaFree(tmp) failed: " << cudaGetErrorString(err));
                }
            }
            else
            {
                copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                           dims, sz, cudaMemcpyDeviceToDevice);
            }
        }
        else
        {
            copyPlanes((uchar*)src->handle, srcofs, srcstep, (uchar*)dst->handle, dstofs, dststep,
                       dims, sz, cudaMemcpyDeviceToDevice);
        }
        dst->markHostCopyObsolete(true);
        dst->markDeviceCopyObsolete(false);
        if (sync)
            cudaSafeCall(cudaStreamSynchronize(0));
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

    static bool hostSpansOverlap(const uchar* baseA, size_t startA, size_t endA,
                                 const uchar* baseB, size_t startB, size_t endB)
    {
        const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(baseA) + startA;
        const std::uintptr_t a1 = reinterpret_cast<std::uintptr_t>(baseA) + endA;
        const std::uintptr_t b0 = reinterpret_cast<std::uintptr_t>(baseB) + startB;
        const std::uintptr_t b1 = reinterpret_cast<std::uintptr_t>(baseB) + endB;
        return a0 < b1 && b0 < a1;
    }

    static void computeRegionSpan(int dims, const size_t sz[],
                                  const size_t ofs[], const size_t step[],
                                  size_t& startOfs, size_t& endOfs,
                                  size_t* packedStep, size_t& packedBytes)
    {
        size_t s = 1;
        size_t relEnd = 0;
        startOfs = 0;
        for (int i = dims - 1; i >= 0; i--)
        {
            const size_t dimSz = sz[i];
            if (packedStep)
                packedStep[i] = s;
            const size_t stride = (i <= dims - 2) ? (step ? step[i] : s) : 1;
            if (ofs && ofs[i] != 0)
            {
                CV_Assert(i > dims - 2 || step != NULL);
                startOfs += ofs[i] * stride;
            }
            relEnd += (dimSz - 1) * stride;
            if (s > std::numeric_limits<size_t>::max() / dimSz)
                CV_Error(Error::StsOutOfRange, "Copy region size overflows size_t");
            s *= dimSz;
        }
        packedBytes = s;
        endOfs = startOfs + relEnd + 1;
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
