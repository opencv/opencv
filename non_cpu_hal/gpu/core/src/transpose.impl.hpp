// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL, GPU: transpose, out of place and of a square matrix in place. Compiled twice through
// ../../precomp.hpp: the kernels with CV_ACCEL_GPU_KERNELS, the operation that launches them without.

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

//! @p rows x @p cols in, cols x rows out
CV_GPU(Error_t) transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                          int rows, int cols, int esz, CV_GPU(Stream_t) s);
//! transpose of the square @p n x @p n matrix @p data in place
CV_GPU(Error_t) transposeInplace(unsigned char* data, size_t step, int n, int esz, CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

const int TILE = 32;

// a TILE x TILE block goes through shared memory, so both the reads and the writes walk along rows
template<typename T>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    __shared__ T tile[TILE][TILE + 1];
    const int x0 = blockIdx.x * TILE;
    for (int y0 = blockIdx.y * TILE; y0 < rows; y0 += gridDim.y * TILE)
    {
        const int x = x0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (x < cols && y0 + j < rows)
                tile[j][threadIdx.x] = ((const T*)(src + (size_t)(y0 + j) * sstep))[x];
        __syncthreads();
        const int ox = y0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (ox < rows && x0 + j < cols)
                ((T*)(dst + (size_t)(x0 + j) * dstep))[ox] = tile[threadIdx.x][j];
        __syncthreads();
    }
}

// each element above the diagonal is swapped with its counterpart below it
template<typename T>
__global__ void transposeInplaceKernel(unsigned char* data, size_t step, int n)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)x; y += gridDim.y * blockDim.y)
    {
        T* a = (T*)(data + (size_t)y * step) + x;
        T* b = (T*)(data + (size_t)x * step) + y;
        const T t = *a;
        *a = *b;
        *b = t;
    }
}

template<int N>
CV_GPU(Error_t) transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, CV_GPU(Stream_t) s)
{
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)transposeKernel<Elem<N> >
                         : (const void*)transposeKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, TILE, TILE), dim3(TILE, BLOCK_Y), args, 0, s);
}

template<int N>
CV_GPU(Error_t) transposeInplaceN(unsigned char* data, size_t step, int n, CV_GPU(Stream_t) s)
{
    void* args[] = { &data, &step, &n };
    const bool wide = aligned(data, step, data, step, alignof(Elem<N>));
    const void* k = wide ? (const void*)transposeInplaceKernel<Elem<N> >
                         : (const void*)transposeInplaceKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(n, n, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y), args, 0, s);
}

} // namespace

CV_GPU(Error_t) transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                          int rows, int cols, int esz, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

CV_GPU(Error_t) transposeInplace(unsigned char* data, size_t step, int n, int esz, CV_GPU(Stream_t) s)
{
    if (n <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(transposeInplaceN, data, step, n, s)
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operation

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! transpose, one of the CoreOps the module mixes together (precomp.hpp)
template<typename Base>
struct GpuTransposeOps : Base
{
    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE;
};

template<typename Base>
bool GpuTransposeOps<Base>::transpose(const UMat& src, UMat& dst, Stream& stream)
{
    if (!runnable(src, dst) || (src.u == dst.u && src.rows != src.cols))
        return false;
    GpuStream& s = gpuStreamOf(stream);
    DeviceGuard g(s.device_);
    const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
    unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
    const int esz = (int)src.elemSize();
    const CV_GPU(Error_t) r = src.u == dst.u
          ? kernels::transposeInplace(dp, dst.step[0], src.rows, esz, s.s_)
          : kernels::transpose(sp, src.step[0], dp, dst.step[0], src.rows, src.cols, esz, s.s_);
    if (noKernelImage(r))
    {
        (void)CV_GPU(GetLastError)();
        return false;
    }
    check(r, "transpose");
    stream.retain({ src, dst });
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS
