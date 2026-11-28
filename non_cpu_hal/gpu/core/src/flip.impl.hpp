// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL, GPU: flip, out of place and in place. Compiled twice through
// ../../precomp.hpp: the kernels with CV_ACCEL_GPU_KERNELS, the operation that launches them without.

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

//! OpenCV's flip codes: 0 around the x axis, > 0 around the y axis, < 0 both
CV_GPU(Error_t) flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, int mode, CV_GPU(Stream_t) s);
//! flip of @p data in place: every element trades places with its mirror image
CV_GPU(Error_t) flipInplace(unsigned char* data, size_t step, int rows, int cols, int esz, int mode,
                            CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

template<typename T>
__global__ void flipKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, bool flipRows, bool flipCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const int sx = flipCols ? cols - 1 - x : x;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        const unsigned sy = flipRows ? (unsigned)rows - 1 - y : y;
        ((T*)(dst + (size_t)y * dstep))[x] = ((const T*)(src + (size_t)sy * sstep))[sx];
    }
}

// each mirrored pair is swapped once, by the thread of its first element
template<typename T>
__global__ void flipInplaceKernel(unsigned char* data, size_t step, int rows, int cols, bool flipRows, bool flipCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const int mx = flipCols ? cols - 1 - x : x;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        const unsigned my = flipRows ? (unsigned)rows - 1 - y : y;
        if (y > my || (y == my && x >= mx))
            continue;
        T* a = (T*)(data + (size_t)y * step) + x;
        T* b = (T*)(data + (size_t)my * step) + mx;
        const T t = *a;
        *a = *b;
        *b = t;
    }
}

template<int N>
CV_GPU(Error_t) flipN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, int mode, CV_GPU(Stream_t) s)
{
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &flipRows, &flipCols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)flipKernel<Elem<N> > : (const void*)flipKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y),
                                args, 0, s);
}

template<int N>
CV_GPU(Error_t) flipInplaceN(unsigned char* data, size_t step, int rows, int cols, int mode, CV_GPU(Stream_t) s)
{
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &data, &step, &rows, &cols, &flipRows, &flipCols };
    const bool wide = aligned(data, step, data, step, alignof(Elem<N>));
    const void* k = wide ? (const void*)flipInplaceKernel<Elem<N> > : (const void*)flipInplaceKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y),
                                args, 0, s);
}

} // namespace

CV_GPU(Error_t) flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, int mode, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(flipN, src, sstep, dst, dstep, rows, cols, mode, s)
}

CV_GPU(Error_t) flipInplace(unsigned char* data, size_t step, int rows, int cols, int esz, int mode,
                            CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(flipInplaceN, data, step, rows, cols, mode, s)
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operation

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! flip, one of the CoreOps the module mixes together (precomp.hpp)
template<typename Base>
struct GpuFlipOps : Base
{
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE;
};

template<typename Base>
bool GpuFlipOps<Base>::flip(const UMat& src, UMat& dst, int flipCode, Stream& stream)
{
    if (!runnable(src, dst))
        return false;
    GpuStream& s = gpuStreamOf(stream);
    DeviceGuard g(s.device_);
    const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
    unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
    const int esz = (int)src.elemSize();
    const CV_GPU(Error_t) r = src.u == dst.u
          ? kernels::flipInplace(dp, dst.step[0], src.rows, src.cols, esz, flipCode, s.s_)
          : kernels::flip(sp, src.step[0], dp, dst.step[0], src.rows, src.cols, esz, flipCode, s.s_);
    if (noKernelImage(r))
    {
        (void)CV_GPU(GetLastError)();
        return false;   // no code for this GPU in the build: nothing was queued, the CPU runs it
    }
    check(r, "flip");
    stream.retain({ src, dst });
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS
