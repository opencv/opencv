/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include "internal_shared.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/limits.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace stereobp
    {
        ///////////////////////////////////////////////////////////////
        /////////////////////// load constants ////////////////////////
        ///////////////////////////////////////////////////////////////

        __constant__ int   cndisp;
        __constant__ float cmax_data_term;
        __constant__ float cdata_weight;
        __constant__ float cmax_disc_term;
        __constant__ float cdisc_single_jump;

        void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump)
        {
            cudaSafeCall( cudaMemcpyToSymbol(cndisp,            &ndisp,            sizeof(int  )) );
            cudaSafeCall( cudaMemcpyToSymbol(cmax_data_term,    &max_data_term,    sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(cdata_weight,      &data_weight,      sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(cmax_disc_term,    &max_disc_term,    sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(cdisc_single_jump, &disc_single_jump, sizeof(float)) );
        }

        ///////////////////////////////////////////////////////////////
        ////////////////////////// comp data //////////////////////////
        ///////////////////////////////////////////////////////////////

        template <int cn> struct PixDiff;
        template <> struct PixDiff<1>
        {
            __device__ __forceinline__ PixDiff(const uchar* ls)
            {
                l = *ls;
            }
            __device__ __forceinline__ float operator()(const uchar* rs) const
            {
                return ::abs((int)l - *rs);
            }
            uchar l;
        };
        template <> struct PixDiff<3>
        {
            __device__ __forceinline__ PixDiff(const uchar* ls)
            {
                l = *((uchar3*)ls);
            }
            __device__ __forceinline__ float operator()(const uchar* rs) const
            {
                const float tr = 0.299f;
                const float tg = 0.587f;
                const float tb = 0.114f;

                float val  = tb * ::abs((int)l.x - rs[0]);
                      val += tg * ::abs((int)l.y - rs[1]);
                      val += tr * ::abs((int)l.z - rs[2]);

                return val;
            }
            uchar3 l;
        };
        template <> struct PixDiff<4>
        {
            __device__ __forceinline__ PixDiff(const uchar* ls)
            {
                l = *((uchar4*)ls);
            }
            __device__ __forceinline__ float operator()(const uchar* rs) const
            {
                const float tr = 0.299f;
                const float tg = 0.587f;
                const float tb = 0.114f;

                uchar4 r = *((uchar4*)rs);

                float val  = tb * ::abs((int)l.x - r.x);
                      val += tg * ::abs((int)l.y - r.y);
                      val += tr * ::abs((int)l.z - r.z);

                return val;
            }
            uchar4 l;
        };

        template <int cn, typename D>
        __global__ void comp_data(const PtrStepSzb left, const PtrStepb right, PtrStep<D> data)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y > 0 && y < left.rows - 1 && x > 0 && x < left.cols - 1)
            {
                const uchar* ls = left.ptr(y) + x * cn;
                const PixDiff<cn> pixDiff(ls);
                const uchar* rs = right.ptr(y) + x * cn;

                D* ds = data.ptr(y) + x;
                const size_t disp_step = data.step * left.rows / sizeof(D);

                for (int disp = 0; disp < cndisp; disp++)
                {
                    if (x - disp >= 1)
                    {
                        float val = pixDiff(rs - disp * cn);

                        ds[disp * disp_step] = saturate_cast<D>(fmin(cdata_weight * val, cdata_weight * cmax_data_term));
                    }
                    else
                    {
                        ds[disp * disp_step] = saturate_cast<D>(cdata_weight * cmax_data_term);
                    }
                }
            }
        }

        template<typename T, typename D>
        void comp_data_gpu(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream);

        template <> void comp_data_gpu<uchar, short>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<1, short><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<short>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        template <> void comp_data_gpu<uchar, float>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<1, float><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<float>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <> void comp_data_gpu<uchar3, short>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<3, short><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<short>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        template <> void comp_data_gpu<uchar3, float>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<3, float><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<float>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <> void comp_data_gpu<uchar4, short>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<4, short><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<short>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        template <> void comp_data_gpu<uchar4, float>(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(left.cols, threads.x);
            grid.y = divUp(left.rows, threads.y);

            comp_data<4, float><<<grid, threads, 0, stream>>>(left, right, (PtrStepSz<float>)data);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ///////////////////////////////////////////////////////////////
        //////////////////////// data step down ///////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T>
        __global__ void data_step_down(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStep<T> src, PtrStep<T> dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst_cols && y < dst_rows)
            {
                for (int d = 0; d < cndisp; ++d)
                {
                    float dst_reg  = src.ptr(d * src_rows + ::min(2*y+0, src_rows-1))[::min(2*x+0, src_cols-1)];
                          dst_reg += src.ptr(d * src_rows + ::min(2*y+1, src_rows-1))[::min(2*x+0, src_cols-1)];
                          dst_reg += src.ptr(d * src_rows + ::min(2*y+0, src_rows-1))[::min(2*x+1, src_cols-1)];
                          dst_reg += src.ptr(d * src_rows + ::min(2*y+1, src_rows-1))[::min(2*x+1, src_cols-1)];

                    dst.ptr(d * dst_rows + y)[x] = saturate_cast<T>(dst_reg);
                }
            }
        }

        template<typename T>
        void data_step_down_gpu(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStepSzb& src, const PtrStepSzb& dst, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(dst_cols, threads.x);
            grid.y = divUp(dst_rows, threads.y);

            data_step_down<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_cols, src_rows, (PtrStepSz<T>)src, (PtrStepSz<T>)dst);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void data_step_down_gpu<short>(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStepSzb& src, const PtrStepSzb& dst, cudaStream_t stream);
        template void data_step_down_gpu<float>(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStepSzb& src, const PtrStepSzb& dst, cudaStream_t stream);

        ///////////////////////////////////////////////////////////////
        /////////////////// level up messages  ////////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T>
        __global__ void level_up_message(int dst_cols, int dst_rows, int src_rows, const PtrStep<T> src, PtrStep<T> dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst_cols && y < dst_rows)
            {
                const size_t dst_disp_step = dst.step * dst_rows / sizeof(T);
                const size_t src_disp_step = src.step * src_rows / sizeof(T);

                T*       dstr = dst.ptr(y  ) + x;
                const T* srcr = src.ptr(y/2) + x/2;

                for (int d = 0; d < cndisp; ++d)
                    dstr[d * dst_disp_step] = srcr[d * src_disp_step];
            }
        }

        template <typename T>
        void level_up_messages_gpu(int dst_idx, int dst_cols, int dst_rows, int src_rows, PtrStepSzb* mus, PtrStepSzb* mds, PtrStepSzb* mls, PtrStepSzb* mrs, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(dst_cols, threads.x);
            grid.y = divUp(dst_rows, threads.y);

            int src_idx = (dst_idx + 1) & 1;

            level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (PtrStepSz<T>)mus[src_idx], (PtrStepSz<T>)mus[dst_idx]);
            cudaSafeCall( cudaGetLastError() );

            level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (PtrStepSz<T>)mds[src_idx], (PtrStepSz<T>)mds[dst_idx]);
            cudaSafeCall( cudaGetLastError() );

            level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (PtrStepSz<T>)mls[src_idx], (PtrStepSz<T>)mls[dst_idx]);
            cudaSafeCall( cudaGetLastError() );

            level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (PtrStepSz<T>)mrs[src_idx], (PtrStepSz<T>)mrs[dst_idx]);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void level_up_messages_gpu<short>(int dst_idx, int dst_cols, int dst_rows, int src_rows, PtrStepSzb* mus, PtrStepSzb* mds, PtrStepSzb* mls, PtrStepSzb* mrs, cudaStream_t stream);
        template void level_up_messages_gpu<float>(int dst_idx, int dst_cols, int dst_rows, int src_rows, PtrStepSzb* mus, PtrStepSzb* mds, PtrStepSzb* mls, PtrStepSzb* mrs, cudaStream_t stream);

        ///////////////////////////////////////////////////////////////
        ////////////////////  calc all iterations /////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T>
        __device__ void calc_min_linear_penalty(T* dst, size_t step)
        {
            float prev = dst[0];
            float cur;
            for (int disp = 1; disp < cndisp; ++disp)
            {
                prev += cdisc_single_jump;
                cur = dst[step * disp];
                if (prev < cur)
                {
                    cur = prev;
                    dst[step * disp] = saturate_cast<T>(prev);
                }
                prev = cur;
            }

            prev = dst[(cndisp - 1) * step];
            for (int disp = cndisp - 2; disp >= 0; disp--)
            {
                prev += cdisc_single_jump;
                cur = dst[step * disp];
                if (prev < cur)
                {
                    cur = prev;
                    dst[step * disp] = saturate_cast<T>(prev);
                }
                prev = cur;
            }
        }

        template <typename T>
        __device__ void message(const T* msg1, const T* msg2, const T* msg3, const T* data, T* dst, size_t msg_disp_step, size_t data_disp_step)
        {
            float minimum = device::numeric_limits<float>::max();

            for(int i = 0; i < cndisp; ++i)
            {
                float dst_reg  = msg1[msg_disp_step * i];
                      dst_reg += msg2[msg_disp_step * i];
                      dst_reg += msg3[msg_disp_step * i];
                      dst_reg += data[data_disp_step * i];

                if (dst_reg < minimum)
                    minimum = dst_reg;

                dst[msg_disp_step * i] = saturate_cast<T>(dst_reg);
            }

            calc_min_linear_penalty(dst, msg_disp_step);

            minimum += cmax_disc_term;

            float sum = 0;
            for(int i = 0; i < cndisp; ++i)
            {
                float dst_reg = dst[msg_disp_step * i];
                if (dst_reg > minimum)
                {
                    dst_reg = minimum;
                    dst[msg_disp_step * i] = saturate_cast<T>(minimum);
                }
                sum += dst_reg;
            }
            sum /= cndisp;

            for(int i = 0; i < cndisp; ++i)
                dst[msg_disp_step * i] -= sum;
        }

        template <typename T>
        __global__ void one_iteration(int t, int elem_step, T* u, T* d, T* l, T* r, const PtrStep<T> data, int cols, int rows)
        {
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + t) & 1);

            if ((y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
            {
                T* us = u + y * elem_step + x;
                T* ds = d + y * elem_step + x;
                T* ls = l + y * elem_step + x;
                T* rs = r + y * elem_step + x;
                const T* dt = data.ptr(y) + x;

                size_t msg_disp_step = elem_step * rows;
                size_t data_disp_step = data.step * rows / sizeof(T);

                message(us + elem_step, ls         + 1, rs - 1, dt, us, msg_disp_step, data_disp_step);
                message(ds - elem_step, ls         + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step);
                message(us + elem_step, ds - elem_step, rs - 1, dt, rs, msg_disp_step, data_disp_step);
                message(us + elem_step, ds - elem_step, ls + 1, dt, ls, msg_disp_step, data_disp_step);
            }
        }

        template <typename T>
        void calc_all_iterations_gpu(int cols, int rows, int iters, const PtrStepSzb& u, const PtrStepSzb& d,
            const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(cols, threads.x << 1);
            grid.y = divUp(rows, threads.y);

            int elem_step = (int)(u.step / sizeof(T));

            for(int t = 0; t < iters; ++t)
            {
                one_iteration<T><<<grid, threads, 0, stream>>>(t, elem_step, (T*)u.data, (T*)d.data, (T*)l.data, (T*)r.data, (PtrStepSz<T>)data, cols, rows);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
            }
        }

        template void calc_all_iterations_gpu<short>(int cols, int rows, int iters, const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, cudaStream_t stream);
        template void calc_all_iterations_gpu<float>(int cols, int rows, int iters, const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, cudaStream_t stream);

        ///////////////////////////////////////////////////////////////
        /////////////////////////// output ////////////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T>
        __global__ void output(const int elem_step, const T* u, const T* d, const T* l, const T* r, const T* data,
            PtrStepSz<short> disp)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y > 0 && y < disp.rows - 1 && x > 0 && x < disp.cols - 1)
            {
                const T* us = u + (y + 1) * elem_step + x;
                const T* ds = d + (y - 1) * elem_step + x;
                const T* ls = l + y * elem_step + (x + 1);
                const T* rs = r + y * elem_step+ (x - 1);
                const T* dt = data + y * elem_step + x;

                size_t disp_step = disp.rows * elem_step;

                int best = 0;
                float best_val = numeric_limits<float>::max();
                for (int d = 0; d < cndisp; ++d)
                {
                    float val  = us[d * disp_step];
                          val += ds[d * disp_step];
                          val += ls[d * disp_step];
                          val += rs[d * disp_step];
                          val += dt[d * disp_step];

                    if (val < best_val)
                    {
                        best_val = val;
                        best = d;
                    }
                }

                disp.ptr(y)[x] = saturate_cast<short>(best);
            }
        }

        template <typename T>
        void output_gpu(const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data,
            const PtrStepSz<short>& disp, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(disp.cols, threads.x);
            grid.y = divUp(disp.rows, threads.y);

            int elem_step = static_cast<int>(u.step/sizeof(T));

            output<T><<<grid, threads, 0, stream>>>(elem_step, (const T*)u.data, (const T*)d.data, (const T*)l.data, (const T*)r.data, (const T*)data.data, disp);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void output_gpu<short>(const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, const PtrStepSz<short>& disp, cudaStream_t stream);
        template void output_gpu<float>(const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, const PtrStepSz<short>& disp, cudaStream_t stream);
    } // namespace stereobp
}}} // namespace cv { namespace gpu { namespace device

#endif /* CUDA_DISABLER */
