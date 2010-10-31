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
// any express or bpied warranties, including, but not limited to, the bpied
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

#include "opencv2/gpu/devmem2d.hpp"
#include "saturate_cast.hpp"
#include "safe_call.hpp"

using namespace cv::gpu;

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

namespace cv { namespace gpu { namespace bp {

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


    template <typename T>
    __global__ void comp_data_gray(const uchar* l, const uchar* r, size_t step, T* data, size_t data_step, int cols, int rows) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
        {
            const uchar* ls = l + y * step + x; 
            const uchar* rs = r + y * step + x; 

            T* ds = data + y * data_step + x;
            size_t disp_step = data_step * rows;

            for (int disp = 0; disp < cndisp; disp++) 
            {
                if (x - disp >= 1)
                {
                    float val  = abs((int)ls[0] - rs[-disp]);
                    
                    ds[disp * disp_step] = saturate_cast<T>(fmin(cdata_weight * val, cdata_weight * cmax_data_term));
                }
                else
                {
                    ds[disp * disp_step] = saturate_cast<T>(cdata_weight * cmax_data_term);
                }
            }
        }
    }

    template <typename T>
    __global__ void comp_data_bgr(const uchar* l, const uchar* r, size_t step, T* data, size_t data_step, int cols, int rows) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
        {
            const uchar* ls = l + y * step + x * 3; 
            const uchar* rs = r + y * step + x * 3; 

            T* ds = data + y * data_step + x;
            size_t disp_step = data_step * rows;

            for (int disp = 0; disp < cndisp; disp++) 
            {
                if (x - disp >= 1)
                {                    
                    const float tr = 0.299f;
                    const float tg = 0.587f;
                    const float tb = 0.114f;

                    float val  = tb * abs((int)ls[0] - rs[0-disp*3]);
                          val += tg * abs((int)ls[1] - rs[1-disp*3]);
                          val += tr * abs((int)ls[2] - rs[2-disp*3]);
                    
                    ds[disp * disp_step] = saturate_cast<T>(fmin(cdata_weight * val, cdata_weight * cmax_data_term));
                }
                else
                {
                    ds[disp * disp_step] = saturate_cast<T>(cdata_weight * cmax_data_term);
                }
            }
        }
    }

    typedef void (*CompDataFunc)(const DevMem2D& l, const DevMem2D& r, int channels, DevMem2D mdata, const cudaStream_t& stream);

    template<typename T>
    void comp_data_(const DevMem2D& l, const DevMem2D& r, int channels, DevMem2D mdata, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(l.cols, threads.x);
        grid.y = divUp(l.rows, threads.y);
        
        if (channels == 1)
            comp_data_gray<T><<<grid, threads, 0, stream>>>(l.data, r.data, l.step, (T*)mdata.data, mdata.step/sizeof(T), l.cols, l.rows);
        else
            comp_data_bgr<T><<<grid, threads, 0, stream>>>(l.data, r.data, l.step, (T*)mdata.data, mdata.step/sizeof(T), l.cols, l.rows);
        
        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void comp_data(int msg_type, const DevMem2D& l, const DevMem2D& r, int channels, DevMem2D mdata, const cudaStream_t& stream)
    {
        static CompDataFunc tab[8] =
        {
            0,                  // uchar
            0,                  // schar
            0,                  // ushort
            comp_data_<short>,  // short
            0,                  // int
            comp_data_<float>,  // float
            0,                  // double
            0                   // user type
        };

        CompDataFunc func = tab[msg_type];
        if (func == 0)
            cv::gpu::error("Unsupported message type", __FILE__, __LINE__);
        func(l, r, channels, mdata, stream);
    }

///////////////////////////////////////////////////////////////
//////////////////////// data step down ///////////////////////
///////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void data_step_down(int dst_cols, int dst_rows, int src_rows, const T* src, size_t src_step, T* dst, size_t dst_step)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < dst_cols && y < dst_rows)
        {
            const size_t dst_disp_step = dst_step * dst_rows;
            const size_t src_disp_step = src_step * src_rows;

            for (int d = 0; d < cndisp; ++d)
            {
                float dst_reg  = src[d * src_disp_step + src_step * (2*y+0) + (2*x+0)];
                      dst_reg += src[d * src_disp_step + src_step * (2*y+1) + (2*x+0)];
                      dst_reg += src[d * src_disp_step + src_step * (2*y+0) + (2*x+1)];
                      dst_reg += src[d * src_disp_step + src_step * (2*y+1) + (2*x+1)];

                dst[d * dst_disp_step + y * dst_step + x] = saturate_cast<T>(dst_reg);
            }
        }
    }

    typedef void (*DataStepDownFunc)(int dst_cols, int dst_rows, int src_rows, const DevMem2D& src, DevMem2D dst, const cudaStream_t& stream);

    template<typename T>
    void data_step_down_(int dst_cols, int dst_rows, int src_rows, const DevMem2D& src, DevMem2D dst, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(dst_cols, threads.x);
        grid.y = divUp(dst_rows, threads.y);

        data_step_down<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (const T*)src.data, src.step/sizeof(T), (T*)dst.data, dst.step/sizeof(T));
        
        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void data_step_down(int dst_cols, int dst_rows, int src_rows, int msg_type, const DevMem2D& src, DevMem2D dst, const cudaStream_t& stream)
    {
        static DataStepDownFunc tab[8] =
        {
            0,                       // uchar
            0,                       // schar
            0,                       // ushort
            data_step_down_<short>,  // short
            0,                       // int
            data_step_down_<float>,  // float
            0,                       // double
            0                        // user type
        };

        DataStepDownFunc func = tab[msg_type];
        if (func == 0)
            cv::gpu::error("Unsupported message type", __FILE__, __LINE__);
        func(dst_cols, dst_rows, src_rows, src, dst, stream);
    }

///////////////////////////////////////////////////////////////
/////////////////// level up messages  ////////////////////////
///////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void level_up_message(int dst_cols, int dst_rows, int src_rows, const T* src, size_t src_step, T* dst, size_t dst_step)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;        

        if (x < dst_cols && y < dst_rows)
        {
            const size_t dst_disp_step = dst_step * dst_rows;
            const size_t src_disp_step = src_step * src_rows;

            T*       dstr = dst + y   * dst_step + x;
            const T* srcr = src + y/2 * src_step + x/2;

            for (int d = 0; d < cndisp; ++d)            
                dstr[d * dst_disp_step] = srcr[d * src_disp_step];
        }
    }

    typedef void (*LevelUpMessagesFunc)(int dst_idx, int dst_cols, int dst_rows, int src_rows, DevMem2D* mus, DevMem2D* mds, DevMem2D* mls, DevMem2D* mrs, const cudaStream_t& stream);

    template<typename T>
    void level_up_messages_(int dst_idx, int dst_cols, int dst_rows, int src_rows, DevMem2D* mus, DevMem2D* mds, DevMem2D* mls, DevMem2D* mrs, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(dst_cols, threads.x);
        grid.y = divUp(dst_rows, threads.y);

        int src_idx = (dst_idx + 1) & 1;

        level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (const T*)mus[src_idx].data, mus[src_idx].step/sizeof(T), (T*)mus[dst_idx].data, mus[dst_idx].step/sizeof(T));
        level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (const T*)mds[src_idx].data, mds[src_idx].step/sizeof(T), (T*)mds[dst_idx].data, mds[dst_idx].step/sizeof(T));
        level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (const T*)mls[src_idx].data, mls[src_idx].step/sizeof(T), (T*)mls[dst_idx].data, mls[dst_idx].step/sizeof(T));
        level_up_message<T><<<grid, threads, 0, stream>>>(dst_cols, dst_rows, src_rows, (const T*)mrs[src_idx].data, mrs[src_idx].step/sizeof(T), (T*)mrs[dst_idx].data, mrs[dst_idx].step/sizeof(T));
        
        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void level_up_messages(int dst_idx, int dst_cols, int dst_rows, int src_rows, int msg_type, DevMem2D* mus, DevMem2D* mds, DevMem2D* mls, DevMem2D* mrs, const cudaStream_t& stream)
    {
        static LevelUpMessagesFunc tab[8] =
        {
            0,                          // uchar
            0,                          // schar
            0,                          // ushort
            level_up_messages_<short>,  // short
            0,                          // int
            level_up_messages_<float>,  // float
            0,                          // double
            0                           // user type
        };

        LevelUpMessagesFunc func = tab[msg_type];
        if (func == 0)
            cv::gpu::error("Unsupported message type", __FILE__, __LINE__);
        func(dst_idx, dst_cols, dst_rows, src_rows, mus, mds, mls, mrs, stream);
    }

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
        float minimum = FLT_MAX;

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
    __global__ void one_iteration(int t, T* u, T* d, T* l, T* r, size_t msg_step, const T* data, size_t data_step, int cols, int rows)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + t) & 1);

        if ( (y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
        {
            T* us = u + y * msg_step + x;
            T* ds = d + y * msg_step + x;
            T* ls = l + y * msg_step + x;
            T* rs = r + y * msg_step + x;
            const T* dt = data + y * data_step + x;

            size_t msg_disp_step = msg_step * rows;
            size_t data_disp_step = data_step * rows;

            message(us + msg_step, ls        + 1, rs - 1, dt, us, msg_disp_step, data_disp_step);
            message(ds - msg_step, ls        + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step);
            message(us + msg_step, ds - msg_step, rs - 1, dt, rs, msg_disp_step, data_disp_step);
            message(us + msg_step, ds - msg_step, ls + 1, dt, ls, msg_disp_step, data_disp_step);                
        }
    }

    typedef void (*CalcAllIterationFunc)(int cols, int rows, int iters, DevMem2D& u, DevMem2D& d, DevMem2D& l, DevMem2D& r, const DevMem2D& data, const cudaStream_t& stream);

    template<typename T>
    void calc_all_iterations_(int cols, int rows, int iters, DevMem2D& u, DevMem2D& d, DevMem2D& l, DevMem2D& r, const DevMem2D& data, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(cols, threads.x << 1);
        grid.y = divUp(rows, threads.y);

        for(int t = 0; t < iters; ++t)
        {
            one_iteration<T><<<grid, threads, 0, stream>>>(t, (T*)u.data, (T*)d.data, (T*)l.data, (T*)r.data, u.step/sizeof(T), (const T*)data.data, data.step/sizeof(T), cols, rows);
            
            if (stream == 0)
                cudaSafeCall( cudaThreadSynchronize() );
        }
    }

    void calc_all_iterations(int cols, int rows, int iters, int msg_type, DevMem2D& u, DevMem2D& d, DevMem2D& l, DevMem2D& r, const DevMem2D& data, const cudaStream_t& stream)
    {
        static CalcAllIterationFunc tab[8] =
        {
            0,                            // uchar
            0,                            // schar
            0,                            // ushort
            calc_all_iterations_<short>,  // short
            0,                            // int
            calc_all_iterations_<float>,  // float
            0,                            // double
            0                             // user type
        };

        CalcAllIterationFunc func = tab[msg_type];
        if (func == 0)
            cv::gpu::error("Unsupported message type", __FILE__, __LINE__);
        func(cols, rows, iters, u, d, l, r, data, stream);
    }

///////////////////////////////////////////////////////////////
/////////////////////////// output ////////////////////////////
///////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void output(int cols, int rows, const T* u, const T* d, const T* l, const T* r, const T* data, size_t step, short* disp, size_t res_step) 
    {   
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
        {
            const T* us = u + (y + 1) * step + x;
            const T* ds = d + (y - 1) * step + x;
            const T* ls = l + y * step + (x + 1);
            const T* rs = r + y * step + (x - 1);
            const T* dt = data + y * step + x;

            size_t disp_step = rows * step;

            int best = 0;
            float best_val = FLT_MAX;
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

            disp[res_step * y + x] = saturate_cast<short>(best);
        }
    }

    typedef void (*OutputFunc)(const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data, DevMem2D disp, const cudaStream_t& stream);

    template<typename T>
    void output_(const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data, DevMem2D disp, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(disp.cols, threads.x);
        grid.y = divUp(disp.rows, threads.y);

        output<T><<<grid, threads, 0, stream>>>(disp.cols, disp.rows, (const T*)u.data, (const T*)d.data, (const T*)l.data, (const T*)r.data, (const T*)data.data, u.step/sizeof(T), (short*)disp.data, disp.step/sizeof(short));

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void output(int msg_type, const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data, DevMem2D disp, const cudaStream_t& stream)
    {            
        static OutputFunc tab[8] =
        {
            0,               // uchar
            0,               // schar
            0,               // ushort
            output_<short>,  // short
            0,               // int
            output_<float>,  // float
            0,               // double
            0                // user type
        };

        OutputFunc func = tab[msg_type];
        if (func == 0)
            cv::gpu::error("Unsupported message type", __FILE__, __LINE__);
        func(u, d, l, r, data, disp, stream);
    }

}}}