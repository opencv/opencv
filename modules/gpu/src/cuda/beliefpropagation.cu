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

#include "opencv2/gpu/devmem2d.hpp"
#include "safe_call.hpp"

using namespace cv::gpu;

static inline int divUp(int a, int b) { return (a % b == 0) ? a/b : a/b + 1; }

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

typedef unsigned char uchar;

namespace beliefpropagation_gpu
{      
    __constant__ int   cndisp;
    __constant__ float cdisc_cost;
    __constant__ float cdata_cost;
    __constant__ float clambda;
};

///////////////////////////////////////////////////////////////
//////////////////  comp data /////////////////////////////////
///////////////////////////////////////////////////////////////

namespace beliefpropagation_gpu
{
    __global__ void comp_data_kernel(uchar* l, uchar* r, size_t step, float* data, size_t data_step, int cols, int rows) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
        {
            uchar *ls = l + y * step + x; 
            uchar *rs = r + y * step + x; 

            float *ds = data + y * data_step + x;
            size_t disp_step = data_step * rows;

            for (int disp = 0; disp < cndisp; disp++) 
            {
                if (x - disp >= 0)
                {
                    int le = ls[0];
                    int re = rs[-disp];
                    float val = abs(le - re);
                    
                    ds[disp * disp_step] = clambda * fmin(val, cdata_cost);
                }
                else
                {
                    ds[disp * disp_step] = cdata_cost;
                }
            }
        }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void load_constants(int ndisp, float disc_cost, float data_cost, float lambda)
    {
        cudaSafeCall( cudaMemcpyToSymbol(beliefpropagation_gpu::cndisp, &ndisp, sizeof(ndisp)) );
        cudaSafeCall( cudaMemcpyToSymbol(beliefpropagation_gpu::cdisc_cost, &disc_cost, sizeof(disc_cost)) );
        cudaSafeCall( cudaMemcpyToSymbol(beliefpropagation_gpu::cdata_cost, &data_cost, sizeof(data_cost)) );
        cudaSafeCall( cudaMemcpyToSymbol(beliefpropagation_gpu::clambda, &lambda, sizeof(lambda)) );        
    }

    extern "C" void comp_data_caller(const DevMem2D& l, const DevMem2D& r, DevMem2D_<float> mdata)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(l.cols, threads.x);
        grid.y = divUp(l.rows, threads.y);

        beliefpropagation_gpu::comp_data_kernel<<<grid, threads>>>(l.ptr, r.ptr, l.step, mdata.ptr, mdata.step/sizeof(float), l.cols, l.rows);
        cudaSafeCall( cudaThreadSynchronize() );
    }
}}}

///////////////////////////////////////////////////////////////
//////////////////  data_step_down ////////////////////////////
///////////////////////////////////////////////////////////////

namespace beliefpropagation_gpu
{    
    __global__ void data_down_kernel(int dst_cols, int dst_rows, int src_rows, float *src, size_t src_step, float *dst, size_t dst_step)
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

                dst[d * dst_disp_step + y * dst_step + x] = dst_reg;
            }
        }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void data_down_kernel_caller(int dst_cols, int dst_rows, int src_rows, const DevMem2D_<float>& src, DevMem2D_<float> dst)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(dst_cols, threads.x);
        grid.y = divUp(dst_rows, threads.y);

        beliefpropagation_gpu::data_down_kernel<<<grid, threads>>>(dst_cols, dst_rows, src_rows, src.ptr, src.step/sizeof(float), dst.ptr, dst.step/sizeof(float));
        cudaSafeCall( cudaThreadSynchronize() );
    }
}}}

///////////////////////////////////////////////////////////////
//////////////////  level up messages  ////////////////////////
///////////////////////////////////////////////////////////////


namespace beliefpropagation_gpu
{    
    __global__ void level_up_kernel(int dst_cols, int dst_rows, int src_rows, float *src, size_t src_step, float *dst, size_t dst_step)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;        

        if (x < dst_cols && y < dst_rows)
        {
            const size_t dst_disp_step = dst_step * dst_rows;
            const size_t src_disp_step = src_step * src_rows;

            float *dstr = dst + y   * dst_step + x;
            float *srcr = src + y/2 * src_step + x/2;

            for (int d = 0; d < cndisp; ++d)            
                dstr[d * dst_disp_step] = srcr[d * src_disp_step];
        }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void level_up(int dst_idx, int dst_cols, int dst_rows, int src_rows, DevMem2D_<float>* mu, DevMem2D_<float>* md, DevMem2D_<float>* ml, DevMem2D_<float>* mr)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(dst_cols, threads.x);
        grid.y = divUp(dst_rows, threads.y);

        int src_idx = (dst_idx + 1) & 1;

        beliefpropagation_gpu::level_up_kernel<<<grid, threads>>>(dst_cols, dst_rows, src_rows, mu[src_idx].ptr, mu[src_idx].step/sizeof(float), mu[dst_idx].ptr, mu[dst_idx].step/sizeof(float));
        beliefpropagation_gpu::level_up_kernel<<<grid, threads>>>(dst_cols, dst_rows, src_rows, md[src_idx].ptr, md[src_idx].step/sizeof(float), md[dst_idx].ptr, md[dst_idx].step/sizeof(float));
        beliefpropagation_gpu::level_up_kernel<<<grid, threads>>>(dst_cols, dst_rows, src_rows, ml[src_idx].ptr, ml[src_idx].step/sizeof(float), ml[dst_idx].ptr, ml[dst_idx].step/sizeof(float));
        beliefpropagation_gpu::level_up_kernel<<<grid, threads>>>(dst_cols, dst_rows, src_rows, mr[src_idx].ptr, mr[src_idx].step/sizeof(float), mr[dst_idx].ptr, mr[dst_idx].step/sizeof(float));

        cudaSafeCall( cudaThreadSynchronize() );
    }
}}}


///////////////////////////////////////////////////////////////
/////////////////  Calcs all iterations ///////////////////////
///////////////////////////////////////////////////////////////


namespace beliefpropagation_gpu
{
    __device__ void calc_min_linear_penalty(float *dst, size_t step)
    {
        float prev = dst[0];
        float cur;
        for (int disp = 1; disp < cndisp; ++disp) 
        {
            prev += 1.0f;
            cur = dst[step * disp];
            if (prev < cur)
                cur = prev;
            dst[step * disp] = prev = cur;
        }

        prev = dst[(cndisp - 1) * step];
        for (int disp = cndisp - 2; disp >= 0; disp--)     
        {
            prev += 1.0f;
            cur = dst[step * disp];
            if (prev < cur)
                cur = prev;
            dst[step * disp] = prev = cur;      
        }
    }

    __device__ void message(float *msg1, float *msg2, float *msg3, float *data, float *dst, size_t msg_disp_step, size_t data_disp_step)
    {
        float minimum = FLT_MAX;

        for(int i = 0; i < cndisp; ++i)
        {
            float dst_reg = msg1[msg_disp_step * i] + msg2[msg_disp_step * i] + msg3[msg_disp_step * i] + data[data_disp_step * i];

            if (dst_reg < minimum)
                minimum = dst_reg;

            dst[msg_disp_step * i] = dst_reg;

        }

        calc_min_linear_penalty(dst, msg_disp_step);

        minimum += cdisc_cost;

        float sum = 0;
        for(int i = 0; i < cndisp; ++i)
        {
            float dst_reg = dst[msg_disp_step * i];
            if (dst_reg > minimum)
            {
                dst[msg_disp_step * i] = dst_reg = minimum;          
            }
            sum += dst_reg;
        }    
        sum /= cndisp;

        for(int i = 0; i < cndisp; ++i)
            dst[msg_disp_step * i] -= sum;
    }

    __global__ void one_iteration(int t, float* u, float *d, float *l, float *r, size_t msg_step, float *data, size_t data_step, int cols, int rows)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + t) & 1);

        if ( (y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
        {
            float *us = u + y * msg_step + x;
            float *ds = d + y * msg_step + x;
            float *ls = l + y * msg_step + x;
            float *rs = r + y * msg_step + x;
            float *dt = data + y * data_step + x;
            size_t msg_disp_step = msg_step * rows;
            size_t data_disp_step = data_step * rows;

            message(us + msg_step, ls        + 1, rs - 1, dt, us, msg_disp_step, data_disp_step);
            message(ds - msg_step, ls        + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step);
            message(us + msg_step, ds - msg_step, rs - 1, dt, rs, msg_disp_step, data_disp_step);
            message(us + msg_step, ds - msg_step, ls + 1, dt, ls, msg_disp_step, data_disp_step);                
        }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void call_all_iterations(int cols, int rows, int iters, DevMem2D_<float>& u, DevMem2D_<float>& d, DevMem2D_<float>& l, DevMem2D_<float>& r, const DevMem2D_<float>& data)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(cols, threads.x << 1);
        grid.y = divUp(rows, threads.y);

        for(int t = 0; t < iters; ++t)
            beliefpropagation_gpu::one_iteration<<<grid, threads>>>(t, u.ptr, d.ptr, l.ptr, r.ptr, u.step/sizeof(float), data.ptr, data.step/sizeof(float), cols, rows);        

        cudaSafeCall( cudaThreadSynchronize() );
    }
}}}


///////////////////////////////////////////////////////////////
//////////////////  Output caller /////////////////////////////
///////////////////////////////////////////////////////////////

namespace beliefpropagation_gpu
{  
    __global__ void output(int cols, int rows, float *u, float *d, float *l, float *r, float* data, size_t step, unsigned char *disp, size_t res_step) 
    {   
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1)
            if (x > 0 && x < cols - 1)
            {
                float *us = u + (y + 1) * step + x;
                float *ds = d + (y - 1) * step + x;
                float *ls = l + y * step + (x + 1);
                float *rs = r + y * step + (x - 1);
                float *dt = data + y * step + x;

                size_t disp_step = rows * step;

                int best = 0;
                float best_val = FLT_MAX;
                for (int d = 0; d < cndisp; ++d) 
                {
                    float val = us[d * disp_step] + ds[d * disp_step] + ls[d * disp_step] + rs[d * disp_step] + dt[d * disp_step];

                    if (val < best_val) 
                    {
                        best_val = val;
                        best = d;
                    }
                }

                disp[res_step * y + x] = best & 0xFF;                           
            }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void output_caller(const DevMem2D_<float>& u, const DevMem2D_<float>& d, const DevMem2D_<float>& l, const DevMem2D_<float>& r, const DevMem2D_<float>& data, DevMem2D disp)
    {    
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(disp.cols, threads.x);
        grid.y = divUp(disp.rows, threads.y);

        beliefpropagation_gpu::output<<<grid, threads>>>(disp.cols, disp.rows, u.ptr, d.ptr, l.ptr, r.ptr, data.ptr, u.step/sizeof(float), disp.ptr, disp.step);
        cudaSafeCall( cudaThreadSynchronize() );
    }
}}}