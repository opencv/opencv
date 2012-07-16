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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#if defined (__ATI__)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (__NVIDIA__)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
///////////////////////////////////////////////////////////////
/////////////////common///////////////////////////////////////
/////////////////////////////////////////////////////////////
short round_short(float v){
    return convert_short_sat_rte(v); 
}
#define FLOAT_MAX 3.402823466e+38f
typedef struct
{
    int   cndisp;
    float cmax_data_term;
    float cdata_weight;
    float cmax_disc_term;
    float cdisc_single_jump;
}con_srtuct_t;
///////////////////////////////////////////////////////////////
////////////////////////// comp data //////////////////////////
///////////////////////////////////////////////////////////////

float pix_diff_1(__global const uchar *ls, __global const uchar *rs)
{
    return abs((int)(*ls) - *rs); 
}

float pix_diff_3(__global const uchar *ls, __global const uchar *rs)
{
    const float tr = 0.299f;
    const float tg = 0.587f;
    const float tb = 0.114f;

    float val;
            
    val =  tb * abs((int)ls[0] - rs[0]);
    val += tg * abs((int)ls[1] - rs[1]);
    val += tr * abs((int)ls[2] - rs[2]);

    return val;
}
float pix_diff_4(__global const uchar *ls, __global const uchar *rs)
{
    uchar4 l, r;
    l = *((__global uchar4 *)ls);
    r = *((__global uchar4 *)rs);

    const float tr = 0.299f;
    const float tg = 0.587f;
    const float tb = 0.114f;

    float val;
             
    val  = tb * abs((int)l.x - r.x);
    val += tg * abs((int)l.y - r.y);
    val += tr * abs((int)l.z - r.z);

    return val;
}

__kernel void comp_data_0(__global uchar *left,  int left_rows,  int left_cols,  int left_step,
                          __global uchar *right, int right_step,
                          __global short  *data, int data_cols,  int data_step,
                          __constant con_srtuct_t *con_st, int cn)
                        //  int cndisp, float cmax_data_term, float cdata_weight, int cn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y > 0 && y < (left_rows - 1) && x > 0 && x < (left_cols - 1))
    {
        const __global uchar* ls = left  + y * left_step  + x * cn;
        const __global uchar* rs = right + y * right_step + x * cn;

        __global short *ds = (__global short *)((__global uchar *)data + y * data_step) + x;

        const unsigned int disp_step = data_cols * left_rows ;

        for (int disp = 0; disp < con_st -> cndisp; disp++)
        {
            if (x - disp >= 1)
            {
                float val = 0;
                if(cn == 1)
                    val = pix_diff_1(ls, rs - disp * cn);
                if(cn == 3)
                    val = pix_diff_3(ls, rs - disp * cn);
                if(cn == 4)
                    val = pix_diff_4(ls, rs - disp *cn);

                ds[disp * disp_step] =  round_short(fmin(con_st -> cdata_weight * val, 
                                                         con_st -> cdata_weight * con_st -> cmax_data_term));
            }
            else
            {
                ds[disp * disp_step] =  round_short(con_st -> cdata_weight * con_st -> cmax_data_term);
            }
        }
    }
}

__kernel void comp_data_1(__global uchar *left,  int left_rows,  int left_cols,  int left_step,
                          __global uchar *right, int right_step,
                          __global float *data,  int data_cols,  int data_step,
                          __constant con_srtuct_t *con_st, int cn)
                          //int cndisp, float cmax_data_term, float cdata_weight, int cn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y > 0 && y < left_rows - 1 && x > 0 && x < left_cols - 1)
    {
        const __global uchar* ls = left  + y * left_step  + x * cn;
        const __global uchar* rs = right + y * right_step + x * cn;

        __global float *ds = (__global float *)((__global char *)data + y * data_step) + x;

        const unsigned int disp_step = data_cols * left_rows;

        for (int disp = 0; disp < con_st -> cndisp; disp++)
        {
            if (x - disp >= 1)
            {
                float val = 0;
                if(cn == 1)
                    val = pix_diff_1(ls, rs - disp * cn);
                if(cn == 3)
                    val = pix_diff_3(ls, rs - disp * cn);
                if(cn == 4)
                    val = pix_diff_4(ls, rs - disp *cn);

                ds[disp * disp_step] = fmin(con_st -> cdata_weight * val, 
                                            con_st -> cdata_weight * con_st -> cmax_data_term);
            }
            else
            {
                ds[disp * disp_step] = con_st -> cdata_weight * con_st -> cmax_data_term;
            }
        }
    }
}

///////////////////////////////////////////////////////////////
//////////////////////// data step down ///////////////////////
///////////////////////////////////////////////////////////////
__kernel void data_step_down_0(__global short *src, int src_rows, int src_cols, 
                               __global short *dst, int dst_rows, int dst_cols, int dst_real_cols, 
                               int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);;
   
    if (x < dst_cols && y < dst_rows)
    {
        for (int d = 0; d < cndisp; ++d)
        {
            //float dst_reg  = src.ptr(d * src_rows + (2*y+0))[(2*x+0)];
            float dst_reg;
            dst_reg  = src[(d * src_rows + (2*y+0)) * src_cols + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_cols + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+0)) * src_cols + 2*x+1];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_cols + 2*x+1];
              
            //dst.ptr(d * dst_rows + y)[x] = saturate_cast<T>(dst_reg);
            dst[(d * dst_rows + y) * dst_real_cols + x] = round_short(dst_reg);
        }
    }
}
__kernel void data_step_down_1(__global float *src, int src_rows, int src_cols,
                               __global float *dst, int dst_rows, int dst_cols, int dst_real_cols, 
                               int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);;
   
    if (x < dst_cols && y < dst_rows)
    {
        for (int d = 0; d < cndisp; ++d)
        {
            //float dst_reg  = src.ptr(d * src_rows + (2*y+0))[(2*x+0)];
            float dst_reg;
            dst_reg = src[(d * src_rows + (2*y+0)) * src_cols + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_cols + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+0)) * src_cols + 2*x+1];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_cols + 2*x+1];
              
            //dst.ptr(d * dst_rows + y)[x] = saturate_cast<T>(dst_reg);
            dst[(d * dst_rows + y) * dst_real_cols + x] = round_short(dst_reg);
        }
    }
}

///////////////////////////////////////////////////////////////
/////////////////// level up messages  ////////////////////////
///////////////////////////////////////////////////////////////
__kernel void level_up_message_0(__global short *src, int src_rows, int src_step,
                                 __global short *dst, int dst_rows, int dst_cols, int dst_step,
                                 int cndisp)
    
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x < dst_cols && y < dst_rows)
    {
        const int dst_disp_step = (dst_step / sizeof(short)) * dst_rows;
        const int src_disp_step = (src_step / sizeof(short)) * src_rows;
       
        __global short        *dstr = (__global short *)((__global char *)dst + y   * dst_step) + x;
        __global const short  *srcr = (__global short *)((__global char *)src + y/2 * src_step) + x/2;
       
        for (int d = 0; d < cndisp; ++d)
            dstr[d * dst_disp_step] = srcr[d * src_disp_step];
    }
}
__kernel void level_up_message_1(__global float *src, int src_rows, int src_step,
                                 __global float *dst, int dst_rows, int dst_cols, int dst_step,
                                 int cndisp)
    
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x < dst_cols && y < dst_rows)
    {
        const int dst_disp_step = (dst_step/sizeof(float)) * dst_rows;
        const int src_disp_step = (src_step/sizeof(float)) * src_rows;
       
        __global float       *dstr = (__global float *)((__global char *)dst + y   * dst_step) + x;
        __global const float *srcr = (__global float *)((__global char *)src + y/2 * src_step) + x/2;
       
        for (int d = 0; d < cndisp; ++d)
            dstr[d * dst_disp_step] = srcr[d * src_disp_step];
    }
}

///////////////////////////////////////////////////////////////
////////////////////  calc all iterations /////////////////////
///////////////////////////////////////////////////////////////
void calc_min_linear_penalty_0(__global short * dst, int disp_step, 
                               int cndisp, float cdisc_single_jump)
{
    float prev = dst[0];
    float cur;

    for (int disp = 1; disp < cndisp; ++disp)
    {
        prev += cdisc_single_jump;
        cur = dst[disp_step * disp];

        if (prev < cur)
        {
            cur = prev;
            dst[disp_step * disp] = round_short(prev);
        }
            
        prev = cur;
    }
        
    prev = dst[(cndisp - 1) * disp_step];
    for (int disp = cndisp - 2; disp >= 0; disp--)
    {
        prev += cdisc_single_jump;
        cur = dst[disp_step * disp];
       
        if (prev < cur)
        {
             cur = prev;
             dst[disp_step * disp] = round_short(prev);
        }
        prev = cur;
    }
}
void message_0(const __global short *msg1, const __global short *msg2,
               const __global short *msg3, const __global short *data, __global short *dst,
               int msg_disp_step, int data_disp_step, int cndisp, float cmax_disc_term, float cdisc_single_jump)
{
    float minimum = FLOAT_MAX;
        
    for(int i = 0; i < cndisp; ++i)
    {
        float dst_reg;
        dst_reg  = msg1[msg_disp_step * i];
        dst_reg += msg2[msg_disp_step * i];
        dst_reg += msg3[msg_disp_step * i];
        dst_reg += data[data_disp_step * i];
       
        if (dst_reg < minimum)
            minimum = dst_reg;
           
        dst[msg_disp_step * i] = round_short(dst_reg);
    }
       
    calc_min_linear_penalty_0(dst, msg_disp_step, cndisp, cdisc_single_jump);
        
    minimum += cmax_disc_term;

    float sum = 0;
    for(int i = 0; i < cndisp; ++i)
    {
        float dst_reg = dst[msg_disp_step * i];
        if (dst_reg > minimum)
        {
            dst_reg = minimum;
            dst[msg_disp_step * i] = round_short(minimum);
        }
        sum += dst_reg;
    }
    sum /= cndisp;
        
    for(int i = 0; i < cndisp; ++i)
        dst[msg_disp_step * i] -= sum;
}
__kernel void one_iteration_0(__global short *u,    int u_step,    int u_cols,
                              __global short *data, int data_step, int data_cols,
                              __global short *d,    __global short *l, __global short *r,
                              int t, int cols, int rows, 
                              int cndisp, float cmax_disc_term, float cdisc_single_jump)
{
    const int y = get_global_id(1);
    const int x = ((get_global_id(0)) << 1) + ((y + t) & 1);
    
    if ((y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
    {
        __global short *us = (__global short *)((__global char *)u + y * u_step) + x;
        __global short *ds = d + y * u_cols + x;
        __global short *ls = l + y * u_cols + x;
        __global short *rs = r + y * u_cols + x;
        const __global  short *dt = (__global short *)((__global char *)data + y * data_step) + x;

        int msg_disp_step = u_cols * rows;
        int data_disp_step = data_cols * rows;

        message_0(us + u_cols, ls      + 1, rs - 1, dt, us, msg_disp_step, data_disp_step, cndisp, 
                cmax_disc_term, cdisc_single_jump);
        message_0(ds - u_cols, ls      + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);

        message_0(us + u_cols, ds - u_cols, rs - 1, dt, rs, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);
        message_0(us + u_cols, ds - u_cols, ls + 1, dt, ls, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);
    }
}
void calc_min_linear_penalty_1(__global float * dst, int step, 
                               int cndisp, float cdisc_single_jump)
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
            dst[step * disp] = prev;
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
             dst[step * disp] = prev;
        }
        prev = cur;
    }
}
void message_1(const __global float *msg1, const __global float *msg2,
               const __global float *msg3, const __global float *data, __global float *dst,
               int msg_disp_step, int data_disp_step, int cndisp, float cmax_disc_term, float cdisc_single_jump)
{
    float minimum = FLOAT_MAX; 
        
    for(int i = 0; i < cndisp; ++i)
    {
        float dst_reg = 0;
        dst_reg  = msg1[msg_disp_step * i];
        dst_reg += msg2[msg_disp_step * i];
        dst_reg += msg3[msg_disp_step * i];
        dst_reg += data[data_disp_step * i];
       
        if (dst_reg < minimum)
            minimum = dst_reg;
           
        dst[msg_disp_step * i] = dst_reg;
    }
       
    calc_min_linear_penalty_1(dst, msg_disp_step, cndisp, cdisc_single_jump);
        
    minimum += cmax_disc_term;

    float sum = 0;
    for(int i = 0; i < cndisp; ++i)
    {
        float dst_reg = dst[msg_disp_step * i];
        if (dst_reg > minimum)
        {
            dst_reg = minimum;
            dst[msg_disp_step * i] = minimum;
        }
        sum += dst_reg;
    }
    sum /= cndisp;
        
    for(int i = 0; i < cndisp; ++i)
        dst[msg_disp_step * i] -= sum;
}
__kernel void one_iteration_1(__global float *u,    int u_step,    int u_cols,
                              __global float *data, int data_step, int data_cols,
                              __global float *d,    __global float *l, __global float *r,
                              int t, int cols, int rows, 
                              int cndisp,float cmax_disc_term, float cdisc_single_jump)
{
    const int y = get_global_id(1);
    const int x = ((get_global_id(0)) << 1) + ((y + t) & 1);
    
    if ((y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
    {
        __global float* us = (__global float *)((__global char *)u + y * u_step) + x;
        __global float* ds = d + y * u_cols + x;
        __global float* ls = l + y * u_cols + x;
        __global float* rs = r + y * u_cols + x;
        const __global float* dt = (__global float *)((__global char *)data + y * data_step) + x;

        int msg_disp_step = u_cols * rows;
        int data_disp_step = data_cols * rows;

        message_1(us + u_cols, ls      + 1, rs - 1, dt, us, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);
        message_1(ds - u_cols, ls      + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step, cndisp, 
                cmax_disc_term, cdisc_single_jump);
        message_1(us + u_cols, ds - u_cols, rs - 1, dt, rs, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);
        message_1(us + u_cols, ds - u_cols, ls + 1, dt, ls, msg_disp_step, data_disp_step, cndisp,
                cmax_disc_term, cdisc_single_jump);
    }
}

///////////////////////////////////////////////////////////////
/////////////////////////// output ////////////////////////////
///////////////////////////////////////////////////////////////
__kernel void output_0(const __global short *u, int u_step, int u_cols,
                       const __global short *d, const __global short *l,
                       const __global short *r, const __global short *data,
                       __global short *disp, int disp_rows, int disp_cols, int disp_step,
                       int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
   
    if (y > 0 && y < disp_rows - 1 && x > 0 && x < disp_cols - 1)
    {
        const __global short *us =(__global short *)((__global char *)u + (y + 1) * u_step) + x;
        const __global short *ds = d + (y - 1) * u_cols + x;
        const __global short *ls = l + y * u_cols + (x + 1);
        const __global short *rs = r + y * u_cols + (x - 1);
        const __global short *dt = data + y * u_cols + x;
       
        int disp_steps = disp_rows * u_cols;

        int best = 0;
        float best_val = FLOAT_MAX;
        for (int d = 0; d < cndisp; ++d)
        {
            float val;
            val  = us[d * disp_steps];
            val += ds[d * disp_steps];
            val += ls[d * disp_steps];
            val += rs[d * disp_steps];
            val += dt[d * disp_steps];
           
            if (val < best_val)
            {
                best_val = val;
                best = d;
            }
        }
            
        ((__global short *)((__global char *)disp + y * disp_step))[x] = convert_short_sat(best);
    }
}
__kernel void output_1(const __global float *u, int u_step, int u_cols,
                       const __global float *d, const __global float *l,
                       const __global float *r, const __global float *data,
                       __global short *disp, int disp_rows, int disp_cols, int disp_step,
                       int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
   
    if (y > 0 && y < disp_rows - 1 && x > 0 && x < disp_cols - 1)
    {
        const __global float *us =(__global float *)((__global char *)u + (y + 1) * u_step) + x;
        const __global float *ds = d + (y - 1) * u_cols + x;
        const __global float *ls = l + y * u_cols + (x + 1);
        const __global float *rs = r + y * u_cols + (x - 1);
        const __global float *dt = data + y * u_cols + x;
       
        int disp_steps = disp_rows * u_cols;
       
        int best = 0;
        float best_val = FLOAT_MAX;
        for (int d = 0; d < cndisp; ++d)
        {
            float val;
            val  = us[d * disp_steps];
            val += ds[d * disp_steps];
            val += ls[d * disp_steps];
            val += rs[d * disp_steps];
            val += dt[d * disp_steps];
           
            if (val < best_val)
            {
                best_val = val;
                best = d;
            }
        }
            
        //disp[y * disp_cols + x] = convert_short_sat(best);
        ((__global short *)((__global char *)disp + y * disp_step))[x] = convert_short_sat(best);
    }
}
