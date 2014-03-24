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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Peng Xiao,   pengxiao@outlook.com
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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef T_FLOAT
#define T float
#define T4 float4
#else
#define T short
#define T4 short4
#endif

///////////////////////////////////////////////////////////////
/////////////////common///////////////////////////////////////
/////////////////////////////////////////////////////////////
inline T saturate_cast(float v){
#ifdef T_SHORT
    return convert_short_sat_rte(v);
#else
    return v;
#endif
}

inline T4 saturate_cast4(float4 v){
#ifdef T_SHORT
    return convert_short4_sat_rte(v);
#else
    return v;
#endif
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

inline float pix_diff_1(const uchar4 l, __global const uchar *rs)
{
    return abs((int)(l.x) - *rs);
}

inline float pix_diff_4(const uchar4 l, __global const uchar *rs)
{
    uchar4 r;
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

inline float pix_diff_3(const uchar4 l, __global const uchar *rs)
{
    return pix_diff_4(l, rs);
}

#ifndef CN
#define CN 4
#endif

#ifndef CNDISP
#define CNDISP 64
#endif

#define CAT(X,Y) X##Y
#define CAT2(X,Y) CAT(X,Y)

#define PIX_DIFF CAT2(pix_diff_, CN)

__kernel void comp_data(__global uchar *left,  int left_rows,  int left_cols,  int left_step,
                        __global uchar *right, int right_step,
                        __global T *data, int data_step,
                        __constant con_srtuct_t *con_st)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y > 0 && y < (left_rows - 1) && x > 0 && x < (left_cols - 1))
    {
        data_step /= sizeof(T);
        const __global uchar* ls = left  + y * left_step  + x * CN;
        const __global uchar* rs = right + y * right_step + x * CN;

        __global T *ds = data + y * data_step + x;

        const unsigned int disp_step = data_step * left_rows;
        const float weightXterm = con_st -> cdata_weight * con_st -> cmax_data_term;
        const uchar4 ls_data = vload4(0, ls);

        for (int disp = 0; disp < con_st -> cndisp; disp++)
        {
            if (x - disp >= 1)
            {
                float val = 0;
                val = PIX_DIFF(ls_data, rs - disp * CN);
                ds[disp * disp_step] =  saturate_cast(fmin(con_st -> cdata_weight * val, weightXterm));
            }
            else
            {
                ds[disp * disp_step] =  saturate_cast(weightXterm);
            }
        }
    }
}

///////////////////////////////////////////////////////////////
//////////////////////// data step down ///////////////////////
///////////////////////////////////////////////////////////////
__kernel void data_step_down(__global T *src, int src_rows,
                             __global T *dst, int dst_rows, int dst_cols,
                             int src_step, int dst_step,
                             int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_step /= sizeof(T);
        dst_step /= sizeof(T);
        int4 coor_step = (int4)(src_rows * src_step);
        int4 coor = (int4)(min(2*y+0, src_rows-1) * src_step + 2*x+0,
                           min(2*y+1, src_rows-1) * src_step + 2*x+0,
                           min(2*y+0, src_rows-1) * src_step + 2*x+1,
                           min(2*y+1, src_rows-1) * src_step + 2*x+1);

        for (int d = 0; d < cndisp; ++d)
        {
            float dst_reg;
            dst_reg  = src[coor.x];
            dst_reg += src[coor.y];
            dst_reg += src[coor.z];
            dst_reg += src[coor.w];
            coor += coor_step;

            dst[(d * dst_rows + y) * dst_step + x] = saturate_cast(dst_reg);
        }
    }
}

///////////////////////////////////////////////////////////////
/////////////////// level up messages  ////////////////////////
///////////////////////////////////////////////////////////////
__kernel void level_up_message(__global T *src, int src_rows, int src_step,
                               __global T *dst, int dst_rows, int dst_cols, int dst_step,
                               int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_step /= sizeof(T);
        dst_step /= sizeof(T);

        const int dst_disp_step = dst_step * dst_rows;
        const int src_disp_step = src_step * src_rows;

        __global T       *dstr = dst + y * dst_step + x;
        __global const T *srcr = src + (y / 2 * src_step) + (x / 2);

        for (int d = 0; d < cndisp; ++d)
            dstr[d * dst_disp_step] = srcr[d * src_disp_step];
    }
}

///////////////////////////////////////////////////////////////
////////////////////  calc all iterations /////////////////////
///////////////////////////////////////////////////////////////
inline void message(__global T *us_, __global T *ds_, __global T *ls_, __global T *rs_,
              const __global T *dt,
              int u_step, int msg_disp_step, int data_disp_step,
              float4 cmax_disc_term, float4 cdisc_single_jump)
{
    __global T *us = us_ + u_step;
    __global T *ds = ds_ - u_step;
    __global T *ls = ls_ + 1;
    __global T *rs = rs_ - 1;

    float4 minimum = (float4)(FLOAT_MAX);

    T4 t_dst[CNDISP];
    float4 dst_reg;
    float4 prev;
    float4 cur;

    T t_us = us[0];
    T t_ds = ds[0];
    T t_ls = ls[0];
    T t_rs = rs[0];
    T t_dt = dt[0];

    prev = (float4)(t_us + t_ls + t_rs + t_dt,
                    t_ds + t_ls + t_rs + t_dt,
                    t_us + t_ds + t_rs + t_dt,
                    t_us + t_ds + t_ls + t_dt);

    minimum = min(prev, minimum);

    t_dst[0] = saturate_cast4(prev);

    for(int i = 1, idx = msg_disp_step; i < CNDISP; ++i, idx+=msg_disp_step)
    {
        t_us = us[idx];
        t_ds = ds[idx];
        t_ls = ls[idx];
        t_rs = rs[idx];
        t_dt = dt[data_disp_step * i];

        dst_reg = (float4)(t_us + t_ls + t_rs + t_dt,
                           t_ds + t_ls + t_rs + t_dt,
                           t_us + t_ds + t_rs + t_dt,
                           t_us + t_ds + t_ls + t_dt);

        minimum = min(dst_reg, minimum);

        prev += cdisc_single_jump;
        prev = min(prev, dst_reg);

        t_dst[i] = saturate_cast4(prev);
    }

    minimum += cmax_disc_term;

    float4 sum = (float4)(0);
    prev = convert_float4(t_dst[CNDISP - 1]);
    for (int disp = CNDISP - 2; disp >= 0; disp--)
    {
        prev += cdisc_single_jump;
        cur = convert_float4(t_dst[disp]);
        prev = min(prev, cur);
        cur = min(prev, minimum);
        sum += cur;

        t_dst[disp] = saturate_cast4(cur);
    }

    dst_reg = convert_float4(t_dst[CNDISP - 1]);
    dst_reg = min(dst_reg, minimum);
    t_dst[CNDISP - 1] = saturate_cast4(dst_reg);
    sum += dst_reg;

    sum /= (float4)(CNDISP);
#pragma unroll
    for(int i = 0, idx = 0; i < CNDISP; ++i, idx+=msg_disp_step)
    {
        T4 dst = t_dst[i];
        us_[idx] = dst.x - sum.x;
        ds_[idx] = dst.y - sum.y;
        rs_[idx] = dst.z - sum.z;
        ls_[idx] = dst.w - sum.w;
    }
}
__kernel void one_iteration(__global T *u,    int u_step,
                            __global T *data, int data_step,
                            __global T *d,    __global T *l, __global T *r,
                            int t, int cols, int rows,
                            float cmax_disc_term, float cdisc_single_jump)
{
    const int y = get_global_id(1);
    const int x = ((get_global_id(0)) << 1) + ((y + t) & 1);

    if ((y > 0) && (y < rows - 1) && (x > 0) && (x < cols - 1))
    {
        u_step    /= sizeof(T);
        data_step /= sizeof(T);

        __global T *us = u + y * u_step + x;
        __global T *ds = d + y * u_step + x;
        __global T *ls = l + y * u_step + x;
        __global T *rs = r + y * u_step + x;
        const __global  T *dt = data + y * data_step + x;

        int msg_disp_step = u_step * rows;
        int data_disp_step = data_step * rows;

        message(us, ds, ls, rs, dt,
                u_step, msg_disp_step, data_disp_step,
                (float4)(cmax_disc_term), (float4)(cdisc_single_jump));
    }
}

///////////////////////////////////////////////////////////////
/////////////////////////// output ////////////////////////////
///////////////////////////////////////////////////////////////
__kernel void output(const __global T *u, int u_step,
                     const __global T *d, const __global T *l,
                     const __global T *r, const __global T *data,
                     __global T *disp, int disp_rows, int disp_cols, int disp_step,
                     int cndisp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y > 0 && y < disp_rows - 1 && x > 0 && x < disp_cols - 1)
    {
        u_step    /= sizeof(T);
        disp_step /= sizeof(T);
        const __global T *us = u + (y + 1) * u_step + x;
        const __global T *ds = d + (y - 1) * u_step + x;
        const __global T *ls = l + y * u_step + (x + 1);
        const __global T *rs = r + y * u_step + (x - 1);
        const __global T *dt = data + y * u_step + x;

        int disp_steps = disp_rows * u_step;

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

        (disp + y * disp_step)[x] = convert_short_sat(best);
    }
}
