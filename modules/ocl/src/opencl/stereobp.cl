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

#if defined (DOUBLE_SUPPORT)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif

#endif

#ifdef T_FLOAT
#define T float
#else
#define T short
#endif

///////////////////////////////////////////////////////////////
/////////////////common///////////////////////////////////////
/////////////////////////////////////////////////////////////
T saturate_cast(float v){
#ifdef T_SHORT
    return convert_short_sat_rte(v);
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


#ifndef CN
#define CN 4
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

        for (int disp = 0; disp < con_st -> cndisp; disp++)
        {
            if (x - disp >= 1)
            {
                float val = 0;
                val = PIX_DIFF(ls, rs - disp * CN);
                ds[disp * disp_step] =  saturate_cast(fmin(con_st -> cdata_weight * val,
                    con_st -> cdata_weight * con_st -> cmax_data_term));
            }
            else
            {
                ds[disp * disp_step] =  saturate_cast(con_st -> cdata_weight * con_st -> cmax_data_term);
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
        for (int d = 0; d < cndisp; ++d)
        {
            float dst_reg;
            dst_reg  = src[(d * src_rows + (2*y+0)) * src_step + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_step + 2*x+0];
            dst_reg += src[(d * src_rows + (2*y+0)) * src_step + 2*x+1];
            dst_reg += src[(d * src_rows + (2*y+1)) * src_step + 2*x+1];

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
void calc_min_linear_penalty(__global T * dst, int disp_step,
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
            dst[disp_step * disp] = saturate_cast(prev);
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
            dst[disp_step * disp] = saturate_cast(prev);
        }
        prev = cur;
    }
}
void message(const __global T *msg1, const __global T *msg2,
             const __global T *msg3, const __global T *data, __global T *dst,
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

        dst[msg_disp_step * i] = saturate_cast(dst_reg);
    }

    calc_min_linear_penalty(dst, msg_disp_step, cndisp, cdisc_single_jump);

    minimum += cmax_disc_term;

    float sum = 0;
    for(int i = 0; i < cndisp; ++i)
    {
        float dst_reg = dst[msg_disp_step * i];
        if (dst_reg > minimum)
        {
            dst_reg = minimum;
            dst[msg_disp_step * i] = saturate_cast(minimum);
        }
        sum += dst_reg;
    }
    sum /= cndisp;

    for(int i = 0; i < cndisp; ++i)
        dst[msg_disp_step * i] -= sum;
}
__kernel void one_iteration(__global T *u,    int u_step,
                            __global T *data, int data_step,
                            __global T *d,    __global T *l, __global T *r,
                            int t, int cols, int rows,
                            int cndisp, float cmax_disc_term, float cdisc_single_jump)
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

        message(us + u_step, ls      + 1, rs - 1, dt, us, msg_disp_step, data_disp_step, cndisp,
            cmax_disc_term, cdisc_single_jump);
        message(ds - u_step, ls      + 1, rs - 1, dt, ds, msg_disp_step, data_disp_step, cndisp,
            cmax_disc_term, cdisc_single_jump);

        message(us + u_step, ds - u_step, rs - 1, dt, rs, msg_disp_step, data_disp_step, cndisp,
            cmax_disc_term, cdisc_single_jump);
        message(us + u_step, ds - u_step, ls + 1, dt, ls, msg_disp_step, data_disp_step, cndisp,
            cmax_disc_term, cdisc_single_jump);
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
