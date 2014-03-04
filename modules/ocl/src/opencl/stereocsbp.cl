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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010,2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Jin Ma, jin@multicorewareinc.com
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

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////get_first_k_initial_global//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

__kernel void get_first_k_initial_global_0(__global short *data_cost_selected_, __global short *selected_disp_pyr,
    __global short *ctemp, int h, int w, int nr_plane,
    int cmsg_step1, int cdisp_step1, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        __global short *selected_disparity = selected_disp_pyr      + y * cmsg_step1 + x;
        __global short *data_cost_selected = data_cost_selected_    + y * cmsg_step1 + x;
        __global short *data_cost          = ctemp + y * cmsg_step1 + x;

        for(int i = 0; i < nr_plane; i++)
        {
            short minimum = SHRT_MAX;
            int id = 0;

            for(int d = 0; d < cndisp; d++)
            {
                short cur = data_cost[d * cdisp_step1];
                if(cur < minimum)
                {
                    minimum = cur;
                    id = d;
                }
            }

            data_cost_selected[i  * cdisp_step1] = minimum;
            selected_disparity[i  * cdisp_step1] = id;
            data_cost         [id * cdisp_step1] = SHRT_MAX;
        }
    }
}

__kernel void get_first_k_initial_global_1(__global  float *data_cost_selected_, __global float *selected_disp_pyr,
    __global  float *ctemp, int h, int w, int nr_plane,
    int cmsg_step1, int cdisp_step1, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        __global   float *selected_disparity = selected_disp_pyr      + y * cmsg_step1 + x;
        __global   float *data_cost_selected = data_cost_selected_    + y * cmsg_step1 + x;
        __global   float *data_cost          = ctemp + y * cmsg_step1 + x;

        for(int i = 0; i < nr_plane; i++)
        {
            float minimum = FLT_MAX;
            int id = 0;

            for(int d = 0; d < cndisp; d++)
            {
                float cur = data_cost[d * cdisp_step1];
                if(cur < minimum)
                {
                    minimum = cur;
                    id = d;
                }
            }

            data_cost_selected[i  * cdisp_step1] = minimum;
            selected_disparity[i  * cdisp_step1] = id;
            data_cost         [id * cdisp_step1] = FLT_MAX;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////get_first_k_initial_local////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void get_first_k_initial_local_0(__global  short *data_cost_selected_, __global short *selected_disp_pyr,
    __global  short *ctemp,int h, int w, int nr_plane,
    int cmsg_step1, int cdisp_step1, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        __global short *selected_disparity = selected_disp_pyr   + y * cmsg_step1 + x;
        __global short *data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
        __global short *data_cost = ctemp + y * cmsg_step1 + x;

        int nr_local_minimum = 0;

        short prev = data_cost[0 * cdisp_step1];
        short cur  = data_cost[1 * cdisp_step1];
        short next = data_cost[2 * cdisp_step1];

        for (int d = 1; d < cndisp - 1 && nr_local_minimum < nr_plane; d++)
        {

            if (cur < prev && cur < next)
            {
                data_cost_selected[nr_local_minimum * cdisp_step1] = cur;
                selected_disparity[nr_local_minimum * cdisp_step1] = d;
                data_cost[d * cdisp_step1] = SHRT_MAX;

                nr_local_minimum++;
            }

            prev = cur;
            cur = next;
            next = data_cost[(d + 1) * cdisp_step1];
        }

        for (int i = nr_local_minimum; i < nr_plane; i++)
        {
            short minimum = SHRT_MAX;
            int id = 0;

            for (int d = 0; d < cndisp; d++)
            {
                cur = data_cost[d * cdisp_step1];
                if (cur < minimum)
                {
                    minimum = cur;
                    id = d;
                }
            }

            data_cost_selected[i * cdisp_step1] = minimum;
            selected_disparity[i * cdisp_step1] = id;
            data_cost[id * cdisp_step1] = SHRT_MAX;
        }
    }
}

__kernel void get_first_k_initial_local_1(__global float *data_cost_selected_, __global float *selected_disp_pyr,
    __global float *ctemp,int h, int w, int nr_plane,
    int cmsg_step1,  int cdisp_step1, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        __global float *selected_disparity = selected_disp_pyr   + y * cmsg_step1 + x;
        __global float *data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
        __global float *data_cost = ctemp + y * cmsg_step1 + x;

        int nr_local_minimum = 0;

        float prev = data_cost[0 * cdisp_step1];
        float cur  = data_cost[1 * cdisp_step1];
        float next = data_cost[2 * cdisp_step1];

        for (int d = 1; d < cndisp - 1 && nr_local_minimum < nr_plane; d++)
        {
            if (cur < prev && cur < next)
            {
                data_cost_selected[nr_local_minimum * cdisp_step1] = cur;
                selected_disparity[nr_local_minimum * cdisp_step1] = d;
                data_cost[d * cdisp_step1] = FLT_MAX ;

                nr_local_minimum++;
            }

            prev = cur;
            cur = next;
            next = data_cost[(d + 1) * cdisp_step1];
        }


        for (int i = nr_local_minimum; i < nr_plane; i++)
        {
            float minimum = FLT_MAX;
            int id = 0;

            for (int d = 0; d < cndisp; d++)
            {
                cur = data_cost[d * cdisp_step1];
                if (cur < minimum)
                {
                    minimum = cur;
                    id = d;
                }
            }

            data_cost_selected[i * cdisp_step1] = minimum;
            selected_disparity[i * cdisp_step1] = id;
            data_cost[id * cdisp_step1] = FLT_MAX;
        }
    }
}

///////////////////////////////////////////////////////////////
/////////////////////// init data cost ////////////////////////
///////////////////////////////////////////////////////////////

float compute_3(__global uchar* left, __global uchar* right,
    float cdata_weight,  float cmax_data_term)
{
    float tb = 0.114f * abs((int)left[0] - right[0]);
    float tg = 0.587f * abs((int)left[1] - right[1]);
    float tr = 0.299f * abs((int)left[2] - right[2]);

    return fmin(cdata_weight * (tr + tg + tb), cdata_weight * cmax_data_term);
}

float compute_1(__global uchar* left, __global uchar* right,
    float cdata_weight,  float cmax_data_term)
{
    return fmin(cdata_weight * abs((int)*left - (int)*right), cdata_weight * cmax_data_term);
}

short round_short(float v)
{
    return convert_short_sat_rte(v);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////init_data_cost///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

__kernel void init_data_cost_0(__global short *ctemp, __global uchar *cleft, __global uchar *cright,
    int h, int w, int level, int channels,
    int cmsg_step1, float cdata_weight, float cmax_data_term, int cdisp_step1,
    int cth, int cimg_step, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        int y0 = y << level;
        int yt = (y + 1) << level;

        int x0 = x << level;
        int xt = (x + 1) << level;

        __global short *data_cost = ctemp + y * cmsg_step1 + x;

        for(int d = 0; d < cndisp; ++d)
        {
            float val = 0.0f;
            for(int yi = y0; yi < yt; yi++)
            {
                for(int xi = x0; xi < xt; xi++)
                {
                    int xr = xi - d;
                    if(d < cth || xr < 0)
                        val += cdata_weight * cmax_data_term;
                    else
                    {
                        __global uchar *lle = cleft  + yi * cimg_step + xi * channels;
                        __global uchar *lri = cright + yi * cimg_step + xr * channels;

                        if(channels == 1)
                            val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                        else
                            val += compute_3(lle, lri, cdata_weight, cmax_data_term);
                    }
                }
            }
            data_cost[cdisp_step1 * d] = round_short(val);
        }
    }
}

__kernel void init_data_cost_1(__global float *ctemp, __global uchar *cleft, __global uchar *cright,
    int h, int w, int level, int channels,
    int cmsg_step1, float cdata_weight, float cmax_data_term, int cdisp_step1,
    int cth, int cimg_step, int cndisp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        int y0 = y << level;
        int yt = (y + 1) << level;

        int x0 = x << level;
        int xt = (x + 1) << level;

        __global float *data_cost = ctemp + y * cmsg_step1 + x;

        for(int d = 0; d < cndisp; ++d)
        {
            float val = 0.0f;
            for(int yi = y0; yi < yt; yi++)
            {
                for(int xi = x0; xi < xt; xi++)
                {
                    int xr = xi - d;
                    if(d < cth || xr < 0)
                        val += cdata_weight * cmax_data_term;
                    else
                    {
                        __global uchar* lle = cleft  + yi * cimg_step + xi * channels;
                        __global uchar* lri = cright + yi * cimg_step + xr * channels;

                        if(channels == 1)
                            val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                        else
                            val += compute_3(lle, lri, cdata_weight, cmax_data_term);
                    }
                }
            }
            data_cost[cdisp_step1 * d] = val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////init_data_cost_reduce//////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void init_data_cost_reduce_0(__global short *ctemp, __global uchar *cleft, __global uchar *cright,
    __local float *smem, int level, int rows, int cols, int h, int winsz, int channels,
    int cndisp,int cimg_step, float cdata_weight, float cmax_data_term, int cth,
    int cdisp_step1, int cmsg_step1)
{
    int x_out = get_group_id(0);
    int y_out = get_group_id(1) % h;
    //int d = (blockIdx.y / h) * blockDim.z + threadIdx.z;
    int d = (get_group_id(1) / h ) * get_local_size(2) + get_local_id(2);

    int tid = get_local_id(0);

    if (d < cndisp)
    {
        int x0 = x_out << level;
        int y0 = y_out << level;

        int len = min(y0 + winsz, rows) - y0;

        float val = 0.0f;
        if (x0 + tid < cols)
        {
            if (x0 + tid - d < 0 || d < cth)
                val = cdata_weight * cmax_data_term * len;
            else
            {
                __global uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                __global uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - d);

                for(int y = 0; y < len; ++y)
                {
                    if(channels == 1)
                        val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                    else
                        val += compute_3(lle, lri, cdata_weight, cmax_data_term);

                    lle += cimg_step;
                    lri += cimg_step;
                }
            }
        }

        __local float* dline = smem + winsz * get_local_id(2);

        dline[tid] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local float* dline = smem + winsz * get_local_id(2);
        if (winsz >= 256)
        {
            if (tid < 128)
                dline[tid] += dline[tid + 128];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local float* dline = smem + winsz * get_local_id(2);
        if (winsz >= 128)
        {
            if (tid <  64)
                dline[tid] += dline[tid + 64];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 64)
            if (tid < 32)
                vdline[tid] += vdline[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 32)
            if (tid < 16)
                vdline[tid] += vdline[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d<cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 16)
            if (tid <  8)
                vdline[tid] += vdline[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d<cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 8)
            if (tid <  4)
                vdline[tid] += vdline[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d<cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 4)
            if (tid <  2)
                vdline[tid] += vdline[tid + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d<cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 2)
            if (tid <  1)
                vdline[tid] += vdline[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local float* dline = smem + winsz * get_local_id(2);
        __global short* data_cost = ctemp + y_out * cmsg_step1 + x_out;
        if (tid == 0)
            data_cost[cdisp_step1 * d] = convert_short_sat_rte(dline[0]);
    }
}

__kernel void init_data_cost_reduce_1(__global float *ctemp, __global uchar *cleft, __global uchar *cright,
    __local float *smem, int level, int rows, int cols, int h, int winsz, int channels,
    int cndisp,int cimg_step, float cdata_weight, float cmax_data_term, int cth,
    int cdisp_step1, int cmsg_step1)
{
    int x_out = get_group_id(0);
    int y_out = get_group_id(1) % h;
    int d = (get_group_id(1) / h ) * get_local_size(2) + get_local_id(2);

    int tid = get_local_id(0);

    if (d < cndisp)
    {
        int x0 = x_out << level;
        int y0 = y_out << level;

        int len = min(y0 + winsz, rows) - y0;

        float val = 0.0f;
        //float val = 528.0f;

        if (x0 + tid < cols)
        {
            if (x0 + tid - d < 0 || d < cth)
                val = cdata_weight * cmax_data_term * len;
            else
            {
                __global uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                __global uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - d);

                for(int y = 0; y < len; ++y)
                {
                    if(channels == 1)
                        val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                    else
                        val += compute_3(lle, lri, cdata_weight, cmax_data_term);

                    lle += cimg_step;
                    lri += cimg_step;
                }
            }
        }

        __local float* dline = smem + winsz * get_local_id(2);

        dline[tid] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local float* dline = smem + winsz * get_local_id(2);
        if (winsz >= 256)
            if (tid < 128)
                dline[tid] += dline[tid + 128];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local float* dline = smem + winsz * get_local_id(2);
        if (winsz >= 128)
            if (tid < 64)
                dline[tid] += dline[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 64)
            if (tid < 32)
                vdline[tid] += vdline[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 32)
            if (tid < 16)
                vdline[tid] += vdline[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 16)
            if (tid < 8)
                vdline[tid] += vdline[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 8)
            if (tid < 4)
                vdline[tid] += vdline[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 4)
            if (tid < 2)
                vdline[tid] += vdline[tid + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 2)
            if (tid < 1)
                vdline[tid] += vdline[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < cndisp)
    {
        __global float *data_cost = ctemp + y_out * cmsg_step1 + x_out;
        __local float* dline = smem + winsz * get_local_id(2);
        if (tid == 0)
            data_cost[cdisp_step1 * d] =  dline[0];
    }
}

///////////////////////////////////////////////////////////////
////////////////////// compute data cost //////////////////////
///////////////////////////////////////////////////////////////

__kernel void compute_data_cost_0(__global const short *selected_disp_pyr, __global short *data_cost_,
    __global uchar *cleft, __global uchar *cright,
    int h, int w, int level, int nr_plane, int channels,
    int cmsg_step1, int cmsg_step2, int cdisp_step1, int cdisp_step2, float cdata_weight,
    float cmax_data_term, int cimg_step, int cth)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        int y0 = y << level;
        int yt = (y + 1) << level;

        int x0 = x << level;
        int xt = (x + 1) << level;

        __global const short *selected_disparity = selected_disp_pyr + y/2 * cmsg_step2 + x/2;
        __global       short *data_cost          = data_cost_ + y * cmsg_step1 + x;

        for(int d = 0; d < nr_plane; d++)
        {
            float val = 0.0f;
            for(int yi = y0; yi < yt; yi++)
            {
                for(int xi = x0; xi < xt; xi++)
                {
                    int sel_disp = selected_disparity[d * cdisp_step2];
                    int xr = xi - sel_disp;

                    if (xr < 0 || sel_disp < cth)
                        val += cdata_weight * cmax_data_term;

                    else
                    {
                        __global uchar* left_x  = cleft + yi * cimg_step + xi * channels;
                        __global uchar* right_x = cright + yi * cimg_step + xr * channels;

                        if(channels == 1)
                            val += compute_1(left_x, right_x, cdata_weight, cmax_data_term);
                        else
                            val += compute_3(left_x, right_x, cdata_weight, cmax_data_term);
                    }
                }
            }
            data_cost[cdisp_step1 * d] = convert_short_sat_rte(val);
        }
    }
}

__kernel void compute_data_cost_1(__global const float *selected_disp_pyr, __global float *data_cost_,
    __global uchar *cleft, __global uchar *cright,
    int h, int w, int level, int nr_plane, int channels,
    int cmsg_step1, int cmsg_step2, int cdisp_step1, int cdisp_step2, float cdata_weight,
    float cmax_data_term, int cimg_step, int cth)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        int y0 = y << level;
        int yt = (y + 1) << level;

        int x0 = x << level;
        int xt = (x + 1) << level;

        __global const float *selected_disparity = selected_disp_pyr + y/2 * cmsg_step2 + x/2;
        __global       float *data_cost          = data_cost_ + y * cmsg_step1 + x;

        for(int d = 0; d < nr_plane; d++)
        {
            float val = 0.0f;
            for(int yi = y0; yi < yt; yi++)
            {
                for(int xi = x0; xi < xt; xi++)
                {
                    int sel_disp = selected_disparity[d * cdisp_step2];
                    int xr = xi - sel_disp;

                    if (xr < 0 || sel_disp < cth)
                        val += cdata_weight * cmax_data_term;
                    else
                    {
                        __global uchar* left_x  = cleft + yi * cimg_step + xi * channels;
                        __global uchar* right_x = cright + yi * cimg_step + xr * channels;

                        if(channels == 1)
                            val += compute_1(left_x, right_x, cdata_weight, cmax_data_term);
                        else
                            val += compute_3(left_x, right_x, cdata_weight, cmax_data_term);
                    }
                }
            }
            data_cost[cdisp_step1 * d] = val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////compute_data_cost_reduce//////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void compute_data_cost_reduce_0(__global const short* selected_disp_pyr, __global short* data_cost_,
    __global uchar *cleft, __global uchar *cright,__local float *smem,
    int level, int rows, int cols, int h, int nr_plane,
    int channels, int winsz,
    int cmsg_step1, int cmsg_step2, int cdisp_step1, int cdisp_step2,
    float cdata_weight,  float cmax_data_term, int cimg_step,int cth)

{
    int x_out = get_group_id(0);
    int y_out = get_group_id(1) % h;
    int d = (get_group_id(1)/ h) * get_local_size(2) + get_local_id(2);

    int tid = get_local_id(0);

    __global const short* selected_disparity = selected_disp_pyr + y_out/2 * cmsg_step2 + x_out/2;
    __global short* data_cost = data_cost_ + y_out * cmsg_step1 + x_out;

    if (d < nr_plane)
    {
        int sel_disp = selected_disparity[d * cdisp_step2];

        int x0 = x_out << level;
        int y0 = y_out << level;

        int len = min(y0 + winsz, rows) - y0;

        float val = 0.0f;
        if (x0 + tid < cols)
        {
            if (x0 + tid - sel_disp < 0 || sel_disp < cth)
                val = cdata_weight * cmax_data_term * len;
            else
            {
                __global uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                __global uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - sel_disp);

                for(int y = 0; y < len; ++y)
                {
                    if(channels == 1)
                        val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                    else
                        val += compute_3(lle, lri, cdata_weight, cmax_data_term);

                    lle += cimg_step;
                    lri += cimg_step;
                }
            }
        }

        __local float* dline = smem + winsz * get_local_id(2);

        dline[tid] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // if (winsz >= 256) { if (tid < 128) { dline[tid] += dline[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    //if (winsz >= 128) { if (tid <  64) { dline[tid] += dline[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 64)
        {
            if (tid < 32)
                vdline[tid] += vdline[tid + 32];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 32)
        {
            if (tid < 16)
                vdline[tid] += vdline[tid + 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 16)
        {
            if (tid < 8)
                vdline[tid] += vdline[tid + 8];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 8)
        {
            if (tid < 4)
                vdline[tid] += vdline[tid + 4];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 4)
        {
            if (tid < 2)
                vdline[tid] += vdline[tid + 2];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 2)
        {
            if (tid < 1)
                vdline[tid] += vdline[tid + 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (tid == 0)
            data_cost[cdisp_step1 * d] = convert_short_sat_rte(vdline[0]);
    }
}

__kernel void compute_data_cost_reduce_1(__global const float *selected_disp_pyr, __global float *data_cost_,
    __global uchar *cleft, __global uchar *cright, __local float *smem,
    int level, int rows, int cols, int h, int nr_plane,
    int channels, int winsz,
    int cmsg_step1, int cmsg_step2, int cdisp_step1,int cdisp_step2, float cdata_weight,
    float cmax_data_term, int cimg_step, int cth)

{
    int x_out = get_group_id(0);
    int y_out = get_group_id(1) % h;
    int d = (get_group_id(1)/ h) * get_local_size(2) + get_local_id(2);

    int tid = get_local_id(0);

    __global const float *selected_disparity = selected_disp_pyr + y_out/2 * cmsg_step2 + x_out/2;
    __global float *data_cost = data_cost_ + y_out * cmsg_step1 + x_out;

    if (d < nr_plane)
    {
        int sel_disp = selected_disparity[d * cdisp_step2];

        int x0 = x_out << level;
        int y0 = y_out << level;

        int len = min(y0 + winsz, rows) - y0;

        float val = 0.0f;
        if (x0 + tid < cols)
        {
            if (x0 + tid - sel_disp < 0 || sel_disp < cth)
                val = cdata_weight * cmax_data_term * len;
            else
            {
                __global uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                __global uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - sel_disp);

                for(int y = 0; y < len; ++y)
                {
                    if(channels == 1)
                        val += compute_1(lle, lri, cdata_weight, cmax_data_term);
                    else
                        val += compute_3(lle, lri, cdata_weight, cmax_data_term);

                    lle += cimg_step;
                    lri += cimg_step;
                }
            }
        }

        __local float* dline = smem + winsz * get_local_id(2);

        dline[tid] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 64)
        {
            if (tid < 32)
                vdline[tid] += vdline[tid + 32];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 32)
        {
            if (tid < 16)
                vdline[tid] += vdline[tid + 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >= 16)
        {
            if (tid <  8)
                vdline[tid] += vdline[tid + 8];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >=  8)
        {
            if (tid <  4)
                vdline[tid] += vdline[tid + 4];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >=  4)
        {
            if (tid <  2)
                vdline[tid] += vdline[tid + 2];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (winsz >=  2)
        {
            if (tid <  1)
                vdline[tid] += vdline[tid + 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(d < nr_plane)
    {
        __local volatile float* vdline = smem + winsz * get_local_id(2);
        if (tid == 0)
            data_cost[cdisp_step1 * d] = vdline[0];
    }
}

///////////////////////////////////////////////////////////////
//////////////////////// init message /////////////////////////
///////////////////////////////////////////////////////////////

void get_first_k_element_increase_0(__global short* u_new, __global short *d_new, __global short *l_new,
    __global short *r_new, __global const short *u_cur, __global const short *d_cur,
    __global const short *l_cur, __global const short *r_cur,
    __global short *data_cost_selected, __global short *disparity_selected_new,
    __global short *data_cost_new, __global const short* data_cost_cur,
    __global const short *disparity_selected_cur,
    int nr_plane, int nr_plane2,
    int cdisp_step1, int cdisp_step2)
{
    for(int i = 0; i < nr_plane; i++)
    {
        short minimum = SHRT_MAX;
        int id = 0;
        for(int j = 0; j < nr_plane2; j++)
        {
            short cur = data_cost_new[j * cdisp_step1];
            if(cur < minimum)
            {
                minimum = cur;
                id = j;
            }
        }

        data_cost_selected[i * cdisp_step1] = data_cost_cur[id * cdisp_step1];
        disparity_selected_new[i * cdisp_step1] = disparity_selected_cur[id * cdisp_step2];

        u_new[i * cdisp_step1] = u_cur[id * cdisp_step2];
        d_new[i * cdisp_step1] = d_cur[id * cdisp_step2];
        l_new[i * cdisp_step1] = l_cur[id * cdisp_step2];
        r_new[i * cdisp_step1] = r_cur[id * cdisp_step2];

        data_cost_new[id * cdisp_step1] = SHRT_MAX;
    }
}

__kernel void init_message_0(__global short *u_new_, __global short *d_new_, __global short *l_new_,
    __global short *r_new_, __global  short *u_cur_, __global const short *d_cur_,
    __global const short *l_cur_, __global const short *r_cur_, __global short *ctemp,
    __global short *selected_disp_pyr_new, __global const short *selected_disp_pyr_cur,
    __global short *data_cost_selected_, __global const short *data_cost_,
    int h, int w, int nr_plane, int h2, int w2, int nr_plane2,
    int cdisp_step1, int cdisp_step2, int cmsg_step1, int cmsg_step2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < h && x < w)
    {
        __global const short *u_cur = u_cur_ + min(h2-1, y/2 + 1) * cmsg_step2 + x/2;
        __global const short *d_cur = d_cur_ + max(0, y/2 - 1)    * cmsg_step2 + x/2;
        __global const short *l_cur = l_cur_ + y/2                * cmsg_step2 + min(w2-1, x/2 + 1);
        __global const short *r_cur = r_cur_ + y/2                * cmsg_step2 + max(0, x/2 - 1);

        __global short *data_cost_new = ctemp + y * cmsg_step1 + x;

        __global const short *disparity_selected_cur = selected_disp_pyr_cur + y/2 * cmsg_step2 + x/2;
        __global const short *data_cost = data_cost_ + y * cmsg_step1 + x;

        for(int d = 0; d < nr_plane2; d++)
        {
            int idx2 = d * cdisp_step2;

            short val  = data_cost[d * cdisp_step1] + u_cur[idx2] + d_cur[idx2] + l_cur[idx2] + r_cur[idx2];
            data_cost_new[d * cdisp_step1] = val;
        }

        __global short *data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
        __global short *disparity_selected_new = selected_disp_pyr_new + y * cmsg_step1 + x;

        __global short *u_new = u_new_ + y * cmsg_step1 + x;
        __global short *d_new = d_new_ + y * cmsg_step1 + x;
        __global short *l_new = l_new_ + y * cmsg_step1 + x;
        __global short *r_new = r_new_ + y * cmsg_step1 + x;

        u_cur = u_cur_ + y/2 * cmsg_step2 + x/2;
        d_cur = d_cur_ + y/2 * cmsg_step2 + x/2;
        l_cur = l_cur_ + y/2 * cmsg_step2 + x/2;
        r_cur = r_cur_ + y/2 * cmsg_step2 + x/2;

        get_first_k_element_increase_0(u_new, d_new, l_new, r_new, u_cur, d_cur, l_cur, r_cur,
            data_cost_selected, disparity_selected_new, data_cost_new,
            data_cost, disparity_selected_cur, nr_plane, nr_plane2,
            cdisp_step1, cdisp_step2);
    }
}

__kernel void init_message_1(__global float *u_new_, __global float *d_new_, __global float *l_new_,
    __global float *r_new_, __global const float *u_cur_, __global const float *d_cur_,
    __global const float *l_cur_, __global const float *r_cur_, __global float *ctemp,
    __global float *selected_disp_pyr_new, __global const float *selected_disp_pyr_cur,
    __global float *data_cost_selected_, __global const float *data_cost_,
    int h, int w, int nr_plane, int h2, int w2, int nr_plane2,
    int cdisp_step1, int cdisp_step2, int cmsg_step1, int cmsg_step2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);


    __global const float *u_cur = u_cur_ + min(h2-1, y/2 + 1) * cmsg_step2 + x/2;
    __global const float *d_cur = d_cur_ + max(0, y/2 - 1)    * cmsg_step2 + x/2;
    __global const float *l_cur = l_cur_ + y/2                * cmsg_step2 + min(w2-1, x/2 + 1);
    __global const float *r_cur = r_cur_ + y/2                * cmsg_step2 + max(0, x/2 - 1);

    __global float *data_cost_new = ctemp + y * cmsg_step1 + x;

    __global const float *disparity_selected_cur = selected_disp_pyr_cur + y/2 * cmsg_step2 + x/2;
    __global const float *data_cost = data_cost_ + y * cmsg_step1 + x;

    if (y < h && x < w)
    {
        for(int d = 0; d < nr_plane2; d++)
        {
            int idx2 = d * cdisp_step2;

            float val  = data_cost[d * cdisp_step1] + u_cur[idx2] + d_cur[idx2] + l_cur[idx2] + r_cur[idx2];
            data_cost_new[d * cdisp_step1] = val;
        }
    }

    __global float *data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
    __global float *disparity_selected_new = selected_disp_pyr_new + y * cmsg_step1 + x;

    __global float *u_new = u_new_ + y * cmsg_step1 + x;
    __global float *d_new = d_new_ + y * cmsg_step1 + x;
    __global float *l_new = l_new_ + y * cmsg_step1 + x;
    __global float *r_new = r_new_ + y * cmsg_step1 + x;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(y < h && x < w)
    {
        u_cur = u_cur_ + y/2 * cmsg_step2 + x/2;
        d_cur = d_cur_ + y/2 * cmsg_step2 + x/2;
        l_cur = l_cur_ + y/2 * cmsg_step2 + x/2;
        r_cur = r_cur_ + y/2 * cmsg_step2 + x/2;

        for(int i = 0; i < nr_plane; i++)
        {
            float minimum = FLT_MAX;
            int id = 0;

            for(int j = 0; j < nr_plane2; j++)
            {
                float cur = data_cost_new[j * cdisp_step1];
                if(cur < minimum)
                {
                    minimum = cur;
                    id = j;
                }
            }
            data_cost_selected[i * cdisp_step1] = data_cost[id * cdisp_step1];
            disparity_selected_new[i * cdisp_step1] = disparity_selected_cur[id * cdisp_step2];
            u_new[i * cdisp_step1] = u_cur[id * cdisp_step2];
            d_new[i * cdisp_step1] = d_cur[id * cdisp_step2];
            l_new[i * cdisp_step1] = l_cur[id * cdisp_step2];
            r_new[i * cdisp_step1] = r_cur[id * cdisp_step2];
            data_cost_new[id * cdisp_step1] = FLT_MAX;
        }
    }
}

///////////////////////////////////////////////////////////////
////////////////////  calc all iterations /////////////////////
///////////////////////////////////////////////////////////////

void message_per_pixel_0(__global const short *data, __global short *msg_dst, __global const short *msg1,
    __global const short *msg2, __global const short *msg3,
    __global const short *dst_disp, __global const short *src_disp,
    int nr_plane, __global short *temp,
    float cmax_disc_term, int cdisp_step1, float cdisc_single_jump)
{
    short minimum = SHRT_MAX;
    for(int d = 0; d < nr_plane; d++)
    {
        int idx = d * cdisp_step1;
        short val  = data[idx] + msg1[idx] + msg2[idx] + msg3[idx];

        if(val < minimum)
            minimum = val;

        msg_dst[idx] = val;
    }

    float sum = 0;
    for(int d = 0; d < nr_plane; d++)
    {
        float cost_min = minimum + cmax_disc_term;
        short src_disp_reg = src_disp[d * cdisp_step1];

        for(int d2 = 0; d2 < nr_plane; d2++)
            cost_min = fmin(cost_min, (msg_dst[d2 * cdisp_step1] +
            cdisc_single_jump * abs(dst_disp[d2 * cdisp_step1] - src_disp_reg)));

        temp[d * cdisp_step1] = convert_short_sat_rte(cost_min);
        sum += cost_min;
    }
    sum /= nr_plane;

    for(int d = 0; d < nr_plane; d++)
        msg_dst[d * cdisp_step1] = convert_short_sat_rte(temp[d * cdisp_step1] - sum);
}

void message_per_pixel_1(__global const float *data, __global float *msg_dst, __global const float *msg1,
    __global const float *msg2, __global const float *msg3,
    __global const float *dst_disp, __global const float *src_disp,
    int nr_plane, __global float *temp,
    float cmax_disc_term, int cdisp_step1, float cdisc_single_jump)
{
    float minimum = FLT_MAX;
    for(int d = 0; d < nr_plane; d++)
    {
        int idx = d * cdisp_step1;
        float val  = data[idx] + msg1[idx] + msg2[idx] + msg3[idx];

        if(val < minimum)
            minimum = val;

        msg_dst[idx] = val;
    }

    float sum = 0;
    for(int d = 0; d < nr_plane; d++)
    {
        float cost_min = minimum + cmax_disc_term;
        float src_disp_reg = src_disp[d * cdisp_step1];

        for(int d2 = 0; d2 < nr_plane; d2++)
            cost_min = fmin(cost_min, (msg_dst[d2 * cdisp_step1] +
            cdisc_single_jump * fabs(dst_disp[d2 * cdisp_step1] - src_disp_reg)));

        temp[d * cdisp_step1] = cost_min;
        sum += cost_min;
    }
    sum /= nr_plane;

    for(int d = 0; d < nr_plane; d++)
        msg_dst[d * cdisp_step1] = temp[d * cdisp_step1] - sum;
}

__kernel void compute_message_0(__global short *u_, __global short *d_, __global short *l_, __global short *r_,
    __global const short *data_cost_selected, __global const short *selected_disp_pyr_cur,
    __global short *ctemp, int h, int w, int nr_plane, int i,
    float cmax_disc_term, int cdisp_step1, int cmsg_step1, float cdisc_single_jump)
{
    int y = get_global_id(1);
    int x = ((get_global_id(0)) << 1) + ((y + i) & 1);

    if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
    {
        __global const short *data = data_cost_selected + y * cmsg_step1 + x;

        __global short *u = u_ + y * cmsg_step1 + x;
        __global short *d = d_ + y * cmsg_step1 + x;
        __global short *l = l_ + y * cmsg_step1 + x;
        __global short *r = r_ + y * cmsg_step1 + x;

        __global const short *disp = selected_disp_pyr_cur + y * cmsg_step1 + x;

        __global short *temp = ctemp + y * cmsg_step1 + x;

        message_per_pixel_0(data, u, r - 1, u + cmsg_step1, l + 1, disp, disp - cmsg_step1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_0(data, d, d - cmsg_step1, r - 1, l + 1, disp, disp + cmsg_step1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_0(data, l, u + cmsg_step1, d - cmsg_step1, l + 1, disp, disp - 1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_0(data, r, u + cmsg_step1, d - cmsg_step1, r - 1, disp, disp + 1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
    }
}

__kernel void compute_message_1(__global float *u_, __global float *d_, __global float *l_, __global float *r_,
    __global const float *data_cost_selected, __global const float *selected_disp_pyr_cur,
    __global float *ctemp, int h, int w, int nr_plane, int i,
    float cmax_disc_term, int cdisp_step1, int cmsg_step1, float cdisc_single_jump)
{
    int y = get_global_id(1);
    int x = ((get_global_id(0)) << 1) + ((y + i) & 1);

    if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
    {
        __global const float *data = data_cost_selected + y * cmsg_step1 + x;

        __global float *u = u_ + y * cmsg_step1 + x;
        __global float *d = d_ + y * cmsg_step1 + x;
        __global float *l = l_ + y * cmsg_step1 + x;
        __global float *r = r_ + y * cmsg_step1 + x;

        __global const float *disp = selected_disp_pyr_cur + y * cmsg_step1 + x;
        __global float *temp = ctemp + y * cmsg_step1 + x;

        message_per_pixel_1(data, u, r - 1, u + cmsg_step1, l + 1, disp, disp - cmsg_step1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_1(data, d, d - cmsg_step1, r - 1, l + 1, disp, disp + cmsg_step1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_1(data, l, u + cmsg_step1, d - cmsg_step1, l + 1, disp, disp - 1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
        message_per_pixel_1(data, r, u + cmsg_step1, d - cmsg_step1, r - 1, disp, disp + 1, nr_plane, temp,
            cmax_disc_term, cdisp_step1, cdisc_single_jump);
    }
}

///////////////////////////////////////////////////////////////
/////////////////////////// output ////////////////////////////
///////////////////////////////////////////////////////////////

__kernel void compute_disp_0(__global const short *u_, __global const short *d_, __global const short *l_,
    __global const short *r_, __global const short * data_cost_selected,
    __global const short *disp_selected_pyr,
    __global short* disp,
    int res_step, int cols, int rows, int nr_plane,
    int cmsg_step1, int cdisp_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
    {
        __global const short *data = data_cost_selected + y * cmsg_step1 + x;
        __global const short *disp_selected = disp_selected_pyr + y * cmsg_step1 + x;

        __global const short *u = u_ + (y+1) * cmsg_step1 + (x+0);
        __global const short *d = d_ + (y-1) * cmsg_step1 + (x+0);
        __global const short *l = l_ + (y+0) * cmsg_step1 + (x+1);
        __global const short *r = r_ + (y+0) * cmsg_step1 + (x-1);

        short best = 0;
        short best_val = SHRT_MAX;

        for (int i = 0; i < nr_plane; ++i)
        {
            int idx = i * cdisp_step1;
            short val = data[idx]+ u[idx] + d[idx] + l[idx] + r[idx];

            if (val < best_val)
            {
                best_val = val;
                best = disp_selected[idx];
            }
        }
        disp[res_step * y + x] = best;
    }
}

__kernel void compute_disp_1(__global const float *u_, __global const float *d_, __global const float *l_,
    __global const float *r_, __global const float *data_cost_selected,
    __global const float *disp_selected_pyr,
    __global short *disp,
    int res_step, int cols, int rows, int nr_plane,
    int cmsg_step1, int cdisp_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
    {
        __global const float *data = data_cost_selected + y * cmsg_step1 + x;
        __global const float *disp_selected = disp_selected_pyr + y * cmsg_step1 + x;

        __global const float *u = u_ + (y+1) * cmsg_step1 + (x+0);
        __global const float *d = d_ + (y-1) * cmsg_step1 + (x+0);
        __global const float *l = l_ + (y+0) * cmsg_step1 + (x+1);
        __global const float *r = r_ + (y+0) * cmsg_step1 + (x-1);

        short best = 0;
        short best_val = SHRT_MAX;
        for (int i = 0; i < nr_plane; ++i)
        {
            int idx = i * cdisp_step1;
            float val = data[idx]+ u[idx] + d[idx] + l[idx] + r[idx];

            if (val < best_val)
            {
                best_val = val;
                best = convert_short_sat_rte(disp_selected[idx]);
            }
        }
        disp[res_step * y + x] = best;
    }
}
