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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define CV_PI M_PI
#else
#define CV_PI M_PI_F
#endif

#define X_ROW 0
#define Y_ROW 1
#define RESPONSE_ROW 2
#define ANGLE_ROW 3
#define OCTAVE_ROW 4
#define SIZE_ROW 5
#define ROWS_COUNT 6


#ifdef CPU
void reduce_32(volatile __local int* smem, volatile int* val, int tid)
{
#define op(A, B) (*A)+(B)

    smem[tid] = *val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 16; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem[tid] = *val = op(val, smem[tid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#undef op
}
#else
void reduce_32(volatile __local int* smem, volatile int* val, int tid)
{
#define op(A, B) (*A)+(B)

    smem[tid] = *val;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    if (tid < 16)
    {
        smem[tid] = *val = op(val, smem[tid + 16]);
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem[tid] = *val = op(val, smem[tid + 8]);
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        smem[tid] = *val = op(val, smem[tid + 4]);
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        smem[tid] = *val = op(val, smem[tid + 2]);
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        smem[tid] = *val = op(val, smem[tid + 1]);
    }
#undef WAVE_SIZE
#undef op
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
// HarrisResponses

__kernel
void HarrisResponses(__global const uchar* img,
                     __global float* keypoints,
                     const int npoints,
                     const int blockSize,
                     const float harris_k,
                     const int img_step,
                     const int keypoints_step)
{
    __local int smem0[8 * 32];
    __local int smem1[8 * 32];
    __local int smem2[8 * 32];

    const int ptidx = mad24(get_group_id(0), get_local_size(1), get_local_id(1));

    if (ptidx < npoints)
    {
        const int pt_x = keypoints[mad24(keypoints_step, X_ROW, ptidx)];
        const int pt_y = keypoints[mad24(keypoints_step, Y_ROW, ptidx)];

        const int r = blockSize / 2;
        const int x0 = pt_x - r;
        const int y0 = pt_y - r;

        int a = 0, b = 0, c = 0;

        for (int ind = get_local_id(0); ind < blockSize * blockSize; ind += get_local_size(0))
        {
            const int i = ind / blockSize;
            const int j = ind % blockSize;

            int center = mad24(y0+i, img_step, x0+j);

            int Ix = (img[center+1] - img[center-1]) * 2 +
                     (img[center-img_step+1] - img[center-img_step-1]) +
                     (img[center+img_step+1] - img[center+img_step-1]);

            int Iy = (img[center+img_step] - img[center-img_step]) * 2 +
                     (img[center+img_step-1] - img[center-img_step-1]) +
                     (img[center+img_step+1] - img[center-img_step+1]);

            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }

        __local int* srow0 = smem0 + get_local_id(1) * get_local_size(0);
        __local int* srow1 = smem1 + get_local_id(1) * get_local_size(0);
        __local int* srow2 = smem2 + get_local_id(1) * get_local_size(0);

        reduce_32(srow0, &a, get_local_id(0));
        reduce_32(srow1, &b, get_local_id(0));
        reduce_32(srow2, &c, get_local_id(0));

        if (get_local_id(0) == 0)
        {
            float scale = (1 << 2) * blockSize * 255.0f;
            scale = 1.0f / scale;
            const float scale_sq_sq = scale * scale * scale * scale;

            float response = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
            keypoints[mad24(keypoints_step, RESPONSE_ROW, ptidx)] = response;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// IC_Angle

__kernel
void IC_Angle(__global const uchar* img,
              __global float* keypoints_,
              __global const int* u_max,
              const int npoints,
              const int half_k,
              const int img_step,
              const int keypoints_step)
{
    __local int smem0[8 * 32];
    __local int smem1[8 * 32];

    __local int* srow0 = smem0 + get_local_id(1) * get_local_size(0);
    __local int* srow1 = smem1 + get_local_id(1) * get_local_size(0);

    const int ptidx = mad24(get_group_id(0), get_local_size(1), get_local_id(1));

    if (ptidx < npoints)
    {
        int m_01 = 0, m_10 = 0;

        const int pt_x = keypoints_[mad24(keypoints_step, X_ROW, ptidx)];
        const int pt_y = keypoints_[mad24(keypoints_step, Y_ROW, ptidx)];

        // Treat the center line differently, v=0
        for (int u = get_local_id(0) - half_k; u <= half_k; u += get_local_size(0))
            m_10 += u * img[mad24(pt_y, img_step, pt_x+u)];

        reduce_32(srow0, &m_10, get_local_id(0));

        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int m_sum = 0;
            const int d = u_max[v];

            for (int u = get_local_id(0) - d; u <= d; u += get_local_size(0))
            {
                int val_plus = img[mad24(pt_y+v, img_step, pt_x+u)];
                int val_minus = img[mad24(pt_y-v, img_step, pt_x+u)];

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }

            reduce_32(srow0, &v_sum, get_local_id(0));
            reduce_32(srow1, &m_sum, get_local_id(0));

            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (get_local_id(0) == 0)
        {
            float kp_dir = atan2((float)m_01, (float)m_10);
            kp_dir += (kp_dir < 0) * (2.0f * CV_PI);
            kp_dir *= 180.0f / CV_PI;

            keypoints_[mad24(keypoints_step, ANGLE_ROW, ptidx)] = kp_dir;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// computeOrbDescriptor

#define GET_VALUE(idx) \
    img[mad24(loc.y + (int)round(pattern[idx] * sina + pattern[pattern_step+idx] * cosa), img_step, \
         loc.x + (int)round(pattern[idx] * cosa - pattern[pattern_step+idx] * sina))]

int calcOrbDescriptor_2(__global const uchar* img,
                        __global const int* pattern,
                        const int2 loc,
                        const float sina,
                        const float cosa,
                        const int i,
                        const int img_step,
                        const int pattern_step)
{
    pattern += 16 * i;

    int t0, t1, val;

    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    val = t0 < t1;

    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;

    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;

    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;

    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;

    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;

    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;

    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    return val;
}

int calcOrbDescriptor_3(__global const uchar* img,
                        __global const int* pattern,
                        const int2 loc,
                        const float sina,
                        const float cosa,
                        const int i,
                        const int img_step,
                        const int pattern_step)
{
    pattern += 12 * i;

    int t0, t1, t2, val;

    t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
    val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

    t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
    val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

    t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
    val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

    t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
    val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

    return val;
}

int calcOrbDescriptor_4(__global const uchar* img,
                        __global const int* pattern,
                        const int2 loc,
                        const float sina,
                        const float cosa,
                        const int i,
                        const int img_step,
                        const int pattern_step)
{
    pattern += 16 * i;

    int t0, t1, t2, t3, k, val;
    int a, b;

    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    t2 = GET_VALUE(2); t3 = GET_VALUE(3);
    a = 0, b = 2;
    if( t1 > t0 ) t0 = t1, a = 1;
    if( t3 > t2 ) t2 = t3, b = 3;
    k = t0 > t2 ? a : b;
    val = k;

    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    t2 = GET_VALUE(6); t3 = GET_VALUE(7);
    a = 0, b = 2;
    if( t1 > t0 ) t0 = t1, a = 1;
    if( t3 > t2 ) t2 = t3, b = 3;
    k = t0 > t2 ? a : b;
    val |= k << 2;

    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    t2 = GET_VALUE(10); t3 = GET_VALUE(11);
    a = 0, b = 2;
    if( t1 > t0 ) t0 = t1, a = 1;
    if( t3 > t2 ) t2 = t3, b = 3;
    k = t0 > t2 ? a : b;
    val |= k << 4;

    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    t2 = GET_VALUE(14); t3 = GET_VALUE(15);
    a = 0, b = 2;
    if( t1 > t0 ) t0 = t1, a = 1;
    if( t3 > t2 ) t2 = t3, b = 3;
    k = t0 > t2 ? a : b;
    val |= k << 6;

    return val;
}

#undef GET_VALUE

__kernel
void computeOrbDescriptor(__global const uchar* img,
                          __global const float* keypoints,
                          __global const int* pattern,
                          __global uchar* desc,
                          const int npoints,
                          const int dsize,
                          const int WTA_K,
                          const int offset,
                          const int img_step,
                          const int keypoints_step,
                          const int pattern_step,
                          const int desc_step)
{
    const int descidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));
    const int ptidx = mad24(get_group_id(1), get_local_size(1), get_local_id(1));

    if (ptidx < npoints && descidx < dsize)
    {
        int2 loc = {(int)keypoints[mad24(keypoints_step, X_ROW, ptidx)],
                    (int)keypoints[mad24(keypoints_step, Y_ROW, ptidx)]};

        float angle = keypoints[mad24(keypoints_step, ANGLE_ROW, ptidx)];
        angle *= (float)(CV_PI / 180.f);

        float sina = sin(angle);
        float cosa = cos(angle);

        if (WTA_K == 2)
            desc[mad24(ptidx+offset, desc_step, descidx)] = calcOrbDescriptor_2(img, pattern, loc, sina, cosa, descidx, img_step, pattern_step);
        else if (WTA_K == 3)
            desc[mad24(ptidx+offset, desc_step, descidx)] = calcOrbDescriptor_3(img, pattern, loc, sina, cosa, descidx, img_step, pattern_step);
        else if (WTA_K == 4)
            desc[mad24(ptidx+offset, desc_step, descidx)] = calcOrbDescriptor_4(img, pattern, loc, sina, cosa, descidx, img_step, pattern_step);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// mergeLocation

__kernel
void mergeLocation(__global const float* keypoints_in,
                   __global float* keypoints_out,
                   const int npoints,
                   const int offset,
                   const float scale,
                   const int octave,
                   const float size,
                   const int keypoints_in_step,
                   const int keypoints_out_step)
{
    //const int ptidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ptidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));

    if (ptidx < npoints)
    {
        float pt_x = keypoints_in[mad24(keypoints_in_step, X_ROW, ptidx)] * scale;
        float pt_y = keypoints_in[mad24(keypoints_in_step, Y_ROW, ptidx)] * scale;
        float response = keypoints_in[mad24(keypoints_in_step, RESPONSE_ROW, ptidx)];
        float angle = keypoints_in[mad24(keypoints_in_step, ANGLE_ROW, ptidx)];

        keypoints_out[mad24(keypoints_out_step, X_ROW, ptidx+offset)] = pt_x;
        keypoints_out[mad24(keypoints_out_step, Y_ROW, ptidx+offset)] = pt_y;
        keypoints_out[mad24(keypoints_out_step, RESPONSE_ROW, ptidx+offset)] = response;
        keypoints_out[mad24(keypoints_out_step, ANGLE_ROW, ptidx+offset)] = angle;
        keypoints_out[mad24(keypoints_out_step, OCTAVE_ROW, ptidx+offset)] = (float)octave;
        keypoints_out[mad24(keypoints_out_step, SIZE_ROW, ptidx+offset)] = size;
    }
}

__kernel
void convertRowsToChannels(__global const float* keypoints_in,
                           __global float* keypoints_out,
                           const int npoints,
                           const int keypoints_in_step,
                           const int keypoints_out_step)
{
    const int ptidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));

    if (ptidx < npoints)
    {
        const int pt_x = keypoints_in[mad24(keypoints_in_step, X_ROW, ptidx)];
        const int pt_y = keypoints_in[mad24(keypoints_in_step, Y_ROW, ptidx)];

        keypoints_out[ptidx*2] = pt_x;
        keypoints_out[ptidx*2+1] = pt_y;
    }
}

__kernel
void convertChannelsToRows(__global const float* keypoints_pos,
                           __global const float* keypoints_resp,
                           __global float* keypoints_out,
                           const int npoints,
                           const int keypoints_pos_step,
                           const int keypoints_resp_step,
                           const int keypoints_out_step)
{
    const int ptidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));

    if (ptidx < npoints)
    {
        const float pt_x = keypoints_pos[ptidx*2];
        const float pt_y = keypoints_pos[ptidx*2+1];
        const float resp = keypoints_resp[ptidx];

        keypoints_out[mad24(keypoints_out_step, X_ROW, ptidx)] = pt_x;
        keypoints_out[mad24(keypoints_out_step, Y_ROW, ptidx)] = pt_y;
        keypoints_out[mad24(keypoints_out_step, RESPONSE_ROW, ptidx)] = resp;
    }
}
