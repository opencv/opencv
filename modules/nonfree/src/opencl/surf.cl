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
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2013, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//    Sen Liu, swjtuls1987@126.com
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

// specialized for non-image2d_t supported platform, intel HD4000, for example
#ifdef DISABLE_IMAGE2D
#define IMAGE_INT32 __global uint  *
#define IMAGE_INT8  __global uchar *
#else
#define IMAGE_INT32 image2d_t
#define IMAGE_INT8  image2d_t
#endif

uint read_sumTex(IMAGE_INT32 img, sampler_t sam, int2 coord, int rows, int cols, int elemPerRow)
{
#ifdef DISABLE_IMAGE2D
    int x = clamp(coord.x, 0, cols);
    int y = clamp(coord.y, 0, rows);
    return img[elemPerRow * y + x];
#else
    return read_imageui(img, sam, coord).x;
#endif
}
uchar read_imgTex(IMAGE_INT8 img, sampler_t sam, float2 coord, int rows, int cols, int elemPerRow)
{
#ifdef DISABLE_IMAGE2D
    int x = clamp(round(coord.x), 0, cols - 1);
    int y = clamp(round(coord.y), 0, rows - 1);
    return img[elemPerRow * y + x];
#else
    return (uchar)read_imageui(img, sam, coord).x;
#endif
}

// dynamically change the precision used for floating type

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#define F double
#else
#define F float
#endif

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifndef FLT_EPSILON
#define FLT_EPSILON (1e-15)
#endif

#ifndef CV_PI_F
#define CV_PI_F 3.14159265f
#endif


// Use integral image to calculate haar wavelets.
// N = 2
// for simple haar paatern
float icvCalcHaarPatternSum_2(
    IMAGE_INT32 sumTex,
    __constant float2 *src,
    int oldSize,
    int newSize,
    int y, int x,
    int rows, int cols, int elemPerRow)
{

    float ratio = (float)newSize / oldSize;

    F d = 0;

    int2 dx1 = convert_int2(round(ratio * src[0]));
    int2 dy1 = convert_int2(round(ratio * src[1]));
    int2 dx2 = convert_int2(round(ratio * src[2]));
    int2 dy2 = convert_int2(round(ratio * src[3]));

    F t = 0;
    t += read_sumTex( sumTex, sampler, (int2)(x + dx1.x, y + dy1.x), rows, cols, elemPerRow );
    t -= read_sumTex( sumTex, sampler, (int2)(x + dx1.x, y + dy2.x), rows, cols, elemPerRow );
    t -= read_sumTex( sumTex, sampler, (int2)(x + dx2.x, y + dy1.x), rows, cols, elemPerRow );
    t += read_sumTex( sumTex, sampler, (int2)(x + dx2.x, y + dy2.x), rows, cols, elemPerRow );
    d += t * src[4].x / ((dx2.x - dx1.x) * (dy2.x - dy1.x));

    t = 0;
    t += read_sumTex( sumTex, sampler, (int2)(x + dx1.y, y + dy1.y), rows, cols, elemPerRow );
    t -= read_sumTex( sumTex, sampler, (int2)(x + dx1.y, y + dy2.y), rows, cols, elemPerRow );
    t -= read_sumTex( sumTex, sampler, (int2)(x + dx2.y, y + dy1.y), rows, cols, elemPerRow );
    t += read_sumTex( sumTex, sampler, (int2)(x + dx2.y, y + dy2.y), rows, cols, elemPerRow );
    d += t * src[4].y / ((dx2.y - dx1.y) * (dy2.y - dy1.y));

    return (float)d;
}

////////////////////////////////////////////////////////////////////////
// Hessian

__inline int calcSize(int octave, int layer)
{
    /* Wavelet size at first layer of first octave. */
    const int HAAR_SIZE0 = 9;

    /* Wavelet size increment between layers. This should be an even number,
    such that the wavelet sizes in an octave are either all even or all odd.
    This ensures that when looking for the neighbours of a sample, the layers
    above and below are aligned correctly. */
    const int HAAR_SIZE_INC = 6;

    return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
}

// Calculate a derivative in an axis-aligned direction (x or y).  The "plus1"
// boxes contribute 1 * (area), and the "minus2" box contributes -2 * (area).
// So the final computation is plus1a + plus1b - 2 * minus2.  The corners are
// labeled A, B, C, and D, with A being the top left, B being top right, C
// being bottom left, and D being bottom right.
F calcAxisAlignedDerivative(
        int plus1a_A, int plus1a_B, int plus1a_C, int plus1a_D, F plus1a_scale,
        int plus1b_A, int plus1b_B, int plus1b_C, int plus1b_D, F plus1b_scale,
        int minus2_A, int minus2_B, int minus2_C, int minus2_D, F minus2_scale)
{
    F plus1a = plus1a_A - plus1a_B - plus1a_C + plus1a_D;
    F plus1b = plus1b_A - plus1b_B - plus1b_C + plus1b_D;
    F minus2 = minus2_A - minus2_B - minus2_C + minus2_D;

    return (plus1a / plus1a_scale -
            2.0f * minus2 / minus2_scale +
            plus1b / plus1b_scale);
}

//calculate targeted layer per-pixel determinant and trace with an integral image
__kernel void icvCalcLayerDetAndTrace(
    IMAGE_INT32 sumTex, // input integral image
    __global float * det,      // output Determinant
    __global float * trace,    // output trace
    int det_step,     // the step of det in bytes
    int trace_step,   // the step of trace in bytes
    int c_img_rows,
    int c_img_cols,
    int c_nOctaveLayers,
    int c_octave,
    int c_layer_rows,
    int sumTex_step
    )
{
    det_step   /= sizeof(*det);
    trace_step /= sizeof(*trace);
    sumTex_step/= sizeof(uint);
    // Determine the indices
    const int gridDim_y  = get_num_groups(1) / (c_nOctaveLayers + 2);
    const int blockIdx_y = get_group_id(1) % gridDim_y;
    const int blockIdx_z = get_group_id(1) / gridDim_y;

    const int j = get_local_id(0) + get_group_id(0) * get_local_size(0);
    const int i = get_local_id(1) + blockIdx_y * get_local_size(1);
    const int layer = blockIdx_z;

    const int size = calcSize(c_octave, layer);

    const int samples_i = 1 + ((c_img_rows - size) >> c_octave);
    const int samples_j = 1 + ((c_img_cols - size) >> c_octave);

    // Ignore pixels where some of the kernel is outside the image
    const int margin = (size >> 1) >> c_octave;

    if (size <= c_img_rows && size <= c_img_cols && i < samples_i && j < samples_j)
    {
        int x = j << c_octave;
        int y = i << c_octave;

        float ratio = (float)size / 9;

        // Precompute some commonly used values, which are used to offset
        // texture coordinates in the integral image.
        int r1 = round(ratio);
        int r2 = round(ratio * 2.0f);
        int r3 = round(ratio * 3.0f);
        int r4 = round(ratio * 4.0f);
        int r5 = round(ratio * 5.0f);
        int r6 = round(ratio * 6.0f);
        int r7 = round(ratio * 7.0f);
        int r8 = round(ratio * 8.0f);
        int r9 = round(ratio * 9.0f);

        // Calculate the approximated derivative in the x-direction
        F d = 0;
        {
            // Some of the pixels needed to compute the derivative are
            // repeated, so we only don't duplicate the fetch here.
            int t02 = read_sumTex( sumTex, sampler, (int2)(x, y + r2), c_img_rows, c_img_cols, sumTex_step );
            int t07 = read_sumTex( sumTex, sampler, (int2)(x, y + r7), c_img_rows, c_img_cols, sumTex_step );
            int t32 = read_sumTex( sumTex, sampler, (int2)(x + r3, y + r2), c_img_rows, c_img_cols, sumTex_step );
            int t37 = read_sumTex( sumTex, sampler, (int2)(x + r3, y + r7), c_img_rows, c_img_cols, sumTex_step );
            int t62 = read_sumTex( sumTex, sampler, (int2)(x + r6, y + r2), c_img_rows, c_img_cols, sumTex_step );
            int t67 = read_sumTex( sumTex, sampler, (int2)(x + r6, y + r7), c_img_rows, c_img_cols, sumTex_step );
            int t92 = read_sumTex( sumTex, sampler, (int2)(x + r9, y + r2), c_img_rows, c_img_cols, sumTex_step );
            int t97 = read_sumTex( sumTex, sampler, (int2)(x + r9, y + r7), c_img_rows, c_img_cols, sumTex_step );

            d = calcAxisAlignedDerivative(t02, t07, t32, t37, (r3) * (r7 - r2),
                                          t62, t67, t92, t97, (r9 - r6) * (r7 - r2),
                                          t32, t37, t62, t67, (r6 - r3) * (r7 - r2));
        }
        const float dx  = (float)d;

        // Calculate the approximated derivative in the y-direction
        d = 0;
        {
            // Some of the pixels needed to compute the derivative are
            // repeated, so we only don't duplicate the fetch here.
            int t20 = read_sumTex( sumTex, sampler, (int2)(x + r2, y), c_img_rows, c_img_cols, sumTex_step );
            int t23 = read_sumTex( sumTex, sampler, (int2)(x + r2, y + r3), c_img_rows, c_img_cols, sumTex_step );
            int t70 = read_sumTex( sumTex, sampler, (int2)(x + r7, y), c_img_rows, c_img_cols, sumTex_step );
            int t73 = read_sumTex( sumTex, sampler, (int2)(x + r7, y + r3), c_img_rows, c_img_cols, sumTex_step );
            int t26 = read_sumTex( sumTex, sampler, (int2)(x + r2, y + r6), c_img_rows, c_img_cols, sumTex_step );
            int t76 = read_sumTex( sumTex, sampler, (int2)(x + r7, y + r6), c_img_rows, c_img_cols, sumTex_step );
            int t29 = read_sumTex( sumTex, sampler, (int2)(x + r2, y + r9), c_img_rows, c_img_cols, sumTex_step );
            int t79 = read_sumTex( sumTex, sampler, (int2)(x + r7, y + r9), c_img_rows, c_img_cols, sumTex_step );

            d = calcAxisAlignedDerivative(t20, t23, t70, t73, (r7 - r2) * (r3),
                                          t26, t29, t76, t79, (r7 - r2) * (r9 - r6),
                                          t23, t26, t73, t76, (r7 - r2) * (r6 - r3));
        }
        const float dy  = (float)d;

        // Calculate the approximated derivative in the xy-direction
        d = 0;
        {
            // There's no saving us here, we just have to get all of the pixels in
            // separate fetches
            F t = 0;
            t += read_sumTex( sumTex, sampler, (int2)(x + r1, y + r1), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r1, y + r4), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r4, y + r1), c_img_rows, c_img_cols, sumTex_step );
            t += read_sumTex( sumTex, sampler, (int2)(x + r4, y + r4), c_img_rows, c_img_cols, sumTex_step );
            d += t / ((r4 - r1) * (r4 - r1));

            t = 0;
            t += read_sumTex( sumTex, sampler, (int2)(x + r5, y + r1), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r5, y + r4), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r8, y + r1), c_img_rows, c_img_cols, sumTex_step );
            t += read_sumTex( sumTex, sampler, (int2)(x + r8, y + r4), c_img_rows, c_img_cols, sumTex_step );
            d -= t / ((r8 - r5) * (r4 - r1));

            t = 0;
            t += read_sumTex( sumTex, sampler, (int2)(x + r1, y + r5), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r1, y + r8), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r4, y + r5), c_img_rows, c_img_cols, sumTex_step );
            t += read_sumTex( sumTex, sampler, (int2)(x + r4, y + r8), c_img_rows, c_img_cols, sumTex_step );
            d -= t / ((r4 - r1) * (r8 - r5));

            t = 0;
            t += read_sumTex( sumTex, sampler, (int2)(x + r5, y + r5), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r5, y + r8), c_img_rows, c_img_cols, sumTex_step );
            t -= read_sumTex( sumTex, sampler, (int2)(x + r8, y + r5), c_img_rows, c_img_cols, sumTex_step );
            t += read_sumTex( sumTex, sampler, (int2)(x + r8, y + r8), c_img_rows, c_img_cols, sumTex_step );
            d += t / ((r8 - r5) * (r8 - r5));
        }
        const float dxy = (float)d;

        det  [j + margin + det_step   * (layer * c_layer_rows + i + margin)] = dx * dy - 0.81f * dxy * dxy;
        trace[j + margin + trace_step * (layer * c_layer_rows + i + margin)] = dx + dy;
    }
}

////////////////////////////////////////////////////////////////////////
// NONMAX

__constant float c_DM[5] = {0, 0, 9, 9, 1};

bool within_check(IMAGE_INT32 maskSumTex, int sum_i, int sum_j, int size, int rows, int cols, int step)
{
    float ratio = (float)size / 9.0f;

    float d = 0;

    int dx1 = round(ratio * c_DM[0]);
    int dy1 = round(ratio * c_DM[1]);
    int dx2 = round(ratio * c_DM[2]);
    int dy2 = round(ratio * c_DM[3]);

    float t = 0;

    t += read_sumTex(maskSumTex, sampler, (int2)(sum_j + dx1, sum_i + dy1), rows, cols, step);
    t -= read_sumTex(maskSumTex, sampler, (int2)(sum_j + dx1, sum_i + dy2), rows, cols, step);
    t -= read_sumTex(maskSumTex, sampler, (int2)(sum_j + dx2, sum_i + dy1), rows, cols, step);
    t += read_sumTex(maskSumTex, sampler, (int2)(sum_j + dx2, sum_i + dy2), rows, cols, step);

    d += t * c_DM[4] / ((dx2 - dx1) * (dy2 - dy1));

    return (d >= 0.5f);
}

// Non-maximal suppression to further filtering the candidates from previous step
__kernel
void icvFindMaximaInLayer_withmask(
    __global const float * det,
    __global const float * trace,
    __global int4 * maxPosBuffer,
    volatile __global int* maxCounter,
    int counter_offset,
    int det_step,     // the step of det in bytes
    int trace_step,   // the step of trace in bytes
    int c_img_rows,
    int c_img_cols,
    int c_nOctaveLayers,
    int c_octave,
    int c_layer_rows,
    int c_layer_cols,
    int c_max_candidates,
    float c_hessianThreshold,
    IMAGE_INT32 maskSumTex,
    int mask_step
)
{
    volatile __local  float N9[768]; // threads.x * threads.y * 3

    det_step   /= sizeof(*det);
    trace_step /= sizeof(*trace);
    maxCounter += counter_offset;
    mask_step  /= sizeof(uint);

    // Determine the indices
    const int gridDim_y  = get_num_groups(1) / c_nOctaveLayers;
    const int blockIdx_y = get_group_id(1)   % gridDim_y;
    const int blockIdx_z = get_group_id(1)   / gridDim_y;

    const int layer = blockIdx_z + 1;

    const int size = calcSize(c_octave, layer);

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    const int margin = ((calcSize(c_octave, layer + 1) >> 1) >> c_octave) + 1;

    const int j = get_local_id(0) + get_group_id(0) * (get_local_size(0) - 2) + margin - 1;
    const int i = get_local_id(1) + blockIdx_y * (get_local_size(1) - 2) + margin - 1;

    // Is this thread within the hessian buffer?
    const int zoff = get_local_size(0) * get_local_size(1);
    const int localLin = get_local_id(0) + get_local_id(1) * get_local_size(0) + zoff;
    N9[localLin - zoff] =
        det[det_step *
            (c_layer_rows * (layer - 1) + min(max(i, 0), c_img_rows - 1)) // y
            + min(max(j, 0), c_img_cols - 1)];                            // x
    N9[localLin       ] =
        det[det_step *
            (c_layer_rows * (layer    ) + min(max(i, 0), c_img_rows - 1)) // y
            + min(max(j, 0), c_img_cols - 1)];                            // x
    N9[localLin + zoff] =
        det[det_step *
            (c_layer_rows * (layer + 1) + min(max(i, 0), c_img_rows - 1)) // y
            + min(max(j, 0), c_img_cols - 1)];                            // x

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < c_layer_rows - margin
            && j < c_layer_cols - margin
            && get_local_id(0) > 0
            && get_local_id(0) < get_local_size(0) - 1
            && get_local_id(1) > 0
            && get_local_id(1) < get_local_size(1) - 1 // these are unnecessary conditions ported from CUDA
       )
    {
        float val0 = N9[localLin];

        if (val0 > c_hessianThreshold)
        {
            // Coordinates for the start of the wavelet in the sum image. There
            // is some integer division involved, so don't try to simplify this
            // (cancel out sampleStep) without checking the result is the same
            const int sum_i = (i - ((size >> 1) >> c_octave)) << c_octave;
            const int sum_j = (j - ((size >> 1) >> c_octave)) << c_octave;

            if (within_check(maskSumTex, sum_i, sum_j, size, c_img_rows, c_img_cols, mask_step))
            {
                // Check to see if we have a max (in its 26 neighbours)
                const bool condmax = val0 > N9[localLin - 1 - get_local_size(0) - zoff]
                                     &&                   val0 > N9[localLin     - get_local_size(0) - zoff]
                                     &&                   val0 > N9[localLin + 1 - get_local_size(0) - zoff]
                                     &&                   val0 > N9[localLin - 1                     - zoff]
                                     &&                   val0 > N9[localLin                         - zoff]
                                     &&                   val0 > N9[localLin + 1                     - zoff]
                                     &&                   val0 > N9[localLin - 1 + get_local_size(0) - zoff]
                                     &&                   val0 > N9[localLin     + get_local_size(0) - zoff]
                                     &&                   val0 > N9[localLin + 1 + get_local_size(0) - zoff]

                                     &&                   val0 > N9[localLin - 1 - get_local_size(0)]
                                     &&                   val0 > N9[localLin     - get_local_size(0)]
                                     &&                   val0 > N9[localLin + 1 - get_local_size(0)]
                                     &&                   val0 > N9[localLin - 1                    ]
                                     &&                   val0 > N9[localLin + 1                    ]
                                     &&                   val0 > N9[localLin - 1 + get_local_size(0)]
                                     &&                   val0 > N9[localLin     + get_local_size(0)]
                                     &&                   val0 > N9[localLin + 1 + get_local_size(0)]

                                     &&                   val0 > N9[localLin - 1 - get_local_size(0) + zoff]
                                     &&                   val0 > N9[localLin     - get_local_size(0) + zoff]
                                     &&                   val0 > N9[localLin + 1 - get_local_size(0) + zoff]
                                     &&                   val0 > N9[localLin - 1                     + zoff]
                                     &&                   val0 > N9[localLin                         + zoff]
                                     &&                   val0 > N9[localLin + 1                     + zoff]
                                     &&                   val0 > N9[localLin - 1 + get_local_size(0) + zoff]
                                     &&                   val0 > N9[localLin     + get_local_size(0) + zoff]
                                     &&                   val0 > N9[localLin + 1 + get_local_size(0) + zoff]
                                     ;

                if(condmax)
                {
                    int ind = atomic_inc(maxCounter);

                    if (ind < c_max_candidates)
                    {
                        const int laplacian = (int) copysign(1.0f, trace[trace_step* (layer * c_layer_rows + i) + j]);

                        maxPosBuffer[ind] = (int4)(j, i, layer, laplacian);
                    }
                }
            }
        }
    }
}

__kernel
void icvFindMaximaInLayer(
    __global float * det,
    __global float * trace,
    __global int4 * maxPosBuffer,
    volatile __global  int* maxCounter,
    int counter_offset,
    int det_step,     // the step of det in bytes
    int trace_step,   // the step of trace in bytes
    int c_img_rows,
    int c_img_cols,
    int c_nOctaveLayers,
    int c_octave,
    int c_layer_rows,
    int c_layer_cols,
    int c_max_candidates,
    float c_hessianThreshold
)
{
    volatile __local  float N9[768]; // threads.x * threads.y * 3

    det_step   /= sizeof(float);
    trace_step /= sizeof(float);
    maxCounter += counter_offset;

    // Determine the indices
    const int gridDim_y  = get_num_groups(1) / c_nOctaveLayers;
    const int blockIdx_y = get_group_id(1)   % gridDim_y;
    const int blockIdx_z = get_group_id(1)   / gridDim_y;

    const int layer = blockIdx_z + 1;

    const int size = calcSize(c_octave, layer);

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    const int margin = ((calcSize(c_octave, layer + 1) >> 1) >> c_octave) + 1;

    const int j = get_local_id(0) + get_group_id(0) * (get_local_size(0) - 2) + margin - 1;
    const int i = get_local_id(1) + blockIdx_y      * (get_local_size(1) - 2) + margin - 1;

    // Is this thread within the hessian buffer?
    const int zoff     = get_local_size(0) * get_local_size(1);
    const int localLin = get_local_id(0) + get_local_id(1) * get_local_size(0) + zoff;

    int l_x = min(max(j, 0), c_img_cols - 1);
    int l_y = c_layer_rows * layer + min(max(i, 0), c_img_rows - 1);

    N9[localLin - zoff] =
        det[det_step * (l_y - c_layer_rows) + l_x];
    N9[localLin       ] =
        det[det_step * (l_y               ) + l_x];
    N9[localLin + zoff] =
        det[det_step * (l_y + c_layer_rows) + l_x];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < c_layer_rows - margin
            && j < c_layer_cols - margin
            && get_local_id(0) > 0
            && get_local_id(0) < get_local_size(0) - 1
            && get_local_id(1) > 0
            && get_local_id(1) < get_local_size(1) - 1 // these are unnecessary conditions ported from CUDA
       )
    {
        float val0 = N9[localLin];
        if (val0 > c_hessianThreshold)
        {
            // Coordinates for the start of the wavelet in the sum image. There
            // is some integer division involved, so don't try to simplify this
            // (cancel out sampleStep) without checking the result is the same

            // Check to see if we have a max (in its 26 neighbours)
            const bool condmax = val0 > N9[localLin - 1 - get_local_size(0) - zoff]
                                 &&                   val0 > N9[localLin     - get_local_size(0) - zoff]
                                 &&                   val0 > N9[localLin + 1 - get_local_size(0) - zoff]
                                 &&                   val0 > N9[localLin - 1                     - zoff]
                                 &&                   val0 > N9[localLin                         - zoff]
                                 &&                   val0 > N9[localLin + 1                     - zoff]
                                 &&                   val0 > N9[localLin - 1 + get_local_size(0) - zoff]
                                 &&                   val0 > N9[localLin     + get_local_size(0) - zoff]
                                 &&                   val0 > N9[localLin + 1 + get_local_size(0) - zoff]

                                 &&                   val0 > N9[localLin - 1 - get_local_size(0)]
                                 &&                   val0 > N9[localLin     - get_local_size(0)]
                                 &&                   val0 > N9[localLin + 1 - get_local_size(0)]
                                 &&                   val0 > N9[localLin - 1                    ]
                                 &&                   val0 > N9[localLin + 1                    ]
                                 &&                   val0 > N9[localLin - 1 + get_local_size(0)]
                                 &&                   val0 > N9[localLin     + get_local_size(0)]
                                 &&                   val0 > N9[localLin + 1 + get_local_size(0)]

                                 &&                   val0 > N9[localLin - 1 - get_local_size(0) + zoff]
                                 &&                   val0 > N9[localLin     - get_local_size(0) + zoff]
                                 &&                   val0 > N9[localLin + 1 - get_local_size(0) + zoff]
                                 &&                   val0 > N9[localLin - 1                     + zoff]
                                 &&                   val0 > N9[localLin                         + zoff]
                                 &&                   val0 > N9[localLin + 1                     + zoff]
                                 &&                   val0 > N9[localLin - 1 + get_local_size(0) + zoff]
                                 &&                   val0 > N9[localLin     + get_local_size(0) + zoff]
                                 &&                   val0 > N9[localLin + 1 + get_local_size(0) + zoff]
                                 ;

            if(condmax)
            {
                int ind = atomic_inc(maxCounter);

                if (ind < c_max_candidates)
                {
                    const int laplacian = (int) copysign(1.0f, trace[trace_step* (layer * c_layer_rows + i) + j]);

                    maxPosBuffer[ind] = (int4)(j, i, layer, laplacian);
                }
            }
        }
    }
}

// solve 3x3 linear system Ax=b for floating point input
inline bool solve3x3_float(const float4 *A, const float *b, float *x)
{
    float det = A[0].x * (A[1].y * A[2].z - A[1].z * A[2].y)
                - A[0].y * (A[1].x * A[2].z - A[1].z * A[2].x)
                + A[0].z * (A[1].x * A[2].y - A[1].y * A[2].x);

    if (det != 0)
    {
        F invdet = 1.0 / det;

        x[0] = invdet *
               (b[0]    * (A[1].y * A[2].z - A[1].z * A[2].y) -
                A[0].y * (b[1]    * A[2].z - A[1].z * b[2]   ) +
                A[0].z * (b[1]    * A[2].y - A[1].y * b[2]   ));

        x[1] = invdet *
               (A[0].x * (b[1]    * A[2].z - A[1].z * b[2]   ) -
                b[0]    * (A[1].x * A[2].z - A[1].z * A[2].x) +
                A[0].z * (A[1].x * b[2]    - b[1]    * A[2].x));

        x[2] = invdet *
               (A[0].x * (A[1].y * b[2]    - b[1]    * A[2].y) -
                A[0].y * (A[1].x * b[2]    - b[1]    * A[2].x) +
                b[0]    * (A[1].x * A[2].y - A[1].y * A[2].x));

        return true;
    }
    return false;
}

#define X_ROW          0
#define Y_ROW          1
#define LAPLACIAN_ROW  2
#define OCTAVE_ROW     3
#define SIZE_ROW       4
#define ANGLE_ROW      5
#define HESSIAN_ROW    6
#define ROWS_COUNT     7

////////////////////////////////////////////////////////////////////////
// INTERPOLATION
__kernel
void icvInterpolateKeypoint(
    __global const float * det,
    __global const int4 * maxPosBuffer,
    __global float * keypoints,
    volatile __global  int * featureCounter,
    int det_step,
    int keypoints_step,
    int c_img_rows,
    int c_img_cols,
    int c_octave,
    int c_layer_rows,
    int c_max_features
)
{
    det_step /= sizeof(*det);
    keypoints_step /= sizeof(*keypoints);
    __global float * featureX       = keypoints + X_ROW * keypoints_step;
    __global float * featureY       = keypoints + Y_ROW * keypoints_step;
    __global int * featureLaplacian = (__global int *)keypoints + LAPLACIAN_ROW * keypoints_step;
    __global int * featureOctave    = (__global int *)keypoints + OCTAVE_ROW * keypoints_step;
    __global float * featureSize    = keypoints + SIZE_ROW * keypoints_step;
    __global float * featureHessian = keypoints + HESSIAN_ROW * keypoints_step;

    const int4 maxPos = maxPosBuffer[get_group_id(0)];

    const int j = maxPos.x - 1 + get_local_id(0);
    const int i = maxPos.y - 1 + get_local_id(1);
    const int layer = maxPos.z - 1 + get_local_id(2);

    volatile __local  float N9[3][3][3];

    N9[get_local_id(2)][get_local_id(1)][get_local_id(0)] =
        det[det_step * (c_layer_rows * layer + i) + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
    {
        float dD[3];

        //dx
        dD[0] = -0.5f * (N9[1][1][2] - N9[1][1][0]);
        //dy
        dD[1] = -0.5f * (N9[1][2][1] - N9[1][0][1]);
        //ds
        dD[2] = -0.5f * (N9[2][1][1] - N9[0][1][1]);

        float4 H[3];

        //dxx
        H[0].x = N9[1][1][0] - 2.0f * N9[1][1][1] + N9[1][1][2];
        //dxy
        H[0].y= 0.25f * (N9[1][2][2] - N9[1][2][0] - N9[1][0][2] + N9[1][0][0]);
        //dxs
        H[0].z= 0.25f * (N9[2][1][2] - N9[2][1][0] - N9[0][1][2] + N9[0][1][0]);
        //dyx = dxy
        H[1].x = H[0].y;
        //dyy
        H[1].y = N9[1][0][1] - 2.0f * N9[1][1][1] + N9[1][2][1];
        //dys
        H[1].z= 0.25f * (N9[2][2][1] - N9[2][0][1] - N9[0][2][1] + N9[0][0][1]);
        //dsx = dxs
        H[2].x = H[0].z;
        //dsy = dys
        H[2].y = H[1].z;
        //dss
        H[2].z = N9[0][1][1] - 2.0f * N9[1][1][1] + N9[2][1][1];

        float x[3];

        if (solve3x3_float(H, dD, x))
        {
            if (fabs(x[0]) <= 1.f && fabs(x[1]) <= 1.f && fabs(x[2]) <= 1.f)
            {
                // if the step is within the interpolation region, perform it

                const int size = calcSize(c_octave, maxPos.z);

                const int sum_i = (maxPos.y - ((size >> 1) >> c_octave)) << c_octave;
                const int sum_j = (maxPos.x - ((size >> 1) >> c_octave)) << c_octave;

                const float center_i = sum_i + (float)(size - 1) / 2;
                const float center_j = sum_j + (float)(size - 1) / 2;

                const float px = center_j + x[0] * (1 << c_octave);
                const float py = center_i + x[1] * (1 << c_octave);

                const int ds = size - calcSize(c_octave, maxPos.z - 1);
                const float psize = round(size + x[2] * ds);

                /* The sampling intervals and wavelet sized for selecting an orientation
                and building the keypoint descriptor are defined relative to 's' */
                const float s = psize * 1.2f / 9.0f;

                /* To find the dominant orientation, the gradients in x and y are
                sampled in a circle of radius 6s using wavelets of size 4s.
                We ensure the gradient wavelet size is even to ensure the
                wavelet pattern is balanced and symmetric around its center */
                const int grad_wav_size = 2 * round(2.0f * s);

                // check when grad_wav_size is too big
                if ((c_img_rows + 1) >= grad_wav_size && (c_img_cols + 1) >= grad_wav_size)
                {
                    // Get a new feature index.
                    int ind = atomic_inc(featureCounter);

                    if (ind < c_max_features)
                    {
                        featureX[ind] = px;
                        featureY[ind] = py;
                        featureLaplacian[ind] = maxPos.w;
                        featureOctave[ind] = c_octave;
                        featureSize[ind] = psize;
                        featureHessian[ind] = N9[1][1][1];
                    }
                } // grad_wav_size check
            } // If the subpixel interpolation worked
        }
    } // If this is thread 0.
}

////////////////////////////////////////////////////////////////////////
// Orientation

#define ORI_WIN			 60
#define ORI_SAMPLES		 113

// The distance between samples in the beginning of the the reduction
#define ORI_RESPONSE_REDUCTION_WIDTH		 48
#define ORI_RESPONSE_ARRAY_SIZE			     (ORI_RESPONSE_REDUCTION_WIDTH * 2)

__constant float c_aptX[ORI_SAMPLES] = {-6, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6};
__constant float c_aptY[ORI_SAMPLES] = {0, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -3, -2, -1, 0, 1, 2, 3, 0};
__constant float c_aptW[ORI_SAMPLES] = {0.001455130288377404f, 0.001707611023448408f, 0.002547456417232752f, 0.003238451667129993f, 0.0035081731621176f,
                                        0.003238451667129993f, 0.002547456417232752f, 0.001707611023448408f, 0.002003900473937392f, 0.0035081731621176f, 0.005233579315245152f,
                                        0.00665318313986063f, 0.00720730796456337f, 0.00665318313986063f, 0.005233579315245152f, 0.0035081731621176f,
                                        0.002003900473937392f, 0.001707611023448408f, 0.0035081731621176f, 0.006141661666333675f, 0.009162282571196556f,
                                        0.01164754293859005f, 0.01261763460934162f, 0.01164754293859005f, 0.009162282571196556f, 0.006141661666333675f,
                                        0.0035081731621176f, 0.001707611023448408f, 0.002547456417232752f, 0.005233579315245152f, 0.009162282571196556f,
                                        0.01366852037608624f, 0.01737609319388866f, 0.0188232995569706f, 0.01737609319388866f, 0.01366852037608624f,
                                        0.009162282571196556f, 0.005233579315245152f, 0.002547456417232752f, 0.003238451667129993f, 0.00665318313986063f,
                                        0.01164754293859005f, 0.01737609319388866f, 0.02208934165537357f, 0.02392910048365593f, 0.02208934165537357f,
                                        0.01737609319388866f, 0.01164754293859005f, 0.00665318313986063f, 0.003238451667129993f, 0.001455130288377404f,
                                        0.0035081731621176f, 0.00720730796456337f, 0.01261763460934162f, 0.0188232995569706f, 0.02392910048365593f,
                                        0.02592208795249462f, 0.02392910048365593f, 0.0188232995569706f, 0.01261763460934162f, 0.00720730796456337f,
                                        0.0035081731621176f, 0.001455130288377404f, 0.003238451667129993f, 0.00665318313986063f, 0.01164754293859005f,
                                        0.01737609319388866f, 0.02208934165537357f, 0.02392910048365593f, 0.02208934165537357f, 0.01737609319388866f,
                                        0.01164754293859005f, 0.00665318313986063f, 0.003238451667129993f, 0.002547456417232752f, 0.005233579315245152f,
                                        0.009162282571196556f, 0.01366852037608624f, 0.01737609319388866f, 0.0188232995569706f, 0.01737609319388866f,
                                        0.01366852037608624f, 0.009162282571196556f, 0.005233579315245152f, 0.002547456417232752f, 0.001707611023448408f,
                                        0.0035081731621176f, 0.006141661666333675f, 0.009162282571196556f, 0.01164754293859005f, 0.01261763460934162f,
                                        0.01164754293859005f, 0.009162282571196556f, 0.006141661666333675f, 0.0035081731621176f, 0.001707611023448408f,
                                        0.002003900473937392f, 0.0035081731621176f, 0.005233579315245152f, 0.00665318313986063f, 0.00720730796456337f,
                                        0.00665318313986063f, 0.005233579315245152f, 0.0035081731621176f, 0.002003900473937392f, 0.001707611023448408f,
                                        0.002547456417232752f, 0.003238451667129993f, 0.0035081731621176f, 0.003238451667129993f, 0.002547456417232752f,
                                        0.001707611023448408f, 0.001455130288377404f
                                       };

__constant float2 c_NX[5] = { (float2)(0, 2), (float2)(0, 0), (float2)(2, 4), (float2)(4, 4), (float2)(-1, 1) };
__constant float2 c_NY[5] = { (float2)(0, 0), (float2)(0, 2), (float2)(4, 4), (float2)(2, 4), (float2)(1, -1) };

void reduce_32_sum(volatile __local  float * data, volatile float* partial_reduction, int tid)
{
#define op(A, B) (*A)+(B)
    data[tid] = *partial_reduction;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    if (tid < 16)
    {
        data[tid] = *partial_reduction = op(partial_reduction, data[tid + 16]);
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        data[tid] = *partial_reduction = op(partial_reduction, data[tid + 8]);
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        data[tid] = *partial_reduction = op(partial_reduction, data[tid + 4]);
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        data[tid] = *partial_reduction = op(partial_reduction, data[tid + 2 ]);
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        data[tid] = *partial_reduction = op(partial_reduction, data[tid + 1 ]);
    }
#undef WAVE_SIZE
#undef op
}

__kernel
void icvCalcOrientation(
    IMAGE_INT32 sumTex,
    __global float * keypoints,
    int keypoints_step,
    int c_img_rows,
    int c_img_cols,
    int sum_step
)
{
    keypoints_step /= sizeof(*keypoints);
    sum_step       /= sizeof(uint);
    __global float* featureX    = keypoints + X_ROW * keypoints_step;
    __global float* featureY    = keypoints + Y_ROW * keypoints_step;
    __global float* featureSize = keypoints + SIZE_ROW * keypoints_step;
    __global float* featureDir  = keypoints + ANGLE_ROW * keypoints_step;


    __local  float s_X[ORI_SAMPLES];
    __local  float s_Y[ORI_SAMPLES];
    __local  float s_angle[ORI_SAMPLES];

    // Need to allocate enough to make the reduction work without accessing
    // past the end of the array.
    __local  float s_sumx[ORI_RESPONSE_ARRAY_SIZE];
    __local  float s_sumy[ORI_RESPONSE_ARRAY_SIZE];
    __local  float s_mod[ORI_RESPONSE_ARRAY_SIZE];

    /* The sampling intervals and wavelet sized for selecting an orientation
    and building the keypoint descriptor are defined relative to 's' */
    const float s = featureSize[get_group_id(0)] * 1.2f / 9.0f;


    /* To find the dominant orientation, the gradients in x and y are
    sampled in a circle of radius 6s using wavelets of size 4s.
    We ensure the gradient wavelet size is even to ensure the
    wavelet pattern is balanced and symmetric around its center */
    const int grad_wav_size = 2 * round(2.0f * s);

    // check when grad_wav_size is too big
    if ((c_img_rows + 1) < grad_wav_size || (c_img_cols + 1) < grad_wav_size)
        return;

    // Calc X, Y, angle and store it to shared memory
    const int tid = get_local_id(0);
    // Initialize values that are only used as part of the reduction later.
    if (tid < ORI_RESPONSE_ARRAY_SIZE - ORI_LOCAL_SIZE) {
        s_mod[tid + ORI_LOCAL_SIZE] = 0.0f;
    }

    float ratio = (float)grad_wav_size / 4;

    int r2 = round(ratio * 2.0);
    int r4 = round(ratio * 4.0);
    for (int i = tid; i < ORI_SAMPLES; i += ORI_LOCAL_SIZE )
    {
        float X = 0.0f, Y = 0.0f, angle = 0.0f;
        const float margin = (float)(grad_wav_size - 1) / 2.0f;
        const int x = round(featureX[get_group_id(0)] + c_aptX[i] * s - margin);
        const int y = round(featureY[get_group_id(0)] + c_aptY[i] * s - margin);

        if (y >= 0 && y < (c_img_rows + 1) - grad_wav_size &&
            x >= 0 && x < (c_img_cols + 1) - grad_wav_size)
        {

            float apt = c_aptW[i];

            // Compute the haar sum without fetching duplicate pixels.
            float t00 = read_sumTex( sumTex, sampler, (int2)(x, y), c_img_rows, c_img_cols, sum_step);
            float t02 = read_sumTex( sumTex, sampler, (int2)(x, y + r2), c_img_rows, c_img_cols, sum_step);
            float t04 = read_sumTex( sumTex, sampler, (int2)(x, y + r4), c_img_rows, c_img_cols, sum_step);
            float t20 = read_sumTex( sumTex, sampler, (int2)(x + r2, y), c_img_rows, c_img_cols, sum_step);
            float t24 = read_sumTex( sumTex, sampler, (int2)(x + r2, y + r4), c_img_rows, c_img_cols, sum_step);
            float t40 = read_sumTex( sumTex, sampler, (int2)(x + r4, y), c_img_rows, c_img_cols, sum_step);
            float t42 = read_sumTex( sumTex, sampler, (int2)(x + r4, y + r2), c_img_rows, c_img_cols, sum_step);
            float t44 = read_sumTex( sumTex, sampler, (int2)(x + r4, y + r4), c_img_rows, c_img_cols, sum_step);

            F t = t00 - t04 - t20 + t24;
            X -= t / ((r2) * (r4));

            t = t20 - t24 - t40 + t44;
            X += t / ((r4 - r2) * (r4));

            t = t00 - t02 - t40 + t42;
            Y += t / ((r2) * (r4));

            t = t02 - t04 - t42 + t44;
            Y -= t  / ((r4) * (r4 - r2));

            X = apt*X;
            Y = apt*Y;

            angle = atan2(Y, X);

            if (angle < 0)
                angle += 2.0f * CV_PI_F;
            angle *= 180.0f / CV_PI_F;

        }

        s_X[i] = X;
        s_Y[i] = Y;
        s_angle[i] = angle;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float bestx = 0, besty = 0, best_mod = 0;
    float sumx = 0.0f, sumy = 0.0f;
    const int dir = tid * ORI_SEARCH_INC;
    #pragma unroll
    for (int i = 0; i < ORI_SAMPLES; ++i) {
        int angle = round(s_angle[i]);

        int d = abs(angle - dir);
        if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2)
        {
            sumx += s_X[i];
            sumy += s_Y[i];
        }
    }
    s_sumx[tid] = sumx;
    s_sumy[tid] = sumy;
    s_mod[tid] = sumx*sumx + sumy*sumy;
    barrier(CLK_LOCAL_MEM_FENCE);

    // This reduction searches for the longest wavelet response vector.  The first
    // step uses all of the work items in the workgroup to narrow the search
    // down to the three candidates.  It requires s_mod to have a few more
    // elements alocated past the work-group size, which are pre-initialized to
    // 0.0f above.
    for(int t = ORI_RESPONSE_REDUCTION_WIDTH; t >= 3; t /= 2) {
        if (tid < t) {
            if (s_mod[tid] < s_mod[tid + t]) {
                s_mod[tid] = s_mod[tid + t];
                s_sumx[tid] = s_sumx[tid + t];
                s_sumy[tid] = s_sumy[tid + t];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Do the final reduction and write out the result.
    if (tid == 0)
    {
        int bestIdx = 0;

        // The loop above narrowed the search of the longest vector to three
        // possibilities.  Pick the best here.
        if (s_mod[1] > s_mod[bestIdx])
            bestIdx = 1;
        if (s_mod[2] > s_mod[bestIdx])
            bestIdx = 2;

        float kp_dir = atan2(s_sumy[bestIdx], s_sumx[bestIdx]);
        if (kp_dir < 0)
            kp_dir += 2.0f * CV_PI_F;
        kp_dir *= 180.0f / CV_PI_F;

        kp_dir = 360.0f - kp_dir;
        if (fabs(kp_dir - 360.f) < FLT_EPSILON)
            kp_dir = 0.f;

        featureDir[get_group_id(0)] = kp_dir;
    }
}

__kernel
void icvSetUpright(
    __global float * keypoints,
    int keypoints_step,
    int nFeatures
)
{
    keypoints_step /= sizeof(*keypoints);
    __global float* featureDir  = keypoints + ANGLE_ROW * keypoints_step;

    if(get_global_id(0) <= nFeatures)
    {
        featureDir[get_global_id(0)] = 270.0f;
    }
}


#undef ORI_SEARCH_INC
#undef ORI_WIN
#undef ORI_SAMPLES

////////////////////////////////////////////////////////////////////////
// Descriptors

#define PATCH_SZ 20

__constant float c_DW[PATCH_SZ * PATCH_SZ] =
{
    3.695352233989979e-006f, 8.444558261544444e-006f, 1.760426494001877e-005f, 3.34794785885606e-005f, 5.808438800158911e-005f, 9.193058212986216e-005f, 0.0001327334757661447f, 0.0001748319627949968f, 0.0002100782439811155f, 0.0002302826324012131f, 0.0002302826324012131f, 0.0002100782439811155f, 0.0001748319627949968f, 0.0001327334757661447f, 9.193058212986216e-005f, 5.808438800158911e-005f, 3.34794785885606e-005f, 1.760426494001877e-005f, 8.444558261544444e-006f, 3.695352233989979e-006f,
    8.444558261544444e-006f, 1.929736572492402e-005f, 4.022897701361217e-005f, 7.650675252079964e-005f, 0.0001327334903180599f, 0.0002100782585330308f, 0.0003033203829545528f, 0.0003995231236331165f, 0.0004800673632416874f, 0.0005262381164357066f, 0.0005262381164357066f, 0.0004800673632416874f, 0.0003995231236331165f, 0.0003033203829545528f, 0.0002100782585330308f, 0.0001327334903180599f, 7.650675252079964e-005f, 4.022897701361217e-005f, 1.929736572492402e-005f, 8.444558261544444e-006f,
    1.760426494001877e-005f, 4.022897701361217e-005f, 8.386484114453197e-005f, 0.0001594926579855382f, 0.0002767078403849155f, 0.0004379475140012801f, 0.0006323281559161842f, 0.0008328808471560478f, 0.001000790391117334f, 0.001097041997127235f, 0.001097041997127235f, 0.001000790391117334f, 0.0008328808471560478f, 0.0006323281559161842f, 0.0004379475140012801f, 0.0002767078403849155f, 0.0001594926579855382f, 8.386484114453197e-005f, 4.022897701361217e-005f, 1.760426494001877e-005f,
    3.34794785885606e-005f, 7.650675252079964e-005f, 0.0001594926579855382f, 0.0003033203247468919f, 0.0005262380582280457f, 0.0008328807889483869f, 0.001202550483867526f, 0.001583957928232849f, 0.001903285388834775f, 0.002086334861814976f, 0.002086334861814976f, 0.001903285388834775f, 0.001583957928232849f, 0.001202550483867526f, 0.0008328807889483869f, 0.0005262380582280457f, 0.0003033203247468919f, 0.0001594926579855382f, 7.650675252079964e-005f, 3.34794785885606e-005f,
    5.808438800158911e-005f, 0.0001327334903180599f, 0.0002767078403849155f, 0.0005262380582280457f, 0.0009129836107604206f, 0.001444985857233405f, 0.002086335094645619f, 0.002748048631474376f, 0.00330205773934722f, 0.003619635012000799f, 0.003619635012000799f, 0.00330205773934722f, 0.002748048631474376f, 0.002086335094645619f, 0.001444985857233405f, 0.0009129836107604206f, 0.0005262380582280457f, 0.0002767078403849155f, 0.0001327334903180599f, 5.808438800158911e-005f,
    9.193058212986216e-005f, 0.0002100782585330308f, 0.0004379475140012801f, 0.0008328807889483869f, 0.001444985857233405f, 0.002286989474669099f, 0.00330205773934722f, 0.004349356517195702f, 0.00522619066759944f, 0.005728822201490402f, 0.005728822201490402f, 0.00522619066759944f, 0.004349356517195702f, 0.00330205773934722f, 0.002286989474669099f, 0.001444985857233405f, 0.0008328807889483869f, 0.0004379475140012801f, 0.0002100782585330308f, 9.193058212986216e-005f,
    0.0001327334757661447f, 0.0003033203829545528f, 0.0006323281559161842f, 0.001202550483867526f, 0.002086335094645619f, 0.00330205773934722f, 0.004767658654600382f, 0.006279794964939356f, 0.007545807864516974f, 0.008271530270576477f, 0.008271530270576477f, 0.007545807864516974f, 0.006279794964939356f, 0.004767658654600382f, 0.00330205773934722f, 0.002086335094645619f, 0.001202550483867526f, 0.0006323281559161842f, 0.0003033203829545528f, 0.0001327334757661447f,
    0.0001748319627949968f, 0.0003995231236331165f, 0.0008328808471560478f, 0.001583957928232849f, 0.002748048631474376f, 0.004349356517195702f, 0.006279794964939356f, 0.008271529339253902f, 0.009939077310264111f, 0.01089497376233339f, 0.01089497376233339f, 0.009939077310264111f, 0.008271529339253902f, 0.006279794964939356f, 0.004349356517195702f, 0.002748048631474376f, 0.001583957928232849f, 0.0008328808471560478f, 0.0003995231236331165f, 0.0001748319627949968f,
    0.0002100782439811155f, 0.0004800673632416874f, 0.001000790391117334f, 0.001903285388834775f, 0.00330205773934722f, 0.00522619066759944f, 0.007545807864516974f, 0.009939077310264111f, 0.01194280479103327f, 0.01309141051024199f, 0.01309141051024199f, 0.01194280479103327f, 0.009939077310264111f, 0.007545807864516974f, 0.00522619066759944f, 0.00330205773934722f, 0.001903285388834775f, 0.001000790391117334f, 0.0004800673632416874f, 0.0002100782439811155f,
    0.0002302826324012131f, 0.0005262381164357066f, 0.001097041997127235f, 0.002086334861814976f, 0.003619635012000799f, 0.005728822201490402f, 0.008271530270576477f, 0.01089497376233339f, 0.01309141051024199f, 0.01435048412531614f, 0.01435048412531614f, 0.01309141051024199f, 0.01089497376233339f, 0.008271530270576477f, 0.005728822201490402f, 0.003619635012000799f, 0.002086334861814976f, 0.001097041997127235f, 0.0005262381164357066f, 0.0002302826324012131f,
    0.0002302826324012131f, 0.0005262381164357066f, 0.001097041997127235f, 0.002086334861814976f, 0.003619635012000799f, 0.005728822201490402f, 0.008271530270576477f, 0.01089497376233339f, 0.01309141051024199f, 0.01435048412531614f, 0.01435048412531614f, 0.01309141051024199f, 0.01089497376233339f, 0.008271530270576477f, 0.005728822201490402f, 0.003619635012000799f, 0.002086334861814976f, 0.001097041997127235f, 0.0005262381164357066f, 0.0002302826324012131f,
    0.0002100782439811155f, 0.0004800673632416874f, 0.001000790391117334f, 0.001903285388834775f, 0.00330205773934722f, 0.00522619066759944f, 0.007545807864516974f, 0.009939077310264111f, 0.01194280479103327f, 0.01309141051024199f, 0.01309141051024199f, 0.01194280479103327f, 0.009939077310264111f, 0.007545807864516974f, 0.00522619066759944f, 0.00330205773934722f, 0.001903285388834775f, 0.001000790391117334f, 0.0004800673632416874f, 0.0002100782439811155f,
    0.0001748319627949968f, 0.0003995231236331165f, 0.0008328808471560478f, 0.001583957928232849f, 0.002748048631474376f, 0.004349356517195702f, 0.006279794964939356f, 0.008271529339253902f, 0.009939077310264111f, 0.01089497376233339f, 0.01089497376233339f, 0.009939077310264111f, 0.008271529339253902f, 0.006279794964939356f, 0.004349356517195702f, 0.002748048631474376f, 0.001583957928232849f, 0.0008328808471560478f, 0.0003995231236331165f, 0.0001748319627949968f,
    0.0001327334757661447f, 0.0003033203829545528f, 0.0006323281559161842f, 0.001202550483867526f, 0.002086335094645619f, 0.00330205773934722f, 0.004767658654600382f, 0.006279794964939356f, 0.007545807864516974f, 0.008271530270576477f, 0.008271530270576477f, 0.007545807864516974f, 0.006279794964939356f, 0.004767658654600382f, 0.00330205773934722f, 0.002086335094645619f, 0.001202550483867526f, 0.0006323281559161842f, 0.0003033203829545528f, 0.0001327334757661447f,
    9.193058212986216e-005f, 0.0002100782585330308f, 0.0004379475140012801f, 0.0008328807889483869f, 0.001444985857233405f, 0.002286989474669099f, 0.00330205773934722f, 0.004349356517195702f, 0.00522619066759944f, 0.005728822201490402f, 0.005728822201490402f, 0.00522619066759944f, 0.004349356517195702f, 0.00330205773934722f, 0.002286989474669099f, 0.001444985857233405f, 0.0008328807889483869f, 0.0004379475140012801f, 0.0002100782585330308f, 9.193058212986216e-005f,
    5.808438800158911e-005f, 0.0001327334903180599f, 0.0002767078403849155f, 0.0005262380582280457f, 0.0009129836107604206f, 0.001444985857233405f, 0.002086335094645619f, 0.002748048631474376f, 0.00330205773934722f, 0.003619635012000799f, 0.003619635012000799f, 0.00330205773934722f, 0.002748048631474376f, 0.002086335094645619f, 0.001444985857233405f, 0.0009129836107604206f, 0.0005262380582280457f, 0.0002767078403849155f, 0.0001327334903180599f, 5.808438800158911e-005f,
    3.34794785885606e-005f, 7.650675252079964e-005f, 0.0001594926579855382f, 0.0003033203247468919f, 0.0005262380582280457f, 0.0008328807889483869f, 0.001202550483867526f, 0.001583957928232849f, 0.001903285388834775f, 0.002086334861814976f, 0.002086334861814976f, 0.001903285388834775f, 0.001583957928232849f, 0.001202550483867526f, 0.0008328807889483869f, 0.0005262380582280457f, 0.0003033203247468919f, 0.0001594926579855382f, 7.650675252079964e-005f, 3.34794785885606e-005f,
    1.760426494001877e-005f, 4.022897701361217e-005f, 8.386484114453197e-005f, 0.0001594926579855382f, 0.0002767078403849155f, 0.0004379475140012801f, 0.0006323281559161842f, 0.0008328808471560478f, 0.001000790391117334f, 0.001097041997127235f, 0.001097041997127235f, 0.001000790391117334f, 0.0008328808471560478f, 0.0006323281559161842f, 0.0004379475140012801f, 0.0002767078403849155f, 0.0001594926579855382f, 8.386484114453197e-005f, 4.022897701361217e-005f, 1.760426494001877e-005f,
    8.444558261544444e-006f, 1.929736572492402e-005f, 4.022897701361217e-005f, 7.650675252079964e-005f, 0.0001327334903180599f, 0.0002100782585330308f, 0.0003033203829545528f, 0.0003995231236331165f, 0.0004800673632416874f, 0.0005262381164357066f, 0.0005262381164357066f, 0.0004800673632416874f, 0.0003995231236331165f, 0.0003033203829545528f, 0.0002100782585330308f, 0.0001327334903180599f, 7.650675252079964e-005f, 4.022897701361217e-005f, 1.929736572492402e-005f, 8.444558261544444e-006f,
    3.695352233989979e-006f, 8.444558261544444e-006f, 1.760426494001877e-005f, 3.34794785885606e-005f, 5.808438800158911e-005f, 9.193058212986216e-005f, 0.0001327334757661447f, 0.0001748319627949968f, 0.0002100782439811155f, 0.0002302826324012131f, 0.0002302826324012131f, 0.0002100782439811155f, 0.0001748319627949968f, 0.0001327334757661447f, 9.193058212986216e-005f, 5.808438800158911e-005f, 3.34794785885606e-005f, 1.760426494001877e-005f, 8.444558261544444e-006f, 3.695352233989979e-006f
};

// utility for linear filter
inline uchar readerGet(
    IMAGE_INT8 src,
    const float centerX, const float centerY, const float win_offset, const float cos_dir, const float sin_dir,
    int i, int j, int rows, int cols, int elemPerRow
)
{
    float pixel_x = centerX + (win_offset + j) * cos_dir + (win_offset + i) * sin_dir;
    float pixel_y = centerY - (win_offset + j) * sin_dir + (win_offset + i) * cos_dir;
    return read_imgTex(src, sampler, (float2)(pixel_x, pixel_y), rows, cols, elemPerRow);
}

inline float linearFilter(
    IMAGE_INT8 src,
    const float centerX, const float centerY, const float win_offset, const float cos_dir, const float sin_dir,
    float y, float x, int rows, int cols, int elemPerRow
)
{
    x -= 0.5f;
    y -= 0.5f;

    float out = 0.0f;

    const int x1 = round(x);
    const int y1 = round(y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;

    uchar src_reg = readerGet(src, centerX, centerY, win_offset, cos_dir, sin_dir, y1, x1, rows, cols, elemPerRow);
    out = out + src_reg * ((x2 - x) * (y2 - y));

    src_reg = readerGet(src, centerX, centerY, win_offset, cos_dir, sin_dir, y1, x2, rows, cols, elemPerRow);
    out = out + src_reg * ((x - x1) * (y2 - y));

    src_reg = readerGet(src, centerX, centerY, win_offset, cos_dir, sin_dir, y2, x1, rows, cols, elemPerRow);
    out = out + src_reg * ((x2 - x) * (y - y1));

    src_reg = readerGet(src, centerX, centerY, win_offset, cos_dir, sin_dir, y2, x2, rows, cols, elemPerRow);
    out = out + src_reg * ((x - x1) * (y - y1));

    return out;
}

void calc_dx_dy(
    IMAGE_INT8 imgTex,
    volatile __local  float *s_dx_bin,
    volatile __local  float *s_dy_bin,
    volatile __local  float *s_PATCH,
    __global const float* featureX,
    __global const float* featureY,
    __global const float* featureSize,
    __global const float* featureDir,
    int rows,
    int cols,
    int elemPerRow
)
{
    const float centerX = featureX[get_group_id(0)];
    const float centerY = featureY[get_group_id(0)];
    const float size = featureSize[get_group_id(0)];
    float descriptor_dir = 360.0f - featureDir[get_group_id(0)];
    if(fabs(descriptor_dir - 360.0f) < FLT_EPSILON)
    {
        descriptor_dir = 0.0f;
    }

    descriptor_dir *= (float)(CV_PI_F / 180.0f);

    /* The sampling intervals and wavelet sized for selecting an orientation
    and building the keypoint descriptor are defined relative to 's' */
    const float s = size * 1.2f / 9.0f;

    /* Extract a window of pixels around the keypoint of size 20s */
    const int win_size = (int)((PATCH_SZ + 1) * s);

    float sin_dir;
    float cos_dir;
    sin_dir = sincos(descriptor_dir, &cos_dir);

    /* Nearest neighbour version (faster) */
    const float win_offset = -(float)(win_size - 1) / 2;

    // Compute sampling points
    // since grids are 2D, need to compute xBlock and yBlock indices
    const int xBlock = (get_group_id(1) & 3);  // get_group_id(1) % 4
    const int yBlock = (get_group_id(1) >> 2); // floor(get_group_id(1)/4)
    const int xIndex = xBlock * 5 + get_local_id(0);
    const int yIndex = yBlock * 5 + get_local_id(1);

    const float icoo = ((float)yIndex / (PATCH_SZ + 1)) * win_size;
    const float jcoo = ((float)xIndex / (PATCH_SZ + 1)) * win_size;

    s_PATCH[get_local_id(1) * 6 + get_local_id(0)] = linearFilter(imgTex, centerX, centerY, win_offset, cos_dir, sin_dir, icoo, jcoo, rows, cols, elemPerRow);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 5 && get_local_id(1) < 5)
    {
        const int tid = get_local_id(1) * 5 + get_local_id(0);

        const float dw = c_DW[yIndex * PATCH_SZ + xIndex];

        const float vx = (
                             s_PATCH[      get_local_id(1) * 6 + get_local_id(0) + 1] -
                             s_PATCH[      get_local_id(1) * 6 + get_local_id(0)    ] +
                             s_PATCH[(get_local_id(1) + 1) * 6 + get_local_id(0) + 1] -
                             s_PATCH[(get_local_id(1) + 1) * 6 + get_local_id(0)    ])
                         * dw;
        const float vy = (
                             s_PATCH[(get_local_id(1) + 1) * 6 + get_local_id(0)    ] -
                             s_PATCH[      get_local_id(1) * 6 + get_local_id(0)    ] +
                             s_PATCH[(get_local_id(1) + 1) * 6 + get_local_id(0) + 1] -
                             s_PATCH[      get_local_id(1) * 6 + get_local_id(0) + 1])
                         * dw;
        s_dx_bin[tid] = vx;
        s_dy_bin[tid] = vy;
    }
}
void reduce_sum25(
    volatile __local  float* sdata1,
    volatile __local  float* sdata2,
    volatile __local  float* sdata3,
    volatile __local  float* sdata4,
    int tid
)
{
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    // first step is to reduce from 25 to 16
    if (tid < 9)
    {
        sdata1[tid] += sdata1[tid + 16];
        sdata2[tid] += sdata2[tid + 16];
        sdata3[tid] += sdata3[tid + 16];
        sdata4[tid] += sdata4[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        sdata1[tid] += sdata1[tid + 8];
        sdata2[tid] += sdata2[tid + 8];
        sdata3[tid] += sdata3[tid + 8];
        sdata4[tid] += sdata4[tid + 8];
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        sdata1[tid] += sdata1[tid + 4];
        sdata2[tid] += sdata2[tid + 4];
        sdata3[tid] += sdata3[tid + 4];
        sdata4[tid] += sdata4[tid + 4];
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        sdata1[tid] += sdata1[tid + 2];
        sdata2[tid] += sdata2[tid + 2];
        sdata3[tid] += sdata3[tid + 2];
        sdata4[tid] += sdata4[tid + 2];
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        sdata1[tid] += sdata1[tid + 1];
        sdata2[tid] += sdata2[tid + 1];
        sdata3[tid] += sdata3[tid + 1];
        sdata4[tid] += sdata4[tid + 1];
    }
#undef WAVE_SIZE
}

__kernel
void compute_descriptors64(
    IMAGE_INT8 imgTex,
    __global float * descriptors,
    __global const float * keypoints,
    int descriptors_step,
    int keypoints_step,
    int rows,
    int cols,
    int img_step
)
{
    descriptors_step /= sizeof(float);
    keypoints_step   /= sizeof(float);
    __global const float * featureX    = keypoints + X_ROW * keypoints_step;
    __global const float * featureY    = keypoints + Y_ROW * keypoints_step;
    __global const float * featureSize = keypoints + SIZE_ROW * keypoints_step;
    __global const float * featureDir  = keypoints + ANGLE_ROW * keypoints_step;

    // 2 floats (dx,dy) for each thread (5x5 sample points in each sub-region)
    volatile __local  float sdx[25];
    volatile __local  float sdy[25];
    volatile __local  float sdxabs[25];
    volatile __local  float sdyabs[25];
    volatile __local  float s_PATCH[6*6];

    calc_dx_dy(imgTex, sdx, sdy, s_PATCH, featureX, featureY, featureSize, featureDir, rows, cols, img_step);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    if (tid < 25)
    {
        sdxabs[tid] = fabs(sdx[tid]); // |dx| array
        sdyabs[tid] = fabs(sdy[tid]); // |dy| array
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce_sum25(sdx, sdy, sdxabs, sdyabs, tid);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 25)
    {
        __global float* descriptors_block = descriptors + descriptors_step * get_group_id(0) + (get_group_id(1) << 2);

        // write dx, dy, |dx|, |dy|
        if (tid == 0)
        {
            descriptors_block[0] = sdx[0];
            descriptors_block[1] = sdy[0];
            descriptors_block[2] = sdxabs[0];
            descriptors_block[3] = sdyabs[0];
        }
    }
}
__kernel
void compute_descriptors128(
    IMAGE_INT8 imgTex,
    __global float * descriptors,
    __global float * keypoints,
    int descriptors_step,
    int keypoints_step,
    int rows,
    int cols,
    int img_step
)
{
    descriptors_step /= sizeof(*descriptors);
    keypoints_step   /= sizeof(*keypoints);

    __global float * featureX   = keypoints + X_ROW * keypoints_step;
    __global float * featureY   = keypoints + Y_ROW * keypoints_step;
    __global float* featureSize = keypoints + SIZE_ROW * keypoints_step;
    __global float* featureDir  = keypoints + ANGLE_ROW * keypoints_step;

    // 2 floats (dx,dy) for each thread (5x5 sample points in each sub-region)
    volatile __local  float sdx[25];
    volatile __local  float sdy[25];

    // sum (reduce) 5x5 area response
    volatile __local  float sd1[25];
    volatile __local  float sd2[25];
    volatile __local  float sdabs1[25];
    volatile __local  float sdabs2[25];
    volatile __local  float s_PATCH[6*6];

    calc_dx_dy(imgTex, sdx, sdy, s_PATCH, featureX, featureY, featureSize, featureDir, rows, cols, img_step);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    if (tid < 25)
    {
        if (sdy[tid] >= 0)
        {
            sd1[tid] = sdx[tid];
            sdabs1[tid] = fabs(sdx[tid]);
            sd2[tid] = 0;
            sdabs2[tid] = 0;
        }
        else
        {
            sd1[tid] = 0;
            sdabs1[tid] = 0;
            sd2[tid] = sdx[tid];
            sdabs2[tid] = fabs(sdx[tid]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce_sum25(sd1, sd2, sdabs1, sdabs2, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    __global float* descriptors_block = descriptors + descriptors_step * get_group_id(0) + (get_group_id(1) << 3);
    if (tid < 25)
    {
        // write dx (dy >= 0), |dx| (dy >= 0), dx (dy < 0), |dx| (dy < 0)
        if (tid == 0)
        {
            descriptors_block[0] = sd1[0];
            descriptors_block[1] = sdabs1[0];
            descriptors_block[2] = sd2[0];
            descriptors_block[3] = sdabs2[0];
        }

        if (sdx[tid] >= 0)
        {
            sd1[tid] = sdy[tid];
            sdabs1[tid] = fabs(sdy[tid]);
            sd2[tid] = 0;
            sdabs2[tid] = 0;
        }
        else
        {
            sd1[tid] = 0;
            sdabs1[tid] = 0;
            sd2[tid] = sdy[tid];
            sdabs2[tid] = fabs(sdy[tid]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    reduce_sum25(sd1, sd2, sdabs1, sdabs2, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 25)
    {
        // write dy (dx >= 0), |dy| (dx >= 0), dy (dx < 0), |dy| (dx < 0)
        if (tid == 0)
        {
            descriptors_block[4] = sd1[0];
            descriptors_block[5] = sdabs1[0];
            descriptors_block[6] = sd2[0];
            descriptors_block[7] = sdabs2[0];
        }
    }
}

void reduce_sum128(volatile __local  float* smem, int tid)
{
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif

    if (tid < 64)
    {
        smem[tid] += smem[tid + 64];
#if WAVE_SIZE < 64
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32)
    {
#endif
        smem[tid] += smem[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem[tid] += smem[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem[tid] += smem[tid + 8];
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        smem[tid] += smem[tid + 4];
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        smem[tid] += smem[tid + 2];
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        smem[tid] += smem[tid + 1];
    }
}


void reduce_sum64(volatile __local  float* smem, int tid)
{
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    if (tid < 32)
    {
        smem[tid] += smem[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem[tid] += smem[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem[tid] += smem[tid + 8];
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        smem[tid] += smem[tid + 4];
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        smem[tid] += smem[tid + 2];
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        smem[tid] += smem[tid + 1];
    }
}

__kernel
void normalize_descriptors128(__global float * descriptors, int descriptors_step)
{
    descriptors_step /= sizeof(*descriptors);
    // no need for thread ID
    __global float* descriptor_base = descriptors + descriptors_step * get_group_id(0);

    // read in the unnormalized descriptor values (squared)
    volatile __local  float sqDesc[128];
    const float lookup = descriptor_base[get_local_id(0)];
    sqDesc[get_local_id(0)] = lookup * lookup;
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce_sum128(sqDesc, get_local_id(0));
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute length (square root)
    volatile __local  float len;
    if (get_local_id(0) == 0)
    {
        len = sqrt(sqDesc[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // normalize and store in output
    descriptor_base[get_local_id(0)] = lookup / len;
}
__kernel
void normalize_descriptors64(__global float * descriptors, int descriptors_step)
{
    descriptors_step /= sizeof(*descriptors);
    // no need for thread ID
    __global float* descriptor_base = descriptors + descriptors_step * get_group_id(0);

    // read in the unnormalized descriptor values (squared)
    volatile __local  float sqDesc[64];
    const float lookup = descriptor_base[get_local_id(0)];
    sqDesc[get_local_id(0)] = lookup * lookup;
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce_sum64(sqDesc, get_local_id(0));
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute length (square root)
    volatile __local  float len;
    if (get_local_id(0) == 0)
    {
        len = sqrt(sqDesc[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // normalize and store in output
    descriptor_base[get_local_id(0)] = lookup / len;
}
