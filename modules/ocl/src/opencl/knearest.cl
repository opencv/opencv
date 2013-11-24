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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma, jin@multicorewareinc.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define TYPE double
#else
#define TYPE float
#endif

#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))
///////////////////////////////////// find_nearest //////////////////////////////////////
__kernel void knn_find_nearest(__global float* sample, int sample_row, int sample_col, int sample_step,
                               int k, __global float* samples_ocl, int sample_ocl_row, int sample_ocl_step,
                               __global float* _results, int _results_step, int _regression, int K1,
                               int sample_ocl_col, int nThreads, __local float* nr)
{
    int k1 = 0;
    int k2 = 0;

    bool regression = false;

    if(_regression)
        regression = true;

    TYPE inv_scale;
#ifdef DOUBLE_SUPPORT
    inv_scale = 1.0/K1;
#else
    inv_scale = 1.0f/K1;
#endif

    int y = get_global_id(1);
    int j, j1;
    int threadY = (y % nThreads);
    __local float* dd = nr + nThreads * k;
    if(y >= sample_row)
    {
        return;
    }
    for(j = 0; j < sample_ocl_row; j++)
    {
        TYPE sum;
#ifdef DOUBLE_SUPPORT
        sum = 0.0;
#else
        sum = 0.0f;
#endif
        float si;
        int t, ii, ii1;
        for(t = 0; t < sample_col - 16; t += 16)
        {
            float16 t0 = vload16(0, sample + y * sample_step + t) - vload16(0, samples_ocl + j * sample_ocl_step + t);
            t0 *= t0;
            sum += t0.s0 + t0.s1 + t0.s2 + t0.s3 + t0.s4 + t0.s5 + t0.s6 + t0.s7 +
                t0.s8 + t0.s9 + t0.sa + t0.sb + t0.sc + t0.sd + t0.se + t0.sf;
        }

        for(; t < sample_col; t++)
        {
#ifdef DOUBLE_SUPPORT
            double t0 = sample[y * sample_step + t] - samples_ocl[j * sample_ocl_step + t];
#else
            float t0 = sample[y * sample_step + t] - samples_ocl[j * sample_ocl_step + t];
#endif
            sum = sum + t0 * t0;
        }

        si = (float)sum;
        for(ii = k1 - 1; ii >= 0; ii--)
        {
            if(as_int(si) > as_int(dd[ii * nThreads + threadY]))
                break;
        }
        if(ii < k - 1)
        {
            for(ii1 = k2 - 1; ii1 > ii; ii1--)
            {
                dd[(ii1 + 1) * nThreads + threadY] = dd[ii1 * nThreads + threadY];
                nr[(ii1 + 1) * nThreads + threadY] = nr[ii1 * nThreads + threadY];
            }

            dd[(ii + 1) * nThreads + threadY] = si;
            nr[(ii + 1) * nThreads + threadY] = samples_ocl[sample_col + j * sample_ocl_step];
        }
        k1 = (k1 + 1) < k ? (k1 + 1) : k;
        k2 = k1 < (k - 1) ? k1 : (k - 1);
    }
    /*! find_nearest_neighbor done!*/
    /*! write_results start!*/
    if (regression)
    {
        TYPE s;
#ifdef DOUBLE_SUPPORT
        s = 0.0;
#else
        s = 0.0f;
#endif
        for(j = 0; j < K1; j++)
            s += nr[j * nThreads + threadY];

        _results[y * _results_step] = (float)(s * inv_scale);
    }
    else
    {
        int prev_start = 0, best_count = 0, cur_count;
        float best_val;

        for(j = K1 - 1; j > 0; j--)
        {
            bool swap_f1 = false;
            for(j1 = 0; j1 < j; j1++)
            {
                if(nr[j1 * nThreads + threadY] > nr[(j1 + 1) * nThreads + threadY])
                {
                    int t;
                    CV_SWAP(nr[j1 * nThreads + threadY], nr[(j1 + 1) * nThreads + threadY], t);
                    swap_f1 = true;
                }
            }
            if(!swap_f1)
                break;
        }

        best_val = 0;
        for(j = 1; j <= K1; j++)
            if(j == K1 || nr[j * nThreads + threadY] != nr[(j - 1) * nThreads + threadY])
            {
                cur_count = j - prev_start;
                if(best_count < cur_count)
                {
                    best_count = cur_count;
                    best_val = nr[(j - 1) * nThreads + threadY];
                }
                prev_start = j;
            }
            _results[y * _results_step] = best_val;
    }
    ///*! write_results done!*/
}
