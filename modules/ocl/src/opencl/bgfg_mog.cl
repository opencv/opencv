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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010,2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma jin@multicorewareinc.com
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

#if defined (CN1)
#define T_FRAME uchar
#define T_MEAN_VAR float
#define CONVERT_TYPE convert_uchar_sat
#define F_ZERO (0.0f)
float cvt(uchar val)
{
    return val;
}

float sqr(float val)
{
    return val * val;
}

float sum(float val)
{
    return val;
}

float clamp1(float var, float learningRate, float diff, float minVar)
{
    return fmax(var + learningRate * (diff * diff - var), minVar);
}

#else

#define T_FRAME uchar4
#define T_MEAN_VAR float4
#define CONVERT_TYPE convert_uchar4_sat
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)

float4 cvt(const uchar4 val)
{
    float4 result;
    result.x = val.x;
    result.y = val.y;
    result.z = val.z;
    result.w = val.w;

    return result;
}

float sqr(const float4 val)
{
    return val.x * val.x + val.y * val.y + val.z * val.z;
}

float sum(const float4 val)
{
    return (val.x + val.y + val.z);
}

void swap4(__global float4* ptr, int x, int y, int k, int rows, int ptr_step)
{
    float4 val = ptr[(k * rows + y) * ptr_step + x];
    ptr[(k * rows + y) * ptr_step + x] = ptr[((k + 1) * rows + y) * ptr_step + x];
    ptr[((k + 1) * rows + y) * ptr_step + x] = val;
}


float4 clamp1(const float4 var, float learningRate, const float4 diff, float minVar)
{
    float4 result;
    result.x = fmax(var.x + learningRate * (diff.x * diff.x - var.x), minVar);
    result.y = fmax(var.y + learningRate * (diff.y * diff.y - var.y), minVar);
    result.z = fmax(var.z + learningRate * (diff.z * diff.z - var.z), minVar);
    result.w = 0.0f;
    return result;
}

#endif

typedef struct
{
    float c_Tb;
    float c_TB;
    float c_Tg;
    float c_varInit;
    float c_varMin;
    float c_varMax;
    float c_tau;
    uchar c_shadowVal;
} con_srtuct_t;

void swap(__global float* ptr, int x, int y, int k, int rows, int ptr_step)
{
    float val = ptr[(k * rows + y) * ptr_step + x];
    ptr[(k * rows + y) * ptr_step + x] = ptr[((k + 1) * rows + y) * ptr_step + x];
    ptr[((k + 1) * rows + y) * ptr_step + x] = val;
}

__kernel void mog_withoutLearning_kernel(__global T_FRAME* frame, __global uchar* fgmask,
    __global float* weight, __global T_MEAN_VAR* mean, __global T_MEAN_VAR* var,
    int frame_row, int frame_col, int frame_step, int fgmask_step,
    int weight_step, int mean_step, int var_step,
    float varThreshold, float backgroundRatio, int fgmask_offset_x,
    int fgmask_offset_y, int frame_offset_x, int frame_offset_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < frame_col && y < frame_row)
    {
        T_MEAN_VAR pix = cvt(frame[(y + frame_offset_y) * frame_step + (x + frame_offset_x)]);

        int kHit = -1;
        int kForeground = -1;

        for (int k = 0; k < (NMIXTURES); ++k)
        {
            if (weight[(k * frame_row + y) * weight_step + x] < 1.192092896e-07f)
                break;

            T_MEAN_VAR mu = mean[(k * frame_row + y) * mean_step + x];
            T_MEAN_VAR _var = var[(k * frame_row + y) + var_step + x];

            T_MEAN_VAR diff = pix - mu;

            if (sqr(diff) < varThreshold * sum(_var))
            {
                kHit = k;
                break;
            }
        }

        if (kHit >= 0)
        {
            float wsum = 0.0f;
            for (int k = 0; k < (NMIXTURES); ++k)
            {
                wsum += weight[(k * frame_row + y) * weight_step + x];

                if (wsum > backgroundRatio)
                {
                    kForeground = k + 1;
                    break;
                }
            }
        }
        if(kHit < 0 || kHit >= kForeground)
            fgmask[(y + fgmask_offset_y) * fgmask_step + (x + fgmask_offset_x)] = (uchar) (-1);
        else
            fgmask[(y + fgmask_offset_y) * fgmask_step + (x + fgmask_offset_x)] = (uchar) (0);
    }
}

__kernel void mog_withLearning_kernel(__global T_FRAME* frame, __global int* fgmask,
    __global float* weight, __global float* sortKey, __global T_MEAN_VAR* mean,
    __global T_MEAN_VAR* var, int frame_row, int frame_col, int frame_step, int fgmask_step,
    int weight_step, int sortKey_step, int mean_step, int var_step,
    float varThreshold, float backgroundRatio, float learningRate, float minVar,
    int fgmask_offset_x, int fgmask_offset_y, int frame_offset_x, int frame_offset_y)
{
    const float w0 = 0.05f;
    const float sk0 = w0 / 30.0f;
    const float var0 = 900.f;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frame_col || y >= frame_row) return;
    float wsum = 0.0f;
    int kHit = -1;
    int kForeground = -1;
    int k = 0;

    T_MEAN_VAR pix = cvt(frame[(y + frame_offset_y) * frame_step + (x + frame_offset_x)]);

    for (; k < (NMIXTURES); ++k)
    {
        float w = weight[(k * frame_row + y) * weight_step + x];
        wsum += w;

        if (w < 1.192092896e-07f)
            break;

        T_MEAN_VAR mu = mean[(k * frame_row + y) * mean_step + x];
        T_MEAN_VAR _var = var[(k * frame_row + y) * var_step + x];

        float sortKey_prev, weight_prev;
        T_MEAN_VAR mean_prev, var_prev;
        if (sqr(pix - mu) < varThreshold * sum(_var))
        {
            wsum -= w;
            float dw = learningRate * (1.0f - w);

            _var = clamp1(_var, learningRate, pix - mu, minVar);

            sortKey_prev = w / sqr(sum(_var));
            sortKey[(k * frame_row + y) * sortKey_step + x] = sortKey_prev;

            weight_prev = w + dw;
            weight[(k * frame_row + y) * weight_step + x] = weight_prev;

            mean_prev = mu + learningRate * (pix - mu);
            mean[(k * frame_row + y) * mean_step + x] = mean_prev;

            var_prev = _var;
            var[(k * frame_row + y) * var_step + x] = var_prev;
        }

        int k1 = k - 1;

        if (k1 >= 0 && sqr(pix - mu) < varThreshold * sum(_var))
        {
            float sortKey_next = sortKey[(k1 * frame_row + y) * sortKey_step + x];
            float weight_next = weight[(k1 * frame_row + y) * weight_step + x];
            T_MEAN_VAR mean_next = mean[(k1 * frame_row + y) * mean_step + x];
            T_MEAN_VAR var_next = var[(k1 * frame_row + y) * var_step + x];

            for (; sortKey_next < sortKey_prev && k1 >= 0; --k1)
            {
                sortKey[(k1 * frame_row + y) * sortKey_step + x] = sortKey_prev;
                sortKey[((k1 + 1) * frame_row + y) * sortKey_step + x] = sortKey_next;

                weight[(k1 * frame_row + y) * weight_step + x] = weight_prev;
                weight[((k1 + 1) * frame_row + y) * weight_step + x] = weight_next;

                mean[(k1 * frame_row + y) * mean_step + x] = mean_prev;
                mean[((k1 + 1) * frame_row + y) * mean_step + x] = mean_next;

                var[(k1 * frame_row + y) * var_step + x] = var_prev;
                var[((k1 + 1) * frame_row + y) * var_step + x] = var_next;

                sortKey_prev = sortKey_next;
                sortKey_next = k1 > 0 ? sortKey[((k1 - 1) * frame_row + y) * sortKey_step + x] : 0.0f;

                weight_prev = weight_next;
                weight_next = k1 > 0 ? weight[((k1 - 1) * frame_row + y) * weight_step + x] : 0.0f;

                mean_prev = mean_next;
                mean_next = k1 > 0 ? mean[((k1 - 1) * frame_row + y) * mean_step + x] : (T_MEAN_VAR)F_ZERO;

                var_prev = var_next;
                var_next = k1 > 0 ? var[((k1 - 1) * frame_row + y) * var_step + x] : (T_MEAN_VAR)F_ZERO;
            }
        }

        kHit = k1 + 1;
        break;
    }

    if (kHit < 0)
    {
        kHit = k = k < ((NMIXTURES) - 1) ? k : ((NMIXTURES) - 1);
        wsum += w0 - weight[(k * frame_row + y) * weight_step + x];

        weight[(k * frame_row + y) * weight_step + x] = w0;
        mean[(k * frame_row + y) * mean_step + x] = pix;
#if defined (CN1)
        var[(k * frame_row + y) * var_step + x] = (T_MEAN_VAR)(var0);
#else
        var[(k * frame_row + y) * var_step + x] = (T_MEAN_VAR)(var0, var0, var0, var0);
#endif
        sortKey[(k * frame_row + y) * sortKey_step + x] = sk0;
    }
    else
    {
        for( ; k < (NMIXTURES); k++)
            wsum += weight[(k * frame_row + y) * weight_step + x];
    }

    float wscale = 1.0f / wsum;
    wsum = 0;
    for (k = 0; k < (NMIXTURES); ++k)
    {
        float w = weight[(k * frame_row + y) * weight_step + x];
        w *= wscale;
        wsum += w;

        weight[(k * frame_row + y) * weight_step + x] = w;
        sortKey[(k * frame_row + y) * sortKey_step + x] *= wscale;

        kForeground = select(kForeground, k + 1, wsum > backgroundRatio && kForeground < 0);
    }
    fgmask[(y + fgmask_offset_y) * fgmask_step + (x + fgmask_offset_x)] = (uchar)(-(kHit >= kForeground));
}


__kernel void getBackgroundImage_kernel(__global float* weight, __global T_MEAN_VAR* mean, __global T_FRAME* dst,
    int dst_row, int dst_col, int weight_step, int mean_step, int dst_step,
    float backgroundRatio)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < dst_col && y < dst_row)
    {
        T_MEAN_VAR meanVal = (T_MEAN_VAR)F_ZERO;
        float totalWeight = 0.0f;

        for (int mode = 0; mode < (NMIXTURES); ++mode)
        {
            float _weight = weight[(mode * dst_row + y) * weight_step + x];

            T_MEAN_VAR _mean = mean[(mode * dst_row + y) * mean_step + x];
            meanVal = meanVal + _weight * _mean;

            totalWeight += _weight;

            if(totalWeight > backgroundRatio)
                break;
        }
        meanVal = meanVal * (1.f / totalWeight);
        dst[y * dst_step + x] = CONVERT_TYPE(meanVal);
    }
}

__kernel void mog2_kernel(__global T_FRAME * frame, __global int* fgmask, __global float* weight, __global T_MEAN_VAR * mean,
        __global int* modesUsed, __global float* variance, int frame_row, int frame_col, int frame_step,
        int fgmask_step, int weight_step, int mean_step, int modesUsed_step, int var_step, float alphaT, float alpha1, float prune,
        int detectShadows_flag, int fgmask_offset_x, int fgmask_offset_y, int frame_offset_x, int frame_offset_y, __constant con_srtuct_t* constants)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < frame_col && y < frame_row)
    {
        T_MEAN_VAR pix = cvt(frame[(y + frame_offset_y) * frame_step + x + frame_offset_x]);

        bool background = false; // true - the pixel classified as background

        bool fitsPDF = false; //if it remains zero a new GMM mode will be added

        int nmodes = modesUsed[y * modesUsed_step + x];
        int nNewModes = nmodes; //current number of modes in GMM

        float totalWeight = 0.0f;

        for (int mode = 0; mode < nmodes; ++mode)
        {
            float _weight = alpha1 * weight[(mode * frame_row + y) * weight_step + x] + prune;
            int swap_count = 0;
            if (!fitsPDF)
            {
                float var = variance[(mode * frame_row + y) * var_step + x];

                T_MEAN_VAR _mean = mean[(mode * frame_row + y) * mean_step + x];

                T_MEAN_VAR diff = _mean - pix;
                float dist2 = sqr(diff);

                if (totalWeight < constants -> c_TB && dist2 < constants -> c_Tb * var)
                    background = true;

                if (dist2 < constants -> c_Tg * var)
                {
                    fitsPDF = true;
                    _weight += alphaT;
                    float k = alphaT / _weight;
                    mean[(mode * frame_row + y) * mean_step + x] = _mean - k * diff;
                    float varnew = var + k * (dist2 - var);
                    varnew = fmax(varnew, constants -> c_varMin);
                    varnew = fmin(varnew, constants -> c_varMax);

                    variance[(mode * frame_row + y) * var_step + x] = varnew;
                    for (int i = mode; i > 0; --i)
                    {
                        if (_weight < weight[((i - 1) * frame_row + y) * weight_step + x])
                            break;
                        swap_count++;
                        swap(weight, x, y, i - 1, frame_row, weight_step);
                        swap(variance, x, y, i - 1, frame_row, var_step);
                        #if defined (CN1)
                        swap(mean, x, y, i - 1, frame_row, mean_step);
                        #else
                        swap4(mean, x, y, i - 1, frame_row, mean_step);
                        #endif
                    }
                }
            } // !fitsPDF

            if (_weight < -prune)
            {
                _weight = 0.0f;
                nmodes--;
            }

            weight[((mode - swap_count) * frame_row + y) * weight_step + x] = _weight; //update weight by the calculated value
            totalWeight += _weight;
        }

        totalWeight = 1.f / totalWeight;
        for (int mode = 0; mode < nmodes; ++mode)
            weight[(mode * frame_row + y) * weight_step + x] *= totalWeight;

        nmodes = nNewModes;

        if (!fitsPDF)
        {
            int mode = nmodes == (NMIXTURES) ? (NMIXTURES) - 1 : nmodes++;

            if (nmodes == 1)
                weight[(mode * frame_row + y) * weight_step + x] = 1.f;
            else
            {
                weight[(mode * frame_row + y) * weight_step + x] = alphaT;

                for (int i = 0; i < nmodes - 1; ++i)
                    weight[(i * frame_row + y) * weight_step + x] *= alpha1;
            }

            mean[(mode * frame_row + y) * mean_step + x] = pix;
            variance[(mode * frame_row + y) * var_step + x] = constants -> c_varInit;

            for (int i = nmodes - 1; i > 0; --i)
            {
                // check one up
                if (alphaT < weight[((i - 1) * frame_row + y) * weight_step + x])
                    break;

                swap(weight, x, y, i - 1, frame_row, weight_step);
                swap(variance, x, y, i - 1, frame_row, var_step);
                #if defined (CN1)
                swap(mean, x, y, i - 1, frame_row, mean_step);
                #else
                swap4(mean, x, y, i - 1, frame_row, mean_step);
                #endif
            }
        }

        modesUsed[y * modesUsed_step + x] = nmodes;

        bool isShadow = false;
        if (detectShadows_flag && !background)
        {
            float tWeight = 0.0f;

            for (int mode = 0; mode < nmodes; ++mode)
            {
                T_MEAN_VAR _mean = mean[(mode * frame_row + y) * mean_step + x];

                T_MEAN_VAR pix_mean = pix * _mean;

                float numerator = sum(pix_mean);
                float denominator = sqr(_mean);

                if (denominator == 0)
                    break;

                if (numerator <= denominator && numerator >= constants -> c_tau * denominator)
                {
                    float a = numerator / denominator;

                    T_MEAN_VAR dD = a * _mean - pix;

                    if (sqr(dD) < constants -> c_Tb * variance[(mode * frame_row + y) * var_step + x] * a * a)
                    {
                        isShadow = true;
                        break;
                    }
                }

                tWeight += weight[(mode * frame_row + y) * weight_step + x];
                if (tWeight > constants -> c_TB)
                    break;
            }
        }

        fgmask[(y + fgmask_offset_y) * fgmask_step + x + fgmask_offset_x] = background ? 0 : isShadow ? constants -> c_shadowVal : 255;
    }
}

__kernel void getBackgroundImage2_kernel(__global int* modesUsed, __global float* weight, __global T_MEAN_VAR* mean,
    __global T_FRAME* dst, float c_TB, int modesUsed_row, int modesUsed_col, int modesUsed_step, int weight_step,
    int mean_step, int dst_step, int dst_x, int dst_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < modesUsed_col && y < modesUsed_row)
    {
        int nmodes = modesUsed[y * modesUsed_step + x];

        T_MEAN_VAR meanVal = (T_MEAN_VAR)F_ZERO;

        float totalWeight = 0.0f;

        for (int mode = 0; mode < nmodes; ++mode)
        {
            float _weight = weight[(mode * modesUsed_row + y) * weight_step + x];

            T_MEAN_VAR _mean = mean[(mode * modesUsed_row + y) * mean_step + x];
            meanVal = meanVal + _weight * _mean;

            totalWeight += _weight;

            if(totalWeight > c_TB)
                break;
        }

        meanVal = meanVal * (1.f / totalWeight);
        dst[(y + dst_y) * dst_step + x + dst_x] = CONVERT_TYPE(meanVal);
    }
}
