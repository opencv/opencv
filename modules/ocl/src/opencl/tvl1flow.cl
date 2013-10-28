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

__kernel void centeredGradientKernel(__global const float* src, int src_col, int src_row, int src_step,
__global float* dx, __global float* dy, int dx_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x < src_col)&&(y < src_row))
    {
        int src_x1 = (x + 1) < (src_col -1)? (x + 1) : (src_col - 1);
        int src_x2 = (x - 1) > 0 ? (x -1) : 0;

        //if(src[y * src_step + src_x1] == src[y * src_step+ src_x2])
        //{
        //    printf("y = %d\n", y);
        //    printf("src_x1 = %d\n", src_x1);
        //    printf("src_x2 = %d\n", src_x2);
        //}
        dx[y * dx_step+ x] = 0.5f * (src[y * src_step + src_x1] - src[y * src_step+ src_x2]);

        int src_y1 = (y+1) < (src_row - 1) ? (y + 1) : (src_row - 1);
        int src_y2 = (y - 1) > 0 ? (y - 1) : 0;
        dy[y * dx_step+ x] = 0.5f * (src[src_y1 * src_step + x] - src[src_y2 * src_step+ x]);
    }

}

float bicubicCoeff(float x_)
{

    float x = fabs(x_);
    if (x <= 1.0f)
    {
        return x * x * (1.5f * x - 2.5f) + 1.0f;
    }
    else if (x < 2.0f)
    {
        return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }

}

__kernel void warpBackwardKernel(__global const float* I0, int I0_step, int I0_col, int I0_row,
    image2d_t tex_I1, image2d_t tex_I1x, image2d_t tex_I1y,
    __global const float* u1, int u1_step,
    __global const float* u2,
    __global float* I1w,
    __global float* I1wx, /*int I1wx_step,*/
    __global float* I1wy, /*int I1wy_step,*/
    __global float* grad, /*int grad_step,*/
    __global float* rho,
    int I1w_step,
    int u2_step,
    int u1_offset_x,
    int u1_offset_y,
    int u2_offset_x,
    int u2_offset_y)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x < I0_col&&y < I0_row)
    {
        //const float u1Val = u1(y, x);
        const float u1Val = u1[(y + u1_offset_y) * u1_step + x + u1_offset_x];
        //const float u2Val = u2(y, x);
        const float u2Val = u2[(y + u2_offset_y) * u2_step + x + u2_offset_x];

        const float wx = x + u1Val;
        const float wy = y + u2Val;

        const int xmin = ceil(wx - 2.0f);
        const int xmax = floor(wx + 2.0f);

        const int ymin = ceil(wy - 2.0f);
        const int ymax = floor(wy + 2.0f);

        float sum  = 0.0f;
        float sumx = 0.0f;
        float sumy = 0.0f;
        float wsum = 0.0f;
        sampler_t sampleri = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            for (int cx = xmin; cx <= xmax; ++cx)
            {
                const float w = bicubicCoeff(wx - cx) * bicubicCoeff(wy - cy);

                //sum  += w * tex2D(tex_I1 , cx, cy);
                int2 cood = (int2)(cx, cy);
                sum += w * read_imagef(tex_I1, sampleri, cood).x;
                //sumx += w * tex2D(tex_I1x, cx, cy);
                sumx += w * read_imagef(tex_I1x, sampleri, cood).x;
                //sumy += w * tex2D(tex_I1y, cx, cy);
                sumy += w * read_imagef(tex_I1y, sampleri, cood).x;

                wsum += w;
            }
        }

        const float coeff = 1.0f / wsum;

        const float I1wVal  = sum  * coeff;
        const float I1wxVal = sumx * coeff;
        const float I1wyVal = sumy * coeff;

        I1w[y * I1w_step + x]  = I1wVal;
        I1wx[y * I1w_step + x] = I1wxVal;
        I1wy[y * I1w_step + x] = I1wyVal;

        const float Ix2 = I1wxVal * I1wxVal;
        const float Iy2 = I1wyVal * I1wyVal;

        // store the |Grad(I1)|^2
        grad[y * I1w_step + x] = Ix2 + Iy2;

        // compute the constant part of the rho function
        const float I0Val = I0[y * I0_step + x];
        rho[y * I1w_step + x] = I1wVal - I1wxVal * u1Val - I1wyVal * u2Val - I0Val;
    }

}

float readImage(__global const float *image,  const int x,  const int y,  const int rows,  const int cols, const int elemCntPerRow)
{
    int i0 = clamp(x, 0, cols - 1);
    int j0 = clamp(y, 0, rows - 1);
    int i1 = clamp(x + 1, 0, cols - 1);
    int j1 = clamp(y + 1, 0, rows - 1);

    return image[j0 * elemCntPerRow + i0];
}

__kernel void warpBackwardKernelNoImage2d(__global const float* I0, int I0_step, int I0_col, int I0_row,
    __global const float* tex_I1, __global const float* tex_I1x, __global const float* tex_I1y,
    __global const float* u1, int u1_step,
    __global const float* u2,
    __global float* I1w,
    __global float* I1wx, /*int I1wx_step,*/
    __global float* I1wy, /*int I1wy_step,*/
    __global float* grad, /*int grad_step,*/
    __global float* rho,
    int I1w_step,
    int u2_step,
    int I1_step,
    int I1x_step)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x < I0_col&&y < I0_row)
    {
        //const float u1Val = u1(y, x);
        const float u1Val = u1[y * u1_step + x];
        //const float u2Val = u2(y, x);
        const float u2Val = u2[y * u2_step + x];

        const float wx = x + u1Val;
        const float wy = y + u2Val;

        const int xmin = ceil(wx - 2.0f);
        const int xmax = floor(wx + 2.0f);

        const int ymin = ceil(wy - 2.0f);
        const int ymax = floor(wy + 2.0f);

        float sum  = 0.0f;
        float sumx = 0.0f;
        float sumy = 0.0f;
        float wsum = 0.0f;

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            for (int cx = xmin; cx <= xmax; ++cx)
            {
                const float w = bicubicCoeff(wx - cx) * bicubicCoeff(wy - cy);

                int2 cood = (int2)(cx, cy);
                sum += w * readImage(tex_I1, cood.x, cood.y, I0_col, I0_row, I1_step);
                sumx += w * readImage(tex_I1x, cood.x, cood.y, I0_col, I0_row, I1x_step);
                sumy += w * readImage(tex_I1y, cood.x, cood.y, I0_col, I0_row, I1x_step);
                wsum += w;
            }
        }

        const float coeff = 1.0f / wsum;

        const float I1wVal  = sum  * coeff;
        const float I1wxVal = sumx * coeff;
        const float I1wyVal = sumy * coeff;

        I1w[y * I1w_step + x]  = I1wVal;
        I1wx[y * I1w_step + x] = I1wxVal;
        I1wy[y * I1w_step + x] = I1wyVal;

        const float Ix2 = I1wxVal * I1wxVal;
        const float Iy2 = I1wyVal * I1wyVal;

        // store the |Grad(I1)|^2
        grad[y * I1w_step + x] = Ix2 + Iy2;

        // compute the constant part of the rho function
        const float I0Val = I0[y * I0_step + x];
        rho[y * I1w_step + x] = I1wVal - I1wxVal * u1Val - I1wyVal * u2Val - I0Val;
    }

}


__kernel void estimateDualVariablesKernel(__global const float* u1, int u1_col, int u1_row, int u1_step,
    __global const float* u2,
    __global float* p11, int p11_step,
    __global float* p12,
    __global float* p21,
    __global float* p22,
    const float taut,
    int u2_step,
    int u1_offset_x,
    int u1_offset_y,
    int u2_offset_x,
    int u2_offset_y)
{

    //const int x = blockIdx.x * blockDim.x + threadIdx.x;
    //const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x < u1_col && y < u1_row)
    {
        int src_x1 = (x + 1) < (u1_col - 1) ? (x + 1) : (u1_col - 1);
        const float u1x = u1[(y + u1_offset_y) * u1_step + src_x1 + u1_offset_x] - u1[(y + u1_offset_y) * u1_step + x + u1_offset_x];

        int src_y1 = (y + 1) < (u1_row - 1) ? (y + 1) : (u1_row - 1);
        const float u1y = u1[(src_y1 + u1_offset_y) * u1_step + x + u1_offset_x] - u1[(y + u1_offset_y) * u1_step + x + u1_offset_x];

        int src_x2 = (x + 1) < (u1_col - 1) ? (x + 1) : (u1_col - 1);
        const float u2x = u2[(y + u2_offset_y) * u2_step + src_x2 + u2_offset_x] - u2[(y + u2_offset_y) * u2_step + x + u2_offset_x];

        int src_y2 = (y + 1) <  (u1_row - 1) ? (y + 1) : (u1_row - 1);
        const float u2y = u2[(src_y2 + u2_offset_y) * u2_step + x + u2_offset_x] - u2[(y + u2_offset_y) * u2_step + x + u2_offset_x];

        const float g1 = hypot(u1x, u1y);
        const float g2 = hypot(u2x, u2y);

        const float ng1 = 1.0f + taut * g1;
        const float ng2 = 1.0f + taut * g2;

        p11[y * p11_step + x] = (p11[y * p11_step + x] + taut * u1x) / ng1;
        p12[y * p11_step + x] = (p12[y * p11_step + x] + taut * u1y) / ng1;
        p21[y * p11_step + x] = (p21[y * p11_step + x] + taut * u2x) / ng2;
        p22[y * p11_step + x] = (p22[y * p11_step + x] + taut * u2y) / ng2;
    }

}

float divergence(__global const float* v1, __global const float* v2, int y, int x, int v1_step, int v2_step)
{

    if (x > 0 && y > 0)
    {
        const float v1x = v1[y * v1_step + x] - v1[y * v1_step + x - 1];
        const float v2y = v2[y * v2_step + x] - v2[(y - 1) * v2_step + x];
        return v1x + v2y;
    }
    else
    {
        if (y > 0)
            return v1[y * v1_step + 0] + v2[y * v2_step + 0] - v2[(y - 1) * v2_step + 0];
        else
        {
            if (x > 0)
                return v1[0 * v1_step + x] - v1[0 * v1_step + x - 1] + v2[0 * v2_step + x];
            else
                return v1[0 * v1_step + 0] + v2[0 * v2_step + 0];
        }
    }

}

__kernel void estimateUKernel(__global const float* I1wx, int I1wx_col, int I1wx_row, int I1wx_step,
    __global const float* I1wy, /*int I1wy_step,*/
    __global const float* grad, /*int grad_step,*/
    __global const float* rho_c, /*int rho_c_step,*/
    __global const float* p11, /*int p11_step,*/
    __global const float* p12, /*int p12_step,*/
    __global const float* p21, /*int p21_step,*/
    __global const float* p22, /*int p22_step,*/
    __global float* u1, int u1_step,
    __global float* u2,
    __global float* error, const float l_t, const float theta, int u2_step,
    int u1_offset_x,
    int u1_offset_y,
    int u2_offset_x,
    int u2_offset_y,
    char calc_error)
{

    //const int x = blockIdx.x * blockDim.x + threadIdx.x;
    //const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = get_global_id(0);
    int y = get_global_id(1);


    if(x < I1wx_col && y < I1wx_row)
    {
        const float I1wxVal = I1wx[y * I1wx_step + x];
        const float I1wyVal = I1wy[y * I1wx_step + x];
        const float gradVal = grad[y * I1wx_step + x];
        const float u1OldVal = u1[(y + u1_offset_y) * u1_step + x + u1_offset_x];
        const float u2OldVal = u2[(y + u2_offset_y) * u2_step + x + u2_offset_x];

        const float rho = rho_c[y * I1wx_step + x] + (I1wxVal * u1OldVal + I1wyVal * u2OldVal);

        // estimate the values of the variable (v1, v2) (thresholding operator TH)

        float d1 = 0.0f;
        float d2 = 0.0f;

        if (rho < -l_t * gradVal)
        {
            d1 = l_t * I1wxVal;
            d2 = l_t * I1wyVal;
        }
        else if (rho > l_t * gradVal)
        {
            d1 = -l_t * I1wxVal;
            d2 = -l_t * I1wyVal;
        }
        else if (gradVal > 1.192092896e-07f)
        {
            const float fi = -rho / gradVal;
            d1 = fi * I1wxVal;
            d2 = fi * I1wyVal;
        }

        const float v1 = u1OldVal + d1;
        const float v2 = u2OldVal + d2;

        // compute the divergence of the dual variable (p1, p2)

        const float div_p1 = divergence(p11, p12, y, x, I1wx_step, I1wx_step);
        const float div_p2 = divergence(p21, p22, y, x, I1wx_step, I1wx_step);

        // estimate the values of the optical flow (u1, u2)

        const float u1NewVal = v1 + theta * div_p1;
        const float u2NewVal = v2 + theta * div_p2;

        u1[(y + u1_offset_y) * u1_step + x + u1_offset_x] = u1NewVal;
        u2[(y + u2_offset_y) * u2_step + x + u2_offset_x] = u2NewVal;

        if(calc_error)
        {
            const float n1 = (u1OldVal - u1NewVal) * (u1OldVal - u1NewVal);
            const float n2 = (u2OldVal - u2NewVal) * (u2OldVal - u2NewVal);
            error[y * I1wx_step + x] = n1 + n2;
        }
    }

}
