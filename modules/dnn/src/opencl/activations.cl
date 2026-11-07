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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)
#define KERNEL_ARG_DTYPE float

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(M_SQRT1_2)
#define M_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2)      */
#endif

__kernel void ReLUForward(const int count, __global const T* in, __global T* out
#ifndef RELU_NO_SLOPE
, KERNEL_ARG_DTYPE negative_slope
#endif
) {
  int index = get_global_id(0);
  if(index < count)
#ifndef RELU_NO_SLOPE
  out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
#else
  out[index] = in[index] > 0 ? in[index] : 0;
#endif
}

__kernel void ReLU6Forward(const int count, __global const T* in, __global T* out,
                           const KERNEL_ARG_DTYPE minValue, const KERNEL_ARG_DTYPE maxValue)
{
  int index = get_global_id(0);
  if(index < count)
  {
    T x = in[index];
    out[index] = clamp(x, convert_T(minValue), convert_T(maxValue));
  }
}

__kernel void ChannelsPReLUForward(const int count, const int channels, const int plane_size,
                                   __global const T* in, __global T* out,
                                   __global const KERNEL_ARG_DTYPE* slope_data)
{
  int index = get_global_id(0);
  int c = (index / plane_size) % channels;
  if(index < count)
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
}

__kernel void PReLUForward(const int count, const int channels, const int plane_size,
                           __global const T* in, __global T* out,
                           __global const KERNEL_ARG_DTYPE* slope_data)
{
  int index = get_global_id(0);
  if(index < count)
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[index];
}

__kernel void TanHForward(const int count, __global T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = tanh(in[index]);
}

__kernel void SigmoidForward(const int count, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = 1.0f / (1.0f + exp(-in[index]));
}

__kernel void SwishForward(const int count, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = in[index] / (1.0f + exp(-in[index]));
}

__kernel void MishForward(const int count, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = in[index] * tanh(log(1.0f + exp(in[index])));
}

__kernel void BNLLForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n) {
    T x = in[index];
    out[index] = x > 0 ? x + log(1.0f + exp(-x)) : log(1.0f + exp(x));
  }
}

__kernel void AbsValForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n)
    out[index] = fabs(in[index]);
}

__kernel void PowForward(const int n, __global const T* in, __global T* out,
                         const KERNEL_ARG_DTYPE power,
                         const KERNEL_ARG_DTYPE scale,
                         const KERNEL_ARG_DTYPE shift)
{
  int index = get_global_id(0);
  if (index < n)
    out[index] = pow(shift + scale * in[index], power);
}

__kernel void ELUForward(const int n, __global const T* in, __global T* out,
                         const KERNEL_ARG_DTYPE alpha)
{
  int index = get_global_id(0);
  if (index < n)
  {
    T src = in[index];
    out[index] = (src >= 0.f) ? src : alpha * (exp(src) - 1);
  }
}

__kernel void ExpForward(const int n, __global const T* in, __global T* out,
                         const KERNEL_ARG_DTYPE normScale,
                         const KERNEL_ARG_DTYPE normShift)
{
  int index = get_global_id(0);
  if (index < n)
  {
    out[index] = exp(normShift + normScale * in[index]);
  }
}

__kernel void CeilForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = ceil(in[index]);
}

__kernel void FloorForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = floor(in[index]);
}

__kernel void LogForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = log(in[index]);
}

__kernel void RoundForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = rint(in[index]);
}

__kernel void SqrtForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = sqrt(in[index]);
}

__kernel void NotForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = floor(1.0f - in[index]);
}

__kernel void AcosForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = acos(in[index]);
}

__kernel void AcoshForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = acosh(in[index]);
}

__kernel void AsinForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = asin(in[index]);
}

__kernel void AsinhForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = asinh(in[index]);
}

__kernel void AtanForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = atan(in[index]);
}

__kernel void AtanhForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = atanh(in[index]);
}

__kernel void CosForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = cos(in[index]);
}

__kernel void CoshForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = cosh(in[index]);
}

__kernel void HardSwishForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = in[index] * max(0.f, min(1.f, in[index] / 6.f + 0.5f));
}

__kernel void SinForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = sin(in[index]);
}

__kernel void SinhForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = sinh(in[index]);
}

__kernel void SoftplusForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = log1p(exp(in[index]));
}

__kernel void SoftsignForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = in[index] / (1.f + fabs(in[index]));
}

__kernel void TanForward(const int n, __global T* in, __global T* out) {
    int index = get_global_id(0);
    if(index < n)
        out[index] = tan(in[index]);
}

__kernel void CeluForward(const int n, __global T* in, __global T* out,
                          const KERNEL_ARG_DTYPE alpha)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = max((T)0.f, in[index]) + (T)min(0.f, alpha * expm1(in[index] / alpha));
}

__kernel void HardSigmoidForward(const int n, __global T* in, __global T* out,
                                 const KERNEL_ARG_DTYPE alpha,
                                 const KERNEL_ARG_DTYPE beta)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = max((T)0.f, (T)min(1.f, alpha * in[index] + beta));
}

__kernel void SeluForward(const int n, __global T* in, __global T* out,
                          const KERNEL_ARG_DTYPE alpha,
                          const KERNEL_ARG_DTYPE gamma)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = gamma * (in[index] > 0.f ? in[index] : alpha * expm1(in[index]));
}

__kernel void ThresholdedReluForward(const int n, __global T* in, __global T* out,
                                     const KERNEL_ARG_DTYPE alpha)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = (in[index] > alpha ? in[index] : 0.f);
}

__kernel void GeluForward(const int n, __global T* in, __global T* out)
{
    int index = get_global_id(0);
    if (index < n)
    {
        T x = in[index];
        out[index] = (T)0.5f * x * ( (T)1.f + erf(x * M_SQRT1_2) );
    }
}

__kernel void GeluApproximationForward(const int n, __global T* in, __global T* out)
{
    // see GeluApproximationConstants from modules/dnn/src/layers/elementwise_layers.cpp
    const T sqrt_2_pi = 0.7978845834732056f;
    const T coef_sqrt_2_pi = 0.044714998453855515f * sqrt_2_pi;

    int index = get_global_id(0);
    if(index < n)
    {
        T x = in[index];
        out[index] = (T)0.5f * x * ( (T)1.f + tanh(x * (sqrt_2_pi + coef_sqrt_2_pi * x * x)) );
    }
}

__kernel void ShrinkForward(const int n, __global T* in, __global T* out,
                            const KERNEL_ARG_DTYPE bias,
                            const KERNEL_ARG_DTYPE lambd)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = in[index] < -lambd ? in[index] + bias : (in[index] > lambd ? in[index] - bias : 0.f);
}

__kernel void SignForward(const int n, __global T* in, __global T* out)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = in[index] > 0.f ? 1.0f : ((in[index] < 0.f) ? -1.0f : 0.0f);
}

__kernel void ReciprocalForward(const int n, __global T* in, __global T* out)
{
    int index = get_global_id(0);
    if(index < n)
        out[index] = 1.0f/in[index];
}
