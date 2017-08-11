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
#define Dtype float

__kernel void TEMPLATE(mul,Dtype)(const int n, __global const Dtype* a,
                                  const int offa,
                                  __global Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] * b[index + offb];
  }
}

__kernel void TEMPLATE(div,Dtype)(const int n, __global const Dtype* a,
                                  const int offa,
                                  __global Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] / b[index + offb];
  }
}

__kernel void TEMPLATE(add_scalar,Dtype)(const int N, const Dtype alpha,
__global Dtype* Y,
                                         const int offY) {
  for (int index = get_global_id(0); index < N; index += get_global_size(0)) {
    Y[offY + index] += alpha;
  }
}

__kernel void TEMPLATE(add,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global const Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] + b[offb + index];
  }
}

__kernel void TEMPLATE(sub,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global const Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] - b[offb + index];
  }
}

__kernel void TEMPLATE(abs,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = fabs((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(exp,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = exp(a[offa + index]);
  }
}

__kernel void TEMPLATE(log,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = log((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(powx,Dtype)(const int n, __global const Dtype* a,
                                   const int offa, Dtype alpha,
                                   __global Dtype* y,
                                   const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    if(alpha == 2.0) {
      y[offy + index] = pow((Dtype)fabs(a[offa + index]), (Dtype)alpha);
    } else {
      y[offy + index] = pow((Dtype)a[offa + index], (Dtype)alpha);
    }
  }
}

__kernel void TEMPLATE(sign,Dtype)(const int n, __global const Dtype* x,
                                   const int offx, __global Dtype* y,
                                   const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = (0.0 < x[index + offx])
        - (x[index + offx] < 0.0);
  }
}

__kernel void TEMPLATE(sgnbit,Dtype)(const int n, __global const Dtype* x,
                                     const int offx, __global Dtype* y,
                                     const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = signbit(x[index + offx]);
  }
}

__kernel void TEMPLATE(axpy,Dtype)(const int n, const Dtype alpha, __global const Dtype* x,
                                   const int offx, __global Dtype* y,
                                   const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype src = x[offx + index];
    Dtype dst = y[offy + index];
    y[offy + index] = alpha * src + dst;
  }
}
