#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __constant
#define __local
#define get_global_id(x) 0
#define get_global_size(x) 0
#define get_local_id(x) 0
#define get_local_size(x) 0
#define FLT_MAX 0
#define FLT_MIN 0
#define cl_khr_fp64
#define cl_amd_fp64
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#define CLK_LOCAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE
#define Dtype float
#define barrier(x)
#define atomic_cmpxchg(x, y, z) x
#define signbit(x) x
#define int_tp long
#define uint_tp unsigned long
#define int_tpc long
#define uint_tpc unsigned long
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

#define TYPE_FLOAT 1
#define TYPE_DOUBLE 2

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#endif

#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define ATOMICS_64_AVAILABLE
#endif

#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

// Types used for parameters, offset computations and so on
#define int_tp int
#define uint_tp unsigned int

// Definitions used to cast the types above as needed
#define int_tpc int
#define uint_tpc unsigned int

#define Dtype float

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

__kernel void TEMPLATE(mul,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa,
                                  __global Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] * b[index + offb];
  }
}

__kernel void TEMPLATE(div,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa,
                                  __global Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] / b[index + offb];
  }
}

__kernel void TEMPLATE(add_scalar,Dtype)(const int_tp N, const Dtype alpha,
__global Dtype* Y,
                                         const int_tp offY) {
  for (int_tp index = get_global_id(0); index < N; index += get_global_size(0)) {
    Y[offY + index] += alpha;
  }
}

__kernel void TEMPLATE(add,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global const Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] + b[offb + index];
  }
}

__kernel void TEMPLATE(sub,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global const Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] - b[offb + index];
  }
}

__kernel void TEMPLATE(abs,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = fabs((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(exp,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = exp(a[offa + index]);
  }
}

__kernel void TEMPLATE(log,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = log((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(powx,Dtype)(const int_tp n, __global const Dtype* a,
                                   const int_tp offa, Dtype alpha,
                                   __global Dtype* y,
                                   const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    if(alpha == 2.0) {
      y[offy + index] = pow((Dtype)fabs(a[offa + index]), (Dtype)alpha);
    } else {
      y[offy + index] = pow((Dtype)a[offa + index], (Dtype)alpha);
    }
  }
}

__kernel void TEMPLATE(sign,Dtype)(const int_tp n, __global const Dtype* x,
                                   const int_tp offx, __global Dtype* y,
                                   const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = (0.0 < x[index + offx])
        - (x[index + offx] < 0.0);
  }
}

__kernel void TEMPLATE(sgnbit,Dtype)(const int_tp n, __global const Dtype* x,
                                     const int_tp offx, __global Dtype* y,
                                     const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = signbit(x[index + offx]);
  }
}

__kernel void TEMPLATE(axpy,Dtype)(const int_tp n, const Dtype alpha, __global const Dtype* x,
                                   const int_tp offx, __global Dtype* y,
                                   const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype src = x[offx + index];
    Dtype dst = y[offy + index];
    y[offy + index] = alpha * src + dst;
  }
}
