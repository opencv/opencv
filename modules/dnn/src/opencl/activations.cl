__kernel void ReLUForward(const int count, __global const T* in, __global T* out
#ifndef RELU_NO_SLOPE
, T negative_slope
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

__kernel void TanHForward(const int count, __global T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = tanh(in[index]);
}

__kernel void SigmoidForward(const int count, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] = 1. / (1. + exp(-in[index]));
}

__kernel void BNLLForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n) {
    out[index] = in[index] > 0 ? in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));
  }
}

__kernel void AbsValForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n)
    out[index] = fabs(in[index]);
}

__kernel void PowForward(const int n, __global const T* in, __global T* out, const T power, const T scale, const T shift) {
  int index = get_global_id(0);
  if (index < n)
    out[index] = pow(shift + scale * in[index], power);
}