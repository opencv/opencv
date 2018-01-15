
__kernel void batchnorm(__global const T *src, int src_offset,
                        __global const float *meanMat,
                        float varMeanScale,
                        __global const float *invStdMat,
                        __global const float *weight,
                        __global const float *bias,
                        int hasWeight, int hasBias,
                        int width, int height, int channel,
                        __global T *dst, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);

    if (x >= width || y >= height || c >= channel)
        return;

    float mean = meanMat[c] * varMeanScale;
    float invstd = invStdMat[c];
    float w = hasWeight ? weight[c] : 1;
    float b = hasBias ? bias[c] : 0;
    int index = y * width + x + c * width * height;
    T val = (src[index + src_offset] - mean) * w * invstd + b;
    dst[index + dst_offset] = val;
}
