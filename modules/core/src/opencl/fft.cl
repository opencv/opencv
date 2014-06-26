__constant float PI = 3.14159265f;
__constant float SQRT_2 = 0.707106781188f;

__constant float sin_120 = 0.866025403784f;
__constant float fft5_2 =  0.559016994374f;
__constant float fft5_3 = -0.951056516295f;
__constant float fft5_4 = -1.538841768587f;
__constant float fft5_5 =  0.363271264002f;

inline float2 mul_float2(float2 a, float2 b){ 
    float2 res; 
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res; 
}

inline float2 sincos_float2(float alpha) {
    float cs, sn;
    sn = sincos(alpha, &cs);  // sincos
    return (float2)(cs, sn);
}

inline float2 twiddle(float2 a) { 
    return (float2)(a.y, -a.x); 
}

inline float2 square(float2 a) { 
    return (float2)(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y); 
}

inline float2 square3(float2 a) { 
    return (float2)(a.x * a.x - a.y * a.y, 3.0f * a.x * a.y); 
}

inline float2 mul_p1q4(float2 a) { 
    return (float2)(SQRT_2) * (float2)(a.x + a.y, -a.x + a.y); 
}

inline float2 mul_p3q4(float2 a) { 
    return (float2)(SQRT_2) * (float2)(-a.x + a.y, -a.x - a.y); 
}

__attribute__((always_inline))
void fft_radix2(__local float2* smem, const int x, const int block_size, const int t)       
{
    const int k = x & (block_size - 1);
    float2 in1, temp;

    if (x < t)
    {
        in1 = smem[x];
        float2 in2 = smem[x+t];

        float theta = -PI * k / block_size;
        float cs;
        float sn = sincos(theta, &cs);
        temp = (float2) (in2.x * cs - in2.y * sn,
                                      in2.y * cs + in2.x * sn);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = (x << 1) - k;
    
        smem[dst_ind] = in1 + temp;
        smem[dst_ind+block_size] = in1 - temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4(__local float2* smem, const int x, const int block_size, const int t)  
{
    const int k = x & (block_size - 1);
    float2 b0, b1, b2, b3;

    if (x < t)
    {
        float theta = -PI * k / (2 * block_size);

        float2 tw = sincos_float2(theta);
        float2 a0 = smem[x];
        float2 a1 = mul_float2(tw, smem[x+t]);
        float2 a2 = smem[x + 2*t];
        float2 a3 = mul_float2(tw, smem[x + 3*t]);
        tw = square(tw);
        a2 = mul_float2(tw, a2);
        a3 = mul_float2(tw, a3);

        b0 = a0 + a2;
        b1 = a0 - a2;
        b2 = a1 + a3;
        b3 = twiddle(a1 - a3);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) << 2) + k;
        smem[dst_ind] = b0 + b2;
        smem[dst_ind + block_size] = b1 + b3;
        smem[dst_ind + 2*block_size] = b0 - b2;
        smem[dst_ind + 3*block_size] = b1 - b3;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix8(__local float2* smem, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    float2 a0, a1, a2, a3, a4, a5, a6, a7;

    if (x < t)
    {
        float theta = -PI * k / (4 * block_size);

        float2 tw = sincos_float2(theta); // W
        a0 = smem[x];
        a1 = mul_float2(tw, smem[x + t]);
        a2 = smem[x + 2 * t];
        a3 = mul_float2(tw, smem[x + 3 * t]);
        a4 = smem[x + 4 * t];
        a5 = mul_float2(tw, smem[x + 5 * t]);
        a6 = smem[x + 6 * t];
        a7 = mul_float2(tw, smem[x + 7 * t]);

        tw = square(tw); // W^2
        a2 = mul_float2(tw, a2);
        a3 = mul_float2(tw, a3);
        a6 = mul_float2(tw, a6);
        a7 = mul_float2(tw, a7);
        tw = square(tw); // W^4
        a4 = mul_float2(tw, a4);
        a5 = mul_float2(tw, a5);
        a6 = mul_float2(tw, a6);
        a7 = mul_float2(tw, a7);

        float2 b0 = a0 + a4;
        float2 b4 = a0 - a4;
        float2 b1 = a1 + a5; 
        float2 b5 = mul_p1q4(a1 - a5);
        float2 b2 = a2 + a6;
        float2 b6 = twiddle(a2 - a6);
        float2 b3 = a3 + a7;
        float2 b7 = mul_p3q4(a3 - a7);

        a0 = b0 + b2;
        a2 = b0 - b2;
        a1 = b1 + b3;
        a3 = twiddle(b1 - b3);
        a4 = b4 + b6;
        a6 = b4 - b6;
        a5 = b5 + b7;
        a7 = twiddle(b5 - b7);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) << 3) + k;
        __local float2* dst = smem + dst_ind;

        dst[0] = a0 + a1;
        dst[block_size] = a4 + a5;
        dst[2 * block_size] = a2 + a3;
        dst[3 * block_size] = a6 + a7;
        dst[4 * block_size] = a0 - a1;
        dst[5 * block_size] = a4 - a5;
        dst[6 * block_size] = a2 - a3;
        dst[7 * block_size] = a6 - a7;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3(__local float2* smem, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    float2 a0, a1, a2, b0, b1;

    if (x < t)
    {
        const float theta = -PI * k * 2 / (3 * block_size);

        a0 = smem[x];
        a1 = mul_float2(sincos_float2(theta), smem[x+t]);
        a2 = mul_float2(sincos_float2(2 * theta), smem[x+2*t]);
        b1 = a1 + a2;
        a2 = twiddle((float2)sin_120*(a1 - a2));
        b0 = a0 - (float2)(0.5f)*b1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) * 3) + k;

        smem[dst_ind] = a0 + b1;
        smem[dst_ind + block_size] = b0 + a2;
        smem[dst_ind + 2*block_size] = b0 - a2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix5(__local float2* smem, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    float2 a0, a1, a2, a3, a4, b0, b1, b2, b5;

    if (x < t)
    {
        const float theta = -PI * k * 2 / (5 * block_size);
    
        a0 = smem[x];
        a1 = mul_float2(sincos_float2(theta), smem[x + t]);
        a2 = mul_float2(sincos_float2(theta*2),smem[x+2*t]);
        a3 = mul_float2(sincos_float2(theta*3),smem[x+3*t]);
        a4 = mul_float2(sincos_float2(theta*4),smem[x+4*t]);

        b1 = a1 + a4;
        a1 -= a4;

        a4 = a3 + a2;
        a3 -= a2;

        b2 = b1 + a4;
        b0 = a0 - (float2)0.25f * b2;

        b1 =  (float2)fft5_2 * (b1 - a4);
        a4 = -(float2)fft5_3 * (a1 + a3);
        a4 = twiddle(a4);

        b5 = (float2)(a4.x - fft5_5 * a1.y, a4.y + fft5_5 * a1.x);

        a4.x += fft5_4 * a3.y; 
        a4.y -= fft5_4 * a3.x;

        a1 = b0 + b1;
        b0 -= b1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) * 5) + k;
        __local float2* dst = smem + dst_ind;

        dst[0] = a0 + b2;
        dst[block_size] = a1 + a4;
        dst[2 * block_size] = b0 + b5;
        dst[3 * block_size] = b0 - b5;
        dst[4 * block_size] = a1 - a4;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void fft_multi_radix(__global const uchar* srcptr, int src_step, int src_offset,
                              __global uchar* dstptr, int dst_step, int dst_offset,
                              const int t, const int nz)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);

    if (y < nz)
    {
        __local float2 smem[LOCAL_SIZE];
        __global const float2* src = (__global const float2*)(srcptr + mad24(y, src_step, mad24(x, (int)(sizeof(float)*2), src_offset)));
        __global float2* dst = (__global float2*)(dstptr + mad24(y, dst_step, mad24(x, (int)(sizeof(float)*2), dst_offset)));

        const int block_size = LOCAL_SIZE/kercn;
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = src[i*block_size];

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
        #pragma unroll
        for (int i=0; i<kercn; i++)
            dst[i*block_size] = smem[x + i*block_size];
    }
}