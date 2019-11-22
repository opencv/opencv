// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#define SQRT_2 0.707106781188f
#define sin_120 0.866025403784f
#define fft5_2  0.559016994374f
#define fft5_3 -0.951056516295f
#define fft5_4 -1.538841768587f
#define fft5_5  0.363271264002f

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__attribute__((always_inline))
CT mul_complex(CT a, CT b) {
    return (CT)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}

__attribute__((always_inline))
CT twiddle(CT a) {
    return (CT)(a.y, -a.x);
}

__attribute__((always_inline))
void butterfly2(CT a0, CT a1, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x & (block_size - 1);
    a1 = mul_complex(twiddles[k], a1);
    const int dst_ind = (x << 1) - k;

    smem[dst_ind] = a0 + a1;
    smem[dst_ind+block_size] = a0 - a1;
}

__attribute__((always_inline))
void butterfly4(CT a0, CT a1, CT a2, CT a3, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x & (block_size - 1);
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k + block_size], a2);
    a3 = mul_complex(twiddles[k + 2*block_size], a3);

    const int dst_ind = ((x - k) << 2) + k;

    CT b0 = a0 + a2;
    a2 = a0 - a2;
    CT b1 = a1 + a3;
    a3 = twiddle(a1 - a3);

    smem[dst_ind]                = b0 + b1;
    smem[dst_ind + block_size]   = a2 + a3;
    smem[dst_ind + 2*block_size] = b0 - b1;
    smem[dst_ind + 3*block_size] = a2 - a3;
}

__attribute__((always_inline))
void butterfly3(CT a0, CT a1, CT a2, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x % block_size;
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k+block_size], a2);
    const int dst_ind = ((x - k) * 3) + k;

    CT b1 = a1 + a2;
    a2 = twiddle(sin_120*(a1 - a2));
    CT b0 = a0 - (CT)(0.5f)*b1;

    smem[dst_ind] = a0 + b1;
    smem[dst_ind + block_size] = b0 + a2;
    smem[dst_ind + 2*block_size] = b0 - a2;
}

__attribute__((always_inline))
void butterfly5(CT a0, CT a1, CT a2, CT a3, CT a4, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x % block_size;
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k + block_size], a2);
    a3 = mul_complex(twiddles[k+2*block_size], a3);
    a4 = mul_complex(twiddles[k+3*block_size], a4);

    const int dst_ind = ((x - k) * 5) + k;
    __local CT* dst = smem + dst_ind;

    CT b0, b1, b5;

    b1 = a1 + a4;
    a1 -= a4;

    a4 = a3 + a2;
    a3 -= a2;

    a2 = b1 + a4;
    b0 = a0 - (CT)0.25f * a2;

    b1 = fft5_2 * (b1 - a4);
    a4 = fft5_3 * (CT)(-a1.y - a3.y, a1.x + a3.x);
    b5 = (CT)(a4.x - fft5_5 * a1.y, a4.y + fft5_5 * a1.x);

    a4.x += fft5_4 * a3.y;
    a4.y -= fft5_4 * a3.x;

    a1 = b0 + b1;
    b0 -= b1;

    dst[0] = a0 + a2;
    dst[block_size] = a1 + a4;
    dst[2 * block_size] = b0 + b5;
    dst[3 * block_size] = b0 - b5;
    dst[4 * block_size] = a1 - a4;
}

__attribute__((always_inline))
void fft_radix2(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1;

    if (x < t)
    {
        a0 = smem[x];
        a1 = smem[x+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly2(a0, a1, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x1 + 2*t/3;
    CT a0, a1, a2, a3, a4, a5;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B4(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/4;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x1 < t/4)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
        a6 = smem[x4]; a7 = smem[x4+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/4)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
        butterfly2(a6, a7, smem, twiddles, x4, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B5(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/5;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    const int x5 = x1 + 4*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    if (x1 < t/5)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
        a6 = smem[x4]; a7 = smem[x4+t];
        a8 = smem[x5]; a9 = smem[x5+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/5)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
        butterfly2(a6, a7, smem, twiddles, x4, block_size);
        butterfly2(a8, a9, smem, twiddles, x5, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1, a2, a3;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x+t]; a2 = smem[x+2*t]; a3 = smem[x+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly4(a0, a1, a2, a3, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t];
        a4 = smem[x2]; a5 = smem[x2+t]; a6 = smem[x2+2*t]; a7 = smem[x2+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly4(a0, a1, a2, a3, smem, twiddles, x1, block_size);
        butterfly4(a4, a5, a6, a7, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x2 + t/3;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t];
        a4 = smem[x2]; a5 = smem[x2+t]; a6 = smem[x2+2*t]; a7 = smem[x2+3*t];
        a8 = smem[x3]; a9 = smem[x3+t]; a10 = smem[x3+2*t]; a11 = smem[x3+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly4(a0, a1, a2, a3, smem, twiddles, x1, block_size);
        butterfly4(a4, a5, a6, a7, smem, twiddles, x2, block_size);
        butterfly4(a8, a9, a10, a11, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix8(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x < t)
    {
        int tw_ind = block_size / 8;

        a0 = smem[x];
        a1 = mul_complex(twiddles[k], smem[x + t]);
        a2 = mul_complex(twiddles[k + block_size],smem[x+2*t]);
        a3 = mul_complex(twiddles[k+2*block_size],smem[x+3*t]);
        a4 = mul_complex(twiddles[k+3*block_size],smem[x+4*t]);
        a5 = mul_complex(twiddles[k+4*block_size],smem[x+5*t]);
        a6 = mul_complex(twiddles[k+5*block_size],smem[x+6*t]);
        a7 = mul_complex(twiddles[k+6*block_size],smem[x+7*t]);

        CT b0, b1, b6, b7;

        b0 = a0 + a4;
        a4 = a0 - a4;
        b1 = a1 + a5;
        a5 = a1 - a5;
        a5 = (CT)(SQRT_2) * (CT)(a5.x + a5.y, -a5.x + a5.y);
        b6 = twiddle(a2 - a6);
        a2 = a2 + a6;
        b7 = a3 - a7;
        b7 = (CT)(SQRT_2) * (CT)(-b7.x + b7.y, -b7.x - b7.y);
        a3 = a3 + a7;

        a0 = b0 + a2;
        a2 = b0 - a2;
        a1 = b1 + a3;
        a3 = twiddle(b1 - a3);
        a6 = a4 - b6;
        a4 = a4 + b6;
        a7 = twiddle(a5 - b7);
        a5 = a5 + b7;

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) << 3) + k;
        __local CT* dst = smem + dst_ind;

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
void fft_radix3(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1, a2;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x+t]; a2 = smem[x+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly3(a0, a1, a2, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3, a4, a5;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x2 + t/3;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
        a6 = smem[x3]; a7 = smem[x3+t]; a8 = smem[x3+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
        butterfly3(a6, a7, a8, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B4(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/4;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;

    if (x1 < t/4)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
        a6 = smem[x3]; a7 = smem[x3+t]; a8 = smem[x3+2*t];
        a9 = smem[x4]; a10 = smem[x4+t]; a11 = smem[x4+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/4)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
        butterfly3(a6, a7, a8, smem, twiddles, x3, block_size);
        butterfly3(a9, a10, a11, smem, twiddles, x4, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix5(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    CT a0, a1, a2, a3, a4;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x + t]; a2 = smem[x+2*t]; a3 = smem[x+3*t]; a4 = smem[x+4*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly5(a0, a1, a2, a3, a4, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix5_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1+t/2;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1 + t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t]; a4 = smem[x1+4*t];
        a5 = smem[x2]; a6 = smem[x2 + t]; a7 = smem[x2+2*t]; a8 = smem[x2+3*t]; a9 = smem[x2+4*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly5(a0, a1, a2, a3, a4, smem, twiddles, x1, block_size);
        butterfly5(a5, a6, a7, a8, a9, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef DFT_SCALE
#define SCALE_VAL(x, scale) x*scale
#else
#define SCALE_VAL(x, scale) x
#endif

__kernel void fft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    __local CT smem[LOCAL_SIZE];  // used in (y < nz) code branch only, but should be declared in the outermost scope of a kernel function
    if (y < nz)
    {
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;
#ifdef IS_1D
        FT scale = (FT) 1/dst_cols;
#else
        FT scale = (FT) 1/(dst_cols*dst_rows);
#endif

#ifdef COMPLEX_INPUT
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = src[i*block_size];
#else
        __global const FT* src = (__global const FT*)(src_ptr + mad24(y, src_step, mad24(x, (int)sizeof(FT), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = (CT)(src[i*block_size], 0.f);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

#ifdef COMPLEX_OUTPUT
#ifdef NO_CONJUGATE
        // copy result without complex conjugate
        const int cols = dst_cols/2 + 1;
#else
        const int cols = dst_cols;
#endif

        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
        #pragma unroll
        for (int i=x; i<cols; i+=block_size)
            dst[i] = SCALE_VAL(smem[i], scale);
#ifdef REAL_INPUT
#ifdef COMPLEX_OUTPUT
#ifdef IS_1D
        for(int i=x+1; i < (dst_cols+1)/2; i+=block_size)
        {
            dst[dst_cols-i] = (CT)(SCALE_VAL(smem[i].x, scale), SCALE_VAL(-smem[i].y, scale));
        }
#endif
#endif
#endif
#else
        // pack row to CCS
        __local FT* smem_1cn = (__local FT*) smem;
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
        for (int i=x; i<dst_cols-1; i+=block_size)
            dst[i+1] = SCALE_VAL(smem_1cn[i+2], scale);
        if (x == 0)
            dst[0] = SCALE_VAL(smem_1cn[0], scale);
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#endif
        #pragma unroll
        for (int i=x; i<dst_cols; i+=block_size)
            dst[i] = 0.f;
    }
}

__kernel void fft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_group_id(0);
    const int y = get_global_id(1);

    __local CT smem[LOCAL_SIZE];  // used in (x < nz) code branch only, but should be declared in the outermost scope of a kernel function
    if (x < nz)
    {
        __global const uchar* src = src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset));
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;
        FT scale = 1.f/(dst_rows*dst_cols);

        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[y+i*block_size] = *((__global const CT*)(src + i*block_size*src_step));

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

#ifdef COMPLEX_OUTPUT
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            *((__global CT*)(dst + i*block_size*dst_step)) = SCALE_VAL(smem[y + i*block_size], scale);
#else
        if (x == 0)
        {
            // pack first column to CCS
            __local FT* smem_1cn = (__local FT*) smem;
            __global uchar* dst = dst_ptr + mad24(y+1, dst_step, dst_offset);
            for (int i=y; i<dst_rows-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + dst_offset)) = SCALE_VAL(smem_1cn[0], scale);
        }
        else if (x == (dst_cols+1)/2)
        {
            // pack last column to CCS (if needed)
            __local FT* smem_1cn = (__local FT*) smem;
            __global uchar* dst = dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), mad24(y+1, dst_step, dst_offset));
            for (int i=y; i<dst_rows-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), dst_offset))) = SCALE_VAL(smem_1cn[0], scale);
        }
        else
        {
            __global uchar* dst = dst_ptr + mad24(x, (int)sizeof(FT)*2, mad24(y, dst_step, dst_offset - (int)sizeof(FT)));
            #pragma unroll
            for (int i=y; i<dst_rows; i+=block_size, dst+=block_size*dst_step)
                vstore2(SCALE_VAL(smem[i], scale), 0, (__global FT*) dst);
        }
#endif
    }
}

__kernel void ifft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                    __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                    __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
#ifdef IS_1D
    const FT scale = (FT) 1/dst_cols;
#else
    const FT scale = (FT) 1/(dst_cols*dst_rows);
#endif

    __local CT smem[LOCAL_SIZE];  // used in (y < nz) code branch only, but should be declared in the outermost scope of a kernel function
    if (y < nz)
    {
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;

#if defined(COMPLEX_INPUT) && !defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            smem[x+i*block_size].x =  src[i*block_size].x;
            smem[x+i*block_size].y = -src[i*block_size].y;
        }
#else

    #if !defined(REAL_INPUT) && defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(2, (int)sizeof(FT), src_offset)));

        #pragma unroll
        for (int i=x; i<(LOCAL_SIZE-1)/2; i+=block_size)
        {
            smem[i+1].x = src[i].x;
            smem[i+1].y = -src[i].y;
            smem[LOCAL_SIZE-i-1] = src[i];
        }
    #else

        #pragma unroll
        for (int i=x; i<(LOCAL_SIZE-1)/2; i+=block_size)
        {
            CT src = vload2(0, (__global const FT*)(src_ptr + mad24(y, src_step, mad24(2*i+1, (int)sizeof(FT), src_offset))));

            smem[i+1].x = src.x;
            smem[i+1].y = -src.y;
            smem[LOCAL_SIZE-i-1] = src;
        }

    #endif

        if (x==0)
        {
            smem[0].x = *(__global const FT*)(src_ptr + mad24(y, src_step, src_offset));
            smem[0].y = 0.f;

            if(LOCAL_SIZE % 2 ==0)
            {
                #if !defined(REAL_INPUT) && defined(NO_CONJUGATE)
                smem[LOCAL_SIZE/2].x = src[LOCAL_SIZE/2-1].x;
                #else
                smem[LOCAL_SIZE/2].x = *(__global const FT*)(src_ptr + mad24(y, src_step, mad24(LOCAL_SIZE-1, (int)sizeof(FT), src_offset)));
                #endif
                smem[LOCAL_SIZE/2].y = 0.f;
            }
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
#ifdef COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            dst[i*block_size].x = SCALE_VAL(smem[x + i*block_size].x, scale);
            dst[i*block_size].y = SCALE_VAL(-smem[x + i*block_size].y, scale);
        }
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(FT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            dst[i*block_size] = SCALE_VAL(smem[x + i*block_size].x, scale);
        }
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#endif
        #pragma unroll
        for (int i=x; i<dst_cols; i+=block_size)
            dst[i] = 0.f;
    }
}

__kernel void ifft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_group_id(0);
    const int y = get_global_id(1);

    __local CT smem[LOCAL_SIZE];  // used in (x < nz) code branch only, but should be declared in the outermost scope of a kernel function
    if (x < nz)
    {
#ifdef COMPLEX_INPUT
        __global const uchar* src = src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset));
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            CT temp = *((__global const CT*)(src + i*block_size*src_step));
            smem[y+i*block_size].x =  temp.x;
            smem[y+i*block_size].y =  -temp.y;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
           __global CT* res = (__global CT*)(dst + i*block_size*dst_step);
            res[0].x = smem[y + i*block_size].x;
            res[0].y = -smem[y + i*block_size].y;
        }
#else
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;

#ifdef EVEN
        if (x!=0 && (x!=(nz-1)))
#else
        if (x!=0)
#endif
        {
            __global const uchar* src = src_ptr + mad24(y, src_step, mad24(2*x-1, (int)sizeof(FT), src_offset));
            #pragma unroll
            for (int i=0; i<kercn; i++)
            {
                CT temp = vload2(0, (__global const FT*)(src + i*block_size*src_step));
                smem[y+i*block_size].x = temp.x;
                smem[y+i*block_size].y = -temp.y;
            }
        }
        else
        {
            int ind = x==0 ? 0: 2*x-1;
            __global const FT* src = (__global const FT*)(src_ptr + mad24(1, src_step, mad24(ind, (int)sizeof(FT), src_offset)));
            int step = src_step/(int)sizeof(FT);

            #pragma unroll
            for (int i=y; i<(LOCAL_SIZE-1)/2; i+=block_size)
            {
                smem[i+1].x = src[2*i*step];
                smem[i+1].y = -src[(2*i+1)*step];

                smem[LOCAL_SIZE-i-1].x = src[2*i*step];;
                smem[LOCAL_SIZE-i-1].y = src[(2*i+1)*step];
            }
            if (y==0)
            {
                smem[0].x = *(__global const FT*)(src_ptr + mad24(ind, (int)sizeof(FT), src_offset));
                smem[0].y = 0.f;

                if(LOCAL_SIZE % 2 ==0)
                {
                    smem[LOCAL_SIZE/2].x = src[(LOCAL_SIZE-2)*step];
                    smem[LOCAL_SIZE/2].y = 0.f;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            __global CT* res = (__global CT*)(dst + i*block_size*dst_step);
            res[0].x =  smem[y + i*block_size].x;
            res[0].y = -smem[y + i*block_size].y;
        }
#endif
    }
}
