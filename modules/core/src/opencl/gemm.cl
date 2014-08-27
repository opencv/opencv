// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define TSIZE (int)sizeof(T)

#define IND_A mad24(y, A_step, A_offset)
#define STEP_A 1

#define IND_B mad24(x, TSIZE, B_offset)
#define STEP_B B_step / TSIZE

#if cn==2
#define MUL(i, a, b)\
    {\
    sum.x += fma(a.x, b.x, - a.y * b.y);\
    sum.y += fma(a.x, b.y, a.y * b.x);\
    }
#else
#define MUL(i, a, b) sum = fma(a, b, sum);
#endif


__kernel void gemm(__global const uchar * A_ptr, int A_step, int A_offset,
                   __global const uchar * B_ptr, int B_step, int B_offset,
                   __global uchar * D_ptr, int D_step, int D_offset, int D_rows, int D_cols,
                   int n, T1 alpha, T1 beta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __global const T* A = (__global const T*)(A_ptr + IND_A);
    __global const T* B = (__global const T*)(B_ptr + IND_B);

    T sum = (T)(0);
    __local T a_local[LOCAL_SIZE*LOCAL_SIZE];
    __local T b_local[LOCAL_SIZE*LOCAL_SIZE];

    for (int p = 0; p < (n+LOCAL_SIZE-1)/LOCAL_SIZE; ++p)
    {
        a_local[mad24(lidy, LOCAL_SIZE, lidx)] = A[mad24(p, LOCAL_SIZE, lidx)];
        b_local[mad24(lidy, LOCAL_SIZE, lidx)] = B[mad24(p, LOCAL_SIZE, lidy)*STEP_B];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (x < D_cols && y < D_rows)
        {
            for (int i = 0; i < LOCAL_SIZE && p * LOCAL_SIZE + i < n; ++i)
                MUL(i, a_local[mad24(lidy, LOCAL_SIZE, i)], b_local[mad24(i, LOCAL_SIZE, lidx)]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < D_cols && y < D_rows)
    {
        __global T* D = (__global T*)(D_ptr + mad24(y, D_step, mad24(x, TSIZE, D_offset)));
#if HAVE_C
        D[0] = mad(alpha, sum, D[0]*beta);
#else
        D[0] = alpha * sum;
#endif
    }
}
