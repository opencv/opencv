#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void countNoneZero(__global const uchar * srcptr, int src_step, int src_offset, int rows, int cols,
                            __global uchar * bufptr, int buf_step, int buf_offset)
{
    int x = get_global_id(0);
    if (x < cols)
    {
        int src_index = mad24(x, (int)sizeof(srcT), src_offset);
        __global int * buf = (__global int *)(bufptr + buf_offset);
        int temp = 0;
        for (int y = 0; y < rows; ++y, src_index += src_step)
        {
            __global const srcT *src = (__global const srcT *)(srcptr + src_index);
            if (0 != (*src))
                temp++;
        }
        buf[x] = temp;
    }
}

#ifndef BUF_COLS
#define BUF_COLS  32
#endif

__kernel void sumLine(__global uchar * bufptr, int buf_step, int buf_offset, int rows, int cols)
{
    int x = get_global_id(0);
    if (x < BUF_COLS)
    {
        int src_index = mad24(x, 4, buf_offset);
        int src_last = mad24(cols, 4, buf_offset);
         __global int * src = (__global int *)(bufptr + src_index);
         __global int * dst = (__global int *)(bufptr + src_index);
         __global int * srcend = (__global int *)(bufptr + src_last);
        int temp = 0;
        for (; src < srcend; src += BUF_COLS)
        {
            temp = add_sat(temp, src[0]);
        }
        dst[0] = temp;
    }
}
