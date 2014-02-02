// OpenCL port of the FAST corner detector.
// Copyright (C) 2014, Itseez Inc. See the license at http://opencv.org

inline int cornerScore(__global const uchar* img, int step)
{
    int k, tofs, v = img[0], a0 = 0, b0;
    int d[16];
    #define LOAD2(idx, ofs) \
        tofs = ofs; d[idx] = (short)(v - img[tofs]); d[idx+8] = (short)(v - img[-tofs])
    LOAD2(0, 3);
    LOAD2(1, -step+3);
    LOAD2(2, -step*2+2);
    LOAD2(3, -step*3+1);
    LOAD2(4, -step*3);
    LOAD2(5, -step*3-1);
    LOAD2(6, -step*2-2);
    LOAD2(7, -step-3);

    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int a = min((int)d[(k+1)&15], (int)d[(k+2)&15]);
        a = min(a, (int)d[(k+3)&15]);
        a = min(a, (int)d[(k+4)&15]);
        a = min(a, (int)d[(k+5)&15]);
        a = min(a, (int)d[(k+6)&15]);
        a = min(a, (int)d[(k+7)&15]);
        a = min(a, (int)d[(k+8)&15]);
        a0 = max(a0, min(a, (int)d[k&15]));
        a0 = max(a0, min(a, (int)d[(k+9)&15]));
    }

    b0 = -a0;
    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int b = max((int)d[(k+1)&15], (int)d[(k+2)&15]);
        b = max(b, (int)d[(k+3)&15]);
        b = max(b, (int)d[(k+4)&15]);
        b = max(b, (int)d[(k+5)&15]);
        b = max(b, (int)d[(k+6)&15]);
        b = max(b, (int)d[(k+7)&15]);
        b = max(b, (int)d[(k+8)&15]);

        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[(k+9)&15]));
    }

    return -b0-1;
}

__kernel
void FAST_findKeypoints(
    __global const uchar * _img, int step, int img_offset,
    int img_rows, int img_cols,
    volatile __global int* kp_loc,
    int max_keypoints, int threshold )
{
    int j = get_global_id(0) + 3;
    int i = get_global_id(1) + 3;

    if (i < img_rows - 3 && j < img_cols - 3)
    {
        __global const uchar* img = _img + mad24(i, step, j + img_offset);
        int v = img[0], t0 = v - threshold, t1 = v + threshold;
        int k, tofs, v0, v1;
        int m0 = 0, m1 = 0;

        #define UPDATE_MASK(idx, ofs) \
            tofs = ofs; v0 = img[tofs]; v1 = img[-tofs]; \
            m0 |= ((v0 < t0) << idx) | ((v1 < t0) << (8 + idx)); \
            m1 |= ((v0 > t1) << idx) | ((v1 > t1) << (8 + idx))

        UPDATE_MASK(0, 3);
        if( (m0 | m1) == 0 )
            return;

        UPDATE_MASK(2, -step*2+2);
        UPDATE_MASK(4, -step*3);
        UPDATE_MASK(6, -step*2-2);

        #define EVEN_MASK (1+4+16+64)

        if( ((m0 | (m0 >> 8)) & EVEN_MASK) != EVEN_MASK &&
            ((m1 | (m1 >> 8)) & EVEN_MASK) != EVEN_MASK )
            return;

        UPDATE_MASK(1, -step+3);
        UPDATE_MASK(3, -step*3+1);
        UPDATE_MASK(5, -step*3-1);
        UPDATE_MASK(7, -step-3);
        if( ((m0 | (m0 >> 8)) & 255) != 255 &&
            ((m1 | (m1 >> 8)) & 255) != 255 )
            return;

        m0 |= m0 << 16;
        m1 |= m1 << 16;

        #define CHECK0(i) ((m0 & (511 << i)) == (511 << i))
        #define CHECK1(i) ((m1 & (511 << i)) == (511 << i))

        if( CHECK0(0) + CHECK0(1) + CHECK0(2) + CHECK0(3) +
            CHECK0(4) + CHECK0(5) + CHECK0(6) + CHECK0(7) +
            CHECK0(8) + CHECK0(9) + CHECK0(10) + CHECK0(11) +
            CHECK0(12) + CHECK0(13) + CHECK0(14) + CHECK0(15) +

            CHECK1(0) + CHECK1(1) + CHECK1(2) + CHECK1(3) +
            CHECK1(4) + CHECK1(5) + CHECK1(6) + CHECK1(7) +
            CHECK1(8) + CHECK1(9) + CHECK1(10) + CHECK1(11) +
            CHECK1(12) + CHECK1(13) + CHECK1(14) + CHECK1(15) == 0 )
            return;

        {
            int idx = atomic_inc(kp_loc);
            if( idx < max_keypoints )
            {
                kp_loc[1 + 2*idx] = j;
                kp_loc[2 + 2*idx] = i;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSupression

__kernel
void FAST_nonmaxSupression(
    __global const int* kp_in, volatile __global int* kp_out,
    __global const uchar * _img, int step, int img_offset,
    int rows, int cols, int counter, int max_keypoints)
{
    const int idx = get_global_id(0);

    if (idx < counter)
    {
        int x = kp_in[1 + 2*idx];
        int y = kp_in[2 + 2*idx];
        __global const uchar* img = _img + mad24(y, step, x + img_offset);

        int s = cornerScore(img, step);

        if( (x < 4 || s > cornerScore(img-1, step)) +
            (y < 4 || s > cornerScore(img-step, step)) != 2 )
            return;
        if( (x >= cols - 4 || s > cornerScore(img+1, step)) +
            (y >= rows - 4 || s > cornerScore(img+step, step)) +
            (x < 4 || y < 4 || s > cornerScore(img-step-1, step)) +
            (x >= cols - 4 || y < 4 || s > cornerScore(img-step+1, step)) +
            (x < 4 || y >= rows - 4 || s > cornerScore(img+step-1, step)) +
            (x >= cols - 4 || y >= rows - 4 || s > cornerScore(img+step+1, step)) == 6)
        {
            int new_idx = atomic_inc(kp_out);
            if( new_idx < max_keypoints )
            {
                kp_out[1 + 3*new_idx] = x;
                kp_out[2 + 3*new_idx] = y;
                kp_out[3 + 3*new_idx] = s;
            }
        }
    }
}
