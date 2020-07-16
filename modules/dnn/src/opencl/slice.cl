// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Specialization constants:
- WSZ: size of OpenCL local group
- DIMS: number of working dimensions
- ELEMSIZE: element size in bytes
- DST_SZ_<i>: dst sizes
- SRC_START_<i>: src index shift (slice .start value)
- SRC_STEP_<i>: src steps (bytes)
- DST_STEP_<i>: dst steps (bytes), derived from DST_SZ_<i> and ELEMSIZE
- BLOCK_DIMS: number of dims for copy block (argmax(count(SRC_STEP_<i> != DST_STEP_<i>) <= 1))
- BLOCK_DIMS_CONTIGUOUS (<= BLOCK_DIMS): SRC_STEP_<i> == DST_STEP_<i> for i in [0, BLOCK_DIMS_CONTIGUOUS)

derived specialization constants:
- BLOCK_SIZE: ELEMSIZE * mul(DST_SZ_<i>) for i in [0, BLOCK_DIMS)

- USE_COPY_1D iff BLOCK_DIMS == BLOCK_DIMS_CONTIGUOUS
- BLOCK_COLS:
  * with USE_COPY_1D: BLOCK_SIZE
  * w/o USE_COPY_1D: ELEMSIZE * mul(DST_SZ_<i>) for i in [0, BLOCK_DIMS_CONTIGUOUS)
- BLOCK_ROWS:
  * with USE_COPY_1D: N/A
  * w/o USE_COPY_1D: ELEMSIZE * mul(DST_SZ_<i>) for i in [BLOCK_DIMS_CONTIGUOUS, BLOCK_DIMS)
- BLOCK_SRC_STRIDE:
  * with USE_COPY_1D: N/A
  * w/o USE_COPY_1D: ELEMSIZE * mul(SRC_STEP_<i>) for i in [0, BLOCK_DIMS_CONTIGUOUS)

Note: SZ, STEP values are in reversed order than OpenCV Mat:
- NCHW SZ: [cols, rows, channels, batch]
- NCHW STEP: [elemsize, cols * elemsize, rows * cols * elemsize, ...] (DIMS+1 value)

*/

/*
local: <WSZ, 1, 1>
global: <WSZ, number_of_copy_blocks, 1>
*/

#define CONCAT_(A, B) A##B
#define CONCAT(A, B) CONCAT_(A, B)

#define BLOCK_COLS_X4 (BLOCK_COLS / 4)
#define BLOCK_COLS_X16 (BLOCK_COLS / 16)

#ifdef USE_COPY_1D

static inline
__attribute__((always_inline))
void copy_block_1d(
    __global const uchar* src0,
    const uint src_offset,
    __global uchar* dst0,
    const uint dst_offset
)
{
    __global const uchar* src = src0 + src_offset;
    __global uchar* dst = dst0 + dst_offset;

    uint processed = 0;

#if BLOCK_COLS_X16 >= 4
    {
        // uchar16 x 4rows per iteration
        uint i = get_local_id(0) * 16;  // uchar16
        while (i < BLOCK_COLS_X16 * 16)
        {
            uint4 idx = (uint4)(i, i + 16 * WSZ, i + 32 * WSZ, i + 48 * WSZ);
            idx = select((uint4)i, idx, idx < (BLOCK_COLS_X16 * 16));

            uchar16 a0 = vload16(0, src + idx.s0);
            uchar16 a1 = vload16(0, src + idx.s1);
            uchar16 a2 = vload16(0, src + idx.s2);
            uchar16 a3 = vload16(0, src + idx.s3);

            vstore16(a0, 0, dst + idx.s0);
            vstore16(a1, 0, dst + idx.s1);
            vstore16(a2, 0, dst + idx.s2);
            vstore16(a3, 0, dst + idx.s3);

            i += WSZ * 16 * 4;
        }
        processed = BLOCK_COLS_X16 * 16;
    }
#else
#define SKIP_1D_BLOCK_COLS_X16 1
#endif

#if BLOCK_COLS_X4 > 0 && (defined(SKIP_1D_BLOCK_COLS_X16) || (BLOCK_COLS_X16 * 16 != BLOCK_COLS_X4 * 4))
    {
        // uchar4 x 4rows per iteration
        uint i = get_local_id(0) * 4 + processed;  // uchar4
        while (i < BLOCK_COLS_X4 * 4)
        {
            uint4 idx = (uint4)(i, i + 4 * WSZ, i + 8 * WSZ, i + 12 * WSZ);
            idx = select((uint4)i, idx, idx < (BLOCK_COLS_X4 * 4));

            uchar4 a0 = vload4(0, src + idx.s0);
            uchar4 a1 = vload4(0, src + idx.s1);
            uchar4 a2 = vload4(0, src + idx.s2);
            uchar4 a3 = vload4(0, src + idx.s3);

            vstore4(a0, 0, dst + idx.s0);
            vstore4(a1, 0, dst + idx.s1);
            vstore4(a2, 0, dst + idx.s2);
            vstore4(a3, 0, dst + idx.s3);

            i += WSZ * 4 * 4;
        }
        processed = BLOCK_COLS_X4 * 4;
    }
#else
#define SKIP_1D_BLOCK_COLS_X4 1
#endif  // BLOCK_COLS_X4 > 0

#if (defined(SKIP_1D_BLOCK_COLS_X16) && defined(SKIP_1D_BLOCK_COLS_X4)) || BLOCK_COLS_X4 * 4 != BLOCK_COLS
    {
        uint i = get_local_id(0) + processed;
        while (i < BLOCK_COLS)
        {
            uchar a0 = src[i];
            dst[i] = a0;

            i += WSZ;
        }
    }
#endif
}

#else  // USE_COPY_1D

static inline
__attribute__((always_inline))
void copy_block_2d(
    __global const uchar* src0,
    const uint src_offset0,
    __global uchar* dst0,
    const uint dst_offset0
)
{
    __global const uchar* src = src0 + src_offset0;
    __global uchar* dst = dst0 + dst_offset0;

    uint i = get_local_id(0) * 4;

#define BLOCK_COLS_FILL_X4 (((BLOCK_COLS + 3) / 4) * 4)
#define BLOCK_SIZE_FILL_X4 (BLOCK_COLS_FILL_X4 * BLOCK_ROWS)

    while (i < BLOCK_SIZE_FILL_X4)
    {
        int row = i / BLOCK_COLS_FILL_X4;
        int col = i % BLOCK_COLS_FILL_X4;

        uint src_offset = row * BLOCK_SRC_STRIDE + col;
#if BLOCK_COLS_FILL_X4 == BLOCK_COLS
        uint dst_offset = i;
#else
        uint dst_offset = row * BLOCK_COLS + col;
#endif

#if BLOCK_COLS_FILL_X4 != BLOCK_COLS
        if (col <= BLOCK_COLS - 4)
#endif
        {
            uchar4 a = vload4(0, src + src_offset);
            vstore4(a, 0, dst + dst_offset);
        }
#if BLOCK_COLS_FILL_X4 != BLOCK_COLS
        else
        {
            /* non-optimized reference code
            while (col < BLOCK_COLS)
            {
                uchar a = src[src_offset];
                dst[dst_offset] = a;
                col++;
                src_offset++;
                dst_offset++;
            }
            */

            uint4 shift = (uint4)(0, 1, 2, 3);
            shift = select((uint4)0, shift, col + shift < BLOCK_COLS);

            dst[dst_offset + shift.s0] = src[src_offset + shift.s0];

#if BLOCK_COLS_FILL_X4 - BLOCK_COLS <= 2
            dst[dst_offset + shift.s1] = src[src_offset + shift.s1];
#endif
#if BLOCK_COLS_FILL_X4 - BLOCK_COLS <= 1
            dst[dst_offset + shift.s2] = src[src_offset + shift.s2];
#endif
        }
#endif  // BLOCK_COLS_FILL_X4 != BLOCK_COLS
        i += WSZ * 4;
    }
}

#endif  // USE_COPY_1D

__kernel void
CONCAT(slice_, DIMS)(
    __global const uchar* src,
    __global uchar* dst
)
{
    uint block_id = get_global_id(1);

    uint dst_offset = block_id * BLOCK_SIZE;

    uint src_offset = 0;

#define CALC_SRC_INDEX(dim) \
    { \
    uint plane_sz = CONCAT(DST_STEP_, dim) / BLOCK_SIZE; \
    CONCAT(idx_, dim) = block_id / plane_sz; \
    block_id = block_id - CONCAT(idx_, dim) * plane_sz; \
    }
#define UPDATE_SRC_OFFSET(dim) \
    src_offset = mad24((uint)(CONCAT(idx_, dim) + CONCAT(SRC_START_, dim)), (uint)CONCAT(SRC_STEP_, dim), (uint)src_offset);
/*
    if (get_global_id(0) == 0 && get_global_id(1) == 0) \
        printf("(%d, %d): @%d src_offset=%d   idx_dim=%d   block_id=%d\n", \
            get_global_id(0), get_global_id(1), \
            dim, src_offset, CONCAT(idx_, dim), block_id \
        );
*/

#if DIMS > 5
#error "invalid configuration"
#endif
#if DIMS > 4
    uint idx_4 = 0;
#if BLOCK_DIMS <= 4
    CALC_SRC_INDEX(4)
#endif
    UPDATE_SRC_OFFSET(4)
#endif
#if DIMS > 3
    uint idx_3 = 0;
#if BLOCK_DIMS <= 3
    CALC_SRC_INDEX(3)
#endif
    UPDATE_SRC_OFFSET(3)
#endif
#if DIMS > 2
    uint idx_2 = 0;
#if BLOCK_DIMS <= 2
    CALC_SRC_INDEX(2)
#endif
    UPDATE_SRC_OFFSET(2)
#endif
#if DIMS > 1
    uint idx_1 = 0;
#if BLOCK_DIMS <= 1
    CALC_SRC_INDEX(1)
#endif
    UPDATE_SRC_OFFSET(1)
#endif
#if DIMS > 0
    uint idx_0 = 0;
    UPDATE_SRC_OFFSET(0)
#endif

/*
    if (get_global_id(0) == 0)
        printf("(%d, %d): src_offset=%d dst_offset=%d\n",
            get_global_id(0), get_global_id(1),
            src_offset, dst_offset
        );
*/

#ifdef USE_COPY_1D
    copy_block_1d(src, src_offset, dst, dst_offset);
#else
    copy_block_2d(src, src_offset, dst, dst_offset);
#endif
}
