// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef BORDER_CONSTANT
//CCCCCC|abcdefgh|CCCCCCC
#define EXTRAPOLATE(x, maxV)
#elif defined BORDER_REPLICATE
//aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = max(min((x), (maxV) - 1), 0); \
    }
#elif defined BORDER_WRAP
//cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = ( (x) + (maxV) ) % (maxV); \
    }
#elif defined BORDER_REFLECT
//fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min( mad24((maxV)-1,2,-(x))+1 , max((x),-(x)-1) ); \
    }
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
//gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min( mad24((maxV)-1,2,-(x)), max((x),-(x)) ); \
    }
#else
#error No extrapolation method
#endif

#define SRC(_x,_y) convert_float(((global SRCTYPE*)(Src+(_y)*src_step))[_x])

#ifdef BORDER_CONSTANT
//CCCCCC|abcdefgh|CCCCCCC
#define ELEM(_x,_y,r_edge,t_edge,const_v) (_x)<0 | (_x) >= (r_edge) | (_y)<0 | (_y) >= (t_edge) ? (const_v) : SRC((_x),(_y))
#else
#define ELEM(_x,_y,r_edge,t_edge,const_v) SRC((_x),(_y))
#endif

#define DSTX(_x,_y) (((global float*)(DstX+DstXOffset+(_y)*DstXPitch))[_x])
#define DSTY(_x,_y) (((global float*)(DstY+DstYOffset+(_y)*DstYPitch))[_x])

#define INIT_AND_READ_LOCAL_SOURCE(width, height, fill_const, kernel_border) \
    int srcX = x + srcOffsetX - (kernel_border); \
    int srcY = y + srcOffsetY - (kernel_border); \
    int xb = srcX; \
    int yb = srcY; \
    \
    EXTRAPOLATE(xb, (width)); \
    EXTRAPOLATE(yb, (height)); \
    lsmem[liy][lix] = ELEM(xb, yb, (width), (height), (fill_const) ); \
    \
    if(lix < ((kernel_border)*2)) \
    { \
        int xb = srcX+BLK_X; \
        EXTRAPOLATE(xb,(width)); \
        lsmem[liy][lix+BLK_X] = ELEM(xb, yb, (width), (height), (fill_const) ); \
    } \
    if(liy< ((kernel_border)*2)) \
    { \
        int yb = srcY+BLK_Y; \
        EXTRAPOLATE(yb, (height)); \
        lsmem[liy+BLK_Y][lix] = ELEM(xb, yb, (width), (height), (fill_const) ); \
    } \
    if(lix<((kernel_border)*2) && liy<((kernel_border)*2)) \
    { \
        int xb = srcX+BLK_X; \
        int yb = srcY+BLK_Y; \
        EXTRAPOLATE(xb,(width)); \
        EXTRAPOLATE(yb,(height)); \
        lsmem[liy+BLK_Y][lix+BLK_X] = ELEM(xb, yb, (width), (height), (fill_const) ); \
    }

__kernel void sobel3(__global const uchar * Src, int src_step, int srcOffsetX, int srcOffsetY,
                     __global uchar * DstX, int DstXPitch, int DstXOffset,
                     __global uchar * DstY, int DstYPitch, int DstYOffset, int dstHeight, int dstWidth,
                     int height, int width, float scale)
{
    __local float lsmem[BLK_Y+2][BLK_X+2];

    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    INIT_AND_READ_LOCAL_SOURCE(width, height, 0, 1)
    barrier(CLK_LOCAL_MEM_FENCE);

    if( x >= dstWidth || y >=dstHeight )  return;

    float u1 = lsmem[liy][lix];
    float u2 = lsmem[liy][lix+1];
    float u3 = lsmem[liy][lix+2];

    float m1 = lsmem[liy+1][lix];
    float m3 = lsmem[liy+1][lix+2];

    float b1 = lsmem[liy+2][lix];
    float b2 = lsmem[liy+2][lix+1];
    float b3 = lsmem[liy+2][lix+2];

    //calc and store dx and dy;//
#ifdef SCHARR
    DSTX(x,y) = mad(10.0f, m3 - m1, 3.0f * (u3 - u1 + b3 - b1)) * scale;
    DSTY(x,y) = mad(10.0f, b2 - u2, 3.0f * (b1 - u1 + b3 - u3)) * scale;
#else
    DSTX(x,y) = mad(2.0f, m3 - m1, u3 - u1 + b3 - b1) * scale;
    DSTY(x,y) = mad(2.0f, b2 - u2, b1 - u1 + b3 - u3) * scale;
#endif
}

__kernel void sobel5(__global const uchar * Src, int src_step, int srcOffsetX, int srcOffsetY,
                     __global uchar * DstX, int DstXPitch, int DstXOffset,
                     __global uchar * DstY, int DstYPitch, int DstYOffset, int dstHeight, int dstWidth,
                     int height, int width, float scale)
{
    __local float lsmem[BLK_Y+4][BLK_X+4];

    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    INIT_AND_READ_LOCAL_SOURCE(width, height, 0, 2)
    barrier(CLK_LOCAL_MEM_FENCE);

    if( x >= dstWidth || y >=dstHeight )  return;

    float t1 = lsmem[liy][lix];
    float t2 = lsmem[liy][lix+1];
    float t3 = lsmem[liy][lix+2];
    float t4 = lsmem[liy][lix+3];
    float t5 = lsmem[liy][lix+4];

    float u1 = lsmem[liy+1][lix];
    float u2 = lsmem[liy+1][lix+1];
    float u3 = lsmem[liy+1][lix+2];
    float u4 = lsmem[liy+1][lix+3];
    float u5 = lsmem[liy+1][lix+4];

    float m1 = lsmem[liy+2][lix];
    float m2 = lsmem[liy+2][lix+1];
    float m4 = lsmem[liy+2][lix+3];
    float m5 = lsmem[liy+2][lix+4];

    float l1 = lsmem[liy+3][lix];
    float l2 = lsmem[liy+3][lix+1];
    float l3 = lsmem[liy+3][lix+2];
    float l4 = lsmem[liy+3][lix+3];
    float l5 = lsmem[liy+3][lix+4];

    float b1 = lsmem[liy+4][lix];
    float b2 = lsmem[liy+4][lix+1];
    float b3 = lsmem[liy+4][lix+2];
    float b4 = lsmem[liy+4][lix+3];
    float b5 = lsmem[liy+4][lix+4];

    //calc and store dx and dy;//
    DSTX(x,y) = scale *
        mad(12.0f, m4 - m2,
            mad(6.0f, m5 - m1,
                mad(8.0f, u4 - u2 + l4 - l2,
                    mad(4.0f, u5 - u1 + l5 - l1,
                        mad(2.0f, t4 - t2 + b4 - b2, t5 - t1 + b5 - b1 )
                        )
                    )
                )
            );

    DSTY(x,y) = scale *
        mad(12.0f, l3 - u3,
            mad(6.0f, b3 - t3,
                mad(8.0f, l2 - u2 + l4 - u4,
                    mad(4.0f, b2 - t2 + b4 - t4,
                        mad(2.0f, l1 - u1 + l5 - u5, b1 - t1 + b5 - t5 )
                        )
                    )
                )
            );
}

__kernel void sobel7(__global const uchar * Src, int src_step, int srcOffsetX, int srcOffsetY,
                     __global uchar * DstX, int DstXPitch, int DstXOffset,
                     __global uchar * DstY, int DstYPitch, int DstYOffset, int dstHeight, int dstWidth,
                     int height, int width, float scale)
{
    __local float lsmem[BLK_Y+6][BLK_X+6];

    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    INIT_AND_READ_LOCAL_SOURCE(width, height, 0, 3)
    barrier(CLK_LOCAL_MEM_FENCE);

    if( x >= dstWidth || y >=dstHeight )  return;

    float tt1 = lsmem[liy][lix];
    float tt2 = lsmem[liy][lix+1];
    float tt3 = lsmem[liy][lix+2];
    float tt4 = lsmem[liy][lix+3];
    float tt5 = lsmem[liy][lix+4];
    float tt6 = lsmem[liy][lix+5];
    float tt7 = lsmem[liy][lix+6];

    float t1 = lsmem[liy+1][lix];
    float t2 = lsmem[liy+1][lix+1];
    float t3 = lsmem[liy+1][lix+2];
    float t4 = lsmem[liy+1][lix+3];
    float t5 = lsmem[liy+1][lix+4];
    float t6 = lsmem[liy+1][lix+5];
    float t7 = lsmem[liy+1][lix+6];

    float u1 = lsmem[liy+2][lix];
    float u2 = lsmem[liy+2][lix+1];
    float u3 = lsmem[liy+2][lix+2];
    float u4 = lsmem[liy+2][lix+3];
    float u5 = lsmem[liy+2][lix+4];
    float u6 = lsmem[liy+2][lix+5];
    float u7 = lsmem[liy+2][lix+6];

    float m1 = lsmem[liy+3][lix];
    float m2 = lsmem[liy+3][lix+1];
    float m3 = lsmem[liy+3][lix+2];
    float m5 = lsmem[liy+3][lix+4];
    float m6 = lsmem[liy+3][lix+5];
    float m7 = lsmem[liy+3][lix+6];

    float l1 = lsmem[liy+4][lix];
    float l2 = lsmem[liy+4][lix+1];
    float l3 = lsmem[liy+4][lix+2];
    float l4 = lsmem[liy+4][lix+3];
    float l5 = lsmem[liy+4][lix+4];
    float l6 = lsmem[liy+4][lix+5];
    float l7 = lsmem[liy+4][lix+6];

    float b1 = lsmem[liy+5][lix];
    float b2 = lsmem[liy+5][lix+1];
    float b3 = lsmem[liy+5][lix+2];
    float b4 = lsmem[liy+5][lix+3];
    float b5 = lsmem[liy+5][lix+4];
    float b6 = lsmem[liy+5][lix+5];
    float b7 = lsmem[liy+5][lix+6];

    float bb1 = lsmem[liy+6][lix];
    float bb2 = lsmem[liy+6][lix+1];
    float bb3 = lsmem[liy+6][lix+2];
    float bb4 = lsmem[liy+6][lix+3];
    float bb5 = lsmem[liy+6][lix+4];
    float bb6 = lsmem[liy+6][lix+5];
    float bb7 = lsmem[liy+6][lix+6];

    //calc and store dx and dy
    DSTX(x,y) = scale *
        mad(100.0f, m5 - m3,
            mad(80.0f, m6 - m2,
                mad(20.0f, m7 - m1,
                    mad(75.0f, u5 - u3 + l5 - l3,
                        mad(60.0f, u6 - u2 + l6 - l2,
                            mad(15.0f, u7 - u1 + l7 - l1,
                                mad(30.0f, t5 - t3 + b5 - b3,
                                    mad(24.0f, t6 - t2 + b6 - b2,
                                        mad(6.0f, t7 - t1 + b7 - b1,
                                            mad(5.0f, tt5 - tt3 + bb5 - bb3,
                                                mad(4.0f, tt6 - tt2 + bb6 - bb2, tt7 - tt1 + bb7 - bb1 )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            );

    DSTY(x,y) = scale *
        mad(100.0f, l4 - u4,
            mad(80.0f, b4 - t4,
                mad(20.0f, bb4 - tt4,
                    mad(75.0f, l5 - u5 + l3 - u3,
                        mad(60.0f, b5 - t5 + b3 - t3,
                            mad(15.0f, bb5 - tt5 + bb3 - tt3,
                                mad(30.0f, l6 - u6 + l2 - u2,
                                    mad(24.0f, b6 - t6 + b2 - t2,
                                        mad(6.0f, bb6 - tt6 + bb2 - tt2,
                                            mad(5.0f, l7 - u7 + l1 - u1,
                                                mad(4.0f, b7 - t7 + b1 - t1, bb7 - tt7 + bb1 - tt1 )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            );
}
