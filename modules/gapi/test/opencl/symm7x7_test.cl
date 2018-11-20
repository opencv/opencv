// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if BORDER_REPLICATE
#define GET_BORDER(elem) (elem)
#define SET_ALL(i, j) a0[i] = a0[j]; a1[i] = a1[j]; a2[i] = a2[j]; b[i] = b[j]; c0[i] = c0[j]; c1[i] = c1[j]; c2[i] = c2[j];
#else
#define GET_BORDER(elem) (BORDER_CONSTANT_VALUE)
#define SET_ALL(i, j) a0[i] = a1[i] = a2[i] = c0[i] = c1[i] = c2[i] = BORDER_CONSTANT_VALUE; b[i] = BORDER_CONSTANT_VALUE;
#endif

#define GET_A0(id, x, l_edge, a1) ((x) <= (l_edge + 2) ? GET_BORDER(a1) : (((const __global uchar*)(id))[-3]))
#define GET_A1(id, x, l_edge, a2) ((x) <= (l_edge + 1) ? GET_BORDER(a2) : (((const __global uchar*)(id))[-2]))
#define GET_A2(id, x, l_edge, b) ((x) <= (l_edge) ? GET_BORDER(b[0]) : (((const __global uchar*)(id))[-1]))
#define GET_C0(id, x, r_edge, b) ((x) >= (r_edge) ? GET_BORDER(b[8 - 1]) : (((const __global uchar*)(id))[8]))
#define GET_C1(id, x, r_edge, c0) ((x) >= (r_edge - 1) ? GET_BORDER(c0) : (((const __global uchar*)(id))[8 + 1]))
#define GET_C2(id, x, r_edge, c1) ((x) >= (r_edge - 2) ? GET_BORDER(c1) : (((const __global uchar*)(id))[8 + 2]))

__kernel void symm_7x7_test(
    __global const uchar * srcptr,
    int srcStep, int srcEndX, int srcEndY,
    __global uchar * dstptr, int dstStep,
    int rows, int cols,
    int tile_y_coord,
    __constant int * coeff)
{
    int lEdge = 0, rEdge = cols - 8;
    int x = (get_global_id(0) < cols/8) ? get_global_id(0) * 8: cols - 8;
    int y = get_global_id(1);

    int yd = min(3, tile_y_coord);
    int dst_id = mad24(y, dstStep, x);
    y+=yd;
    int src_id = mad24(y, srcStep, x);
    int y_limit = y + tile_y_coord;
    y_limit-=yd;

    const __global uchar* psrc = (const __global uchar*)(srcptr + src_id);
    __global uchar* pdst = (__global uchar*)(dstptr + dst_id);

#define BSIZE (7)
    //float8 row0, row1, row2, xVal;

    float a0[BSIZE]; float a1[BSIZE]; float a2[BSIZE];
    float8 b[BSIZE];
    float c0[BSIZE]; float c1[BSIZE]; float c2[BSIZE];

    //a0[0] a1[0] a2[0] b[0] c0[0] c1[0] c2[0]
    //a0[1] a1[1] a2[1] b[1] c0[1] c1[1] c2[1]
    //a0[2] a1[2] a2[2] b[2] c0[2] c1[2] c2[2]
    //a0[3] a1[3] a2[3] b[3] c0[3] c1[3] c2[3]
    //a0[4] a1[4] a2[4] b[4] c0[4] c1[4] c2[4]
    //a0[5] a1[5] a2[5] b[5] c0[5] c1[5] c2[5]
    //a0[6] a1[6] a2[6] b[6] c0[6] c1[6] c2[6]

    //start by filling in row 3
    b[3] = convert_float8(vload8(0, (const __global uchar*)psrc));


    //if we even have to worry about border checking...
    if( (y_limit <=2 ) || (y_limit >= srcEndY - 3) || (x >= rEdge-2) || (x <= lEdge + 2) )
    {
        a2[3] = GET_A2(psrc, x, lEdge, b[3]);
        a1[3] = GET_A1(psrc, x, lEdge, a2[3]);
        a0[3] = GET_A0(psrc, x, lEdge, a1[3]);
        c0[3] = GET_C0(psrc, x, rEdge, b[3]);
        c1[3] = GET_C1(psrc, x, rEdge, c0[3]);
        c2[3] = GET_C2(psrc, x, rEdge, c1[3]);

        if(y_limit > 0)
        {
          b[2] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep)));
          a2[2] = GET_A2(psrc - srcStep, x, lEdge, b[2]);
          a1[2] = GET_A1(psrc - srcStep, x, lEdge, a2[2]);
          a0[2] = GET_A0(psrc - srcStep, x, lEdge, a1[2]);
          c0[2] = GET_C0(psrc - srcStep, x, rEdge, b[2]);
          c1[2] = GET_C1(psrc - srcStep, x, rEdge, c0[2]);
          c2[2] = GET_C2(psrc - srcStep, x, rEdge, c1[2]);
        }
        else
        {
          SET_ALL(2, 3);
        }

        if( y_limit > 1 )
        {
          b[1] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep*2)));
          a2[1] = GET_A2(psrc - srcStep*2, x, lEdge, b[1]);
          a1[1] = GET_A1(psrc - srcStep*2, x, lEdge, a2[1]);
          a0[1] = GET_A0(psrc - srcStep*2, x, lEdge, a1[1]);
          c0[1] = GET_C0(psrc - srcStep*2, x, rEdge, b[1]);
          c1[1] = GET_C1(psrc - srcStep*2, x, rEdge, c0[1]);
          c2[1] = GET_C2(psrc - srcStep*2, x, rEdge, c1[1]);
        }
        else
        {
          SET_ALL(1, 2);
        }

        if( y_limit > 2 )
        {
          b[0] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep*3)));
          a2[0] = GET_A2(psrc - srcStep*3, x, lEdge, b[0]);
          a1[0] = GET_A1(psrc - srcStep*3, x, lEdge, a2[0]);
          a0[0] = GET_A0(psrc - srcStep*3, x, lEdge, a1[0]);
          c0[0] = GET_C0(psrc - srcStep*3, x, rEdge, b[0]);
          c1[0] = GET_C1(psrc - srcStep*3, x, rEdge, c0[0]);
          c2[0] = GET_C2(psrc - srcStep*3, x, rEdge, c1[0]);
        }
        else
        {
          SET_ALL(0, 1);
        }

        //row 4
        if( y_limit < srcEndY - 1 )
        {
          b[4] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep)));
          a2[4] = GET_A2(psrc + srcStep, x, lEdge, b[4]);
          a1[4] = GET_A1(psrc + srcStep, x, lEdge, a2[4]);
          a0[4] = GET_A0(psrc + srcStep, x, lEdge, a1[4]);
          c0[4] = GET_C0(psrc + srcStep, x, rEdge, b[4]);
          c1[4] = GET_C1(psrc + srcStep, x, rEdge, c0[4]);
          c2[4] = GET_C2(psrc + srcStep, x, rEdge, c1[4]);
        }
        else
        {
          SET_ALL(4, 3);
        }

        if( y_limit < srcEndY - 2 )
        {
          b[5] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep*2)));
          a2[5] = GET_A2(psrc + srcStep*2, x, lEdge, b[5]);
          a1[5] = GET_A1(psrc + srcStep*2, x, lEdge, a2[5]);
          a0[5] = GET_A0(psrc + srcStep*2, x, lEdge, a1[5]);
          c0[5] = GET_C0(psrc + srcStep*2, x, rEdge, b[5]);
          c1[5] = GET_C1(psrc + srcStep*2, x, rEdge, c0[5]);
          c2[5] = GET_C2(psrc + srcStep*2, x, rEdge, c1[5]);
        }
        else
        {
          SET_ALL(5, 4);
        }

        //this one can move into the loop
        if( y_limit < srcEndY - 3 )
        {
          b[6] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep*3)));
          a2[6] = GET_A2(psrc + srcStep*3, x, lEdge, b[6]);
          a1[6] = GET_A1(psrc + srcStep*3, x, lEdge, a2[6]);
          a0[6] = GET_A0(psrc + srcStep*3, x, lEdge, a1[6]);
          c0[6] = GET_C0(psrc + srcStep*3, x, rEdge, b[6]);
          c1[6] = GET_C1(psrc + srcStep*3, x, rEdge, c0[6]);
          c2[6] = GET_C2(psrc + srcStep*3, x, rEdge, c1[6]);
        }
        else
        {
          SET_ALL(6, 5);
        }
    }
    else
    {
      a2[3] = (((const __global uchar*)(psrc))[-1]);
      a1[3] = (((const __global uchar*)(psrc))[-2]);
      a0[3] = (((const __global uchar*)(psrc))[-3]);
      c0[3] = (((const __global uchar*)(psrc))[8]);
      c1[3] = (((const __global uchar*)(psrc))[8 + 1]);
      c2[3] = (((const __global uchar*)(psrc))[8 + 2]);

      b[2] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep)));
      a2[2] = (((const __global uchar*)(psrc - srcStep))[-1]);
      a1[2] = (((const __global uchar*)(psrc - srcStep))[-2]);
      a0[2] = (((const __global uchar*)(psrc - srcStep))[-3]);
      c0[2] = (((const __global uchar*)(psrc - srcStep))[8]);
      c1[2] = (((const __global uchar*)(psrc - srcStep))[8 + 1]);
      c2[2] = (((const __global uchar*)(psrc - srcStep))[8 + 2]);
      b[1] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep*2)));
      a2[1] = (((const __global uchar*)(psrc - srcStep*2))[-1]);
      a1[1] = (((const __global uchar*)(psrc - srcStep*2))[-2]);
      a0[1] = (((const __global uchar*)(psrc - srcStep*2))[-3]);
      c0[1] = (((const __global uchar*)(psrc - srcStep*2))[8]);
      c1[1] = (((const __global uchar*)(psrc - srcStep*2))[8 + 1]);
      c2[1] = (((const __global uchar*)(psrc - srcStep*2))[8 + 2]);
      b[0] = convert_float8(vload8(0, (const __global uchar*)(psrc - srcStep*3)));
      a2[0] = (((const __global uchar*)(psrc - srcStep*3))[-1]);
      a1[0] = (((const __global uchar*)(psrc - srcStep*3))[-2]);
      a0[0] = (((const __global uchar*)(psrc - srcStep*3))[-3]);
      c0[0] = (((const __global uchar*)(psrc - srcStep*3))[8]);
      c1[0] = (((const __global uchar*)(psrc - srcStep*3))[8 + 1]);
      c2[0] = (((const __global uchar*)(psrc - srcStep*3))[8 + 2]);
      b[4] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep)));
      a2[4] = (((const __global uchar*)(psrc + srcStep))[-1]);
      a1[4] = (((const __global uchar*)(psrc + srcStep))[-2]);
      a0[4] = (((const __global uchar*)(psrc + srcStep))[-3]);
      c0[4] = (((const __global uchar*)(psrc + srcStep))[8]);
      c1[4] = (((const __global uchar*)(psrc + srcStep))[8 + 1]);
      c2[4] = (((const __global uchar*)(psrc + srcStep))[8 + 2]);
      b[5] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep*2)));
      a2[5] = (((const __global uchar*)(psrc + srcStep*2))[-1]);
      a1[5] = (((const __global uchar*)(psrc + srcStep*2))[-2]);
      a0[5] = (((const __global uchar*)(psrc + srcStep*2))[-3]);
      c0[5] = (((const __global uchar*)(psrc + srcStep*2))[8]);
      c1[5] = (((const __global uchar*)(psrc + srcStep*2))[8 + 1]);
      c2[5] = (((const __global uchar*)(psrc + srcStep*2))[8 + 2]);
      b[6] = convert_float8(vload8(0, (const __global uchar*)(psrc + srcStep*3)));
      a2[6] = (((const __global uchar*)(psrc + srcStep*3))[-1]);
      a1[6] = (((const __global uchar*)(psrc + srcStep*3))[-2]);
      a0[6] = (((const __global uchar*)(psrc + srcStep*3))[-3]);
      c0[6] = (((const __global uchar*)(psrc + srcStep*3))[8]);
      c1[6] = (((const __global uchar*)(psrc + srcStep*3))[8 + 1]);
      c2[6] = (((const __global uchar*)(psrc + srcStep*3))[8 + 2]);
    }
    float a0_sum[3]; float a1_sum[3]; float a2_sum[3];
    float8 b_sum[3];
    float c0_sum[3]; float c1_sum[3]; float c2_sum[3];

    a0_sum[0] = a0[0] + a0[6];
    a0_sum[1] = a0[1] + a0[5];
    a0_sum[2] = a0[2] + a0[4];

    a1_sum[0] = a1[0] + a1[6];
    a1_sum[1] = a1[1] + a1[5];
    a1_sum[2] = a1[2] + a1[4];

    a2_sum[0] = a2[0] + a2[6];
    a2_sum[1] = a2[1] + a2[5];
    a2_sum[2] = a2[2] + a2[4];

    c0_sum[0] = c0[0] + c0[6];
    c0_sum[1] = c0[1] + c0[5];
    c0_sum[2] = c0[2] + c0[4];

    c1_sum[0] = c1[0] + c1[6];
    c1_sum[1] = c1[1] + c1[5];
    c1_sum[2] = c1[2] + c1[4];

    c2_sum[0] = c2[0] + c2[6];
    c2_sum[1] = c2[1] + c2[5];
    c2_sum[2] = c2[2] + c2[4];

    b_sum[0] = b[0] + b[6];
    b_sum[1] = b[1] + b[5];
    b_sum[2] = b[2] + b[4];

    float8 A = b[3];
    float8 intermediate = A * (float)coeff[0];

    float8 B = b_sum[2] +
               (float8)(a2[3], b[3].s0123, b[3].s456) +
               (float8)(b[3].s123, b[3].s4567, c0[3]);
    intermediate += B * (float)coeff[1];

    float8 C = (float8)(a2_sum[2], b_sum[2].s0123, b_sum[2].s456) +
               (float8)(b_sum[2].s123, b_sum[2].s4567, c0_sum[2]);
    intermediate += C * (float)coeff[2];

    float8 D = b_sum[1] +
               (float8)(a1[3], a2[3], b[3].s0123, b[3].s45) +
               (float8)(b[3].s23, b[3].s4567, c0[3], c1[3]);
    intermediate += D * (float)coeff[3];

    float8 E = (float8)(a2_sum[1], b_sum[1].s0123, b_sum[1].s456) +
              (float8)( b_sum[1].s123, b_sum[1].s4567, c0_sum[1]) +
              (float8)( a1_sum[2], a2_sum[2], b_sum[2].s0123, b_sum[2].s45) +
              (float8)( b_sum[2].s23, b_sum[2].s4567, c0_sum[2], c1_sum[2]);
    intermediate += E * (float)coeff[4];

    float8 F = (float8)(a1_sum[1], a2_sum[1], b_sum[1].s0123, b_sum[1].s45) +
               (float8)(b_sum[1].s23, b_sum[1].s4567, c0_sum[1], c1_sum[1]);
    intermediate += F * (float)coeff[5];

    float8 G = b_sum[0] +
               (float8)(a0[3], a1[3], a2[3], b[3].s0123, b[3].s4) +
               (float8)(b[3].s3, b[3].s4567, c0[3], c1[3], c2[3]);
    intermediate += G * (float)coeff[6];

    float8 H = (float8)(a2_sum[0], b_sum[0].s0123, b_sum[0].s456) +
               (float8)(b_sum[0].s123, b_sum[0].s4567, c0_sum[0]) +
               (float8)(a0_sum[2], a1_sum[2], a2_sum[2], b_sum[2].s0123, b_sum[2].s4) +
               (float8)(b_sum[2].s3, b_sum[2].s4567, c0_sum[2], c1_sum[2], c2_sum[2]);
    intermediate += H * (float)coeff[7];

    float8 I = (float8)(a1_sum[0], a2_sum[0], b_sum[0].s0123, b_sum[0].s45) +
               (float8)(b_sum[0].s23, b_sum[0].s4567, c0_sum[0], c1_sum[0]) +
               (float8)(a0_sum[1], a1_sum[1], a2_sum[1], b_sum[1].s0123, b_sum[1].s4) +
               (float8)(b_sum[1].s3, b_sum[1].s4567, c0_sum[1], c1_sum[1], c2_sum[1]);
    intermediate += I * (float)coeff[8];


    float8 J = (float8)(a0_sum[0], a1_sum[0], a2_sum[0], b_sum[0].s0123, b_sum[0].s4) +
               (float8)(b_sum[0].s3, b_sum[0].s4567, c0_sum[0], c1_sum[0], c2_sum[0]);
    intermediate += J * (float)coeff[9];

    intermediate *= SCALE;

    vstore8(convert_uchar8_sat(intermediate), 0, (__global uchar*)(pdst));

}
