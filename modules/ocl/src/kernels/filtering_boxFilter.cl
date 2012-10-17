/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Ying, zhangying913@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BORDER_REPLICATE
//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   :(i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
//BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)-1 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

//blur function does not support BORDER_WRAP
#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////8uC1////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#define THREADS 256
#define ELEM(i, l_edge, r_edge, elem1, elem2) (i) >= (l_edge) && (i) < (r_edge) ? (elem1) : (elem2)
__kernel void boxFilter_C1_D0(__global const uchar * restrict src, __global uchar *dst, float alpha,
                                     int src_offset, int src_whole_rows, int src_whole_cols, int src_step,
                                     int dst_offset, int dst_rows, int dst_cols, int dst_step
                                     )
{

    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);
    int src_x_off = src_offset % src_step;
    int src_y_off = src_offset / src_step;
    int dst_x_off = dst_offset % dst_step;
    int dst_y_off = dst_offset / dst_step;

    int head_off = dst_x_off%4;
    int startX = ((gX * (THREADS-ksX+1)-anX) * 4) - head_off + src_x_off;
    int startY = (gY << 1) - anY + src_y_off;
    int dst_startX = (gX * (THREADS-ksX+1) * 4) - head_off + dst_x_off;
    int dst_startY = (gY << 1) + dst_y_off;

    uint4 data[ksY+1];
    __local uint4 temp[(THREADS<<1)];

#ifdef BORDER_CONSTANT

        for(int i=0; i < ksY+1; i++)
        {
            if(startY+i >=0 && startY+i < src_whole_rows && startX+col*4 >=0 && startX+col*4+3<src_whole_cols)
                data[i] = convert_uint4(vload4(col,(__global uchar*)(src+(startY+i)*src_step + startX)));
            else
            {
                data[i]=0;
                int con = startY+i >=0 && startY+i < src_whole_rows && startX+col*4 >=0 && startX+col*4<src_whole_cols;
                if(con)data[i].s0 = *(src+(startY+i)*src_step + startX + col*4);
                con = startY+i >=0 && startY+i < src_whole_rows && startX+col*4+1 >=0 && startX+col*4+1<src_whole_cols;
                if(con)data[i].s1 = *(src+(startY+i)*src_step + startX + col*4+1) ;
                con = startY+i >=0 && startY+i < src_whole_rows && startX+col*4+2 >=0 && startX+col*4+2<src_whole_cols;
                if(con)data[i].s2 = *(src+(startY+i)*src_step + startX + col*4+2);
                con = startY+i >=0 && startY+i < src_whole_rows && startX+col*4+3 >=0 && startX+col*4+3<src_whole_cols;
                if(con)data[i].s3 = *(src+(startY+i)*src_step + startX + col*4+3);
            }
        }

#else
   int not_all_in_range;
   for(int i=0; i < ksY+1; i++)
   {
      not_all_in_range = (startX+col*4<0) | (startX+col*4+3>src_whole_cols-1)
                        | (startY+i<0) | (startY+i>src_whole_rows-1);
      if(not_all_in_range)
      {
          int selected_row;
          int4 selected_col;
          selected_row = ADDR_H(startY+i, 0, src_whole_rows);
          selected_row = ADDR_B(startY+i, src_whole_rows, selected_row);

          selected_col.x = ADDR_L(startX+col*4, 0, src_whole_cols);
          selected_col.x = ADDR_R(startX+col*4, src_whole_cols, selected_col.x);

          selected_col.y = ADDR_L(startX+col*4+1, 0, src_whole_cols);
          selected_col.y = ADDR_R(startX+col*4+1, src_whole_cols, selected_col.y);

          selected_col.z = ADDR_L(startX+col*4+2, 0, src_whole_cols);
          selected_col.z = ADDR_R(startX+col*4+2, src_whole_cols, selected_col.z);

          selected_col.w = ADDR_L(startX+col*4+3, 0, src_whole_cols);
          selected_col.w = ADDR_R(startX+col*4+3, src_whole_cols, selected_col.w);

          data[i].x = *(src + selected_row * src_step + selected_col.x);
          data[i].y = *(src + selected_row * src_step + selected_col.y);
          data[i].z = *(src + selected_row * src_step + selected_col.z);
          data[i].w = *(src + selected_row * src_step + selected_col.w);
      }
      else
      {
          data[i] =  convert_uint4(vload4(col,(__global uchar*)(src+(startY+i)*src_step + startX)));
      }
   }
#endif
    uint4 sum0 = 0, sum1 = 0, sum2 = 0;
    for(int i=1; i < ksY; i++)
    {
        sum0 += (data[i]);
    }
    sum1 = sum0 + (data[0]);
    sum2 = sum0 + (data[ksY]);

    temp[col] = sum1;
    temp[col+THREADS] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(col >= anX && col < (THREADS-ksX+anX+1))
    {
        int posX = dst_startX - dst_x_off + (col-anX)*4;
        int posY = (gY << 1);
        uint4 tmp_sum1=0, tmp_sum2=0;
        for(int i=-anX; i<=anX; i++)
        {
           tmp_sum1 += vload4(col, (__local uint*)temp+i);
        }

        for(int i=-anX; i<=anX; i++)
        {
           tmp_sum2 += vload4(col, (__local uint*)(temp+THREADS)+i);
        }

        if(posY < dst_rows && posX < dst_cols)
        {
           if(posX >= 0 && posX < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX + (col-anX)*4) = tmp_sum1.x/alpha;
           if(posX+1 >= 0 && posX+1 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+1 + (col-anX)*4) = tmp_sum1.y/alpha;
           if(posX+2 >= 0 && posX+2 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+2 + (col-anX)*4) = tmp_sum1.z/alpha;
           if(posX+3 >= 0 && posX+3 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+3 + (col-anX)*4) = tmp_sum1.w/alpha;
        }
        if(posY+1 < dst_rows && posX < dst_cols)
        {
           dst_startY+=1;
           if(posX >= 0 && posX < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX + (col-anX)*4) = tmp_sum2.x/alpha;
           if(posX+1 >= 0 && posX+1 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+1 + (col-anX)*4) = tmp_sum2.y/alpha;
           if(posX+2 >= 0 && posX+2 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+2 + (col-anX)*4) = tmp_sum2.z/alpha;
           if(posX+3 >= 0 && posX+3 < dst_cols)
               *(dst+dst_startY * dst_step + dst_startX+3 + (col-anX)*4) = tmp_sum2.w/alpha;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////8uC4////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void boxFilter_C4_D0(__global const uchar4 * restrict src, __global uchar4 *dst, float alpha,
                                     int src_offset, int src_whole_rows, int src_whole_cols, int src_step,
                                     int dst_offset, int dst_rows, int dst_cols, int dst_step
                                     )
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);

    int src_x_off = (src_offset % src_step) >> 2;
    int src_y_off = src_offset / src_step;
    int dst_x_off = (dst_offset % dst_step) >> 2;
    int dst_y_off = dst_offset / dst_step;

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;
    int startY = (gY << 1) - anY + src_y_off;
    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY << 1) + dst_y_off;
      //int end_addr = (src_whole_rows-1)*(src_step>>2) + src_whole_cols-4;

      int end_addr = src_whole_cols-4;
    uint4 data[ksY+1];
    __local uint4 temp[2][THREADS];
#ifdef BORDER_CONSTANT
    bool con;
    uint4 ss;
    for(int i=0; i < ksY+1; i++)
    {
        con = startX+col >= 0 && startX+col < src_whole_cols && startY+i >= 0 && startY+i < src_whole_rows;

            //int cur_addr = clamp((startY+i)*(src_step>>2)+(startX+col),0,end_addr);
        //ss = convert_uint4(src[cur_addr]);

        int cur_col = clamp(startX + col, 0, src_whole_cols);
        if(con)
          ss = convert_uint4(src[(startY+i)*(src_step>>2) + cur_col]);

        data[i] = con ? ss : 0;
    }
#else
   for(int i=0; i < ksY+1; i++)
   {
          int selected_row;
          int selected_col;
          selected_row = ADDR_H(startY+i, 0, src_whole_rows);
          selected_row = ADDR_B(startY+i, src_whole_rows, selected_row);

          selected_col = ADDR_L(startX+col, 0, src_whole_cols);
          selected_col = ADDR_R(startX+col, src_whole_cols, selected_col);


          data[i] = convert_uint4(src[selected_row * (src_step>>2) + selected_col]);
   }

#endif
    uint4 sum0 = 0, sum1 = 0, sum2 = 0;
    for(int i=1; i < ksY; i++)
    {
        sum0 += (data[i]);
    }
    sum1 = sum0 + (data[0]);
    sum2 = sum0 + (data[ksY]);
    temp[0][col] = sum1;
    temp[1][col] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(col < (THREADS-(ksX-1)))
    {
        col += anX;
        int posX = dst_startX - dst_x_off + col - anX;
        int posY = (gY << 1);

        uint4 tmp_sum[2]={(uint4)(0,0,0,0),(uint4)(0,0,0,0)};
        for(int k=0; k<2; k++)
            for(int i=-anX; i<=anX; i++)
            {
                tmp_sum[k] += temp[k][col+i];
            }
        for(int i=0; i<2; i++)
        {
            if(posX >= 0 && posX < dst_cols && (posY+i) >= 0 && (posY+i) < dst_rows)
                dst[(dst_startY+i) * (dst_step>>2)+ dst_startX + col - anX] = convert_uchar4(convert_float4(tmp_sum[i])/alpha);
        }

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////32fC1////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void boxFilter_C1_D5(__global const float *restrict src, __global float *dst, float alpha,
                                     int src_offset, int src_whole_rows, int src_whole_cols, int src_step,
                                     int dst_offset, int dst_rows, int dst_cols, int dst_step
                                     )
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);

    int src_x_off = (src_offset % src_step) >> 2;
    int src_y_off = src_offset / src_step;
    int dst_x_off = (dst_offset % dst_step) >> 2;
    int dst_y_off = dst_offset / dst_step;

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;
    int startY = (gY << 1) - anY + src_y_off;
    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY << 1) + dst_y_off;
    int end_addr = (src_whole_rows-1)*(src_step>>2) + src_whole_cols-4;
    float data[ksY+1];
    __local float temp[2][THREADS];
#ifdef BORDER_CONSTANT
    bool con;
    float ss;
    for(int i=0; i < ksY+1; i++)
    {
        con = startX+col >= 0 && startX+col < src_whole_cols && startY+i >= 0 && startY+i < src_whole_rows;
          //int cur_addr = clamp((startY+i)*(src_step>>2)+(startX+col),0,end_addr);
        //ss = src[cur_addr];

        int cur_col = clamp(startX + col, 0, src_whole_cols);
        //ss = src[(startY+i)*(src_step>>2) + cur_col];
        ss = (startY+i)<src_whole_rows&&(startY+i)>=0&&cur_col>=0&&cur_col<src_whole_cols?src[(startY+i)*(src_step>>2) + cur_col]:0;

        data[i] = con ? ss : 0.f;
    }
#else
   for(int i=0; i < ksY+1; i++)
   {
          int selected_row;
          int selected_col;
          selected_row = ADDR_H(startY+i, 0, src_whole_rows);
          selected_row = ADDR_B(startY+i, src_whole_rows, selected_row);

          selected_col = ADDR_L(startX+col, 0, src_whole_cols);
          selected_col = ADDR_R(startX+col, src_whole_cols, selected_col);

          data[i] = src[selected_row * (src_step>>2) + selected_col];
   }

#endif
    float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
    for(int i=1; i < ksY; i++)
    {
        sum0 += (data[i]);
    }
    sum1 = sum0 + (data[0]);
    sum2 = sum0 + (data[ksY]);
    temp[0][col] = sum1;
    temp[1][col] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(col < (THREADS-(ksX-1)))
    {
        col += anX;
        int posX = dst_startX - dst_x_off + col - anX;
        int posY = (gY << 1);

        float tmp_sum[2]={0.0, 0.0};
        for(int k=0; k<2; k++)
            for(int i=-anX; i<=anX; i++)
            {
                tmp_sum[k] += temp[k][col+i];
            }
        for(int i=0; i<2; i++)
        {
            if(posX >= 0 && posX < dst_cols && (posY+i) >= 0 && (posY+i) < dst_rows)
                dst[(dst_startY+i) * (dst_step>>2)+ dst_startX + col - anX] = tmp_sum[i]/alpha;
        }

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////32fC4////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void boxFilter_C4_D5(__global const float4 *restrict src, __global float4 *dst, float alpha,
                                     int src_offset, int src_whole_rows, int src_whole_cols, int src_step,
                                     int dst_offset, int dst_rows, int dst_cols, int dst_step
                                     )
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);

    int src_x_off = (src_offset % src_step) >> 4;
    int src_y_off = src_offset / src_step;
    int dst_x_off = (dst_offset % dst_step) >> 4;
    int dst_y_off = dst_offset / dst_step;

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;
    int startY = (gY << 1) - anY + src_y_off;
    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY << 1) + dst_y_off;
    int end_addr = (src_whole_rows-1)*(src_step>>4) + src_whole_cols-16;
    float4 data[ksY+1];
    __local float4 temp[2][THREADS];
#ifdef BORDER_CONSTANT
    bool con;
    float4 ss;
    for(int i=0; i < ksY+1; i++)
    {
        con = startX+col >= 0 && startX+col < src_whole_cols && startY+i >= 0 && startY+i < src_whole_rows;
            //int cur_addr = clamp((startY+i)*(src_step>>4)+(startX+col),0,end_addr);
        //ss = src[cur_addr];

        int cur_col = clamp(startX + col, 0, src_whole_cols);
        //ss = src[(startY+i)*(src_step>>4) + cur_col];
        ss = (startY+i)<src_whole_rows&&(startY+i)>=0&&cur_col>=0&&cur_col<src_whole_cols?src[(startY+i)*(src_step>>4) + cur_col]:0;

        data[i] = con ? ss : (float4)(0.0,0.0,0.0,0.0);
    }
#else
   for(int i=0; i < ksY+1; i++)
   {
          int selected_row;
          int selected_col;
          selected_row = ADDR_H(startY+i, 0, src_whole_rows);
          selected_row = ADDR_B(startY+i, src_whole_rows, selected_row);

          selected_col = ADDR_L(startX+col, 0, src_whole_cols);
          selected_col = ADDR_R(startX+col, src_whole_cols, selected_col);

          data[i] = src[selected_row * (src_step>>4) + selected_col];
   }

#endif
    float4 sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
    for(int i=1; i < ksY; i++)
    {
        sum0 += (data[i]);
    }
    sum1 = sum0 + (data[0]);
    sum2 = sum0 + (data[ksY]);
    temp[0][col] = sum1;
    temp[1][col] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(col < (THREADS-(ksX-1)))
    {
        col += anX;
        int posX = dst_startX - dst_x_off + col - anX;
        int posY = (gY << 1);

        float4 tmp_sum[2]={(float4)(0.0,0.0,0.0,0.0), (float4)(0.0,0.0,0.0,0.0)};
        for(int k=0; k<2; k++)
            for(int i=-anX; i<=anX; i++)
            {
                tmp_sum[k] += temp[k][col+i];
            }
        for(int i=0; i<2; i++)
        {
            if(posX >= 0 && posX < dst_cols && (posY+i) >= 0 && (posY+i) < dst_rows)
                dst[(dst_startY+i) * (dst_step>>4)+ dst_startX + col - anX] = tmp_sum[i]/alpha;
        }

    }
}
