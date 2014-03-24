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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Shengen Yan,yanshengen@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wenju He, wenju@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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

inline short2 do_mean_shift(int x0, int y0, __global uchar4* out,int out_step,
               __global uchar4* in, int in_step, int dst_off, int src_off,
               int cols, int rows, int sp, int sr, int maxIter, float eps)
{
    int isr2 = sr*sr;
    in_step = in_step >> 2;
    out_step = out_step >> 2;
    src_off = src_off >> 2;
    dst_off = dst_off >> 2;
    int idx = src_off + y0 * in_step + x0;
    uchar4 c = in[idx];
    int base = dst_off + get_global_id(1)*out_step + get_global_id(0) ;

    // iterate meanshift procedure
    for( int iter = 0; iter < maxIter; iter++ )
    {
        int count = 0;
        int4 s = (int4)0;
        int sx = 0, sy = 0;

        //mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
        //deal with the image boundary
        int minx = (x0-sp)>0 ? x0-sp : 0;
        int miny = (y0-sp)>0 ? y0-sp : 0;
        int maxx = (x0+sp)<cols ? x0+sp : cols-1;
        int maxy = (y0+sp)<rows ? y0+sp : rows-1;

        for( int y = miny; y <= maxy; y++)
        {
            int rowCount = 0;
            int x = minx;
            for( ; x+3 <= maxx; x+=4 )
            {
                int id = src_off + y*in_step + x;
                uchar16 t = (uchar16)(in[id],in[id+1],in[id+2],in[id+3]);
                int norm2_1 = (t.s0 - c.x) * (t.s0 - c.x) + (t.s1 - c.y) * (t.s1 - c.y) +
                              (t.s2 - c.z) * (t.s2 - c.z);
                int norm2_2 = (t.s4 - c.x) * (t.s4 - c.x) + (t.s5 - c.y) * (t.s5 - c.y) +
                              (t.s6 - c.z) * (t.s6 - c.z);
                int norm2_3 = (t.s8 - c.x) * (t.s8 - c.x) + (t.s9 - c.y) * (t.s9 - c.y) +
                              (t.sa - c.z) * (t.sa - c.z);
                int norm2_4 = (t.sc - c.x) * (t.sc - c.x) + (t.sd - c.y) * (t.sd - c.y) +
                              (t.se - c.z) * (t.se - c.z);
                if( norm2_1 <= isr2 )
                {
                    s.x += t.s0; s.y += t.s1; s.z += t.s2;
                    sx += x; rowCount++;
                }
                if( norm2_2 <= isr2 )
                {
                    s.x += t.s4; s.y += t.s5; s.z += t.s6;
                    sx += x+1; rowCount++;
                }
                if( norm2_3 <= isr2 )
                {
                    s.x += t.s8; s.y += t.s9; s.z += t.sa;
                    sx += x+2; rowCount++;
                }
                if( norm2_4 <= isr2 )
                {
                    s.x += t.sc; s.y += t.sd; s.z += t.se;
                    sx += x+3; rowCount++;
                }
            }
            if(x == maxx)
            {
                int id = src_off + y*in_step + x;
                uchar4 t = in[id];
                int norm2 = (t.s0 - c.x) * (t.s0 - c.x) + (t.s1 - c.y) * (t.s1 - c.y) +
                            (t.s2 - c.z) * (t.s2 - c.z);
                if( norm2 <= isr2 )
                {
                    s.x += t.s0; s.y += t.s1; s.z += t.s2;
                    sx += x; rowCount++;
                }

            }
            if(x+1 == maxx)
            {
                  int id = src_off + y*in_step + x;
                  uchar8 t = (uchar8)(in[id],in[id+1]);
                  int norm2_1 = (t.s0 - c.x) * (t.s0 - c.x) + (t.s1 - c.y) * (t.s1 - c.y) +
                                (t.s2 - c.z) * (t.s2 - c.z);
                  int norm2_2 = (t.s4 - c.x) * (t.s4 - c.x) + (t.s5 - c.y) * (t.s5 - c.y) +
                                (t.s6 - c.z) * (t.s6 - c.z);
                  if( norm2_1 <= isr2 )
                  {
                      s.x += t.s0; s.y += t.s1; s.z += t.s2;
                      sx += x; rowCount++;
                  }
                  if( norm2_2 <= isr2 )
                  {
                      s.x += t.s4; s.y += t.s5; s.z += t.s6;
                      sx += x+1; rowCount++;
                  }
            }
            if(x+2 == maxx)
            {
                  int id = src_off + y*in_step + x;
                  uchar16 t = (uchar16)(in[id],in[id+1],in[id+2],in[id+3]);
                  int norm2_1 = (t.s0 - c.x) * (t.s0 - c.x) + (t.s1 - c.y) * (t.s1 - c.y) +
                                (t.s2 - c.z) * (t.s2 - c.z);
                  int norm2_2 = (t.s4 - c.x) * (t.s4 - c.x) + (t.s5 - c.y) * (t.s5 - c.y) +
                                (t.s6 - c.z) * (t.s6 - c.z);
                  int norm2_3 = (t.s8 - c.x) * (t.s8 - c.x) + (t.s9 - c.y) * (t.s9 - c.y) +
                                (t.sa - c.z) * (t.sa - c.z);
                  if( norm2_1 <= isr2 )
                  {
                      s.x += t.s0; s.y += t.s1; s.z += t.s2;
                      sx += x; rowCount++;
                  }
                  if( norm2_2 <= isr2 )
                  {
                      s.x += t.s4; s.y += t.s5; s.z += t.s6;
                      sx += x+1; rowCount++;
                  }
                  if( norm2_3 <= isr2 )
                  {
                      s.x += t.s8; s.y += t.s9; s.z += t.sa;
                      sx += x+2; rowCount++;
                  }
            }
            if(rowCount == 0)
               continue;
            count += rowCount;
            if(y == 0)
               continue;
            sy += y*rowCount;
        }

        if( count == 0 )
            break;

        int x1 = sx/count;
        int y1 = sy/count;
        s.x = s.x/count;
        s.y = s.y/count;
        s.z = s.z/count;

        int4 tmp = s - convert_int4(c);
        int norm2 = tmp.x * tmp.x + tmp.y *  tmp.y +
                    tmp.z * tmp.z;

        bool stopFlag = (x1 == x0 && y1 == y0) || (abs(x1-x0) + abs(y1-y0) + norm2 <= eps);

        x0 = x1;
        y0 = y1;
        c.x = s.x;
        c.y = s.y;
        c.z = s.z;

        if( stopFlag )
            break;
    }

    out[base] = c;

    return (short2)((short)x0, (short)y0);
}


__kernel void meanshift_kernel(__global uchar4* out, int out_step,
                               __global uchar4* in, int in_step,
                        int dst_off, int src_off, int cols, int rows,
                        int sp, int sr, int maxIter, float eps)
{
    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    if( x0 < cols && y0 < rows )
        do_mean_shift(x0, y0, out, out_step, in, in_step, dst_off, src_off,
                          cols, rows, sp, sr, maxIter, eps);
}

__kernel void meanshiftproc_kernel( __global uchar4* in, __global uchar4* outr,
                             __global short2* outsp, int instep, int outrstep,
                             int outspstep, int in_off, int outr_off, int outsp_off,
                             int cols, int rows, int sp, int sr, int maxIter, float eps )
{
    int x0 = get_global_id(0);
    int y0 = get_global_id(1);

    if( x0 < cols && y0 < rows )
    {
        //int basesp = (blockIdx.y * blockDim.y + threadIdx.y) * outspstep + (blockIdx.x * blockDim.x + threadIdx.x) * 2 * sizeof(short);
        //*(short2*)(outsp + basesp) = do_mean_shift(x0, y0, outr, outrstep, cols, rows, sp, sr, maxIter, eps);
        // we have ensured before that ((outspstep & 0x11)==0).
        outsp_off >>= 2;
        outspstep >>= 2;
        int basesp = outsp_off + y0 * outspstep + x0;
        outsp[basesp] = do_mean_shift(x0, y0, outr, outrstep, in, instep, outr_off, in_off, cols, rows, sp, sr, maxIter, eps);
//        outsp[basesp] =(short2)((short)x0,(short)y0);
    }
}
