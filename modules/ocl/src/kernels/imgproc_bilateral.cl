//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Rock Li, Rock.li@amd.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
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
//


//#pragma OPENCL EXTENSION cl_amd_printf :enable
__kernel
void bilateral4(__global uchar4 *dst,
		__global uchar4 *src,
		int rows,
		int cols,
		int channels,
		int radius,
		int wholerows,
		int wholecols,
		int src_step,
		int dst_step,
		int src_offset,
		int dst_offset,
		__constant float *sigClr,
		__constant float *sigSpc)
{
	uint lidx = get_local_id(0);
	uint lidy = get_local_id(1);
	
	uint gdx = get_global_id(0);
	uint gdy = get_global_id(1);

	uint gidx = gdx >=cols?cols-1:gdx;
	uint gidy = gdy >=rows?rows-1:gdy;

	uchar4 p,q,tmp;

	float4 pf = 0,pq = 0,pd = 0;
        float wt =0;

	int r = radius;
	int ij = 0;
	int ct = 0;

	uint index_src = src_offset/4 + gidy*src_step/4 + gidx;
	uint index_dst = dst_offset/4 + gidy*dst_step/4 + gidx;

	p = src[index_src];

	uint gx,gy;
	uint src_index,dst_index;

	for(int ii = -r;ii<r+1;ii++)
	{
		for(int jj =-r;jj<r+1;jj++)
			{
					ij = ii*ii+jj*jj;
					if(ij > mul24(radius,radius)) continue;
					gx = gidx + jj;
					gy = gidy + ii;

					src_index = src_offset/4 + gy *	 src_step/4 + gx;
					q = src[src_index];
					

					ct = abs(p.x-q.x)+abs(p.y-q.y)+abs(p.z-q.z);
					wt =sigClr[ct]*sigSpc[(ii+radius)*(2*radius+1)+jj+radius];

				        pf.x += q.x*wt;
					pf.y += q.y*wt;
					pf.z += q.z*wt;
//					pf.w += q.w*wt;

					pq += wt;

			}
	}

	pd = pf/pq;
	dst[index_dst] = convert_uchar4_rte(pd);
}

__kernel
void bilateral(__global uchar *dst,
		__global uchar *src,
		int rows,
		int cols,
		int channels,
		int radius,
		int wholerows,
		int wholecols,
		int src_step,
		int dst_step,
		int src_offset,
		int dst_offset,
		__constant float *sigClr,
		__constant float *sigSpc)
{
	uint lidx = get_local_id(0);
	uint lidy = get_local_id(1);
	
	uint gdx = get_global_id(0);
	uint gdy = get_global_id(1);

	uint gidx = gdx >=cols?cols-1:gdx;
	uint gidy = gdy >=rows?rows-1:gdy;

	uchar p,q,tmp;

	float pf = 0,pq = 0,wt = 0,pd = 0;

	int r =radius;
	int ij = 0;
	int ct = 0;

	uint index_src = src_offset + gidy*src_step + gidx;
	uint index_dst = dst_offset + gidy*dst_step + gidx;

	p = src[index_src];

	uint gx,gy;
	uint src_index,dst_index;

	for(int ii = -r;ii<r+1;ii++)
	{
		for(int jj =-r;jj<r+1;jj++)
			{
					ij = ii*ii+jj*jj;
					if(ij > mul24(radius,radius)) continue;

					gx = gidx + jj;
					gy = gidy + ii;

					
					src_index = src_offset + gy * src_step + gx;
					q = src[src_index];

					ct = abs(p-q);
					wt =sigClr[ct]*sigSpc[(ii+radius)*(2*radius+1)+jj+radius];

					pf += q*wt;
					
					pq += wt;
			}
	}
	pd = pf/pq;
	dst[index_dst] = convert_uchar_rte(pd);

}

