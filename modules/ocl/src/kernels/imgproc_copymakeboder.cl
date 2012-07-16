//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin zero.lin@amd.com
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
//


#define get(a,b,c) (( b >= top & b < srcRows+top & a >= left & a < srcCols+left )? c : 8)
__kernel void copyConstBorder_C1_D0(__global uchar * src, __global uchar * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, uchar nVal, int srcStep, int dstStep)
{
	int idx = get_global_id(0);
	int tpr = (dstCols + 3 + (dstOffset&3))>>2;
	int dx  = ((idx%(tpr))<<2) - (dstOffset&3);
    int dy = idx/(tpr);
    
	__global uchar4 * d=(__global uchar4 *)(dst + dstOffset + dy*dstStep + dx);
	int start=srcOffset + (dy-top)*srcStep + (dx-left);
	uchar8 s=*((__global uchar8 *)(src + ((start>>2)<<2) ));
	uchar4 v;
	
	uchar sv[9]={s.s0,s.s1,s.s2,s.s3,s.s4,s.s5,s.s6,s.s7,nVal};
	
	int det=start&3;
	v.x=sv[get(dx,dy,det)];
	v.y=sv[get(dx+1,dy,det+1)];
	v.z=sv[get(dx+2,dy,det+2)];
	v.w=sv[get(dx+3,dy,det+3)];
	
	if(dy<dstRows)
	{
		uchar4 res = *d;
		res.x = (dx>=0 && dx<dstCols) ? v.x : res.x;
		res.y = (dx+1>=0 && dx+1<dstCols) ? v.y : res.y;
		res.z = (dx+2>=0 && dx+2<dstCols) ? v.z : res.z;
		res.w = (dx+3>=0 && dx+3<dstCols) ? v.w : res.w;
	
		*d=res;
	}
}
#undef get(a,b,c)

#define get(a,b,c,d) (( b >= top & b < srcRows+top & a >= left & a < srcCols+left )? c : d)
__kernel void copyConstBorder_C1_D4(__global int * src, __global int * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, int nVal, int srcStep, int dstStep)
{
    int idx = get_global_id(0);
	int tpr = (dstCols + 3)>>2;
	int dx  = (idx%(tpr))<<2;
    int dy = idx/(tpr);
    
	__global int4 * d=(__global int4 *)(dst+dy*dstStep+dx);
	int4 s=*((__global int4 *)(src + srcOffset + (dy-top)*srcStep + (dx-left) ));
	int4 v;
	
	v.x=get(dx,dy,s.x,nVal);
	v.y=get(dx+1,dy,s.y,nVal);
	v.z=get(dx+2,dy,s.z,nVal);
	v.w=get(dx+3,dy,s.w,nVal);
	
	if(dy<dstRows)
	{
		int4 res = *d;
		v.y = (dx+1<dstCols) ? v.y : res.y;
		v.z = (dx+2<dstCols) ? v.z : res.z;
		v.w = (dx+3<dstCols) ? v.w : res.w;
	
		*d=v;
	}
}
#undef get(a,b,c,d)

#define get(a,b,c) ( a < srcCols+left ? b : c)
__kernel void copyReplicateBorder_C1_D4(__global int * src, __global int * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, int nVal, int srcStep, int dstStep)
{
    int idx = get_global_id(0);
	int tpr = (dstCols + 3)>>2;
	int dx  = (idx%(tpr))<<2;
    int dy = idx/(tpr);

	__global int4 * d=(__global int4 *)(dst + dstOffset + dy*dstStep + dx);
	int c=clamp(dx-left,0,srcCols-1);
	int4 s=*((__global int4 *)(src + srcOffset + clamp(dy-top,0,srcRows-1) * srcStep + c ));
	int sa[4]={s.x,s.y,s.z,s.w};
	int4 v;
	
	v.x=get(dx,sa[max(0,(dx-left)-c)],sa[srcCols-1-c]);
	v.y=get(dx+1,sa[max(0,(dx+1-left)-c)],sa[srcCols-1-c]);
	v.z=get(dx+2,sa[max(0,(dx+2-left)-c)],sa[srcCols-1-c]);
	v.w=get(dx+3,sa[max(0,(dx+3-left)-c)],sa[srcCols-1-c]);
	
	if(dy<dstRows)
	{
		int4 res = *d;
		v.y = (dx+1<dstCols) ? v.y : res.y;
		v.z = (dx+2<dstCols) ? v.z : res.z;
		v.w = (dx+3<dstCols) ? v.w : res.w;
	
		*d=v;
	}
}

__kernel void copyReplicateBorder_C1_D0(__global uchar * src, __global uchar * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, uchar nVal, int srcStep, int dstStep)
{
	int idx = get_global_id(0);
	int tpr = (dstCols + 3 + (dstOffset&3))>>2;
	int dx  = ((idx%(tpr))<<2) - (dstOffset&3);
    int dy = idx/(tpr);
    
	__global uchar4 * d=(__global uchar4 *)(dst + dstOffset + dy*dstStep + dx);
	int c=clamp(dx-left,0,srcCols-1);
	int start= srcOffset + clamp(dy-top,0,srcRows-1) * srcStep + c;
	uchar8 s=*((__global uchar8 *)(src + ((start>>2)<<2) ));
	uchar4 v;
	
	uchar sa[8]={s.s0,s.s1,s.s2,s.s3,s.s4,s.s5,s.s6,s.s7};
	
	int det=start&3;
	v.x=get(dx,sa[max(0,(dx-left)-c)+det],sa[srcCols-1-c+det]);
	v.y=get(dx+1,sa[max(0,(dx+1-left)-c)+det],sa[srcCols-1-c+det]);
	v.z=get(dx+2,sa[max(0,(dx+2-left)-c)+det],sa[srcCols-1-c+det]);
	v.w=get(dx+3,sa[max(0,(dx+3-left)-c)+det],sa[srcCols-1-c+det]);
	
	if(dy<dstRows)
	{
		uchar4 res = *d;
		res.x = (dx>=0 && dx<dstCols) ? v.x : res.x;
		res.y = (dx+1>=0 && dx+1<dstCols) ? v.y : res.y;
		res.z = (dx+2>=0 && dx+2<dstCols) ? v.z : res.z;
		res.w = (dx+3>=0 && dx+3<dstCols) ? v.w : res.w;
	
		*d=res;
	}
}
#undef get(a,b,c)

//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define edge(x,size,rx) rx = abs(x) % ((size<<1)-2); rx = (rx>=size?(size<<1)-2:rx<<1) - rx;
__kernel void copyReflectBorder_C1_D4(__global int * src, __global int * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, int nVal, int srcStep, int dstStep)
{
    int idx = get_global_id(0);
	int tpr = (dstCols + 3)>>2;
	int dx  = (idx%(tpr))<<2;
    int dy = idx/(tpr);

	__global int4 * d=(__global int4 *)(dst + dstOffset + dy*dstStep + dx);
	uint4 id;
	edge(dx-left,srcCols,id.x);
	edge(dx-left+1,srcCols,id.x);
	edge(dx-left+2,srcCols,id.x);
	edge(dx-left+3,srcCols,id.x);



	int start=min(id.x,id.w);
	int4 s=*((__global int4 *)(src + srcOffset + clamp(dy-top,0,srcRows-1) * srcStep + start));
	int sa[4]={s.x,s.y,s.z,s.w};

	int4 v=(int4)(sa[(id.x-start)],sa[(id.y-start)],sa[(id.z-start)],sa[(id.w-start)]);
	
	
	if(dy<dstRows)
	{
		int4 res = *d;
		v.y = (dx+1<dstCols) ? v.y : res.y;
		v.z = (dx+2<dstCols) ? v.z : res.z;
		v.w = (dx+3<dstCols) ? v.w : res.w;
	
		*d=v;
	}
}

__kernel void copyReflectBorder_C1_D0(__global uchar * src, __global uchar * dst, int srcOffset, int dstOffset, 
								int srcCols, int srcRows, int dstCols, int dstRows, 
								int top, int left, uchar nVal, int srcStep, int dstStep)
{
    int idx = get_global_id(0);
	int tpr = (dstCols + 3 + (dstOffset&3))>>2;
	int dx  = ((idx%(tpr))<<2) - (dstOffset&3);
    int dy = idx/(tpr);
    
	__global uchar4 * d=(__global uchar4 *)(dst + dstOffset + dy*dstStep + dx);
	uint4 id;
	edge(dx-left,srcCols,id.x);
	edge(dx-left+1,srcCols,id.x);
	edge(dx-left+2,srcCols,id.x);
	edge(dx-left+3,srcCols,id.x);

	int start=min(id.x,id.w) + srcOffset;
	uchar8 s=*((__global uchar8 *)(src + clamp(dy-top,0,srcRows-1) * srcStep + ((start>>2)<<2) ));
	uchar sa[8]={s.s0,s.s1,s.s2,s.s3,s.s4,s.s5,s.s6,s.s7};
	
	int det=start&3;
	uchar4 v=(uchar4)(sa[(id.x-start)+det],sa[(id.y-start)+det],sa[(id.z-start)+det],sa[(id.w-start)+det]);
	
	if(dy<dstRows)
	{
		uchar4 res = *d;
		res.x = (dx>=0 && dx<dstCols) ? v.x : res.x;
		res.y = (dx+1>=0 && dx+1<dstCols) ? v.y : res.y;
		res.z = (dx+2>=0 && dx+2<dstCols) ? v.z : res.z;
		res.w = (dx+3>=0 && dx+3<dstCols) ? v.w : res.w;
	
		*d=res;
	}
}
#undef edge(x,size,rx)

