// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 



#ifndef CHARLS_LOSSLESSTRAITS
#define CHARLS_LOSSLESSTRAITS

#include "header.h"

//
// optimized trait classes for lossless compression of 8 bit color and 8/16 bit monochrome images.
// This class is assumes MAXVAL correspond to a whole number of bits, and no custom RESET value is set when encoding.
// The point of this is to have the most optimized code for the most common and most demanding scenario. 

template <class sample, LONG bitsperpixel>
struct LosslessTraitsImplT 
{
	typedef sample SAMPLE;
	enum { 
		NEAR  = 0,
		bpp   = bitsperpixel,
		qbpp  = bitsperpixel,
		RANGE = (1 << bpp),
		MAXVAL= (1 << bpp) - 1,
		LIMIT = 2 * (bitsperpixel + MAX(8,bitsperpixel)),
		RESET = BASIC_RESET
	};

	static inlinehint LONG ComputeErrVal(LONG d)
	{ return ModRange(d); }
		
	static inlinehint bool IsNear(LONG lhs, LONG rhs) 
		{ return lhs == rhs; }

	static inlinehint LONG ModRange(LONG Errval) 
	{
		return LONG(Errval << (LONG_BITCOUNT  - bpp)) >> (LONG_BITCOUNT  - bpp); 
	}
	
	static inlinehint SAMPLE ComputeReconstructedSample(LONG Px, LONG ErrVal)
	{
		return SAMPLE(MAXVAL & (Px + ErrVal)); 
	}

	static inlinehint LONG CorrectPrediction(LONG Pxc) 
	{
		if ((Pxc & MAXVAL) == Pxc)
			return Pxc;
		
		return (~(Pxc >> (LONG_BITCOUNT-1))) & MAXVAL;		
	}

};

template <class SAMPLE, LONG bpp>
struct LosslessTraitsT : public LosslessTraitsImplT<SAMPLE, bpp> 
{
	typedef SAMPLE PIXEL;
};



template<>
struct LosslessTraitsT<BYTE,8> : public LosslessTraitsImplT<BYTE, 8> 
{
	typedef SAMPLE PIXEL;

	static inlinehint signed char ModRange(LONG Errval) 
		{ return (signed char)Errval; }

	static inlinehint LONG ComputeErrVal(LONG d)
	{ return (signed char)(d); }

	static inlinehint BYTE ComputeReconstructedSample(LONG Px, LONG ErrVal)
		{ return BYTE(Px + ErrVal);  }
	
};



template<>
struct LosslessTraitsT<USHORT,16> : public LosslessTraitsImplT<USHORT,16> 
{
	typedef SAMPLE PIXEL;

	static inlinehint short ModRange(LONG Errval) 
		{ return short(Errval); }

	static inlinehint LONG ComputeErrVal(LONG d)
	{ return short(d); }

	static inlinehint SAMPLE ComputeReconstructedSample(LONG Px, LONG ErrVal)
		{ return SAMPLE(Px + ErrVal);  }

};




template<class SAMPLE, LONG bpp>
struct LosslessTraitsT<Triplet<SAMPLE>,bpp> : public LosslessTraitsImplT<SAMPLE,bpp>
{
	typedef Triplet<SAMPLE> PIXEL;

	static inlinehint bool IsNear(LONG lhs, LONG rhs) 
		{ return lhs == rhs; }

	static inlinehint bool IsNear(PIXEL lhs, PIXEL rhs) 
		{ return lhs == rhs; }


	static inlinehint SAMPLE ComputeReconstructedSample(LONG Px, LONG ErrVal)
		{ return SAMPLE(Px + ErrVal);  }


};

#endif
