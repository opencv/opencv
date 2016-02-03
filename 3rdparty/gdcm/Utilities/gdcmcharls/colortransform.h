// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 
#ifndef CHARLS_COLORTRANSFORM
#define CHARLS_COLORTRANSFORM

//
// This file defines simple classes that define (lossless) color transforms.
// They are invoked in processline.h to convert between decoded values and the internal line buffers. 
// Color transforms work best for computer generated images. 
//

template<class sample>
struct TransformNoneImpl
{
	typedef sample SAMPLE;

	inlinehint Triplet<SAMPLE> operator() (int v1, int v2, int v3)
	{ return Triplet<SAMPLE>(v1, v2, v3); }
};


template<class sample>
struct TransformNone : public TransformNoneImpl<sample>
{
	typedef struct TransformNoneImpl<sample> INVERSE;
};



template<class sample>
struct TransformHp1
{
	enum { RANGE = 1 << sizeof(sample)*8 };
	typedef sample SAMPLE;

	struct INVERSE
	{
		INVERSE(const TransformHp1&) {};

		inlinehint Triplet<SAMPLE> operator() (int v1, int v2, int v3)
		{ return Triplet<SAMPLE>(v1 + v2 - RANGE/2, v2, v3 + v2 - RANGE/2); }
	};

	inlinehint Triplet<SAMPLE> operator() (int R, int G, int B)
	{
		Triplet<SAMPLE> hp1;
		hp1.v2 = SAMPLE(G);
		hp1.v1 = SAMPLE(R - G + RANGE/2);
		hp1.v3 = SAMPLE(B - G + RANGE/2);
		return hp1;
	}
};





template<class sample>
struct TransformHp2
{
	enum { RANGE = 1 << sizeof(sample)*8 };
	typedef sample SAMPLE;

	struct INVERSE
	{
		INVERSE(const TransformHp2&) {};

		inlinehint   Triplet<SAMPLE> operator() (int v1, int v2, int v3)
		{
			Triplet<SAMPLE> rgb;
			rgb.R  = SAMPLE(v1 + v2 - RANGE/2);          // new R
			rgb.G  = SAMPLE(v2);                     // new G				
			rgb.B  = SAMPLE(v3 + ((rgb.R + rgb.G) >> 1) - RANGE/2); // new B
			return rgb;
		}
	};

	inlinehint Triplet<SAMPLE> operator() (int R, int G, int B)
	{
		return Triplet<SAMPLE>(R - G + RANGE/2, G, B - ((R+G )>>1) - RANGE/2);
	}


};



template<class sample>
struct TransformHp3
{
	enum { RANGE = 1 << sizeof(sample)*8 };
	typedef sample SAMPLE;

	struct INVERSE
	{
		INVERSE(const TransformHp3&) {};

		inlinehint Triplet<SAMPLE> operator() (int v1, int v2, int v3)
		{
			int G = v1 - ((v3 + v2)>>2) + RANGE/4;
			Triplet<SAMPLE> rgb;
			rgb.R  = SAMPLE(v3 + G - RANGE/2); // new R
			rgb.G  = SAMPLE(G);             // new G				
			rgb.B  = SAMPLE(v2 + G - RANGE/2); // new B
			return rgb;
		}
	};

	inlinehint Triplet<SAMPLE> operator() (int R, int G, int B)
	{
		Triplet<SAMPLE> hp3;		
		hp3.v2 = SAMPLE(B - G + RANGE/2);
		hp3.v3 = SAMPLE(R - G + RANGE/2);
		hp3.v1 = SAMPLE(G + ((hp3.v2 + hp3.v3)>>2)) - RANGE/4;
		return hp3;
	}
};


// Transform class that shifts bits towards the high bit when bitcount is not 8 or 16
// needed to make the HP color transforms work correctly.

template<class TRANSFORM>
struct TransformShifted
{	
	typedef typename TRANSFORM::SAMPLE SAMPLE;

	struct INVERSE
	{
		INVERSE(const TransformShifted& transform) : 
			_shift(transform._shift),
			_inverseTransform(transform._colortransform)
		{}

		inlinehint Triplet<SAMPLE> operator() (int v1, int v2, int v3)
		{
			Triplet<SAMPLE> result = _inverseTransform(v1 << _shift, v2 << _shift, v3 << _shift);
			
			return Triplet<SAMPLE>(result.R >> _shift, result.G >> _shift, result.B >> _shift);
		}

		inlinehint Quad<SAMPLE> operator() (int v1, int v2, int v3, int v4)
		{
			Triplet<SAMPLE> result = _inverseTransform(v1 << _shift, v2 << _shift, v3 << _shift);
			
			return Quad<SAMPLE>(result.R >> _shift, result.G >> _shift, result.B >> _shift, v4);
		}

		int _shift;
		typename TRANSFORM::INVERSE _inverseTransform;
	};


	TransformShifted(int shift) :
		 _shift(shift)
	{
	}

	inlinehint Triplet<SAMPLE> operator() (int R, int G, int B)
	{
		Triplet<SAMPLE> result = _colortransform(R << _shift, G << _shift, B << _shift);

		return Triplet<SAMPLE>(result.R >> _shift, result.G >> _shift, result.B >> _shift);
	}

	inlinehint Quad<SAMPLE> operator() (int R, int G, int B, int A)
	{
		Triplet<SAMPLE> result = _colortransform(R << _shift, G << _shift, B << _shift);

		return Quad<SAMPLE>(result.R >> _shift, result.G >> _shift, result.B >> _shift, A);
	}

	int _shift;
	TRANSFORM _colortransform;
};



#endif
