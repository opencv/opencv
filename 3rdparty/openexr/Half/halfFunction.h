///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// Primary authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>


//---------------------------------------------------------------------------
//
//	halfFunction<T> -- a class for fast evaluation
//			   of half --> T functions
//
//	The constructor for a halfFunction object,
//
//	    halfFunction (function,
//			  domainMin, domainMax,
//			  defaultValue,
//			  posInfValue, negInfValue,
//			  nanValue);
//
//	evaluates the function for all finite half values in the interval
//	[domainMin, domainMax], and stores the results in a lookup table.
//	For finite half values that are not in [domainMin, domainMax], the
//	constructor stores defaultValue in the table.  For positive infinity,
//	negative infinity and NANs, posInfValue, negInfValue and nanValue
//	are stored in the table.
//
//	The tabulated function can then be evaluated quickly for arbitrary
//	half values by calling the the halfFunction object's operator()
//	method.
//
//	Example:
//
//	    #include <math.h>
//	    #include <halfFunction.h>
//
//	    halfFunction<half> hsin (sin);
//
//	    halfFunction<half> hsqrt (sqrt,		// function
//				      0, HALF_MAX,	// domain
//				      half::qNan(),	// sqrt(x) for x < 0
//				      half::posInf(),	// sqrt(+inf)
//				      half::qNan(),	// sqrt(-inf)
//				      half::qNan());	// sqrt(nan)
//
//	    half x = hsin (1);
//	    half y = hsqrt (3.5);
//
//---------------------------------------------------------------------------

#ifndef _HALF_FUNCTION_H_
#define _HALF_FUNCTION_H_

#include "half.h"

#include "IlmBaseConfig.h"
#ifndef ILMBASE_HAVE_LARGE_STACK  
#include <string.h>     // need this for memset
#else 
#endif

#include <float.h>


template <class T>
class halfFunction
{
  public:

    //------------
    // Constructor
    //------------

    template <class Function>
    halfFunction (Function f,
		  half domainMin = -HALF_MAX,
		  half domainMax =  HALF_MAX,
		  T defaultValue = 0,
		  T posInfValue  = 0,
		  T negInfValue  = 0,
		  T nanValue     = 0);

#ifndef ILMBASE_HAVE_LARGE_STACK
    ~halfFunction () { delete [] _lut; }    
#endif
    
    //-----------
    // Evaluation
    //-----------

    T		operator () (half x) const;

  private:

#ifdef ILMBASE_HAVE_LARGE_STACK
    T		_lut[1 << 16];
#else
    T *         _lut;
#endif
};


//---------------
// Implementation
//---------------

template <class T>
template <class Function>
halfFunction<T>::halfFunction (Function f,
			       half domainMin,
			       half domainMax,
			       T defaultValue,
			       T posInfValue,
			       T negInfValue,
			       T nanValue)
{
#ifndef ILMBASE_HAVE_LARGE_STACK
    _lut = new T[1<<16];
    memset (_lut, 0 , (1<<16) * sizeof(T));
#endif
    
    for (int i = 0; i < (1 << 16); i++)
    {
	half x;
	x.setBits (i);

	if (x.isNan())
	    _lut[i] = nanValue;
	else if (x.isInfinity())
	    _lut[i] = x.isNegative()? negInfValue: posInfValue;
	else if (x < domainMin || x > domainMax)
	    _lut[i] = defaultValue;
	else
	    _lut[i] = f (x);
    }
}


template <class T>
inline T
halfFunction<T>::operator () (half x) const
{
    return _lut[x.bits()];
}


#endif
