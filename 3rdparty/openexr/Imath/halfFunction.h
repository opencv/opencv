//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

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

/// @cond Doxygen_Suppress

#include "half.h"

#include "ImathConfig.h"
#ifndef IMATH_HAVE_LARGE_STACK
#    include <string.h> // need this for memset
#else
#endif

#include <float.h>

template <class T> class halfFunction
{
  public:
    //------------
    // Constructor
    //------------

    template <class Function>
    halfFunction (Function f,
                  half domainMin = -HALF_MAX,
                  half domainMax = HALF_MAX,
                  T defaultValue = 0,
                  T posInfValue  = 0,
                  T negInfValue  = 0,
                  T nanValue     = 0);

#ifndef IMATH_HAVE_LARGE_STACK
    ~halfFunction() { delete[] _lut; }
    halfFunction (const halfFunction&) = delete;
    halfFunction& operator= (const halfFunction&) = delete;
    halfFunction (halfFunction&&)                 = delete;
    halfFunction& operator= (halfFunction&&) = delete;
#endif

    //-----------
    // Evaluation
    //-----------

    T operator() (half x) const;

  private:
#ifdef IMATH_HAVE_LARGE_STACK
    T _lut[1 << 16];
#else
    T* _lut;
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
#ifndef IMATH_HAVE_LARGE_STACK
    _lut = new T[1 << 16];
#endif

    for (int i = 0; i < (1 << 16); i++)
    {
        half x;
        x.setBits (i);

        if (x.isNan())
            _lut[i] = nanValue;
        else if (x.isInfinity())
            _lut[i] = x.isNegative() ? negInfValue : posInfValue;
        else if (x < domainMin || x > domainMax)
            _lut[i] = defaultValue;
        else
            _lut[i] = f (x);
    }
}

template <class T>
inline T
halfFunction<T>::operator() (half x) const
{
    return _lut[x.bits()];
}


/// @endcond


#endif
