//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// An interval class
//

#ifndef INCLUDED_IMATHINTERVAL_H
#define INCLUDED_IMATHINTERVAL_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// An Interval has a min and a max and some miscellaneous
/// functions. It is basically a Box<T> that allows T to be a scalar.
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Interval
{
  public:

    /// @{
    /// @name Direct access to bounds
    
    /// The minimum value of the interval
    T min;

    /// The minimum value of the interval
    T max;

    /// @}
    
    /// @{
    /// @name Constructors

    /// Initialize to the empty interval
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Interval() IMATH_NOEXCEPT;

    /// Intitialize to a single point
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Interval (const T& point) IMATH_NOEXCEPT;

    /// Intitialize to a given (min,max)
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Interval (const T& minT, const T& maxT) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Comparison

    /// Equality
    IMATH_HOSTDEVICE constexpr bool operator== (const Interval<T>& src) const IMATH_NOEXCEPT;
    /// Inequality
    IMATH_HOSTDEVICE constexpr bool operator!= (const Interval<T>& src) const IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Manipulation

    /// Set the interval to be empty. An interval is empty if the
    /// minimum is greater than the maximum.
    IMATH_HOSTDEVICE void makeEmpty() IMATH_NOEXCEPT;

    /// Extend the interval to include the given point.
    IMATH_HOSTDEVICE void extendBy (const T& point) IMATH_NOEXCEPT;

    /// Extend the interval to include the given interval
    IMATH_HOSTDEVICE void extendBy (const Interval<T>& interval) IMATH_NOEXCEPT;

    /// Make the interval include the entire range of the base type.
    IMATH_HOSTDEVICE void makeInfinite() IMATH_NOEXCEPT;

    /// @}

    /// @{
    ///	@name Query

    /// Return the size of the interval. The size is (max-min). An empty box has a size of 0.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T size() const IMATH_NOEXCEPT;

    /// Return the center of the interval. The center is defined as
    /// (max+min)/2. The center of an empty interval is undefined.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T center() const IMATH_NOEXCEPT;

    /// Return true if the given point is inside the interval, false otherwise.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool intersects (const T& point) const IMATH_NOEXCEPT;

    /// Return true if the given interval is inside the interval, false otherwise.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool intersects (const Interval<T>& interval) const IMATH_NOEXCEPT;

    /// Return true if the interval is empty, false otherwise. An
    /// empty interval's minimum is greater than its maximum.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool isEmpty() const IMATH_NOEXCEPT;

    /// Return true if the interval is larger than a single point,
    /// false otherwise.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool hasVolume() const IMATH_NOEXCEPT;

    /// Return true if the interval contains all points, false
    /// otherwise.  An infinite box has a mimimum of std::numeric_limits<T>::lowest()
    /// and a maximum of std::numeric_limits<T>::max()
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool isInfinite() const IMATH_NOEXCEPT;

    /// @}
};

/// Stream output, as "(min max)"
template <class T> std::ostream& operator<< (std::ostream& s, const Interval<T>& v);

/// Interval of type float
typedef Interval<float> Intervalf;

/// Interval of type double
typedef Interval<double> Intervald;

/// Interval of type short
typedef Interval<short> Intervals;

/// Interval of type integer
typedef Interval<int> Intervali;

template <class T>
IMATH_HOSTDEVICE inline IMATH_CONSTEXPR14 Interval<T>::Interval() IMATH_NOEXCEPT
{
    makeEmpty();
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Interval<T>::Interval (const T& point) IMATH_NOEXCEPT
{
    min = point;
    max = point;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Interval<T>::Interval (const T& minV, const T& maxV) IMATH_NOEXCEPT
{
    min = minV;
    max = maxV;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Interval<T>::operator== (const Interval<T>& src) const IMATH_NOEXCEPT
{
    return (min == src.min && max == src.max);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Interval<T>::operator!= (const Interval<T>& src) const IMATH_NOEXCEPT
{
    return (min != src.min || max != src.max);
}

template <class T>
IMATH_HOSTDEVICE inline void
Interval<T>::makeEmpty() IMATH_NOEXCEPT
{
    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::lowest();
}

template <class T>
IMATH_HOSTDEVICE inline void
Interval<T>::makeInfinite() IMATH_NOEXCEPT
{
    min = std::numeric_limits<T>::lowest();
    max = std::numeric_limits<T>::max();
}


template <class T>
IMATH_HOSTDEVICE inline void
Interval<T>::extendBy (const T& point) IMATH_NOEXCEPT
{
    if (point < min)
        min = point;

    if (point > max)
        max = point;
}

template <class T>
IMATH_HOSTDEVICE inline void
Interval<T>::extendBy (const Interval<T>& interval) IMATH_NOEXCEPT
{
    if (interval.min < min)
        min = interval.min;

    if (interval.max > max)
        max = interval.max;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Interval<T>::intersects (const T& point) const IMATH_NOEXCEPT
{
    return point >= min && point <= max;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Interval<T>::intersects (const Interval<T>& interval) const IMATH_NOEXCEPT
{
    return interval.max >= min && interval.min <= max;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Interval<T>::size() const IMATH_NOEXCEPT
{
    if (isEmpty())
        return T(0);
    
    return max - min;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Interval<T>::center() const IMATH_NOEXCEPT
{
    return (max + min) / 2;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Interval<T>::isEmpty() const IMATH_NOEXCEPT
{
    return max < min;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Interval<T>::hasVolume() const IMATH_NOEXCEPT
{
    return max > min;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Interval<T>::isInfinite() const IMATH_NOEXCEPT
{
    if (min != std::numeric_limits<T>::lowest() || max != std::numeric_limits<T>::max())
        return false;

    return true;
}

/// Stream output
template <class T>
std::ostream&
operator<< (std::ostream& s, const Interval<T>& v)
{
    return s << '(' << v.min << ' ' << v.max << ')';
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHINTERVAL_H
