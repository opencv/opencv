//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMATHFORWARD_H
#define INCLUDED_IMATHFORWARD_H

#include "ImathNamespace.h"
#include "ImathExport.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

/// @cond Doxygen_Suppress

//
// Basic template type declarations.
//

//
// Note: declarations with attributes generate warnings with
// -Wattributes or -Wall if the type is already defined, i.e. the
// header is already included. To avoid the warning, only make the
// forward declaration if the header has not yet been included.
//

#ifndef INCLUDED_IMATHBOX_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Box;
#endif
#ifndef INCLUDED_IMATHCOLOR_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Color3;
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Color4;
#endif
#ifndef INCLUDED_IMATHEULER_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Euler;
#endif
#ifndef INCLUDED_IMATHFRUSTUM_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Frustum;
#endif
#ifndef INCLUDED_IMATHFRUSTUMTEST_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE FrustumTest;
#endif
#ifndef INCLUDED_IMATHINTERVAL_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Interval;
#endif
#ifndef INCLUDED_IMATHLINE_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Line3;
#endif
#ifndef INCLUDED_IMATHMATRIX_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Matrix33;
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Matrix44;
#endif
#ifndef INCLUDED_IMATHPLANE_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Plane3;
#endif
#ifndef INCLUDED_IMATHQUAT_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Quat;
#endif
#ifndef INCLUDED_IMATHSHEAR_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Shear6;
#endif
#ifndef INCLUDED_IMATHSPHERE_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Sphere3;
#endif
#ifndef INCLUDED_IMATHVEC_H
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Vec2;
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Vec3;
template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Vec4;
#endif

#ifndef INCLUDED_IMATHRANDOM_H
class IMATH_EXPORT_TYPE Rand32;
class IMATH_EXPORT_TYPE Rand48;
#endif

/// @endcond

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHFORWARD_H
