//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Convenience functions that call GLU with Imath types
//

#ifndef INCLUDED_IMATHGLU_H
#define INCLUDED_IMATHGLU_H

#include <GL/gl.h>
#include <GL/glu.h>

#include "ImathVec.h"

/// Call gluLookAt with the given position, interest, and up-vector.
inline void
gluLookAt (const IMATH_INTERNAL_NAMESPACE::V3f& pos,
           const IMATH_INTERNAL_NAMESPACE::V3f& interest,
           const IMATH_INTERNAL_NAMESPACE::V3f& up)
{
    gluLookAt (pos.x, pos.y, pos.z, interest.x, interest.y, interest.z, up.x, up.y, up.z);
}

#endif
