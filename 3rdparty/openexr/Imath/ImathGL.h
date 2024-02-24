//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Convenience functions that call GL with Imath types
//

#ifndef INCLUDED_IMATHGL_H
#define INCLUDED_IMATHGL_H

#include <GL/gl.h>

#include "ImathFun.h"
#include "ImathMatrix.h"
#include "ImathNamespace.h"
#include "ImathVec.h"

/// Call glVertex3f
inline void
glVertex (const IMATH_INTERNAL_NAMESPACE::V3f& v)
{
    glVertex3f (v.x, v.y, v.z);
}

/// Call glVertex2f
inline void
glVertex (const IMATH_INTERNAL_NAMESPACE::V2f& v)
{
    glVertex2f (v.x, v.y);
}

/// Call glNormal3f
inline void
glNormal (const IMATH_INTERNAL_NAMESPACE::V3f& n)
{
    glNormal3f (n.x, n.y, n.z);
}

/// Call glColor3f
inline void
glColor (const IMATH_INTERNAL_NAMESPACE::V3f& c)
{
    glColor3f (c.x, c.y, c.z);
}

/// Call glTranslatef
inline void
glTranslate (const IMATH_INTERNAL_NAMESPACE::V3f& t)
{
    glTranslatef (t.x, t.y, t.z);
}

/// Call glTexCoord2f
inline void
glTexCoord (const IMATH_INTERNAL_NAMESPACE::V2f& t)
{
    glTexCoord2f (t.x, t.y);
}

/// Disable GL textures
inline void
glDisableTexture()
{
    glActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, 0);
    glDisable (GL_TEXTURE_2D);

    glActiveTexture (GL_TEXTURE0);
}

namespace
{

const float GL_FLOAT_MAX = 1.8e+19; // sqrt (FLT_MAX)

inline bool
badFloat (float f)
{
    return !IMATH_INTERNAL_NAMESPACE::finitef (f) || f < -GL_FLOAT_MAX || f > GL_FLOAT_MAX;
}

} // namespace

/// Throw an exception if m is not a valid matrix for GL
inline void
throwBadMatrix (const IMATH_INTERNAL_NAMESPACE::M44f& m)
{
    if (badFloat (m[0][0]) || badFloat (m[0][1]) || badFloat (m[0][2]) || badFloat (m[0][3]) ||
        badFloat (m[1][0]) || badFloat (m[1][1]) || badFloat (m[1][2]) || badFloat (m[1][3]) ||
        badFloat (m[2][0]) || badFloat (m[2][1]) || badFloat (m[2][2]) || badFloat (m[2][3]) ||
        badFloat (m[3][0]) || badFloat (m[3][1]) || badFloat (m[3][2]) || badFloat (m[3][3]))
        throw std::invalid_argument ("GL matrix overflow");
}

/// Call glMultmatrixf. Throw an exception if m is not a valid matrix for GL.
inline void
glMultMatrix (const IMATH_INTERNAL_NAMESPACE::M44f& m)
{
    throwBadMatrix (m);
    glMultMatrixf ((GLfloat*) m[0]);
}

/// Call glMultmatrixf. Throw an exception if m is not a valid matrix for GL.
inline void
glMultMatrix (const IMATH_INTERNAL_NAMESPACE::M44f* m)
{
    throwBadMatrix (*m);
    glMultMatrixf ((GLfloat*) (*m)[0]);
}

/// Call glLoadmatrixf. Throw an exception if m is not a valid matrix for GL.
inline void
glLoadMatrix (const IMATH_INTERNAL_NAMESPACE::M44f& m)
{
    throwBadMatrix (m);
    glLoadMatrixf ((GLfloat*) m[0]);
}

/// Call glLoadmatrixf. Throw an exception if m is not a valid matrix for GL.
inline void
glLoadMatrix (const IMATH_INTERNAL_NAMESPACE::M44f* m)
{
    throwBadMatrix (*m);
    glLoadMatrixf ((GLfloat*) (*m)[0]);
}

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// A class object that pushes/pops the GL matrix. This object assists with
/// proper cleanup of the state when exceptions are thrown.
///

class GLPushMatrix
{
  public:
    GLPushMatrix() { glPushMatrix(); }
    ~GLPushMatrix() { glPopMatrix(); }
};

///
/// A class object that pushes/pops the current GL attribute state. This object assists with
/// proper cleanup of the state when exceptions are thrown.
///

class GLPushAttrib
{
  public:
    /// call glPushAttrib()
    GLPushAttrib (GLbitfield mask) { glPushAttrib (mask); }

    /// call glPopAttrib()
    ~GLPushAttrib() { glPopAttrib(); }
};

///
/// A class object that wraps glBegin/glEnd. The constructor calls
/// glBegin(). The destructor calls glEnd().
///

class GLBegin
{
  public:

    /// Call glBegin()
    GLBegin (GLenum mode) { glBegin (mode); }

    /// Call glEnd()
    ~GLBegin() { glEnd(); }
};

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
