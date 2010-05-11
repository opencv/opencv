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


#ifndef INCLUDED_IMATHGL_H
#define INCLUDED_IMATHGL_H

#include <GL/gl.h>

#include "ImathVec.h"
#include "ImathMatrix.h"
#include "IexMathExc.h"
#include "ImathFun.h"

inline void glVertex    ( const Imath::V3f &v ) { glVertex3f(v.x,v.y,v.z);   }
inline void glVertex    ( const Imath::V2f &v ) { glVertex2f(v.x,v.y);       }
inline void glNormal    ( const Imath::V3f &n ) { glNormal3f(n.x,n.y,n.z);   }
inline void glColor     ( const Imath::V3f &c ) { glColor3f(c.x,c.y,c.z);    }
inline void glTranslate ( const Imath::V3f &t ) { glTranslatef(t.x,t.y,t.z); }

inline void glTexCoord( const Imath::V2f &t )
{
    glTexCoord2f(t.x,t.y);
}

inline void glDisableTexture()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE0);
}

namespace {
    
const float GL_FLOAT_MAX = 1.8e+19; // sqrt (FLT_MAX)

inline bool
badFloat (float f)
{
    return !Imath::finitef (f) || f < - GL_FLOAT_MAX || f > GL_FLOAT_MAX;
}

} // namespace
	
inline void
throwBadMatrix (const Imath::M44f& m)
{
    if (badFloat (m[0][0]) ||
	badFloat (m[0][1]) ||
	badFloat (m[0][2]) ||
	badFloat (m[0][3]) || 
	badFloat (m[1][0]) ||
	badFloat (m[1][1]) ||
	badFloat (m[1][2]) ||
	badFloat (m[1][3]) || 
	badFloat (m[2][0]) ||
	badFloat (m[2][1]) ||
	badFloat (m[2][2]) ||
	badFloat (m[2][3]) || 
	badFloat (m[3][0]) ||
	badFloat (m[3][1]) ||
	badFloat (m[3][2]) ||
	badFloat (m[3][3]))
	throw Iex::OverflowExc ("GL matrix overflow");
}

inline void 
glMultMatrix( const Imath::M44f& m ) 
{ 
    throwBadMatrix (m);
    glMultMatrixf( (GLfloat*)m[0] ); 
}

inline void 
glMultMatrix( const Imath::M44f* m ) 
{ 
    throwBadMatrix (*m);
    glMultMatrixf( (GLfloat*)(*m)[0] ); 
}

inline void 
glLoadMatrix( const Imath::M44f& m ) 
{ 
    throwBadMatrix (m);
    glLoadMatrixf( (GLfloat*)m[0] ); 
}

inline void 
glLoadMatrix( const Imath::M44f* m ) 
{ 
    throwBadMatrix (*m);
    glLoadMatrixf( (GLfloat*)(*m)[0] ); 
}


namespace Imath {

//
// Class objects that push/pop the GL state. These objects assist with
// proper cleanup of the state when exceptions are thrown.
//

class GLPushMatrix {
  public:

    GLPushMatrix ()			{ glPushMatrix(); }
    ~GLPushMatrix()			{ glPopMatrix(); }
};

class GLPushAttrib {
  public:

    GLPushAttrib (GLbitfield mask)	{ glPushAttrib (mask); }
    ~GLPushAttrib()			{ glPopAttrib(); }
};

class GLBegin {
  public:

    GLBegin (GLenum mode)		{ glBegin (mode); }
    ~GLBegin()				{ glEnd(); }
};

} // namespace Imath

#endif
