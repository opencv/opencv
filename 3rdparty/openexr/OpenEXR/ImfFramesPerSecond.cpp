//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Convenience functions related to the framesPerSecond attribute
//
//-----------------------------------------------------------------------------

#include <ImfFramesPerSecond.h>
#include "ImathFun.h"

using namespace IMATH_NAMESPACE;
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Rational
guessExactFps (double fps)
{
    return guessExactFps (Rational (fps));
}


Rational
guessExactFps (const Rational &fps)
{
    const double e = 0.002;

    if (abs (double (fps) - double (fps_23_976())) < e)
	return fps_23_976();

    if (abs (double (fps) - double (fps_29_97())) < e)
	return fps_29_97();

    if (abs (double (fps) - double (fps_47_952())) < e)
	return fps_47_952();

    if (abs (double (fps) - double (fps_59_94())) < e)
	return fps_59_94();

    return fps;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
