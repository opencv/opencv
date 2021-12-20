//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FRAMES_PER_SECOND_H
#define INCLUDED_IMF_FRAMES_PER_SECOND_H

//-----------------------------------------------------------------------------
//
//	Convenience functions related to the framesPerSecond attribute
//
//	Functions that return the exact values for commonly used frame rates:
//
//	    name		frames per second
//
//	    fps_23_976()	23.976023...
//	    fps_24()		24.0		35mm film frames
//	    fps_25()		25.0		PAL video frames
//	    fps_29_97()		29.970029...	NTSC video frames
//	    fps_30()		30.0		60Hz HDTV frames
//	    fps_47_952()	47.952047...
//	    fps_48()		48.0
//	    fps_50()		50.0		PAL video fields
//	    fps_59_94()		59.940059...	NTSC video fields
//	    fps_60()		60.0		60Hz HDTV fields
//
//	Functions that try to convert inexact frame rates into exact ones:
//
//	    Given a frame rate, fps, that is close to one of the pre-defined
//	    frame rates fps_23_976(), fps_29_97(), fps_47_952() or fps_59_94(),
//	    guessExactFps(fps) returns the corresponding pre-defined frame
//	    rate.  If fps is not close to one of the pre-defined frame rates,
//	    then guessExactFps(fps) returns Rational(fps).
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfRational.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


inline Rational	fps_23_976 ()	{return Rational (24000, 1001);}
inline Rational	fps_24 ()	{return Rational (24, 1);}
inline Rational	fps_25 ()	{return Rational (25, 1);}
inline Rational	fps_29_97 ()	{return Rational (30000, 1001);}
inline Rational	fps_30 ()	{return Rational (30, 1);}
inline Rational	fps_47_952 ()	{return Rational (48000, 1001);}
inline Rational	fps_48 ()	{return Rational (48, 1);}
inline Rational	fps_50 ()	{return Rational (50, 1);}
inline Rational	fps_59_94 ()	{return Rational (60000, 1001);}
inline Rational	fps_60 ()	{return Rational (60, 1);}

IMF_EXPORT Rational	guessExactFps (double fps);
IMF_EXPORT Rational	guessExactFps (const Rational &fps);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
