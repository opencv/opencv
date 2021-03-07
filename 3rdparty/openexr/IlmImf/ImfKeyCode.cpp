///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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


//-----------------------------------------------------------------------------
//
//	class KeyCode
//
//-----------------------------------------------------------------------------

#include <ImfKeyCode.h>
#include "Iex.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

   
KeyCode::KeyCode (int filmMfcCode,
		  int filmType,
		  int prefix,
		  int count,
		  int perfOffset,
		  int perfsPerFrame,
		  int perfsPerCount)
{
    setFilmMfcCode (filmMfcCode);
    setFilmType (filmType);
    setPrefix (prefix);
    setCount (count);
    setPerfOffset (perfOffset);
    setPerfsPerFrame (perfsPerFrame);
    setPerfsPerCount (perfsPerCount);
}


KeyCode::KeyCode (const KeyCode &other)
{
    _filmMfcCode = other._filmMfcCode;
    _filmType = other._filmType;
    _prefix = other._prefix;
    _count = other._count;
    _perfOffset = other._perfOffset;
    _perfsPerFrame = other._perfsPerFrame;
    _perfsPerCount = other._perfsPerCount;
}


KeyCode &
KeyCode::operator = (const KeyCode &other)
{
    _filmMfcCode = other._filmMfcCode;
    _filmType = other._filmType;
    _prefix = other._prefix;
    _count = other._count;
    _perfOffset = other._perfOffset;
    _perfsPerFrame = other._perfsPerFrame;
    _perfsPerCount = other._perfsPerCount;

    return *this;
}


int		
KeyCode::filmMfcCode () const
{
    return _filmMfcCode;
}


void	
KeyCode::setFilmMfcCode (int filmMfcCode)
{
    if (filmMfcCode < 0 || filmMfcCode > 99)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code film manufacturer code "
			   "(must be between 0 and 99).");

    _filmMfcCode = filmMfcCode;
}

int		
KeyCode::filmType () const
{
    return _filmType;
}


void	
KeyCode::setFilmType (int filmType)
{
    if (filmType < 0 || filmType > 99)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code film type "
			   "(must be between 0 and 99).");

    _filmType = filmType;
}

int		
KeyCode::prefix () const
{
    return _prefix;
}


void	
KeyCode::setPrefix (int prefix)
{
    if (prefix < 0 || prefix > 999999)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code prefix "
			   "(must be between 0 and 999999).");

    _prefix = prefix;
}


int		
KeyCode::count () const
{
    return _count;
}


void	
KeyCode::setCount (int count)
{
    if (count < 0 || count > 9999)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code count "
			   "(must be between 0 and 9999).");

    _count = count;
}


int		
KeyCode::perfOffset () const
{
    return _perfOffset;
}


void	
KeyCode::setPerfOffset (int perfOffset)
{
    if (perfOffset < 0 || perfOffset > 119)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code perforation offset "
			   "(must be between 0 and 119).");

    _perfOffset = perfOffset;
}


int	
KeyCode::perfsPerFrame () const
{
    return _perfsPerFrame;
}


void
KeyCode::setPerfsPerFrame (int perfsPerFrame)
{
    if (perfsPerFrame < 1 || perfsPerFrame > 15)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code number of perforations per frame "
			   "(must be between 1 and 15).");

    _perfsPerFrame = perfsPerFrame;
}


int	
KeyCode::perfsPerCount () const
{
    return _perfsPerCount;
}


void
KeyCode::setPerfsPerCount (int perfsPerCount)
{
    if (perfsPerCount < 20 || perfsPerCount > 120)
	throw IEX_NAMESPACE::ArgExc ("Invalid key code number of perforations per count "
			   "(must be between 20 and 120).");

    _perfsPerCount = perfsPerCount;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
