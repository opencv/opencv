//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------
//
//	A function to control which IEEE floating
//	point exceptions will be translated into
//	C++ MathExc exceptions.
//
//-----------------------------------------------------

#include "IexMathFloatExc.h"
#include "IexMacros.h"
#include "IexMathExc.h"
#include "IexMathFpu.h"

#if 0
    #include <iostream>
    #define debug(x) (std::cout << x << std::flush)
#else
    #define debug(x)
#endif

IEX_INTERNAL_NAMESPACE_SOURCE_ENTER


namespace {

void
fpeHandler (int type, const char explanation[])
{
    switch (type)
    {
      case IEEE_OVERFLOW:
	throw OverflowExc (explanation);

      case IEEE_UNDERFLOW:
	throw UnderflowExc (explanation);

      case IEEE_DIVZERO:
	throw DivzeroExc (explanation);

      case IEEE_INEXACT:
	throw InexactExc (explanation);

      case IEEE_INVALID:
	throw InvalidFpOpExc (explanation);
    }

    throw MathExc (explanation);
}

} // namespace


void
mathExcOn (int when)
{
    debug ("mathExcOn (when = 0x" << std::hex << when << ")\n");

    setFpExceptions (when);
    setFpExceptionHandler (fpeHandler);
}


int
getMathExcOn ()
{
    int when = fpExceptions();

    debug ("getMathExcOn () == 0x" << std::hex << when << ")\n");

    return when;
}

MathExcOn::MathExcOn (int when)
: _changed (false)
{
    _saved = getMathExcOn();

    if (_saved != when)
    {
        _changed = true;
        mathExcOn (when);
    }
}

MathExcOn::~MathExcOn ()
{
    if (_changed)
        mathExcOn (_saved);
}

void
MathExcOn::handleOutstandingExceptions()
{
    handleExceptionsSetInRegisters();
}


IEX_INTERNAL_NAMESPACE_SOURCE_EXIT
