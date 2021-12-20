//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IEX_H
#define INCLUDED_IEX_H


//--------------------------------
//
//	Exception handling
//
//--------------------------------


#include "IexMacros.h"
#include "IexBaseExc.h"
#include "IexMathExc.h"
#include "IexThrowErrnoExc.h"

// Note that we do not include file IexErrnoExc.h here.  That file
// defines over 150 classes and significantly slows down compilation.
// If you throw ErrnoExc exceptions using the throwErrnoExc() function,
// you don't need IexErrnoExc.h.  You have to include IexErrnoExc.h
// only if you want to catch specific subclasses of ErrnoExc.


#endif
