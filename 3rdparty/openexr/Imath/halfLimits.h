//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_HALF_LIMITS_H
#define INCLUDED_HALF_LIMITS_H

// Warn if half.h hasn't being included
#ifndef IMATH_HALF_H_
//
// This file is now deprecated. It previously included the
// specialization of std::numeric_limits<half>, but those now appear
// directly in half.h, because they should be regarded as inseperable
// from the half class.
//

#ifdef __GNUC__
#warning "ImathLimits is deprecated; use #include <half.h>"
#else
#pragma message("ImathLimits is deprecated; use #include <half.h>")
#endif

#include "half.h"
#endif
#endif
