//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IEXNAMESPACE_H
#define INCLUDED_IEXNAMESPACE_H

//
// The purpose of this file is to make it possible to specify an
// IEX_INTERNAL_NAMESPACE as a preprocessor definition and have all of the
// Iex symbols defined within that namespace rather than the standard
// Iex namespace.  Those symbols are made available to client code through
// the IEX_NAMESPACE in addition to the IEX_INTERNAL_NAMESPACE.
//
// To ensure source code compatibility, the IEX_NAMESPACE defaults to Iex
// and then "using namespace IEX_INTERNAL_NAMESPACE;" brings all of the
// declarations from the IEX_INTERNAL_NAMESPACE into the IEX_NAMESPACE.  This
// means that client code can continue to use syntax like Iex::BaseExc, but
// at link time it will resolve to a mangled symbol based on the
// IEX_INTERNAL_NAMESPACE.
//
// As an example, if one needed to build against a newer version of Iex and
// have it run alongside an older version in the same application, it is now
// possible to use an internal namespace to prevent collisions between the
// older versions of Iex symbols and the newer ones.  To do this, the
// following could be defined at build time:
//
// IEX_INTERNAL_NAMESPACE = Iex_v2
//
// This means that declarations inside Iex headers look like this (after the
// preprocessor has done its work):
//
// namespace Iex_v2 {
//     ...
//     class declarations
//     ...
// }
//
// namespace Iex {
//     using namespace Iex_v2;
// }
//

//
// Open Source version of this file pulls in the IexConfig.h file
// for the configure time options.
//
#include "IexConfig.h"

#ifndef IEX_NAMESPACE
#define IEX_NAMESPACE Iex
#endif

#ifndef IEX_INTERNAL_NAMESPACE
#define IEX_INTERNAL_NAMESPACE IEX_NAMESPACE
#endif

//
// We need to be sure that we import the internal namespace into the public one.
// To do this, we use the small bit of code below which initially defines
// IEX_INTERNAL_NAMESPACE (so it can be referenced) and then defines
// IEX_NAMESPACE and pulls the internal symbols into the public namespace.
//

namespace IEX_INTERNAL_NAMESPACE {}
namespace IEX_NAMESPACE {
    using namespace IEX_INTERNAL_NAMESPACE;
}

//
// There are identical pairs of HEADER/SOURCE ENTER/EXIT macros so that
// future extension to the namespace mechanism is possible without changing
// project source code.
//

#define IEX_INTERNAL_NAMESPACE_HEADER_ENTER namespace IEX_INTERNAL_NAMESPACE {
#define IEX_INTERNAL_NAMESPACE_HEADER_EXIT }

#define IEX_INTERNAL_NAMESPACE_SOURCE_ENTER namespace IEX_INTERNAL_NAMESPACE {
#define IEX_INTERNAL_NAMESPACE_SOURCE_EXIT }

#endif // INCLUDED_IEXNAMESPACE_H
