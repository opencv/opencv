//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMFNAMESPACE_H
#define INCLUDED_IMFNAMESPACE_H

//
// The purpose of this file is to have all of the Imath symbols defined within 
// the OPENEXR_IMF_INTERNAL_NAMESPACE namespace rather than the standard Imath
// namespace. Those symbols are made available to client code through the 
// OPENEXR_IMF_NAMESPACE in addition to the OPENEXR_IMF_INTERNAL_NAMESPACE.
//
// To ensure source code compatibility, the OPENEXR_IMF_NAMESPACE defaults to
// Imath and then "using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;" brings all
// of the declarations from the OPENEXR_IMF_INTERNAL_NAMESPACE into the
// OPENEXR_IMF_NAMESPACE.
// This means that client code can continue to use syntax like
// Imf::Header, but at link time it will resolve to a
// mangled symbol based on the OPENEXR_IMF_INTERNAL_NAMESPACE.
//
// As an example, if one needed to build against a newer version of Imath and
// have it run alongside an older version in the same application, it is now
// possible to use an internal namespace to prevent collisions between the
// older versions of Imath symbols and the newer ones.  To do this, the
// following could be defined at build time:
//
// OPENEXR_IMF_INTERNAL_NAMESPACE = Imf_v2
//
// This means that declarations inside Imath headers look like this (after
// the preprocessor has done its work):
//
// namespace Imf_v2 {
//     ...
//     class declarations
//     ...
// }
//
// namespace Imf {
//     using namespace IMF_NAMESPACE_v2;
// }
//

//
// Open Source version of this file pulls in the OpenEXRConfig.h file
// for the configure time options.
//
#include "OpenEXRConfig.h"


#ifndef OPENEXR_IMF_NAMESPACE
#define OPENEXR_IMF_NAMESPACE Imf
#endif

#ifndef OPENEXR_IMF_INTERNAL_NAMESPACE
#define OPENEXR_IMF_INTERNAL_NAMESPACE OPENEXR_IMF_NAMESPACE
#endif

//
// We need to be sure that we import the internal namespace into the public one.
// To do this, we use the small bit of code below which initially defines
// OPENEXR_IMF_INTERNAL_NAMESPACE (so it can be referenced) and then defines
// OPENEXR_IMF_NAMESPACE and pulls the internal symbols into the public
// namespace.
//

namespace OPENEXR_IMF_INTERNAL_NAMESPACE {}
namespace OPENEXR_IMF_NAMESPACE {
     using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;
}

//
// There are identical pairs of HEADER/SOURCE ENTER/EXIT macros so that
// future extension to the namespace mechanism is possible without changing
// project source code.
//

#define OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER namespace OPENEXR_IMF_INTERNAL_NAMESPACE {
#define OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT }

#define OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER namespace OPENEXR_IMF_INTERNAL_NAMESPACE {
#define OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT }


#endif /* INCLUDED_IMFNAMESPACE_H */
