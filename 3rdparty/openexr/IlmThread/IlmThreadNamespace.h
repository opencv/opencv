//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILMTHREADNAMESPACE_H
#define INCLUDED_ILMTHREADNAMESPACE_H

//
// The purpose of this file is to make it possible to specify an
// ILMTHREAD_INTERNAL_NAMESPACE as a preprocessor definition and have all of
// the IlmThread symbols defined within that namespace rather than the
// standard IlmThread namespace.  Those symbols are made available to client
// code through the ILMTHREAD_NAMESPACE in addition to the
// ILMTHREAD_INTERNAL_NAMESPACE.
//
// To ensure source code compatibility, the ILMTHREAD_NAMESPACE defaults to
// IlmThread and then "using namespace ILMTHREAD_INTERNAL_NAMESPACE;" brings
// all of the declarations from the ILMTHREAD_INTERNAL_NAMESPACE into the
// ILMTHREAD_NAMESPACE.  This means that client code can continue to use
// syntax like IlmThread::Thread, but at link time it will resolve to a
// mangled symbol based on the ILMTHREAD_INTERNAL_NAMESPACE.
//
// As an example, if one needed to build against a newer version of IlmThread
// and have it run alongside an older version in the same application, it is
// now possible to use an internal namespace to prevent collisions between
// the older versions of IlmThread symbols and the newer ones.  To do this,
// the following could be defined at build time:
//
// ILMTHREAD_INTERNAL_NAMESPACE = IlmThread_v2
//
// This means that declarations inside IlmThread headers look like this
// (after the preprocessor has done its work):
//
// namespace IlmThread_v2 {
//     ...
//     class declarations
//     ...
// }
//
// namespace IlmThread {
//     using namespace IlmThread_v2;
// }
//

//
// Open Source version of this file pulls in the IlmThreadConfig.h file
// for the configure time options.
//
#include "IlmThreadConfig.h"

#ifndef ILMTHREAD_NAMESPACE
#define ILMTHREAD_NAMESPACE IlmThread
#endif

#ifndef ILMTHREAD_INTERNAL_NAMESPACE
#define ILMTHREAD_INTERNAL_NAMESPACE ILMTHREAD_NAMESPACE
#endif

//
// We need to be sure that we import the internal namespace into the public one.
// To do this, we use the small bit of code below which initially defines
// ILMTHREAD_INTERNAL_NAMESPACE (so it can be referenced) and then defines
// ILMTHREAD_NAMESPACE and pulls the internal symbols into the public
// namespace.
//

namespace ILMTHREAD_INTERNAL_NAMESPACE {}
namespace ILMTHREAD_NAMESPACE {
     using namespace ILMTHREAD_INTERNAL_NAMESPACE;
}

//
// There are identical pairs of HEADER/SOURCE ENTER/EXIT macros so that
// future extension to the namespace mechanism is possible without changing
// project source code.
//

#define ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER namespace ILMTHREAD_INTERNAL_NAMESPACE {
#define ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT }

#define ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER namespace ILMTHREAD_INTERNAL_NAMESPACE {
#define ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT }

#endif // INCLUDED_ILMTHREADNAMESPACE_H
