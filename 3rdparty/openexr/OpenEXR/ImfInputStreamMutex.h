//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFINPUTSTREAMMUTEX_H_
#define IMFINPUTSTREAMMUTEX_H_

#include "ImfForward.h"

#include "IlmThreadConfig.h"

#if ILMTHREAD_THREADING_ENABLED
#include <mutex>
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Used to wrap OPENEXR_IMF_INTERNAL_NAMESPACE::IStream as a mutex.
//
struct InputStreamMutex
#if ILMTHREAD_THREADING_ENABLED
    : public std::mutex
#endif
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream* is = nullptr;
    uint64_t currentPosition = 0;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif /* IMFINPUTSTREAMMUTEX_H_ */
