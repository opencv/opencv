//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFOUTPUTSTREAMMUTEX_H_
#define IMFOUTPUTSTREAMMUTEX_H_

#include "ImfForward.h"

#include "IlmThreadConfig.h"

#if ILMTHREAD_THREADING_ENABLED
#include <mutex>
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Used to wrap OPENEXR_IMF_INTERNAL_NAMESPACE::OStream as a Mutex.
//
struct OutputStreamMutex
#if ILMTHREAD_THREADING_ENABLED
    : public std::mutex
#endif
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::OStream* os = nullptr;
    uint64_t currentPosition = 0;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif /* IMFOUTPUTSTREAMMUTEX_H_ */
