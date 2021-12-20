//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILMTHREADFORWARD_H
#define INCLUDED_ILMTHREADFORWARD_H

#include "IlmThreadConfig.h"
#include "IlmThreadNamespace.h"

#if ILMTHREAD_THREADING_ENABLED
namespace std { class mutex; }
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER

class Thread;
#if ILMTHREAD_THREADING_ENABLED
using Mutex = std::mutex;
#else
class Mutex;
#endif
class Lock;
class ThreadPool;
class Task;
class TaskGroup;
class Semaphore;

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILMTHREADFORWARD_H
