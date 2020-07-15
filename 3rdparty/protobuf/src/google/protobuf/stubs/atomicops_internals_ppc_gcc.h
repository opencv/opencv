// Protocol Buffers - Google's data interchange format
// Copyright 2015 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

// Author: ogabbay@advaoptical.com (Oded Gabbay)
// Cleaned up by: bsilver16384@gmail.com (Brian Silverman)
//
// This file is an internal atomic implementation, use atomicops.h instead.

#ifndef GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_PPC_GCC_H_
#define GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_PPC_GCC_H_

#define ATOMICOPS_COMPILER_BARRIER() __asm__ __volatile__("" : : : "memory")

namespace google {
namespace protobuf {
namespace internal {

inline Atomic32 NoBarrier_CompareAndSwap(volatile Atomic32 *ptr,
                                         Atomic32 old_value,
                                         Atomic32 new_value) {
  Atomic32 prev;

  __asm__ __volatile__(
      "0:                                  \n\t"
      "lwarx %[prev],0,%[ptr]              \n\t"
      "cmpw 0,%[prev],%[old_value]         \n\t"
      "bne- 1f                             \n\t"
      "stwcx. %[new_value],0,%[ptr]        \n\t"
      "bne- 0b                             \n\t"
      "1:                                  \n\t"
      : [prev] "=&r"(prev), "+m"(*ptr)
      : [ptr] "r"(ptr), [old_value] "r"(old_value), [new_value] "r"(new_value)
      : "cc", "memory");

  return prev;
}

inline Atomic32 NoBarrier_AtomicExchange(volatile Atomic32 *ptr,
                                         Atomic32 new_value) {
  Atomic32 old;

  __asm__ __volatile__(
      "0:                                  \n\t"
      "lwarx %[old],0,%[ptr]               \n\t"
      "stwcx. %[new_value],0,%[ptr]        \n\t"
      "bne- 0b                             \n\t"
      : [old] "=&r"(old), "+m"(*ptr)
      : [ptr] "r"(ptr), [new_value] "r"(new_value)
      : "cc", "memory");

  return old;
}

inline Atomic32 NoBarrier_AtomicIncrement(volatile Atomic32 *ptr,
                                          Atomic32 increment) {
  Atomic32 temp;

  __asm__ __volatile__(
      "0:                                  \n\t"
      "lwarx %[temp],0,%[ptr]              \n\t"
      "add %[temp],%[increment],%[temp]    \n\t"
      "stwcx. %[temp],0,%[ptr]             \n\t"
      "bne- 0b                             \n\t"
      : [temp] "=&r"(temp)
      : [increment] "r"(increment), [ptr] "r"(ptr)
      : "cc", "memory");

  return temp;
}

inline Atomic32 Barrier_AtomicIncrement(volatile Atomic32 *ptr,
                                        Atomic32 increment) {
  MemoryBarrierInternal();
  Atomic32 res = NoBarrier_AtomicIncrement(ptr, increment);
  MemoryBarrierInternal();
  return res;
}

inline Atomic32 Acquire_CompareAndSwap(volatile Atomic32 *ptr,
                                       Atomic32 old_value, Atomic32 new_value) {
  Atomic32 res = NoBarrier_CompareAndSwap(ptr, old_value, new_value);
  MemoryBarrierInternal();
  return res;
}

inline Atomic32 Release_CompareAndSwap(volatile Atomic32 *ptr,
                                       Atomic32 old_value, Atomic32 new_value) {
  MemoryBarrierInternal();
  Atomic32 res = NoBarrier_CompareAndSwap(ptr, old_value, new_value);
  return res;
}

inline void NoBarrier_Store(volatile Atomic32 *ptr, Atomic32 value) {
  *ptr = value;
}

inline void MemoryBarrierInternal() { __asm__ __volatile__("sync" : : : "memory"); }

inline void Acquire_Store(volatile Atomic32 *ptr, Atomic32 value) {
  *ptr = value;
  MemoryBarrierInternal();
}

inline void Release_Store(volatile Atomic32 *ptr, Atomic32 value) {
  MemoryBarrierInternal();
  *ptr = value;
}

inline Atomic32 NoBarrier_Load(volatile const Atomic32 *ptr) { return *ptr; }

inline Atomic32 Acquire_Load(volatile const Atomic32 *ptr) {
  Atomic32 value = *ptr;
  MemoryBarrierInternal();
  return value;
}

inline Atomic32 Release_Load(volatile const Atomic32 *ptr) {
  MemoryBarrierInternal();
  return *ptr;
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#undef ATOMICOPS_COMPILER_BARRIER

#endif  // GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_PPC_GCC_H_
