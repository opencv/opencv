// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
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

#ifndef GOOGLE_PROTOBUF_EXPLICITLY_CONSTRUCTED_H__
#define GOOGLE_PROTOBUF_EXPLICITLY_CONSTRUCTED_H__

#include <stdint.h>

#include <utility>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>

// clang-format off
#include <google/protobuf/port_def.inc>
// clang-format on

namespace google {
namespace protobuf {
namespace internal {

// Wraps a variable whose constructor and destructor are explicitly
// called. It is particularly useful for a global variable, without its
// constructor and destructor run on start and end of the program lifetime.
// This circumvents the initial construction order fiasco, while keeping
// the address of the empty string a compile time constant.
//
// Pay special attention to the initialization state of the object.
// 1. The object is "uninitialized" to begin with.
// 2. Call Construct() or DefaultConstruct() only if the object is
//    uninitialized. After the call, the object becomes "initialized".
// 3. Call get() and get_mutable() only if the object is initialized.
// 4. Call Destruct() only if the object is initialized.
//    After the call, the object becomes uninitialized.
template <typename T>
class ExplicitlyConstructed {
 public:
  void DefaultConstruct() { new (&union_) T(); }

  template <typename... Args>
  void Construct(Args&&... args) {
    new (&union_) T(std::forward<Args>(args)...);
  }

  void Destruct() { get_mutable()->~T(); }

  constexpr const T& get() const { return reinterpret_cast<const T&>(union_); }
  T* get_mutable() { return reinterpret_cast<T*>(&union_); }

 private:
  union AlignedUnion {
    alignas(T) char space[sizeof(T)];
    int64_t align_to_int64;
    void* align_to_ptr;
  } union_;
};

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_EXPLICITLY_CONSTRUCTED_H__
