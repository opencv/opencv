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

// StatusOr<T> is the union of a Status object and a T
// object. StatusOr models the concept of an object that is either a
// usable value, or an error Status explaining why such a value is
// not present. To this end, StatusOr<T> does not allow its Status
// value to be OkStatus(). Further, StatusOr<T*> does not allow the
// contained pointer to be nullptr.
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example client usage for a StatusOr<T>, where T is not a pointer:
//
//  StatusOr<float> result = DoBigCalculationThatCouldFail();
//  if (result.ok()) {
//    float answer = result.value();
//    printf("Big calculation yielded: %f", answer);
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<T*>:
//
//  StatusOr<Foo*> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo(result.value());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example factory implementation returning StatusOr<T*>:
//
//  StatusOr<Foo*> FooFactory::MakeNewFoo(int arg) {
//    if (arg <= 0) {
//      return InvalidArgumentError("Arg must be positive");
//    } else {
//      return new Foo(arg);
//    }
//  }
//

#ifndef GOOGLE_PROTOBUF_STUBS_STATUSOR_H_
#define GOOGLE_PROTOBUF_STUBS_STATUSOR_H_

#include <new>
#include <string>
#include <utility>

#include <google/protobuf/stubs/status.h>

#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace util {
namespace statusor_internal {

template<typename T>
class StatusOr {
  template<typename U> friend class StatusOr;

 public:
  using value_type = T;

  // Construct a new StatusOr with Status::UNKNOWN status.
  // Construct a new StatusOr with UnknownError() status.
  explicit StatusOr();

  // Construct a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to value() will CHECK-fail.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
  // REQUIRES: status != OkStatus(). This requirement is DCHECKed.
  // In optimized builds, passing OkStatus() here will have the effect
  // of passing PosixErrorSpace::EINVAL as a fallback.
  StatusOr(const Status& status);  // NOLINT

  // Construct a new StatusOr with the given value. If T is a plain pointer,
  // value must not be nullptr. After calling this constructor, calls to
  // value() will succeed, and calls to status() will return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when when the return type is StatusOr<T>.
  //
  // REQUIRES: if T is a plain pointer, value != nullptr. This requirement is
  // DCHECKed. In optimized builds, passing a null pointer here will have
  // the effect of passing PosixErrorSpace::EINVAL as a fallback.
  StatusOr(const T& value);  // NOLINT

  // Copy constructor.
  StatusOr(const StatusOr& other);

  // Conversion copy constructor, T must be copy constructible from U
  template<typename U>
  StatusOr(const StatusOr<U>& other);

  // Assignment operator.
  StatusOr& operator=(const StatusOr& other);

  // Conversion assignment operator, T must be assignable from U
  template<typename U>
  StatusOr& operator=(const StatusOr<U>& other);

  // Returns a reference to our status. If this contains a T, then
  // returns OkStatus().
  const Status& status() const;

  // Returns this->status().ok()
  bool ok() const;

  // Returns a reference to our current value, or CHECK-fails if !this->ok().
  const T& value () const;

 private:
  Status status_;
  T value_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

class PROTOBUF_EXPORT StatusOrHelper {
 public:
  // Move type-agnostic error handling to the .cc.
  static void Crash(const util::Status& status);

  // Customized behavior for StatusOr<T> vs. StatusOr<T*>
  template<typename T>
  struct Specialize;
};

template<typename T>
struct StatusOrHelper::Specialize {
  // For non-pointer T, a reference can never be nullptr.
  static inline bool IsValueNull(const T& /*t*/) { return false; }
};

template<typename T>
struct StatusOrHelper::Specialize<T*> {
  static inline bool IsValueNull(const T* t) { return t == nullptr; }
};

template <typename T>
inline StatusOr<T>::StatusOr() : status_(util::UnknownError("")) {}

template<typename T>
inline StatusOr<T>::StatusOr(const Status& status) {
  if (status.ok()) {
    status_ = util::InternalError("OkStatus() is not a valid argument.");
  } else {
    status_ = status;
  }
}

template<typename T>
inline StatusOr<T>::StatusOr(const T& value) {
  if (StatusOrHelper::Specialize<T>::IsValueNull(value)) {
    status_ = util::InternalError("nullptr is not a valid argument.");
  } else {
    status_ = util::OkStatus();
    value_ = value;
  }
}

template<typename T>
inline StatusOr<T>::StatusOr(const StatusOr<T>& other)
    : status_(other.status_), value_(other.value_) {
}

template<typename T>
inline StatusOr<T>& StatusOr<T>::operator=(const StatusOr<T>& other) {
  status_ = other.status_;
  value_ = other.value_;
  return *this;
}

template<typename T>
template<typename U>
inline StatusOr<T>::StatusOr(const StatusOr<U>& other)
    : status_(other.status_), value_(other.status_.ok() ? other.value_ : T()) {
}

template<typename T>
template<typename U>
inline StatusOr<T>& StatusOr<T>::operator=(const StatusOr<U>& other) {
  status_ = other.status_;
  if (status_.ok()) value_ = other.value_;
  return *this;
}

template<typename T>
inline const Status& StatusOr<T>::status() const {
  return status_;
}

template<typename T>
inline bool StatusOr<T>::ok() const {
  return status().ok();
}

template<typename T>
inline const T& StatusOr<T>::value() const {
  if (!status_.ok()) {
    StatusOrHelper::Crash(status_);
  }
  return value_;
}

}  // namespace statusor_internal

using ::google::protobuf::util::statusor_internal::StatusOr;

}  // namespace util
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_STUBS_STATUSOR_H_
