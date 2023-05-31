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

#ifndef GOOGLE_PROTOBUF_STUBS_STATUS_H_
#define GOOGLE_PROTOBUF_STUBS_STATUS_H_

#include <string>

#include <google/protobuf/stubs/stringpiece.h>

#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace util {
namespace status_internal {

// These values must match error codes defined in google/rpc/code.proto.
enum class StatusCode : int {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kUnauthenticated = 16,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
};

class PROTOBUF_EXPORT Status {
 public:
  // Creates a "successful" status.
  Status();

  // Create a status in the canonical error space with the specified
  // code, and error message.  If "code == 0", error_message is
  // ignored and a Status object identical to Status::kOk is
  // constructed.
  Status(StatusCode error_code, StringPiece error_message);
  Status(const Status&);
  Status& operator=(const Status& x);
  ~Status() {}

  // Accessor
  bool ok() const { return error_code_ == StatusCode::kOk; }
  StatusCode code() const { return error_code_; }
  StringPiece message() const {
    return error_message_;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const {
    return !operator==(x);
  }

  // Return a combination of the error code name and message.
  std::string ToString() const;

 private:
  StatusCode error_code_;
  std::string error_message_;
};

// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `OkStatus()` when constructing such an OK status.
PROTOBUF_EXPORT Status OkStatus();

// Prints a human-readable representation of 'x' to 'os'.
PROTOBUF_EXPORT std::ostream& operator<<(std::ostream& os, const Status& x);

// These convenience functions return `true` if a given status matches the
// `StatusCode` error code of its associated function.
PROTOBUF_EXPORT bool IsAborted(const Status& status);
PROTOBUF_EXPORT bool IsAlreadyExists(const Status& status);
PROTOBUF_EXPORT bool IsCancelled(const Status& status);
PROTOBUF_EXPORT bool IsDataLoss(const Status& status);
PROTOBUF_EXPORT bool IsDeadlineExceeded(const Status& status);
PROTOBUF_EXPORT bool IsFailedPrecondition(const Status& status);
PROTOBUF_EXPORT bool IsInternal(const Status& status);
PROTOBUF_EXPORT bool IsInvalidArgument(const Status& status);
PROTOBUF_EXPORT bool IsNotFound(const Status& status);
PROTOBUF_EXPORT bool IsOutOfRange(const Status& status);
PROTOBUF_EXPORT bool IsPermissionDenied(const Status& status);
PROTOBUF_EXPORT bool IsResourceExhausted(const Status& status);
PROTOBUF_EXPORT bool IsUnauthenticated(const Status& status);
PROTOBUF_EXPORT bool IsUnavailable(const Status& status);
PROTOBUF_EXPORT bool IsUnimplemented(const Status& status);
PROTOBUF_EXPORT bool IsUnknown(const Status& status);

// These convenience functions create an `Status` object with an error code as
// indicated by the associated function name, using the error message passed in
// `message`.
//
// These functions are intentionally named `*Error` rather than `*Status` to
// match the names from Abseil:
// https://github.com/abseil/abseil-cpp/blob/2e9532cc6c701a8323d0cffb468999ab804095ab/absl/status/status.h#L716
PROTOBUF_EXPORT Status AbortedError(StringPiece message);
PROTOBUF_EXPORT Status AlreadyExistsError(StringPiece message);
PROTOBUF_EXPORT Status CancelledError(StringPiece message);
PROTOBUF_EXPORT Status DataLossError(StringPiece message);
PROTOBUF_EXPORT Status DeadlineExceededError(StringPiece message);
PROTOBUF_EXPORT Status FailedPreconditionError(StringPiece message);
PROTOBUF_EXPORT Status InternalError(StringPiece message);
PROTOBUF_EXPORT Status InvalidArgumentError(StringPiece message);
PROTOBUF_EXPORT Status NotFoundError(StringPiece message);
PROTOBUF_EXPORT Status OutOfRangeError(StringPiece message);
PROTOBUF_EXPORT Status PermissionDeniedError(StringPiece message);
PROTOBUF_EXPORT Status ResourceExhaustedError(StringPiece message);
PROTOBUF_EXPORT Status UnauthenticatedError(StringPiece message);
PROTOBUF_EXPORT Status UnavailableError(StringPiece message);
PROTOBUF_EXPORT Status UnimplementedError(StringPiece message);
PROTOBUF_EXPORT Status UnknownError(StringPiece message);

}  // namespace status_internal

using ::google::protobuf::util::status_internal::Status;
using ::google::protobuf::util::status_internal::StatusCode;

using ::google::protobuf::util::status_internal::IsAborted;
using ::google::protobuf::util::status_internal::IsAlreadyExists;
using ::google::protobuf::util::status_internal::IsCancelled;
using ::google::protobuf::util::status_internal::IsDataLoss;
using ::google::protobuf::util::status_internal::IsDeadlineExceeded;
using ::google::protobuf::util::status_internal::IsFailedPrecondition;
using ::google::protobuf::util::status_internal::IsInternal;
using ::google::protobuf::util::status_internal::IsInvalidArgument;
using ::google::protobuf::util::status_internal::IsNotFound;
using ::google::protobuf::util::status_internal::IsOutOfRange;
using ::google::protobuf::util::status_internal::IsPermissionDenied;
using ::google::protobuf::util::status_internal::IsResourceExhausted;
using ::google::protobuf::util::status_internal::IsUnauthenticated;
using ::google::protobuf::util::status_internal::IsUnavailable;
using ::google::protobuf::util::status_internal::IsUnimplemented;
using ::google::protobuf::util::status_internal::IsUnknown;

using ::google::protobuf::util::status_internal::AbortedError;
using ::google::protobuf::util::status_internal::AlreadyExistsError;
using ::google::protobuf::util::status_internal::CancelledError;
using ::google::protobuf::util::status_internal::DataLossError;
using ::google::protobuf::util::status_internal::DeadlineExceededError;
using ::google::protobuf::util::status_internal::FailedPreconditionError;
using ::google::protobuf::util::status_internal::InternalError;
using ::google::protobuf::util::status_internal::InvalidArgumentError;
using ::google::protobuf::util::status_internal::NotFoundError;
using ::google::protobuf::util::status_internal::OkStatus;
using ::google::protobuf::util::status_internal::OutOfRangeError;
using ::google::protobuf::util::status_internal::PermissionDeniedError;
using ::google::protobuf::util::status_internal::ResourceExhaustedError;
using ::google::protobuf::util::status_internal::UnauthenticatedError;
using ::google::protobuf::util::status_internal::UnavailableError;
using ::google::protobuf::util::status_internal::UnimplementedError;
using ::google::protobuf::util::status_internal::UnknownError;

}  // namespace util
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_STUBS_STATUS_H_
