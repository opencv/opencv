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
#include <google/protobuf/stubs/status.h>

#include <ostream>
#include <stdio.h>
#include <string>
#include <utility>

namespace google {
namespace protobuf {
namespace util {
namespace status_internal {
namespace {

inline std::string StatusCodeToString(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kCancelled:
      return "CANCELLED";
    case StatusCode::kUnknown:
      return "UNKNOWN";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case StatusCode::kDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kAlreadyExists:
      return "ALREADY_EXISTS";
    case StatusCode::kPermissionDenied:
      return "PERMISSION_DENIED";
    case StatusCode::kUnauthenticated:
      return "UNAUTHENTICATED";
    case StatusCode::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::kFailedPrecondition:
      return "FAILED_PRECONDITION";
    case StatusCode::kAborted:
      return "ABORTED";
    case StatusCode::kOutOfRange:
      return "OUT_OF_RANGE";
    case StatusCode::kUnimplemented:
      return "UNIMPLEMENTED";
    case StatusCode::kInternal:
      return "INTERNAL";
    case StatusCode::kUnavailable:
      return "UNAVAILABLE";
    case StatusCode::kDataLoss:
      return "DATA_LOSS";
  }

  // No default clause, clang will abort if a code is missing from
  // above switch.
  return "UNKNOWN";
}

}  // namespace

Status::Status() : error_code_(StatusCode::kOk) {}

Status::Status(StatusCode error_code, StringPiece error_message)
    : error_code_(error_code) {
  if (error_code != StatusCode::kOk) {
    error_message_ = error_message.ToString();
  }
}

Status::Status(const Status& other)
    : error_code_(other.error_code_), error_message_(other.error_message_) {
}

Status& Status::operator=(const Status& other) {
  error_code_ = other.error_code_;
  error_message_ = other.error_message_;
  return *this;
}

bool Status::operator==(const Status& x) const {
  return error_code_ == x.error_code_ &&
      error_message_ == x.error_message_;
}

std::string Status::ToString() const {
  if (error_code_ == StatusCode::kOk) {
    return "OK";
  } else {
    if (error_message_.empty()) {
      return StatusCodeToString(error_code_);
    } else {
      return StatusCodeToString(error_code_) + ":" + error_message_;
    }
  }
}

Status OkStatus() { return Status(); }

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

bool IsAborted(const Status& status) {
  return status.code() == StatusCode::kAborted;
}

bool IsAlreadyExists(const Status& status) {
  return status.code() == StatusCode::kAlreadyExists;
}

bool IsCancelled(const Status& status) {
  return status.code() == StatusCode::kCancelled;
}

bool IsDataLoss(const Status& status) {
  return status.code() == StatusCode::kDataLoss;
}

bool IsDeadlineExceeded(const Status& status) {
  return status.code() == StatusCode::kDeadlineExceeded;
}

bool IsFailedPrecondition(const Status& status) {
  return status.code() == StatusCode::kFailedPrecondition;
}

bool IsInternal(const Status& status) {
  return status.code() == StatusCode::kInternal;
}

bool IsInvalidArgument(const Status& status) {
  return status.code() == StatusCode::kInvalidArgument;
}

bool IsNotFound(const Status& status) {
  return status.code() == StatusCode::kNotFound;
}

bool IsOutOfRange(const Status& status) {
  return status.code() == StatusCode::kOutOfRange;
}

bool IsPermissionDenied(const Status& status) {
  return status.code() == StatusCode::kPermissionDenied;
}

bool IsResourceExhausted(const Status& status) {
  return status.code() == StatusCode::kResourceExhausted;
}

bool IsUnauthenticated(const Status& status) {
  return status.code() == StatusCode::kUnauthenticated;
}

bool IsUnavailable(const Status& status) {
  return status.code() == StatusCode::kUnavailable;
}

bool IsUnimplemented(const Status& status) {
  return status.code() == StatusCode::kUnimplemented;
}

bool IsUnknown(const Status& status) {
  return status.code() == StatusCode::kUnknown;
}

Status AbortedError(StringPiece message) {
  return Status(StatusCode::kAborted, message);
}

Status AlreadyExistsError(StringPiece message) {
  return Status(StatusCode::kAlreadyExists, message);
}

Status CancelledError(StringPiece message) {
  return Status(StatusCode::kCancelled, message);
}

Status DataLossError(StringPiece message) {
  return Status(StatusCode::kDataLoss, message);
}

Status DeadlineExceededError(StringPiece message) {
  return Status(StatusCode::kDeadlineExceeded, message);
}

Status FailedPreconditionError(StringPiece message) {
  return Status(StatusCode::kFailedPrecondition, message);
}

Status InternalError(StringPiece message) {
  return Status(StatusCode::kInternal, message);
}

Status InvalidArgumentError(StringPiece message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status NotFoundError(StringPiece message) {
  return Status(StatusCode::kNotFound, message);
}

Status OutOfRangeError(StringPiece message) {
  return Status(StatusCode::kOutOfRange, message);
}

Status PermissionDeniedError(StringPiece message) {
  return Status(StatusCode::kPermissionDenied, message);
}

Status ResourceExhaustedError(StringPiece message) {
  return Status(StatusCode::kResourceExhausted, message);
}

Status UnauthenticatedError(StringPiece message) {
  return Status(StatusCode::kUnauthenticated, message);
}

Status UnavailableError(StringPiece message) {
  return Status(StatusCode::kUnavailable, message);
}

Status UnimplementedError(StringPiece message) {
  return Status(StatusCode::kUnimplemented, message);
}

Status UnknownError(StringPiece message) {
  return Status(StatusCode::kUnknown, message);
}

}  // namespace status_internal
}  // namespace util
}  // namespace protobuf
}  // namespace google
