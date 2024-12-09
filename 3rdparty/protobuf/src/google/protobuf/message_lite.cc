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

// Authors: wink@google.com (Wink Saville),
//          kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/message_lite.h>

#include <climits>
#include <cstdint>
#include <string>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/stringprintf.h>
#include <google/protobuf/parse_context.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/stl_util.h>
#include <google/protobuf/stubs/mutex.h>

#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {

std::string MessageLite::InitializationErrorString() const {
  return "(cannot determine missing fields for lite message)";
}

std::string MessageLite::DebugString() const {
  std::uintptr_t address = reinterpret_cast<std::uintptr_t>(this);
  return StrCat("MessageLite at 0x", strings::Hex(address));
}

namespace {

// When serializing, we first compute the byte size, then serialize the message.
// If serialization produces a different number of bytes than expected, we
// call this function, which crashes.  The problem could be due to a bug in the
// protobuf implementation but is more likely caused by concurrent modification
// of the message.  This function attempts to distinguish between the two and
// provide a useful error message.
void ByteSizeConsistencyError(size_t byte_size_before_serialization,
                              size_t byte_size_after_serialization,
                              size_t bytes_produced_by_serialization,
                              const MessageLite& message) {
  GOOGLE_CHECK_EQ(byte_size_before_serialization, byte_size_after_serialization)
      << message.GetTypeName()
      << " was modified concurrently during serialization.";
  GOOGLE_CHECK_EQ(bytes_produced_by_serialization, byte_size_before_serialization)
      << "Byte size calculation and serialization were inconsistent.  This "
         "may indicate a bug in protocol buffers or it may be caused by "
         "concurrent modification of "
      << message.GetTypeName() << ".";
  GOOGLE_LOG(FATAL) << "This shouldn't be called if all the sizes are equal.";
}

std::string InitializationErrorMessage(const char* action,
                                       const MessageLite& message) {
  // Note:  We want to avoid depending on strutil in the lite library, otherwise
  //   we'd use:
  //
  // return strings::Substitute(
  //   "Can't $0 message of type \"$1\" because it is missing required "
  //   "fields: $2",
  //   action, message.GetTypeName(),
  //   message.InitializationErrorString());

  std::string result;
  result += "Can't ";
  result += action;
  result += " message of type \"";
  result += message.GetTypeName();
  result += "\" because it is missing required fields: ";
  result += message.InitializationErrorString();
  return result;
}

inline StringPiece as_string_view(const void* data, int size) {
  return StringPiece(static_cast<const char*>(data), size);
}

// Returns true of all required fields are present / have values.
inline bool CheckFieldPresence(const internal::ParseContext& ctx,
                               const MessageLite& msg,
                               MessageLite::ParseFlags parse_flags) {
  (void)ctx;  // Parameter is used by Google-internal code.
  if (PROTOBUF_PREDICT_FALSE((parse_flags & MessageLite::kMergePartial) != 0)) {
    return true;
  }
  return msg.IsInitializedWithErrors();
}

}  // namespace

void MessageLite::LogInitializationErrorMessage() const {
  GOOGLE_LOG(ERROR) << InitializationErrorMessage("parse", *this);
}

namespace internal {

template <bool aliasing>
bool MergeFromImpl(StringPiece input, MessageLite* msg,
                   MessageLite::ParseFlags parse_flags) {
  const char* ptr;
  internal::ParseContext ctx(io::CodedInputStream::GetDefaultRecursionLimit(),
                             aliasing, &ptr, input);
  ptr = msg->_InternalParse(ptr, &ctx);
  // ctx has an explicit limit set (length of string_view).
  if (PROTOBUF_PREDICT_TRUE(ptr && ctx.EndedAtLimit())) {
    return CheckFieldPresence(ctx, *msg, parse_flags);
  }
  return false;
}

template <bool aliasing>
bool MergeFromImpl(io::ZeroCopyInputStream* input, MessageLite* msg,
                   MessageLite::ParseFlags parse_flags) {
  const char* ptr;
  internal::ParseContext ctx(io::CodedInputStream::GetDefaultRecursionLimit(),
                             aliasing, &ptr, input);
  ptr = msg->_InternalParse(ptr, &ctx);
  // ctx has no explicit limit (hence we end on end of stream)
  if (PROTOBUF_PREDICT_TRUE(ptr && ctx.EndedAtEndOfStream())) {
    return CheckFieldPresence(ctx, *msg, parse_flags);
  }
  return false;
}

template <bool aliasing>
bool MergeFromImpl(BoundedZCIS input, MessageLite* msg,
                   MessageLite::ParseFlags parse_flags) {
  const char* ptr;
  internal::ParseContext ctx(io::CodedInputStream::GetDefaultRecursionLimit(),
                             aliasing, &ptr, input.zcis, input.limit);
  ptr = msg->_InternalParse(ptr, &ctx);
  if (PROTOBUF_PREDICT_FALSE(!ptr)) return false;
  ctx.BackUp(ptr);
  if (PROTOBUF_PREDICT_TRUE(ctx.EndedAtLimit())) {
    return CheckFieldPresence(ctx, *msg, parse_flags);
  }
  return false;
}

template bool MergeFromImpl<false>(StringPiece input, MessageLite* msg,
                                   MessageLite::ParseFlags parse_flags);
template bool MergeFromImpl<true>(StringPiece input, MessageLite* msg,
                                  MessageLite::ParseFlags parse_flags);
template bool MergeFromImpl<false>(io::ZeroCopyInputStream* input,
                                   MessageLite* msg,
                                   MessageLite::ParseFlags parse_flags);
template bool MergeFromImpl<true>(io::ZeroCopyInputStream* input,
                                  MessageLite* msg,
                                  MessageLite::ParseFlags parse_flags);
template bool MergeFromImpl<false>(BoundedZCIS input, MessageLite* msg,
                                   MessageLite::ParseFlags parse_flags);
template bool MergeFromImpl<true>(BoundedZCIS input, MessageLite* msg,
                                  MessageLite::ParseFlags parse_flags);

}  // namespace internal

class ZeroCopyCodedInputStream : public io::ZeroCopyInputStream {
 public:
  ZeroCopyCodedInputStream(io::CodedInputStream* cis) : cis_(cis) {}
  bool Next(const void** data, int* size) final {
    if (!cis_->GetDirectBufferPointer(data, size)) return false;
    cis_->Skip(*size);
    return true;
  }
  void BackUp(int count) final { cis_->Advance(-count); }
  bool Skip(int count) final { return cis_->Skip(count); }
  int64_t ByteCount() const final { return 0; }

  bool aliasing_enabled() { return cis_->aliasing_enabled_; }

 private:
  io::CodedInputStream* cis_;
};

bool MessageLite::MergeFromImpl(io::CodedInputStream* input,
                                MessageLite::ParseFlags parse_flags) {
  ZeroCopyCodedInputStream zcis(input);
  const char* ptr;
  internal::ParseContext ctx(input->RecursionBudget(), zcis.aliasing_enabled(),
                             &ptr, &zcis);
  // MergePartialFromCodedStream allows terminating the wireformat by 0 or
  // end-group tag. Leaving it up to the caller to verify correct ending by
  // calling LastTagWas on input. We need to maintain this behavior.
  ctx.TrackCorrectEnding();
  ctx.data().pool = input->GetExtensionPool();
  ctx.data().factory = input->GetExtensionFactory();
  ptr = _InternalParse(ptr, &ctx);
  if (PROTOBUF_PREDICT_FALSE(!ptr)) return false;
  ctx.BackUp(ptr);
  if (!ctx.EndedAtEndOfStream()) {
    GOOGLE_DCHECK(ctx.LastTag() != 1);  // We can't end on a pushed limit.
    if (ctx.IsExceedingLimit(ptr)) return false;
    input->SetLastTag(ctx.LastTag());
  } else {
    input->SetConsumed();
  }
  return CheckFieldPresence(ctx, *this, parse_flags);
}

bool MessageLite::MergePartialFromCodedStream(io::CodedInputStream* input) {
  return MergeFromImpl(input, kMergePartial);
}

bool MessageLite::MergeFromCodedStream(io::CodedInputStream* input) {
  return MergeFromImpl(input, kMerge);
}

bool MessageLite::ParseFromCodedStream(io::CodedInputStream* input) {
  Clear();
  return MergeFromImpl(input, kParse);
}

bool MessageLite::ParsePartialFromCodedStream(io::CodedInputStream* input) {
  Clear();
  return MergeFromImpl(input, kParsePartial);
}

bool MessageLite::ParseFromZeroCopyStream(io::ZeroCopyInputStream* input) {
  return ParseFrom<kParse>(input);
}

bool MessageLite::ParsePartialFromZeroCopyStream(
    io::ZeroCopyInputStream* input) {
  return ParseFrom<kParsePartial>(input);
}

bool MessageLite::ParseFromFileDescriptor(int file_descriptor) {
  io::FileInputStream input(file_descriptor);
  return ParseFromZeroCopyStream(&input) && input.GetErrno() == 0;
}

bool MessageLite::ParsePartialFromFileDescriptor(int file_descriptor) {
  io::FileInputStream input(file_descriptor);
  return ParsePartialFromZeroCopyStream(&input) && input.GetErrno() == 0;
}

bool MessageLite::ParseFromIstream(std::istream* input) {
  io::IstreamInputStream zero_copy_input(input);
  return ParseFromZeroCopyStream(&zero_copy_input) && input->eof();
}

bool MessageLite::ParsePartialFromIstream(std::istream* input) {
  io::IstreamInputStream zero_copy_input(input);
  return ParsePartialFromZeroCopyStream(&zero_copy_input) && input->eof();
}

bool MessageLite::MergePartialFromBoundedZeroCopyStream(
    io::ZeroCopyInputStream* input, int size) {
  return ParseFrom<kMergePartial>(internal::BoundedZCIS{input, size});
}

bool MessageLite::MergeFromBoundedZeroCopyStream(io::ZeroCopyInputStream* input,
                                                 int size) {
  return ParseFrom<kMerge>(internal::BoundedZCIS{input, size});
}

bool MessageLite::ParseFromBoundedZeroCopyStream(io::ZeroCopyInputStream* input,
                                                 int size) {
  return ParseFrom<kParse>(internal::BoundedZCIS{input, size});
}

bool MessageLite::ParsePartialFromBoundedZeroCopyStream(
    io::ZeroCopyInputStream* input, int size) {
  return ParseFrom<kParsePartial>(internal::BoundedZCIS{input, size});
}

bool MessageLite::ParseFromString(ConstStringParam data) {
  return ParseFrom<kParse>(data);
}

bool MessageLite::ParsePartialFromString(ConstStringParam data) {
  return ParseFrom<kParsePartial>(data);
}

bool MessageLite::ParseFromArray(const void* data, int size) {
  return ParseFrom<kParse>(as_string_view(data, size));
}

bool MessageLite::ParsePartialFromArray(const void* data, int size) {
  return ParseFrom<kParsePartial>(as_string_view(data, size));
}

bool MessageLite::MergeFromString(ConstStringParam data) {
  return ParseFrom<kMerge>(data);
}


// ===================================================================

inline uint8_t* SerializeToArrayImpl(const MessageLite& msg, uint8_t* target,
                                     int size) {
  constexpr bool debug = false;
  if (debug) {
    // Force serialization to a stream with a block size of 1, which forces
    // all writes to the stream to cross buffers triggering all fallback paths
    // in the unittests when serializing to string / array.
    io::ArrayOutputStream stream(target, size, 1);
    uint8_t* ptr;
    io::EpsCopyOutputStream out(
        &stream, io::CodedOutputStream::IsDefaultSerializationDeterministic(),
        &ptr);
    ptr = msg._InternalSerialize(ptr, &out);
    out.Trim(ptr);
    GOOGLE_DCHECK(!out.HadError() && stream.ByteCount() == size);
    return target + size;
  } else {
    io::EpsCopyOutputStream out(
        target, size,
        io::CodedOutputStream::IsDefaultSerializationDeterministic());
    auto res = msg._InternalSerialize(target, &out);
    GOOGLE_DCHECK(target + size == res);
    return res;
  }
}

uint8_t* MessageLite::SerializeWithCachedSizesToArray(uint8_t* target) const {
  // We only optimize this when using optimize_for = SPEED.  In other cases
  // we just use the CodedOutputStream path.
  return SerializeToArrayImpl(*this, target, GetCachedSize());
}

bool MessageLite::SerializeToCodedStream(io::CodedOutputStream* output) const {
  GOOGLE_DCHECK(IsInitialized()) << InitializationErrorMessage("serialize", *this);
  return SerializePartialToCodedStream(output);
}

bool MessageLite::SerializePartialToCodedStream(
    io::CodedOutputStream* output) const {
  const size_t size = ByteSizeLong();  // Force size to be cached.
  if (size > INT_MAX) {
    GOOGLE_LOG(ERROR) << GetTypeName()
               << " exceeded maximum protobuf size of 2GB: " << size;
    return false;
  }

  int original_byte_count = output->ByteCount();
  SerializeWithCachedSizes(output);
  if (output->HadError()) {
    return false;
  }
  int final_byte_count = output->ByteCount();

  if (final_byte_count - original_byte_count != static_cast<int64_t>(size)) {
    ByteSizeConsistencyError(size, ByteSizeLong(),
                             final_byte_count - original_byte_count, *this);
  }

  return true;
}

bool MessageLite::SerializeToZeroCopyStream(
    io::ZeroCopyOutputStream* output) const {
  GOOGLE_DCHECK(IsInitialized()) << InitializationErrorMessage("serialize", *this);
  return SerializePartialToZeroCopyStream(output);
}

bool MessageLite::SerializePartialToZeroCopyStream(
    io::ZeroCopyOutputStream* output) const {
  const size_t size = ByteSizeLong();  // Force size to be cached.
  if (size > INT_MAX) {
    GOOGLE_LOG(ERROR) << GetTypeName()
               << " exceeded maximum protobuf size of 2GB: " << size;
    return false;
  }

  uint8_t* target;
  io::EpsCopyOutputStream stream(
      output, io::CodedOutputStream::IsDefaultSerializationDeterministic(),
      &target);
  target = _InternalSerialize(target, &stream);
  stream.Trim(target);
  if (stream.HadError()) return false;
  return true;
}

bool MessageLite::SerializeToFileDescriptor(int file_descriptor) const {
  io::FileOutputStream output(file_descriptor);
  return SerializeToZeroCopyStream(&output) && output.Flush();
}

bool MessageLite::SerializePartialToFileDescriptor(int file_descriptor) const {
  io::FileOutputStream output(file_descriptor);
  return SerializePartialToZeroCopyStream(&output) && output.Flush();
}

bool MessageLite::SerializeToOstream(std::ostream* output) const {
  {
    io::OstreamOutputStream zero_copy_output(output);
    if (!SerializeToZeroCopyStream(&zero_copy_output)) return false;
  }
  return output->good();
}

bool MessageLite::SerializePartialToOstream(std::ostream* output) const {
  io::OstreamOutputStream zero_copy_output(output);
  return SerializePartialToZeroCopyStream(&zero_copy_output);
}

bool MessageLite::AppendToString(std::string* output) const {
  GOOGLE_DCHECK(IsInitialized()) << InitializationErrorMessage("serialize", *this);
  return AppendPartialToString(output);
}

bool MessageLite::AppendPartialToString(std::string* output) const {
  size_t old_size = output->size();
  size_t byte_size = ByteSizeLong();
  if (byte_size > INT_MAX) {
    GOOGLE_LOG(ERROR) << GetTypeName()
               << " exceeded maximum protobuf size of 2GB: " << byte_size;
    return false;
  }

  STLStringResizeUninitializedAmortized(output, old_size + byte_size);
  uint8_t* start =
      reinterpret_cast<uint8_t*>(io::mutable_string_data(output) + old_size);
  SerializeToArrayImpl(*this, start, byte_size);
  return true;
}

bool MessageLite::SerializeToString(std::string* output) const {
  output->clear();
  return AppendToString(output);
}

bool MessageLite::SerializePartialToString(std::string* output) const {
  output->clear();
  return AppendPartialToString(output);
}

bool MessageLite::SerializeToArray(void* data, int size) const {
  GOOGLE_DCHECK(IsInitialized()) << InitializationErrorMessage("serialize", *this);
  return SerializePartialToArray(data, size);
}

bool MessageLite::SerializePartialToArray(void* data, int size) const {
  const size_t byte_size = ByteSizeLong();
  if (byte_size > INT_MAX) {
    GOOGLE_LOG(ERROR) << GetTypeName()
               << " exceeded maximum protobuf size of 2GB: " << byte_size;
    return false;
  }
  if (size < static_cast<int64_t>(byte_size)) return false;
  uint8_t* start = reinterpret_cast<uint8_t*>(data);
  SerializeToArrayImpl(*this, start, byte_size);
  return true;
}

std::string MessageLite::SerializeAsString() const {
  // If the compiler implements the (Named) Return Value Optimization,
  // the local variable 'output' will not actually reside on the stack
  // of this function, but will be overlaid with the object that the
  // caller supplied for the return value to be constructed in.
  std::string output;
  if (!AppendToString(&output)) output.clear();
  return output;
}

std::string MessageLite::SerializePartialAsString() const {
  std::string output;
  if (!AppendPartialToString(&output)) output.clear();
  return output;
}


namespace internal {

template <>
MessageLite* GenericTypeHandler<MessageLite>::NewFromPrototype(
    const MessageLite* prototype, Arena* arena) {
  return prototype->New(arena);
}
template <>
void GenericTypeHandler<MessageLite>::Merge(const MessageLite& from,
                                            MessageLite* to) {
  to->CheckTypeAndMergeFrom(from);
}
template <>
void GenericTypeHandler<std::string>::Merge(const std::string& from,
                                            std::string* to) {
  *to = from;
}

// Non-inline variants of std::string specializations for
// various InternalMetadata routines.
template <>
void InternalMetadata::DoClear<std::string>() {
  mutable_unknown_fields<std::string>()->clear();
}

template <>
void InternalMetadata::DoMergeFrom<std::string>(const std::string& other) {
  mutable_unknown_fields<std::string>()->append(other);
}

template <>
void InternalMetadata::DoSwap<std::string>(std::string* other) {
  mutable_unknown_fields<std::string>()->swap(*other);
}

}  // namespace internal


// ===================================================================
// Shutdown support.

namespace internal {

struct ShutdownData {
  ~ShutdownData() {
    std::reverse(functions.begin(), functions.end());
    for (auto pair : functions) pair.first(pair.second);
  }

  static ShutdownData* get() {
    static auto* data = new ShutdownData;
    return data;
  }

  std::vector<std::pair<void (*)(const void*), const void*>> functions;
  Mutex mutex;
};

static void RunZeroArgFunc(const void* arg) {
  void (*func)() = reinterpret_cast<void (*)()>(const_cast<void*>(arg));
  func();
}

void OnShutdown(void (*func)()) {
  OnShutdownRun(RunZeroArgFunc, reinterpret_cast<void*>(func));
}

void OnShutdownRun(void (*f)(const void*), const void* arg) {
  auto shutdown_data = ShutdownData::get();
  MutexLock lock(&shutdown_data->mutex);
  shutdown_data->functions.push_back(std::make_pair(f, arg));
}

}  // namespace internal

void ShutdownProtobufLibrary() {
  // This function should be called only once, but accepts multiple calls.
  static bool is_shutdown = false;
  if (!is_shutdown) {
    delete internal::ShutdownData::get();
    is_shutdown = true;
  }
}


}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
