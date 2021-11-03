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

#include <google/protobuf/generated_message_bases.h>

#include <google/protobuf/parse_context.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/wire_format_lite.h>

// Must be last:
#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace internal {

// =============================================================================
// ZeroFieldsBase

void ZeroFieldsBase::Clear() {
  _internal_metadata_.Clear<UnknownFieldSet>();  //
}

ZeroFieldsBase::~ZeroFieldsBase() {
  if (GetArenaForAllocation() != nullptr) return;
  _internal_metadata_.Delete<UnknownFieldSet>();
}

size_t ZeroFieldsBase::ByteSizeLong() const {
  return MaybeComputeUnknownFieldsSize(0, &_cached_size_);
}

const char* ZeroFieldsBase::_InternalParse(const char* ptr,
                                           internal::ParseContext* ctx) {
#define CHK_(x)                       \
  if (PROTOBUF_PREDICT_FALSE(!(x))) { \
    goto failure;                     \
  }

  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = internal::ReadTag(ptr, &tag);
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag, _internal_metadata_.mutable_unknown_fields<UnknownFieldSet>(), ptr,
        ctx);
    CHK_(ptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::uint8_t* ZeroFieldsBase::_InternalSerialize(
    ::uint8_t* target, io::EpsCopyOutputStream* stream) const {
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<UnknownFieldSet>(
            UnknownFieldSet::default_instance),
        target, stream);
  }
  return target;
}

void ZeroFieldsBase::MergeImpl(Message* to_param, const Message& from_param) {
  auto* to = static_cast<ZeroFieldsBase*>(to_param);
  const auto* from = static_cast<const ZeroFieldsBase*>(&from_param);
  GOOGLE_DCHECK_NE(from, to);
  to->_internal_metadata_.MergeFrom<UnknownFieldSet>(from->_internal_metadata_);
}

void ZeroFieldsBase::CopyImpl(Message* to_param, const Message& from_param) {
  auto* to = static_cast<ZeroFieldsBase*>(to_param);
  const auto* from = static_cast<const ZeroFieldsBase*>(&from_param);
  if (from == to) return;
  to->_internal_metadata_.Clear<UnknownFieldSet>();
  to->_internal_metadata_.MergeFrom<UnknownFieldSet>(from->_internal_metadata_);
}

void ZeroFieldsBase::InternalSwap(ZeroFieldsBase* other) {
  _internal_metadata_.Swap<UnknownFieldSet>(&other->_internal_metadata_);
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
