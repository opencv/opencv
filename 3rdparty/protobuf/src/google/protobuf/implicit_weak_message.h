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

#ifndef GOOGLE_PROTOBUF_IMPLICIT_WEAK_MESSAGE_H__
#define GOOGLE_PROTOBUF_IMPLICIT_WEAK_MESSAGE_H__

#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>

#ifdef SWIG
#error "You cannot SWIG proto headers"
#endif

#include <google/protobuf/port_def.inc>

// This file is logically internal-only and should only be used by protobuf
// generated code.

namespace google {
namespace protobuf {
namespace internal {

// An implementation of MessageLite that treats all data as unknown. This type
// acts as a placeholder for an implicit weak field in the case where the true
// message type does not get linked into the binary.
class PROTOBUF_EXPORT ImplicitWeakMessage : public MessageLite {
 public:
  ImplicitWeakMessage() {}
  explicit ImplicitWeakMessage(Arena* arena) : MessageLite(arena) {}

  static const ImplicitWeakMessage* default_instance();

  std::string GetTypeName() const override { return ""; }

  MessageLite* New(Arena* arena) const override {
    return Arena::CreateMessage<ImplicitWeakMessage>(arena);
  }

  void Clear() override { data_.clear(); }

  bool IsInitialized() const override { return true; }

  void CheckTypeAndMergeFrom(const MessageLite& other) override {
    data_.append(static_cast<const ImplicitWeakMessage&>(other).data_);
  }

  const char* _InternalParse(const char* ptr, ParseContext* ctx) final;

  size_t ByteSizeLong() const override { return data_.size(); }

  uint8_t* _InternalSerialize(uint8_t* target,
                              io::EpsCopyOutputStream* stream) const final {
    return stream->WriteRaw(data_.data(), static_cast<int>(data_.size()),
                            target);
  }

  int GetCachedSize() const override { return static_cast<int>(data_.size()); }

  typedef void InternalArenaConstructable_;

 private:
  std::string data_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ImplicitWeakMessage);
};

// A type handler for use with implicit weak repeated message fields.
template <typename ImplicitWeakType>
class ImplicitWeakTypeHandler {
 public:
  typedef MessageLite Type;
  static constexpr bool Moveable = false;

  static inline MessageLite* NewFromPrototype(const MessageLite* prototype,
                                              Arena* arena = nullptr) {
    return prototype->New(arena);
  }

  static inline void Delete(MessageLite* value, Arena* arena) {
    if (arena == nullptr) {
      delete value;
    }
  }
  static inline Arena* GetArena(MessageLite* value) {
    return value->GetArena();
  }
  static inline void Clear(MessageLite* value) { value->Clear(); }
  static void Merge(const MessageLite& from, MessageLite* to) {
    to->CheckTypeAndMergeFrom(from);
  }
};

}  // namespace internal

template <typename T>
struct WeakRepeatedPtrField {
  using TypeHandler = internal::ImplicitWeakTypeHandler<T>;
  constexpr WeakRepeatedPtrField() : weak() {}
  explicit WeakRepeatedPtrField(Arena* arena) : weak(arena) {}
  ~WeakRepeatedPtrField() { weak.template Destroy<TypeHandler>(); }

  typedef internal::RepeatedPtrIterator<MessageLite> iterator;
  typedef internal::RepeatedPtrIterator<const MessageLite> const_iterator;
  typedef internal::RepeatedPtrOverPtrsIterator<MessageLite*, void*>
      pointer_iterator;
  typedef internal::RepeatedPtrOverPtrsIterator<const MessageLite* const,
                                                const void* const>
      const_pointer_iterator;

  iterator begin() { return iterator(base().raw_data()); }
  const_iterator begin() const { return iterator(base().raw_data()); }
  const_iterator cbegin() const { return begin(); }
  iterator end() { return begin() + base().size(); }
  const_iterator end() const { return begin() + base().size(); }
  const_iterator cend() const { return end(); }
  pointer_iterator pointer_begin() {
    return pointer_iterator(base().raw_mutable_data());
  }
  const_pointer_iterator pointer_begin() const {
    return const_pointer_iterator(base().raw_mutable_data());
  }
  pointer_iterator pointer_end() {
    return pointer_iterator(base().raw_mutable_data() + base().size());
  }
  const_pointer_iterator pointer_end() const {
    return const_pointer_iterator(base().raw_mutable_data() + base().size());
  }

  MessageLite* AddWeak(const MessageLite* prototype) {
    return base().AddWeak(prototype);
  }
  T* Add() { return weak.Add(); }
  void Clear() { base().template Clear<TypeHandler>(); }
  void MergeFrom(const WeakRepeatedPtrField& other) {
    base().template MergeFrom<TypeHandler>(other.base());
  }
  void InternalSwap(WeakRepeatedPtrField* other) {
    base().InternalSwap(&other->base());
  }

  const internal::RepeatedPtrFieldBase& base() const { return weak; }
  internal::RepeatedPtrFieldBase& base() { return weak; }
  // Union disables running the destructor. Which would create a strong link.
  // Instead we explicitly destroy the underlying base through the virtual
  // destructor.
  union {
    RepeatedPtrField<T> weak;
  };
};

}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_IMPLICIT_WEAK_MESSAGE_H__
