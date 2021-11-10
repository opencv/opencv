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

#ifndef GOOGLE_PROTOBUF_METADATA_LITE_H__
#define GOOGLE_PROTOBUF_METADATA_LITE_H__

#include <string>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/port.h>

#include <google/protobuf/port_def.inc>

#ifdef SWIG
#error "You cannot SWIG proto headers"
#endif

namespace google {
namespace protobuf {
namespace internal {

// This is the representation for messages that support arena allocation. It
// uses a tagged pointer to either store the owning Arena pointer, if there are
// no unknown fields, or a pointer to a block of memory with both the owning
// Arena pointer and the UnknownFieldSet, if there are unknown fields. Besides,
// it also uses the tag to distinguish whether the owning Arena pointer is also
// used by sub-structure allocation. This optimization allows for
// "zero-overhead" storage of the Arena pointer, relative to the above baseline
// implementation.
//
// The tagged pointer uses the least two significant bits to disambiguate cases.
// It uses bit 0 == 0 to indicate an arena pointer and bit 0 == 1 to indicate a
// UFS+Arena-container pointer. Besides it uses bit 1 == 0 to indicate arena
// allocation and bit 1 == 1 to indicate heap allocation.
class InternalMetadata {
 public:
  constexpr InternalMetadata() : ptr_(0) {}
  explicit InternalMetadata(Arena* arena, bool is_message_owned = false)
      : ptr_(is_message_owned
                 ? reinterpret_cast<intptr_t>(arena) | kMessageOwnedArenaTagMask
                 : reinterpret_cast<intptr_t>(arena)) {
    GOOGLE_DCHECK(!is_message_owned || arena != nullptr);
  }

  ~InternalMetadata() {
    if (HasMessageOwnedArenaTag()) {
      delete arena();
    }
  }

  template <typename T>
  void Delete() {
    // Note that Delete<> should be called not more than once.
    if (have_unknown_fields()) {
      DeleteOutOfLineHelper<T>();
    }
  }

  PROTOBUF_NDEBUG_INLINE Arena* owning_arena() const {
    return HasMessageOwnedArenaTag() ? nullptr : arena();
  }

  PROTOBUF_NDEBUG_INLINE Arena* arena() const {
    if (PROTOBUF_PREDICT_FALSE(have_unknown_fields())) {
      return PtrValue<ContainerBase>()->arena;
    } else {
      return PtrValue<Arena>();
    }
  }

  PROTOBUF_NDEBUG_INLINE bool have_unknown_fields() const {
    return HasUnknownFieldsTag();
  }

  PROTOBUF_NDEBUG_INLINE void* raw_arena_ptr() const {
    return reinterpret_cast<void*>(ptr_);
  }

  template <typename T>
  PROTOBUF_NDEBUG_INLINE const T& unknown_fields(
      const T& (*default_instance)()) const {
    if (PROTOBUF_PREDICT_FALSE(have_unknown_fields())) {
      return PtrValue<Container<T>>()->unknown_fields;
    } else {
      return default_instance();
    }
  }

  template <typename T>
  PROTOBUF_NDEBUG_INLINE T* mutable_unknown_fields() {
    if (PROTOBUF_PREDICT_TRUE(have_unknown_fields())) {
      return &PtrValue<Container<T>>()->unknown_fields;
    } else {
      return mutable_unknown_fields_slow<T>();
    }
  }

  template <typename T>
  PROTOBUF_NDEBUG_INLINE void Swap(InternalMetadata* other) {
    // Semantics here are that we swap only the unknown fields, not the arena
    // pointer. We cannot simply swap ptr_ with other->ptr_ because we need to
    // maintain our own arena ptr. Also, our ptr_ and other's ptr_ may be in
    // different states (direct arena pointer vs. container with UFS) so we
    // cannot simply swap ptr_ and then restore the arena pointers. We reuse
    // UFS's swap implementation instead.
    if (have_unknown_fields() || other->have_unknown_fields()) {
      DoSwap<T>(other->mutable_unknown_fields<T>());
    }
  }

  PROTOBUF_NDEBUG_INLINE void InternalSwap(InternalMetadata* other) {
    std::swap(ptr_, other->ptr_);
  }

  template <typename T>
  PROTOBUF_NDEBUG_INLINE void MergeFrom(const InternalMetadata& other) {
    if (other.have_unknown_fields()) {
      DoMergeFrom<T>(other.unknown_fields<T>(nullptr));
    }
  }

  template <typename T>
  PROTOBUF_NDEBUG_INLINE void Clear() {
    if (have_unknown_fields()) {
      DoClear<T>();
    }
  }

 private:
  intptr_t ptr_;

  // Tagged pointer implementation.
  static constexpr intptr_t kUnknownFieldsTagMask = 1;
  static constexpr intptr_t kMessageOwnedArenaTagMask = 2;
  static constexpr intptr_t kPtrTagMask =
      kUnknownFieldsTagMask | kMessageOwnedArenaTagMask;
  static constexpr intptr_t kPtrValueMask = ~kPtrTagMask;

  // Accessors for pointer tag and pointer value.
  PROTOBUF_ALWAYS_INLINE bool HasUnknownFieldsTag() const {
    return ptr_ & kUnknownFieldsTagMask;
  }
  PROTOBUF_ALWAYS_INLINE bool HasMessageOwnedArenaTag() const {
    return ptr_ & kMessageOwnedArenaTagMask;
  }

  template <typename U>
  U* PtrValue() const {
    return reinterpret_cast<U*>(ptr_ & kPtrValueMask);
  }

  // If ptr_'s tag is kTagContainer, it points to an instance of this struct.
  struct ContainerBase {
    Arena* arena;
  };

  template <typename T>
  struct Container : public ContainerBase {
    T unknown_fields;
  };

  template <typename T>
  PROTOBUF_NOINLINE void DeleteOutOfLineHelper() {
    if (arena() == nullptr) {
      delete PtrValue<Container<T>>();
    }
  }

  template <typename T>
  PROTOBUF_NOINLINE T* mutable_unknown_fields_slow() {
    Arena* my_arena = arena();
    Container<T>* container = Arena::Create<Container<T>>(my_arena);
    intptr_t message_owned_arena_tag = ptr_ & kMessageOwnedArenaTagMask;
    // Two-step assignment works around a bug in clang's static analyzer:
    // https://bugs.llvm.org/show_bug.cgi?id=34198.
    ptr_ = reinterpret_cast<intptr_t>(container);
    ptr_ |= kUnknownFieldsTagMask | message_owned_arena_tag;
    container->arena = my_arena;
    return &(container->unknown_fields);
  }

  // Templated functions.

  template <typename T>
  PROTOBUF_NOINLINE void DoClear() {
    mutable_unknown_fields<T>()->Clear();
  }

  template <typename T>
  PROTOBUF_NOINLINE void DoMergeFrom(const T& other) {
    mutable_unknown_fields<T>()->MergeFrom(other);
  }

  template <typename T>
  PROTOBUF_NOINLINE void DoSwap(T* other) {
    mutable_unknown_fields<T>()->Swap(other);
  }
};

// String Template specializations.

template <>
PROTOBUF_EXPORT void InternalMetadata::DoClear<std::string>();
template <>
PROTOBUF_EXPORT void InternalMetadata::DoMergeFrom<std::string>(
    const std::string& other);
template <>
PROTOBUF_EXPORT void InternalMetadata::DoSwap<std::string>(std::string* other);

// This helper RAII class is needed to efficiently parse unknown fields. We
// should only call mutable_unknown_fields if there are actual unknown fields.
// The obvious thing to just use a stack string and swap it at the end of
// the parse won't work, because the destructor of StringOutputStream needs to
// be called before we can modify the string (it check-fails). Using
// LiteUnknownFieldSetter setter(&_internal_metadata_);
// StringOutputStream stream(setter.buffer());
// guarantees that the string is only swapped after stream is destroyed.
class PROTOBUF_EXPORT LiteUnknownFieldSetter {
 public:
  explicit LiteUnknownFieldSetter(InternalMetadata* metadata)
      : metadata_(metadata) {
    if (metadata->have_unknown_fields()) {
      buffer_.swap(*metadata->mutable_unknown_fields<std::string>());
    }
  }
  ~LiteUnknownFieldSetter() {
    if (!buffer_.empty())
      metadata_->mutable_unknown_fields<std::string>()->swap(buffer_);
  }
  std::string* buffer() { return &buffer_; }

 private:
  InternalMetadata* metadata_;
  std::string buffer_;
};

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_METADATA_LITE_H__
