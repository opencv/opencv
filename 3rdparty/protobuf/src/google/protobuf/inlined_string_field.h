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

#ifndef GOOGLE_PROTOBUF_INLINED_STRING_FIELD_H__
#define GOOGLE_PROTOBUF_INLINED_STRING_FIELD_H__

#include <string>
#include <utility>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/port.h>
#include <google/protobuf/stubs/strutil.h>

// Must be included last.
#include <google/protobuf/port_def.inc>

#ifdef SWIG
#error "You cannot SWIG proto headers"
#endif

namespace google {
namespace protobuf {

class Arena;

namespace internal {

// InlinedStringField wraps a std::string instance and exposes an API similar to
// ArenaStringPtr's wrapping of a std::string* instance.
//
// default_value parameters are taken for consistency with ArenaStringPtr, but
// are not used for most methods. With inlining, these should be removed from
// the generated binary.
//
// InlinedStringField has a donating mechanism that allows string buffer
// allocated on arena. A string is donated means both the string container and
// the data buffer are on arena. The donating mechanism here is similar to the
// one in ArenaStringPtr with some differences:
//
// When an InlinedStringField is constructed, the donating state is true. This
// is because the string container is directly stored in the message on the
// arena:
//
//   Construction: donated=true
//   Arena:
//   +-----------------------+
//   |Message foo:           |
//   | +-------------------+ |
//   | |InlinedStringField:| |
//   | | +-----+           | |
//   | | | | | |           | |
//   | | +-----+           | |
//   | +-------------------+ |
//   +-----------------------+
//
// When lvalue Set is called, the donating state is still true. String data will
// be allocated on the arena:
//
//   Lvalue Set: donated=true
//   Arena:
//   +-----------------------+
//   |Message foo:           |
//   | +-------------------+ |
//   | |InlinedStringField:| |
//   | | +-----+           | |
//   | | | | | |           | |
//   | | +|----+           | |
//   | +--|----------------+ |
//   |    V                  |
//   |  +----------------+   |
//   |  |'f','o','o',... |   |
//   |  +----------------+   |
//   +-----------------------+
//
// Some operations will undonate a donated string, including: Mutable,
// SetAllocated, Rvalue Set, and Swap with a non-donated string.
//
// For more details of the donating states transitions, go/pd-inlined-string.
class PROTOBUF_EXPORT InlinedStringField {
 public:
  InlinedStringField() { Init(); }
  inline void Init() { new (get_mutable()) std::string(); }
  // Add the dummy parameter just to make InlinedStringField(nullptr)
  // unambiguous.
  constexpr InlinedStringField(
      const ExplicitlyConstructed<std::string>* /*default_value*/,
      bool /*dummy*/)
      : value_{} {}
  explicit InlinedStringField(const std::string& default_value);
  explicit InlinedStringField(Arena* arena);
  ~InlinedStringField() { Destruct(); }

  // Lvalue Set. To save space, we pack the donating states of multiple
  // InlinedStringFields into an uint32_t `donating_states`. The `mask`
  // indicates the position of the bit for this InlinedStringField. `donated` is
  // whether this field is donated.
  //
  // The caller should guarantee that:
  //
  //   `donated == ((donating_states & ~mask) != 0)`
  //
  // This method never changes the `donating_states`.
  void Set(const std::string* default_value, ConstStringParam value,
           Arena* arena, bool donated, uint32_t* /*donating_states*/,
           uint32_t /*mask*/) {
    (void)arena;
    (void)donated;
    SetNoArena(default_value, value);
  }

  // Rvalue Set. If this field is donated, this method will undonate this field
  // by mutating the `donating_states` according to `mask`.
  void Set(const std::string* default_value, std::string&& value, Arena* arena,
           bool donated, uint32_t* donating_states, uint32_t mask);

  template <typename FirstParam>
  void Set(FirstParam p1, const char* str, ::google::protobuf::Arena* arena, bool donated,
           uint32_t* donating_states, uint32_t mask) {
    Set(p1, ConstStringParam(str), arena, donated, donating_states, mask);
  }

  template <typename FirstParam>
  void Set(FirstParam p1, const char* str, size_t size, ::google::protobuf::Arena* arena,
           bool donated, uint32_t* donating_states, uint32_t mask) {
    ConstStringParam sp{str, size};  // for string_view and `const string &`
    Set(p1, sp, arena, donated, donating_states, mask);
  }

  template <typename FirstParam, typename RefWrappedType>
  void Set(FirstParam p1,
           std::reference_wrapper<RefWrappedType> const_string_ref,
           ::google::protobuf::Arena* arena, bool donated, uint32_t* donating_states,
           uint32_t mask) {
    Set(p1, const_string_ref.get(), arena, donated, donating_states, mask);
  }

  template <typename FirstParam, typename SecondParam>
  void SetBytes(FirstParam p1, SecondParam&& p2, ::google::protobuf::Arena* arena,
                bool donated, uint32_t* donating_states, uint32_t mask) {
    Set(p1, static_cast<SecondParam&&>(p2), arena, donated, donating_states,
        mask);
  }

  template <typename FirstParam>
  void SetBytes(FirstParam p1, const void* str, size_t size,
                ::google::protobuf::Arena* arena, bool donated, uint32_t* donating_states,
                uint32_t mask) {
    // Must work whether ConstStringParam is string_view or `const string &`
    ConstStringParam sp{static_cast<const char*>(str), size};
    Set(p1, sp, arena, donated, donating_states, mask);
  }

  PROTOBUF_NDEBUG_INLINE void SetNoArena(const std::string* default_value,
                                         StringPiece value);
  PROTOBUF_NDEBUG_INLINE void SetNoArena(const std::string* default_value,
                                         std::string&& value);

  // Basic accessors.
  PROTOBUF_NDEBUG_INLINE const std::string& Get() const { return GetNoArena(); }
  PROTOBUF_NDEBUG_INLINE const std::string& GetNoArena() const;

  // Mutable returns a std::string* instance that is heap-allocated. If this
  // field is donated, this method undonates this field by mutating the
  // `donating_states` according to `mask`, and copies the content of the
  // original string to the returning string.
  std::string* Mutable(const LazyString& default_value, Arena* arena,
                       bool donated, uint32_t* donating_states, uint32_t mask);
  std::string* Mutable(ArenaStringPtr::EmptyDefault, Arena* arena, bool donated,
                       uint32_t* donating_states, uint32_t mask);

  // Release returns a std::string* instance that is heap-allocated and is not
  // Own()'d by any arena. If the field is not set, this returns nullptr. The
  // caller retains ownership. Clears this field back to nullptr state. Used to
  // implement release_<field>() methods on generated classes.
  PROTOBUF_NODISCARD std::string* Release(const std::string* default_value,
                                          Arena* arena, bool donated);
  PROTOBUF_NODISCARD std::string* ReleaseNonDefault(
      const std::string* default_value, Arena* arena);
  std::string* ReleaseNonDefaultNoArena(const std::string* default_value);

  // Takes a std::string that is heap-allocated, and takes ownership. The
  // std::string's destructor is registered with the arena. Used to implement
  // set_allocated_<field> in generated classes.
  //
  // If this field is donated, this method undonates this field by mutating the
  // `donating_states` according to `mask`.
  void SetAllocated(const std::string* default_value, std::string* value,
                    Arena* arena, bool donated, uint32_t* donating_states,
                    uint32_t mask);

  void SetAllocatedNoArena(const std::string* default_value,
                           std::string* value);

  // When one of `this` and `from` is donated and the other is not donated, this
  // method will undonate the donated one and swap the two heap-allocated
  // strings.
  PROTOBUF_NDEBUG_INLINE void Swap(InlinedStringField* from,
                                   const std::string* default_value,
                                   Arena* arena, bool donated,
                                   bool from_donated, uint32_t* donating_states,
                                   uint32_t* from_donating_states,
                                   uint32_t mask);

  // Frees storage (if not on an arena).
  PROTOBUF_NDEBUG_INLINE void Destroy(const std::string* default_value,
                                      Arena* arena) {
    if (arena == nullptr) {
      DestroyNoArena(default_value);
    }
  }
  PROTOBUF_NDEBUG_INLINE void DestroyNoArena(const std::string* default_value);

  // Clears content, but keeps allocated std::string, to avoid the overhead of
  // heap operations. After this returns, the content (as seen by the user) will
  // always be the empty std::string.
  PROTOBUF_NDEBUG_INLINE void ClearToEmpty() { ClearNonDefaultToEmpty(); }
  PROTOBUF_NDEBUG_INLINE void ClearNonDefaultToEmpty() {
    get_mutable()->clear();
  }

  // Clears content, but keeps allocated std::string if arena != nullptr, to
  // avoid the overhead of heap operations. After this returns, the content (as
  // seen by the user) will always be equal to |default_value|.
  void ClearToDefault(const LazyString& default_value, Arena* arena,
                      bool donated);

  // Returns a mutable pointer, but doesn't initialize the string to the
  // default value.
  PROTOBUF_NDEBUG_INLINE std::string* MutableNoArenaNoDefault(
      const std::string* /*default_value*/);

  // Generated code / reflection only! Returns a mutable pointer to the string.
  PROTOBUF_NDEBUG_INLINE std::string* UnsafeMutablePointer();

  // InlinedStringField doesn't have things like the `default_value` pointer in
  // ArenaStringPtr.
  bool IsDefault(const std::string* /*default_value*/) const { return false; }

 private:
  void Destruct() { get_mutable()->~basic_string(); }

  PROTOBUF_NDEBUG_INLINE std::string* get_mutable();
  PROTOBUF_NDEBUG_INLINE const std::string* get_const() const;

  alignas(std::string) char value_[sizeof(std::string)];

  std::string* MutableSlow(::google::protobuf::Arena* arena, bool donated,
                           uint32_t* donating_states, uint32_t mask);


  // When constructed in an Arena, we want our destructor to be skipped.
  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
};

inline std::string* InlinedStringField::get_mutable() {
  return reinterpret_cast<std::string*>(&value_);
}

inline const std::string* InlinedStringField::get_const() const {
  return reinterpret_cast<const std::string*>(&value_);
}

inline InlinedStringField::InlinedStringField(
    const std::string& default_value) {
  new (get_mutable()) std::string(default_value);
}

inline InlinedStringField::InlinedStringField(Arena* arena) {
  Init();
  if (arena != nullptr) {
    arena->OwnDestructor(get_mutable());
  }
}

inline const std::string& InlinedStringField::GetNoArena() const {
  return *get_const();
}

inline void InlinedStringField::SetAllocatedNoArena(
    const std::string* /*default_value*/, std::string* value) {
  if (value == nullptr) {
    // Currently, inlined string field can't have non empty default.
    get_mutable()->clear();
  } else {
    get_mutable()->assign(std::move(*value));
    delete value;
  }
}

inline void InlinedStringField::DestroyNoArena(const std::string*) {
  // This is invoked from the generated message's ArenaDtor, which is used to
  // clean up objects not allocated on the Arena.
  this->~InlinedStringField();
}

inline std::string* InlinedStringField::ReleaseNonDefaultNoArena(
    const std::string* /*default_value*/) {
  // Currently, inlined string field can't have non empty default.
  auto* released = new std::string();
  get_mutable()->swap(*released);
  return released;
}

inline void InlinedStringField::SetNoArena(const std::string* /*default_value*/,
                                           StringPiece value) {
  get_mutable()->assign(value.data(), value.length());
}

inline void InlinedStringField::SetNoArena(const std::string* /*default_value*/,
                                           std::string&& value) {
  get_mutable()->assign(std::move(value));
}

inline void InlinedStringField::Swap(
    InlinedStringField* from, const std::string* /*default_value*/,
    Arena* arena, bool donated, bool from_donated, uint32_t* donating_states,
    uint32_t* from_donating_states, uint32_t mask) {
#if GOOGLE_PROTOBUF_INTERNAL_DONATE_STEAL_INLINE
  // If one is donated and the other is not, undonate the donated one.
  if (donated && !from_donated) {
    MutableSlow(arena, donated, donating_states, mask);
  } else if (!donated && from_donated) {
    from->MutableSlow(arena, from_donated, from_donating_states, mask);
  }
  // Then, swap the two undonated strings.
#else
  (void)arena;
  (void)donated;
  (void)from_donated;
  (void)donating_states;
  (void)from_donating_states;
  (void)mask;
#endif
  get_mutable()->swap(*from->get_mutable());
}

inline std::string* InlinedStringField::MutableNoArenaNoDefault(
    const std::string*) {
  return get_mutable();
}

inline std::string* InlinedStringField::UnsafeMutablePointer() {
  return get_mutable();
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_INLINED_STRING_FIELD_H__
