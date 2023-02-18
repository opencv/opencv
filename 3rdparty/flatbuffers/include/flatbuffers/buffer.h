/*
 * Copyright 2021 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_BUFFER_H_
#define FLATBUFFERS_BUFFER_H_

#include <algorithm>

#include "flatbuffers/base.h"

namespace flatbuffers {

// Wrapper for uoffset_t to allow safe template specialization.
// Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
template<typename T> struct Offset {
  uoffset_t o;
  Offset() : o(0) {}
  Offset(uoffset_t _o) : o(_o) {}
  Offset<void> Union() const { return Offset<void>(o); }
  bool IsNull() const { return !o; }
};

inline void EndianCheck() {
  int endiantest = 1;
  // If this fails, see FLATBUFFERS_LITTLEENDIAN above.
  FLATBUFFERS_ASSERT(*reinterpret_cast<char *>(&endiantest) ==
                     FLATBUFFERS_LITTLEENDIAN);
  (void)endiantest;
}

template<typename T> FLATBUFFERS_CONSTEXPR size_t AlignOf() {
  // clang-format off
  #ifdef _MSC_VER
    return __alignof(T);
  #else
    #ifndef alignof
      return __alignof__(T);
    #else
      return alignof(T);
    #endif
  #endif
  // clang-format on
}

// Lexicographically compare two strings (possibly containing nulls), and
// return true if the first is less than the second.
static inline bool StringLessThan(const char *a_data, uoffset_t a_size,
                                  const char *b_data, uoffset_t b_size) {
  const auto cmp = memcmp(a_data, b_data, (std::min)(a_size, b_size));
  return cmp == 0 ? a_size < b_size : cmp < 0;
}

// When we read serialized data from memory, in the case of most scalars,
// we want to just read T, but in the case of Offset, we want to actually
// perform the indirection and return a pointer.
// The template specialization below does just that.
// It is wrapped in a struct since function templates can't overload on the
// return type like this.
// The typedef is for the convenience of callers of this function
// (avoiding the need for a trailing return decltype)
template<typename T> struct IndirectHelper {
  typedef T return_type;
  typedef T mutable_return_type;
  static const size_t element_stride = sizeof(T);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    return EndianScalar((reinterpret_cast<const T *>(p))[i]);
  }
  static return_type Read(uint8_t *p, uoffset_t i) {
    return Read(const_cast<const uint8_t *>(p), i);
  }
};
template<typename T> struct IndirectHelper<Offset<T>> {
  typedef const T *return_type;
  typedef T *mutable_return_type;
  static const size_t element_stride = sizeof(uoffset_t);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    p += i * sizeof(uoffset_t);
    return reinterpret_cast<return_type>(p + ReadScalar<uoffset_t>(p));
  }
  static mutable_return_type Read(uint8_t *p, uoffset_t i) {
    p += i * sizeof(uoffset_t);
    return reinterpret_cast<mutable_return_type>(p + ReadScalar<uoffset_t>(p));
  }
};
template<typename T> struct IndirectHelper<const T *> {
  typedef const T *return_type;
  typedef T *mutable_return_type;
  static const size_t element_stride = sizeof(T);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    return reinterpret_cast<return_type>(p + i * sizeof(T));
  }
  static mutable_return_type Read(uint8_t *p, uoffset_t i) {
    return reinterpret_cast<mutable_return_type>(p + i * sizeof(T));
  }
};

/// @brief Get a pointer to the file_identifier section of the buffer.
/// @return Returns a const char pointer to the start of the file_identifier
/// characters in the buffer.  The returned char * has length
/// 'flatbuffers::FlatBufferBuilder::kFileIdentifierLength'.
/// This function is UNDEFINED for FlatBuffers whose schema does not include
/// a file_identifier (likely points at padding or the start of a the root
/// vtable).
inline const char *GetBufferIdentifier(const void *buf,
                                       bool size_prefixed = false) {
  return reinterpret_cast<const char *>(buf) +
         ((size_prefixed) ? 2 * sizeof(uoffset_t) : sizeof(uoffset_t));
}

// Helper to see if the identifier in a buffer has the expected value.
inline bool BufferHasIdentifier(const void *buf, const char *identifier,
                                bool size_prefixed = false) {
  return strncmp(GetBufferIdentifier(buf, size_prefixed), identifier,
                 flatbuffers::kFileIdentifierLength) == 0;
}

/// @cond FLATBUFFERS_INTERNAL
// Helpers to get a typed pointer to the root object contained in the buffer.
template<typename T> T *GetMutableRoot(void *buf) {
  EndianCheck();
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(buf) +
      EndianScalar(*reinterpret_cast<uoffset_t *>(buf)));
}

template<typename T> T *GetMutableSizePrefixedRoot(void *buf) {
  return GetMutableRoot<T>(reinterpret_cast<uint8_t *>(buf) +
                           sizeof(uoffset_t));
}

template<typename T> const T *GetRoot(const void *buf) {
  return GetMutableRoot<T>(const_cast<void *>(buf));
}

template<typename T> const T *GetSizePrefixedRoot(const void *buf) {
  return GetRoot<T>(reinterpret_cast<const uint8_t *>(buf) + sizeof(uoffset_t));
}

}  // namespace flatbuffers

#endif  // FLATBUFFERS_BUFFER_H_
