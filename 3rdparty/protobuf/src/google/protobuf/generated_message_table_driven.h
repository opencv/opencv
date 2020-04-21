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

#ifndef GOOGLE_PROTOBUF_GENERATED_MESSAGE_TABLE_DRIVEN_H__
#define GOOGLE_PROTOBUF_GENERATED_MESSAGE_TABLE_DRIVEN_H__

#include <google/protobuf/map.h>
#include <google/protobuf/map_entry_lite.h>
#include <google/protobuf/map_field_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/wire_format_lite_inl.h>

#if LANG_CXX11
#define PROTOBUF_CONSTEXPR constexpr

// We require C++11 and Clang to use constexpr for variables, as GCC 4.8
// requires constexpr to be consistent between declarations of variables
// unnecessarily (see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58541).
// VS 2017 Update 3 also supports this usage of constexpr.
#if defined(__clang__) || (defined(_MSC_VER) && _MSC_VER >= 1911)
#define PROTOBUF_CONSTEXPR_VAR constexpr
#else  // !__clang__
#define PROTOBUF_CONSTEXPR_VAR
#endif  // !_clang

#else
#define PROTOBUF_CONSTEXPR
#define PROTOBUF_CONSTEXPR_VAR
#endif

namespace google {
namespace protobuf {
namespace internal {

// Processing-type masks.
static PROTOBUF_CONSTEXPR const unsigned char kOneofMask = 0x40;
static PROTOBUF_CONSTEXPR const unsigned char kRepeatedMask = 0x20;
// Mask for the raw type: either a WireFormatLite::FieldType or one of the
// ProcessingTypes below, without the oneof or repeated flag.
static PROTOBUF_CONSTEXPR const unsigned char kTypeMask = 0x1f;

// Wire type masks.
static PROTOBUF_CONSTEXPR const unsigned char kNotPackedMask = 0x10;
static PROTOBUF_CONSTEXPR const unsigned char kInvalidMask = 0x20;

enum ProcessingTypes {
  TYPE_STRING_CORD = 19,
  TYPE_STRING_STRING_PIECE = 20,
  TYPE_BYTES_CORD = 21,
  TYPE_BYTES_STRING_PIECE = 22,
  TYPE_MAP = 23,
};

#if LANG_CXX11
static_assert(TYPE_MAP < kRepeatedMask, "Invalid enum");
#endif

// TODO(ckennelly):  Add a static assertion to ensure that these masks do not
// conflict with wiretypes.

// ParseTableField is kept small to help simplify instructions for computing
// offsets, as we will always need this information to parse a field.
// Additional data, needed for some types, is stored in
// AuxillaryParseTableField.
struct ParseTableField {
  uint32 offset;
  // The presence_index ordinarily represents a has_bit index, but for fields
  // inside a oneof it represents the index in _oneof_case_.
  uint32 presence_index;
  unsigned char normal_wiretype;
  unsigned char packed_wiretype;

  // processing_type is given by:
  //   (FieldDescriptor->type() << 1) | FieldDescriptor->is_packed()
  unsigned char processing_type;

  unsigned char tag_size;
};

struct ParseTable;

union AuxillaryParseTableField {
  typedef bool (*EnumValidator)(int);

  // Enums
  struct enum_aux {
    EnumValidator validator;
  };
  enum_aux enums;
  // Group, messages
  struct message_aux {
    // ExplicitlyInitialized<T> -> T requires a reinterpret_cast, which prevents
    // the tables from being constructed as a constexpr.  We use void to avoid
    // the cast.
    const void* default_message_void;
    const MessageLite* default_message() const {
      return static_cast<const MessageLite*>(default_message_void);
    }
    const ParseTable* parse_table;
  };
  message_aux messages;
  // Strings
  struct string_aux {
    const void* default_ptr;
    const char* field_name;
  };
  string_aux strings;

  struct map_aux {
    bool (*parse_map)(io::CodedInputStream*, void*);
  };
  map_aux maps;

#if LANG_CXX11
  AuxillaryParseTableField() = default;
#else
  AuxillaryParseTableField() { }
#endif
  PROTOBUF_CONSTEXPR AuxillaryParseTableField(
      AuxillaryParseTableField::enum_aux e) : enums(e) {}
  PROTOBUF_CONSTEXPR AuxillaryParseTableField(
      AuxillaryParseTableField::message_aux m) : messages(m) {}
  PROTOBUF_CONSTEXPR AuxillaryParseTableField(
      AuxillaryParseTableField::string_aux s) : strings(s) {}
  PROTOBUF_CONSTEXPR AuxillaryParseTableField(
      AuxillaryParseTableField::map_aux m)
      : maps(m) {}
};

struct ParseTable {
  const ParseTableField* fields;
  const AuxillaryParseTableField* aux;
  int max_field_number;
  // TODO(ckennelly): Do something with this padding.

  // TODO(ckennelly): Vet these for sign extension.
  int64 has_bits_offset;
  int64 oneof_case_offset;
  int64 extension_offset;
  int64 arena_offset;

  // ExplicitlyInitialized<T> -> T requires a reinterpret_cast, which prevents
  // the tables from being constructed as a constexpr.  We use void to avoid
  // the cast.
  const void* default_instance_void;
  const MessageLite* default_instance() const {
    return static_cast<const MessageLite*>(default_instance_void);
  }

  bool unknown_field_set;
};

// TODO(jhen): Remove the __NVCC__ check when we get a version of nvcc that
// supports these checks.
#if LANG_CXX11 && !defined(__NVCC__)
static_assert(sizeof(ParseTableField) <= 16, "ParseTableField is too large");
// The tables must be composed of POD components to ensure link-time
// initialization.
static_assert(std::is_pod<ParseTableField>::value, "");
static_assert(std::is_pod<AuxillaryParseTableField>::value, "");
static_assert(std::is_pod<AuxillaryParseTableField::enum_aux>::value, "");
static_assert(std::is_pod<AuxillaryParseTableField::message_aux>::value, "");
static_assert(std::is_pod<AuxillaryParseTableField::string_aux>::value, "");
static_assert(std::is_pod<ParseTable>::value, "");
#endif

// TODO(ckennelly): Consolidate these implementations into a single one, using
// dynamic dispatch to the appropriate unknown field handler.
bool MergePartialFromCodedStream(MessageLite* msg, const ParseTable& table,
                                 io::CodedInputStream* input);
bool MergePartialFromCodedStreamLite(MessageLite* msg, const ParseTable& table,
                                 io::CodedInputStream* input);

template <typename Entry>
bool ParseMap(io::CodedInputStream* input, void* map_field) {
  typedef typename MapEntryToMapField<Entry>::MapFieldType MapFieldType;
  typedef google::protobuf::Map<typename Entry::EntryKeyType,
                      typename Entry::EntryValueType>
      MapType;
  typedef typename Entry::template Parser<MapFieldType, MapType> ParserType;

  ParserType parser(static_cast<MapFieldType*>(map_field));
  return ::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(input,
                                                                  &parser);
}

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_GENERATED_MESSAGE_TABLE_DRIVEN_H__
