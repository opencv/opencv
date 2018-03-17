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

// Author: kenton@google.com (Kenton Varda)
//         atenasio@google.com (Chris Atenasio) (ZigZag transform)
//         wink@google.com (Wink Saville) (refactored from wire_format.h)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// This header is logically internal, but is made public because it is used
// from protocol-compiler-generated code, which may reside in other components.

#ifndef GOOGLE_PROTOBUF_WIRE_FORMAT_LITE_H__
#define GOOGLE_PROTOBUF_WIRE_FORMAT_LITE_H__

#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/repeated_field.h>

// Do UTF-8 validation on string type in Debug build only
#ifndef NDEBUG
#define GOOGLE_PROTOBUF_UTF8_VALIDATION_ENABLED
#endif

// Avoid conflict with iOS where <ConditionalMacros.h> #defines TYPE_BOOL.
//
// If some one needs the macro TYPE_BOOL in a file that includes this header, it's
// possible to bring it back using push/pop_macro as follows.
//
// #pragma push_macro("TYPE_BOOL")
// #include this header and/or all headers that need the macro to be undefined.
// #pragma pop_macro("TYPE_BOOL")
#undef TYPE_BOOL

namespace google {

namespace protobuf {
  template <typename T> class RepeatedField;  // repeated_field.h
}

namespace protobuf {
namespace internal {

class StringPieceField;

// This class is for internal use by the protocol buffer library and by
// protocol-complier-generated message classes.  It must not be called
// directly by clients.
//
// This class contains helpers for implementing the binary protocol buffer
// wire format without the need for reflection. Use WireFormat when using
// reflection.
//
// This class is really a namespace that contains only static methods.
class LIBPROTOBUF_EXPORT WireFormatLite {
 public:

  // -----------------------------------------------------------------
  // Helper constants and functions related to the format.  These are
  // mostly meant for internal and generated code to use.

  // The wire format is composed of a sequence of tag/value pairs, each
  // of which contains the value of one field (or one element of a repeated
  // field).  Each tag is encoded as a varint.  The lower bits of the tag
  // identify its wire type, which specifies the format of the data to follow.
  // The rest of the bits contain the field number.  Each type of field (as
  // declared by FieldDescriptor::Type, in descriptor.h) maps to one of
  // these wire types.  Immediately following each tag is the field's value,
  // encoded in the format specified by the wire type.  Because the tag
  // identifies the encoding of this data, it is possible to skip
  // unrecognized fields for forwards compatibility.

  enum WireType {
    WIRETYPE_VARINT           = 0,
    WIRETYPE_FIXED64          = 1,
    WIRETYPE_LENGTH_DELIMITED = 2,
    WIRETYPE_START_GROUP      = 3,
    WIRETYPE_END_GROUP        = 4,
    WIRETYPE_FIXED32          = 5,
  };

  // Lite alternative to FieldDescriptor::Type.  Must be kept in sync.
  enum FieldType {
    TYPE_DOUBLE         = 1,
    TYPE_FLOAT          = 2,
    TYPE_INT64          = 3,
    TYPE_UINT64         = 4,
    TYPE_INT32          = 5,
    TYPE_FIXED64        = 6,
    TYPE_FIXED32        = 7,
    TYPE_BOOL           = 8,
    TYPE_STRING         = 9,
    TYPE_GROUP          = 10,
    TYPE_MESSAGE        = 11,
    TYPE_BYTES          = 12,
    TYPE_UINT32         = 13,
    TYPE_ENUM           = 14,
    TYPE_SFIXED32       = 15,
    TYPE_SFIXED64       = 16,
    TYPE_SINT32         = 17,
    TYPE_SINT64         = 18,
    MAX_FIELD_TYPE      = 18,
  };

  // Lite alternative to FieldDescriptor::CppType.  Must be kept in sync.
  enum CppType {
    CPPTYPE_INT32       = 1,
    CPPTYPE_INT64       = 2,
    CPPTYPE_UINT32      = 3,
    CPPTYPE_UINT64      = 4,
    CPPTYPE_DOUBLE      = 5,
    CPPTYPE_FLOAT       = 6,
    CPPTYPE_BOOL        = 7,
    CPPTYPE_ENUM        = 8,
    CPPTYPE_STRING      = 9,
    CPPTYPE_MESSAGE     = 10,
    MAX_CPPTYPE         = 10,
  };

  // Helper method to get the CppType for a particular Type.
  static CppType FieldTypeToCppType(FieldType type);

  // Given a FieldDescriptor::Type return its WireType
  static inline WireFormatLite::WireType WireTypeForFieldType(
      WireFormatLite::FieldType type) {
    return kWireTypeForFieldType[type];
  }

  // Number of bits in a tag which identify the wire type.
  static const int kTagTypeBits = 3;
  // Mask for those bits.
  static const uint32 kTagTypeMask = (1 << kTagTypeBits) - 1;

  // Helper functions for encoding and decoding tags.  (Inlined below and in
  // _inl.h)
  //
  // This is different from MakeTag(field->number(), field->type()) in the case
  // of packed repeated fields.
  static uint32 MakeTag(int field_number, WireType type);
  static WireType GetTagWireType(uint32 tag);
  static int GetTagFieldNumber(uint32 tag);

  // Compute the byte size of a tag.  For groups, this includes both the start
  // and end tags.
  static inline size_t TagSize(int field_number,
                               WireFormatLite::FieldType type);

  // Skips a field value with the given tag.  The input should start
  // positioned immediately after the tag.  Skipped values are simply discarded,
  // not recorded anywhere.  See WireFormat::SkipField() for a version that
  // records to an UnknownFieldSet.
  static bool SkipField(io::CodedInputStream* input, uint32 tag);

  // Skips a field value with the given tag.  The input should start
  // positioned immediately after the tag. Skipped values are recorded to a
  // CodedOutputStream.
  static bool SkipField(io::CodedInputStream* input, uint32 tag,
                        io::CodedOutputStream* output);

  // Reads and ignores a message from the input.  Skipped values are simply
  // discarded, not recorded anywhere.  See WireFormat::SkipMessage() for a
  // version that records to an UnknownFieldSet.
  static bool SkipMessage(io::CodedInputStream* input);

  // Reads and ignores a message from the input.  Skipped values are recorded
  // to a CodedOutputStream.
  static bool SkipMessage(io::CodedInputStream* input,
                          io::CodedOutputStream* output);

// This macro does the same thing as WireFormatLite::MakeTag(), but the
// result is usable as a compile-time constant, which makes it usable
// as a switch case or a template input.  WireFormatLite::MakeTag() is more
// type-safe, though, so prefer it if possible.
#define GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(FIELD_NUMBER, TYPE)                  \
  static_cast<uint32>(                                                   \
    (static_cast<uint32>(FIELD_NUMBER) << ::google::protobuf::internal::WireFormatLite::kTagTypeBits) \
      | (TYPE))

  // These are the tags for the old MessageSet format, which was defined as:
  //   message MessageSet {
  //     repeated group Item = 1 {
  //       required int32 type_id = 2;
  //       required string message = 3;
  //     }
  //   }
  static const int kMessageSetItemNumber = 1;
  static const int kMessageSetTypeIdNumber = 2;
  static const int kMessageSetMessageNumber = 3;
  static const int kMessageSetItemStartTag =
    GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetItemNumber,
                                WireFormatLite::WIRETYPE_START_GROUP);
  static const int kMessageSetItemEndTag =
    GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetItemNumber,
                                WireFormatLite::WIRETYPE_END_GROUP);
  static const int kMessageSetTypeIdTag =
    GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetTypeIdNumber,
                                WireFormatLite::WIRETYPE_VARINT);
  static const int kMessageSetMessageTag =
    GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetMessageNumber,
                                WireFormatLite::WIRETYPE_LENGTH_DELIMITED);

  // Byte size of all tags of a MessageSet::Item combined.
  static const size_t kMessageSetItemTagsSize;

  // Helper functions for converting between floats/doubles and IEEE-754
  // uint32s/uint64s so that they can be written.  (Assumes your platform
  // uses IEEE-754 floats.)
  static uint32 EncodeFloat(float value);
  static float DecodeFloat(uint32 value);
  static uint64 EncodeDouble(double value);
  static double DecodeDouble(uint64 value);

  // Helper functions for mapping signed integers to unsigned integers in
  // such a way that numbers with small magnitudes will encode to smaller
  // varints.  If you simply static_cast a negative number to an unsigned
  // number and varint-encode it, it will always take 10 bytes, defeating
  // the purpose of varint.  So, for the "sint32" and "sint64" field types,
  // we ZigZag-encode the values.
  static uint32 ZigZagEncode32(int32 n);
  static int32  ZigZagDecode32(uint32 n);
  static uint64 ZigZagEncode64(int64 n);
  static int64  ZigZagDecode64(uint64 n);

  // =================================================================
  // Methods for reading/writing individual field.  The implementations
  // of these methods are defined in wire_format_lite_inl.h; you must #include
  // that file to use these.

#ifdef NDEBUG
#define INL GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
#else
// Avoid excessive inlining in non-optimized builds. Without other optimizations
// the inlining is not going to provide benefits anyway and the huge resulting
// functions, especially in the proto-generated serialization functions, produce
// stack frames so large that many tests run into stack overflows (b/32192897).
#define INL
#endif

  // Read fields, not including tags.  The assumption is that you already
  // read the tag to determine what field to read.

  // For primitive fields, we just use a templatized routine parameterized by
  // the represented type and the FieldType. These are specialized with the
  // appropriate definition for each declared type.
  template <typename CType, enum FieldType DeclaredType>
  INL static bool ReadPrimitive(io::CodedInputStream* input, CType* value);

  // Reads repeated primitive values, with optimizations for repeats.
  // tag_size and tag should both be compile-time constants provided by the
  // protocol compiler.
  template <typename CType, enum FieldType DeclaredType>
  INL static bool ReadRepeatedPrimitive(int tag_size, uint32 tag,
                                        io::CodedInputStream* input,
                                        RepeatedField<CType>* value);

  // Identical to ReadRepeatedPrimitive, except will not inline the
  // implementation.
  template <typename CType, enum FieldType DeclaredType>
  static bool ReadRepeatedPrimitiveNoInline(int tag_size, uint32 tag,
                                            io::CodedInputStream* input,
                                            RepeatedField<CType>* value);

  // Reads a primitive value directly from the provided buffer. It returns a
  // pointer past the segment of data that was read.
  //
  // This is only implemented for the types with fixed wire size, e.g.
  // float, double, and the (s)fixed* types.
  template <typename CType, enum FieldType DeclaredType> INL
  static const uint8* ReadPrimitiveFromArray(const uint8* buffer, CType* value);

  // Reads a primitive packed field.
  //
  // This is only implemented for packable types.
  template <typename CType, enum FieldType DeclaredType>
  INL static bool ReadPackedPrimitive(io::CodedInputStream* input,
                                      RepeatedField<CType>* value);

  // Identical to ReadPackedPrimitive, except will not inline the
  // implementation.
  template <typename CType, enum FieldType DeclaredType>
  static bool ReadPackedPrimitiveNoInline(io::CodedInputStream* input,
                                          RepeatedField<CType>* value);

  // Read a packed enum field. If the is_valid function is not NULL, values for
  // which is_valid(value) returns false are silently dropped.
  static bool ReadPackedEnumNoInline(io::CodedInputStream* input,
                                     bool (*is_valid)(int),
                                     RepeatedField<int>* values);

  // Read a packed enum field. If the is_valid function is not NULL, values for
  // which is_valid(value) returns false are appended to unknown_fields_stream.
  static bool ReadPackedEnumPreserveUnknowns(
      io::CodedInputStream* input, int field_number, bool (*is_valid)(int),
      io::CodedOutputStream* unknown_fields_stream, RepeatedField<int>* values);

  // Read a string.  ReadString(..., string* value) requires an existing string.
  static inline bool ReadString(io::CodedInputStream* input, string* value);
  // ReadString(..., string** p) is internal-only, and should only be called
  // from generated code. It starts by setting *p to "new string"
  // if *p == &GetEmptyStringAlreadyInited().  It then invokes
  // ReadString(io::CodedInputStream* input, *p).  This is useful for reducing
  // code size.
  static inline bool ReadString(io::CodedInputStream* input, string** p);
  // Analogous to ReadString().
  static bool ReadBytes(io::CodedInputStream* input, string* value);
  static bool ReadBytes(io::CodedInputStream* input, string** p);

  enum Operation {
    PARSE = 0,
    SERIALIZE = 1,
  };

  // Returns true if the data is valid UTF-8.
  static bool VerifyUtf8String(const char* data, int size,
                               Operation op,
                               const char* field_name);

  static inline bool ReadGroup(int field_number, io::CodedInputStream* input,
                               MessageLite* value);
  static inline bool ReadMessage(io::CodedInputStream* input,
                                 MessageLite* value);

  // Like above, but de-virtualize the call to MergePartialFromCodedStream().
  // The pointer must point at an instance of MessageType, *not* a subclass (or
  // the subclass must not override MergePartialFromCodedStream()).
  template <typename MessageType>
  static inline bool ReadGroupNoVirtual(int field_number,
                                        io::CodedInputStream* input,
                                        MessageType* value);
  template<typename MessageType>
  static inline bool ReadMessageNoVirtual(io::CodedInputStream* input,
                                          MessageType* value);

  // The same, but do not modify input's recursion depth.  This is useful
  // when reading a bunch of groups or messages in a loop, because then the
  // recursion depth can be incremented before the loop and decremented after.
  template<typename MessageType>
  static inline bool ReadGroupNoVirtualNoRecursionDepth(
      int field_number, io::CodedInputStream* input, MessageType* value);

  template<typename MessageType>
  static inline bool ReadMessageNoVirtualNoRecursionDepth(
      io::CodedInputStream* input, MessageType* value);

  // Write a tag.  The Write*() functions typically include the tag, so
  // normally there's no need to call this unless using the Write*NoTag()
  // variants.
  INL static void WriteTag(int field_number, WireType type,
                           io::CodedOutputStream* output);

  // Write fields, without tags.
  INL static void WriteInt32NoTag(int32 value, io::CodedOutputStream* output);
  INL static void WriteInt64NoTag(int64 value, io::CodedOutputStream* output);
  INL static void WriteUInt32NoTag(uint32 value, io::CodedOutputStream* output);
  INL static void WriteUInt64NoTag(uint64 value, io::CodedOutputStream* output);
  INL static void WriteSInt32NoTag(int32 value, io::CodedOutputStream* output);
  INL static void WriteSInt64NoTag(int64 value, io::CodedOutputStream* output);
  INL static void WriteFixed32NoTag(uint32 value,
                                    io::CodedOutputStream* output);
  INL static void WriteFixed64NoTag(uint64 value,
                                    io::CodedOutputStream* output);
  INL static void WriteSFixed32NoTag(int32 value,
                                     io::CodedOutputStream* output);
  INL static void WriteSFixed64NoTag(int64 value,
                                     io::CodedOutputStream* output);
  INL static void WriteFloatNoTag(float value, io::CodedOutputStream* output);
  INL static void WriteDoubleNoTag(double value, io::CodedOutputStream* output);
  INL static void WriteBoolNoTag(bool value, io::CodedOutputStream* output);
  INL static void WriteEnumNoTag(int value, io::CodedOutputStream* output);

  // Write array of primitive fields, without tags
  static void WriteFloatArray(const float* a, int n,
                              io::CodedOutputStream* output);
  static void WriteDoubleArray(const double* a, int n,
                               io::CodedOutputStream* output);
  static void WriteFixed32Array(const uint32* a, int n,
                                io::CodedOutputStream* output);
  static void WriteFixed64Array(const uint64* a, int n,
                                io::CodedOutputStream* output);
  static void WriteSFixed32Array(const int32* a, int n,
                                 io::CodedOutputStream* output);
  static void WriteSFixed64Array(const int64* a, int n,
                                 io::CodedOutputStream* output);
  static void WriteBoolArray(const bool* a, int n,
                             io::CodedOutputStream* output);

  // Write fields, including tags.
  static void WriteInt32(int field_number, int32 value,
                         io::CodedOutputStream* output);
  static void WriteInt64(int field_number, int64 value,
                         io::CodedOutputStream* output);
  static void WriteUInt32(int field_number, uint32 value,
                          io::CodedOutputStream* output);
  static void WriteUInt64(int field_number, uint64 value,
                          io::CodedOutputStream* output);
  static void WriteSInt32(int field_number, int32 value,
                          io::CodedOutputStream* output);
  static void WriteSInt64(int field_number, int64 value,
                          io::CodedOutputStream* output);
  static void WriteFixed32(int field_number, uint32 value,
                           io::CodedOutputStream* output);
  static void WriteFixed64(int field_number, uint64 value,
                           io::CodedOutputStream* output);
  static void WriteSFixed32(int field_number, int32 value,
                            io::CodedOutputStream* output);
  static void WriteSFixed64(int field_number, int64 value,
                            io::CodedOutputStream* output);
  static void WriteFloat(int field_number, float value,
                         io::CodedOutputStream* output);
  static void WriteDouble(int field_number, double value,
                          io::CodedOutputStream* output);
  static void WriteBool(int field_number, bool value,
                        io::CodedOutputStream* output);
  static void WriteEnum(int field_number, int value,
                        io::CodedOutputStream* output);

  static void WriteString(int field_number, const string& value,
                          io::CodedOutputStream* output);
  static void WriteBytes(int field_number, const string& value,
                         io::CodedOutputStream* output);
  static void WriteStringMaybeAliased(int field_number, const string& value,
                                      io::CodedOutputStream* output);
  static void WriteBytesMaybeAliased(int field_number, const string& value,
                                     io::CodedOutputStream* output);

  static void WriteGroup(int field_number, const MessageLite& value,
                         io::CodedOutputStream* output);
  static void WriteMessage(int field_number, const MessageLite& value,
                           io::CodedOutputStream* output);
  // Like above, but these will check if the output stream has enough
  // space to write directly to a flat array.
  static void WriteGroupMaybeToArray(int field_number, const MessageLite& value,
                                     io::CodedOutputStream* output);
  static void WriteMessageMaybeToArray(int field_number,
                                       const MessageLite& value,
                                       io::CodedOutputStream* output);

  // Like above, but de-virtualize the call to SerializeWithCachedSizes().  The
  // pointer must point at an instance of MessageType, *not* a subclass (or
  // the subclass must not override SerializeWithCachedSizes()).
  template <typename MessageType>
  static inline void WriteGroupNoVirtual(int field_number,
                                         const MessageType& value,
                                         io::CodedOutputStream* output);
  template <typename MessageType>
  static inline void WriteMessageNoVirtual(int field_number,
                                           const MessageType& value,
                                           io::CodedOutputStream* output);

  // Like above, but use only *ToArray methods of CodedOutputStream.
  INL static uint8* WriteTagToArray(int field_number, WireType type,
                                    uint8* target);

  // Write fields, without tags.
  INL static uint8* WriteInt32NoTagToArray(int32 value, uint8* target);
  INL static uint8* WriteInt64NoTagToArray(int64 value, uint8* target);
  INL static uint8* WriteUInt32NoTagToArray(uint32 value, uint8* target);
  INL static uint8* WriteUInt64NoTagToArray(uint64 value, uint8* target);
  INL static uint8* WriteSInt32NoTagToArray(int32 value, uint8* target);
  INL static uint8* WriteSInt64NoTagToArray(int64 value, uint8* target);
  INL static uint8* WriteFixed32NoTagToArray(uint32 value, uint8* target);
  INL static uint8* WriteFixed64NoTagToArray(uint64 value, uint8* target);
  INL static uint8* WriteSFixed32NoTagToArray(int32 value, uint8* target);
  INL static uint8* WriteSFixed64NoTagToArray(int64 value, uint8* target);
  INL static uint8* WriteFloatNoTagToArray(float value, uint8* target);
  INL static uint8* WriteDoubleNoTagToArray(double value, uint8* target);
  INL static uint8* WriteBoolNoTagToArray(bool value, uint8* target);
  INL static uint8* WriteEnumNoTagToArray(int value, uint8* target);

  // Write fields, without tags.  These require that value.size() > 0.
  template<typename T>
  INL static uint8* WritePrimitiveNoTagToArray(
      const RepeatedField<T>& value,
      uint8* (*Writer)(T, uint8*), uint8* target);
  template<typename T>
  INL static uint8* WriteFixedNoTagToArray(
      const RepeatedField<T>& value,
      uint8* (*Writer)(T, uint8*), uint8* target);

  INL static uint8* WriteInt32NoTagToArray(
      const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteInt64NoTagToArray(
      const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteUInt32NoTagToArray(
      const RepeatedField<uint32>& value, uint8* output);
  INL static uint8* WriteUInt64NoTagToArray(
      const RepeatedField<uint64>& value, uint8* output);
  INL static uint8* WriteSInt32NoTagToArray(
      const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteSInt64NoTagToArray(
      const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteFixed32NoTagToArray(
      const RepeatedField<uint32>& value, uint8* output);
  INL static uint8* WriteFixed64NoTagToArray(
      const RepeatedField<uint64>& value, uint8* output);
  INL static uint8* WriteSFixed32NoTagToArray(
      const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteSFixed64NoTagToArray(
      const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteFloatNoTagToArray(
      const RepeatedField< float>& value, uint8* output);
  INL static uint8* WriteDoubleNoTagToArray(
      const RepeatedField<double>& value, uint8* output);
  INL static uint8* WriteBoolNoTagToArray(
      const RepeatedField<  bool>& value, uint8* output);
  INL static uint8* WriteEnumNoTagToArray(
      const RepeatedField<   int>& value, uint8* output);

  // Write fields, including tags.
  INL static uint8* WriteInt32ToArray(int field_number, int32 value,
                                      uint8* target);
  INL static uint8* WriteInt64ToArray(int field_number, int64 value,
                                      uint8* target);
  INL static uint8* WriteUInt32ToArray(int field_number, uint32 value,
                                       uint8* target);
  INL static uint8* WriteUInt64ToArray(int field_number, uint64 value,
                                       uint8* target);
  INL static uint8* WriteSInt32ToArray(int field_number, int32 value,
                                       uint8* target);
  INL static uint8* WriteSInt64ToArray(int field_number, int64 value,
                                       uint8* target);
  INL static uint8* WriteFixed32ToArray(int field_number, uint32 value,
                                        uint8* target);
  INL static uint8* WriteFixed64ToArray(int field_number, uint64 value,
                                        uint8* target);
  INL static uint8* WriteSFixed32ToArray(int field_number, int32 value,
                                         uint8* target);
  INL static uint8* WriteSFixed64ToArray(int field_number, int64 value,
                                         uint8* target);
  INL static uint8* WriteFloatToArray(int field_number, float value,
                                      uint8* target);
  INL static uint8* WriteDoubleToArray(int field_number, double value,
                                       uint8* target);
  INL static uint8* WriteBoolToArray(int field_number, bool value,
                                     uint8* target);
  INL static uint8* WriteEnumToArray(int field_number, int value,
                                     uint8* target);

  template<typename T>
  INL static uint8* WritePrimitiveToArray(
      int field_number,
      const RepeatedField<T>& value,
      uint8* (*Writer)(int, T, uint8*), uint8* target);

  INL static uint8* WriteInt32ToArray(
      int field_number, const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteInt64ToArray(
      int field_number, const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteUInt32ToArray(
      int field_number, const RepeatedField<uint32>& value, uint8* output);
  INL static uint8* WriteUInt64ToArray(
      int field_number, const RepeatedField<uint64>& value, uint8* output);
  INL static uint8* WriteSInt32ToArray(
      int field_number, const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteSInt64ToArray(
      int field_number, const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteFixed32ToArray(
      int field_number, const RepeatedField<uint32>& value, uint8* output);
  INL static uint8* WriteFixed64ToArray(
      int field_number, const RepeatedField<uint64>& value, uint8* output);
  INL static uint8* WriteSFixed32ToArray(
      int field_number, const RepeatedField< int32>& value, uint8* output);
  INL static uint8* WriteSFixed64ToArray(
      int field_number, const RepeatedField< int64>& value, uint8* output);
  INL static uint8* WriteFloatToArray(
      int field_number, const RepeatedField< float>& value, uint8* output);
  INL static uint8* WriteDoubleToArray(
      int field_number, const RepeatedField<double>& value, uint8* output);
  INL static uint8* WriteBoolToArray(
      int field_number, const RepeatedField<  bool>& value, uint8* output);
  INL static uint8* WriteEnumToArray(
      int field_number, const RepeatedField<   int>& value, uint8* output);

  INL static uint8* WriteStringToArray(int field_number, const string& value,
                                       uint8* target);
  INL static uint8* WriteBytesToArray(int field_number, const string& value,
                                      uint8* target);

  // Whether to serialize deterministically (e.g., map keys are
  // sorted) is a property of a CodedOutputStream, and in the process
  // of serialization, the "ToArray" variants may be invoked.  But they don't
  // have a CodedOutputStream available, so they get an additional parameter
  // telling them whether to serialize deterministically.
  INL static uint8* InternalWriteGroupToArray(int field_number,
                                              const MessageLite& value,
                                              bool deterministic,
                                              uint8* target);
  INL static uint8* InternalWriteMessageToArray(int field_number,
                                                const MessageLite& value,
                                                bool deterministic,
                                                uint8* target);

  // Like above, but de-virtualize the call to SerializeWithCachedSizes().  The
  // pointer must point at an instance of MessageType, *not* a subclass (or
  // the subclass must not override SerializeWithCachedSizes()).
  template <typename MessageType>
  INL static uint8* InternalWriteGroupNoVirtualToArray(int field_number,
                                                       const MessageType& value,
                                                       bool deterministic,
                                                       uint8* target);
  template <typename MessageType>
  INL static uint8* InternalWriteMessageNoVirtualToArray(
      int field_number, const MessageType& value, bool deterministic,
      uint8* target);

  // For backward-compatibility, the last four methods also have versions
  // that are non-deterministic always.
  INL static uint8* WriteGroupToArray(int field_number,
                                      const MessageLite& value, uint8* target) {
    return InternalWriteGroupToArray(field_number, value, false, target);
  }
  INL static uint8* WriteMessageToArray(int field_number,
                                        const MessageLite& value,
                                        uint8* target) {
    return InternalWriteMessageToArray(field_number, value, false, target);
  }
  template <typename MessageType>
  INL static uint8* WriteGroupNoVirtualToArray(int field_number,
                                               const MessageType& value,
                                               uint8* target) {
    return InternalWriteGroupNoVirtualToArray(field_number, value, false,
                                              target);
  }
  template <typename MessageType>
  INL static uint8* WriteMessageNoVirtualToArray(int field_number,
                                                 const MessageType& value,
                                                 uint8* target) {
    return InternalWriteMessageNoVirtualToArray(field_number, value, false,
                                                target);
  }

#undef INL

  // Compute the byte size of a field.  The XxSize() functions do NOT include
  // the tag, so you must also call TagSize().  (This is because, for repeated
  // fields, you should only call TagSize() once and multiply it by the element
  // count, but you may have to call XxSize() for each individual element.)
  static inline size_t Int32Size   ( int32 value);
  static inline size_t Int64Size   ( int64 value);
  static inline size_t UInt32Size  (uint32 value);
  static inline size_t UInt64Size  (uint64 value);
  static inline size_t SInt32Size  ( int32 value);
  static inline size_t SInt64Size  ( int64 value);
  static inline size_t EnumSize    (   int value);

  static        size_t Int32Size (const RepeatedField< int32>& value);
  static inline size_t Int64Size (const RepeatedField< int64>& value);
  static        size_t UInt32Size(const RepeatedField<uint32>& value);
  static inline size_t UInt64Size(const RepeatedField<uint64>& value);
  static        size_t SInt32Size(const RepeatedField< int32>& value);
  static inline size_t SInt64Size(const RepeatedField< int64>& value);
  static        size_t EnumSize  (const RepeatedField<   int>& value);

  // These types always have the same size.
  static const size_t kFixed32Size  = 4;
  static const size_t kFixed64Size  = 8;
  static const size_t kSFixed32Size = 4;
  static const size_t kSFixed64Size = 8;
  static const size_t kFloatSize    = 4;
  static const size_t kDoubleSize   = 8;
  static const size_t kBoolSize     = 1;

  static inline size_t StringSize(const string& value);
  static inline size_t BytesSize (const string& value);

  static inline size_t GroupSize  (const MessageLite& value);
  static inline size_t MessageSize(const MessageLite& value);

  // Like above, but de-virtualize the call to ByteSize().  The
  // pointer must point at an instance of MessageType, *not* a subclass (or
  // the subclass must not override ByteSize()).
  template<typename MessageType>
  static inline size_t GroupSizeNoVirtual  (const MessageType& value);
  template<typename MessageType>
  static inline size_t MessageSizeNoVirtual(const MessageType& value);

  // Given the length of data, calculate the byte size of the data on the
  // wire if we encode the data as a length delimited field.
  static inline size_t LengthDelimitedSize(size_t length);

 private:
  // A helper method for the repeated primitive reader. This method has
  // optimizations for primitive types that have fixed size on the wire, and
  // can be read using potentially faster paths.
  template <typename CType, enum FieldType DeclaredType>
  GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
  static bool ReadRepeatedFixedSizePrimitive(
      int tag_size,
      uint32 tag,
      google::protobuf::io::CodedInputStream* input,
      RepeatedField<CType>* value);

  // Like ReadRepeatedFixedSizePrimitive but for packed primitive fields.
  template <typename CType, enum FieldType DeclaredType>
  GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
  static bool ReadPackedFixedSizePrimitive(
      google::protobuf::io::CodedInputStream* input, RepeatedField<CType>* value);

  static const CppType kFieldTypeToCppTypeMap[];
  static const WireFormatLite::WireType kWireTypeForFieldType[];

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(WireFormatLite);
};

// A class which deals with unknown values.  The default implementation just
// discards them.  WireFormat defines a subclass which writes to an
// UnknownFieldSet.  This class is used by ExtensionSet::ParseField(), since
// ExtensionSet is part of the lite library but UnknownFieldSet is not.
class LIBPROTOBUF_EXPORT FieldSkipper {
 public:
  FieldSkipper() {}
  virtual ~FieldSkipper() {}

  // Skip a field whose tag has already been consumed.
  virtual bool SkipField(io::CodedInputStream* input, uint32 tag);

  // Skip an entire message or group, up to an end-group tag (which is consumed)
  // or end-of-stream.
  virtual bool SkipMessage(io::CodedInputStream* input);

  // Deal with an already-parsed unrecognized enum value.  The default
  // implementation does nothing, but the UnknownFieldSet-based implementation
  // saves it as an unknown varint.
  virtual void SkipUnknownEnum(int field_number, int value);
};

// Subclass of FieldSkipper which saves skipped fields to a CodedOutputStream.

class LIBPROTOBUF_EXPORT CodedOutputStreamFieldSkipper : public FieldSkipper {
 public:
  explicit CodedOutputStreamFieldSkipper(io::CodedOutputStream* unknown_fields)
      : unknown_fields_(unknown_fields) {}
  virtual ~CodedOutputStreamFieldSkipper() {}

  // implements FieldSkipper -----------------------------------------
  virtual bool SkipField(io::CodedInputStream* input, uint32 tag);
  virtual bool SkipMessage(io::CodedInputStream* input);
  virtual void SkipUnknownEnum(int field_number, int value);

 protected:
  io::CodedOutputStream* unknown_fields_;
};


// inline methods ====================================================

inline WireFormatLite::CppType
WireFormatLite::FieldTypeToCppType(FieldType type) {
  return kFieldTypeToCppTypeMap[type];
}

inline uint32 WireFormatLite::MakeTag(int field_number, WireType type) {
  return GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(field_number, type);
}

inline WireFormatLite::WireType WireFormatLite::GetTagWireType(uint32 tag) {
  return static_cast<WireType>(tag & kTagTypeMask);
}

inline int WireFormatLite::GetTagFieldNumber(uint32 tag) {
  return static_cast<int>(tag >> kTagTypeBits);
}

inline size_t WireFormatLite::TagSize(int field_number,
                                      WireFormatLite::FieldType type) {
  size_t result = io::CodedOutputStream::VarintSize32(
    static_cast<uint32>(field_number << kTagTypeBits));
  if (type == TYPE_GROUP) {
    // Groups have both a start and an end tag.
    return result * 2;
  } else {
    return result;
  }
}

inline uint32 WireFormatLite::EncodeFloat(float value) {
  union {float f; uint32 i;};
  f = value;
  return i;
}

inline float WireFormatLite::DecodeFloat(uint32 value) {
  union {float f; uint32 i;};
  i = value;
  return f;
}

inline uint64 WireFormatLite::EncodeDouble(double value) {
  union {double f; uint64 i;};
  f = value;
  return i;
}

inline double WireFormatLite::DecodeDouble(uint64 value) {
  union {double f; uint64 i;};
  i = value;
  return f;
}

// ZigZag Transform:  Encodes signed integers so that they can be
// effectively used with varint encoding.
//
// varint operates on unsigned integers, encoding smaller numbers into
// fewer bytes.  If you try to use it on a signed integer, it will treat
// this number as a very large unsigned integer, which means that even
// small signed numbers like -1 will take the maximum number of bytes
// (10) to encode.  ZigZagEncode() maps signed integers to unsigned
// in such a way that those with a small absolute value will have smaller
// encoded values, making them appropriate for encoding using varint.
//
//       int32 ->     uint32
// -------------------------
//           0 ->          0
//          -1 ->          1
//           1 ->          2
//          -2 ->          3
//         ... ->        ...
//  2147483647 -> 4294967294
// -2147483648 -> 4294967295
//
//        >> encode >>
//        << decode <<

inline uint32 WireFormatLite::ZigZagEncode32(int32 n) {
  // Note:  the right-shift must be arithmetic
  // Note:  left shift must be unsigned because of overflow
  return (static_cast<uint32>(n) << 1) ^ static_cast<uint32>(n >> 31);
}

inline int32 WireFormatLite::ZigZagDecode32(uint32 n) {
  // Note:  Using unsigned types prevent undefined behavior
  return static_cast<int32>((n >> 1) ^ -(n & 1));
}

inline uint64 WireFormatLite::ZigZagEncode64(int64 n) {
  // Note:  the right-shift must be arithmetic
  // Note:  left shift must be unsigned because of overflow
  return (static_cast<uint64>(n) << 1) ^ static_cast<uint64>(n >> 63);
}

inline int64 WireFormatLite::ZigZagDecode64(uint64 n) {
  // Note:  Using unsigned types prevent undefined behavior
  return static_cast<int64>((n >> 1) ^ -(n & 1));
}

// String is for UTF-8 text only, but, even so, ReadString() can simply
// call ReadBytes().

inline bool WireFormatLite::ReadString(io::CodedInputStream* input,
                                       string* value) {
  return ReadBytes(input, value);
}

inline bool WireFormatLite::ReadString(io::CodedInputStream* input,
                                       string** p) {
  return ReadBytes(input, p);
}

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_WIRE_FORMAT_LITE_H__
