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
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/extension_set.h>

#include <tuple>
#include <unordered_set>
#include <utility>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/extension_set_inl.h>
#include <google/protobuf/parse_context.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/map_util.h>
#include <google/protobuf/stubs/hash.h>

// clang-format off
#include <google/protobuf/port_def.inc>  // must be last.
// clang-format on
namespace google {
namespace protobuf {
namespace internal {

namespace {

inline WireFormatLite::FieldType real_type(FieldType type) {
  GOOGLE_DCHECK(type > 0 && type <= WireFormatLite::MAX_FIELD_TYPE);
  return static_cast<WireFormatLite::FieldType>(type);
}

inline WireFormatLite::CppType cpp_type(FieldType type) {
  return WireFormatLite::FieldTypeToCppType(real_type(type));
}

inline bool is_packable(WireFormatLite::WireType type) {
  switch (type) {
    case WireFormatLite::WIRETYPE_VARINT:
    case WireFormatLite::WIRETYPE_FIXED64:
    case WireFormatLite::WIRETYPE_FIXED32:
      return true;
    case WireFormatLite::WIRETYPE_LENGTH_DELIMITED:
    case WireFormatLite::WIRETYPE_START_GROUP:
    case WireFormatLite::WIRETYPE_END_GROUP:
      return false;

      // Do not add a default statement. Let the compiler complain when someone
      // adds a new wire type.
  }
  GOOGLE_LOG(FATAL) << "can't reach here.";
  return false;
}

// Registry stuff.

// Note that we cannot use hetererogeneous lookup for std containers since we
// need to support C++11.
struct ExtensionEq {
  bool operator()(const ExtensionInfo& lhs, const ExtensionInfo& rhs) const {
    return lhs.message == rhs.message && lhs.number == rhs.number;
  }
};

struct ExtensionHasher {
  std::size_t operator()(const ExtensionInfo& info) const {
    return std::hash<const MessageLite*>{}(info.message) ^
           std::hash<int>{}(info.number);
  }
};

using ExtensionRegistry =
    std::unordered_set<ExtensionInfo, ExtensionHasher, ExtensionEq>;

static const ExtensionRegistry* global_registry = nullptr;

// This function is only called at startup, so there is no need for thread-
// safety.
void Register(const ExtensionInfo& info) {
  static auto local_static_registry = OnShutdownDelete(new ExtensionRegistry);
  global_registry = local_static_registry;
  if (!InsertIfNotPresent(local_static_registry, info)) {
    GOOGLE_LOG(FATAL) << "Multiple extension registrations for type \""
               << info.message->GetTypeName() << "\", field number "
               << info.number << ".";
  }
}

const ExtensionInfo* FindRegisteredExtension(const MessageLite* extendee,
                                             int number) {
  if (!global_registry) return nullptr;

  ExtensionInfo info;
  info.message = extendee;
  info.number = number;

  auto it = global_registry->find(info);
  if (it == global_registry->end()) {
    return nullptr;
  } else {
    return &*it;
  }
}

}  // namespace

ExtensionFinder::~ExtensionFinder() {}

bool GeneratedExtensionFinder::Find(int number, ExtensionInfo* output) {
  const ExtensionInfo* extension = FindRegisteredExtension(extendee_, number);
  if (extension == nullptr) {
    return false;
  } else {
    *output = *extension;
    return true;
  }
}

void ExtensionSet::RegisterExtension(const MessageLite* extendee, int number,
                                     FieldType type, bool is_repeated,
                                     bool is_packed) {
  GOOGLE_CHECK_NE(type, WireFormatLite::TYPE_ENUM);
  GOOGLE_CHECK_NE(type, WireFormatLite::TYPE_MESSAGE);
  GOOGLE_CHECK_NE(type, WireFormatLite::TYPE_GROUP);
  ExtensionInfo info(extendee, number, type, is_repeated, is_packed);
  Register(info);
}

static bool CallNoArgValidityFunc(const void* arg, int number) {
  // Note:  Must use C-style cast here rather than reinterpret_cast because
  //   the C++ standard at one point did not allow casts between function and
  //   data pointers and some compilers enforce this for C++-style casts.  No
  //   compiler enforces it for C-style casts since lots of C-style code has
  //   relied on these kinds of casts for a long time, despite being
  //   technically undefined.  See:
  //     http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#195
  // Also note:  Some compilers do not allow function pointers to be "const".
  //   Which makes sense, I suppose, because it's meaningless.
  return ((EnumValidityFunc*)arg)(number);
}

void ExtensionSet::RegisterEnumExtension(const MessageLite* extendee,
                                         int number, FieldType type,
                                         bool is_repeated, bool is_packed,
                                         EnumValidityFunc* is_valid) {
  GOOGLE_CHECK_EQ(type, WireFormatLite::TYPE_ENUM);
  ExtensionInfo info(extendee, number, type, is_repeated, is_packed);
  info.enum_validity_check.func = CallNoArgValidityFunc;
  // See comment in CallNoArgValidityFunc() about why we use a c-style cast.
  info.enum_validity_check.arg = (void*)is_valid;
  Register(info);
}

void ExtensionSet::RegisterMessageExtension(const MessageLite* extendee,
                                            int number, FieldType type,
                                            bool is_repeated, bool is_packed,
                                            const MessageLite* prototype) {
  GOOGLE_CHECK(type == WireFormatLite::TYPE_MESSAGE ||
        type == WireFormatLite::TYPE_GROUP);
  ExtensionInfo info(extendee, number, type, is_repeated, is_packed);
  info.message_info = {prototype};
  Register(info);
}

// ===================================================================
// Constructors and basic methods.

ExtensionSet::ExtensionSet(Arena* arena)
    : arena_(arena),
      flat_capacity_(0),
      flat_size_(0),
      map_{flat_capacity_ == 0
               ? nullptr
               : Arena::CreateArray<KeyValue>(arena_, flat_capacity_)} {}

ExtensionSet::~ExtensionSet() {
  // Deletes all allocated extensions.
  if (arena_ == nullptr) {
    ForEach([](int /* number */, Extension& ext) { ext.Free(); });
    if (PROTOBUF_PREDICT_FALSE(is_large())) {
      delete map_.large;
    } else {
      DeleteFlatMap(map_.flat, flat_capacity_);
    }
  }
}

void ExtensionSet::DeleteFlatMap(const ExtensionSet::KeyValue* flat,
                                 uint16_t flat_capacity) {
#ifdef __cpp_sized_deallocation
  // Arena::CreateArray already requires a trivially destructible type, but
  // ensure this constraint is not violated in the future.
  static_assert(std::is_trivially_destructible<KeyValue>::value,
                "CreateArray requires a trivially destructible type");
  // A const-cast is needed, but this is safe as we are about to deallocate the
  // array.
  ::operator delete[](const_cast<ExtensionSet::KeyValue*>(flat),
                      sizeof(*flat) * flat_capacity);
#else   // !__cpp_sized_deallocation
  delete[] flat;
#endif  // !__cpp_sized_deallocation
}

// Defined in extension_set_heavy.cc.
// void ExtensionSet::AppendToList(const Descriptor* extendee,
//                                 const DescriptorPool* pool,
//                                 vector<const FieldDescriptor*>* output) const

bool ExtensionSet::Has(int number) const {
  const Extension* ext = FindOrNull(number);
  if (ext == nullptr) return false;
  GOOGLE_DCHECK(!ext->is_repeated);
  return !ext->is_cleared;
}

bool ExtensionSet::HasLazy(int number) const {
  return Has(number) && FindOrNull(number)->is_lazy;
}

int ExtensionSet::NumExtensions() const {
  int result = 0;
  ForEach([&result](int /* number */, const Extension& ext) {
    if (!ext.is_cleared) {
      ++result;
    }
  });
  return result;
}

int ExtensionSet::ExtensionSize(int number) const {
  const Extension* ext = FindOrNull(number);
  return ext == nullptr ? 0 : ext->GetSize();
}

FieldType ExtensionSet::ExtensionType(int number) const {
  const Extension* ext = FindOrNull(number);
  if (ext == nullptr) {
    GOOGLE_LOG(DFATAL) << "Don't lookup extension types if they aren't present (1). ";
    return 0;
  }
  if (ext->is_cleared) {
    GOOGLE_LOG(DFATAL) << "Don't lookup extension types if they aren't present (2). ";
  }
  return ext->type;
}

void ExtensionSet::ClearExtension(int number) {
  Extension* ext = FindOrNull(number);
  if (ext == nullptr) return;
  ext->Clear();
}

// ===================================================================
// Field accessors

namespace {

enum { REPEATED_FIELD, OPTIONAL_FIELD };

}  // namespace

#define GOOGLE_DCHECK_TYPE(EXTENSION, LABEL, CPPTYPE)                                 \
  GOOGLE_DCHECK_EQ((EXTENSION).is_repeated ? REPEATED_FIELD : OPTIONAL_FIELD, LABEL); \
  GOOGLE_DCHECK_EQ(cpp_type((EXTENSION).type), WireFormatLite::CPPTYPE_##CPPTYPE)

// -------------------------------------------------------------------
// Primitives

#define PRIMITIVE_ACCESSORS(UPPERCASE, LOWERCASE, CAMELCASE)                  \
                                                                              \
  LOWERCASE ExtensionSet::Get##CAMELCASE(int number, LOWERCASE default_value) \
      const {                                                                 \
    const Extension* extension = FindOrNull(number);                          \
    if (extension == nullptr || extension->is_cleared) {                      \
      return default_value;                                                   \
    } else {                                                                  \
      GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, UPPERCASE);                     \
      return extension->LOWERCASE##_value;                                    \
    }                                                                         \
  }                                                                           \
                                                                              \
  const LOWERCASE& ExtensionSet::GetRef##CAMELCASE(                           \
      int number, const LOWERCASE& default_value) const {                     \
    const Extension* extension = FindOrNull(number);                          \
    if (extension == nullptr || extension->is_cleared) {                      \
      return default_value;                                                   \
    } else {                                                                  \
      GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, UPPERCASE);                     \
      return extension->LOWERCASE##_value;                                    \
    }                                                                         \
  }                                                                           \
                                                                              \
  void ExtensionSet::Set##CAMELCASE(int number, FieldType type,               \
                                    LOWERCASE value,                          \
                                    const FieldDescriptor* descriptor) {      \
    Extension* extension;                                                     \
    if (MaybeNewExtension(number, descriptor, &extension)) {                  \
      extension->type = type;                                                 \
      GOOGLE_DCHECK_EQ(cpp_type(extension->type),                                    \
                WireFormatLite::CPPTYPE_##UPPERCASE);                         \
      extension->is_repeated = false;                                         \
    } else {                                                                  \
      GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, UPPERCASE);                     \
    }                                                                         \
    extension->is_cleared = false;                                            \
    extension->LOWERCASE##_value = value;                                     \
  }                                                                           \
                                                                              \
  LOWERCASE ExtensionSet::GetRepeated##CAMELCASE(int number, int index)       \
      const {                                                                 \
    const Extension* extension = FindOrNull(number);                          \
    GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";   \
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, UPPERCASE);                       \
    return extension->repeated_##LOWERCASE##_value->Get(index);               \
  }                                                                           \
                                                                              \
  const LOWERCASE& ExtensionSet::GetRefRepeated##CAMELCASE(int number,        \
                                                           int index) const { \
    const Extension* extension = FindOrNull(number);                          \
    GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";   \
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, UPPERCASE);                       \
    return extension->repeated_##LOWERCASE##_value->Get(index);               \
  }                                                                           \
                                                                              \
  void ExtensionSet::SetRepeated##CAMELCASE(int number, int index,            \
                                            LOWERCASE value) {                \
    Extension* extension = FindOrNull(number);                                \
    GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";   \
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, UPPERCASE);                       \
    extension->repeated_##LOWERCASE##_value->Set(index, value);               \
  }                                                                           \
                                                                              \
  void ExtensionSet::Add##CAMELCASE(int number, FieldType type, bool packed,  \
                                    LOWERCASE value,                          \
                                    const FieldDescriptor* descriptor) {      \
    Extension* extension;                                                     \
    if (MaybeNewExtension(number, descriptor, &extension)) {                  \
      extension->type = type;                                                 \
      GOOGLE_DCHECK_EQ(cpp_type(extension->type),                                    \
                WireFormatLite::CPPTYPE_##UPPERCASE);                         \
      extension->is_repeated = true;                                          \
      extension->is_packed = packed;                                          \
      extension->repeated_##LOWERCASE##_value =                               \
          Arena::CreateMessage<RepeatedField<LOWERCASE>>(arena_);             \
    } else {                                                                  \
      GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, UPPERCASE);                     \
      GOOGLE_DCHECK_EQ(extension->is_packed, packed);                                \
    }                                                                         \
    extension->repeated_##LOWERCASE##_value->Add(value);                      \
  }

PRIMITIVE_ACCESSORS(INT32, int32_t, Int32)
PRIMITIVE_ACCESSORS(INT64, int64_t, Int64)
PRIMITIVE_ACCESSORS(UINT32, uint32_t, UInt32)
PRIMITIVE_ACCESSORS(UINT64, uint64_t, UInt64)
PRIMITIVE_ACCESSORS(FLOAT, float, Float)
PRIMITIVE_ACCESSORS(DOUBLE, double, Double)
PRIMITIVE_ACCESSORS(BOOL, bool, Bool)

#undef PRIMITIVE_ACCESSORS

const void* ExtensionSet::GetRawRepeatedField(int number,
                                              const void* default_value) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr) {
    return default_value;
  }
  // We assume that all the RepeatedField<>* pointers have the same
  // size and alignment within the anonymous union in Extension.
  return extension->repeated_int32_t_value;
}

void* ExtensionSet::MutableRawRepeatedField(int number, FieldType field_type,
                                            bool packed,
                                            const FieldDescriptor* desc) {
  Extension* extension;

  // We instantiate an empty Repeated{,Ptr}Field if one doesn't exist for this
  // extension.
  if (MaybeNewExtension(number, desc, &extension)) {
    extension->is_repeated = true;
    extension->type = field_type;
    extension->is_packed = packed;

    switch (WireFormatLite::FieldTypeToCppType(
        static_cast<WireFormatLite::FieldType>(field_type))) {
      case WireFormatLite::CPPTYPE_INT32:
        extension->repeated_int32_t_value =
            Arena::CreateMessage<RepeatedField<int32_t>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_INT64:
        extension->repeated_int64_t_value =
            Arena::CreateMessage<RepeatedField<int64_t>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_UINT32:
        extension->repeated_uint32_t_value =
            Arena::CreateMessage<RepeatedField<uint32_t>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_UINT64:
        extension->repeated_uint64_t_value =
            Arena::CreateMessage<RepeatedField<uint64_t>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_DOUBLE:
        extension->repeated_double_value =
            Arena::CreateMessage<RepeatedField<double>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_FLOAT:
        extension->repeated_float_value =
            Arena::CreateMessage<RepeatedField<float>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_BOOL:
        extension->repeated_bool_value =
            Arena::CreateMessage<RepeatedField<bool>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_ENUM:
        extension->repeated_enum_value =
            Arena::CreateMessage<RepeatedField<int>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_STRING:
        extension->repeated_string_value =
            Arena::CreateMessage<RepeatedPtrField<std::string>>(arena_);
        break;
      case WireFormatLite::CPPTYPE_MESSAGE:
        extension->repeated_message_value =
            Arena::CreateMessage<RepeatedPtrField<MessageLite>>(arena_);
        break;
    }
  }

  // We assume that all the RepeatedField<>* pointers have the same
  // size and alignment within the anonymous union in Extension.
  return extension->repeated_int32_t_value;
}

// Compatible version using old call signature. Does not create extensions when
// the don't already exist; instead, just GOOGLE_CHECK-fails.
void* ExtensionSet::MutableRawRepeatedField(int number) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Extension not found.";
  // We assume that all the RepeatedField<>* pointers have the same
  // size and alignment within the anonymous union in Extension.
  return extension->repeated_int32_t_value;
}

// -------------------------------------------------------------------
// Enums

int ExtensionSet::GetEnum(int number, int default_value) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr || extension->is_cleared) {
    // Not present.  Return the default value.
    return default_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, ENUM);
    return extension->enum_value;
  }
}

const int& ExtensionSet::GetRefEnum(int number,
                                    const int& default_value) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr || extension->is_cleared) {
    // Not present.  Return the default value.
    return default_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, ENUM);
    return extension->enum_value;
  }
}

void ExtensionSet::SetEnum(int number, FieldType type, int value,
                           const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_ENUM);
    extension->is_repeated = false;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, ENUM);
  }
  extension->is_cleared = false;
  extension->enum_value = value;
}

int ExtensionSet::GetRepeatedEnum(int number, int index) const {
  const Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, ENUM);
  return extension->repeated_enum_value->Get(index);
}

const int& ExtensionSet::GetRefRepeatedEnum(int number, int index) const {
  const Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, ENUM);
  return extension->repeated_enum_value->Get(index);
}

void ExtensionSet::SetRepeatedEnum(int number, int index, int value) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, ENUM);
  extension->repeated_enum_value->Set(index, value);
}

void ExtensionSet::AddEnum(int number, FieldType type, bool packed, int value,
                           const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_ENUM);
    extension->is_repeated = true;
    extension->is_packed = packed;
    extension->repeated_enum_value =
        Arena::CreateMessage<RepeatedField<int>>(arena_);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, ENUM);
    GOOGLE_DCHECK_EQ(extension->is_packed, packed);
  }
  extension->repeated_enum_value->Add(value);
}

// -------------------------------------------------------------------
// Strings

const std::string& ExtensionSet::GetString(
    int number, const std::string& default_value) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr || extension->is_cleared) {
    // Not present.  Return the default value.
    return default_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, STRING);
    return *extension->string_value;
  }
}

std::string* ExtensionSet::MutableString(int number, FieldType type,
                                         const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_STRING);
    extension->is_repeated = false;
    extension->string_value = Arena::Create<std::string>(arena_);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, STRING);
  }
  extension->is_cleared = false;
  return extension->string_value;
}

const std::string& ExtensionSet::GetRepeatedString(int number,
                                                   int index) const {
  const Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, STRING);
  return extension->repeated_string_value->Get(index);
}

std::string* ExtensionSet::MutableRepeatedString(int number, int index) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, STRING);
  return extension->repeated_string_value->Mutable(index);
}

std::string* ExtensionSet::AddString(int number, FieldType type,
                                     const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_STRING);
    extension->is_repeated = true;
    extension->is_packed = false;
    extension->repeated_string_value =
        Arena::CreateMessage<RepeatedPtrField<std::string>>(arena_);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, STRING);
  }
  return extension->repeated_string_value->Add();
}

// -------------------------------------------------------------------
// Messages

const MessageLite& ExtensionSet::GetMessage(
    int number, const MessageLite& default_value) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr) {
    // Not present.  Return the default value.
    return default_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    if (extension->is_lazy) {
      return extension->lazymessage_value->GetMessage(default_value, arena_);
    } else {
      return *extension->message_value;
    }
  }
}

// Defined in extension_set_heavy.cc.
// const MessageLite& ExtensionSet::GetMessage(int number,
//                                             const Descriptor* message_type,
//                                             MessageFactory* factory) const

MessageLite* ExtensionSet::MutableMessage(int number, FieldType type,
                                          const MessageLite& prototype,
                                          const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_MESSAGE);
    extension->is_repeated = false;
    extension->is_lazy = false;
    extension->message_value = prototype.New(arena_);
    extension->is_cleared = false;
    return extension->message_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    extension->is_cleared = false;
    if (extension->is_lazy) {
      return extension->lazymessage_value->MutableMessage(prototype, arena_);
    } else {
      return extension->message_value;
    }
  }
}

// Defined in extension_set_heavy.cc.
// MessageLite* ExtensionSet::MutableMessage(int number, FieldType type,
//                                           const Descriptor* message_type,
//                                           MessageFactory* factory)

void ExtensionSet::SetAllocatedMessage(int number, FieldType type,
                                       const FieldDescriptor* descriptor,
                                       MessageLite* message) {
  if (message == nullptr) {
    ClearExtension(number);
    return;
  }
  GOOGLE_DCHECK(message->GetOwningArena() == nullptr ||
         message->GetOwningArena() == arena_);
  Arena* message_arena = message->GetOwningArena();
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_MESSAGE);
    extension->is_repeated = false;
    extension->is_lazy = false;
    if (message_arena == arena_) {
      extension->message_value = message;
    } else if (message_arena == nullptr) {
      extension->message_value = message;
      arena_->Own(message);  // not nullptr because not equal to message_arena
    } else {
      extension->message_value = message->New(arena_);
      extension->message_value->CheckTypeAndMergeFrom(*message);
    }
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    if (extension->is_lazy) {
      extension->lazymessage_value->SetAllocatedMessage(message, arena_);
    } else {
      if (arena_ == nullptr) {
        delete extension->message_value;
      }
      if (message_arena == arena_) {
        extension->message_value = message;
      } else if (message_arena == nullptr) {
        extension->message_value = message;
        arena_->Own(message);  // not nullptr because not equal to message_arena
      } else {
        extension->message_value = message->New(arena_);
        extension->message_value->CheckTypeAndMergeFrom(*message);
      }
    }
  }
  extension->is_cleared = false;
}

void ExtensionSet::UnsafeArenaSetAllocatedMessage(
    int number, FieldType type, const FieldDescriptor* descriptor,
    MessageLite* message) {
  if (message == nullptr) {
    ClearExtension(number);
    return;
  }
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_MESSAGE);
    extension->is_repeated = false;
    extension->is_lazy = false;
    extension->message_value = message;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    if (extension->is_lazy) {
      extension->lazymessage_value->UnsafeArenaSetAllocatedMessage(message,
                                                                   arena_);
    } else {
      if (arena_ == nullptr) {
        delete extension->message_value;
      }
      extension->message_value = message;
    }
  }
  extension->is_cleared = false;
}

MessageLite* ExtensionSet::ReleaseMessage(int number,
                                          const MessageLite& prototype) {
  Extension* extension = FindOrNull(number);
  if (extension == nullptr) {
    // Not present.  Return nullptr.
    return nullptr;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    MessageLite* ret = nullptr;
    if (extension->is_lazy) {
      ret = extension->lazymessage_value->ReleaseMessage(prototype, arena_);
      if (arena_ == nullptr) {
        delete extension->lazymessage_value;
      }
    } else {
      if (arena_ == nullptr) {
        ret = extension->message_value;
      } else {
        // ReleaseMessage() always returns a heap-allocated message, and we are
        // on an arena, so we need to make a copy of this message to return.
        ret = extension->message_value->New();
        ret->CheckTypeAndMergeFrom(*extension->message_value);
      }
    }
    Erase(number);
    return ret;
  }
}

MessageLite* ExtensionSet::UnsafeArenaReleaseMessage(
    int number, const MessageLite& prototype) {
  Extension* extension = FindOrNull(number);
  if (extension == nullptr) {
    // Not present.  Return nullptr.
    return nullptr;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL_FIELD, MESSAGE);
    MessageLite* ret = nullptr;
    if (extension->is_lazy) {
      ret = extension->lazymessage_value->UnsafeArenaReleaseMessage(prototype,
                                                                    arena_);
      if (arena_ == nullptr) {
        delete extension->lazymessage_value;
      }
    } else {
      ret = extension->message_value;
    }
    Erase(number);
    return ret;
  }
}

// Defined in extension_set_heavy.cc.
// MessageLite* ExtensionSet::ReleaseMessage(const FieldDescriptor* descriptor,
//                                           MessageFactory* factory);

const MessageLite& ExtensionSet::GetRepeatedMessage(int number,
                                                    int index) const {
  const Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, MESSAGE);
  return extension->repeated_message_value->Get(index);
}

MessageLite* ExtensionSet::MutableRepeatedMessage(int number, int index) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, MESSAGE);
  return extension->repeated_message_value->Mutable(index);
}

MessageLite* ExtensionSet::AddMessage(int number, FieldType type,
                                      const MessageLite& prototype,
                                      const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(number, descriptor, &extension)) {
    extension->type = type;
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), WireFormatLite::CPPTYPE_MESSAGE);
    extension->is_repeated = true;
    extension->repeated_message_value =
        Arena::CreateMessage<RepeatedPtrField<MessageLite>>(arena_);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, REPEATED_FIELD, MESSAGE);
  }

  // RepeatedPtrField<MessageLite> does not know how to Add() since it cannot
  // allocate an abstract object, so we have to be tricky.
  MessageLite* result = reinterpret_cast<internal::RepeatedPtrFieldBase*>(
                            extension->repeated_message_value)
                            ->AddFromCleared<GenericTypeHandler<MessageLite>>();
  if (result == nullptr) {
    result = prototype.New(arena_);
    extension->repeated_message_value->AddAllocated(result);
  }
  return result;
}

// Defined in extension_set_heavy.cc.
// MessageLite* ExtensionSet::AddMessage(int number, FieldType type,
//                                       const Descriptor* message_type,
//                                       MessageFactory* factory)

#undef GOOGLE_DCHECK_TYPE

void ExtensionSet::RemoveLast(int number) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK(extension->is_repeated);

  switch (cpp_type(extension->type)) {
    case WireFormatLite::CPPTYPE_INT32:
      extension->repeated_int32_t_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_INT64:
      extension->repeated_int64_t_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_UINT32:
      extension->repeated_uint32_t_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_UINT64:
      extension->repeated_uint64_t_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_FLOAT:
      extension->repeated_float_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_DOUBLE:
      extension->repeated_double_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_BOOL:
      extension->repeated_bool_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_ENUM:
      extension->repeated_enum_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_STRING:
      extension->repeated_string_value->RemoveLast();
      break;
    case WireFormatLite::CPPTYPE_MESSAGE:
      extension->repeated_message_value->RemoveLast();
      break;
  }
}

MessageLite* ExtensionSet::ReleaseLast(int number) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK(extension->is_repeated);
  GOOGLE_DCHECK(cpp_type(extension->type) == WireFormatLite::CPPTYPE_MESSAGE);
  return extension->repeated_message_value->ReleaseLast();
}

MessageLite* ExtensionSet::UnsafeArenaReleaseLast(int number) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK(extension->is_repeated);
  GOOGLE_DCHECK(cpp_type(extension->type) == WireFormatLite::CPPTYPE_MESSAGE);
  return extension->repeated_message_value->UnsafeArenaReleaseLast();
}

void ExtensionSet::SwapElements(int number, int index1, int index2) {
  Extension* extension = FindOrNull(number);
  GOOGLE_CHECK(extension != nullptr) << "Index out-of-bounds (field is empty).";
  GOOGLE_DCHECK(extension->is_repeated);

  switch (cpp_type(extension->type)) {
    case WireFormatLite::CPPTYPE_INT32:
      extension->repeated_int32_t_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_INT64:
      extension->repeated_int64_t_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_UINT32:
      extension->repeated_uint32_t_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_UINT64:
      extension->repeated_uint64_t_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_FLOAT:
      extension->repeated_float_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_DOUBLE:
      extension->repeated_double_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_BOOL:
      extension->repeated_bool_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_ENUM:
      extension->repeated_enum_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_STRING:
      extension->repeated_string_value->SwapElements(index1, index2);
      break;
    case WireFormatLite::CPPTYPE_MESSAGE:
      extension->repeated_message_value->SwapElements(index1, index2);
      break;
  }
}

// ===================================================================

void ExtensionSet::Clear() {
  ForEach([](int /* number */, Extension& ext) { ext.Clear(); });
}

namespace {
// Computes the size of a std::set_union without constructing the union.
template <typename ItX, typename ItY>
size_t SizeOfUnion(ItX it_xs, ItX end_xs, ItY it_ys, ItY end_ys) {
  size_t result = 0;
  while (it_xs != end_xs && it_ys != end_ys) {
    ++result;
    if (it_xs->first < it_ys->first) {
      ++it_xs;
    } else if (it_xs->first == it_ys->first) {
      ++it_xs;
      ++it_ys;
    } else {
      ++it_ys;
    }
  }
  result += std::distance(it_xs, end_xs);
  result += std::distance(it_ys, end_ys);
  return result;
}
}  // namespace

void ExtensionSet::MergeFrom(const MessageLite* extendee,
                             const ExtensionSet& other) {
  if (PROTOBUF_PREDICT_TRUE(!is_large())) {
    if (PROTOBUF_PREDICT_TRUE(!other.is_large())) {
      GrowCapacity(SizeOfUnion(flat_begin(), flat_end(), other.flat_begin(),
                               other.flat_end()));
    } else {
      GrowCapacity(SizeOfUnion(flat_begin(), flat_end(),
                               other.map_.large->begin(),
                               other.map_.large->end()));
    }
  }
  other.ForEach([extendee, this, &other](int number, const Extension& ext) {
    this->InternalExtensionMergeFrom(extendee, number, ext, other.arena_);
  });
}

void ExtensionSet::InternalExtensionMergeFrom(const MessageLite* extendee,
                                              int number,
                                              const Extension& other_extension,
                                              Arena* other_arena) {
  if (other_extension.is_repeated) {
    Extension* extension;
    bool is_new =
        MaybeNewExtension(number, other_extension.descriptor, &extension);
    if (is_new) {
      // Extension did not already exist in set.
      extension->type = other_extension.type;
      extension->is_packed = other_extension.is_packed;
      extension->is_repeated = true;
    } else {
      GOOGLE_DCHECK_EQ(extension->type, other_extension.type);
      GOOGLE_DCHECK_EQ(extension->is_packed, other_extension.is_packed);
      GOOGLE_DCHECK(extension->is_repeated);
    }

    switch (cpp_type(other_extension.type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE, REPEATED_TYPE) \
  case WireFormatLite::CPPTYPE_##UPPERCASE:              \
    if (is_new) {                                        \
      extension->repeated_##LOWERCASE##_value =          \
          Arena::CreateMessage<REPEATED_TYPE>(arena_);   \
    }                                                    \
    extension->repeated_##LOWERCASE##_value->MergeFrom(  \
        *other_extension.repeated_##LOWERCASE##_value);  \
    break;

      HANDLE_TYPE(INT32, int32_t, RepeatedField<int32_t>);
      HANDLE_TYPE(INT64, int64_t, RepeatedField<int64_t>);
      HANDLE_TYPE(UINT32, uint32_t, RepeatedField<uint32_t>);
      HANDLE_TYPE(UINT64, uint64_t, RepeatedField<uint64_t>);
      HANDLE_TYPE(FLOAT, float, RepeatedField<float>);
      HANDLE_TYPE(DOUBLE, double, RepeatedField<double>);
      HANDLE_TYPE(BOOL, bool, RepeatedField<bool>);
      HANDLE_TYPE(ENUM, enum, RepeatedField<int>);
      HANDLE_TYPE(STRING, string, RepeatedPtrField<std::string>);
#undef HANDLE_TYPE

      case WireFormatLite::CPPTYPE_MESSAGE:
        if (is_new) {
          extension->repeated_message_value =
              Arena::CreateMessage<RepeatedPtrField<MessageLite>>(arena_);
        }
        // We can't call RepeatedPtrField<MessageLite>::MergeFrom() because
        // it would attempt to allocate new objects.
        RepeatedPtrField<MessageLite>* other_repeated_message =
            other_extension.repeated_message_value;
        for (int i = 0; i < other_repeated_message->size(); i++) {
          const MessageLite& other_message = other_repeated_message->Get(i);
          MessageLite* target =
              reinterpret_cast<internal::RepeatedPtrFieldBase*>(
                  extension->repeated_message_value)
                  ->AddFromCleared<GenericTypeHandler<MessageLite>>();
          if (target == nullptr) {
            target = other_message.New(arena_);
            extension->repeated_message_value->AddAllocated(target);
          }
          target->CheckTypeAndMergeFrom(other_message);
        }
        break;
    }
  } else {
    if (!other_extension.is_cleared) {
      switch (cpp_type(other_extension.type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE, CAMELCASE)  \
  case WireFormatLite::CPPTYPE_##UPPERCASE:           \
    Set##CAMELCASE(number, other_extension.type,      \
                   other_extension.LOWERCASE##_value, \
                   other_extension.descriptor);       \
    break;

        HANDLE_TYPE(INT32, int32_t, Int32);
        HANDLE_TYPE(INT64, int64_t, Int64);
        HANDLE_TYPE(UINT32, uint32_t, UInt32);
        HANDLE_TYPE(UINT64, uint64_t, UInt64);
        HANDLE_TYPE(FLOAT, float, Float);
        HANDLE_TYPE(DOUBLE, double, Double);
        HANDLE_TYPE(BOOL, bool, Bool);
        HANDLE_TYPE(ENUM, enum, Enum);
#undef HANDLE_TYPE
        case WireFormatLite::CPPTYPE_STRING:
          SetString(number, other_extension.type, *other_extension.string_value,
                    other_extension.descriptor);
          break;
        case WireFormatLite::CPPTYPE_MESSAGE: {
          Extension* extension;
          bool is_new =
              MaybeNewExtension(number, other_extension.descriptor, &extension);
          if (is_new) {
            extension->type = other_extension.type;
            extension->is_packed = other_extension.is_packed;
            extension->is_repeated = false;
            if (other_extension.is_lazy) {
              extension->is_lazy = true;
              extension->lazymessage_value =
                  other_extension.lazymessage_value->New(arena_);
              extension->lazymessage_value->MergeFrom(
                  GetPrototypeForLazyMessage(extendee, number),
                  *other_extension.lazymessage_value, arena_);
            } else {
              extension->is_lazy = false;
              extension->message_value =
                  other_extension.message_value->New(arena_);
              extension->message_value->CheckTypeAndMergeFrom(
                  *other_extension.message_value);
            }
          } else {
            GOOGLE_DCHECK_EQ(extension->type, other_extension.type);
            GOOGLE_DCHECK_EQ(extension->is_packed, other_extension.is_packed);
            GOOGLE_DCHECK(!extension->is_repeated);
            if (other_extension.is_lazy) {
              if (extension->is_lazy) {
                extension->lazymessage_value->MergeFrom(
                    GetPrototypeForLazyMessage(extendee, number),
                    *other_extension.lazymessage_value, arena_);
              } else {
                extension->message_value->CheckTypeAndMergeFrom(
                    other_extension.lazymessage_value->GetMessage(
                        *extension->message_value, other_arena));
              }
            } else {
              if (extension->is_lazy) {
                extension->lazymessage_value
                    ->MutableMessage(*other_extension.message_value, arena_)
                    ->CheckTypeAndMergeFrom(*other_extension.message_value);
              } else {
                extension->message_value->CheckTypeAndMergeFrom(
                    *other_extension.message_value);
              }
            }
          }
          extension->is_cleared = false;
          break;
        }
      }
    }
  }
}

void ExtensionSet::Swap(const MessageLite* extendee, ExtensionSet* other) {
#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
  if (GetArena() != nullptr && GetArena() == other->GetArena()) {
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
  if (GetArena() == other->GetArena()) {
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
    InternalSwap(other);
  } else {
    // TODO(cfallin, rohananil): We maybe able to optimize a case where we are
    // swapping from heap to arena-allocated extension set, by just Own()'ing
    // the extensions.
    ExtensionSet extension_set;
    extension_set.MergeFrom(extendee, *other);
    other->Clear();
    other->MergeFrom(extendee, *this);
    Clear();
    MergeFrom(extendee, extension_set);
  }
}

void ExtensionSet::InternalSwap(ExtensionSet* other) {
  using std::swap;
  swap(arena_, other->arena_);
  swap(flat_capacity_, other->flat_capacity_);
  swap(flat_size_, other->flat_size_);
  swap(map_, other->map_);
}

void ExtensionSet::SwapExtension(const MessageLite* extendee,
                                 ExtensionSet* other, int number) {
  if (this == other) return;

  if (GetArena() == other->GetArena()) {
    UnsafeShallowSwapExtension(other, number);
    return;
  }

  Extension* this_ext = FindOrNull(number);
  Extension* other_ext = other->FindOrNull(number);

  if (this_ext == other_ext) return;

  if (this_ext != nullptr && other_ext != nullptr) {
    // TODO(cfallin, rohananil): We could further optimize these cases,
    // especially avoid creation of ExtensionSet, and move MergeFrom logic
    // into Extensions itself (which takes arena as an argument).
    // We do it this way to reuse the copy-across-arenas logic already
    // implemented in ExtensionSet's MergeFrom.
    ExtensionSet temp;
    temp.InternalExtensionMergeFrom(extendee, number, *other_ext,
                                    other->GetArena());
    Extension* temp_ext = temp.FindOrNull(number);

    other_ext->Clear();
    other->InternalExtensionMergeFrom(extendee, number, *this_ext,
                                      this->GetArena());
    this_ext->Clear();
    InternalExtensionMergeFrom(extendee, number, *temp_ext, temp.GetArena());
  } else if (this_ext == nullptr) {
    InternalExtensionMergeFrom(extendee, number, *other_ext, other->GetArena());
    if (other->GetArena() == nullptr) other_ext->Free();
    other->Erase(number);
  } else {
    other->InternalExtensionMergeFrom(extendee, number, *this_ext,
                                      this->GetArena());
    if (GetArena() == nullptr) this_ext->Free();
    Erase(number);
  }
}

void ExtensionSet::UnsafeShallowSwapExtension(ExtensionSet* other, int number) {
  if (this == other) return;

  Extension* this_ext = FindOrNull(number);
  Extension* other_ext = other->FindOrNull(number);

  if (this_ext == other_ext) return;

  GOOGLE_DCHECK_EQ(GetArena(), other->GetArena());

  if (this_ext != nullptr && other_ext != nullptr) {
    std::swap(*this_ext, *other_ext);
  } else if (this_ext == nullptr) {
    *Insert(number).first = *other_ext;
    other->Erase(number);
  } else {
    *other->Insert(number).first = *this_ext;
    Erase(number);
  }
}

bool ExtensionSet::IsInitialized() const {
  // Extensions are never required.  However, we need to check that all
  // embedded messages are initialized.
  if (PROTOBUF_PREDICT_FALSE(is_large())) {
    for (const auto& kv : *map_.large) {
      if (!kv.second.IsInitialized()) return false;
    }
    return true;
  }
  for (const KeyValue* it = flat_begin(); it != flat_end(); ++it) {
    if (!it->second.IsInitialized()) return false;
  }
  return true;
}

bool ExtensionSet::FindExtensionInfoFromTag(uint32_t tag,
                                            ExtensionFinder* extension_finder,
                                            int* field_number,
                                            ExtensionInfo* extension,
                                            bool* was_packed_on_wire) {
  *field_number = WireFormatLite::GetTagFieldNumber(tag);
  WireFormatLite::WireType wire_type = WireFormatLite::GetTagWireType(tag);
  return FindExtensionInfoFromFieldNumber(wire_type, *field_number,
                                          extension_finder, extension,
                                          was_packed_on_wire);
}

bool ExtensionSet::FindExtensionInfoFromFieldNumber(
    int wire_type, int field_number, ExtensionFinder* extension_finder,
    ExtensionInfo* extension, bool* was_packed_on_wire) const {
  if (!extension_finder->Find(field_number, extension)) {
    return false;
  }

  WireFormatLite::WireType expected_wire_type =
      WireFormatLite::WireTypeForFieldType(real_type(extension->type));

  // Check if this is a packed field.
  *was_packed_on_wire = false;
  if (extension->is_repeated &&
      wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED &&
      is_packable(expected_wire_type)) {
    *was_packed_on_wire = true;
    return true;
  }
  // Otherwise the wire type must match.
  return expected_wire_type == wire_type;
}

bool ExtensionSet::ParseField(uint32_t tag, io::CodedInputStream* input,
                              ExtensionFinder* extension_finder,
                              FieldSkipper* field_skipper) {
  int number;
  bool was_packed_on_wire;
  ExtensionInfo extension;
  if (!FindExtensionInfoFromTag(tag, extension_finder, &number, &extension,
                                &was_packed_on_wire)) {
    return field_skipper->SkipField(input, tag);
  } else {
    return ParseFieldWithExtensionInfo(number, was_packed_on_wire, extension,
                                       input, field_skipper);
  }
}

const char* ExtensionSet::ParseField(uint64_t tag, const char* ptr,
                                     const MessageLite* extendee,
                                     internal::InternalMetadata* metadata,
                                     internal::ParseContext* ctx) {
  GeneratedExtensionFinder finder(extendee);
  int number = tag >> 3;
  bool was_packed_on_wire;
  ExtensionInfo extension;
  if (!FindExtensionInfoFromFieldNumber(tag & 7, number, &finder, &extension,
                                        &was_packed_on_wire)) {
    return UnknownFieldParse(
        tag, metadata->mutable_unknown_fields<std::string>(), ptr, ctx);
  }
  return ParseFieldWithExtensionInfo<std::string>(
      number, was_packed_on_wire, extension, metadata, ptr, ctx);
}

const char* ExtensionSet::ParseMessageSetItem(
    const char* ptr, const MessageLite* extendee,
    internal::InternalMetadata* metadata, internal::ParseContext* ctx) {
  return ParseMessageSetItemTmpl<MessageLite, std::string>(ptr, extendee,
                                                           metadata, ctx);
}

bool ExtensionSet::ParseFieldWithExtensionInfo(int number,
                                               bool was_packed_on_wire,
                                               const ExtensionInfo& extension,
                                               io::CodedInputStream* input,
                                               FieldSkipper* field_skipper) {
  // Explicitly not read extension.is_packed, instead check whether the field
  // was encoded in packed form on the wire.
  if (was_packed_on_wire) {
    uint32_t size;
    if (!input->ReadVarint32(&size)) return false;
    io::CodedInputStream::Limit limit = input->PushLimit(size);

    switch (extension.type) {
#define HANDLE_TYPE(UPPERCASE, CPP_CAMELCASE, CPP_LOWERCASE)                \
  case WireFormatLite::TYPE_##UPPERCASE:                                    \
    while (input->BytesUntilLimit() > 0) {                                  \
      CPP_LOWERCASE value;                                                  \
      if (!WireFormatLite::ReadPrimitive<CPP_LOWERCASE,                     \
                                         WireFormatLite::TYPE_##UPPERCASE>( \
              input, &value))                                               \
        return false;                                                       \
      Add##CPP_CAMELCASE(number, WireFormatLite::TYPE_##UPPERCASE,          \
                         extension.is_packed, value, extension.descriptor); \
    }                                                                       \
    break

      HANDLE_TYPE(INT32, Int32, int32_t);
      HANDLE_TYPE(INT64, Int64, int64_t);
      HANDLE_TYPE(UINT32, UInt32, uint32_t);
      HANDLE_TYPE(UINT64, UInt64, uint64_t);
      HANDLE_TYPE(SINT32, Int32, int32_t);
      HANDLE_TYPE(SINT64, Int64, int64_t);
      HANDLE_TYPE(FIXED32, UInt32, uint32_t);
      HANDLE_TYPE(FIXED64, UInt64, uint64_t);
      HANDLE_TYPE(SFIXED32, Int32, int32_t);
      HANDLE_TYPE(SFIXED64, Int64, int64_t);
      HANDLE_TYPE(FLOAT, Float, float);
      HANDLE_TYPE(DOUBLE, Double, double);
      HANDLE_TYPE(BOOL, Bool, bool);
#undef HANDLE_TYPE

      case WireFormatLite::TYPE_ENUM:
        while (input->BytesUntilLimit() > 0) {
          int value;
          if (!WireFormatLite::ReadPrimitive<int, WireFormatLite::TYPE_ENUM>(
                  input, &value))
            return false;
          if (extension.enum_validity_check.func(
                  extension.enum_validity_check.arg, value)) {
            AddEnum(number, WireFormatLite::TYPE_ENUM, extension.is_packed,
                    value, extension.descriptor);
          } else {
            // Invalid value.  Treat as unknown.
            field_skipper->SkipUnknownEnum(number, value);
          }
        }
        break;

      case WireFormatLite::TYPE_STRING:
      case WireFormatLite::TYPE_BYTES:
      case WireFormatLite::TYPE_GROUP:
      case WireFormatLite::TYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Non-primitive types can't be packed.";
        break;
    }

    input->PopLimit(limit);
  } else {
    switch (extension.type) {
#define HANDLE_TYPE(UPPERCASE, CPP_CAMELCASE, CPP_LOWERCASE)                \
  case WireFormatLite::TYPE_##UPPERCASE: {                                  \
    CPP_LOWERCASE value;                                                    \
    if (!WireFormatLite::ReadPrimitive<CPP_LOWERCASE,                       \
                                       WireFormatLite::TYPE_##UPPERCASE>(   \
            input, &value))                                                 \
      return false;                                                         \
    if (extension.is_repeated) {                                            \
      Add##CPP_CAMELCASE(number, WireFormatLite::TYPE_##UPPERCASE,          \
                         extension.is_packed, value, extension.descriptor); \
    } else {                                                                \
      Set##CPP_CAMELCASE(number, WireFormatLite::TYPE_##UPPERCASE, value,   \
                         extension.descriptor);                             \
    }                                                                       \
  } break

      HANDLE_TYPE(INT32, Int32, int32_t);
      HANDLE_TYPE(INT64, Int64, int64_t);
      HANDLE_TYPE(UINT32, UInt32, uint32_t);
      HANDLE_TYPE(UINT64, UInt64, uint64_t);
      HANDLE_TYPE(SINT32, Int32, int32_t);
      HANDLE_TYPE(SINT64, Int64, int64_t);
      HANDLE_TYPE(FIXED32, UInt32, uint32_t);
      HANDLE_TYPE(FIXED64, UInt64, uint64_t);
      HANDLE_TYPE(SFIXED32, Int32, int32_t);
      HANDLE_TYPE(SFIXED64, Int64, int64_t);
      HANDLE_TYPE(FLOAT, Float, float);
      HANDLE_TYPE(DOUBLE, Double, double);
      HANDLE_TYPE(BOOL, Bool, bool);
#undef HANDLE_TYPE

      case WireFormatLite::TYPE_ENUM: {
        int value;
        if (!WireFormatLite::ReadPrimitive<int, WireFormatLite::TYPE_ENUM>(
                input, &value))
          return false;

        if (!extension.enum_validity_check.func(
                extension.enum_validity_check.arg, value)) {
          // Invalid value.  Treat as unknown.
          field_skipper->SkipUnknownEnum(number, value);
        } else if (extension.is_repeated) {
          AddEnum(number, WireFormatLite::TYPE_ENUM, extension.is_packed, value,
                  extension.descriptor);
        } else {
          SetEnum(number, WireFormatLite::TYPE_ENUM, value,
                  extension.descriptor);
        }
        break;
      }

      case WireFormatLite::TYPE_STRING: {
        std::string* value =
            extension.is_repeated
                ? AddString(number, WireFormatLite::TYPE_STRING,
                            extension.descriptor)
                : MutableString(number, WireFormatLite::TYPE_STRING,
                                extension.descriptor);
        if (!WireFormatLite::ReadString(input, value)) return false;
        break;
      }

      case WireFormatLite::TYPE_BYTES: {
        std::string* value =
            extension.is_repeated
                ? AddString(number, WireFormatLite::TYPE_BYTES,
                            extension.descriptor)
                : MutableString(number, WireFormatLite::TYPE_BYTES,
                                extension.descriptor);
        if (!WireFormatLite::ReadBytes(input, value)) return false;
        break;
      }

      case WireFormatLite::TYPE_GROUP: {
        MessageLite* value =
            extension.is_repeated
                ? AddMessage(number, WireFormatLite::TYPE_GROUP,
                             *extension.message_info.prototype,
                             extension.descriptor)
                : MutableMessage(number, WireFormatLite::TYPE_GROUP,
                                 *extension.message_info.prototype,
                                 extension.descriptor);
        if (!WireFormatLite::ReadGroup(number, input, value)) return false;
        break;
      }

      case WireFormatLite::TYPE_MESSAGE: {
        MessageLite* value =
            extension.is_repeated
                ? AddMessage(number, WireFormatLite::TYPE_MESSAGE,
                             *extension.message_info.prototype,
                             extension.descriptor)
                : MutableMessage(number, WireFormatLite::TYPE_MESSAGE,
                                 *extension.message_info.prototype,
                                 extension.descriptor);
        if (!WireFormatLite::ReadMessage(input, value)) return false;
        break;
      }
    }
  }

  return true;
}

bool ExtensionSet::ParseField(uint32_t tag, io::CodedInputStream* input,
                              const MessageLite* extendee) {
  FieldSkipper skipper;
  GeneratedExtensionFinder finder(extendee);
  return ParseField(tag, input, &finder, &skipper);
}

bool ExtensionSet::ParseField(uint32_t tag, io::CodedInputStream* input,
                              const MessageLite* extendee,
                              io::CodedOutputStream* unknown_fields) {
  CodedOutputStreamFieldSkipper skipper(unknown_fields);
  GeneratedExtensionFinder finder(extendee);
  return ParseField(tag, input, &finder, &skipper);
}

bool ExtensionSet::ParseMessageSetLite(io::CodedInputStream* input,
                                       ExtensionFinder* extension_finder,
                                       FieldSkipper* field_skipper) {
  while (true) {
    const uint32_t tag = input->ReadTag();
    switch (tag) {
      case 0:
        return true;
      case WireFormatLite::kMessageSetItemStartTag:
        if (!ParseMessageSetItemLite(input, extension_finder, field_skipper)) {
          return false;
        }
        break;
      default:
        if (!ParseField(tag, input, extension_finder, field_skipper)) {
          return false;
        }
        break;
    }
  }
}

bool ExtensionSet::ParseMessageSetItemLite(io::CodedInputStream* input,
                                           ExtensionFinder* extension_finder,
                                           FieldSkipper* field_skipper) {
  struct MSLite {
    bool ParseField(int type_id, io::CodedInputStream* input) {
      return me->ParseField(
          WireFormatLite::WIRETYPE_LENGTH_DELIMITED + 8 * type_id, input,
          extension_finder, field_skipper);
    }

    bool SkipField(uint32_t tag, io::CodedInputStream* input) {
      return field_skipper->SkipField(input, tag);
    }

    ExtensionSet* me;
    ExtensionFinder* extension_finder;
    FieldSkipper* field_skipper;
  };

  return ParseMessageSetItemImpl(input,
                                 MSLite{this, extension_finder, field_skipper});
}

bool ExtensionSet::ParseMessageSet(io::CodedInputStream* input,
                                   const MessageLite* extendee,
                                   std::string* unknown_fields) {
  io::StringOutputStream zcis(unknown_fields);
  io::CodedOutputStream output(&zcis);
  CodedOutputStreamFieldSkipper skipper(&output);
  GeneratedExtensionFinder finder(extendee);
  return ParseMessageSetLite(input, &finder, &skipper);
}

uint8_t* ExtensionSet::_InternalSerializeImpl(
    const MessageLite* extendee, int start_field_number, int end_field_number,
    uint8_t* target, io::EpsCopyOutputStream* stream) const {
  if (PROTOBUF_PREDICT_FALSE(is_large())) {
    const auto& end = map_.large->end();
    for (auto it = map_.large->lower_bound(start_field_number);
         it != end && it->first < end_field_number; ++it) {
      target = it->second.InternalSerializeFieldWithCachedSizesToArray(
          extendee, this, it->first, target, stream);
    }
    return target;
  }
  const KeyValue* end = flat_end();
  for (const KeyValue* it = std::lower_bound(
           flat_begin(), end, start_field_number, KeyValue::FirstComparator());
       it != end && it->first < end_field_number; ++it) {
    target = it->second.InternalSerializeFieldWithCachedSizesToArray(
        extendee, this, it->first, target, stream);
  }
  return target;
}

uint8_t* ExtensionSet::InternalSerializeMessageSetWithCachedSizesToArray(
    const MessageLite* extendee, uint8_t* target,
    io::EpsCopyOutputStream* stream) const {
  const ExtensionSet* extension_set = this;
  ForEach([&target, extendee, stream, extension_set](int number,
                                                     const Extension& ext) {
    target = ext.InternalSerializeMessageSetItemWithCachedSizesToArray(
        extendee, extension_set, number, target, stream);
  });
  return target;
}

size_t ExtensionSet::ByteSize() const {
  size_t total_size = 0;
  ForEach([&total_size](int number, const Extension& ext) {
    total_size += ext.ByteSize(number);
  });
  return total_size;
}

// Defined in extension_set_heavy.cc.
// int ExtensionSet::SpaceUsedExcludingSelf() const

bool ExtensionSet::MaybeNewExtension(int number,
                                     const FieldDescriptor* descriptor,
                                     Extension** result) {
  bool extension_is_new = false;
  std::tie(*result, extension_is_new) = Insert(number);
  (*result)->descriptor = descriptor;
  return extension_is_new;
}

// ===================================================================
// Methods of ExtensionSet::Extension

void ExtensionSet::Extension::Clear() {
  if (is_repeated) {
    switch (cpp_type(type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)   \
  case WireFormatLite::CPPTYPE_##UPPERCASE: \
    repeated_##LOWERCASE##_value->Clear();  \
    break

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, enum);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(MESSAGE, message);
#undef HANDLE_TYPE
    }
  } else {
    if (!is_cleared) {
      switch (cpp_type(type)) {
        case WireFormatLite::CPPTYPE_STRING:
          string_value->clear();
          break;
        case WireFormatLite::CPPTYPE_MESSAGE:
          if (is_lazy) {
            lazymessage_value->Clear();
          } else {
            message_value->Clear();
          }
          break;
        default:
          // No need to do anything.  Get*() will return the default value
          // as long as is_cleared is true and Set*() will overwrite the
          // previous value.
          break;
      }

      is_cleared = true;
    }
  }
}

size_t ExtensionSet::Extension::ByteSize(int number) const {
  size_t result = 0;

  if (is_repeated) {
    if (is_packed) {
      switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                 \
  case WireFormatLite::TYPE_##UPPERCASE:                             \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) { \
      result += WireFormatLite::CAMELCASE##Size(                     \
          repeated_##LOWERCASE##_value->Get(i));                     \
    }                                                                \
    break

        HANDLE_TYPE(INT32, Int32, int32_t);
        HANDLE_TYPE(INT64, Int64, int64_t);
        HANDLE_TYPE(UINT32, UInt32, uint32_t);
        HANDLE_TYPE(UINT64, UInt64, uint64_t);
        HANDLE_TYPE(SINT32, SInt32, int32_t);
        HANDLE_TYPE(SINT64, SInt64, int64_t);
        HANDLE_TYPE(ENUM, Enum, enum);
#undef HANDLE_TYPE

        // Stuff with fixed size.
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)             \
  case WireFormatLite::TYPE_##UPPERCASE:                         \
    result += WireFormatLite::k##CAMELCASE##Size *               \
              FromIntSize(repeated_##LOWERCASE##_value->size()); \
    break
        HANDLE_TYPE(FIXED32, Fixed32, uint32_t);
        HANDLE_TYPE(FIXED64, Fixed64, uint64_t);
        HANDLE_TYPE(SFIXED32, SFixed32, int32_t);
        HANDLE_TYPE(SFIXED64, SFixed64, int64_t);
        HANDLE_TYPE(FLOAT, Float, float);
        HANDLE_TYPE(DOUBLE, Double, double);
        HANDLE_TYPE(BOOL, Bool, bool);
#undef HANDLE_TYPE

        case WireFormatLite::TYPE_STRING:
        case WireFormatLite::TYPE_BYTES:
        case WireFormatLite::TYPE_GROUP:
        case WireFormatLite::TYPE_MESSAGE:
          GOOGLE_LOG(FATAL) << "Non-primitive types can't be packed.";
          break;
      }

      cached_size = ToCachedSize(result);
      if (result > 0) {
        result += io::CodedOutputStream::VarintSize32(result);
        result += io::CodedOutputStream::VarintSize32(WireFormatLite::MakeTag(
            number, WireFormatLite::WIRETYPE_LENGTH_DELIMITED));
      }
    } else {
      size_t tag_size = WireFormatLite::TagSize(number, real_type(type));

      switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                        \
  case WireFormatLite::TYPE_##UPPERCASE:                                    \
    result += tag_size * FromIntSize(repeated_##LOWERCASE##_value->size()); \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) {        \
      result += WireFormatLite::CAMELCASE##Size(                            \
          repeated_##LOWERCASE##_value->Get(i));                            \
    }                                                                       \
    break

        HANDLE_TYPE(INT32, Int32, int32_t);
        HANDLE_TYPE(INT64, Int64, int64_t);
        HANDLE_TYPE(UINT32, UInt32, uint32_t);
        HANDLE_TYPE(UINT64, UInt64, uint64_t);
        HANDLE_TYPE(SINT32, SInt32, int32_t);
        HANDLE_TYPE(SINT64, SInt64, int64_t);
        HANDLE_TYPE(STRING, String, string);
        HANDLE_TYPE(BYTES, Bytes, string);
        HANDLE_TYPE(ENUM, Enum, enum);
        HANDLE_TYPE(GROUP, Group, message);
        HANDLE_TYPE(MESSAGE, Message, message);
#undef HANDLE_TYPE

        // Stuff with fixed size.
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)             \
  case WireFormatLite::TYPE_##UPPERCASE:                         \
    result += (tag_size + WireFormatLite::k##CAMELCASE##Size) *  \
              FromIntSize(repeated_##LOWERCASE##_value->size()); \
    break
        HANDLE_TYPE(FIXED32, Fixed32, uint32_t);
        HANDLE_TYPE(FIXED64, Fixed64, uint64_t);
        HANDLE_TYPE(SFIXED32, SFixed32, int32_t);
        HANDLE_TYPE(SFIXED64, SFixed64, int64_t);
        HANDLE_TYPE(FLOAT, Float, float);
        HANDLE_TYPE(DOUBLE, Double, double);
        HANDLE_TYPE(BOOL, Bool, bool);
#undef HANDLE_TYPE
      }
    }
  } else if (!is_cleared) {
    result += WireFormatLite::TagSize(number, real_type(type));
    switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)      \
  case WireFormatLite::TYPE_##UPPERCASE:                  \
    result += WireFormatLite::CAMELCASE##Size(LOWERCASE); \
    break

      HANDLE_TYPE(INT32, Int32, int32_t_value);
      HANDLE_TYPE(INT64, Int64, int64_t_value);
      HANDLE_TYPE(UINT32, UInt32, uint32_t_value);
      HANDLE_TYPE(UINT64, UInt64, uint64_t_value);
      HANDLE_TYPE(SINT32, SInt32, int32_t_value);
      HANDLE_TYPE(SINT64, SInt64, int64_t_value);
      HANDLE_TYPE(STRING, String, *string_value);
      HANDLE_TYPE(BYTES, Bytes, *string_value);
      HANDLE_TYPE(ENUM, Enum, enum_value);
      HANDLE_TYPE(GROUP, Group, *message_value);
#undef HANDLE_TYPE
      case WireFormatLite::TYPE_MESSAGE: {
        if (is_lazy) {
          size_t size = lazymessage_value->ByteSizeLong();
          result += io::CodedOutputStream::VarintSize32(size) + size;
        } else {
          result += WireFormatLite::MessageSize(*message_value);
        }
        break;
      }

      // Stuff with fixed size.
#define HANDLE_TYPE(UPPERCASE, CAMELCASE)         \
  case WireFormatLite::TYPE_##UPPERCASE:          \
    result += WireFormatLite::k##CAMELCASE##Size; \
    break
        HANDLE_TYPE(FIXED32, Fixed32);
        HANDLE_TYPE(FIXED64, Fixed64);
        HANDLE_TYPE(SFIXED32, SFixed32);
        HANDLE_TYPE(SFIXED64, SFixed64);
        HANDLE_TYPE(FLOAT, Float);
        HANDLE_TYPE(DOUBLE, Double);
        HANDLE_TYPE(BOOL, Bool);
#undef HANDLE_TYPE
    }
  }

  return result;
}

int ExtensionSet::Extension::GetSize() const {
  GOOGLE_DCHECK(is_repeated);
  switch (cpp_type(type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)   \
  case WireFormatLite::CPPTYPE_##UPPERCASE: \
    return repeated_##LOWERCASE##_value->size()

    HANDLE_TYPE(INT32, int32_t);
    HANDLE_TYPE(INT64, int64_t);
    HANDLE_TYPE(UINT32, uint32_t);
    HANDLE_TYPE(UINT64, uint64_t);
    HANDLE_TYPE(FLOAT, float);
    HANDLE_TYPE(DOUBLE, double);
    HANDLE_TYPE(BOOL, bool);
    HANDLE_TYPE(ENUM, enum);
    HANDLE_TYPE(STRING, string);
    HANDLE_TYPE(MESSAGE, message);
#undef HANDLE_TYPE
  }

  GOOGLE_LOG(FATAL) << "Can't get here.";
  return 0;
}

// This function deletes all allocated objects. This function should be only
// called if the Extension was created without an arena.
void ExtensionSet::Extension::Free() {
  if (is_repeated) {
    switch (cpp_type(type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)   \
  case WireFormatLite::CPPTYPE_##UPPERCASE: \
    delete repeated_##LOWERCASE##_value;    \
    break

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, enum);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(MESSAGE, message);
#undef HANDLE_TYPE
    }
  } else {
    switch (cpp_type(type)) {
      case WireFormatLite::CPPTYPE_STRING:
        delete string_value;
        break;
      case WireFormatLite::CPPTYPE_MESSAGE:
        if (is_lazy) {
          delete lazymessage_value;
        } else {
          delete message_value;
        }
        break;
      default:
        break;
    }
  }
}

// Defined in extension_set_heavy.cc.
// int ExtensionSet::Extension::SpaceUsedExcludingSelf() const

bool ExtensionSet::Extension::IsInitialized() const {
  if (cpp_type(type) == WireFormatLite::CPPTYPE_MESSAGE) {
    if (is_repeated) {
      for (int i = 0; i < repeated_message_value->size(); i++) {
        if (!repeated_message_value->Get(i).IsInitialized()) {
          return false;
        }
      }
    } else {
      if (!is_cleared) {
        if (is_lazy) {
          if (!lazymessage_value->IsInitialized()) return false;
        } else {
          if (!message_value->IsInitialized()) return false;
        }
      }
    }
  }
  return true;
}

// Dummy key method to avoid weak vtable.
void ExtensionSet::LazyMessageExtension::UnusedKeyMethod() {}

const ExtensionSet::Extension* ExtensionSet::FindOrNull(int key) const {
  if (flat_size_ == 0) {
    return nullptr;
  } else if (PROTOBUF_PREDICT_TRUE(!is_large())) {
    auto it = std::lower_bound(flat_begin(), flat_end() - 1, key,
                               KeyValue::FirstComparator());
    return it->first == key ? &it->second : nullptr;
  } else {
    return FindOrNullInLargeMap(key);
  }
}

const ExtensionSet::Extension* ExtensionSet::FindOrNullInLargeMap(
    int key) const {
  assert(is_large());
  LargeMap::const_iterator it = map_.large->find(key);
  if (it != map_.large->end()) {
    return &it->second;
  }
  return nullptr;
}

ExtensionSet::Extension* ExtensionSet::FindOrNull(int key) {
  const auto* const_this = this;
  return const_cast<ExtensionSet::Extension*>(const_this->FindOrNull(key));
}

ExtensionSet::Extension* ExtensionSet::FindOrNullInLargeMap(int key) {
  const auto* const_this = this;
  return const_cast<ExtensionSet::Extension*>(
      const_this->FindOrNullInLargeMap(key));
}

std::pair<ExtensionSet::Extension*, bool> ExtensionSet::Insert(int key) {
  if (PROTOBUF_PREDICT_FALSE(is_large())) {
    auto maybe = map_.large->insert({key, Extension()});
    return {&maybe.first->second, maybe.second};
  }
  KeyValue* end = flat_end();
  KeyValue* it =
      std::lower_bound(flat_begin(), end, key, KeyValue::FirstComparator());
  if (it != end && it->first == key) {
    return {&it->second, false};
  }
  if (flat_size_ < flat_capacity_) {
    std::copy_backward(it, end, end + 1);
    ++flat_size_;
    it->first = key;
    it->second = Extension();
    return {&it->second, true};
  }
  GrowCapacity(flat_size_ + 1);
  return Insert(key);
}

void ExtensionSet::GrowCapacity(size_t minimum_new_capacity) {
  if (PROTOBUF_PREDICT_FALSE(is_large())) {
    return;  // LargeMap does not have a "reserve" method.
  }
  if (flat_capacity_ >= minimum_new_capacity) {
    return;
  }

  auto new_flat_capacity = flat_capacity_;
  do {
    new_flat_capacity = new_flat_capacity == 0 ? 1 : new_flat_capacity * 4;
  } while (new_flat_capacity < minimum_new_capacity);

  const KeyValue* begin = flat_begin();
  const KeyValue* end = flat_end();
  AllocatedData new_map;
  if (new_flat_capacity > kMaximumFlatCapacity) {
    new_map.large = Arena::Create<LargeMap>(arena_);
    LargeMap::iterator hint = new_map.large->begin();
    for (const KeyValue* it = begin; it != end; ++it) {
      hint = new_map.large->insert(hint, {it->first, it->second});
    }
    flat_size_ = static_cast<uint16_t>(-1);
    GOOGLE_DCHECK(is_large());
  } else {
    new_map.flat = Arena::CreateArray<KeyValue>(arena_, new_flat_capacity);
    std::copy(begin, end, new_map.flat);
  }

  if (arena_ == nullptr) {
    DeleteFlatMap(begin, flat_capacity_);
  }
  flat_capacity_ = new_flat_capacity;
  map_ = new_map;
}

#if (__cplusplus < 201703) && \
    (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))
// static
constexpr uint16_t ExtensionSet::kMaximumFlatCapacity;
#endif  //  (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900
        //  && _MSC_VER < 1912))

void ExtensionSet::Erase(int key) {
  if (PROTOBUF_PREDICT_FALSE(is_large())) {
    map_.large->erase(key);
    return;
  }
  KeyValue* end = flat_end();
  KeyValue* it =
      std::lower_bound(flat_begin(), end, key, KeyValue::FirstComparator());
  if (it != end && it->first == key) {
    std::copy(it + 1, end, it);
    --flat_size_;
  }
}

// ==================================================================
// Default repeated field instances for iterator-compatible accessors

const RepeatedPrimitiveDefaults* RepeatedPrimitiveDefaults::default_instance() {
  static auto instance = OnShutdownDelete(new RepeatedPrimitiveDefaults);
  return instance;
}

const RepeatedStringTypeTraits::RepeatedFieldType*
RepeatedStringTypeTraits::GetDefaultRepeatedField() {
  static auto instance = OnShutdownDelete(new RepeatedFieldType);
  return instance;
}

uint8_t* ExtensionSet::Extension::InternalSerializeFieldWithCachedSizesToArray(
    const MessageLite* extendee, const ExtensionSet* extension_set, int number,
    uint8_t* target, io::EpsCopyOutputStream* stream) const {
  if (is_repeated) {
    if (is_packed) {
      if (cached_size == 0) return target;

      target = stream->EnsureSpace(target);
      target = WireFormatLite::WriteTagToArray(
          number, WireFormatLite::WIRETYPE_LENGTH_DELIMITED, target);
      target = WireFormatLite::WriteInt32NoTagToArray(cached_size, target);

      switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                 \
  case WireFormatLite::TYPE_##UPPERCASE:                             \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) { \
      target = stream->EnsureSpace(target);                          \
      target = WireFormatLite::Write##CAMELCASE##NoTagToArray(       \
          repeated_##LOWERCASE##_value->Get(i), target);             \
    }                                                                \
    break

        HANDLE_TYPE(INT32, Int32, int32_t);
        HANDLE_TYPE(INT64, Int64, int64_t);
        HANDLE_TYPE(UINT32, UInt32, uint32_t);
        HANDLE_TYPE(UINT64, UInt64, uint64_t);
        HANDLE_TYPE(SINT32, SInt32, int32_t);
        HANDLE_TYPE(SINT64, SInt64, int64_t);
        HANDLE_TYPE(FIXED32, Fixed32, uint32_t);
        HANDLE_TYPE(FIXED64, Fixed64, uint64_t);
        HANDLE_TYPE(SFIXED32, SFixed32, int32_t);
        HANDLE_TYPE(SFIXED64, SFixed64, int64_t);
        HANDLE_TYPE(FLOAT, Float, float);
        HANDLE_TYPE(DOUBLE, Double, double);
        HANDLE_TYPE(BOOL, Bool, bool);
        HANDLE_TYPE(ENUM, Enum, enum);
#undef HANDLE_TYPE

        case WireFormatLite::TYPE_STRING:
        case WireFormatLite::TYPE_BYTES:
        case WireFormatLite::TYPE_GROUP:
        case WireFormatLite::TYPE_MESSAGE:
          GOOGLE_LOG(FATAL) << "Non-primitive types can't be packed.";
          break;
      }
    } else {
      switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                 \
  case WireFormatLite::TYPE_##UPPERCASE:                             \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) { \
      target = stream->EnsureSpace(target);                          \
      target = WireFormatLite::Write##CAMELCASE##ToArray(            \
          number, repeated_##LOWERCASE##_value->Get(i), target);     \
    }                                                                \
    break

        HANDLE_TYPE(INT32, Int32, int32_t);
        HANDLE_TYPE(INT64, Int64, int64_t);
        HANDLE_TYPE(UINT32, UInt32, uint32_t);
        HANDLE_TYPE(UINT64, UInt64, uint64_t);
        HANDLE_TYPE(SINT32, SInt32, int32_t);
        HANDLE_TYPE(SINT64, SInt64, int64_t);
        HANDLE_TYPE(FIXED32, Fixed32, uint32_t);
        HANDLE_TYPE(FIXED64, Fixed64, uint64_t);
        HANDLE_TYPE(SFIXED32, SFixed32, int32_t);
        HANDLE_TYPE(SFIXED64, SFixed64, int64_t);
        HANDLE_TYPE(FLOAT, Float, float);
        HANDLE_TYPE(DOUBLE, Double, double);
        HANDLE_TYPE(BOOL, Bool, bool);
        HANDLE_TYPE(ENUM, Enum, enum);
#undef HANDLE_TYPE
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                 \
  case WireFormatLite::TYPE_##UPPERCASE:                             \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) { \
      target = stream->EnsureSpace(target);                          \
      target = stream->WriteString(                                  \
          number, repeated_##LOWERCASE##_value->Get(i), target);     \
    }                                                                \
    break
        HANDLE_TYPE(STRING, String, string);
        HANDLE_TYPE(BYTES, Bytes, string);
#undef HANDLE_TYPE
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, LOWERCASE)                     \
  case WireFormatLite::TYPE_##UPPERCASE:                                 \
    for (int i = 0; i < repeated_##LOWERCASE##_value->size(); i++) {     \
      target = stream->EnsureSpace(target);                              \
      target = WireFormatLite::InternalWrite##CAMELCASE(                 \
          number, repeated_##LOWERCASE##_value->Get(i), target, stream); \
    }                                                                    \
    break

        HANDLE_TYPE(GROUP, Group, message);
        HANDLE_TYPE(MESSAGE, Message, message);
#undef HANDLE_TYPE
      }
    }
  } else if (!is_cleared) {
    switch (real_type(type)) {
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, VALUE)                               \
  case WireFormatLite::TYPE_##UPPERCASE:                                       \
    target = stream->EnsureSpace(target);                                      \
    target = WireFormatLite::Write##CAMELCASE##ToArray(number, VALUE, target); \
    break

      HANDLE_TYPE(INT32, Int32, int32_t_value);
      HANDLE_TYPE(INT64, Int64, int64_t_value);
      HANDLE_TYPE(UINT32, UInt32, uint32_t_value);
      HANDLE_TYPE(UINT64, UInt64, uint64_t_value);
      HANDLE_TYPE(SINT32, SInt32, int32_t_value);
      HANDLE_TYPE(SINT64, SInt64, int64_t_value);
      HANDLE_TYPE(FIXED32, Fixed32, uint32_t_value);
      HANDLE_TYPE(FIXED64, Fixed64, uint64_t_value);
      HANDLE_TYPE(SFIXED32, SFixed32, int32_t_value);
      HANDLE_TYPE(SFIXED64, SFixed64, int64_t_value);
      HANDLE_TYPE(FLOAT, Float, float_value);
      HANDLE_TYPE(DOUBLE, Double, double_value);
      HANDLE_TYPE(BOOL, Bool, bool_value);
      HANDLE_TYPE(ENUM, Enum, enum_value);
#undef HANDLE_TYPE
#define HANDLE_TYPE(UPPERCASE, CAMELCASE, VALUE)         \
  case WireFormatLite::TYPE_##UPPERCASE:                 \
    target = stream->EnsureSpace(target);                \
    target = stream->WriteString(number, VALUE, target); \
    break
      HANDLE_TYPE(STRING, String, *string_value);
      HANDLE_TYPE(BYTES, Bytes, *string_value);
#undef HANDLE_TYPE
      case WireFormatLite::TYPE_GROUP:
        target = stream->EnsureSpace(target);
        target = WireFormatLite::InternalWriteGroup(number, *message_value,
                                                    target, stream);
        break;
      case WireFormatLite::TYPE_MESSAGE:
        if (is_lazy) {
          const auto* prototype =
              extension_set->GetPrototypeForLazyMessage(extendee, number);
          target = lazymessage_value->WriteMessageToArray(prototype, number,
                                                          target, stream);
        } else {
          target = stream->EnsureSpace(target);
          target = WireFormatLite::InternalWriteMessage(number, *message_value,
                                                        target, stream);
        }
        break;
    }
  }
  return target;
}

const MessageLite* ExtensionSet::GetPrototypeForLazyMessage(
    const MessageLite* extendee, int number) const {
  GeneratedExtensionFinder finder(extendee);
  bool was_packed_on_wire = false;
  ExtensionInfo extension_info;
  if (!FindExtensionInfoFromFieldNumber(
          WireFormatLite::WireType::WIRETYPE_LENGTH_DELIMITED, number, &finder,
          &extension_info, &was_packed_on_wire)) {
    return nullptr;
  }
  return extension_info.message_info.prototype;
}

uint8_t*
ExtensionSet::Extension::InternalSerializeMessageSetItemWithCachedSizesToArray(
    const MessageLite* extendee, const ExtensionSet* extension_set, int number,
    uint8_t* target, io::EpsCopyOutputStream* stream) const {
  if (type != WireFormatLite::TYPE_MESSAGE || is_repeated) {
    // Not a valid MessageSet extension, but serialize it the normal way.
    GOOGLE_LOG(WARNING) << "Invalid message set extension.";
    return InternalSerializeFieldWithCachedSizesToArray(extendee, extension_set,
                                                        number, target, stream);
  }

  if (is_cleared) return target;

  target = stream->EnsureSpace(target);
  // Start group.
  target = io::CodedOutputStream::WriteTagToArray(
      WireFormatLite::kMessageSetItemStartTag, target);
  // Write type ID.
  target = WireFormatLite::WriteUInt32ToArray(
      WireFormatLite::kMessageSetTypeIdNumber, number, target);
  // Write message.
  if (is_lazy) {
    const auto* prototype =
        extension_set->GetPrototypeForLazyMessage(extendee, number);
    target = lazymessage_value->WriteMessageToArray(
        prototype, WireFormatLite::kMessageSetMessageNumber, target, stream);
  } else {
    target = WireFormatLite::InternalWriteMessage(
        WireFormatLite::kMessageSetMessageNumber, *message_value, target,
        stream);
  }
  // End group.
  target = stream->EnsureSpace(target);
  target = io::CodedOutputStream::WriteTagToArray(
      WireFormatLite::kMessageSetItemEndTag, target);
  return target;
}

size_t ExtensionSet::Extension::MessageSetItemByteSize(int number) const {
  if (type != WireFormatLite::TYPE_MESSAGE || is_repeated) {
    // Not a valid MessageSet extension, but compute the byte size for it the
    // normal way.
    return ByteSize(number);
  }

  if (is_cleared) return 0;

  size_t our_size = WireFormatLite::kMessageSetItemTagsSize;

  // type_id
  our_size += io::CodedOutputStream::VarintSize32(number);

  // message
  size_t message_size = 0;
  if (is_lazy) {
    message_size = lazymessage_value->ByteSizeLong();
  } else {
    message_size = message_value->ByteSizeLong();
  }

  our_size += io::CodedOutputStream::VarintSize32(message_size);
  our_size += message_size;

  return our_size;
}

size_t ExtensionSet::MessageSetByteSize() const {
  size_t total_size = 0;
  ForEach([&total_size](int number, const Extension& ext) {
    total_size += ext.MessageSetItemByteSize(number);
  });
  return total_size;
}


}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
