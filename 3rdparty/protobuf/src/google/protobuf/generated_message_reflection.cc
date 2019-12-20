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

#include <algorithm>
#include <set>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/map_field.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format.h>


#define GOOGLE_PROTOBUF_HAS_ONEOF

namespace google {
namespace protobuf {
namespace internal {

namespace {
bool IsMapFieldInApi(const FieldDescriptor* field) {
  return field->is_map();
}
}  // anonymous namespace

bool ParseNamedEnum(const EnumDescriptor* descriptor,
                    const string& name,
                    int* value) {
  const EnumValueDescriptor* d = descriptor->FindValueByName(name);
  if (d == NULL) return false;
  *value = d->number();
  return true;
}

const string& NameOfEnum(const EnumDescriptor* descriptor, int value) {
  const EnumValueDescriptor* d = descriptor->FindValueByNumber(value);
  return (d == NULL ? GetEmptyString() : d->name());
}

// ===================================================================
// Helpers for reporting usage errors (e.g. trying to use GetInt32() on
// a string field).

namespace {

template <class To>
To* GetPointerAtOffset(Message* message, uint32 offset) {
  return reinterpret_cast<To*>(reinterpret_cast<char*>(message) + offset);
}

template <class To>
const To* GetConstPointerAtOffset(const Message* message, uint32 offset) {
  return reinterpret_cast<const To*>(reinterpret_cast<const char*>(message) +
                                     offset);
}

template <class To>
const To& GetConstRefAtOffset(const Message& message, uint32 offset) {
  return *GetConstPointerAtOffset<To>(&message, offset);
}

void ReportReflectionUsageError(
    const Descriptor* descriptor, const FieldDescriptor* field,
    const char* method, const char* description) {
  GOOGLE_LOG(FATAL)
    << "Protocol Buffer reflection usage error:\n"
       "  Method      : google::protobuf::Reflection::" << method << "\n"
       "  Message type: " << descriptor->full_name() << "\n"
       "  Field       : " << field->full_name() << "\n"
       "  Problem     : " << description;
}

const char* cpptype_names_[FieldDescriptor::MAX_CPPTYPE + 1] = {
  "INVALID_CPPTYPE",
  "CPPTYPE_INT32",
  "CPPTYPE_INT64",
  "CPPTYPE_UINT32",
  "CPPTYPE_UINT64",
  "CPPTYPE_DOUBLE",
  "CPPTYPE_FLOAT",
  "CPPTYPE_BOOL",
  "CPPTYPE_ENUM",
  "CPPTYPE_STRING",
  "CPPTYPE_MESSAGE"
};

static void ReportReflectionUsageTypeError(
    const Descriptor* descriptor, const FieldDescriptor* field,
    const char* method,
    FieldDescriptor::CppType expected_type) {
  GOOGLE_LOG(FATAL)
    << "Protocol Buffer reflection usage error:\n"
       "  Method      : google::protobuf::Reflection::" << method << "\n"
       "  Message type: " << descriptor->full_name() << "\n"
       "  Field       : " << field->full_name() << "\n"
       "  Problem     : Field is not the right type for this message:\n"
       "    Expected  : " << cpptype_names_[expected_type] << "\n"
       "    Field type: " << cpptype_names_[field->cpp_type()];
}

static void ReportReflectionUsageEnumTypeError(
    const Descriptor* descriptor, const FieldDescriptor* field,
    const char* method, const EnumValueDescriptor* value) {
  GOOGLE_LOG(FATAL)
    << "Protocol Buffer reflection usage error:\n"
       "  Method      : google::protobuf::Reflection::" << method << "\n"
       "  Message type: " << descriptor->full_name() << "\n"
       "  Field       : " << field->full_name() << "\n"
       "  Problem     : Enum value did not match field type:\n"
       "    Expected  : " << field->enum_type()->full_name() << "\n"
       "    Actual    : " << value->full_name();
}

#define USAGE_CHECK(CONDITION, METHOD, ERROR_DESCRIPTION)                      \
  if (!(CONDITION))                                                            \
    ReportReflectionUsageError(descriptor_, field, #METHOD, ERROR_DESCRIPTION)
#define USAGE_CHECK_EQ(A, B, METHOD, ERROR_DESCRIPTION)                        \
  USAGE_CHECK((A) == (B), METHOD, ERROR_DESCRIPTION)
#define USAGE_CHECK_NE(A, B, METHOD, ERROR_DESCRIPTION)                        \
  USAGE_CHECK((A) != (B), METHOD, ERROR_DESCRIPTION)

#define USAGE_CHECK_TYPE(METHOD, CPPTYPE)                                      \
  if (field->cpp_type() != FieldDescriptor::CPPTYPE_##CPPTYPE)                 \
    ReportReflectionUsageTypeError(descriptor_, field, #METHOD,                \
                                   FieldDescriptor::CPPTYPE_##CPPTYPE)

#define USAGE_CHECK_ENUM_VALUE(METHOD)                                         \
  if (value->type() != field->enum_type())                                     \
    ReportReflectionUsageEnumTypeError(descriptor_, field, #METHOD, value)

#define USAGE_CHECK_MESSAGE_TYPE(METHOD)                                       \
  USAGE_CHECK_EQ(field->containing_type(), descriptor_,                        \
                 METHOD, "Field does not match message type.");
#define USAGE_CHECK_SINGULAR(METHOD)                                           \
  USAGE_CHECK_NE(field->label(), FieldDescriptor::LABEL_REPEATED, METHOD,      \
                 "Field is repeated; the method requires a singular field.")
#define USAGE_CHECK_REPEATED(METHOD)                                           \
  USAGE_CHECK_EQ(field->label(), FieldDescriptor::LABEL_REPEATED, METHOD,      \
                 "Field is singular; the method requires a repeated field.")

#define USAGE_CHECK_ALL(METHOD, LABEL, CPPTYPE)                       \
    USAGE_CHECK_MESSAGE_TYPE(METHOD);                                 \
    USAGE_CHECK_##LABEL(METHOD);                                      \
    USAGE_CHECK_TYPE(METHOD, CPPTYPE)

}  // namespace

// ===================================================================

GeneratedMessageReflection::GeneratedMessageReflection(
    const Descriptor* descriptor, const ReflectionSchema& schema,
    const DescriptorPool* pool, MessageFactory* factory)
    : descriptor_(descriptor),
      schema_(schema),
      descriptor_pool_((pool == NULL) ? DescriptorPool::generated_pool()
                                      : pool),
      message_factory_(factory),
      last_non_weak_field_index_(-1) {
  last_non_weak_field_index_ = descriptor_->field_count() - 1;
}

GeneratedMessageReflection::~GeneratedMessageReflection() {}

const UnknownFieldSet& GeneratedMessageReflection::GetUnknownFields(
    const Message& message) const {
  if (descriptor_->file()->syntax() == FileDescriptor::SYNTAX_PROTO3 &&
      !GetProto3PreserveUnknownsDefault()) {
    // We have to ensure that any mutations made to the return value of
    // MutableUnknownFields() are not reflected here when Proto3 defaults to
    // discard unknowns.
    return *UnknownFieldSet::default_instance();
  } else {
    return GetInternalMetadataWithArena(message).unknown_fields();
  }
}

UnknownFieldSet* GeneratedMessageReflection::MutableUnknownFields(
    Message* message) const {
  return MutableInternalMetadataWithArena(message)->mutable_unknown_fields();
}

size_t GeneratedMessageReflection::SpaceUsedLong(const Message& message) const {
  // object_size_ already includes the in-memory representation of each field
  // in the message, so we only need to account for additional memory used by
  // the fields.
  size_t total_size = schema_.GetObjectSize();

  total_size += GetUnknownFields(message).SpaceUsedExcludingSelfLong();

  if (schema_.HasExtensionSet()) {
    total_size += GetExtensionSet(message).SpaceUsedExcludingSelfLong();
  }
  for (int i = 0; i <= last_non_weak_field_index_; i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->is_repeated()) {
      switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
        case FieldDescriptor::CPPTYPE_##UPPERCASE :                           \
          total_size += GetRaw<RepeatedField<LOWERCASE> >(message, field)     \
                          .SpaceUsedExcludingSelfLong();                      \
          break

        HANDLE_TYPE( INT32,  int32);
        HANDLE_TYPE( INT64,  int64);
        HANDLE_TYPE(UINT32, uint32);
        HANDLE_TYPE(UINT64, uint64);
        HANDLE_TYPE(DOUBLE, double);
        HANDLE_TYPE( FLOAT,  float);
        HANDLE_TYPE(  BOOL,   bool);
        HANDLE_TYPE(  ENUM,    int);
#undef HANDLE_TYPE

        case FieldDescriptor::CPPTYPE_STRING:
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING:
              total_size += GetRaw<RepeatedPtrField<string> >(message, field)
                                .SpaceUsedExcludingSelfLong();
              break;
          }
          break;

        case FieldDescriptor::CPPTYPE_MESSAGE:
          if (IsMapFieldInApi(field)) {
            total_size += GetRaw<MapFieldBase>(message, field)
                              .SpaceUsedExcludingSelfLong();
          } else {
            // We don't know which subclass of RepeatedPtrFieldBase the type is,
            // so we use RepeatedPtrFieldBase directly.
            total_size +=
                GetRaw<RepeatedPtrFieldBase>(message, field)
                    .SpaceUsedExcludingSelfLong<GenericTypeHandler<Message> >();
          }

          break;
      }
    } else {
      if (field->containing_oneof() && !HasOneofField(message, field)) {
        continue;
      }
      switch (field->cpp_type()) {
        case FieldDescriptor::CPPTYPE_INT32 :
        case FieldDescriptor::CPPTYPE_INT64 :
        case FieldDescriptor::CPPTYPE_UINT32:
        case FieldDescriptor::CPPTYPE_UINT64:
        case FieldDescriptor::CPPTYPE_DOUBLE:
        case FieldDescriptor::CPPTYPE_FLOAT :
        case FieldDescriptor::CPPTYPE_BOOL  :
        case FieldDescriptor::CPPTYPE_ENUM  :
          // Field is inline, so we've already counted it.
          break;

        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              // Initially, the string points to the default value stored in
              // the prototype. Only count the string if it has been changed
              // from the default value.
              const string* default_ptr =
                  &DefaultRaw<ArenaStringPtr>(field).Get();
              const string* ptr =
                  &GetField<ArenaStringPtr>(message, field).Get();

              if (ptr != default_ptr) {
                // string fields are represented by just a pointer, so also
                // include sizeof(string) as well.
                total_size +=
                    sizeof(*ptr) + StringSpaceUsedExcludingSelfLong(*ptr);
              }
              break;
            }
          }
          break;
        }

        case FieldDescriptor::CPPTYPE_MESSAGE:
          if (schema_.IsDefaultInstance(message)) {
            // For singular fields, the prototype just stores a pointer to the
            // external type's prototype, so there is no extra memory usage.
          } else {
            const Message* sub_message = GetRaw<const Message*>(message, field);
            if (sub_message != NULL) {
              total_size += sub_message->SpaceUsedLong();
            }
          }
          break;
      }
    }
  }
  return total_size;
}

void GeneratedMessageReflection::SwapField(
    Message* message1,
    Message* message2,
    const FieldDescriptor* field) const {
  if (field->is_repeated()) {
    switch (field->cpp_type()) {
#define SWAP_ARRAYS(CPPTYPE, TYPE)                                      \
      case FieldDescriptor::CPPTYPE_##CPPTYPE:                          \
        MutableRaw<RepeatedField<TYPE> >(message1, field)->Swap(        \
            MutableRaw<RepeatedField<TYPE> >(message2, field));         \
        break;

      SWAP_ARRAYS(INT32 , int32 );
      SWAP_ARRAYS(INT64 , int64 );
      SWAP_ARRAYS(UINT32, uint32);
      SWAP_ARRAYS(UINT64, uint64);
      SWAP_ARRAYS(FLOAT , float );
      SWAP_ARRAYS(DOUBLE, double);
      SWAP_ARRAYS(BOOL  , bool  );
      SWAP_ARRAYS(ENUM  , int   );
#undef SWAP_ARRAYS

      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            MutableRaw<RepeatedPtrFieldBase>(message1, field)->
                Swap<GenericTypeHandler<string> >(
                    MutableRaw<RepeatedPtrFieldBase>(message2, field));
            break;
        }
        break;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (IsMapFieldInApi(field)) {
          MutableRaw<MapFieldBase>(message1, field)->
            MutableRepeatedField()->
              Swap<GenericTypeHandler<google::protobuf::Message> >(
                MutableRaw<MapFieldBase>(message2, field)->
                  MutableRepeatedField());
        } else {
          MutableRaw<RepeatedPtrFieldBase>(message1, field)->
            Swap<GenericTypeHandler<google::protobuf::Message> >(
              MutableRaw<RepeatedPtrFieldBase>(message2, field));
        }
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field->cpp_type();
    }
  } else {
    switch (field->cpp_type()) {
#define SWAP_VALUES(CPPTYPE, TYPE)                                      \
      case FieldDescriptor::CPPTYPE_##CPPTYPE:                          \
        std::swap(*MutableRaw<TYPE>(message1, field),                   \
                  *MutableRaw<TYPE>(message2, field));                  \
        break;

      SWAP_VALUES(INT32 , int32 );
      SWAP_VALUES(INT64 , int64 );
      SWAP_VALUES(UINT32, uint32);
      SWAP_VALUES(UINT64, uint64);
      SWAP_VALUES(FLOAT , float );
      SWAP_VALUES(DOUBLE, double);
      SWAP_VALUES(BOOL  , bool  );
      SWAP_VALUES(ENUM  , int   );
#undef SWAP_VALUES
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (GetArena(message1) == GetArena(message2)) {
          std::swap(*MutableRaw<Message*>(message1, field),
                    *MutableRaw<Message*>(message2, field));
        } else {
          Message** sub_msg1 = MutableRaw<Message*>(message1, field);
          Message** sub_msg2 = MutableRaw<Message*>(message2, field);
          if (*sub_msg1 == NULL && *sub_msg2 == NULL) break;
          if (*sub_msg1 && *sub_msg2) {
            (*sub_msg1)->GetReflection()->Swap(*sub_msg1, *sub_msg2);
            break;
          }
          if (*sub_msg1 == NULL) {
            *sub_msg1 = (*sub_msg2)->New(message1->GetArena());
            (*sub_msg1)->CopyFrom(**sub_msg2);
            ClearField(message2, field);
          } else {
            *sub_msg2 = (*sub_msg1)->New(message2->GetArena());
            (*sub_msg2)->CopyFrom(**sub_msg1);
            ClearField(message1, field);
          }
        }
        break;

      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            {
              Arena* arena1 = GetArena(message1);
              Arena* arena2 = GetArena(message2);
              ArenaStringPtr* string1 =
                  MutableRaw<ArenaStringPtr>(message1, field);
              ArenaStringPtr* string2 =
                  MutableRaw<ArenaStringPtr>(message2, field);
              if (arena1 == arena2) {
                string1->Swap(string2);
              } else {
                const string* default_ptr =
                    &DefaultRaw<ArenaStringPtr>(field).Get();
                const string temp = string1->Get();
                string1->Set(default_ptr, string2->Get(), arena1);
                string2->Set(default_ptr, temp, arena2);
              }
            }
            break;
        }
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field->cpp_type();
    }
  }
}

void GeneratedMessageReflection::SwapOneofField(
    Message* message1,
    Message* message2,
    const OneofDescriptor* oneof_descriptor) const {
  uint32 oneof_case1 = GetOneofCase(*message1, oneof_descriptor);
  uint32 oneof_case2 = GetOneofCase(*message2, oneof_descriptor);

  int32 temp_int32;
  int64 temp_int64;
  uint32 temp_uint32;
  uint64 temp_uint64;
  float temp_float;
  double temp_double;
  bool temp_bool;
  int temp_int;
  Message* temp_message = NULL;
  string temp_string;

  // Stores message1's oneof field to a temp variable.
  const FieldDescriptor* field1 = NULL;
  if (oneof_case1 > 0) {
    field1 = descriptor_->FindFieldByNumber(oneof_case1);
    //oneof_descriptor->field(oneof_case1);
    switch (field1->cpp_type()) {
#define GET_TEMP_VALUE(CPPTYPE, TYPE)                                   \
      case FieldDescriptor::CPPTYPE_##CPPTYPE:                          \
        temp_##TYPE = GetField<TYPE>(*message1, field1);                \
        break;

      GET_TEMP_VALUE(INT32 , int32 );
      GET_TEMP_VALUE(INT64 , int64 );
      GET_TEMP_VALUE(UINT32, uint32);
      GET_TEMP_VALUE(UINT64, uint64);
      GET_TEMP_VALUE(FLOAT , float );
      GET_TEMP_VALUE(DOUBLE, double);
      GET_TEMP_VALUE(BOOL  , bool  );
      GET_TEMP_VALUE(ENUM  , int   );
#undef GET_TEMP_VALUE
      case FieldDescriptor::CPPTYPE_MESSAGE:
        temp_message = ReleaseMessage(message1, field1);
        break;

      case FieldDescriptor::CPPTYPE_STRING:
        temp_string = GetString(*message1, field1);
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field1->cpp_type();
    }
  }

  // Sets message1's oneof field from the message2's oneof field.
  if (oneof_case2 > 0) {
    const FieldDescriptor* field2 =
        descriptor_->FindFieldByNumber(oneof_case2);
    switch (field2->cpp_type()) {
#define SET_ONEOF_VALUE1(CPPTYPE, TYPE)                                 \
      case FieldDescriptor::CPPTYPE_##CPPTYPE:                          \
        SetField<TYPE>(message1, field2, GetField<TYPE>(*message2, field2)); \
        break;

      SET_ONEOF_VALUE1(INT32 , int32 );
      SET_ONEOF_VALUE1(INT64 , int64 );
      SET_ONEOF_VALUE1(UINT32, uint32);
      SET_ONEOF_VALUE1(UINT64, uint64);
      SET_ONEOF_VALUE1(FLOAT , float );
      SET_ONEOF_VALUE1(DOUBLE, double);
      SET_ONEOF_VALUE1(BOOL  , bool  );
      SET_ONEOF_VALUE1(ENUM  , int   );
#undef SET_ONEOF_VALUE1
      case FieldDescriptor::CPPTYPE_MESSAGE:
        SetAllocatedMessage(message1,
                            ReleaseMessage(message2, field2),
                            field2);
        break;

      case FieldDescriptor::CPPTYPE_STRING:
        SetString(message1, field2, GetString(*message2, field2));
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field2->cpp_type();
    }
  } else {
    ClearOneof(message1, oneof_descriptor);
  }

  // Sets message2's oneof field from the temp variable.
  if (oneof_case1 > 0) {
    switch (field1->cpp_type()) {
#define SET_ONEOF_VALUE2(CPPTYPE, TYPE)                                 \
      case FieldDescriptor::CPPTYPE_##CPPTYPE:                          \
        SetField<TYPE>(message2, field1, temp_##TYPE);                  \
        break;

      SET_ONEOF_VALUE2(INT32 , int32 );
      SET_ONEOF_VALUE2(INT64 , int64 );
      SET_ONEOF_VALUE2(UINT32, uint32);
      SET_ONEOF_VALUE2(UINT64, uint64);
      SET_ONEOF_VALUE2(FLOAT , float );
      SET_ONEOF_VALUE2(DOUBLE, double);
      SET_ONEOF_VALUE2(BOOL  , bool  );
      SET_ONEOF_VALUE2(ENUM  , int   );
#undef SET_ONEOF_VALUE2
      case FieldDescriptor::CPPTYPE_MESSAGE:
        SetAllocatedMessage(message2, temp_message, field1);
        break;

      case FieldDescriptor::CPPTYPE_STRING:
        SetString(message2, field1, temp_string);
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field1->cpp_type();
    }
  } else {
    ClearOneof(message2, oneof_descriptor);
  }
}

void GeneratedMessageReflection::Swap(
    Message* message1,
    Message* message2) const {
  if (message1 == message2) return;

  // TODO(kenton):  Other Reflection methods should probably check this too.
  GOOGLE_CHECK_EQ(message1->GetReflection(), this)
    << "First argument to Swap() (of type \""
    << message1->GetDescriptor()->full_name()
    << "\") is not compatible with this reflection object (which is for type \""
    << descriptor_->full_name()
    << "\").  Note that the exact same class is required; not just the same "
       "descriptor.";
  GOOGLE_CHECK_EQ(message2->GetReflection(), this)
    << "Second argument to Swap() (of type \""
    << message2->GetDescriptor()->full_name()
    << "\") is not compatible with this reflection object (which is for type \""
    << descriptor_->full_name()
    << "\").  Note that the exact same class is required; not just the same "
       "descriptor.";

  // Check that both messages are in the same arena (or both on the heap). We
  // need to copy all data if not, due to ownership semantics.
  if (GetArena(message1) != GetArena(message2)) {
    // Slow copy path.
    // Use our arena as temp space, if available.
    Message* temp = message1->New(GetArena(message1));
    temp->MergeFrom(*message2);
    message2->CopyFrom(*message1);
    Swap(message1, temp);
    if (GetArena(message1) == NULL) {
      delete temp;
    }
    return;
  }

  if (schema_.HasHasbits()) {
    uint32* has_bits1 = MutableHasBits(message1);
    uint32* has_bits2 = MutableHasBits(message2);

    int fields_with_has_bits = 0;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      if (field->is_repeated() || field->containing_oneof()) {
        continue;
      }
      fields_with_has_bits++;
    }

    int has_bits_size = (fields_with_has_bits + 31) / 32;

    for (int i = 0; i < has_bits_size; i++) {
      std::swap(has_bits1[i], has_bits2[i]);
    }
  }

  for (int i = 0; i <= last_non_weak_field_index_; i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->containing_oneof()) continue;
    SwapField(message1, message2, field);
  }
  const int oneof_decl_count = descriptor_->oneof_decl_count();
  for (int i = 0; i < oneof_decl_count; i++) {
    SwapOneofField(message1, message2, descriptor_->oneof_decl(i));
  }

  if (schema_.HasExtensionSet()) {
    MutableExtensionSet(message1)->Swap(MutableExtensionSet(message2));
  }

  MutableUnknownFields(message1)->Swap(MutableUnknownFields(message2));
}

void GeneratedMessageReflection::SwapFields(
    Message* message1,
    Message* message2,
    const std::vector<const FieldDescriptor*>& fields) const {
  if (message1 == message2) return;

  // TODO(kenton):  Other Reflection methods should probably check this too.
  GOOGLE_CHECK_EQ(message1->GetReflection(), this)
    << "First argument to SwapFields() (of type \""
    << message1->GetDescriptor()->full_name()
    << "\") is not compatible with this reflection object (which is for type \""
    << descriptor_->full_name()
    << "\").  Note that the exact same class is required; not just the same "
       "descriptor.";
  GOOGLE_CHECK_EQ(message2->GetReflection(), this)
    << "Second argument to SwapFields() (of type \""
    << message2->GetDescriptor()->full_name()
    << "\") is not compatible with this reflection object (which is for type \""
    << descriptor_->full_name()
    << "\").  Note that the exact same class is required; not just the same "
       "descriptor.";

  std::set<int> swapped_oneof;

  const int fields_size = static_cast<int>(fields.size());
  for (int i = 0; i < fields_size; i++) {
    const FieldDescriptor* field = fields[i];
    if (field->is_extension()) {
      MutableExtensionSet(message1)->SwapExtension(
          MutableExtensionSet(message2),
          field->number());
    } else {
      if (field->containing_oneof()) {
        int oneof_index = field->containing_oneof()->index();
        // Only swap the oneof field once.
        if (swapped_oneof.find(oneof_index) != swapped_oneof.end()) {
          continue;
        }
        swapped_oneof.insert(oneof_index);
        SwapOneofField(message1, message2, field->containing_oneof());
      } else {
        // Swap has bit for non-repeated fields.  We have already checked for
        // oneof already.
        if (!field->is_repeated()) {
          SwapBit(message1, message2, field);
        }
        // Swap field.
        SwapField(message1, message2, field);
      }
    }
  }
}

// -------------------------------------------------------------------

bool GeneratedMessageReflection::HasField(const Message& message,
                                          const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(HasField);
  USAGE_CHECK_SINGULAR(HasField);

  if (field->is_extension()) {
    return GetExtensionSet(message).Has(field->number());
  } else {
    if (field->containing_oneof()) {
      return HasOneofField(message, field);
    } else {
      return HasBit(message, field);
    }
  }
}

int GeneratedMessageReflection::FieldSize(const Message& message,
                                          const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(FieldSize);
  USAGE_CHECK_REPEATED(FieldSize);

  if (field->is_extension()) {
    return GetExtensionSet(message).ExtensionSize(field->number());
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
      case FieldDescriptor::CPPTYPE_##UPPERCASE :                             \
        return GetRaw<RepeatedField<LOWERCASE> >(message, field).size()

      HANDLE_TYPE( INT32,  int32);
      HANDLE_TYPE( INT64,  int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE( FLOAT,  float);
      HANDLE_TYPE(  BOOL,   bool);
      HANDLE_TYPE(  ENUM,    int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (IsMapFieldInApi(field)) {
          return GetRaw<MapFieldBase>(message, field).GetRepeatedField().size();
        } else {
          return GetRaw<RepeatedPtrFieldBase>(message, field).size();
        }
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return 0;
  }
}

void GeneratedMessageReflection::ClearField(
    Message* message, const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(ClearField);

  if (field->is_extension()) {
    MutableExtensionSet(message)->ClearExtension(field->number());
  } else if (!field->is_repeated()) {
    if (field->containing_oneof()) {
      ClearOneofField(message, field);
      return;
    }
    if (HasBit(*message, field)) {
      ClearBit(message, field);

      // We need to set the field back to its default value.
      switch (field->cpp_type()) {
#define CLEAR_TYPE(CPPTYPE, TYPE)                                            \
        case FieldDescriptor::CPPTYPE_##CPPTYPE:                             \
          *MutableRaw<TYPE>(message, field) =                                \
            field->default_value_##TYPE();                                   \
          break;

        CLEAR_TYPE(INT32 , int32 );
        CLEAR_TYPE(INT64 , int64 );
        CLEAR_TYPE(UINT32, uint32);
        CLEAR_TYPE(UINT64, uint64);
        CLEAR_TYPE(FLOAT , float );
        CLEAR_TYPE(DOUBLE, double);
        CLEAR_TYPE(BOOL  , bool  );
#undef CLEAR_TYPE

        case FieldDescriptor::CPPTYPE_ENUM:
          *MutableRaw<int>(message, field) =
            field->default_value_enum()->number();
          break;

        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              const string* default_ptr =
                  &DefaultRaw<ArenaStringPtr>(field).Get();
              MutableRaw<ArenaStringPtr>(message, field)->SetAllocated(
                  default_ptr, NULL, GetArena(message));
              break;
            }
          }
          break;
        }

        case FieldDescriptor::CPPTYPE_MESSAGE:
          if (!schema_.HasHasbits()) {
            // Proto3 does not have has-bits and we need to set a message field
            // to NULL in order to indicate its un-presence.
            if (GetArena(message) == NULL) {
              delete *MutableRaw<Message*>(message, field);
            }
            *MutableRaw<Message*>(message, field) = NULL;
          } else {
            (*MutableRaw<Message*>(message, field))->Clear();
          }
          break;
      }
    }
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
      case FieldDescriptor::CPPTYPE_##UPPERCASE :                             \
        MutableRaw<RepeatedField<LOWERCASE> >(message, field)->Clear();       \
        break

      HANDLE_TYPE( INT32,  int32);
      HANDLE_TYPE( INT64,  int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE( FLOAT,  float);
      HANDLE_TYPE(  BOOL,   bool);
      HANDLE_TYPE(  ENUM,    int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING: {
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            MutableRaw<RepeatedPtrField<string> >(message, field)->Clear();
            break;
        }
        break;
      }

      case FieldDescriptor::CPPTYPE_MESSAGE: {
        if (IsMapFieldInApi(field)) {
          MutableRaw<MapFieldBase>(message, field)
              ->MutableRepeatedField()
              ->Clear<GenericTypeHandler<Message> >();
        } else {
          // We don't know which subclass of RepeatedPtrFieldBase the type is,
          // so we use RepeatedPtrFieldBase directly.
          MutableRaw<RepeatedPtrFieldBase>(message, field)
              ->Clear<GenericTypeHandler<Message> >();
        }
        break;
      }
    }
  }
}

void GeneratedMessageReflection::RemoveLast(
    Message* message,
    const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(RemoveLast);
  USAGE_CHECK_REPEATED(RemoveLast);

  if (field->is_extension()) {
    MutableExtensionSet(message)->RemoveLast(field->number());
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
      case FieldDescriptor::CPPTYPE_##UPPERCASE :                             \
        MutableRaw<RepeatedField<LOWERCASE> >(message, field)->RemoveLast();  \
        break

      HANDLE_TYPE( INT32,  int32);
      HANDLE_TYPE( INT64,  int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE( FLOAT,  float);
      HANDLE_TYPE(  BOOL,   bool);
      HANDLE_TYPE(  ENUM,    int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            MutableRaw<RepeatedPtrField<string> >(message, field)->RemoveLast();
            break;
        }
        break;

      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (IsMapFieldInApi(field)) {
          MutableRaw<MapFieldBase>(message, field)
              ->MutableRepeatedField()
              ->RemoveLast<GenericTypeHandler<Message> >();
        } else {
          MutableRaw<RepeatedPtrFieldBase>(message, field)
            ->RemoveLast<GenericTypeHandler<Message> >();
        }
        break;
    }
  }
}

Message* GeneratedMessageReflection::ReleaseLast(
    Message* message,
    const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(ReleaseLast, REPEATED, MESSAGE);

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->ReleaseLast(field->number()));
  } else {
    if (IsMapFieldInApi(field)) {
      return MutableRaw<MapFieldBase>(message, field)
          ->MutableRepeatedField()
          ->ReleaseLast<GenericTypeHandler<Message> >();
    } else {
      return MutableRaw<RepeatedPtrFieldBase>(message, field)
        ->ReleaseLast<GenericTypeHandler<Message> >();
    }
  }
}

void GeneratedMessageReflection::SwapElements(
    Message* message,
    const FieldDescriptor* field,
    int index1,
    int index2) const {
  USAGE_CHECK_MESSAGE_TYPE(Swap);
  USAGE_CHECK_REPEATED(Swap);

  if (field->is_extension()) {
    MutableExtensionSet(message)->SwapElements(field->number(), index1, index2);
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
      case FieldDescriptor::CPPTYPE_##UPPERCASE :                             \
        MutableRaw<RepeatedField<LOWERCASE> >(message, field)                 \
            ->SwapElements(index1, index2);                                   \
        break

      HANDLE_TYPE( INT32,  int32);
      HANDLE_TYPE( INT64,  int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE( FLOAT,  float);
      HANDLE_TYPE(  BOOL,   bool);
      HANDLE_TYPE(  ENUM,    int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (IsMapFieldInApi(field)) {
          MutableRaw<MapFieldBase>(message, field)
              ->MutableRepeatedField()
              ->SwapElements(index1, index2);
        } else {
          MutableRaw<RepeatedPtrFieldBase>(message, field)
            ->SwapElements(index1, index2);
        }
        break;
    }
  }
}

namespace {
// Comparison functor for sorting FieldDescriptors by field number.
struct FieldNumberSorter {
  bool operator()(const FieldDescriptor* left,
                  const FieldDescriptor* right) const {
    return left->number() < right->number();
  }
};

inline bool IsIndexInHasBitSet(
    const uint32* has_bit_set, uint32 has_bit_index) {
  GOOGLE_DCHECK_NE(has_bit_index, ~0u);
  return ((has_bit_set[has_bit_index / 32] >> (has_bit_index % 32)) &
          static_cast<uint32>(1)) != 0;
}
}  // namespace

void GeneratedMessageReflection::ListFields(
    const Message& message,
    std::vector<const FieldDescriptor*>* output) const {
  output->clear();

  // Optimization:  The default instance never has any fields set.
  if (schema_.IsDefaultInstance(message)) return;

  // Optimization: Avoid calling GetHasBits() and HasOneofField() many times
  // within the field loop.  We allow this violation of ReflectionSchema
  // encapsulation because this function takes a noticable about of CPU
  // fleetwide and properly allowing this optimization through public interfaces
  // seems more trouble than it is worth.
  const uint32* const has_bits =
      schema_.HasHasbits() ? GetHasBits(message) : NULL;
  const uint32* const has_bits_indices = schema_.has_bit_indices_;
  const uint32* const oneof_case_array =
      GetConstPointerAtOffset<uint32>(&message, schema_.oneof_case_offset_);
  output->reserve(descriptor_->field_count());
  for (int i = 0; i <= last_non_weak_field_index_; i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->is_repeated()) {
      if (FieldSize(message, field) > 0) {
        output->push_back(field);
      }
    } else {
      const OneofDescriptor* containing_oneof = field->containing_oneof();
      if (containing_oneof) {
        // Equivalent to: HasOneofField(message, field)
        if (oneof_case_array[containing_oneof->index()] == field->number()) {
          output->push_back(field);
        }
      } else if (has_bits) {
        // Equivalent to: HasBit(message, field)
        if (IsIndexInHasBitSet(has_bits, has_bits_indices[i])) {
          output->push_back(field);
        }
      } else if (HasBit(message, field)) {  // Fall back on proto3-style HasBit.
        output->push_back(field);
      }
    }
  }
  if (schema_.HasExtensionSet()) {
    GetExtensionSet(message).AppendToList(descriptor_, descriptor_pool_,
                                          output);
  }

  // ListFields() must sort output by field number.
  std::sort(output->begin(), output->end(), FieldNumberSorter());
}

// -------------------------------------------------------------------

#undef DEFINE_PRIMITIVE_ACCESSORS
#define DEFINE_PRIMITIVE_ACCESSORS(TYPENAME, TYPE, PASSTYPE, CPPTYPE)        \
  PASSTYPE GeneratedMessageReflection::Get##TYPENAME(                        \
      const Message& message, const FieldDescriptor* field) const {          \
    USAGE_CHECK_ALL(Get##TYPENAME, SINGULAR, CPPTYPE);                       \
    if (field->is_extension()) {                                             \
      return GetExtensionSet(message).Get##TYPENAME(                         \
        field->number(), field->default_value_##PASSTYPE());                 \
    } else {                                                                 \
      return GetField<TYPE>(message, field);                                 \
    }                                                                        \
  }                                                                          \
                                                                             \
  void GeneratedMessageReflection::Set##TYPENAME(                            \
      Message* message, const FieldDescriptor* field,                        \
      PASSTYPE value) const {                                                \
    USAGE_CHECK_ALL(Set##TYPENAME, SINGULAR, CPPTYPE);                       \
    if (field->is_extension()) {                                             \
      return MutableExtensionSet(message)->Set##TYPENAME(                    \
        field->number(), field->type(), value, field);                       \
    } else {                                                                 \
      SetField<TYPE>(message, field, value);                                 \
    }                                                                        \
  }                                                                          \
                                                                             \
  PASSTYPE GeneratedMessageReflection::GetRepeated##TYPENAME(                \
      const Message& message,                                                \
      const FieldDescriptor* field, int index) const {                       \
    USAGE_CHECK_ALL(GetRepeated##TYPENAME, REPEATED, CPPTYPE);               \
    if (field->is_extension()) {                                             \
      return GetExtensionSet(message).GetRepeated##TYPENAME(                 \
        field->number(), index);                                             \
    } else {                                                                 \
      return GetRepeatedField<TYPE>(message, field, index);                  \
    }                                                                        \
  }                                                                          \
                                                                             \
  void GeneratedMessageReflection::SetRepeated##TYPENAME(                    \
      Message* message, const FieldDescriptor* field,                        \
      int index, PASSTYPE value) const {                                     \
    USAGE_CHECK_ALL(SetRepeated##TYPENAME, REPEATED, CPPTYPE);               \
    if (field->is_extension()) {                                             \
      MutableExtensionSet(message)->SetRepeated##TYPENAME(                   \
        field->number(), index, value);                                      \
    } else {                                                                 \
      SetRepeatedField<TYPE>(message, field, index, value);                  \
    }                                                                        \
  }                                                                          \
                                                                             \
  void GeneratedMessageReflection::Add##TYPENAME(                            \
      Message* message, const FieldDescriptor* field,                        \
      PASSTYPE value) const {                                                \
    USAGE_CHECK_ALL(Add##TYPENAME, REPEATED, CPPTYPE);                       \
    if (field->is_extension()) {                                             \
      MutableExtensionSet(message)->Add##TYPENAME(                           \
        field->number(), field->type(), field->options().packed(), value,    \
        field);                                                              \
    } else {                                                                 \
      AddField<TYPE>(message, field, value);                                 \
    }                                                                        \
  }

DEFINE_PRIMITIVE_ACCESSORS(Int32 , int32 , int32 , INT32 )
DEFINE_PRIMITIVE_ACCESSORS(Int64 , int64 , int64 , INT64 )
DEFINE_PRIMITIVE_ACCESSORS(UInt32, uint32, uint32, UINT32)
DEFINE_PRIMITIVE_ACCESSORS(UInt64, uint64, uint64, UINT64)
DEFINE_PRIMITIVE_ACCESSORS(Float , float , float , FLOAT )
DEFINE_PRIMITIVE_ACCESSORS(Double, double, double, DOUBLE)
DEFINE_PRIMITIVE_ACCESSORS(Bool  , bool  , bool  , BOOL  )
#undef DEFINE_PRIMITIVE_ACCESSORS

// -------------------------------------------------------------------

string GeneratedMessageReflection::GetString(
    const Message& message, const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(GetString, SINGULAR, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetString(field->number(),
                                              field->default_value_string());
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        return GetField<ArenaStringPtr>(message, field).Get();
      }
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return GetEmptyString();  // Make compiler happy.
  }
}

const string& GeneratedMessageReflection::GetStringReference(
    const Message& message,
    const FieldDescriptor* field, string* scratch) const {
  USAGE_CHECK_ALL(GetStringReference, SINGULAR, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetString(field->number(),
                                              field->default_value_string());
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        return GetField<ArenaStringPtr>(message, field).Get();
      }
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return GetEmptyString();  // Make compiler happy.
  }
}


void GeneratedMessageReflection::SetString(
    Message* message, const FieldDescriptor* field,
    const string& value) const {
  USAGE_CHECK_ALL(SetString, SINGULAR, STRING);
  if (field->is_extension()) {
    return MutableExtensionSet(message)->SetString(field->number(),
                                                   field->type(), value, field);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        const string* default_ptr = &DefaultRaw<ArenaStringPtr>(field).Get();
        if (field->containing_oneof() && !HasOneofField(*message, field)) {
          ClearOneof(message, field->containing_oneof());
          MutableField<ArenaStringPtr>(message, field)->UnsafeSetDefault(
              default_ptr);
        }
        MutableField<ArenaStringPtr>(message, field)->Set(default_ptr,
            value, GetArena(message));
        break;
      }
    }
  }
}


string GeneratedMessageReflection::GetRepeatedString(
    const Message& message, const FieldDescriptor* field, int index) const {
  USAGE_CHECK_ALL(GetRepeatedString, REPEATED, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetRepeatedString(field->number(), index);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        return GetRepeatedPtrField<string>(message, field, index);
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return GetEmptyString();  // Make compiler happy.
  }
}

const string& GeneratedMessageReflection::GetRepeatedStringReference(
    const Message& message, const FieldDescriptor* field,
    int index, string* scratch) const {
  USAGE_CHECK_ALL(GetRepeatedStringReference, REPEATED, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetRepeatedString(field->number(), index);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        return GetRepeatedPtrField<string>(message, field, index);
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return GetEmptyString();  // Make compiler happy.
  }
}


void GeneratedMessageReflection::SetRepeatedString(
    Message* message, const FieldDescriptor* field,
    int index, const string& value) const {
  USAGE_CHECK_ALL(SetRepeatedString, REPEATED, STRING);
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetRepeatedString(
      field->number(), index, value);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        *MutableRepeatedField<string>(message, field, index) = value;
        break;
    }
  }
}


void GeneratedMessageReflection::AddString(
    Message* message, const FieldDescriptor* field,
    const string& value) const {
  USAGE_CHECK_ALL(AddString, REPEATED, STRING);
  if (field->is_extension()) {
    MutableExtensionSet(message)->AddString(field->number(),
                                            field->type(), value, field);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        *AddField<string>(message, field) = value;
        break;
    }
  }
}


// -------------------------------------------------------------------

inline bool CreateUnknownEnumValues(const FileDescriptor* file) {
  return file->syntax() == FileDescriptor::SYNTAX_PROTO3;
}

const EnumValueDescriptor* GeneratedMessageReflection::GetEnum(
    const Message& message, const FieldDescriptor* field) const {
  // Usage checked by GetEnumValue.
  int value = GetEnumValue(message, field);
  return field->enum_type()->FindValueByNumberCreatingIfUnknown(value);
}

int GeneratedMessageReflection::GetEnumValue(
    const Message& message, const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(GetEnumValue, SINGULAR, ENUM);

  int32 value;
  if (field->is_extension()) {
    value = GetExtensionSet(message).GetEnum(
      field->number(), field->default_value_enum()->number());
  } else {
    value = GetField<int>(message, field);
  }
  return value;
}

void GeneratedMessageReflection::SetEnum(
    Message* message, const FieldDescriptor* field,
    const EnumValueDescriptor* value) const {
  // Usage checked by SetEnumValue.
  USAGE_CHECK_ENUM_VALUE(SetEnum);
  SetEnumValueInternal(message, field, value->number());
}

void GeneratedMessageReflection::SetEnumValue(
    Message* message, const FieldDescriptor* field,
    int value) const {
  USAGE_CHECK_ALL(SetEnumValue, SINGULAR, ENUM);
  if (!CreateUnknownEnumValues(descriptor_->file())) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == NULL) {
      GOOGLE_LOG(DFATAL) << "SetEnumValue accepts only valid integer values: value "
                  << value << " unexpected for field " << field->full_name();
      // In production builds, DFATAL will not terminate the program, so we have
      // to do something reasonable: just set the default value.
      value = field->default_value_enum()->number();
    }
  }
  SetEnumValueInternal(message, field, value);
}

void GeneratedMessageReflection::SetEnumValueInternal(
    Message* message, const FieldDescriptor* field,
    int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetEnum(field->number(), field->type(),
                                          value, field);
  } else {
    SetField<int>(message, field, value);
  }
}

const EnumValueDescriptor* GeneratedMessageReflection::GetRepeatedEnum(
    const Message& message, const FieldDescriptor* field, int index) const {
  // Usage checked by GetRepeatedEnumValue.
  int value = GetRepeatedEnumValue(message, field, index);
  return field->enum_type()->FindValueByNumberCreatingIfUnknown(value);
}

int GeneratedMessageReflection::GetRepeatedEnumValue(
    const Message& message, const FieldDescriptor* field, int index) const {
  USAGE_CHECK_ALL(GetRepeatedEnumValue, REPEATED, ENUM);

  int value;
  if (field->is_extension()) {
    value = GetExtensionSet(message).GetRepeatedEnum(field->number(), index);
  } else {
    value = GetRepeatedField<int>(message, field, index);
  }
  return value;
}

void GeneratedMessageReflection::SetRepeatedEnum(
    Message* message,
    const FieldDescriptor* field, int index,
    const EnumValueDescriptor* value) const {
  // Usage checked by SetRepeatedEnumValue.
  USAGE_CHECK_ENUM_VALUE(SetRepeatedEnum);
  SetRepeatedEnumValueInternal(message, field, index, value->number());
}

void GeneratedMessageReflection::SetRepeatedEnumValue(
    Message* message,
    const FieldDescriptor* field, int index,
    int value) const {
  USAGE_CHECK_ALL(SetRepeatedEnum, REPEATED, ENUM);
  if (!CreateUnknownEnumValues(descriptor_->file())) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == NULL) {
      GOOGLE_LOG(DFATAL) << "SetRepeatedEnumValue accepts only valid integer values: "
                  << "value " << value << " unexpected for field "
                  << field->full_name();
      // In production builds, DFATAL will not terminate the program, so we have
      // to do something reasonable: just set the default value.
      value = field->default_value_enum()->number();
    }
  }
  SetRepeatedEnumValueInternal(message, field, index, value);
}

void GeneratedMessageReflection::SetRepeatedEnumValueInternal(
    Message* message,
    const FieldDescriptor* field, int index,
    int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetRepeatedEnum(
      field->number(), index, value);
  } else {
    SetRepeatedField<int>(message, field, index, value);
  }
}

void GeneratedMessageReflection::AddEnum(
    Message* message, const FieldDescriptor* field,
    const EnumValueDescriptor* value) const {
  // Usage checked by AddEnumValue.
  USAGE_CHECK_ENUM_VALUE(AddEnum);
  AddEnumValueInternal(message, field, value->number());
}

void GeneratedMessageReflection::AddEnumValue(
    Message* message, const FieldDescriptor* field,
    int value) const {
  USAGE_CHECK_ALL(AddEnum, REPEATED, ENUM);
  if (!CreateUnknownEnumValues(descriptor_->file())) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == NULL) {
      GOOGLE_LOG(DFATAL) << "AddEnumValue accepts only valid integer values: value "
                  << value << " unexpected for field " << field->full_name();
      // In production builds, DFATAL will not terminate the program, so we have
      // to do something reasonable: just set the default value.
      value = field->default_value_enum()->number();
    }
  }
  AddEnumValueInternal(message, field, value);
}

void GeneratedMessageReflection::AddEnumValueInternal(
    Message* message, const FieldDescriptor* field,
    int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->AddEnum(field->number(), field->type(),
                                          field->options().packed(),
                                          value, field);
  } else {
    AddField<int>(message, field, value);
  }
}

// -------------------------------------------------------------------

const Message& GeneratedMessageReflection::GetMessage(
    const Message& message, const FieldDescriptor* field,
    MessageFactory* factory) const {
  USAGE_CHECK_ALL(GetMessage, SINGULAR, MESSAGE);

  if (factory == NULL) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<const Message&>(
        GetExtensionSet(message).GetMessage(
          field->number(), field->message_type(), factory));
  } else {
    const Message* result = GetRaw<const Message*>(message, field);
    if (result == NULL) {
      result = DefaultRaw<const Message*>(field);
    }
    return *result;
  }
}

Message* GeneratedMessageReflection::MutableMessage(
    Message* message, const FieldDescriptor* field,
    MessageFactory* factory) const {
  USAGE_CHECK_ALL(MutableMessage, SINGULAR, MESSAGE);

  if (factory == NULL) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->MutableMessage(field, factory));
  } else {
    Message* result;

    Message** result_holder = MutableRaw<Message*>(message, field);

    if (field->containing_oneof()) {
      if (!HasOneofField(*message, field)) {
        ClearOneof(message, field->containing_oneof());
        result_holder = MutableField<Message*>(message, field);
        const Message* default_message = DefaultRaw<const Message*>(field);
        *result_holder = default_message->New(message->GetArena());
      }
    } else {
      SetBit(message, field);
    }

    if (*result_holder == NULL) {
      const Message* default_message = DefaultRaw<const Message*>(field);
      *result_holder = default_message->New(message->GetArena());
    }
    result = *result_holder;
    return result;
  }
}

void GeneratedMessageReflection::UnsafeArenaSetAllocatedMessage(
    Message* message,
    Message* sub_message,
    const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(SetAllocatedMessage, SINGULAR, MESSAGE);

  if (field->is_extension()) {
    MutableExtensionSet(message)->UnsafeArenaSetAllocatedMessage(
        field->number(), field->type(), field, sub_message);
  } else {
    if (field->containing_oneof()) {
      if (sub_message == NULL) {
        ClearOneof(message, field->containing_oneof());
        return;
      }
        ClearOneof(message, field->containing_oneof());
        *MutableRaw<Message*>(message, field) = sub_message;
      SetOneofCase(message, field);
      return;
    }

    if (sub_message == NULL) {
      ClearBit(message, field);
    } else {
      SetBit(message, field);
    }
    Message** sub_message_holder = MutableRaw<Message*>(message, field);
    if (GetArena(message) == NULL) {
      delete *sub_message_holder;
    }
    *sub_message_holder = sub_message;
  }
}

void GeneratedMessageReflection::SetAllocatedMessage(
    Message* message,
    Message* sub_message,
    const FieldDescriptor* field) const {
  // If message and sub-message are in different memory ownership domains
  // (different arenas, or one is on heap and one is not), then we may need to
  // do a copy.
  if (sub_message != NULL &&
      sub_message->GetArena() != message->GetArena()) {
    if (sub_message->GetArena() == NULL && message->GetArena() != NULL) {
      // Case 1: parent is on an arena and child is heap-allocated. We can add
      // the child to the arena's Own() list to free on arena destruction, then
      // set our pointer.
      message->GetArena()->Own(sub_message);
      UnsafeArenaSetAllocatedMessage(message, sub_message, field);
    } else {
      // Case 2: all other cases. We need to make a copy. MutableMessage() will
      // either get the existing message object, or instantiate a new one as
      // appropriate w.r.t. our arena.
      Message* sub_message_copy = MutableMessage(message, field);
      sub_message_copy->CopyFrom(*sub_message);
    }
  } else {
    // Same memory ownership domains.
    UnsafeArenaSetAllocatedMessage(message, sub_message, field);
  }
}

Message* GeneratedMessageReflection::UnsafeArenaReleaseMessage(
    Message* message,
    const FieldDescriptor* field,
    MessageFactory* factory) const {
  USAGE_CHECK_ALL(ReleaseMessage, SINGULAR, MESSAGE);

  if (factory == NULL) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->UnsafeArenaReleaseMessage(field,
                                                                factory));
  } else {
    if (!(field->is_repeated() || field->containing_oneof())) {
      ClearBit(message, field);
    }
    if (field->containing_oneof()) {
      if (HasOneofField(*message, field)) {
        *MutableOneofCase(message, field->containing_oneof()) = 0;
      } else {
        return NULL;
      }
    }
    Message** result = MutableRaw<Message*>(message, field);
    Message* ret = *result;
    *result = NULL;
    return ret;
  }
}

Message* GeneratedMessageReflection::ReleaseMessage(
    Message* message,
    const FieldDescriptor* field,
    MessageFactory* factory) const {
  Message* released = UnsafeArenaReleaseMessage(message, field, factory);
  if (GetArena(message) != NULL && released != NULL) {
    Message* copy_from_arena = released->New();
    copy_from_arena->CopyFrom(*released);
    released = copy_from_arena;
  }
  return released;
}

const Message& GeneratedMessageReflection::GetRepeatedMessage(
    const Message& message, const FieldDescriptor* field, int index) const {
  USAGE_CHECK_ALL(GetRepeatedMessage, REPEATED, MESSAGE);

  if (field->is_extension()) {
    return static_cast<const Message&>(
        GetExtensionSet(message).GetRepeatedMessage(field->number(), index));
  } else {
    if (IsMapFieldInApi(field)) {
      return GetRaw<MapFieldBase>(message, field)
          .GetRepeatedField()
          .Get<GenericTypeHandler<Message> >(index);
    } else {
      return GetRaw<RepeatedPtrFieldBase>(message, field)
          .Get<GenericTypeHandler<Message> >(index);
    }
  }
}

Message* GeneratedMessageReflection::MutableRepeatedMessage(
    Message* message, const FieldDescriptor* field, int index) const {
  USAGE_CHECK_ALL(MutableRepeatedMessage, REPEATED, MESSAGE);

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->MutableRepeatedMessage(
          field->number(), index));
  } else {
    if (IsMapFieldInApi(field)) {
      return MutableRaw<MapFieldBase>(message, field)
          ->MutableRepeatedField()
          ->Mutable<GenericTypeHandler<Message> >(index);
    } else {
      return MutableRaw<RepeatedPtrFieldBase>(message, field)
        ->Mutable<GenericTypeHandler<Message> >(index);
    }
  }
}

Message* GeneratedMessageReflection::AddMessage(
    Message* message, const FieldDescriptor* field,
    MessageFactory* factory) const {
  USAGE_CHECK_ALL(AddMessage, REPEATED, MESSAGE);

  if (factory == NULL) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->AddMessage(field, factory));
  } else {
    Message* result = NULL;

    // We can't use AddField<Message>() because RepeatedPtrFieldBase doesn't
    // know how to allocate one.
    RepeatedPtrFieldBase* repeated = NULL;
    if (IsMapFieldInApi(field)) {
      repeated =
          MutableRaw<MapFieldBase>(message, field)->MutableRepeatedField();
    } else {
      repeated = MutableRaw<RepeatedPtrFieldBase>(message, field);
    }
    result = repeated->AddFromCleared<GenericTypeHandler<Message> >();
    if (result == NULL) {
      // We must allocate a new object.
      const Message* prototype;
      if (repeated->size() == 0) {
        prototype = factory->GetPrototype(field->message_type());
      } else {
        prototype = &repeated->Get<GenericTypeHandler<Message> >(0);
      }
      result = prototype->New(message->GetArena());
      // We can guarantee here that repeated and result are either both heap
      // allocated or arena owned. So it is safe to call the unsafe version
      // of AddAllocated.
      repeated->UnsafeArenaAddAllocated<GenericTypeHandler<Message> >(result);
    }

    return result;
  }
}

void GeneratedMessageReflection::AddAllocatedMessage(
    Message* message, const FieldDescriptor* field,
    Message* new_entry) const {
  USAGE_CHECK_ALL(AddAllocatedMessage, REPEATED, MESSAGE);

  if (field->is_extension()) {
    MutableExtensionSet(message)->AddAllocatedMessage(field, new_entry);
  } else {
    RepeatedPtrFieldBase* repeated = NULL;
    if (IsMapFieldInApi(field)) {
      repeated =
          MutableRaw<MapFieldBase>(message, field)->MutableRepeatedField();
    } else {
      repeated = MutableRaw<RepeatedPtrFieldBase>(message, field);
    }
    repeated->AddAllocated<GenericTypeHandler<Message> >(new_entry);
  }
}

void* GeneratedMessageReflection::MutableRawRepeatedField(
    Message* message, const FieldDescriptor* field,
    FieldDescriptor::CppType cpptype,
    int ctype, const Descriptor* desc) const {
  USAGE_CHECK_REPEATED("MutableRawRepeatedField");
  if (field->cpp_type() != cpptype)
    ReportReflectionUsageTypeError(descriptor_,
        field, "MutableRawRepeatedField", cpptype);
  if (ctype >= 0)
    GOOGLE_CHECK_EQ(field->options().ctype(), ctype) << "subtype mismatch";
  if (desc != NULL)
    GOOGLE_CHECK_EQ(field->message_type(), desc) << "wrong submessage type";
  if (field->is_extension()) {
    return MutableExtensionSet(message)->MutableRawRepeatedField(
        field->number(), field->type(), field->is_packed(), field);
  } else {
    // Trigger transform for MapField
    if (IsMapFieldInApi(field)) {
      return MutableRawNonOneof<MapFieldBase>(message, field)
          ->MutableRepeatedField();
    }
    return MutableRawNonOneof<void>(message, field);
  }
}

const void* GeneratedMessageReflection::GetRawRepeatedField(
    const Message& message, const FieldDescriptor* field,
    FieldDescriptor::CppType cpptype,
    int ctype, const Descriptor* desc) const {
  USAGE_CHECK_REPEATED("GetRawRepeatedField");
  if (field->cpp_type() != cpptype)
    ReportReflectionUsageTypeError(descriptor_,
        field, "GetRawRepeatedField", cpptype);
  if (ctype >= 0)
    GOOGLE_CHECK_EQ(field->options().ctype(), ctype) << "subtype mismatch";
  if (desc != NULL)
    GOOGLE_CHECK_EQ(field->message_type(), desc) << "wrong submessage type";
  if (field->is_extension()) {
    // Should use extension_set::GetRawRepeatedField. However, the required
    // parameter "default repeated value" is not very easy to get here.
    // Map is not supported in extensions, it is acceptable to use
    // extension_set::MutableRawRepeatedField which does not change the message.
    return MutableExtensionSet(const_cast<Message*>(&message))
        ->MutableRawRepeatedField(
        field->number(), field->type(), field->is_packed(), field);
  } else {
    // Trigger transform for MapField
    if (IsMapFieldInApi(field)) {
      return &(GetRawNonOneof<MapFieldBase>(message, field).GetRepeatedField());
    }
    return &GetRawNonOneof<char>(message, field);
  }
}

const FieldDescriptor* GeneratedMessageReflection::GetOneofFieldDescriptor(
    const Message& message,
    const OneofDescriptor* oneof_descriptor) const {
  uint32 field_number = GetOneofCase(message, oneof_descriptor);
  if (field_number == 0) {
    return NULL;
  }
  return descriptor_->FindFieldByNumber(field_number);
}

bool GeneratedMessageReflection::ContainsMapKey(
    const Message& message,
    const FieldDescriptor* field,
    const MapKey& key) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "LookupMapValue",
              "Field is not a map field.");
  return GetRaw<MapFieldBase>(message, field).ContainsMapKey(key);
}

bool GeneratedMessageReflection::InsertOrLookupMapValue(
    Message* message,
    const FieldDescriptor* field,
    const MapKey& key,
    MapValueRef* val) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "InsertOrLookupMapValue",
              "Field is not a map field.");
  val->SetType(field->message_type()->FindFieldByName("value")->cpp_type());
  return MutableRaw<MapFieldBase>(message, field)->InsertOrLookupMapValue(
      key, val);
}

bool GeneratedMessageReflection::DeleteMapValue(
    Message* message,
    const FieldDescriptor* field,
    const MapKey& key) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "DeleteMapValue",
              "Field is not a map field.");
  return MutableRaw<MapFieldBase>(message, field)->DeleteMapValue(key);
}

MapIterator GeneratedMessageReflection::MapBegin(
    Message* message,
    const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "MapBegin",
              "Field is not a map field.");
  MapIterator iter(message, field);
  GetRaw<MapFieldBase>(*message, field).MapBegin(&iter);
  return iter;
}

MapIterator GeneratedMessageReflection::MapEnd(
    Message* message,
    const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "MapEnd",
              "Field is not a map field.");
  MapIterator iter(message, field);
  GetRaw<MapFieldBase>(*message, field).MapEnd(&iter);
  return iter;
}

int GeneratedMessageReflection::MapSize(
    const Message& message,
    const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "MapSize",
              "Field is not a map field.");
  return GetRaw<MapFieldBase>(message, field).size();
}

// -----------------------------------------------------------------------------

const FieldDescriptor* GeneratedMessageReflection::FindKnownExtensionByName(
    const string& name) const {
  if (!schema_.HasExtensionSet()) return NULL;

  const FieldDescriptor* result = descriptor_pool_->FindExtensionByName(name);
  if (result != NULL && result->containing_type() == descriptor_) {
    return result;
  }

  if (descriptor_->options().message_set_wire_format()) {
    // MessageSet extensions may be identified by type name.
    const Descriptor* type = descriptor_pool_->FindMessageTypeByName(name);
    if (type != NULL) {
      // Look for a matching extension in the foreign type's scope.
      const int type_extension_count = type->extension_count();
      for (int i = 0; i < type_extension_count; i++) {
        const FieldDescriptor* extension = type->extension(i);
        if (extension->containing_type() == descriptor_ &&
            extension->type() == FieldDescriptor::TYPE_MESSAGE &&
            extension->is_optional() &&
            extension->message_type() == type) {
          // Found it.
          return extension;
        }
      }
    }
  }

  return NULL;
}

const FieldDescriptor* GeneratedMessageReflection::FindKnownExtensionByNumber(
    int number) const {
  if (!schema_.HasExtensionSet()) return NULL;
  return descriptor_pool_->FindExtensionByNumber(descriptor_, number);
}

bool GeneratedMessageReflection::SupportsUnknownEnumValues() const {
  return CreateUnknownEnumValues(descriptor_->file());
}

// ===================================================================
// Some private helpers.

// These simple template accessors obtain pointers (or references) to
// the given field.

template <class Type>
const Type& GeneratedMessageReflection::GetRawNonOneof(
    const Message& message, const FieldDescriptor* field) const {
  return GetConstRefAtOffset<Type>(message,
                                   schema_.GetFieldOffsetNonOneof(field));
}

template <class Type>
Type* GeneratedMessageReflection::MutableRawNonOneof(
    Message* message, const FieldDescriptor* field) const {
  return GetPointerAtOffset<Type>(message,
                                  schema_.GetFieldOffsetNonOneof(field));
}

template <typename Type>
const Type& GeneratedMessageReflection::GetRaw(
    const Message& message, const FieldDescriptor* field) const {
  if (field->containing_oneof() && !HasOneofField(message, field)) {
    return DefaultRaw<Type>(field);
  }
  return GetConstRefAtOffset<Type>(message, schema_.GetFieldOffset(field));
}

template <typename Type>
Type* GeneratedMessageReflection::MutableRaw(Message* message,
                                   const FieldDescriptor* field) const {
  return GetPointerAtOffset<Type>(message, schema_.GetFieldOffset(field));
}


inline const uint32* GeneratedMessageReflection::GetHasBits(
    const Message& message) const {
  GOOGLE_DCHECK(schema_.HasHasbits());
  return &GetConstRefAtOffset<uint32>(message, schema_.HasBitsOffset());
}

inline uint32* GeneratedMessageReflection::MutableHasBits(
    Message* message) const {
  GOOGLE_DCHECK(schema_.HasHasbits());
  return GetPointerAtOffset<uint32>(message, schema_.HasBitsOffset());
}

inline uint32 GeneratedMessageReflection::GetOneofCase(
    const Message& message, const OneofDescriptor* oneof_descriptor) const {
  return GetConstRefAtOffset<uint32>(
      message, schema_.GetOneofCaseOffset(oneof_descriptor));
}

inline uint32* GeneratedMessageReflection::MutableOneofCase(
    Message* message, const OneofDescriptor* oneof_descriptor) const {
  return GetPointerAtOffset<uint32>(
      message, schema_.GetOneofCaseOffset(oneof_descriptor));
}

inline const ExtensionSet& GeneratedMessageReflection::GetExtensionSet(
    const Message& message) const {
  return GetConstRefAtOffset<ExtensionSet>(message,
                                           schema_.GetExtensionSetOffset());
}

inline ExtensionSet* GeneratedMessageReflection::MutableExtensionSet(
    Message* message) const {
  return GetPointerAtOffset<ExtensionSet>(message,
                                          schema_.GetExtensionSetOffset());
}

inline Arena* GeneratedMessageReflection::GetArena(Message* message) const {
  return GetInternalMetadataWithArena(*message).arena();
}

inline const InternalMetadataWithArena&
GeneratedMessageReflection::GetInternalMetadataWithArena(
    const Message& message) const {
  return GetConstRefAtOffset<InternalMetadataWithArena>(
      message, schema_.GetMetadataOffset());
}

inline InternalMetadataWithArena*
GeneratedMessageReflection::MutableInternalMetadataWithArena(
    Message* message) const {
  return GetPointerAtOffset<InternalMetadataWithArena>(
      message, schema_.GetMetadataOffset());
}

template <typename Type>
inline const Type& GeneratedMessageReflection::DefaultRaw(
    const FieldDescriptor* field) const {
  return *reinterpret_cast<const Type*>(schema_.GetFieldDefault(field));
}

// Simple accessors for manipulating has_bits_.
inline bool GeneratedMessageReflection::HasBit(
    const Message& message, const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  if (schema_.HasHasbits()) {
    return IsIndexInHasBitSet(GetHasBits(message), schema_.HasBitIndex(field));
  }

  // proto3: no has-bits. All fields present except messages, which are
  // present only if their message-field pointer is non-NULL.
  if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
    return !schema_.IsDefaultInstance(message) &&
        GetRaw<const Message*>(message, field) != NULL;
  } else {
    // Non-message field (and non-oneof, since that was handled in HasField()
    // before calling us), and singular (again, checked in HasField). So, this
    // field must be a scalar.

    // Scalar primitive (numeric or string/bytes) fields are present if
    // their value is non-zero (numeric) or non-empty (string/bytes).  N.B.:
    // we must use this definition here, rather than the "scalar fields
    // always present" in the proto3 docs, because MergeFrom() semantics
    // require presence as "present on wire", and reflection-based merge
    // (which uses HasField()) needs to be consistent with this.
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default: {
            return GetField<ArenaStringPtr>(message, field).Get().size() > 0;
          }
        }
        return false;
      case FieldDescriptor::CPPTYPE_BOOL:
        return GetRaw<bool>(message, field) != false;
      case FieldDescriptor::CPPTYPE_INT32:
        return GetRaw<int32>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_INT64:
        return GetRaw<int64>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_UINT32:
        return GetRaw<uint32>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_UINT64:
        return GetRaw<uint64>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_FLOAT:
        return GetRaw<float>(message, field) != 0.0;
      case FieldDescriptor::CPPTYPE_DOUBLE:
        return GetRaw<double>(message, field) != 0.0;
      case FieldDescriptor::CPPTYPE_ENUM:
        return GetRaw<int>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        // handled above; avoid warning
        break;
    }
    GOOGLE_LOG(FATAL) << "Reached impossible case in HasBit().";
    return false;
  }
}

inline void GeneratedMessageReflection::SetBit(
    Message* message, const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  if (!schema_.HasHasbits()) {
    return;
  }
  const uint32 index = schema_.HasBitIndex(field);
  MutableHasBits(message)[index / 32] |=
      (static_cast<uint32>(1) << (index % 32));
}

inline void GeneratedMessageReflection::ClearBit(
    Message* message, const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  if (!schema_.HasHasbits()) {
    return;
  }
  const uint32 index = schema_.HasBitIndex(field);
  MutableHasBits(message)[index / 32] &=
      ~(static_cast<uint32>(1) << (index % 32));
}

inline void GeneratedMessageReflection::SwapBit(
    Message* message1, Message* message2, const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  if (!schema_.HasHasbits()) {
    return;
  }
  bool temp_has_bit = HasBit(*message1, field);
  if (HasBit(*message2, field)) {
    SetBit(message1, field);
  } else {
    ClearBit(message1, field);
  }
  if (temp_has_bit) {
    SetBit(message2, field);
  } else {
    ClearBit(message2, field);
  }
}

inline bool GeneratedMessageReflection::HasOneof(
    const Message& message, const OneofDescriptor* oneof_descriptor) const {
  return (GetOneofCase(message, oneof_descriptor) > 0);
}

inline bool GeneratedMessageReflection::HasOneofField(
    const Message& message, const FieldDescriptor* field) const {
  return (GetOneofCase(message, field->containing_oneof()) == field->number());
}

inline void GeneratedMessageReflection::SetOneofCase(
    Message* message, const FieldDescriptor* field) const {
  *MutableOneofCase(message, field->containing_oneof()) = field->number();
}

inline void GeneratedMessageReflection::ClearOneofField(
    Message* message, const FieldDescriptor* field) const {
  if (HasOneofField(*message, field)) {
    ClearOneof(message, field->containing_oneof());
  }
}

inline void GeneratedMessageReflection::ClearOneof(
    Message* message, const OneofDescriptor* oneof_descriptor) const {
  // TODO(jieluo): Consider to cache the unused object instead of deleting
  // it. It will be much faster if an application switches a lot from
  // a few oneof fields.  Time/space tradeoff
  uint32 oneof_case = GetOneofCase(*message, oneof_descriptor);
  if (oneof_case > 0) {
    const FieldDescriptor* field = descriptor_->FindFieldByNumber(oneof_case);
    if (GetArena(message) == NULL) {
      switch (field->cpp_type()) {
        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              const string* default_ptr =
                  &DefaultRaw<ArenaStringPtr>(field).Get();
              MutableField<ArenaStringPtr>(message, field)->
                  Destroy(default_ptr, GetArena(message));
              break;
            }
          }
          break;
        }

        case FieldDescriptor::CPPTYPE_MESSAGE:
          delete *MutableRaw<Message*>(message, field);
          break;
        default:
          break;
      }
    }

    *MutableOneofCase(message, oneof_descriptor) = 0;
  }
}

// Template implementations of basic accessors.  Inline because each
// template instance is only called from one location.  These are
// used for all types except messages.
template <typename Type>
inline const Type& GeneratedMessageReflection::GetField(
    const Message& message, const FieldDescriptor* field) const {
  return GetRaw<Type>(message, field);
}

template <typename Type>
inline void GeneratedMessageReflection::SetField(
    Message* message, const FieldDescriptor* field, const Type& value) const {
  if (field->containing_oneof() && !HasOneofField(*message, field)) {
    ClearOneof(message, field->containing_oneof());
  }
  *MutableRaw<Type>(message, field) = value;
  field->containing_oneof() ?
      SetOneofCase(message, field) : SetBit(message, field);
}

template <typename Type>
inline Type* GeneratedMessageReflection::MutableField(
    Message* message, const FieldDescriptor* field) const {
  field->containing_oneof() ?
      SetOneofCase(message, field) : SetBit(message, field);
  return MutableRaw<Type>(message, field);
}

template <typename Type>
inline const Type& GeneratedMessageReflection::GetRepeatedField(
    const Message& message, const FieldDescriptor* field, int index) const {
  return GetRaw<RepeatedField<Type> >(message, field).Get(index);
}

template <typename Type>
inline const Type& GeneratedMessageReflection::GetRepeatedPtrField(
    const Message& message, const FieldDescriptor* field, int index) const {
  return GetRaw<RepeatedPtrField<Type> >(message, field).Get(index);
}

template <typename Type>
inline void GeneratedMessageReflection::SetRepeatedField(
    Message* message, const FieldDescriptor* field,
    int index, Type value) const {
  MutableRaw<RepeatedField<Type> >(message, field)->Set(index, value);
}

template <typename Type>
inline Type* GeneratedMessageReflection::MutableRepeatedField(
    Message* message, const FieldDescriptor* field, int index) const {
  RepeatedPtrField<Type>* repeated =
    MutableRaw<RepeatedPtrField<Type> >(message, field);
  return repeated->Mutable(index);
}

template <typename Type>
inline void GeneratedMessageReflection::AddField(
    Message* message, const FieldDescriptor* field, const Type& value) const {
  MutableRaw<RepeatedField<Type> >(message, field)->Add(value);
}

template <typename Type>
inline Type* GeneratedMessageReflection::AddField(
    Message* message, const FieldDescriptor* field) const {
  RepeatedPtrField<Type>* repeated =
    MutableRaw<RepeatedPtrField<Type> >(message, field);
  return repeated->Add();
}

MessageFactory* GeneratedMessageReflection::GetMessageFactory() const {
  return message_factory_;
}

void* GeneratedMessageReflection::RepeatedFieldData(
    Message* message, const FieldDescriptor* field,
    FieldDescriptor::CppType cpp_type,
    const Descriptor* message_type) const {
  GOOGLE_CHECK(field->is_repeated());
  GOOGLE_CHECK(field->cpp_type() == cpp_type ||
        (field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM &&
         cpp_type == FieldDescriptor::CPPTYPE_INT32))
      << "The type parameter T in RepeatedFieldRef<T> API doesn't match "
      << "the actual field type (for enums T should be the generated enum "
      << "type or int32).";
  if (message_type != NULL) {
    GOOGLE_CHECK_EQ(message_type, field->message_type());
  }
  if (field->is_extension()) {
    return MutableExtensionSet(message)->MutableRawRepeatedField(
        field->number(), field->type(), field->is_packed(), field);
  } else {
    return MutableRawNonOneof<char>(message, field);
  }
}

MapFieldBase* GeneratedMessageReflection::MapData(
    Message* message, const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field),
              "GetMapData",
              "Field is not a map field.");
  return MutableRaw<MapFieldBase>(message, field);
}

namespace {

// Helper function to transform migration schema into reflection schema.
ReflectionSchema MigrationToReflectionSchema(
    const Message* const* default_instance, const uint32* offsets,
    MigrationSchema migration_schema) {
  ReflectionSchema result;
  result.default_instance_ = *default_instance;
  // First 6 offsets are offsets to the special fields. The following offsets
  // are the proto fields.
  result.offsets_ = offsets + migration_schema.offsets_index + 5;
  result.has_bit_indices_ = offsets + migration_schema.has_bit_indices_index;
  result.has_bits_offset_ = offsets[migration_schema.offsets_index + 0];
  result.metadata_offset_ = offsets[migration_schema.offsets_index + 1];
  result.extensions_offset_ = offsets[migration_schema.offsets_index + 2];
  result.oneof_case_offset_ = offsets[migration_schema.offsets_index + 3];
  result.object_size_ = migration_schema.object_size;
  result.weak_field_map_offset_ = offsets[migration_schema.offsets_index + 4];
  return result;
}

template<typename Schema>
class AssignDescriptorsHelper {
 public:
  AssignDescriptorsHelper(MessageFactory* factory,
                          Metadata* file_level_metadata,
                          const EnumDescriptor** file_level_enum_descriptors,
                          const Schema* schemas,
                          const Message* const* default_instance_data,
                          const uint32* offsets)
      : factory_(factory),
        file_level_metadata_(file_level_metadata),
        file_level_enum_descriptors_(file_level_enum_descriptors),
        schemas_(schemas),
        default_instance_data_(default_instance_data),
        offsets_(offsets) {}

  void AssignMessageDescriptor(const Descriptor* descriptor) {
    for (int i = 0; i < descriptor->nested_type_count(); i++) {
      AssignMessageDescriptor(descriptor->nested_type(i));
    }

    file_level_metadata_->descriptor = descriptor;

    file_level_metadata_->reflection = new GeneratedMessageReflection(
        descriptor,
        MigrationToReflectionSchema(default_instance_data_, offsets_,
                                    *schemas_),
        ::google::protobuf::DescriptorPool::generated_pool(), factory_);
    for (int i = 0; i < descriptor->enum_type_count(); i++) {
      AssignEnumDescriptor(descriptor->enum_type(i));
    }
    schemas_++;
    default_instance_data_++;
    file_level_metadata_++;
  }

  void AssignEnumDescriptor(const EnumDescriptor* descriptor) {
    *file_level_enum_descriptors_ = descriptor;
    file_level_enum_descriptors_++;
  }

  const Metadata* GetCurrentMetadataPtr() const { return file_level_metadata_; }

 private:
  MessageFactory* factory_;
  Metadata* file_level_metadata_;
  const EnumDescriptor** file_level_enum_descriptors_;
  const Schema* schemas_;
  const Message* const * default_instance_data_;
  const uint32* offsets_;
};

// We have the routines that assign descriptors and build reflection
// automatically delete the allocated reflection. MetadataOwner owns
// all the allocated reflection instances.
struct MetadataOwner {
  void AddArray(const Metadata* begin, const Metadata* end) {
    MutexLock lock(&mu_);
    metadata_arrays_.push_back(std::make_pair(begin, end));
  }

  static MetadataOwner* Instance() {
    static MetadataOwner* res = new MetadataOwner;
    return res;
  }

 private:
  // Use the constructor to register the shutdown code. Because c++ makes sure
  // this called only once.
  MetadataOwner() { OnShutdown(&DeleteMetadata); }
  ~MetadataOwner() {
    for (int i = 0; i < metadata_arrays_.size(); i++) {
      for (const Metadata* m = metadata_arrays_[i].first;
           m < metadata_arrays_[i].second; m++) {
        delete m->reflection;
      }
    }
  }

  static void DeleteMetadata() {
    delete Instance();
  }

  Mutex mu_;
  std::vector<std::pair<const Metadata*, const Metadata*> > metadata_arrays_;
};

}  // namespace

void AssignDescriptors(
    const string& filename, const MigrationSchema* schemas,
    const Message* const* default_instances_, const uint32* offsets,
    MessageFactory* factory,
    // update the following descriptor arrays.
    Metadata* file_level_metadata,
    const EnumDescriptor** file_level_enum_descriptors,
    const ServiceDescriptor** file_level_service_descriptors) {
  const ::google::protobuf::FileDescriptor* file =
      ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(filename);
  GOOGLE_CHECK(file != NULL);

  if (!factory) factory = MessageFactory::generated_factory();

  AssignDescriptorsHelper<MigrationSchema> helper(factory, file_level_metadata,
                                 file_level_enum_descriptors, schemas,
                                 default_instances_, offsets);

  for (int i = 0; i < file->message_type_count(); i++) {
    helper.AssignMessageDescriptor(file->message_type(i));
  }

  for (int i = 0; i < file->enum_type_count(); i++) {
    helper.AssignEnumDescriptor(file->enum_type(i));
  }
  if (file->options().cc_generic_services()) {
    for (int i = 0; i < file->service_count(); i++) {
      file_level_service_descriptors[i] = file->service(i);
    }
  }
  MetadataOwner::Instance()->AddArray(
      file_level_metadata, helper.GetCurrentMetadataPtr());
}

void RegisterAllTypesInternal(const Metadata* file_level_metadata, int size) {
  for (int i = 0; i < size; i++) {
    const GeneratedMessageReflection* reflection =
        static_cast<const GeneratedMessageReflection*>(
           file_level_metadata[i].reflection);
    if (reflection) {
      // It's not a map type
      ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
          file_level_metadata[i].descriptor,
          reflection->schema_.default_instance_);
    }
  }
}

void RegisterAllTypes(const Metadata* file_level_metadata, int size) {
  RegisterAllTypesInternal(file_level_metadata, size);
}

void UnknownFieldSetSerializer(const uint8* base, uint32 offset, uint32 tag,
                               uint32 has_offset,
                               ::google::protobuf::io::CodedOutputStream* output) {
  const void* ptr = base + offset;
  const InternalMetadataWithArena* metadata =
      static_cast<const InternalMetadataWithArena*>(ptr);
  if (metadata->have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        metadata->unknown_fields(), output);
  }
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google
