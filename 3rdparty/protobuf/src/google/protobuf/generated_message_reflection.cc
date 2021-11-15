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

#include <google/protobuf/generated_message_reflection.h>

#include <algorithm>
#include <set>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/map_field.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/stubs/mutex.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/stubs/strutil.h>


// clang-format off
#include <google/protobuf/port_def.inc>
// clang-format on

#define GOOGLE_PROTOBUF_HAS_ONEOF

using google::protobuf::internal::ArenaStringPtr;
using google::protobuf::internal::DescriptorTable;
using google::protobuf::internal::ExtensionSet;
using google::protobuf::internal::GenericTypeHandler;
using google::protobuf::internal::GetEmptyString;
using google::protobuf::internal::InlinedStringField;
using google::protobuf::internal::InternalMetadata;
using google::protobuf::internal::LazyField;
using google::protobuf::internal::MapFieldBase;
using google::protobuf::internal::MigrationSchema;
using google::protobuf::internal::OnShutdownDelete;
using google::protobuf::internal::ReflectionSchema;
using google::protobuf::internal::RepeatedPtrFieldBase;
using google::protobuf::internal::StringSpaceUsedExcludingSelfLong;
using google::protobuf::internal::WrappedMutex;

namespace google {
namespace protobuf {

namespace {
bool IsMapFieldInApi(const FieldDescriptor* field) { return field->is_map(); }

#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
Message* MaybeForceCopy(Arena* arena, Message* msg) {
  if (arena != nullptr || msg == nullptr) return msg;

  Message* copy = msg->New();
  copy->MergeFrom(*msg);
  delete msg;
  return copy;
}
#endif  // PROTOBUF_FORCE_COPY_IN_RELEASE
}  // anonymous namespace

namespace internal {

bool ParseNamedEnum(const EnumDescriptor* descriptor, ConstStringParam name,
                    int* value) {
  const EnumValueDescriptor* d = descriptor->FindValueByName(name);
  if (d == nullptr) return false;
  *value = d->number();
  return true;
}

const std::string& NameOfEnum(const EnumDescriptor* descriptor, int value) {
  const EnumValueDescriptor* d = descriptor->FindValueByNumber(value);
  return (d == nullptr ? GetEmptyString() : d->name());
}

}  // namespace internal

// ===================================================================
// Helpers for reporting usage errors (e.g. trying to use GetInt32() on
// a string field).

namespace {

using internal::GetConstPointerAtOffset;
using internal::GetConstRefAtOffset;
using internal::GetPointerAtOffset;

void ReportReflectionUsageError(const Descriptor* descriptor,
                                const FieldDescriptor* field,
                                const char* method, const char* description) {
  GOOGLE_LOG(FATAL) << "Protocol Buffer reflection usage error:\n"
                "  Method      : google::protobuf::Reflection::"
             << method
             << "\n"
                "  Message type: "
             << descriptor->full_name()
             << "\n"
                "  Field       : "
             << field->full_name()
             << "\n"
                "  Problem     : "
             << description;
}

const char* cpptype_names_[FieldDescriptor::MAX_CPPTYPE + 1] = {
    "INVALID_CPPTYPE", "CPPTYPE_INT32",  "CPPTYPE_INT64",  "CPPTYPE_UINT32",
    "CPPTYPE_UINT64",  "CPPTYPE_DOUBLE", "CPPTYPE_FLOAT",  "CPPTYPE_BOOL",
    "CPPTYPE_ENUM",    "CPPTYPE_STRING", "CPPTYPE_MESSAGE"};

static void ReportReflectionUsageTypeError(
    const Descriptor* descriptor, const FieldDescriptor* field,
    const char* method, FieldDescriptor::CppType expected_type) {
  GOOGLE_LOG(FATAL)
      << "Protocol Buffer reflection usage error:\n"
         "  Method      : google::protobuf::Reflection::"
      << method
      << "\n"
         "  Message type: "
      << descriptor->full_name()
      << "\n"
         "  Field       : "
      << field->full_name()
      << "\n"
         "  Problem     : Field is not the right type for this message:\n"
         "    Expected  : "
      << cpptype_names_[expected_type]
      << "\n"
         "    Field type: "
      << cpptype_names_[field->cpp_type()];
}

static void ReportReflectionUsageEnumTypeError(
    const Descriptor* descriptor, const FieldDescriptor* field,
    const char* method, const EnumValueDescriptor* value) {
  GOOGLE_LOG(FATAL) << "Protocol Buffer reflection usage error:\n"
                "  Method      : google::protobuf::Reflection::"
             << method
             << "\n"
                "  Message type: "
             << descriptor->full_name()
             << "\n"
                "  Field       : "
             << field->full_name()
             << "\n"
                "  Problem     : Enum value did not match field type:\n"
                "    Expected  : "
             << field->enum_type()->full_name()
             << "\n"
                "    Actual    : "
             << value->full_name();
}

inline void CheckInvalidAccess(const internal::ReflectionSchema& schema,
                               const FieldDescriptor* field) {
  GOOGLE_CHECK(!schema.IsFieldStripped(field))
      << "invalid access to a stripped field " << field->full_name();
}

#define USAGE_CHECK(CONDITION, METHOD, ERROR_DESCRIPTION) \
  if (!(CONDITION))                                       \
  ReportReflectionUsageError(descriptor_, field, #METHOD, ERROR_DESCRIPTION)
#define USAGE_CHECK_EQ(A, B, METHOD, ERROR_DESCRIPTION) \
  USAGE_CHECK((A) == (B), METHOD, ERROR_DESCRIPTION)
#define USAGE_CHECK_NE(A, B, METHOD, ERROR_DESCRIPTION) \
  USAGE_CHECK((A) != (B), METHOD, ERROR_DESCRIPTION)

#define USAGE_CHECK_TYPE(METHOD, CPPTYPE)                      \
  if (field->cpp_type() != FieldDescriptor::CPPTYPE_##CPPTYPE) \
  ReportReflectionUsageTypeError(descriptor_, field, #METHOD,  \
                                 FieldDescriptor::CPPTYPE_##CPPTYPE)

#define USAGE_CHECK_ENUM_VALUE(METHOD)     \
  if (value->type() != field->enum_type()) \
  ReportReflectionUsageEnumTypeError(descriptor_, field, #METHOD, value)

#define USAGE_CHECK_MESSAGE_TYPE(METHOD)                        \
  USAGE_CHECK_EQ(field->containing_type(), descriptor_, METHOD, \
                 "Field does not match message type.");
#define USAGE_CHECK_SINGULAR(METHOD)                                      \
  USAGE_CHECK_NE(field->label(), FieldDescriptor::LABEL_REPEATED, METHOD, \
                 "Field is repeated; the method requires a singular field.")
#define USAGE_CHECK_REPEATED(METHOD)                                      \
  USAGE_CHECK_EQ(field->label(), FieldDescriptor::LABEL_REPEATED, METHOD, \
                 "Field is singular; the method requires a repeated field.")

#define USAGE_CHECK_ALL(METHOD, LABEL, CPPTYPE) \
  USAGE_CHECK_MESSAGE_TYPE(METHOD);             \
  USAGE_CHECK_##LABEL(METHOD);                  \
  USAGE_CHECK_TYPE(METHOD, CPPTYPE)

}  // namespace

// ===================================================================

Reflection::Reflection(const Descriptor* descriptor,
                       const internal::ReflectionSchema& schema,
                       const DescriptorPool* pool, MessageFactory* factory)
    : descriptor_(descriptor),
      schema_(schema),
      descriptor_pool_(
          (pool == nullptr) ? DescriptorPool::internal_generated_pool() : pool),
      message_factory_(factory),
      last_non_weak_field_index_(-1) {
  last_non_weak_field_index_ = descriptor_->field_count() - 1;
}

const UnknownFieldSet& Reflection::GetUnknownFields(
    const Message& message) const {
  return GetInternalMetadata(message).unknown_fields<UnknownFieldSet>(
      UnknownFieldSet::default_instance);
}

UnknownFieldSet* Reflection::MutableUnknownFields(Message* message) const {
  return MutableInternalMetadata(message)
      ->mutable_unknown_fields<UnknownFieldSet>();
}

bool Reflection::IsLazyExtension(const Message& message,
                                 const FieldDescriptor* field) const {
  return field->is_extension() &&
         GetExtensionSet(message).HasLazy(field->number());
}

bool Reflection::IsLazilyVerifiedLazyField(const FieldDescriptor* field) const {
  return field->options().lazy();
}

bool Reflection::IsEagerlyVerifiedLazyField(
    const FieldDescriptor* field) const {
  return (field->type() == FieldDescriptor::TYPE_MESSAGE &&
          schema_.IsEagerlyVerifiedLazyField(field));
}

bool Reflection::IsInlined(const FieldDescriptor* field) const {
  return schema_.IsFieldInlined(field);
}

size_t Reflection::SpaceUsedLong(const Message& message) const {
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
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                           \
  case FieldDescriptor::CPPTYPE_##UPPERCASE:                        \
    total_size += GetRaw<RepeatedField<LOWERCASE> >(message, field) \
                      .SpaceUsedExcludingSelfLong();                \
    break

        HANDLE_TYPE(INT32, int32_t);
        HANDLE_TYPE(INT64, int64_t);
        HANDLE_TYPE(UINT32, uint32_t);
        HANDLE_TYPE(UINT64, uint64_t);
        HANDLE_TYPE(DOUBLE, double);
        HANDLE_TYPE(FLOAT, float);
        HANDLE_TYPE(BOOL, bool);
        HANDLE_TYPE(ENUM, int);
#undef HANDLE_TYPE

        case FieldDescriptor::CPPTYPE_STRING:
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING:
              total_size +=
                  GetRaw<RepeatedPtrField<std::string> >(message, field)
                      .SpaceUsedExcludingSelfLong();
              break;
          }
          break;

        case FieldDescriptor::CPPTYPE_MESSAGE:
          if (IsMapFieldInApi(field)) {
            total_size += GetRaw<internal::MapFieldBase>(message, field)
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
      if (schema_.InRealOneof(field) && !HasOneofField(message, field)) {
        continue;
      }
      switch (field->cpp_type()) {
        case FieldDescriptor::CPPTYPE_INT32:
        case FieldDescriptor::CPPTYPE_INT64:
        case FieldDescriptor::CPPTYPE_UINT32:
        case FieldDescriptor::CPPTYPE_UINT64:
        case FieldDescriptor::CPPTYPE_DOUBLE:
        case FieldDescriptor::CPPTYPE_FLOAT:
        case FieldDescriptor::CPPTYPE_BOOL:
        case FieldDescriptor::CPPTYPE_ENUM:
          // Field is inline, so we've already counted it.
          break;

        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              if (IsInlined(field)) {
                const std::string* ptr =
                    &GetField<InlinedStringField>(message, field).GetNoArena();
                total_size += StringSpaceUsedExcludingSelfLong(*ptr);
                break;
              }

              const std::string* ptr =
                  GetField<ArenaStringPtr>(message, field).GetPointer();

              // Initially, the string points to the default value stored
              // in the prototype. Only count the string if it has been
              // changed from the default value.
              // Except oneof fields, those never point to a default instance,
              // and there is no default instance to point to.
              if (schema_.InRealOneof(field) ||
                  ptr != DefaultRaw<ArenaStringPtr>(field).GetPointer()) {
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
            if (sub_message != nullptr) {
              total_size += sub_message->SpaceUsedLong();
            }
          }
          break;
      }
    }
  }
  return total_size;
}

namespace {

template <bool unsafe_shallow_swap>
struct OneofFieldMover {
  template <typename FromType, typename ToType>
  void operator()(const FieldDescriptor* field, FromType* from, ToType* to) {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32:
        to->SetInt32(from->GetInt32());
        break;
      case FieldDescriptor::CPPTYPE_INT64:
        to->SetInt64(from->GetInt64());
        break;
      case FieldDescriptor::CPPTYPE_UINT32:
        to->SetUint32(from->GetUint32());
        break;
      case FieldDescriptor::CPPTYPE_UINT64:
        to->SetUint64(from->GetUint64());
        break;
      case FieldDescriptor::CPPTYPE_FLOAT:
        to->SetFloat(from->GetFloat());
        break;
      case FieldDescriptor::CPPTYPE_DOUBLE:
        to->SetDouble(from->GetDouble());
        break;
      case FieldDescriptor::CPPTYPE_BOOL:
        to->SetBool(from->GetBool());
        break;
      case FieldDescriptor::CPPTYPE_ENUM:
        to->SetEnum(from->GetEnum());
        break;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (!unsafe_shallow_swap) {
          to->SetMessage(from->GetMessage());
        } else {
          to->UnsafeSetMessage(from->UnsafeGetMessage());
        }
        break;
      case FieldDescriptor::CPPTYPE_STRING:
        if (!unsafe_shallow_swap) {
          to->SetString(from->GetString());
          break;
        }
        switch (field->options().ctype()) {
          default:
          case FieldOptions::STRING: {
            to->SetArenaStringPtr(from->GetArenaStringPtr());
            break;
          }
        }
        break;
      default:
        GOOGLE_LOG(FATAL) << "unimplemented type: " << field->cpp_type();
    }
    if (unsafe_shallow_swap) {
      // Not clearing oneof case after move may cause unwanted "ClearOneof"
      // where the residual message or string value is deleted and causes
      // use-after-free (only for unsafe swap).
      from->ClearOneofCase();
    }
  }
};

}  // namespace

namespace internal {

class SwapFieldHelper {
 public:
  template <bool unsafe_shallow_swap>
  static void SwapRepeatedStringField(const Reflection* r, Message* lhs,
                                      Message* rhs,
                                      const FieldDescriptor* field);

  template <bool unsafe_shallow_swap>
  static void SwapInlinedStrings(const Reflection* r, Message* lhs,
                                 Message* rhs, const FieldDescriptor* field);

  template <bool unsafe_shallow_swap>
  static void SwapNonInlinedStrings(const Reflection* r, Message* lhs,
                                    Message* rhs, const FieldDescriptor* field);

  template <bool unsafe_shallow_swap>
  static void SwapStringField(const Reflection* r, Message* lhs, Message* rhs,
                              const FieldDescriptor* field);

  static void SwapArenaStringPtr(const std::string* default_ptr,
                                 ArenaStringPtr* lhs, Arena* lhs_arena,
                                 ArenaStringPtr* rhs, Arena* rhs_arena);

  template <bool unsafe_shallow_swap>
  static void SwapRepeatedMessageField(const Reflection* r, Message* lhs,
                                       Message* rhs,
                                       const FieldDescriptor* field);

  template <bool unsafe_shallow_swap>
  static void SwapMessageField(const Reflection* r, Message* lhs, Message* rhs,
                               const FieldDescriptor* field);

  static void SwapMessage(const Reflection* r, Message* lhs, Arena* lhs_arena,
                          Message* rhs, Arena* rhs_arena,
                          const FieldDescriptor* field);
};

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapRepeatedStringField(const Reflection* r, Message* lhs,
                                              Message* rhs,
                                              const FieldDescriptor* field) {
  switch (field->options().ctype()) {
    default:
    case FieldOptions::STRING: {
      auto* lhs_string = r->MutableRaw<RepeatedPtrFieldBase>(lhs, field);
      auto* rhs_string = r->MutableRaw<RepeatedPtrFieldBase>(rhs, field);
      if (unsafe_shallow_swap) {
        lhs_string->InternalSwap(rhs_string);
      } else {
        lhs_string->Swap<GenericTypeHandler<std::string>>(rhs_string);
      }
      break;
    }
  }
}

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapInlinedStrings(const Reflection* r, Message* lhs,
                                         Message* rhs,
                                         const FieldDescriptor* field) {
  // Inlined string field.
  Arena* lhs_arena = lhs->GetArenaForAllocation();
  Arena* rhs_arena = rhs->GetArenaForAllocation();
  auto* lhs_string = r->MutableRaw<InlinedStringField>(lhs, field);
  auto* rhs_string = r->MutableRaw<InlinedStringField>(rhs, field);
  const uint32 index = r->schema_.InlinedStringIndex(field);
  uint32* lhs_state = &r->MutableInlinedStringDonatedArray(lhs)[index / 32];
  uint32* rhs_state = &r->MutableInlinedStringDonatedArray(rhs)[index / 32];
  const uint32 mask = ~(static_cast<uint32>(1) << (index % 32));
  if (unsafe_shallow_swap || lhs_arena == rhs_arena) {
    lhs_string->Swap(rhs_string, /*default_value=*/nullptr, lhs_arena,
                     r->IsInlinedStringDonated(*lhs, field),
                     r->IsInlinedStringDonated(*rhs, field),
                     /*donating_states=*/lhs_state, rhs_state, mask);
  } else {
    const std::string temp = lhs_string->Get();
    lhs_string->Set(nullptr, rhs_string->Get(), lhs_arena,
                    r->IsInlinedStringDonated(*lhs, field), lhs_state, mask);
    rhs_string->Set(nullptr, temp, rhs_arena,
                    r->IsInlinedStringDonated(*rhs, field), rhs_state, mask);
  }
}

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapNonInlinedStrings(const Reflection* r, Message* lhs,
                                            Message* rhs,
                                            const FieldDescriptor* field) {
  ArenaStringPtr* lhs_string = r->MutableRaw<ArenaStringPtr>(lhs, field);
  ArenaStringPtr* rhs_string = r->MutableRaw<ArenaStringPtr>(rhs, field);
  if (unsafe_shallow_swap) {
    ArenaStringPtr::UnsafeShallowSwap(lhs_string, rhs_string);
  } else {
    SwapFieldHelper::SwapArenaStringPtr(
        r->DefaultRaw<ArenaStringPtr>(field).GetPointer(),  //
        lhs_string, lhs->GetArenaForAllocation(),           //
        rhs_string, rhs->GetArenaForAllocation());
  }
}

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapStringField(const Reflection* r, Message* lhs,
                                      Message* rhs,
                                      const FieldDescriptor* field) {
  switch (field->options().ctype()) {
    default:
    case FieldOptions::STRING: {
      if (r->IsInlined(field)) {
        SwapFieldHelper::SwapInlinedStrings<unsafe_shallow_swap>(r, lhs, rhs,
                                                                 field);
      } else {
        SwapFieldHelper::SwapNonInlinedStrings<unsafe_shallow_swap>(r, lhs, rhs,
                                                                    field);
      }
      break;
    }
  }
}

void SwapFieldHelper::SwapArenaStringPtr(const std::string* default_ptr,
                                         ArenaStringPtr* lhs, Arena* lhs_arena,
                                         ArenaStringPtr* rhs,
                                         Arena* rhs_arena) {
  if (lhs_arena == rhs_arena) {
    ArenaStringPtr::InternalSwap(default_ptr, lhs, lhs_arena, rhs, rhs_arena);
  } else if (lhs->IsDefault(default_ptr) && rhs->IsDefault(default_ptr)) {
    // Nothing to do.
  } else if (lhs->IsDefault(default_ptr)) {
    lhs->Set(default_ptr, rhs->Get(), lhs_arena);
    // rhs needs to be destroyed before overwritten.
    rhs->Destroy(default_ptr, rhs_arena);
    rhs->UnsafeSetDefault(default_ptr);
  } else if (rhs->IsDefault(default_ptr)) {
    rhs->Set(default_ptr, lhs->Get(), rhs_arena);
    // lhs needs to be destroyed before overwritten.
    lhs->Destroy(default_ptr, lhs_arena);
    lhs->UnsafeSetDefault(default_ptr);
  } else {
    std::string temp = lhs->Get();
    lhs->Set(default_ptr, rhs->Get(), lhs_arena);
    rhs->Set(default_ptr, std::move(temp), rhs_arena);
  }
}

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapRepeatedMessageField(const Reflection* r,
                                               Message* lhs, Message* rhs,
                                               const FieldDescriptor* field) {
  if (IsMapFieldInApi(field)) {
    auto* lhs_map = r->MutableRaw<MapFieldBase>(lhs, field);
    auto* rhs_map = r->MutableRaw<MapFieldBase>(rhs, field);
    if (unsafe_shallow_swap) {
      lhs_map->UnsafeShallowSwap(rhs_map);
    } else {
      lhs_map->Swap(rhs_map);
    }
  } else {
    auto* lhs_rm = r->MutableRaw<RepeatedPtrFieldBase>(lhs, field);
    auto* rhs_rm = r->MutableRaw<RepeatedPtrFieldBase>(rhs, field);
    if (unsafe_shallow_swap) {
      lhs_rm->InternalSwap(rhs_rm);
    } else {
      lhs_rm->Swap<GenericTypeHandler<Message>>(rhs_rm);
    }
  }
}

template <bool unsafe_shallow_swap>
void SwapFieldHelper::SwapMessageField(const Reflection* r, Message* lhs,
                                       Message* rhs,
                                       const FieldDescriptor* field) {
  if (unsafe_shallow_swap) {
    std::swap(*r->MutableRaw<Message*>(lhs, field),
              *r->MutableRaw<Message*>(rhs, field));
  } else {
    SwapMessage(r, lhs, lhs->GetArenaForAllocation(), rhs,
                rhs->GetArenaForAllocation(), field);
  }
}

void SwapFieldHelper::SwapMessage(const Reflection* r, Message* lhs,
                                  Arena* lhs_arena, Message* rhs,
                                  Arena* rhs_arena,
                                  const FieldDescriptor* field) {
  Message** lhs_sub = r->MutableRaw<Message*>(lhs, field);
  Message** rhs_sub = r->MutableRaw<Message*>(rhs, field);

  if (*lhs_sub == *rhs_sub) return;

#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
  if (lhs_arena != nullptr && lhs_arena == rhs_arena) {
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
  if (lhs_arena == rhs_arena) {
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
    std::swap(*lhs_sub, *rhs_sub);
    return;
  }

  if (*lhs_sub != nullptr && *rhs_sub != nullptr) {
    (*lhs_sub)->GetReflection()->Swap(*lhs_sub, *rhs_sub);
  } else if (*lhs_sub == nullptr && r->HasBit(*rhs, field)) {
    *lhs_sub = (*rhs_sub)->New(lhs_arena);
    (*lhs_sub)->CopyFrom(**rhs_sub);
    r->ClearField(rhs, field);
    // Ensures has bit is unchanged after ClearField.
    r->SetBit(rhs, field);
  } else if (*rhs_sub == nullptr && r->HasBit(*lhs, field)) {
    *rhs_sub = (*lhs_sub)->New(rhs_arena);
    (*rhs_sub)->CopyFrom(**lhs_sub);
    r->ClearField(lhs, field);
    // Ensures has bit is unchanged after ClearField.
    r->SetBit(lhs, field);
  }
}

}  // namespace internal

void Reflection::SwapField(Message* message1, Message* message2,
                           const FieldDescriptor* field) const {
  if (field->is_repeated()) {
    switch (field->cpp_type()) {
#define SWAP_ARRAYS(CPPTYPE, TYPE)                                 \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                         \
    MutableRaw<RepeatedField<TYPE> >(message1, field)              \
        ->Swap(MutableRaw<RepeatedField<TYPE> >(message2, field)); \
    break;

      SWAP_ARRAYS(INT32, int32_t);
      SWAP_ARRAYS(INT64, int64_t);
      SWAP_ARRAYS(UINT32, uint32_t);
      SWAP_ARRAYS(UINT64, uint64_t);
      SWAP_ARRAYS(FLOAT, float);
      SWAP_ARRAYS(DOUBLE, double);
      SWAP_ARRAYS(BOOL, bool);
      SWAP_ARRAYS(ENUM, int);
#undef SWAP_ARRAYS

      case FieldDescriptor::CPPTYPE_STRING:
        internal::SwapFieldHelper::SwapRepeatedStringField<false>(
            this, message1, message2, field);
        break;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        internal::SwapFieldHelper::SwapRepeatedMessageField<false>(
            this, message1, message2, field);
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field->cpp_type();
    }
  } else {
    switch (field->cpp_type()) {
#define SWAP_VALUES(CPPTYPE, TYPE)                 \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:         \
    std::swap(*MutableRaw<TYPE>(message1, field),  \
              *MutableRaw<TYPE>(message2, field)); \
    break;

      SWAP_VALUES(INT32, int32_t);
      SWAP_VALUES(INT64, int64_t);
      SWAP_VALUES(UINT32, uint32_t);
      SWAP_VALUES(UINT64, uint64_t);
      SWAP_VALUES(FLOAT, float);
      SWAP_VALUES(DOUBLE, double);
      SWAP_VALUES(BOOL, bool);
      SWAP_VALUES(ENUM, int);
#undef SWAP_VALUES
      case FieldDescriptor::CPPTYPE_MESSAGE:
        internal::SwapFieldHelper::SwapMessageField<false>(this, message1,
                                                           message2, field);
        break;

      case FieldDescriptor::CPPTYPE_STRING:
        internal::SwapFieldHelper::SwapStringField<false>(this, message1,
                                                          message2, field);
        break;

      default:
        GOOGLE_LOG(FATAL) << "Unimplemented type: " << field->cpp_type();
    }
  }
}

void Reflection::UnsafeShallowSwapField(Message* message1, Message* message2,
                                        const FieldDescriptor* field) const {
  if (!field->is_repeated()) {
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      internal::SwapFieldHelper::SwapMessageField<true>(this, message1,
                                                        message2, field);
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_STRING) {
      internal::SwapFieldHelper::SwapStringField<true>(this, message1, message2,
                                                       field);
    } else {
      SwapField(message1, message2, field);
    }
    return;
  }

  switch (field->cpp_type()) {
#define SHALLOW_SWAP_ARRAYS(CPPTYPE, TYPE)                                \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                                \
    MutableRaw<RepeatedField<TYPE>>(message1, field)                      \
        ->InternalSwap(MutableRaw<RepeatedField<TYPE>>(message2, field)); \
    break;

    SHALLOW_SWAP_ARRAYS(INT32, int32_t);
    SHALLOW_SWAP_ARRAYS(INT64, int64_t);
    SHALLOW_SWAP_ARRAYS(UINT32, uint32_t);
    SHALLOW_SWAP_ARRAYS(UINT64, uint64_t);
    SHALLOW_SWAP_ARRAYS(FLOAT, float);
    SHALLOW_SWAP_ARRAYS(DOUBLE, double);
    SHALLOW_SWAP_ARRAYS(BOOL, bool);
    SHALLOW_SWAP_ARRAYS(ENUM, int);
#undef SHALLOW_SWAP_ARRAYS

    case FieldDescriptor::CPPTYPE_STRING:
      internal::SwapFieldHelper::SwapRepeatedStringField<true>(this, message1,
                                                               message2, field);
      break;
    case FieldDescriptor::CPPTYPE_MESSAGE:
      internal::SwapFieldHelper::SwapRepeatedMessageField<true>(
          this, message1, message2, field);
      break;

    default:
      GOOGLE_LOG(FATAL) << "Unimplemented type: " << field->cpp_type();
  }
}

// Swaps oneof field between lhs and rhs. If unsafe_shallow_swap is true, it
// directly swaps oneof values; otherwise, it may involve copy/delete. Note that
// two messages may have different oneof cases. So, it has to be done in three
// steps (i.e. lhs -> temp, rhs -> lhs, temp -> rhs).
template <bool unsafe_shallow_swap>
void Reflection::SwapOneofField(Message* lhs, Message* rhs,
                                const OneofDescriptor* oneof_descriptor) const {
  // Wraps a local variable to temporarily store oneof value.
  struct LocalVarWrapper {
#define LOCAL_VAR_ACCESSOR(type, var, name)               \
  type Get##name() const { return oneof_val.type_##var; } \
  void Set##name(type v) { oneof_val.type_##var = v; }

    LOCAL_VAR_ACCESSOR(int32_t, int32, Int32);
    LOCAL_VAR_ACCESSOR(int64_t, int64, Int64);
    LOCAL_VAR_ACCESSOR(uint32_t, uint32, Uint32);
    LOCAL_VAR_ACCESSOR(uint64_t, uint64, Uint64);
    LOCAL_VAR_ACCESSOR(float, float, Float);
    LOCAL_VAR_ACCESSOR(double, double, Double);
    LOCAL_VAR_ACCESSOR(bool, bool, Bool);
    LOCAL_VAR_ACCESSOR(int, enum, Enum);
    LOCAL_VAR_ACCESSOR(Message*, message, Message);
    LOCAL_VAR_ACCESSOR(ArenaStringPtr, arena_string_ptr, ArenaStringPtr);
    const std::string& GetString() const { return string_val; }
    void SetString(const std::string& v) { string_val = v; }
    Message* UnsafeGetMessage() const { return GetMessage(); }
    void UnsafeSetMessage(Message* v) { SetMessage(v); }
    void ClearOneofCase() {}

    union {
      int32_t type_int32;
      int64_t type_int64;
      uint32_t type_uint32;
      uint64_t type_uint64;
      float type_float;
      double type_double;
      bool type_bool;
      int type_enum;
      Message* type_message;
      internal::ArenaStringPtr type_arena_string_ptr;
    } oneof_val;

    // std::string cannot be in union.
    std::string string_val;
  };

  // Wraps a message pointer to read and write a field.
  struct MessageWrapper {
#define MESSAGE_FIELD_ACCESSOR(type, var, name)         \
  type Get##name() const {                              \
    return reflection->GetField<type>(*message, field); \
  }                                                     \
  void Set##name(type v) { reflection->SetField<type>(message, field, v); }

    MESSAGE_FIELD_ACCESSOR(int32_t, int32, Int32);
    MESSAGE_FIELD_ACCESSOR(int64_t, int64, Int64);
    MESSAGE_FIELD_ACCESSOR(uint32_t, uint32, Uint32);
    MESSAGE_FIELD_ACCESSOR(uint64_t, uint64, Uint64);
    MESSAGE_FIELD_ACCESSOR(float, float, Float);
    MESSAGE_FIELD_ACCESSOR(double, double, Double);
    MESSAGE_FIELD_ACCESSOR(bool, bool, Bool);
    MESSAGE_FIELD_ACCESSOR(int, enum, Enum);
    MESSAGE_FIELD_ACCESSOR(ArenaStringPtr, arena_string_ptr, ArenaStringPtr);
    std::string GetString() const {
      return reflection->GetString(*message, field);
    }
    void SetString(const std::string& v) {
      reflection->SetString(message, field, v);
    }
    Message* GetMessage() const {
      return reflection->ReleaseMessage(message, field);
    }
    void SetMessage(Message* v) {
      reflection->SetAllocatedMessage(message, v, field);
    }
    Message* UnsafeGetMessage() const {
      return reflection->UnsafeArenaReleaseMessage(message, field);
    }
    void UnsafeSetMessage(Message* v) {
      reflection->UnsafeArenaSetAllocatedMessage(message, v, field);
    }
    void ClearOneofCase() {
      *reflection->MutableOneofCase(message, field->containing_oneof()) = 0;
    }

    const Reflection* reflection;
    Message* message;
    const FieldDescriptor* field;
  };

  GOOGLE_DCHECK(!oneof_descriptor->is_synthetic());
  uint32 oneof_case_lhs = GetOneofCase(*lhs, oneof_descriptor);
  uint32 oneof_case_rhs = GetOneofCase(*rhs, oneof_descriptor);

  LocalVarWrapper temp;
  MessageWrapper lhs_wrapper, rhs_wrapper;
  const FieldDescriptor* field_lhs = nullptr;
  OneofFieldMover<unsafe_shallow_swap> mover;
  // lhs --> temp
  if (oneof_case_lhs > 0) {
    field_lhs = descriptor_->FindFieldByNumber(oneof_case_lhs);
    lhs_wrapper = {this, lhs, field_lhs};
    mover(field_lhs, &lhs_wrapper, &temp);
  }
  // rhs --> lhs
  if (oneof_case_rhs > 0) {
    const FieldDescriptor* f = descriptor_->FindFieldByNumber(oneof_case_rhs);
    lhs_wrapper = {this, lhs, f};
    rhs_wrapper = {this, rhs, f};
    mover(f, &rhs_wrapper, &lhs_wrapper);
  } else if (!unsafe_shallow_swap) {
    ClearOneof(lhs, oneof_descriptor);
  }
  // temp --> rhs
  if (oneof_case_lhs > 0) {
    rhs_wrapper = {this, rhs, field_lhs};
    mover(field_lhs, &temp, &rhs_wrapper);
  } else if (!unsafe_shallow_swap) {
    ClearOneof(rhs, oneof_descriptor);
  }

  if (unsafe_shallow_swap) {
    *MutableOneofCase(lhs, oneof_descriptor) = oneof_case_rhs;
    *MutableOneofCase(rhs, oneof_descriptor) = oneof_case_lhs;
  }
}

void Reflection::Swap(Message* message1, Message* message2) const {
  if (message1 == message2) return;

  // TODO(kenton):  Other Reflection methods should probably check this too.
  GOOGLE_CHECK_EQ(message1->GetReflection(), this)
      << "First argument to Swap() (of type \""
      << message1->GetDescriptor()->full_name()
      << "\") is not compatible with this reflection object (which is for type "
         "\""
      << descriptor_->full_name()
      << "\").  Note that the exact same class is required; not just the same "
         "descriptor.";
  GOOGLE_CHECK_EQ(message2->GetReflection(), this)
      << "Second argument to Swap() (of type \""
      << message2->GetDescriptor()->full_name()
      << "\") is not compatible with this reflection object (which is for type "
         "\""
      << descriptor_->full_name()
      << "\").  Note that the exact same class is required; not just the same "
         "descriptor.";

  // Check that both messages are in the same arena (or both on the heap). We
  // need to copy all data if not, due to ownership semantics.
#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
  if (message1->GetOwningArena() == nullptr ||
      message1->GetOwningArena() != message2->GetOwningArena()) {
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
  if (message1->GetOwningArena() != message2->GetOwningArena()) {
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
    // One of the two is guaranteed to have an arena.  Switch things around
    // to guarantee that message1 has an arena.
    Arena* arena = message1->GetOwningArena();
    if (arena == nullptr) {
      arena = message2->GetOwningArena();
      std::swap(message1, message2);  // Swapping names for pointers!
    }

    Message* temp = message1->New(arena);
    temp->MergeFrom(*message2);
    message2->CopyFrom(*message1);
#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    message1->CopyFrom(*temp);
    if (arena == nullptr) delete temp;
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
    Swap(message1, temp);
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
    return;
  }

  GOOGLE_DCHECK_EQ(message1->GetOwningArena(), message2->GetOwningArena());

  UnsafeArenaSwap(message1, message2);
}

template <bool unsafe_shallow_swap>
void Reflection::SwapFieldsImpl(
    Message* message1, Message* message2,
    const std::vector<const FieldDescriptor*>& fields) const {
  if (message1 == message2) return;

  // TODO(kenton):  Other Reflection methods should probably check this too.
  GOOGLE_CHECK_EQ(message1->GetReflection(), this)
      << "First argument to SwapFields() (of type \""
      << message1->GetDescriptor()->full_name()
      << "\") is not compatible with this reflection object (which is for type "
         "\""
      << descriptor_->full_name()
      << "\").  Note that the exact same class is required; not just the same "
         "descriptor.";
  GOOGLE_CHECK_EQ(message2->GetReflection(), this)
      << "Second argument to SwapFields() (of type \""
      << message2->GetDescriptor()->full_name()
      << "\") is not compatible with this reflection object (which is for type "
         "\""
      << descriptor_->full_name()
      << "\").  Note that the exact same class is required; not just the same "
         "descriptor.";

  std::set<int> swapped_oneof;

  GOOGLE_DCHECK(!unsafe_shallow_swap || message1->GetArenaForAllocation() ==
                                     message2->GetArenaForAllocation());

  const Message* prototype =
      message_factory_->GetPrototype(message1->GetDescriptor());
  for (const auto* field : fields) {
    CheckInvalidAccess(schema_, field);
    if (field->is_extension()) {
      if (unsafe_shallow_swap) {
        MutableExtensionSet(message1)->UnsafeShallowSwapExtension(
            MutableExtensionSet(message2), field->number());
      } else {
        MutableExtensionSet(message1)->SwapExtension(
            prototype, MutableExtensionSet(message2), field->number());
      }
    } else {
      if (schema_.InRealOneof(field)) {
        int oneof_index = field->containing_oneof()->index();
        // Only swap the oneof field once.
        if (swapped_oneof.find(oneof_index) != swapped_oneof.end()) {
          continue;
        }
        swapped_oneof.insert(oneof_index);
        SwapOneofField<unsafe_shallow_swap>(message1, message2,
                                            field->containing_oneof());
      } else {
        // Swap field.
        if (unsafe_shallow_swap) {
          UnsafeShallowSwapField(message1, message2, field);
        } else {
          SwapField(message1, message2, field);
        }
        // Swap has bit for non-repeated fields.  We have already checked for
        // oneof already. This has to be done after SwapField, because SwapField
        // may depend on the information in has bits.
        if (!field->is_repeated()) {
          SwapBit(message1, message2, field);
        }
      }
    }
  }
}

void Reflection::SwapFields(
    Message* message1, Message* message2,
    const std::vector<const FieldDescriptor*>& fields) const {
  SwapFieldsImpl<false>(message1, message2, fields);
}

void Reflection::UnsafeShallowSwapFields(
    Message* message1, Message* message2,
    const std::vector<const FieldDescriptor*>& fields) const {
  SwapFieldsImpl<true>(message1, message2, fields);
}

void Reflection::UnsafeArenaSwapFields(
    Message* lhs, Message* rhs,
    const std::vector<const FieldDescriptor*>& fields) const {
  GOOGLE_DCHECK_EQ(lhs->GetArenaForAllocation(), rhs->GetArenaForAllocation());
  UnsafeShallowSwapFields(lhs, rhs, fields);
}

// -------------------------------------------------------------------

bool Reflection::HasField(const Message& message,
                          const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(HasField);
  USAGE_CHECK_SINGULAR(HasField);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    return GetExtensionSet(message).Has(field->number());
  } else {
    if (schema_.InRealOneof(field)) {
      return HasOneofField(message, field);
    } else {
      return HasBit(message, field);
    }
  }
}

void Reflection::UnsafeArenaSwap(Message* lhs, Message* rhs) const {
  if (lhs == rhs) return;

  MutableInternalMetadata(lhs)->InternalSwap(MutableInternalMetadata(rhs));

  for (int i = 0; i <= last_non_weak_field_index_; i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (schema_.InRealOneof(field)) continue;
    if (schema_.IsFieldStripped(field)) continue;
    UnsafeShallowSwapField(lhs, rhs, field);
  }
  const int oneof_decl_count = descriptor_->oneof_decl_count();
  for (int i = 0; i < oneof_decl_count; i++) {
    const OneofDescriptor* oneof = descriptor_->oneof_decl(i);
    if (!oneof->is_synthetic()) {
      SwapOneofField<true>(lhs, rhs, oneof);
    }
  }

  // Swapping bits need to happen after swapping fields, because the latter may
  // depend on the has bit information.
  if (schema_.HasHasbits()) {
    uint32* lhs_has_bits = MutableHasBits(lhs);
    uint32* rhs_has_bits = MutableHasBits(rhs);

    int fields_with_has_bits = 0;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      if (field->is_repeated() || schema_.InRealOneof(field)) {
        continue;
      }
      fields_with_has_bits++;
    }

    int has_bits_size = (fields_with_has_bits + 31) / 32;

    for (int i = 0; i < has_bits_size; i++) {
      std::swap(lhs_has_bits[i], rhs_has_bits[i]);
    }
  }

  if (schema_.HasExtensionSet()) {
    MutableExtensionSet(lhs)->InternalSwap(MutableExtensionSet(rhs));
  }
}

int Reflection::FieldSize(const Message& message,
                          const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(FieldSize);
  USAGE_CHECK_REPEATED(FieldSize);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    return GetExtensionSet(message).ExtensionSize(field->number());
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)    \
  case FieldDescriptor::CPPTYPE_##UPPERCASE: \
    return GetRaw<RepeatedField<LOWERCASE> >(message, field).size()

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (IsMapFieldInApi(field)) {
          const internal::MapFieldBase& map =
              GetRaw<MapFieldBase>(message, field);
          if (map.IsRepeatedFieldValid()) {
            return map.GetRepeatedField().size();
          } else {
            // No need to materialize the repeated field if it is out of sync:
            // its size will be the same as the map's size.
            return map.size();
          }
        } else {
          return GetRaw<RepeatedPtrFieldBase>(message, field).size();
        }
    }

    GOOGLE_LOG(FATAL) << "Can't get here.";
    return 0;
  }
}

void Reflection::ClearField(Message* message,
                            const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(ClearField);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    MutableExtensionSet(message)->ClearExtension(field->number());
  } else if (!field->is_repeated()) {
    if (schema_.InRealOneof(field)) {
      ClearOneofField(message, field);
      return;
    }
    if (HasBit(*message, field)) {
      ClearBit(message, field);

      // We need to set the field back to its default value.
      switch (field->cpp_type()) {
#define CLEAR_TYPE(CPPTYPE, TYPE)                                      \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                             \
    *MutableRaw<TYPE>(message, field) = field->default_value_##TYPE(); \
    break;

        CLEAR_TYPE(INT32, int32_t);
        CLEAR_TYPE(INT64, int64_t);
        CLEAR_TYPE(UINT32, uint32_t);
        CLEAR_TYPE(UINT64, uint64_t);
        CLEAR_TYPE(FLOAT, float);
        CLEAR_TYPE(DOUBLE, double);
        CLEAR_TYPE(BOOL, bool);
#undef CLEAR_TYPE

        case FieldDescriptor::CPPTYPE_ENUM:
          *MutableRaw<int>(message, field) =
              field->default_value_enum()->number();
          break;

        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              if (IsInlined(field)) {
                // Currently, string with default value can't be inlined. So we
                // don't have to handle default value here.
                MutableRaw<InlinedStringField>(message, field)->ClearToEmpty();
                break;
              }
              const std::string* default_ptr =
                  DefaultRaw<ArenaStringPtr>(field).GetPointer();
              MutableRaw<ArenaStringPtr>(message, field)
                  ->SetAllocated(default_ptr, nullptr,
                                 message->GetArenaForAllocation());
              break;
            }
          }
          break;
        }

        case FieldDescriptor::CPPTYPE_MESSAGE:
          if (schema_.HasBitIndex(field) == static_cast<uint32_t>(-1)) {
            // Proto3 does not have has-bits and we need to set a message field
            // to nullptr in order to indicate its un-presence.
            if (message->GetArenaForAllocation() == nullptr) {
              delete *MutableRaw<Message*>(message, field);
            }
            *MutableRaw<Message*>(message, field) = nullptr;
          } else {
            (*MutableRaw<Message*>(message, field))->Clear();
          }
          break;
      }
    }
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                           \
  case FieldDescriptor::CPPTYPE_##UPPERCASE:                        \
    MutableRaw<RepeatedField<LOWERCASE> >(message, field)->Clear(); \
    break

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING: {
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            MutableRaw<RepeatedPtrField<std::string> >(message, field)->Clear();
            break;
        }
        break;
      }

      case FieldDescriptor::CPPTYPE_MESSAGE: {
        if (IsMapFieldInApi(field)) {
          MutableRaw<MapFieldBase>(message, field)->Clear();
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

void Reflection::RemoveLast(Message* message,
                            const FieldDescriptor* field) const {
  USAGE_CHECK_MESSAGE_TYPE(RemoveLast);
  USAGE_CHECK_REPEATED(RemoveLast);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    MutableExtensionSet(message)->RemoveLast(field->number());
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                \
  case FieldDescriptor::CPPTYPE_##UPPERCASE:                             \
    MutableRaw<RepeatedField<LOWERCASE> >(message, field)->RemoveLast(); \
    break

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, int);
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // TODO(kenton):  Support other string reps.
          case FieldOptions::STRING:
            MutableRaw<RepeatedPtrField<std::string> >(message, field)
                ->RemoveLast();
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

Message* Reflection::ReleaseLast(Message* message,
                                 const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(ReleaseLast, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  Message* released;
  if (field->is_extension()) {
    released = static_cast<Message*>(
        MutableExtensionSet(message)->ReleaseLast(field->number()));
  } else {
    if (IsMapFieldInApi(field)) {
      released = MutableRaw<MapFieldBase>(message, field)
                     ->MutableRepeatedField()
                     ->ReleaseLast<GenericTypeHandler<Message>>();
    } else {
      released = MutableRaw<RepeatedPtrFieldBase>(message, field)
                     ->ReleaseLast<GenericTypeHandler<Message>>();
    }
  }
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  return MaybeForceCopy(message->GetArenaForAllocation(), released);
#else   // PROTOBUF_FORCE_COPY_IN_RELEASE
  return released;
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
}

Message* Reflection::UnsafeArenaReleaseLast(
    Message* message, const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(UnsafeArenaReleaseLast, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->UnsafeArenaReleaseLast(field->number()));
  } else {
    if (IsMapFieldInApi(field)) {
      return MutableRaw<MapFieldBase>(message, field)
          ->MutableRepeatedField()
          ->UnsafeArenaReleaseLast<GenericTypeHandler<Message>>();
    } else {
      return MutableRaw<RepeatedPtrFieldBase>(message, field)
          ->UnsafeArenaReleaseLast<GenericTypeHandler<Message>>();
    }
  }
}

void Reflection::SwapElements(Message* message, const FieldDescriptor* field,
                              int index1, int index2) const {
  USAGE_CHECK_MESSAGE_TYPE(Swap);
  USAGE_CHECK_REPEATED(Swap);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    MutableExtensionSet(message)->SwapElements(field->number(), index1, index2);
  } else {
    switch (field->cpp_type()) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                 \
  case FieldDescriptor::CPPTYPE_##UPPERCASE:              \
    MutableRaw<RepeatedField<LOWERCASE> >(message, field) \
        ->SwapElements(index1, index2);                   \
    break

      HANDLE_TYPE(INT32, int32_t);
      HANDLE_TYPE(INT64, int64_t);
      HANDLE_TYPE(UINT32, uint32_t);
      HANDLE_TYPE(UINT64, uint64_t);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(ENUM, int);
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

bool IsIndexInHasBitSet(const uint32_t* has_bit_set, uint32_t has_bit_index) {
  GOOGLE_DCHECK_NE(has_bit_index, ~0u);
  return ((has_bit_set[has_bit_index / 32] >> (has_bit_index % 32)) &
          static_cast<uint32_t>(1)) != 0;
}

bool CreateUnknownEnumValues(const FileDescriptor* file) {
  return file->syntax() == FileDescriptor::SYNTAX_PROTO3;
}
}  // namespace

namespace internal {
bool CreateUnknownEnumValues(const FieldDescriptor* field) {
  bool open_enum = false;
  return field->file()->syntax() == FileDescriptor::SYNTAX_PROTO3 || open_enum;
}
}  // namespace internal
using internal::CreateUnknownEnumValues;

void Reflection::ListFieldsMayFailOnStripped(
    const Message& message, bool should_fail,
    std::vector<const FieldDescriptor*>* output) const {
  output->clear();

  // Optimization:  The default instance never has any fields set.
  if (schema_.IsDefaultInstance(message)) return;

  // Optimization: Avoid calling GetHasBits() and HasOneofField() many times
  // within the field loop.  We allow this violation of ReflectionSchema
  // encapsulation because this function takes a noticeable about of CPU
  // fleetwide and properly allowing this optimization through public interfaces
  // seems more trouble than it is worth.
  const uint32_t* const has_bits =
      schema_.HasHasbits() ? GetHasBits(message) : nullptr;
  const uint32_t* const has_bits_indices = schema_.has_bit_indices_;
  output->reserve(descriptor_->field_count());
  const int last_non_weak_field_index = last_non_weak_field_index_;
  for (int i = 0; i <= last_non_weak_field_index; i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (!should_fail && schema_.IsFieldStripped(field)) {
      continue;
    }
    if (field->is_repeated()) {
      if (FieldSize(message, field) > 0) {
        output->push_back(field);
      }
    } else {
      const OneofDescriptor* containing_oneof = field->containing_oneof();
      if (schema_.InRealOneof(field)) {
        const uint32_t* const oneof_case_array =
            GetConstPointerAtOffset<uint32_t>(&message,
                                              schema_.oneof_case_offset_);
        // Equivalent to: HasOneofField(message, field)
        if (static_cast<int64_t>(oneof_case_array[containing_oneof->index()]) ==
            field->number()) {
          output->push_back(field);
        }
      } else if (has_bits && has_bits_indices[i] != static_cast<uint32_t>(-1)) {
        CheckInvalidAccess(schema_, field);
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

void Reflection::ListFields(const Message& message,
                            std::vector<const FieldDescriptor*>* output) const {
  ListFieldsMayFailOnStripped(message, true, output);
}

void Reflection::ListFieldsOmitStripped(
    const Message& message, std::vector<const FieldDescriptor*>* output) const {
  ListFieldsMayFailOnStripped(message, false, output);
}

// -------------------------------------------------------------------

#undef DEFINE_PRIMITIVE_ACCESSORS
#define DEFINE_PRIMITIVE_ACCESSORS(TYPENAME, TYPE, PASSTYPE, CPPTYPE)          \
  PASSTYPE Reflection::Get##TYPENAME(const Message& message,                   \
                                     const FieldDescriptor* field) const {     \
    USAGE_CHECK_ALL(Get##TYPENAME, SINGULAR, CPPTYPE);                         \
    if (field->is_extension()) {                                               \
      return GetExtensionSet(message).Get##TYPENAME(                           \
          field->number(), field->default_value_##PASSTYPE());                 \
    } else if (schema_.InRealOneof(field) && !HasOneofField(message, field)) { \
      return field->default_value_##PASSTYPE();                                \
    } else {                                                                   \
      return GetField<TYPE>(message, field);                                   \
    }                                                                          \
  }                                                                            \
                                                                               \
  void Reflection::Set##TYPENAME(                                              \
      Message* message, const FieldDescriptor* field, PASSTYPE value) const {  \
    USAGE_CHECK_ALL(Set##TYPENAME, SINGULAR, CPPTYPE);                         \
    if (field->is_extension()) {                                               \
      return MutableExtensionSet(message)->Set##TYPENAME(                      \
          field->number(), field->type(), value, field);                       \
    } else {                                                                   \
      SetField<TYPE>(message, field, value);                                   \
    }                                                                          \
  }                                                                            \
                                                                               \
  PASSTYPE Reflection::GetRepeated##TYPENAME(                                  \
      const Message& message, const FieldDescriptor* field, int index) const { \
    USAGE_CHECK_ALL(GetRepeated##TYPENAME, REPEATED, CPPTYPE);                 \
    if (field->is_extension()) {                                               \
      return GetExtensionSet(message).GetRepeated##TYPENAME(field->number(),   \
                                                            index);            \
    } else {                                                                   \
      return GetRepeatedField<TYPE>(message, field, index);                    \
    }                                                                          \
  }                                                                            \
                                                                               \
  void Reflection::SetRepeated##TYPENAME(Message* message,                     \
                                         const FieldDescriptor* field,         \
                                         int index, PASSTYPE value) const {    \
    USAGE_CHECK_ALL(SetRepeated##TYPENAME, REPEATED, CPPTYPE);                 \
    if (field->is_extension()) {                                               \
      MutableExtensionSet(message)->SetRepeated##TYPENAME(field->number(),     \
                                                          index, value);       \
    } else {                                                                   \
      SetRepeatedField<TYPE>(message, field, index, value);                    \
    }                                                                          \
  }                                                                            \
                                                                               \
  void Reflection::Add##TYPENAME(                                              \
      Message* message, const FieldDescriptor* field, PASSTYPE value) const {  \
    USAGE_CHECK_ALL(Add##TYPENAME, REPEATED, CPPTYPE);                         \
    if (field->is_extension()) {                                               \
      MutableExtensionSet(message)->Add##TYPENAME(                             \
          field->number(), field->type(), field->options().packed(), value,    \
          field);                                                              \
    } else {                                                                   \
      AddField<TYPE>(message, field, value);                                   \
    }                                                                          \
  }

DEFINE_PRIMITIVE_ACCESSORS(Int32, int32_t, int32_t, INT32)
DEFINE_PRIMITIVE_ACCESSORS(Int64, int64_t, int64_t, INT64)
DEFINE_PRIMITIVE_ACCESSORS(UInt32, uint32_t, uint32_t, UINT32)
DEFINE_PRIMITIVE_ACCESSORS(UInt64, uint64_t, uint64_t, UINT64)
DEFINE_PRIMITIVE_ACCESSORS(Float, float, float, FLOAT)
DEFINE_PRIMITIVE_ACCESSORS(Double, double, double, DOUBLE)
DEFINE_PRIMITIVE_ACCESSORS(Bool, bool, bool, BOOL)
#undef DEFINE_PRIMITIVE_ACCESSORS

// -------------------------------------------------------------------

std::string Reflection::GetString(const Message& message,
                                  const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(GetString, SINGULAR, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetString(field->number(),
                                              field->default_value_string());
  } else {
    if (schema_.InRealOneof(field) && !HasOneofField(message, field)) {
      return field->default_value_string();
    }
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        if (IsInlined(field)) {
          return GetField<InlinedStringField>(message, field).GetNoArena();
        }

        if (auto* value =
                GetField<ArenaStringPtr>(message, field).GetPointer()) {
          return *value;
        }
        return field->default_value_string();
      }
    }
  }
}

const std::string& Reflection::GetStringReference(const Message& message,
                                                  const FieldDescriptor* field,
                                                  std::string* scratch) const {
  (void)scratch;  // Parameter is used by Google-internal code.
  USAGE_CHECK_ALL(GetStringReference, SINGULAR, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetString(field->number(),
                                              field->default_value_string());
  } else {
    if (schema_.InRealOneof(field) && !HasOneofField(message, field)) {
      return field->default_value_string();
    }
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        if (IsInlined(field)) {
          return GetField<InlinedStringField>(message, field).GetNoArena();
        }

        if (auto* value =
                GetField<ArenaStringPtr>(message, field).GetPointer()) {
          return *value;
        }
        return field->default_value_string();
      }
    }
  }
}


void Reflection::SetString(Message* message, const FieldDescriptor* field,
                           std::string value) const {
  USAGE_CHECK_ALL(SetString, SINGULAR, STRING);
  if (field->is_extension()) {
    return MutableExtensionSet(message)->SetString(
        field->number(), field->type(), std::move(value), field);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING: {
        if (IsInlined(field)) {
          const uint32_t index = schema_.InlinedStringIndex(field);
          uint32_t* states =
              &MutableInlinedStringDonatedArray(message)[index / 32];
          uint32_t mask = ~(static_cast<uint32_t>(1) << (index % 32));
          MutableField<InlinedStringField>(message, field)
              ->Set(nullptr, value, message->GetArenaForAllocation(),
                    IsInlinedStringDonated(*message, field), states, mask);
          break;
        }

        // Oneof string fields are never set as a default instance.
        // We just need to pass some arbitrary default string to make it work.
        // This allows us to not have the real default accessible from
        // reflection.
        const std::string* default_ptr =
            schema_.InRealOneof(field)
                ? nullptr
                : DefaultRaw<ArenaStringPtr>(field).GetPointer();
        if (schema_.InRealOneof(field) && !HasOneofField(*message, field)) {
          ClearOneof(message, field->containing_oneof());
          MutableField<ArenaStringPtr>(message, field)
              ->UnsafeSetDefault(default_ptr);
        }
        MutableField<ArenaStringPtr>(message, field)
            ->Set(default_ptr, std::move(value),
                  message->GetArenaForAllocation());
        break;
      }
    }
  }
}


std::string Reflection::GetRepeatedString(const Message& message,
                                          const FieldDescriptor* field,
                                          int index) const {
  USAGE_CHECK_ALL(GetRepeatedString, REPEATED, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetRepeatedString(field->number(), index);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        return GetRepeatedPtrField<std::string>(message, field, index);
    }
  }
}

const std::string& Reflection::GetRepeatedStringReference(
    const Message& message, const FieldDescriptor* field, int index,
    std::string* scratch) const {
  (void)scratch;  // Parameter is used by Google-internal code.
  USAGE_CHECK_ALL(GetRepeatedStringReference, REPEATED, STRING);
  if (field->is_extension()) {
    return GetExtensionSet(message).GetRepeatedString(field->number(), index);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        return GetRepeatedPtrField<std::string>(message, field, index);
    }
  }
}


void Reflection::SetRepeatedString(Message* message,
                                   const FieldDescriptor* field, int index,
                                   std::string value) const {
  USAGE_CHECK_ALL(SetRepeatedString, REPEATED, STRING);
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetRepeatedString(field->number(), index,
                                                    std::move(value));
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        MutableRepeatedField<std::string>(message, field, index)
            ->assign(std::move(value));
        break;
    }
  }
}


void Reflection::AddString(Message* message, const FieldDescriptor* field,
                           std::string value) const {
  USAGE_CHECK_ALL(AddString, REPEATED, STRING);
  if (field->is_extension()) {
    MutableExtensionSet(message)->AddString(field->number(), field->type(),
                                            std::move(value), field);
  } else {
    switch (field->options().ctype()) {
      default:  // TODO(kenton):  Support other string reps.
      case FieldOptions::STRING:
        AddField<std::string>(message, field)->assign(std::move(value));
        break;
    }
  }
}


// -------------------------------------------------------------------

const EnumValueDescriptor* Reflection::GetEnum(
    const Message& message, const FieldDescriptor* field) const {
  // Usage checked by GetEnumValue.
  int value = GetEnumValue(message, field);
  return field->enum_type()->FindValueByNumberCreatingIfUnknown(value);
}

int Reflection::GetEnumValue(const Message& message,
                             const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(GetEnumValue, SINGULAR, ENUM);

  int32_t value;
  if (field->is_extension()) {
    value = GetExtensionSet(message).GetEnum(
        field->number(), field->default_value_enum()->number());
  } else if (schema_.InRealOneof(field) && !HasOneofField(message, field)) {
    value = field->default_value_enum()->number();
  } else {
    value = GetField<int>(message, field);
  }
  return value;
}

void Reflection::SetEnum(Message* message, const FieldDescriptor* field,
                         const EnumValueDescriptor* value) const {
  // Usage checked by SetEnumValue.
  USAGE_CHECK_ENUM_VALUE(SetEnum);
  SetEnumValueInternal(message, field, value->number());
}

void Reflection::SetEnumValue(Message* message, const FieldDescriptor* field,
                              int value) const {
  USAGE_CHECK_ALL(SetEnumValue, SINGULAR, ENUM);
  if (!CreateUnknownEnumValues(field)) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == nullptr) {
      MutableUnknownFields(message)->AddVarint(field->number(), value);
      return;
    }
  }
  SetEnumValueInternal(message, field, value);
}

void Reflection::SetEnumValueInternal(Message* message,
                                      const FieldDescriptor* field,
                                      int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetEnum(field->number(), field->type(), value,
                                          field);
  } else {
    SetField<int>(message, field, value);
  }
}

const EnumValueDescriptor* Reflection::GetRepeatedEnum(
    const Message& message, const FieldDescriptor* field, int index) const {
  // Usage checked by GetRepeatedEnumValue.
  int value = GetRepeatedEnumValue(message, field, index);
  return field->enum_type()->FindValueByNumberCreatingIfUnknown(value);
}

int Reflection::GetRepeatedEnumValue(const Message& message,
                                     const FieldDescriptor* field,
                                     int index) const {
  USAGE_CHECK_ALL(GetRepeatedEnumValue, REPEATED, ENUM);

  int value;
  if (field->is_extension()) {
    value = GetExtensionSet(message).GetRepeatedEnum(field->number(), index);
  } else {
    value = GetRepeatedField<int>(message, field, index);
  }
  return value;
}

void Reflection::SetRepeatedEnum(Message* message, const FieldDescriptor* field,
                                 int index,
                                 const EnumValueDescriptor* value) const {
  // Usage checked by SetRepeatedEnumValue.
  USAGE_CHECK_ENUM_VALUE(SetRepeatedEnum);
  SetRepeatedEnumValueInternal(message, field, index, value->number());
}

void Reflection::SetRepeatedEnumValue(Message* message,
                                      const FieldDescriptor* field, int index,
                                      int value) const {
  USAGE_CHECK_ALL(SetRepeatedEnum, REPEATED, ENUM);
  if (!CreateUnknownEnumValues(field)) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == nullptr) {
      MutableUnknownFields(message)->AddVarint(field->number(), value);
      return;
    }
  }
  SetRepeatedEnumValueInternal(message, field, index, value);
}

void Reflection::SetRepeatedEnumValueInternal(Message* message,
                                              const FieldDescriptor* field,
                                              int index, int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->SetRepeatedEnum(field->number(), index,
                                                  value);
  } else {
    SetRepeatedField<int>(message, field, index, value);
  }
}

void Reflection::AddEnum(Message* message, const FieldDescriptor* field,
                         const EnumValueDescriptor* value) const {
  // Usage checked by AddEnumValue.
  USAGE_CHECK_ENUM_VALUE(AddEnum);
  AddEnumValueInternal(message, field, value->number());
}

void Reflection::AddEnumValue(Message* message, const FieldDescriptor* field,
                              int value) const {
  USAGE_CHECK_ALL(AddEnum, REPEATED, ENUM);
  if (!CreateUnknownEnumValues(field)) {
    // Check that the value is valid if we don't support direct storage of
    // unknown enum values.
    const EnumValueDescriptor* value_desc =
        field->enum_type()->FindValueByNumber(value);
    if (value_desc == nullptr) {
      MutableUnknownFields(message)->AddVarint(field->number(), value);
      return;
    }
  }
  AddEnumValueInternal(message, field, value);
}

void Reflection::AddEnumValueInternal(Message* message,
                                      const FieldDescriptor* field,
                                      int value) const {
  if (field->is_extension()) {
    MutableExtensionSet(message)->AddEnum(field->number(), field->type(),
                                          field->options().packed(), value,
                                          field);
  } else {
    AddField<int>(message, field, value);
  }
}

// -------------------------------------------------------------------

const Message* Reflection::GetDefaultMessageInstance(
    const FieldDescriptor* field) const {
  // If we are using the generated factory, we cache the prototype in the field
  // descriptor for faster access.
  // The default instances of generated messages are not cross-linked, which
  // means they contain null pointers on their message fields and can't be used
  // to get the default of submessages.
  if (message_factory_ == MessageFactory::generated_factory()) {
    auto& ptr = field->default_generated_instance_;
    auto* res = ptr.load(std::memory_order_acquire);
    if (res == nullptr) {
      // First time asking for this field's default. Load it and cache it.
      res = message_factory_->GetPrototype(field->message_type());
      ptr.store(res, std::memory_order_release);
    }
    return res;
  }

  // For other factories, we try the default's object field.
  // In particular, the DynamicMessageFactory will cross link the default
  // instances to allow for this. But only do this for real fields.
  // This is an optimization to avoid going to GetPrototype() below, as that
  // requires a lock and a map lookup.
  if (!field->is_extension() && !field->options().weak() &&
      !IsLazyField(field) && !schema_.InRealOneof(field)) {
    auto* res = DefaultRaw<const Message*>(field);
    if (res != nullptr) {
      return res;
    }
  }
  // Otherwise, just go to the factory.
  return message_factory_->GetPrototype(field->message_type());
}

const Message& Reflection::GetMessage(const Message& message,
                                      const FieldDescriptor* field,
                                      MessageFactory* factory) const {
  USAGE_CHECK_ALL(GetMessage, SINGULAR, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (factory == nullptr) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<const Message&>(GetExtensionSet(message).GetMessage(
        field->number(), field->message_type(), factory));
  } else {
    if (schema_.InRealOneof(field) && !HasOneofField(message, field)) {
      return *GetDefaultMessageInstance(field);
    }
    const Message* result = GetRaw<const Message*>(message, field);
    if (result == nullptr) {
      result = GetDefaultMessageInstance(field);
    }
    return *result;
  }
}

Message* Reflection::MutableMessage(Message* message,
                                    const FieldDescriptor* field,
                                    MessageFactory* factory) const {
  USAGE_CHECK_ALL(MutableMessage, SINGULAR, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (factory == nullptr) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->MutableMessage(field, factory));
  } else {
    Message* result;

    Message** result_holder = MutableRaw<Message*>(message, field);

    if (schema_.InRealOneof(field)) {
      if (!HasOneofField(*message, field)) {
        ClearOneof(message, field->containing_oneof());
        result_holder = MutableField<Message*>(message, field);
        const Message* default_message = GetDefaultMessageInstance(field);
        *result_holder = default_message->New(message->GetArenaForAllocation());
      }
    } else {
      SetBit(message, field);
    }

    if (*result_holder == nullptr) {
      const Message* default_message = GetDefaultMessageInstance(field);
      *result_holder = default_message->New(message->GetArenaForAllocation());
    }
    result = *result_holder;
    return result;
  }
}

void Reflection::UnsafeArenaSetAllocatedMessage(
    Message* message, Message* sub_message,
    const FieldDescriptor* field) const {
  USAGE_CHECK_ALL(SetAllocatedMessage, SINGULAR, MESSAGE);
  CheckInvalidAccess(schema_, field);


  if (field->is_extension()) {
    MutableExtensionSet(message)->UnsafeArenaSetAllocatedMessage(
        field->number(), field->type(), field, sub_message);
  } else {
    if (schema_.InRealOneof(field)) {
      if (sub_message == nullptr) {
        ClearOneof(message, field->containing_oneof());
        return;
      }
        ClearOneof(message, field->containing_oneof());
        *MutableRaw<Message*>(message, field) = sub_message;
      SetOneofCase(message, field);
      return;
    }

    if (sub_message == nullptr) {
      ClearBit(message, field);
    } else {
      SetBit(message, field);
    }
    Message** sub_message_holder = MutableRaw<Message*>(message, field);
    if (message->GetArenaForAllocation() == nullptr) {
      delete *sub_message_holder;
    }
    *sub_message_holder = sub_message;
  }
}

void Reflection::SetAllocatedMessage(Message* message, Message* sub_message,
                                     const FieldDescriptor* field) const {
  GOOGLE_DCHECK(sub_message == nullptr || sub_message->GetOwningArena() == nullptr ||
         sub_message->GetOwningArena() == message->GetArenaForAllocation());
  CheckInvalidAccess(schema_, field);

  // If message and sub-message are in different memory ownership domains
  // (different arenas, or one is on heap and one is not), then we may need to
  // do a copy.
  if (sub_message != nullptr &&
      sub_message->GetOwningArena() != message->GetArenaForAllocation()) {
    if (sub_message->GetOwningArena() == nullptr &&
        message->GetArenaForAllocation() != nullptr) {
      // Case 1: parent is on an arena and child is heap-allocated. We can add
      // the child to the arena's Own() list to free on arena destruction, then
      // set our pointer.
      message->GetArenaForAllocation()->Own(sub_message);
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

Message* Reflection::UnsafeArenaReleaseMessage(Message* message,
                                               const FieldDescriptor* field,
                                               MessageFactory* factory) const {
  USAGE_CHECK_ALL(ReleaseMessage, SINGULAR, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (factory == nullptr) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->UnsafeArenaReleaseMessage(field,
                                                                factory));
  } else {
    if (!(field->is_repeated() || schema_.InRealOneof(field))) {
      ClearBit(message, field);
    }
    if (schema_.InRealOneof(field)) {
      if (HasOneofField(*message, field)) {
        *MutableOneofCase(message, field->containing_oneof()) = 0;
      } else {
        return nullptr;
      }
    }
    Message** result = MutableRaw<Message*>(message, field);
    Message* ret = *result;
    *result = nullptr;
    return ret;
  }
}

Message* Reflection::ReleaseMessage(Message* message,
                                    const FieldDescriptor* field,
                                    MessageFactory* factory) const {
  CheckInvalidAccess(schema_, field);

  Message* released = UnsafeArenaReleaseMessage(message, field, factory);
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  released = MaybeForceCopy(message->GetArenaForAllocation(), released);
#endif  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (message->GetArenaForAllocation() != nullptr && released != nullptr) {
    Message* copy_from_arena = released->New();
    copy_from_arena->CopyFrom(*released);
    released = copy_from_arena;
  }
  return released;
}

const Message& Reflection::GetRepeatedMessage(const Message& message,
                                              const FieldDescriptor* field,
                                              int index) const {
  USAGE_CHECK_ALL(GetRepeatedMessage, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

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

Message* Reflection::MutableRepeatedMessage(Message* message,
                                            const FieldDescriptor* field,
                                            int index) const {
  USAGE_CHECK_ALL(MutableRepeatedMessage, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->MutableRepeatedMessage(field->number(),
                                                             index));
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

Message* Reflection::AddMessage(Message* message, const FieldDescriptor* field,
                                MessageFactory* factory) const {
  USAGE_CHECK_ALL(AddMessage, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (factory == nullptr) factory = message_factory_;

  if (field->is_extension()) {
    return static_cast<Message*>(
        MutableExtensionSet(message)->AddMessage(field, factory));
  } else {
    Message* result = nullptr;

    // We can't use AddField<Message>() because RepeatedPtrFieldBase doesn't
    // know how to allocate one.
    RepeatedPtrFieldBase* repeated = nullptr;
    if (IsMapFieldInApi(field)) {
      repeated =
          MutableRaw<MapFieldBase>(message, field)->MutableRepeatedField();
    } else {
      repeated = MutableRaw<RepeatedPtrFieldBase>(message, field);
    }
    result = repeated->AddFromCleared<GenericTypeHandler<Message> >();
    if (result == nullptr) {
      // We must allocate a new object.
      const Message* prototype;
      if (repeated->size() == 0) {
        prototype = factory->GetPrototype(field->message_type());
      } else {
        prototype = &repeated->Get<GenericTypeHandler<Message> >(0);
      }
      result = prototype->New(message->GetArenaForAllocation());
      // We can guarantee here that repeated and result are either both heap
      // allocated or arena owned. So it is safe to call the unsafe version
      // of AddAllocated.
      repeated->UnsafeArenaAddAllocated<GenericTypeHandler<Message> >(result);
    }

    return result;
  }
}

void Reflection::AddAllocatedMessage(Message* message,
                                     const FieldDescriptor* field,
                                     Message* new_entry) const {
  USAGE_CHECK_ALL(AddAllocatedMessage, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    MutableExtensionSet(message)->AddAllocatedMessage(field, new_entry);
  } else {
    RepeatedPtrFieldBase* repeated = nullptr;
    if (IsMapFieldInApi(field)) {
      repeated =
          MutableRaw<MapFieldBase>(message, field)->MutableRepeatedField();
    } else {
      repeated = MutableRaw<RepeatedPtrFieldBase>(message, field);
    }
    repeated->AddAllocated<GenericTypeHandler<Message> >(new_entry);
  }
}

void Reflection::UnsafeArenaAddAllocatedMessage(Message* message,
                                                const FieldDescriptor* field,
                                                Message* new_entry) const {
  USAGE_CHECK_ALL(UnsafeArenaAddAllocatedMessage, REPEATED, MESSAGE);
  CheckInvalidAccess(schema_, field);

  if (field->is_extension()) {
    MutableExtensionSet(message)->UnsafeArenaAddAllocatedMessage(field,
                                                                 new_entry);
  } else {
    RepeatedPtrFieldBase* repeated = nullptr;
    if (IsMapFieldInApi(field)) {
      repeated =
          MutableRaw<MapFieldBase>(message, field)->MutableRepeatedField();
    } else {
      repeated = MutableRaw<RepeatedPtrFieldBase>(message, field);
    }
    repeated->UnsafeArenaAddAllocated<GenericTypeHandler<Message>>(new_entry);
  }
}

void* Reflection::MutableRawRepeatedField(Message* message,
                                          const FieldDescriptor* field,
                                          FieldDescriptor::CppType cpptype,
                                          int ctype,
                                          const Descriptor* desc) const {
  (void)ctype;  // Parameter is used by Google-internal code.
  USAGE_CHECK_REPEATED("MutableRawRepeatedField");
  CheckInvalidAccess(schema_, field);

  if (field->cpp_type() != cpptype &&
      (field->cpp_type() != FieldDescriptor::CPPTYPE_ENUM ||
       cpptype != FieldDescriptor::CPPTYPE_INT32))
    ReportReflectionUsageTypeError(descriptor_, field,
                                   "MutableRawRepeatedField", cpptype);
  if (desc != nullptr)
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

const void* Reflection::GetRawRepeatedField(const Message& message,
                                            const FieldDescriptor* field,
                                            FieldDescriptor::CppType cpptype,
                                            int ctype,
                                            const Descriptor* desc) const {
  USAGE_CHECK_REPEATED("GetRawRepeatedField");
  if (field->cpp_type() != cpptype)
    ReportReflectionUsageTypeError(descriptor_, field, "GetRawRepeatedField",
                                   cpptype);
  if (ctype >= 0)
    GOOGLE_CHECK_EQ(field->options().ctype(), ctype) << "subtype mismatch";
  if (desc != nullptr)
    GOOGLE_CHECK_EQ(field->message_type(), desc) << "wrong submessage type";
  if (field->is_extension()) {
    // Should use extension_set::GetRawRepeatedField. However, the required
    // parameter "default repeated value" is not very easy to get here.
    // Map is not supported in extensions, it is acceptable to use
    // extension_set::MutableRawRepeatedField which does not change the message.
    return MutableExtensionSet(const_cast<Message*>(&message))
        ->MutableRawRepeatedField(field->number(), field->type(),
                                  field->is_packed(), field);
  } else {
    // Trigger transform for MapField
    if (IsMapFieldInApi(field)) {
      return &(GetRawNonOneof<MapFieldBase>(message, field).GetRepeatedField());
    }
    return &GetRawNonOneof<char>(message, field);
  }
}

const FieldDescriptor* Reflection::GetOneofFieldDescriptor(
    const Message& message, const OneofDescriptor* oneof_descriptor) const {
  if (oneof_descriptor->is_synthetic()) {
    const FieldDescriptor* field = oneof_descriptor->field(0);
    return HasField(message, field) ? field : nullptr;
  }
  uint32_t field_number = GetOneofCase(message, oneof_descriptor);
  if (field_number == 0) {
    return nullptr;
  }
  return descriptor_->FindFieldByNumber(field_number);
}

bool Reflection::ContainsMapKey(const Message& message,
                                const FieldDescriptor* field,
                                const MapKey& key) const {
  USAGE_CHECK(IsMapFieldInApi(field), "LookupMapValue",
              "Field is not a map field.");
  return GetRaw<MapFieldBase>(message, field).ContainsMapKey(key);
}

bool Reflection::InsertOrLookupMapValue(Message* message,
                                        const FieldDescriptor* field,
                                        const MapKey& key,
                                        MapValueRef* val) const {
  USAGE_CHECK(IsMapFieldInApi(field), "InsertOrLookupMapValue",
              "Field is not a map field.");
  val->SetType(field->message_type()->FindFieldByName("value")->cpp_type());
  return MutableRaw<MapFieldBase>(message, field)
      ->InsertOrLookupMapValue(key, val);
}

bool Reflection::LookupMapValue(const Message& message,
                                const FieldDescriptor* field, const MapKey& key,
                                MapValueConstRef* val) const {
  USAGE_CHECK(IsMapFieldInApi(field), "LookupMapValue",
              "Field is not a map field.");
  val->SetType(field->message_type()->FindFieldByName("value")->cpp_type());
  return GetRaw<MapFieldBase>(message, field).LookupMapValue(key, val);
}

bool Reflection::DeleteMapValue(Message* message, const FieldDescriptor* field,
                                const MapKey& key) const {
  USAGE_CHECK(IsMapFieldInApi(field), "DeleteMapValue",
              "Field is not a map field.");
  return MutableRaw<MapFieldBase>(message, field)->DeleteMapValue(key);
}

MapIterator Reflection::MapBegin(Message* message,
                                 const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field), "MapBegin", "Field is not a map field.");
  MapIterator iter(message, field);
  GetRaw<MapFieldBase>(*message, field).MapBegin(&iter);
  return iter;
}

MapIterator Reflection::MapEnd(Message* message,
                               const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field), "MapEnd", "Field is not a map field.");
  MapIterator iter(message, field);
  GetRaw<MapFieldBase>(*message, field).MapEnd(&iter);
  return iter;
}

int Reflection::MapSize(const Message& message,
                        const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field), "MapSize", "Field is not a map field.");
  return GetRaw<MapFieldBase>(message, field).size();
}

// -----------------------------------------------------------------------------

const FieldDescriptor* Reflection::FindKnownExtensionByName(
    const std::string& name) const {
  if (!schema_.HasExtensionSet()) return nullptr;
  return descriptor_pool_->FindExtensionByPrintableName(descriptor_, name);
}

const FieldDescriptor* Reflection::FindKnownExtensionByNumber(
    int number) const {
  if (!schema_.HasExtensionSet()) return nullptr;
  return descriptor_pool_->FindExtensionByNumber(descriptor_, number);
}

bool Reflection::SupportsUnknownEnumValues() const {
  return CreateUnknownEnumValues(descriptor_->file());
}

// ===================================================================
// Some private helpers.

// These simple template accessors obtain pointers (or references) to
// the given field.

template <class Type>
const Type& Reflection::GetRawNonOneof(const Message& message,
                                       const FieldDescriptor* field) const {
  return GetConstRefAtOffset<Type>(message,
                                   schema_.GetFieldOffsetNonOneof(field));
}

template <class Type>
Type* Reflection::MutableRawNonOneof(Message* message,
                                     const FieldDescriptor* field) const {
  return GetPointerAtOffset<Type>(message,
                                  schema_.GetFieldOffsetNonOneof(field));
}

template <typename Type>
Type* Reflection::MutableRaw(Message* message,
                             const FieldDescriptor* field) const {
  return GetPointerAtOffset<Type>(message, schema_.GetFieldOffset(field));
}

const uint32_t* Reflection::GetHasBits(const Message& message) const {
  GOOGLE_DCHECK(schema_.HasHasbits());
  return &GetConstRefAtOffset<uint32_t>(message, schema_.HasBitsOffset());
}

uint32_t* Reflection::MutableHasBits(Message* message) const {
  GOOGLE_DCHECK(schema_.HasHasbits());
  return GetPointerAtOffset<uint32_t>(message, schema_.HasBitsOffset());
}

uint32_t* Reflection::MutableOneofCase(
    Message* message, const OneofDescriptor* oneof_descriptor) const {
  GOOGLE_DCHECK(!oneof_descriptor->is_synthetic());
  return GetPointerAtOffset<uint32_t>(
      message, schema_.GetOneofCaseOffset(oneof_descriptor));
}

const ExtensionSet& Reflection::GetExtensionSet(const Message& message) const {
  return GetConstRefAtOffset<ExtensionSet>(message,
                                           schema_.GetExtensionSetOffset());
}

ExtensionSet* Reflection::MutableExtensionSet(Message* message) const {
  return GetPointerAtOffset<ExtensionSet>(message,
                                          schema_.GetExtensionSetOffset());
}

const InternalMetadata& Reflection::GetInternalMetadata(
    const Message& message) const {
  return GetConstRefAtOffset<InternalMetadata>(message,
                                               schema_.GetMetadataOffset());
}

InternalMetadata* Reflection::MutableInternalMetadata(Message* message) const {
  return GetPointerAtOffset<InternalMetadata>(message,
                                              schema_.GetMetadataOffset());
}

const uint32_t* Reflection::GetInlinedStringDonatedArray(
    const Message& message) const {
  GOOGLE_DCHECK(schema_.HasInlinedString());
  return &GetConstRefAtOffset<uint32_t>(message,
                                        schema_.InlinedStringDonatedOffset());
}

uint32_t* Reflection::MutableInlinedStringDonatedArray(Message* message) const {
  GOOGLE_DCHECK(schema_.HasHasbits());
  return GetPointerAtOffset<uint32_t>(message,
                                      schema_.InlinedStringDonatedOffset());
}

// Simple accessors for manipulating _inlined_string_donated_;
bool Reflection::IsInlinedStringDonated(const Message& message,
                                        const FieldDescriptor* field) const {
  return IsIndexInHasBitSet(GetInlinedStringDonatedArray(message),
                            schema_.InlinedStringIndex(field));
}

// Simple accessors for manipulating has_bits_.
bool Reflection::HasBit(const Message& message,
                        const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  if (schema_.HasBitIndex(field) != static_cast<uint32_t>(-1)) {
    return IsIndexInHasBitSet(GetHasBits(message), schema_.HasBitIndex(field));
  }

  // Intentionally check here because HasBitIndex(field) != -1 means valid.
  CheckInvalidAccess(schema_, field);

  // proto3: no has-bits. All fields present except messages, which are
  // present only if their message-field pointer is non-null.
  if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
    return !schema_.IsDefaultInstance(message) &&
           GetRaw<const Message*>(message, field) != nullptr;
  } else {
    // Non-message field (and non-oneof, since that was handled in HasField()
    // before calling us), and singular (again, checked in HasField). So, this
    // field must be a scalar.

    // Scalar primitive (numeric or string/bytes) fields are present if
    // their value is non-zero (numeric) or non-empty (string/bytes). N.B.:
    // we must use this definition here, rather than the "scalar fields
    // always present" in the proto3 docs, because MergeFrom() semantics
    // require presence as "present on wire", and reflection-based merge
    // (which uses HasField()) needs to be consistent with this.
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default: {
            if (IsInlined(field)) {
              return !GetField<InlinedStringField>(message, field)
                          .GetNoArena()
                          .empty();
            }

            return GetField<ArenaStringPtr>(message, field).Get().size() > 0;
          }
        }
        return false;
      case FieldDescriptor::CPPTYPE_BOOL:
        return GetRaw<bool>(message, field) != false;
      case FieldDescriptor::CPPTYPE_INT32:
        return GetRaw<int32_t>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_INT64:
        return GetRaw<int64_t>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_UINT32:
        return GetRaw<uint32_t>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_UINT64:
        return GetRaw<uint64_t>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_FLOAT:
        static_assert(sizeof(uint32_t) == sizeof(float),
                      "Code assumes uint32_t and float are the same size.");
        return GetRaw<uint32_t>(message, field) != 0;
      case FieldDescriptor::CPPTYPE_DOUBLE:
        static_assert(sizeof(uint64_t) == sizeof(double),
                      "Code assumes uint64_t and double are the same size.");
        return GetRaw<uint64_t>(message, field) != 0;
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

void Reflection::SetBit(Message* message, const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  const uint32_t index = schema_.HasBitIndex(field);
  if (index == static_cast<uint32_t>(-1)) return;
  MutableHasBits(message)[index / 32] |=
      (static_cast<uint32_t>(1) << (index % 32));
}

void Reflection::ClearBit(Message* message,
                          const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!field->options().weak());
  const uint32_t index = schema_.HasBitIndex(field);
  if (index == static_cast<uint32_t>(-1)) return;
  MutableHasBits(message)[index / 32] &=
      ~(static_cast<uint32_t>(1) << (index % 32));
}

void Reflection::SwapBit(Message* message1, Message* message2,
                         const FieldDescriptor* field) const {
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

bool Reflection::HasOneof(const Message& message,
                          const OneofDescriptor* oneof_descriptor) const {
  if (oneof_descriptor->is_synthetic()) {
    return HasField(message, oneof_descriptor->field(0));
  }
  return (GetOneofCase(message, oneof_descriptor) > 0);
}

void Reflection::SetOneofCase(Message* message,
                              const FieldDescriptor* field) const {
  *MutableOneofCase(message, field->containing_oneof()) = field->number();
}

void Reflection::ClearOneofField(Message* message,
                                 const FieldDescriptor* field) const {
  if (HasOneofField(*message, field)) {
    ClearOneof(message, field->containing_oneof());
  }
}

void Reflection::ClearOneof(Message* message,
                            const OneofDescriptor* oneof_descriptor) const {
  if (oneof_descriptor->is_synthetic()) {
    ClearField(message, oneof_descriptor->field(0));
    return;
  }
  // TODO(jieluo): Consider to cache the unused object instead of deleting
  // it. It will be much faster if an application switches a lot from
  // a few oneof fields.  Time/space tradeoff
  uint32_t oneof_case = GetOneofCase(*message, oneof_descriptor);
  if (oneof_case > 0) {
    const FieldDescriptor* field = descriptor_->FindFieldByNumber(oneof_case);
    if (message->GetArenaForAllocation() == nullptr) {
      switch (field->cpp_type()) {
        case FieldDescriptor::CPPTYPE_STRING: {
          switch (field->options().ctype()) {
            default:  // TODO(kenton):  Support other string reps.
            case FieldOptions::STRING: {
              // Oneof string fields are never set as a default instance.
              // We just need to pass some arbitrary default string to make it
              // work. This allows us to not have the real default accessible
              // from reflection.
              MutableField<ArenaStringPtr>(message, field)
                  ->Destroy(nullptr, message->GetArenaForAllocation());
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

#define HANDLE_TYPE(TYPE, CPPTYPE, CTYPE)                                  \
  template <>                                                              \
  const RepeatedField<TYPE>& Reflection::GetRepeatedFieldInternal<TYPE>(   \
      const Message& message, const FieldDescriptor* field) const {        \
    return *static_cast<RepeatedField<TYPE>*>(MutableRawRepeatedField(     \
        const_cast<Message*>(&message), field, CPPTYPE, CTYPE, nullptr));  \
  }                                                                        \
                                                                           \
  template <>                                                              \
  RepeatedField<TYPE>* Reflection::MutableRepeatedFieldInternal<TYPE>(     \
      Message * message, const FieldDescriptor* field) const {             \
    return static_cast<RepeatedField<TYPE>*>(                              \
        MutableRawRepeatedField(message, field, CPPTYPE, CTYPE, nullptr)); \
  }

HANDLE_TYPE(int32_t, FieldDescriptor::CPPTYPE_INT32, -1);
HANDLE_TYPE(int64_t, FieldDescriptor::CPPTYPE_INT64, -1);
HANDLE_TYPE(uint32_t, FieldDescriptor::CPPTYPE_UINT32, -1);
HANDLE_TYPE(uint64_t, FieldDescriptor::CPPTYPE_UINT64, -1);
HANDLE_TYPE(float, FieldDescriptor::CPPTYPE_FLOAT, -1);
HANDLE_TYPE(double, FieldDescriptor::CPPTYPE_DOUBLE, -1);
HANDLE_TYPE(bool, FieldDescriptor::CPPTYPE_BOOL, -1);


#undef HANDLE_TYPE

void* Reflection::MutableRawRepeatedString(Message* message,
                                           const FieldDescriptor* field,
                                           bool is_string) const {
  (void)is_string;  // Parameter is used by Google-internal code.
  return MutableRawRepeatedField(message, field,
                                 FieldDescriptor::CPPTYPE_STRING,
                                 FieldOptions::STRING, nullptr);
}

// Template implementations of basic accessors.  Inline because each
// template instance is only called from one location.  These are
// used for all types except messages.
template <typename Type>
const Type& Reflection::GetField(const Message& message,
                                 const FieldDescriptor* field) const {
  return GetRaw<Type>(message, field);
}

template <typename Type>
void Reflection::SetField(Message* message, const FieldDescriptor* field,
                          const Type& value) const {
  bool real_oneof = schema_.InRealOneof(field);
  if (real_oneof && !HasOneofField(*message, field)) {
    ClearOneof(message, field->containing_oneof());
  }
  *MutableRaw<Type>(message, field) = value;
  real_oneof ? SetOneofCase(message, field) : SetBit(message, field);
}

template <typename Type>
Type* Reflection::MutableField(Message* message,
                               const FieldDescriptor* field) const {
  schema_.InRealOneof(field) ? SetOneofCase(message, field)
                             : SetBit(message, field);
  return MutableRaw<Type>(message, field);
}

template <typename Type>
const Type& Reflection::GetRepeatedField(const Message& message,
                                         const FieldDescriptor* field,
                                         int index) const {
  return GetRaw<RepeatedField<Type> >(message, field).Get(index);
}

template <typename Type>
const Type& Reflection::GetRepeatedPtrField(const Message& message,
                                            const FieldDescriptor* field,
                                            int index) const {
  return GetRaw<RepeatedPtrField<Type> >(message, field).Get(index);
}

template <typename Type>
void Reflection::SetRepeatedField(Message* message,
                                  const FieldDescriptor* field, int index,
                                  Type value) const {
  MutableRaw<RepeatedField<Type> >(message, field)->Set(index, value);
}

template <typename Type>
Type* Reflection::MutableRepeatedField(Message* message,
                                       const FieldDescriptor* field,
                                       int index) const {
  RepeatedPtrField<Type>* repeated =
      MutableRaw<RepeatedPtrField<Type> >(message, field);
  return repeated->Mutable(index);
}

template <typename Type>
void Reflection::AddField(Message* message, const FieldDescriptor* field,
                          const Type& value) const {
  MutableRaw<RepeatedField<Type> >(message, field)->Add(value);
}

template <typename Type>
Type* Reflection::AddField(Message* message,
                           const FieldDescriptor* field) const {
  RepeatedPtrField<Type>* repeated =
      MutableRaw<RepeatedPtrField<Type> >(message, field);
  return repeated->Add();
}

MessageFactory* Reflection::GetMessageFactory() const {
  return message_factory_;
}

void* Reflection::RepeatedFieldData(Message* message,
                                    const FieldDescriptor* field,
                                    FieldDescriptor::CppType cpp_type,
                                    const Descriptor* message_type) const {
  GOOGLE_CHECK(field->is_repeated());
  GOOGLE_CHECK(field->cpp_type() == cpp_type ||
        (field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM &&
         cpp_type == FieldDescriptor::CPPTYPE_INT32))
      << "The type parameter T in RepeatedFieldRef<T> API doesn't match "
      << "the actual field type (for enums T should be the generated enum "
      << "type or int32_t).";
  if (message_type != nullptr) {
    GOOGLE_CHECK_EQ(message_type, field->message_type());
  }
  if (field->is_extension()) {
    return MutableExtensionSet(message)->MutableRawRepeatedField(
        field->number(), field->type(), field->is_packed(), field);
  } else {
    return MutableRawNonOneof<char>(message, field);
  }
}

MapFieldBase* Reflection::MutableMapData(Message* message,
                                         const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field), "GetMapData",
              "Field is not a map field.");
  return MutableRaw<MapFieldBase>(message, field);
}

const MapFieldBase* Reflection::GetMapData(const Message& message,
                                           const FieldDescriptor* field) const {
  USAGE_CHECK(IsMapFieldInApi(field), "GetMapData",
              "Field is not a map field.");
  return &(GetRaw<MapFieldBase>(message, field));
}

namespace {

// Helper function to transform migration schema into reflection schema.
ReflectionSchema MigrationToReflectionSchema(
    const Message* const* default_instance, const uint32_t* offsets,
    MigrationSchema migration_schema) {
  ReflectionSchema result;
  result.default_instance_ = *default_instance;
  // First 7 offsets are offsets to the special fields. The following offsets
  // are the proto fields.
  result.offsets_ = offsets + migration_schema.offsets_index + 6;
  result.has_bit_indices_ = offsets + migration_schema.has_bit_indices_index;
  result.has_bits_offset_ = offsets[migration_schema.offsets_index + 0];
  result.metadata_offset_ = offsets[migration_schema.offsets_index + 1];
  result.extensions_offset_ = offsets[migration_schema.offsets_index + 2];
  result.oneof_case_offset_ = offsets[migration_schema.offsets_index + 3];
  result.object_size_ = migration_schema.object_size;
  result.weak_field_map_offset_ = offsets[migration_schema.offsets_index + 4];
  result.inlined_string_donated_offset_ =
      offsets[migration_schema.offsets_index + 5];
  result.inlined_string_indices_ =
      offsets + migration_schema.inlined_string_indices_index;
  return result;
}

}  // namespace

class AssignDescriptorsHelper {
 public:
  AssignDescriptorsHelper(MessageFactory* factory,
                          Metadata* file_level_metadata,
                          const EnumDescriptor** file_level_enum_descriptors,
                          const MigrationSchema* schemas,
                          const Message* const* default_instance_data,
                          const uint32_t* offsets)
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

    file_level_metadata_->reflection =
        new Reflection(descriptor,
                       MigrationToReflectionSchema(default_instance_data_,
                                                   offsets_, *schemas_),
                       DescriptorPool::internal_generated_pool(), factory_);
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
  const MigrationSchema* schemas_;
  const Message* const* default_instance_data_;
  const uint32_t* offsets_;
};

namespace {

// We have the routines that assign descriptors and build reflection
// automatically delete the allocated reflection. MetadataOwner owns
// all the allocated reflection instances.
struct MetadataOwner {
  ~MetadataOwner() {
    for (auto range : metadata_arrays_) {
      for (const Metadata* m = range.first; m < range.second; m++) {
        delete m->reflection;
      }
    }
  }

  void AddArray(const Metadata* begin, const Metadata* end) {
    mu_.Lock();
    metadata_arrays_.push_back(std::make_pair(begin, end));
    mu_.Unlock();
  }

  static MetadataOwner* Instance() {
    static MetadataOwner* res = OnShutdownDelete(new MetadataOwner);
    return res;
  }

 private:
  MetadataOwner() = default;  // private because singleton

  WrappedMutex mu_;
  std::vector<std::pair<const Metadata*, const Metadata*> > metadata_arrays_;
};

void AddDescriptors(const DescriptorTable* table);

void AssignDescriptorsImpl(const DescriptorTable* table, bool eager) {
  // Ensure the file descriptor is added to the pool.
  {
    // This only happens once per proto file. So a global mutex to serialize
    // calls to AddDescriptors.
    static WrappedMutex mu{GOOGLE_PROTOBUF_LINKER_INITIALIZED};
    mu.Lock();
    AddDescriptors(table);
    mu.Unlock();
  }
  if (eager) {
    // Normally we do not want to eagerly build descriptors of our deps.
    // However if this proto is optimized for code size (ie using reflection)
    // and it has a message extending a custom option of a descriptor with that
    // message being optimized for code size as well. Building the descriptors
    // in this file requires parsing the serialized file descriptor, which now
    // requires parsing the message extension, which potentially requires
    // building the descriptor of the message extending one of the options.
    // However we are already updating descriptor pool under a lock. To prevent
    // this the compiler statically looks for this case and we just make sure we
    // first build the descriptors of all our dependencies, preventing the
    // deadlock.
    int num_deps = table->num_deps;
    for (int i = 0; i < num_deps; i++) {
      // In case of weak fields deps[i] could be null.
      if (table->deps[i]) AssignDescriptors(table->deps[i], true);
    }
  }

  // Fill the arrays with pointers to descriptors and reflection classes.
  const FileDescriptor* file =
      DescriptorPool::internal_generated_pool()->FindFileByName(
          table->filename);
  GOOGLE_CHECK(file != nullptr);

  MessageFactory* factory = MessageFactory::generated_factory();

  AssignDescriptorsHelper helper(
      factory, table->file_level_metadata, table->file_level_enum_descriptors,
      table->schemas, table->default_instances, table->offsets);

  for (int i = 0; i < file->message_type_count(); i++) {
    helper.AssignMessageDescriptor(file->message_type(i));
  }

  for (int i = 0; i < file->enum_type_count(); i++) {
    helper.AssignEnumDescriptor(file->enum_type(i));
  }
  if (file->options().cc_generic_services()) {
    for (int i = 0; i < file->service_count(); i++) {
      table->file_level_service_descriptors[i] = file->service(i);
    }
  }
  MetadataOwner::Instance()->AddArray(table->file_level_metadata,
                                      helper.GetCurrentMetadataPtr());
}

void AddDescriptorsImpl(const DescriptorTable* table) {
  // Reflection refers to the default fields so make sure they are initialized.
  internal::InitProtobufDefaults();

  // Ensure all dependent descriptors are registered to the generated descriptor
  // pool and message factory.
  int num_deps = table->num_deps;
  for (int i = 0; i < num_deps; i++) {
    // In case of weak fields deps[i] could be null.
    if (table->deps[i]) AddDescriptors(table->deps[i]);
  }

  // Register the descriptor of this file.
  DescriptorPool::InternalAddGeneratedFile(table->descriptor, table->size);
  MessageFactory::InternalRegisterGeneratedFile(table);
}

void AddDescriptors(const DescriptorTable* table) {
  // AddDescriptors is not thread safe. Callers need to ensure calls are
  // properly serialized. This function is only called pre-main by global
  // descriptors and we can assume single threaded access or it's called
  // by AssignDescriptorImpl which uses a mutex to sequence calls.
  if (table->is_initialized) return;
  table->is_initialized = true;
  AddDescriptorsImpl(table);
}

}  // namespace

// Separate function because it needs to be a friend of
// Reflection
void RegisterAllTypesInternal(const Metadata* file_level_metadata, int size) {
  for (int i = 0; i < size; i++) {
    const Reflection* reflection = file_level_metadata[i].reflection;
    MessageFactory::InternalRegisterGeneratedMessage(
        file_level_metadata[i].descriptor,
        reflection->schema_.default_instance_);
  }
}

namespace internal {

Metadata AssignDescriptors(const DescriptorTable* (*table)(),
                           internal::once_flag* once,
                           const Metadata& metadata) {
  call_once(*once, [=] {
    auto* t = table();
    AssignDescriptorsImpl(t, t->is_eager);
  });

  return metadata;
}

void AssignDescriptors(const DescriptorTable* table, bool eager) {
  if (!eager) eager = table->is_eager;
  call_once(*table->once, AssignDescriptorsImpl, table, eager);
}

AddDescriptorsRunner::AddDescriptorsRunner(const DescriptorTable* table) {
  AddDescriptors(table);
}

void RegisterFileLevelMetadata(const DescriptorTable* table) {
  AssignDescriptors(table);
  RegisterAllTypesInternal(table->file_level_metadata, table->num_messages);
}

void UnknownFieldSetSerializer(const uint8_t* base, uint32_t offset,
                               uint32_t /*tag*/, uint32_t /*has_offset*/,
                               io::CodedOutputStream* output) {
  const void* ptr = base + offset;
  const InternalMetadata* metadata = static_cast<const InternalMetadata*>(ptr);
  if (metadata->have_unknown_fields()) {
    internal::WireFormat::SerializeUnknownFields(
        metadata->unknown_fields<UnknownFieldSet>(
            UnknownFieldSet::default_instance),
        output);
  }
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
