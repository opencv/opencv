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
//
// Contains methods defined in extension_set.h which cannot be part of the
// lite library because they use descriptors or reflection.

#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/extension_set_inl.h>
#include <google/protobuf/parse_context.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/message.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/wire_format_lite.h>


#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace internal {

// A FieldSkipper used to store unknown MessageSet fields into UnknownFieldSet.
class MessageSetFieldSkipper : public UnknownFieldSetFieldSkipper {
 public:
  explicit MessageSetFieldSkipper(UnknownFieldSet* unknown_fields)
      : UnknownFieldSetFieldSkipper(unknown_fields) {}
  ~MessageSetFieldSkipper() override {}

  virtual bool SkipMessageSetField(io::CodedInputStream* input,
                                   int field_number);
};
bool MessageSetFieldSkipper::SkipMessageSetField(io::CodedInputStream* input,
                                                 int field_number) {
  uint32_t length;
  if (!input->ReadVarint32(&length)) return false;
  if (unknown_fields_ == nullptr) {
    return input->Skip(length);
  } else {
    return input->ReadString(unknown_fields_->AddLengthDelimited(field_number),
                             length);
  }
}


// Implementation of ExtensionFinder which finds extensions in a given
// DescriptorPool, using the given MessageFactory to construct sub-objects.
// This class is implemented in extension_set_heavy.cc.
class DescriptorPoolExtensionFinder : public ExtensionFinder {
 public:
  DescriptorPoolExtensionFinder(const DescriptorPool* pool,
                                MessageFactory* factory,
                                const Descriptor* containing_type)
      : pool_(pool), factory_(factory), containing_type_(containing_type) {}
  ~DescriptorPoolExtensionFinder() override {}

  bool Find(int number, ExtensionInfo* output) override;

 private:
  const DescriptorPool* pool_;
  MessageFactory* factory_;
  const Descriptor* containing_type_;
};

void ExtensionSet::AppendToList(
    const Descriptor* containing_type, const DescriptorPool* pool,
    std::vector<const FieldDescriptor*>* output) const {
  ForEach([containing_type, pool, &output](int number, const Extension& ext) {
    bool has = false;
    if (ext.is_repeated) {
      has = ext.GetSize() > 0;
    } else {
      has = !ext.is_cleared;
    }

    if (has) {
      // TODO(kenton): Looking up each field by number is somewhat unfortunate.
      //   Is there a better way?  The problem is that descriptors are lazily-
      //   initialized, so they might not even be constructed until
      //   AppendToList() is called.

      if (ext.descriptor == nullptr) {
        output->push_back(pool->FindExtensionByNumber(containing_type, number));
      } else {
        output->push_back(ext.descriptor);
      }
    }
  });
}

inline FieldDescriptor::Type real_type(FieldType type) {
  GOOGLE_DCHECK(type > 0 && type <= FieldDescriptor::MAX_TYPE);
  return static_cast<FieldDescriptor::Type>(type);
}

inline FieldDescriptor::CppType cpp_type(FieldType type) {
  return FieldDescriptor::TypeToCppType(
      static_cast<FieldDescriptor::Type>(type));
}

inline WireFormatLite::FieldType field_type(FieldType type) {
  GOOGLE_DCHECK(type > 0 && type <= WireFormatLite::MAX_FIELD_TYPE);
  return static_cast<WireFormatLite::FieldType>(type);
}

#define GOOGLE_DCHECK_TYPE(EXTENSION, LABEL, CPPTYPE)                         \
  GOOGLE_DCHECK_EQ((EXTENSION).is_repeated ? FieldDescriptor::LABEL_REPEATED  \
                                    : FieldDescriptor::LABEL_OPTIONAL, \
            FieldDescriptor::LABEL_##LABEL);                           \
  GOOGLE_DCHECK_EQ(cpp_type((EXTENSION).type), FieldDescriptor::CPPTYPE_##CPPTYPE)

const MessageLite& ExtensionSet::GetMessage(int number,
                                            const Descriptor* message_type,
                                            MessageFactory* factory) const {
  const Extension* extension = FindOrNull(number);
  if (extension == nullptr || extension->is_cleared) {
    // Not present.  Return the default value.
    return *factory->GetPrototype(message_type);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL, MESSAGE);
    if (extension->is_lazy) {
      return extension->lazymessage_value->GetMessage(
          *factory->GetPrototype(message_type), arena_);
    } else {
      return *extension->message_value;
    }
  }
}

MessageLite* ExtensionSet::MutableMessage(const FieldDescriptor* descriptor,
                                          MessageFactory* factory) {
  Extension* extension;
  if (MaybeNewExtension(descriptor->number(), descriptor, &extension)) {
    extension->type = descriptor->type();
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), FieldDescriptor::CPPTYPE_MESSAGE);
    extension->is_repeated = false;
    extension->is_packed = false;
    const MessageLite* prototype =
        factory->GetPrototype(descriptor->message_type());
    extension->is_lazy = false;
    extension->message_value = prototype->New(arena_);
    extension->is_cleared = false;
    return extension->message_value;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL, MESSAGE);
    extension->is_cleared = false;
    if (extension->is_lazy) {
      return extension->lazymessage_value->MutableMessage(
          *factory->GetPrototype(descriptor->message_type()), arena_);
    } else {
      return extension->message_value;
    }
  }
}

MessageLite* ExtensionSet::ReleaseMessage(const FieldDescriptor* descriptor,
                                          MessageFactory* factory) {
  Extension* extension = FindOrNull(descriptor->number());
  if (extension == nullptr) {
    // Not present.  Return nullptr.
    return nullptr;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL, MESSAGE);
    MessageLite* ret = nullptr;
    if (extension->is_lazy) {
      ret = extension->lazymessage_value->ReleaseMessage(
          *factory->GetPrototype(descriptor->message_type()), arena_);
      if (arena_ == nullptr) {
        delete extension->lazymessage_value;
      }
    } else {
      if (arena_ != nullptr) {
        ret = extension->message_value->New();
        ret->CheckTypeAndMergeFrom(*extension->message_value);
      } else {
        ret = extension->message_value;
      }
    }
    Erase(descriptor->number());
    return ret;
  }
}

MessageLite* ExtensionSet::UnsafeArenaReleaseMessage(
    const FieldDescriptor* descriptor, MessageFactory* factory) {
  Extension* extension = FindOrNull(descriptor->number());
  if (extension == nullptr) {
    // Not present.  Return nullptr.
    return nullptr;
  } else {
    GOOGLE_DCHECK_TYPE(*extension, OPTIONAL, MESSAGE);
    MessageLite* ret = nullptr;
    if (extension->is_lazy) {
      ret = extension->lazymessage_value->UnsafeArenaReleaseMessage(
          *factory->GetPrototype(descriptor->message_type()), arena_);
      if (arena_ == nullptr) {
        delete extension->lazymessage_value;
      }
    } else {
      ret = extension->message_value;
    }
    Erase(descriptor->number());
    return ret;
  }
}

ExtensionSet::Extension* ExtensionSet::MaybeNewRepeatedExtension(
    const FieldDescriptor* descriptor) {
  Extension* extension;
  if (MaybeNewExtension(descriptor->number(), descriptor, &extension)) {
    extension->type = descriptor->type();
    GOOGLE_DCHECK_EQ(cpp_type(extension->type), FieldDescriptor::CPPTYPE_MESSAGE);
    extension->is_repeated = true;
    extension->repeated_message_value =
        Arena::CreateMessage<RepeatedPtrField<MessageLite> >(arena_);
  } else {
    GOOGLE_DCHECK_TYPE(*extension, REPEATED, MESSAGE);
  }
  return extension;
}

MessageLite* ExtensionSet::AddMessage(const FieldDescriptor* descriptor,
                                      MessageFactory* factory) {
  Extension* extension = MaybeNewRepeatedExtension(descriptor);

  // RepeatedPtrField<Message> does not know how to Add() since it cannot
  // allocate an abstract object, so we have to be tricky.
  MessageLite* result =
      reinterpret_cast<internal::RepeatedPtrFieldBase*>(
          extension->repeated_message_value)
          ->AddFromCleared<GenericTypeHandler<MessageLite> >();
  if (result == nullptr) {
    const MessageLite* prototype;
    if (extension->repeated_message_value->empty()) {
      prototype = factory->GetPrototype(descriptor->message_type());
      GOOGLE_CHECK(prototype != nullptr);
    } else {
      prototype = &extension->repeated_message_value->Get(0);
    }
    result = prototype->New(arena_);
    extension->repeated_message_value->AddAllocated(result);
  }
  return result;
}

void ExtensionSet::AddAllocatedMessage(const FieldDescriptor* descriptor,
                                       MessageLite* new_entry) {
  Extension* extension = MaybeNewRepeatedExtension(descriptor);

  extension->repeated_message_value->AddAllocated(new_entry);
}

void ExtensionSet::UnsafeArenaAddAllocatedMessage(
    const FieldDescriptor* descriptor, MessageLite* new_entry) {
  Extension* extension = MaybeNewRepeatedExtension(descriptor);

  extension->repeated_message_value->UnsafeArenaAddAllocated(new_entry);
}

static bool ValidateEnumUsingDescriptor(const void* arg, int number) {
  return reinterpret_cast<const EnumDescriptor*>(arg)->FindValueByNumber(
             number) != nullptr;
}

bool DescriptorPoolExtensionFinder::Find(int number, ExtensionInfo* output) {
  const FieldDescriptor* extension =
      pool_->FindExtensionByNumber(containing_type_, number);
  if (extension == nullptr) {
    return false;
  } else {
    output->type = extension->type();
    output->is_repeated = extension->is_repeated();
    output->is_packed = extension->options().packed();
    output->descriptor = extension;
    if (extension->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      output->message_info.prototype =
          factory_->GetPrototype(extension->message_type());
      GOOGLE_CHECK(output->message_info.prototype != nullptr)
          << "Extension factory's GetPrototype() returned nullptr; extension: "
          << extension->full_name();
    } else if (extension->cpp_type() == FieldDescriptor::CPPTYPE_ENUM) {
      output->enum_validity_check.func = ValidateEnumUsingDescriptor;
      output->enum_validity_check.arg = extension->enum_type();
    }

    return true;
  }
}


bool ExtensionSet::FindExtension(int wire_type, uint32_t field,
                                 const Message* containing_type,
                                 const internal::ParseContext* ctx,
                                 ExtensionInfo* extension,
                                 bool* was_packed_on_wire) {
  if (ctx->data().pool == nullptr) {
    GeneratedExtensionFinder finder(containing_type);
    if (!FindExtensionInfoFromFieldNumber(wire_type, field, &finder, extension,
                                          was_packed_on_wire)) {
      return false;
    }
  } else {
    DescriptorPoolExtensionFinder finder(ctx->data().pool, ctx->data().factory,
                                         containing_type->GetDescriptor());
    if (!FindExtensionInfoFromFieldNumber(wire_type, field, &finder, extension,
                                          was_packed_on_wire)) {
      return false;
    }
  }
  return true;
}

const char* ExtensionSet::ParseField(uint64_t tag, const char* ptr,
                                     const Message* containing_type,
                                     internal::InternalMetadata* metadata,
                                     internal::ParseContext* ctx) {
  int number = tag >> 3;
  bool was_packed_on_wire;
  ExtensionInfo extension;
  if (!FindExtension(tag & 7, number, containing_type, ctx, &extension,
                     &was_packed_on_wire)) {
    return UnknownFieldParse(
        tag, metadata->mutable_unknown_fields<UnknownFieldSet>(), ptr, ctx);
  }
  return ParseFieldWithExtensionInfo<UnknownFieldSet>(
      number, was_packed_on_wire, extension, metadata, ptr, ctx);
}

const char* ExtensionSet::ParseFieldMaybeLazily(
    uint64_t tag, const char* ptr, const Message* containing_type,
    internal::InternalMetadata* metadata, internal::ParseContext* ctx) {
  return ParseField(tag, ptr, containing_type, metadata, ctx);
}

const char* ExtensionSet::ParseMessageSetItem(
    const char* ptr, const Message* containing_type,
    internal::InternalMetadata* metadata, internal::ParseContext* ctx) {
  return ParseMessageSetItemTmpl<Message, UnknownFieldSet>(ptr, containing_type,
                                                           metadata, ctx);
}

bool ExtensionSet::ParseField(uint32_t tag, io::CodedInputStream* input,
                              const Message* containing_type,
                              UnknownFieldSet* unknown_fields) {
  UnknownFieldSetFieldSkipper skipper(unknown_fields);
  if (input->GetExtensionPool() == nullptr) {
    GeneratedExtensionFinder finder(containing_type);
    return ParseField(tag, input, &finder, &skipper);
  } else {
    DescriptorPoolExtensionFinder finder(input->GetExtensionPool(),
                                         input->GetExtensionFactory(),
                                         containing_type->GetDescriptor());
    return ParseField(tag, input, &finder, &skipper);
  }
}

bool ExtensionSet::ParseMessageSet(io::CodedInputStream* input,
                                   const Message* containing_type,
                                   UnknownFieldSet* unknown_fields) {
  MessageSetFieldSkipper skipper(unknown_fields);
  if (input->GetExtensionPool() == nullptr) {
    GeneratedExtensionFinder finder(containing_type);
    return ParseMessageSet(input, &finder, &skipper);
  } else {
    DescriptorPoolExtensionFinder finder(input->GetExtensionPool(),
                                         input->GetExtensionFactory(),
                                         containing_type->GetDescriptor());
    return ParseMessageSet(input, &finder, &skipper);
  }
}

int ExtensionSet::SpaceUsedExcludingSelf() const {
  return internal::FromIntSize(SpaceUsedExcludingSelfLong());
}

size_t ExtensionSet::SpaceUsedExcludingSelfLong() const {
  size_t total_size = Size() * sizeof(KeyValue);
  ForEach([&total_size](int /* number */, const Extension& ext) {
    total_size += ext.SpaceUsedExcludingSelfLong();
  });
  return total_size;
}

inline size_t ExtensionSet::RepeatedMessage_SpaceUsedExcludingSelfLong(
    RepeatedPtrFieldBase* field) {
  return field->SpaceUsedExcludingSelfLong<GenericTypeHandler<Message> >();
}

size_t ExtensionSet::Extension::SpaceUsedExcludingSelfLong() const {
  size_t total_size = 0;
  if (is_repeated) {
    switch (cpp_type(type)) {
#define HANDLE_TYPE(UPPERCASE, LOWERCASE)                                     \
  case FieldDescriptor::CPPTYPE_##UPPERCASE:                                  \
    total_size += sizeof(*repeated_##LOWERCASE##_value) +                     \
                  repeated_##LOWERCASE##_value->SpaceUsedExcludingSelfLong(); \
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
#undef HANDLE_TYPE

      case FieldDescriptor::CPPTYPE_MESSAGE:
        // repeated_message_value is actually a RepeatedPtrField<MessageLite>,
        // but MessageLite has no SpaceUsedLong(), so we must directly call
        // RepeatedPtrFieldBase::SpaceUsedExcludingSelfLong() with a different
        // type handler.
        total_size += sizeof(*repeated_message_value) +
                      RepeatedMessage_SpaceUsedExcludingSelfLong(
                          reinterpret_cast<internal::RepeatedPtrFieldBase*>(
                              repeated_message_value));
        break;
    }
  } else {
    switch (cpp_type(type)) {
      case FieldDescriptor::CPPTYPE_STRING:
        total_size += sizeof(*string_value) +
                      StringSpaceUsedExcludingSelfLong(*string_value);
        break;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (is_lazy) {
          total_size += lazymessage_value->SpaceUsedLong();
        } else {
          total_size += down_cast<Message*>(message_value)->SpaceUsedLong();
        }
        break;
      default:
        // No extra storage costs for primitive types.
        break;
    }
  }
  return total_size;
}

uint8_t* ExtensionSet::SerializeMessageSetWithCachedSizesToArray(
    const MessageLite* extendee, uint8_t* target) const {
  io::EpsCopyOutputStream stream(
      target, MessageSetByteSize(),
      io::CodedOutputStream::IsDefaultSerializationDeterministic());
  return InternalSerializeMessageSetWithCachedSizesToArray(extendee, target,
                                                           &stream);
}

bool ExtensionSet::ParseFieldMaybeLazily(
    int wire_type, int field_number, io::CodedInputStream* input,
    ExtensionFinder* extension_finder, MessageSetFieldSkipper* field_skipper) {
  return ParseField(
      WireFormatLite::MakeTag(field_number,
                              static_cast<WireFormatLite::WireType>(wire_type)),
      input, extension_finder, field_skipper);
}

bool ExtensionSet::ParseMessageSet(io::CodedInputStream* input,
                                   ExtensionFinder* extension_finder,
                                   MessageSetFieldSkipper* field_skipper) {
  while (true) {
    const uint32_t tag = input->ReadTag();
    switch (tag) {
      case 0:
        return true;
      case WireFormatLite::kMessageSetItemStartTag:
        if (!ParseMessageSetItem(input, extension_finder, field_skipper)) {
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

bool ExtensionSet::ParseMessageSetItem(io::CodedInputStream* input,
                                       ExtensionFinder* extension_finder,
                                       MessageSetFieldSkipper* field_skipper) {
  struct MSFull {
    bool ParseField(int type_id, io::CodedInputStream* input) {
      return me->ParseFieldMaybeLazily(
          WireFormatLite::WIRETYPE_LENGTH_DELIMITED, type_id, input,
          extension_finder, field_skipper);
    }

    bool SkipField(uint32_t tag, io::CodedInputStream* input) {
      return field_skipper->SkipField(input, tag);
    }

    ExtensionSet* me;
    ExtensionFinder* extension_finder;
    MessageSetFieldSkipper* field_skipper;
  };

  return ParseMessageSetItemImpl(input,
                                 MSFull{this, extension_finder, field_skipper});
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
