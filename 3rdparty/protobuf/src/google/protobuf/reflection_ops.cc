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
#include <google/protobuf/reflection_ops.h>

#include <string>
#include <vector>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/map_field.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>

#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace internal {

static const Reflection* GetReflectionOrDie(const Message& m) {
  const Reflection* r = m.GetReflection();
  if (r == nullptr) {
    const Descriptor* d = m.GetDescriptor();
    const std::string& mtype = d ? d->name() : "unknown";
    // RawMessage is one known type for which GetReflection() returns nullptr.
    GOOGLE_LOG(FATAL) << "Message does not support reflection (type " << mtype << ").";
  }
  return r;
}

void ReflectionOps::Copy(const Message& from, Message* to) {
  if (&from == to) return;
  Clear(to);
  Merge(from, to);
}

void ReflectionOps::Merge(const Message& from, Message* to) {
  GOOGLE_CHECK_NE(&from, to);

  const Descriptor* descriptor = from.GetDescriptor();
  GOOGLE_CHECK_EQ(to->GetDescriptor(), descriptor)
      << "Tried to merge messages of different types "
      << "(merge " << descriptor->full_name() << " to "
      << to->GetDescriptor()->full_name() << ")";

  const Reflection* from_reflection = GetReflectionOrDie(from);
  const Reflection* to_reflection = GetReflectionOrDie(*to);
  bool is_from_generated = (from_reflection->GetMessageFactory() ==
                            google::protobuf::MessageFactory::generated_factory());
  bool is_to_generated = (to_reflection->GetMessageFactory() ==
                          google::protobuf::MessageFactory::generated_factory());

  std::vector<const FieldDescriptor*> fields;
  from_reflection->ListFieldsOmitStripped(from, &fields);
  for (const FieldDescriptor* field : fields) {
    if (field->is_repeated()) {
      // Use map reflection if both are in map status and have the
      // same map type to avoid sync with repeated field.
      // Note: As from and to messages have the same descriptor, the
      // map field types are the same if they are both generated
      // messages or both dynamic messages.
      if (is_from_generated == is_to_generated && field->is_map()) {
        const MapFieldBase* from_field =
            from_reflection->GetMapData(from, field);
        MapFieldBase* to_field = to_reflection->MutableMapData(to, field);
        if (to_field->IsMapValid() && from_field->IsMapValid()) {
          to_field->MergeFrom(*from_field);
          continue;
        }
      }
      int count = from_reflection->FieldSize(from, field);
      for (int j = 0; j < count; j++) {
        switch (field->cpp_type()) {
#define HANDLE_TYPE(CPPTYPE, METHOD)                                      \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                                \
    to_reflection->Add##METHOD(                                           \
        to, field, from_reflection->GetRepeated##METHOD(from, field, j)); \
    break;

          HANDLE_TYPE(INT32, Int32);
          HANDLE_TYPE(INT64, Int64);
          HANDLE_TYPE(UINT32, UInt32);
          HANDLE_TYPE(UINT64, UInt64);
          HANDLE_TYPE(FLOAT, Float);
          HANDLE_TYPE(DOUBLE, Double);
          HANDLE_TYPE(BOOL, Bool);
          HANDLE_TYPE(STRING, String);
          HANDLE_TYPE(ENUM, Enum);
#undef HANDLE_TYPE

          case FieldDescriptor::CPPTYPE_MESSAGE:
            const Message& from_child =
                from_reflection->GetRepeatedMessage(from, field, j);
            if (from_reflection == to_reflection) {
              to_reflection
                  ->AddMessage(to, field,
                               from_child.GetReflection()->GetMessageFactory())
                  ->MergeFrom(from_child);
            } else {
              to_reflection->AddMessage(to, field)->MergeFrom(from_child);
            }
            break;
        }
      }
    } else {
      switch (field->cpp_type()) {
#define HANDLE_TYPE(CPPTYPE, METHOD)                                       \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                                 \
    to_reflection->Set##METHOD(to, field,                                  \
                               from_reflection->Get##METHOD(from, field)); \
    break;

        HANDLE_TYPE(INT32, Int32);
        HANDLE_TYPE(INT64, Int64);
        HANDLE_TYPE(UINT32, UInt32);
        HANDLE_TYPE(UINT64, UInt64);
        HANDLE_TYPE(FLOAT, Float);
        HANDLE_TYPE(DOUBLE, Double);
        HANDLE_TYPE(BOOL, Bool);
        HANDLE_TYPE(STRING, String);
        HANDLE_TYPE(ENUM, Enum);
#undef HANDLE_TYPE

        case FieldDescriptor::CPPTYPE_MESSAGE:
          const Message& from_child = from_reflection->GetMessage(from, field);
          if (from_reflection == to_reflection) {
            to_reflection
                ->MutableMessage(
                    to, field, from_child.GetReflection()->GetMessageFactory())
                ->MergeFrom(from_child);
          } else {
            to_reflection->MutableMessage(to, field)->MergeFrom(from_child);
          }
          break;
      }
    }
  }

  to_reflection->MutableUnknownFields(to)->MergeFrom(
      from_reflection->GetUnknownFields(from));
}

void ReflectionOps::Clear(Message* message) {
  const Reflection* reflection = GetReflectionOrDie(*message);

  std::vector<const FieldDescriptor*> fields;
  reflection->ListFieldsOmitStripped(*message, &fields);
  for (const FieldDescriptor* field : fields) {
    reflection->ClearField(message, field);
  }

  reflection->MutableUnknownFields(message)->Clear();
}

bool ReflectionOps::IsInitialized(const Message& message, bool check_fields,
                                  bool check_descendants) {
  const Descriptor* descriptor = message.GetDescriptor();
  const Reflection* reflection = GetReflectionOrDie(message);
  if (const int field_count = descriptor->field_count()) {
    const FieldDescriptor* begin = descriptor->field(0);
    const FieldDescriptor* end = begin + field_count;
    GOOGLE_DCHECK_EQ(descriptor->field(field_count - 1), end - 1);

    if (check_fields) {
      // Check required fields of this message.
      for (const FieldDescriptor* field = begin; field != end; ++field) {
        if (field->is_required() && !reflection->HasField(message, field)) {
          return false;
        }
      }
    }

    if (check_descendants) {
      for (const FieldDescriptor* field = begin; field != end; ++field) {
        if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
          const Descriptor* message_type = field->message_type();
          if (PROTOBUF_PREDICT_FALSE(message_type->options().map_entry())) {
            if (message_type->field(1)->cpp_type() ==
                FieldDescriptor::CPPTYPE_MESSAGE) {
              const MapFieldBase* map_field =
                  reflection->GetMapData(message, field);
              if (map_field->IsMapValid()) {
                MapIterator it(const_cast<Message*>(&message), field);
                MapIterator end_map(const_cast<Message*>(&message), field);
                for (map_field->MapBegin(&it), map_field->MapEnd(&end_map);
                     it != end_map; ++it) {
                  if (!it.GetValueRef().GetMessageValue().IsInitialized()) {
                    return false;
                  }
                }
              }
            }
          } else if (field->is_repeated()) {
            const int size = reflection->FieldSize(message, field);
            for (int j = 0; j < size; j++) {
              if (!reflection->GetRepeatedMessage(message, field, j)
                       .IsInitialized()) {
                return false;
              }
            }
          } else if (reflection->HasField(message, field)) {
            if (!reflection->GetMessage(message, field).IsInitialized()) {
              return false;
            }
          }
        }
      }
    }
  }
  if (check_descendants && reflection->HasExtensionSet(message) &&
      !reflection->GetExtensionSet(message).IsInitialized()) {
    return false;
  }
  return true;
}

bool ReflectionOps::IsInitialized(const Message& message) {
  const Descriptor* descriptor = message.GetDescriptor();
  const Reflection* reflection = GetReflectionOrDie(message);

  // Check required fields of this message.
  {
    const int field_count = descriptor->field_count();
    for (int i = 0; i < field_count; i++) {
      if (descriptor->field(i)->is_required()) {
        if (!reflection->HasField(message, descriptor->field(i))) {
          return false;
        }
      }
    }
  }

  // Check that sub-messages are initialized.
  std::vector<const FieldDescriptor*> fields;
  // Should be safe to skip stripped fields because required fields are not
  // stripped.
  reflection->ListFieldsOmitStripped(message, &fields);
  for (const FieldDescriptor* field : fields) {
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {

      if (field->is_map()) {
        const FieldDescriptor* value_field = field->message_type()->field(1);
        if (value_field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
          const MapFieldBase* map_field =
              reflection->GetMapData(message, field);
          if (map_field->IsMapValid()) {
            MapIterator iter(const_cast<Message*>(&message), field);
            MapIterator end(const_cast<Message*>(&message), field);
            for (map_field->MapBegin(&iter), map_field->MapEnd(&end);
                 iter != end; ++iter) {
              if (!iter.GetValueRef().GetMessageValue().IsInitialized()) {
                return false;
              }
            }
            continue;
          }
        } else {
          continue;
        }
      }

      if (field->is_repeated()) {
        int size = reflection->FieldSize(message, field);

        for (int j = 0; j < size; j++) {
          if (!reflection->GetRepeatedMessage(message, field, j)
                   .IsInitialized()) {
            return false;
          }
        }
      } else {
        if (!reflection->GetMessage(message, field).IsInitialized()) {
          return false;
        }
      }
    }
  }

  return true;
}

static bool IsMapValueMessageTyped(const FieldDescriptor* map_field) {
  return map_field->message_type()->field(1)->cpp_type() ==
         FieldDescriptor::CPPTYPE_MESSAGE;
}

void ReflectionOps::DiscardUnknownFields(Message* message) {
  const Reflection* reflection = GetReflectionOrDie(*message);

  reflection->MutableUnknownFields(message)->Clear();

  // Walk through the fields of this message and DiscardUnknownFields on any
  // messages present.
  std::vector<const FieldDescriptor*> fields;
  reflection->ListFields(*message, &fields);
  for (const FieldDescriptor* field : fields) {
    // Skip over non-message fields.
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    // Discard the unknown fields in maps that contain message values.
    if (field->is_map() && IsMapValueMessageTyped(field)) {
      const MapFieldBase* map_field =
          reflection->MutableMapData(message, field);
      if (map_field->IsMapValid()) {
        MapIterator iter(message, field);
        MapIterator end(message, field);
        for (map_field->MapBegin(&iter), map_field->MapEnd(&end); iter != end;
             ++iter) {
          iter.MutableValueRef()->MutableMessageValue()->DiscardUnknownFields();
        }
      }
      // Discard every unknown field inside messages in a repeated field.
    } else if (field->is_repeated()) {
      int size = reflection->FieldSize(*message, field);
      for (int j = 0; j < size; j++) {
        reflection->MutableRepeatedMessage(message, field, j)
            ->DiscardUnknownFields();
      }
      // Discard the unknown fields inside an optional message.
    } else {
      reflection->MutableMessage(message, field)->DiscardUnknownFields();
    }
  }
}

static std::string SubMessagePrefix(const std::string& prefix,
                                    const FieldDescriptor* field, int index) {
  std::string result(prefix);
  if (field->is_extension()) {
    result.append("(");
    result.append(field->full_name());
    result.append(")");
  } else {
    result.append(field->name());
  }
  if (index != -1) {
    result.append("[");
    result.append(StrCat(index));
    result.append("]");
  }
  result.append(".");
  return result;
}

void ReflectionOps::FindInitializationErrors(const Message& message,
                                             const std::string& prefix,
                                             std::vector<std::string>* errors) {
  const Descriptor* descriptor = message.GetDescriptor();
  const Reflection* reflection = GetReflectionOrDie(message);

  // Check required fields of this message.
  {
    const int field_count = descriptor->field_count();
    for (int i = 0; i < field_count; i++) {
      if (descriptor->field(i)->is_required()) {
        if (!reflection->HasField(message, descriptor->field(i))) {
          errors->push_back(prefix + descriptor->field(i)->name());
        }
      }
    }
  }

  // Check sub-messages.
  std::vector<const FieldDescriptor*> fields;
  reflection->ListFieldsOmitStripped(message, &fields);
  for (const FieldDescriptor* field : fields) {
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {

      if (field->is_repeated()) {
        int size = reflection->FieldSize(message, field);

        for (int j = 0; j < size; j++) {
          const Message& sub_message =
              reflection->GetRepeatedMessage(message, field, j);
          FindInitializationErrors(sub_message,
                                   SubMessagePrefix(prefix, field, j), errors);
        }
      } else {
        const Message& sub_message = reflection->GetMessage(message, field);
        FindInitializationErrors(sub_message,
                                 SubMessagePrefix(prefix, field, -1), errors);
      }
    }
  }
}

void GenericSwap(Message* lhs, Message* rhs) {
#ifndef PROTOBUF_FORCE_COPY_IN_SWAP
  GOOGLE_DCHECK(Arena::InternalHelper<Message>::GetOwningArena(lhs) !=
         Arena::InternalHelper<Message>::GetOwningArena(rhs));
  GOOGLE_DCHECK(Arena::InternalHelper<Message>::GetOwningArena(lhs) != nullptr ||
         Arena::InternalHelper<Message>::GetOwningArena(rhs) != nullptr);
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
  // At least one of these must have an arena, so make `rhs` point to it.
  Arena* arena = Arena::InternalHelper<Message>::GetOwningArena(rhs);
  if (arena == nullptr) {
    std::swap(lhs, rhs);
    arena = Arena::InternalHelper<Message>::GetOwningArena(rhs);
  }

  // Improve efficiency by placing the temporary on an arena so that messages
  // are copied twice rather than three times.
  Message* tmp = rhs->New(arena);
  tmp->CheckTypeAndMergeFrom(*lhs);
  lhs->Clear();
  lhs->CheckTypeAndMergeFrom(*rhs);
#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
  rhs->Clear();
  rhs->CheckTypeAndMergeFrom(*tmp);
  if (arena == nullptr) delete tmp;
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
  rhs->GetReflection()->Swap(tmp, rhs);
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
