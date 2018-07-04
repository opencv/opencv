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

#include <google/protobuf/any.h>

#include <google/protobuf/generated_message_util.h>

namespace google {
namespace protobuf {
namespace internal {

namespace {
string GetTypeUrl(const Descriptor* message,
                  const string& type_url_prefix) {
  if (!type_url_prefix.empty() &&
      type_url_prefix[type_url_prefix.size() - 1] == '/') {
    return type_url_prefix + message->full_name();
  } else {
    return type_url_prefix + "/" + message->full_name();
  }
}
}  // namespace

const char kAnyFullTypeName[] = "google.protobuf.Any";
const char kTypeGoogleApisComPrefix[] = "type.googleapis.com/";
const char kTypeGoogleProdComPrefix[] = "type.googleprod.com/";

AnyMetadata::AnyMetadata(UrlType* type_url, ValueType* value)
    : type_url_(type_url), value_(value) {
}

void AnyMetadata::PackFrom(const Message& message) {
  PackFrom(message, kTypeGoogleApisComPrefix);
}

void AnyMetadata::PackFrom(const Message& message,
                           const string& type_url_prefix) {
  type_url_->SetNoArena(&::google::protobuf::internal::GetEmptyString(),
                        GetTypeUrl(message.GetDescriptor(), type_url_prefix));
  message.SerializeToString(value_->MutableNoArena(
      &::google::protobuf::internal::GetEmptyStringAlreadyInited()));
}

bool AnyMetadata::UnpackTo(Message* message) const {
  if (!InternalIs(message->GetDescriptor())) {
    return false;
  }
  return message->ParseFromString(value_->GetNoArena());
}

bool AnyMetadata::InternalIs(const Descriptor* descriptor) const {
  const string type_url = type_url_->GetNoArena();
  string full_name;
  if (!ParseAnyTypeUrl(type_url, &full_name)) {
    return false;
  }
  return full_name == descriptor->full_name();
}

bool ParseAnyTypeUrl(const string& type_url, string* full_type_name) {
  size_t pos = type_url.find_last_of("/");
  if (pos == string::npos || pos + 1 == type_url.size()) {
    return false;
  }
  *full_type_name = type_url.substr(pos + 1);
  return true;
}


bool GetAnyFieldDescriptors(const Message& message,
                            const FieldDescriptor** type_url_field,
                            const FieldDescriptor** value_field) {
    const Descriptor* descriptor = message.GetDescriptor();
    if (descriptor->full_name() != kAnyFullTypeName) {
      return false;
    }
    *type_url_field = descriptor->FindFieldByNumber(1);
    *value_field = descriptor->FindFieldByNumber(2);
    return (*type_url_field != NULL &&
            (*type_url_field)->type() == FieldDescriptor::TYPE_STRING &&
            *value_field != NULL &&
            (*value_field)->type() == FieldDescriptor::TYPE_BYTES);
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google
