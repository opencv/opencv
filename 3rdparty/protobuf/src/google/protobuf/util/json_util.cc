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

#include <google/protobuf/util/json_util.h>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/util/internal/default_value_objectwriter.h>
#include <google/protobuf/util/internal/error_listener.h>
#include <google/protobuf/util/internal/json_objectwriter.h>
#include <google/protobuf/util/internal/json_stream_parser.h>
#include <google/protobuf/util/internal/protostream_objectsource.h>
#include <google/protobuf/util/internal/protostream_objectwriter.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/util/type_resolver_util.h>
#include <google/protobuf/stubs/bytestream.h>
#include <google/protobuf/stubs/status_macros.h>

namespace google {
namespace protobuf {
namespace util {

namespace internal {
void ZeroCopyStreamByteSink::Append(const char* bytes, size_t len) {
  while (len > 0) {
    void* buffer;
    int length;
    if (!stream_->Next(&buffer, &length)) {
      // There isn't a way for ByteSink to report errors.
      return;
    }
    if (len < length) {
      memcpy(buffer, bytes, len);
      stream_->BackUp(length - len);
      break;
    } else {
      memcpy(buffer, bytes, length);
      bytes += length;
      len -= length;
    }
  }
}
}  // namespace internal

util::Status BinaryToJsonStream(TypeResolver* resolver,
                                  const string& type_url,
                                  io::ZeroCopyInputStream* binary_input,
                                  io::ZeroCopyOutputStream* json_output,
                                  const JsonPrintOptions& options) {
  io::CodedInputStream in_stream(binary_input);
  google::protobuf::Type type;
  RETURN_IF_ERROR(resolver->ResolveMessageType(type_url, &type));
  converter::ProtoStreamObjectSource proto_source(&in_stream, resolver, type);
  io::CodedOutputStream out_stream(json_output);
  converter::JsonObjectWriter json_writer(options.add_whitespace ? " " : "",
                                          &out_stream);
  if (options.always_print_primitive_fields) {
    converter::DefaultValueObjectWriter default_value_writer(
        resolver, type, &json_writer);
    return proto_source.WriteTo(&default_value_writer);
  } else {
    return proto_source.WriteTo(&json_writer);
  }
}

util::Status BinaryToJsonString(TypeResolver* resolver,
                                  const string& type_url,
                                  const string& binary_input,
                                  string* json_output,
                                  const JsonPrintOptions& options) {
  io::ArrayInputStream input_stream(binary_input.data(), binary_input.size());
  io::StringOutputStream output_stream(json_output);
  return BinaryToJsonStream(resolver, type_url, &input_stream, &output_stream,
                            options);
}

namespace {
class StatusErrorListener : public converter::ErrorListener {
 public:
  StatusErrorListener() : status_(util::Status::OK) {}
  virtual ~StatusErrorListener() {}

  util::Status GetStatus() { return status_; }

  virtual void InvalidName(const converter::LocationTrackerInterface& loc,
                           StringPiece unknown_name, StringPiece message) {
    status_ = util::Status(util::error::INVALID_ARGUMENT,
                             loc.ToString() + ": " + message.ToString());
  }

  virtual void InvalidValue(const converter::LocationTrackerInterface& loc,
                            StringPiece type_name, StringPiece value) {
    status_ =
        util::Status(util::error::INVALID_ARGUMENT,
                       loc.ToString() + ": invalid value " + value.ToString() +
                           " for type " + type_name.ToString());
  }

  virtual void MissingField(const converter::LocationTrackerInterface& loc,
                            StringPiece missing_name) {
    status_ = util::Status(
        util::error::INVALID_ARGUMENT,
        loc.ToString() + ": missing field " + missing_name.ToString());
  }

 private:
  util::Status status_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(StatusErrorListener);
};
}  // namespace

util::Status JsonToBinaryStream(TypeResolver* resolver,
                                  const string& type_url,
                                  io::ZeroCopyInputStream* json_input,
                                  io::ZeroCopyOutputStream* binary_output,
                                  const JsonParseOptions& options) {
  google::protobuf::Type type;
  RETURN_IF_ERROR(resolver->ResolveMessageType(type_url, &type));
  internal::ZeroCopyStreamByteSink sink(binary_output);
  StatusErrorListener listener;
  converter::ProtoStreamObjectWriter::Options proto_writer_options;
  proto_writer_options.ignore_unknown_fields = options.ignore_unknown_fields;
  converter::ProtoStreamObjectWriter proto_writer(resolver, type, &sink,
                                                  &listener,
                                                  proto_writer_options);

  converter::JsonStreamParser parser(&proto_writer);
  const void* buffer;
  int length;
  while (json_input->Next(&buffer, &length)) {
    if (length == 0) continue;
    RETURN_IF_ERROR(
        parser.Parse(StringPiece(static_cast<const char*>(buffer), length)));
  }
  RETURN_IF_ERROR(parser.FinishParse());

  return listener.GetStatus();
}

util::Status JsonToBinaryString(TypeResolver* resolver,
                                  const string& type_url,
                                  const string& json_input,
                                  string* binary_output,
                                  const JsonParseOptions& options) {
  io::ArrayInputStream input_stream(json_input.data(), json_input.size());
  io::StringOutputStream output_stream(binary_output);
  return JsonToBinaryStream(
      resolver, type_url, &input_stream, &output_stream, options);
}

namespace {
const char* kTypeUrlPrefix = "type.googleapis.com";
TypeResolver* generated_type_resolver_ = NULL;
GOOGLE_PROTOBUF_DECLARE_ONCE(generated_type_resolver_init_);

string GetTypeUrl(const Message& message) {
  return string(kTypeUrlPrefix) + "/" + message.GetDescriptor()->full_name();
}

void DeleteGeneratedTypeResolver() { delete generated_type_resolver_; }

void InitGeneratedTypeResolver() {
  generated_type_resolver_ = NewTypeResolverForDescriptorPool(
      kTypeUrlPrefix, DescriptorPool::generated_pool());
  ::google::protobuf::internal::OnShutdown(&DeleteGeneratedTypeResolver);
}

TypeResolver* GetGeneratedTypeResolver() {
  ::google::protobuf::GoogleOnceInit(&generated_type_resolver_init_, &InitGeneratedTypeResolver);
  return generated_type_resolver_;
}
}  // namespace

util::Status MessageToJsonString(const Message& message, string* output,
                                   const JsonOptions& options) {
  const DescriptorPool* pool = message.GetDescriptor()->file()->pool();
  TypeResolver* resolver =
      pool == DescriptorPool::generated_pool()
          ? GetGeneratedTypeResolver()
          : NewTypeResolverForDescriptorPool(kTypeUrlPrefix, pool);
  util::Status result =
      BinaryToJsonString(resolver, GetTypeUrl(message),
                         message.SerializeAsString(), output, options);
  if (pool != DescriptorPool::generated_pool()) {
    delete resolver;
  }
  return result;
}

util::Status JsonStringToMessage(const string& input, Message* message,
                                   const JsonParseOptions& options) {
  const DescriptorPool* pool = message->GetDescriptor()->file()->pool();
  TypeResolver* resolver =
      pool == DescriptorPool::generated_pool()
          ? GetGeneratedTypeResolver()
          : NewTypeResolverForDescriptorPool(kTypeUrlPrefix, pool);
  string binary;
  util::Status result = JsonToBinaryString(
      resolver, GetTypeUrl(*message), input, &binary, options);
  if (result.ok() && !message->ParseFromString(binary)) {
    result =
        util::Status(util::error::INVALID_ARGUMENT,
                       "JSON transcoder produced invalid protobuf output.");
  }
  if (pool != DescriptorPool::generated_pool()) {
    delete resolver;
  }
  return result;
}

}  // namespace util
}  // namespace protobuf
}  // namespace google
