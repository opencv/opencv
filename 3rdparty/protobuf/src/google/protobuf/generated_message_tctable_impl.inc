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

// clang-format off
#ifdef PROTOBUF_TCT_SOURCE
#define PROTOBUF_TCT_EXTERN
#else
#define PROTOBUF_TCT_EXTERN extern
#endif
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularFixed<uint64_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedFixed<uint64_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedFixed<uint64_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularFixed<uint32_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedFixed<uint32_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedFixed<uint32_t, uint8_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<uint64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<uint64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<uint64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<uint32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<uint32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<uint32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<int64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<int64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<int64_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<int32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<int32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<int32_t, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<bool, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<bool, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<bool, uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8ValidateOnly>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint8_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8ValidateOnly>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularFixed<uint64_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedFixed<uint64_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedFixed<uint64_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularFixed<uint32_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedFixed<uint32_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedFixed<uint32_t, uint16_t>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<uint64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<uint64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<uint64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<uint32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<uint32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<uint32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<int64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<int64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<int64_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<int32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<int32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<int32_t, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kZigZag>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularVarint<bool, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedVarint<bool, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::PackedVarint<bool, uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoConversion>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kNoUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::SingularString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8ValidateOnly>(PROTOBUF_TC_PARAM_DECL);
PROTOBUF_TCT_EXTERN template const char* TcParser::RepeatedString<uint16_t, ::PROTOBUF_NAMESPACE_ID::internal::TcParser::kUtf8ValidateOnly>(PROTOBUF_TC_PARAM_DECL);
#undef PROTOBUF_TCT_EXTERN
// clang-format on
