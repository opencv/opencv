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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_EXPECTING_OBJECTWRITER_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_EXPECTING_OBJECTWRITER_H__

// An implementation of ObjectWriter that automatically sets the
// gmock expectations for the response to a method. Every method
// returns the object itself for chaining.
//
// Usage:
//   // Setup
//   MockObjectWriter mock;
//   ExpectingObjectWriter ow(&mock);
//
//   // Set expectation
//   ow.StartObject("")
//       ->RenderString("key", "value")
//     ->EndObject();
//
//   // Actual testing
//   mock.StartObject(StringPiece())
//         ->RenderString("key", "value")
//       ->EndObject();

#include <cstdint>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/util/internal/object_writer.h>
#include <gmock/gmock.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace util {
namespace converter {

using testing::Eq;
using testing::IsEmpty;
using testing::NanSensitiveDoubleEq;
using testing::NanSensitiveFloatEq;
using testing::Return;
using testing::StrEq;
using testing::TypedEq;

class MockObjectWriter : public ObjectWriter {
 public:
  MockObjectWriter() {}

  MOCK_METHOD(ObjectWriter*, StartObject, (StringPiece), (override));
  MOCK_METHOD(ObjectWriter*, EndObject, (), (override));
  MOCK_METHOD(ObjectWriter*, StartList, (StringPiece), (override));
  MOCK_METHOD(ObjectWriter*, EndList, (), (override));
  MOCK_METHOD(ObjectWriter*, RenderBool, (StringPiece, bool), (override));
  MOCK_METHOD(ObjectWriter*, RenderInt32, (StringPiece, int32_t),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderUint32, (StringPiece, uint32_t),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderInt64, (StringPiece, int64_t),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderUint64, (StringPiece, uint64_t),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderDouble, (StringPiece, double),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderFloat, (StringPiece, float),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderString,
              (StringPiece, StringPiece), (override));
  MOCK_METHOD(ObjectWriter*, RenderBytes, (StringPiece, StringPiece),
              (override));
  MOCK_METHOD(ObjectWriter*, RenderNull, (StringPiece), (override));
};

class ExpectingObjectWriter : public ObjectWriter {
 public:
  explicit ExpectingObjectWriter(MockObjectWriter* mock) : mock_(mock) {}

  virtual ObjectWriter* StartObject(StringPiece name) {
    (name.empty() ? EXPECT_CALL(*mock_, StartObject(IsEmpty()))
                  : EXPECT_CALL(*mock_, StartObject(Eq(std::string(name)))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* EndObject() {
    EXPECT_CALL(*mock_, EndObject())
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* StartList(StringPiece name) {
    (name.empty() ? EXPECT_CALL(*mock_, StartList(IsEmpty()))
                  : EXPECT_CALL(*mock_, StartList(Eq(std::string(name)))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* EndList() {
    EXPECT_CALL(*mock_, EndList())
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderBool(StringPiece name, bool value) {
    (name.empty()
         ? EXPECT_CALL(*mock_, RenderBool(IsEmpty(), TypedEq<bool>(value)))
         : EXPECT_CALL(*mock_,
                       RenderBool(Eq(std::string(name)), TypedEq<bool>(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderInt32(StringPiece name, int32_t value) {
    (name.empty()
         ? EXPECT_CALL(*mock_, RenderInt32(IsEmpty(), TypedEq<int32_t>(value)))
         : EXPECT_CALL(*mock_, RenderInt32(Eq(std::string(name)),
                                           TypedEq<int32_t>(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderUint32(StringPiece name, uint32_t value) {
    (name.empty() ? EXPECT_CALL(*mock_, RenderUint32(IsEmpty(),
                                                     TypedEq<uint32_t>(value)))
                  : EXPECT_CALL(*mock_, RenderUint32(Eq(std::string(name)),
                                                     TypedEq<uint32_t>(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderInt64(StringPiece name, int64_t value) {
    (name.empty()
         ? EXPECT_CALL(*mock_, RenderInt64(IsEmpty(), TypedEq<int64_t>(value)))
         : EXPECT_CALL(*mock_, RenderInt64(Eq(std::string(name)),
                                           TypedEq<int64_t>(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderUint64(StringPiece name, uint64_t value) {
    (name.empty() ? EXPECT_CALL(*mock_, RenderUint64(IsEmpty(),
                                                     TypedEq<uint64_t>(value)))
                  : EXPECT_CALL(*mock_, RenderUint64(Eq(std::string(name)),
                                                     TypedEq<uint64_t>(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderDouble(StringPiece name, double value) {
    (name.empty()
         ? EXPECT_CALL(*mock_,
                       RenderDouble(IsEmpty(), NanSensitiveDoubleEq(value)))
         : EXPECT_CALL(*mock_, RenderDouble(Eq(std::string(name)),
                                            NanSensitiveDoubleEq(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderFloat(StringPiece name, float value) {
    (name.empty()
         ? EXPECT_CALL(*mock_,
                       RenderFloat(IsEmpty(), NanSensitiveFloatEq(value)))
         : EXPECT_CALL(*mock_, RenderFloat(Eq(std::string(name)),
                                           NanSensitiveFloatEq(value))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderString(StringPiece name,
                                     StringPiece value) {
    (name.empty() ? EXPECT_CALL(*mock_, RenderString(IsEmpty(),
                                                     TypedEq<StringPiece>(
                                                         std::string(value))))
                  : EXPECT_CALL(*mock_, RenderString(Eq(std::string(name)),
                                                     TypedEq<StringPiece>(
                                                         std::string(value)))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }
  virtual ObjectWriter* RenderBytes(StringPiece name, StringPiece value) {
    (name.empty()
         ? EXPECT_CALL(*mock_, RenderBytes(IsEmpty(), TypedEq<StringPiece>(
                                                          value.ToString())))
         : EXPECT_CALL(*mock_,
                       RenderBytes(Eq(std::string(name)),
                                   TypedEq<StringPiece>(value.ToString()))))
        .WillOnce(Return(mock_))
        .RetiresOnSaturation();
    return this;
  }

  virtual ObjectWriter* RenderNull(StringPiece name) {
    (name.empty() ? EXPECT_CALL(*mock_, RenderNull(IsEmpty()))
                  : EXPECT_CALL(*mock_, RenderNull(Eq(std::string(name))))
                        .WillOnce(Return(mock_))
                        .RetiresOnSaturation());
    return this;
  }

 private:
  MockObjectWriter* mock_;

  GOOGLE_DISALLOW_IMPLICIT_CONSTRUCTORS(ExpectingObjectWriter);
};

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_EXPECTING_OBJECTWRITER_H__
