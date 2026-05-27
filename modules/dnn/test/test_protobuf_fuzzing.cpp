// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

// Regression tests for crash-000:
// UBSan null-pointer-passed-to-memcpy in
// EpsCopyInputStream::ReadPackedFixed<float> when parsing a Caffe binary model
// whose packed `repeated float` payload length is not a multiple of sizeof(float).
//
// See https://github.com/opencv/opencv/issues/29152
// Fix: 3rdparty/protobuf/src/google/protobuf/parse_context.h

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test { namespace {

// ---------------------------------------------------------------------------
// Helper: build a raw Caffe binary model buffer whose V1LayerParameter.blobs_lr
// field (packed repeated float, wire field 7) carries a payload of `payloadLen`
// bytes.  Only payloadLen matters for triggering the bug; the rest of the bytes
// are taken from the original PoC.
// ---------------------------------------------------------------------------
static std::vector<uchar> makeMalformedCaffeModel(uint8_t payloadLen)
{
    // NetParameter field 2 (layers, wire_type=2), total inner length depends on
    // payloadLen.  The inner message is:
    //   field 2/wire64 (8 bytes) + field 8/wire32 (4 bytes) + field 7/wire2 tag
    //   (2 bytes) + varint payloadLen (1 byte) + payloadLen bytes of payload
    // inner_len = 8 + 4 + 2 + 1 + payloadLen = 15 + payloadLen
    uint8_t inner_len = static_cast<uint8_t>(15 + payloadLen);

    std::vector<uchar> buf = {
        0x12, inner_len,                                      // field 2 (layers), len
          0x11, 0x0e, 0x0e, 0x0e, 0x0e, 0x00, 0x1a, 0xb7,   // field 2, 64-bit fixed
          0xb5, 0x45, 0x13, 0xa2, 0x47, 0x13,                // field 8, 32-bit fixed
          0xba, 0x00,                                         // field 7 (blobs_lr) tag
          payloadLen,                                         // varint: payload length
    };
    // payload bytes (content doesn't matter — length is what triggers the bug)
    for (uint8_t i = 0; i < payloadLen; ++i)
        buf.push_back(static_cast<uchar>(i));
    return buf;
}

// A deliberately invalid one-byte prototxt that makes readNetFromCaffe skip
// the text-proto path and fall through to the binary path.
static const std::vector<uchar> kBadProto = { 0x0c };

// ---------------------------------------------------------------------------
// Crash / UBSan regression — issue #29152
//
// Before the fix: ReadPackedFixed<float> with payload < sizeof(float) calls
//   AddNAlreadyReserved(0) which returns nullptr on a default-constructed
//   RepeatedField, then memcpy(nullptr, ptr, 0) fires UBSan __nonnull.
//
// After the fix: the malformed payload is rejected cleanly; protobuf signals
//   parse failure, which propagates as a cv::Exception from ReadNetParamsFromBinaryBufferOrDie.
//   No undefined behavior is invoked.
// ---------------------------------------------------------------------------

// payload = 1 byte → num = 1/4 = 0 → was UB; now cv::Exception thrown cleanly
TEST(Dnn_Protobuf, readNetFromCaffe_packed_float_payload_1byte_29152)
{
    auto model = makeMalformedCaffeModel(1);
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

// payload = 2 bytes (< sizeof(float)) — was UB
TEST(Dnn_Protobuf, readNetFromCaffe_packed_float_payload_2bytes_29152)
{
    auto model = makeMalformedCaffeModel(2);
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

// payload = 3 bytes (< sizeof(float)) — was UB
TEST(Dnn_Protobuf, readNetFromCaffe_packed_float_payload_3bytes_29152)
{
    auto model = makeMalformedCaffeModel(3);
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

// payload = 5 bytes (not a multiple of 4, > sizeof(float))
// Tail-block fix: size % sizeof(T) != 0 → return nullptr → cv::Exception
TEST(Dnn_Protobuf, readNetFromCaffe_packed_float_payload_5bytes_29152)
{
    auto model = makeMalformedCaffeModel(5);
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

// Exact PoC from the bug report (23-byte model, payload=1)
TEST(Dnn_Protobuf, readNetFromCaffe_exact_poc_crash000_29152)
{
    const std::vector<uchar> model = {
        0x12, 0x11,
          0x11, 0x0e, 0x0e, 0x0e, 0x0e, 0x00, 0x1a, 0xb7,
          0xb5, 0x45, 0x13, 0xa2, 0x47, 0x13,
          0xba, 0x00,
          0x01,
        0x25, 0xb7, 0x9f, 0x45
    };
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

// Completely empty model buffer — malformed, must throw cleanly
TEST(Dnn_Protobuf, readNetFromCaffe_empty_model_throws_29152)
{
    const std::vector<uchar> model;
    EXPECT_THROW(cv::dnn::readNetFromCaffe(kBadProto, model), cv::Exception);
}

}} // namespace opencv_test
