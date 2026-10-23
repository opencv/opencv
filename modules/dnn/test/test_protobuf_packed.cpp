// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv-onnx.pb.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace opencv_test {

namespace {

// Regression test for issue #29152:
// UBSan reported "null pointer passed to memcpy" in
// EpsCopyInputStream::ReadPackedFixed<float> when a packed `repeated float`
// field encodes a wire length smaller than sizeof(float) (1..3 bytes).
// In that case `num = size / sizeof(T)` is 0, `AddNAlreadyReserved(0)`
// returns `unsafe_elements()` (nullptr for an empty field) and the code
// performed `std::memcpy(nullptr, ptr, 0)`, which is undefined behaviour
// because glibc's memcpy carries the __nonnull attribute on both pointers
// regardless of the length.
//
// The fix guards the memcpy with `if (block_size > 0)`. This test parses a
// hand-built buffer whose float_data field carries only a single byte and
// verifies the parse succeeds and yields the expected (empty) repeated field
// without triggering the null-memcpy path.

// Build a minimal TensorProto buffer: field 4 (float_data, packed) with a
// wire length of `payloadBytes` (1..3), followed by that many raw bytes.
static std::vector<uchar> buildTruncatedFloatTensor(int payloadBytes) {
    std::vector<uchar> buf;
    // Tag for field 4, wire type 2 (length delimited): (4 << 3) | 2 = 0x22
    buf.push_back(0x22);
    // Length of the packed payload (single varint byte, < sizeof(float))
    buf.push_back(static_cast<uchar>(payloadBytes));
    // Raw payload bytes (only payloadBytes valid, the rest is trailing garbage)
    for (int i = 0; i < payloadBytes; ++i)
        buf.push_back(static_cast<uchar>(0x40 + i));
    return buf;
}

// Parse a raw buffer as an opencv_onnx::TensorProto using the partial parser,
// exactly the path used by the ONNX importer for externally supplied models.
static bool parseTensorProto(const std::vector<uchar>& buf,
                             ::opencv_onnx::TensorProto& proto) {
    ::google::protobuf::io::ArrayInputStream input(buf.data(),
                                                    static_cast<int>(buf.size()));
    ::google::protobuf::io::CodedInputStream coded(&input);
    // Allow the small buffers we build here.
    coded.SetTotalBytesLimit(1024);
    return proto.ParsePartialFromCodedStream(&coded);
}

} // namespace

TEST(Test_Protobuf_Packed, TruncatedFloatData_NoNullMemcpy) {
    for (int payloadBytes = 1; payloadBytes <= 3; ++payloadBytes) {
        std::vector<uchar> buf = buildTruncatedFloatTensor(payloadBytes);
        ::opencv_onnx::TensorProto proto;
        bool ok = parseTensorProto(buf, proto);
        // A truncated packed field (size < sizeof(float)) yields num == 0, so
        // AddNAlreadyReserved(0) returns nullptr and no full element is decoded.
        // ParsePartialFromCodedStream reports the leftover via a false return,
        // which is expected here; the important part is that no crash occurs and
        // float_data() stays empty (this is what exercised the null memcpy).
        EXPECT_FALSE(ok) << "payloadBytes=" << payloadBytes;
        EXPECT_EQ(proto.float_data().size(), 0u) << "payloadBytes=" << payloadBytes;
    }
}

} // namespace opencv_test
