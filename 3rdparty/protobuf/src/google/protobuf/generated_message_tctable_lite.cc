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

#include <cstdint>

#include <google/protobuf/parse_context.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_tctable_decl.h>
#include <google/protobuf/generated_message_tctable_impl.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/wire_format_lite.h>

// clang-format off
#include <google/protobuf/port_def.inc>
// clang-format on

namespace google {
namespace protobuf {
namespace internal {

#ifndef NDEBUG
template void AlignFail<4>(uintptr_t);
template void AlignFail<8>(uintptr_t);
#endif

const char* TcParser::GenericFallbackLite(PROTOBUF_TC_PARAM_DECL) {
  return GenericFallbackImpl<MessageLite, std::string>(PROTOBUF_TC_PARAM_PASS);
}

namespace {

// Offset returns the address `offset` bytes after `base`.
inline void* Offset(void* base, uint32_t offset) {
  return static_cast<uint8_t*>(base) + offset;
}

// InvertPacked changes tag bits from the given wire type to length
// delimited. This is the difference expected between packed and non-packed
// repeated fields.
template <WireFormatLite::WireType Wt>
inline PROTOBUF_ALWAYS_INLINE void InvertPacked(TcFieldData& data) {
  data.data ^= Wt ^ WireFormatLite::WIRETYPE_LENGTH_DELIMITED;
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// Fixed fields
//////////////////////////////////////////////////////////////////////////////

template <typename LayoutType, typename TagType>
const char* TcParser::SingularFixed(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    return table->fallback(PROTOBUF_TC_PARAM_PASS);
  }
  ptr += sizeof(TagType);  // Consume tag
  hasbits |= (uint64_t{1} << data.hasbit_idx());
  std::memcpy(Offset(msg, data.offset()), ptr, sizeof(LayoutType));
  ptr += sizeof(LayoutType);
  PROTOBUF_MUSTTAIL return ToTagDispatch(PROTOBUF_TC_PARAM_PASS);
}

template <typename LayoutType, typename TagType>
const char* TcParser::RepeatedFixed(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    // Check if the field can be parsed as packed repeated:
    constexpr WireFormatLite::WireType fallback_wt =
        sizeof(LayoutType) == 4 ? WireFormatLite::WIRETYPE_FIXED32
                                : WireFormatLite::WIRETYPE_FIXED64;
    InvertPacked<fallback_wt>(data);
    if (data.coded_tag<TagType>() == 0) {
      return PackedFixed<LayoutType, TagType>(PROTOBUF_TC_PARAM_PASS);
    } else {
      return table->fallback(PROTOBUF_TC_PARAM_PASS);
    }
  }
  auto& field = RefAt<RepeatedField<LayoutType>>(msg, data.offset());
  int idx = field.size();
  auto elem = field.Add();
  int space = field.Capacity() - idx;
  idx = 0;
  auto expected_tag = UnalignedLoad<TagType>(ptr);
  do {
    ptr += sizeof(TagType);
    std::memcpy(elem + (idx++), ptr, sizeof(LayoutType));
    ptr += sizeof(LayoutType);
    if (idx >= space) break;
    if (!ctx->DataAvailable(ptr)) break;
  } while (UnalignedLoad<TagType>(ptr) == expected_tag);
  field.AddNAlreadyReserved(idx - 1);
  return ToParseLoop(PROTOBUF_TC_PARAM_PASS);
}

template <typename LayoutType, typename TagType>
const char* TcParser::PackedFixed(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    // Try parsing as non-packed repeated:
    constexpr WireFormatLite::WireType fallback_wt =
        sizeof(LayoutType) == 4 ? WireFormatLite::WIRETYPE_FIXED32
                                : WireFormatLite::WIRETYPE_FIXED64;
    InvertPacked<fallback_wt>(data);
    if (data.coded_tag<TagType>() == 0) {
      return RepeatedFixed<LayoutType, TagType>(PROTOBUF_TC_PARAM_PASS);
    } else {
      return table->fallback(PROTOBUF_TC_PARAM_PASS);
    }
  }
  ptr += sizeof(TagType);
  // Since ctx->ReadPackedFixed does not use TailCall<> or Return<>, sync any
  // pending hasbits now:
  SyncHasbits(msg, hasbits, table);
  auto& field = RefAt<RepeatedField<LayoutType>>(msg, data.offset());
  int size = ReadSize(&ptr);
  // TODO(dlj): add a tailcalling variant of ReadPackedFixed.
  return ctx->ReadPackedFixed(ptr, size,
                              static_cast<RepeatedField<LayoutType>*>(&field));
}

//////////////////////////////////////////////////////////////////////////////
// Varint fields
//////////////////////////////////////////////////////////////////////////////

namespace {

inline PROTOBUF_ALWAYS_INLINE std::pair<const char*, uint64_t>
Parse64FallbackPair(const char* p, int64_t res1) {
  auto ptr = reinterpret_cast<const int8_t*>(p);

  // The algorithm relies on sign extension for each byte to set all high bits
  // when the varint continues. It also relies on asserting all of the lower
  // bits for each successive byte read. This allows the result to be aggregated
  // using a bitwise AND. For example:
  //
  //          8       1          64     57 ... 24     17  16      9  8       1
  // ptr[0] = 1aaa aaaa ; res1 = 1111 1111 ... 1111 1111  1111 1111  1aaa aaaa
  // ptr[1] = 1bbb bbbb ; res2 = 1111 1111 ... 1111 1111  11bb bbbb  b111 1111
  // ptr[2] = 1ccc cccc ; res3 = 0000 0000 ... 000c cccc  cc11 1111  1111 1111
  //                             ---------------------------------------------
  //        res1 & res2 & res3 = 0000 0000 ... 000c cccc  ccbb bbbb  baaa aaaa
  //
  // On x86-64, a shld from a single register filled with enough 1s in the high
  // bits can accomplish all this in one instruction. It so happens that res1
  // has 57 high bits of ones, which is enough for the largest shift done.
  GOOGLE_DCHECK_EQ(res1 >> 7, -1);
  uint64_t ones = res1;  // save the high 1 bits from res1 (input to SHLD)
  uint64_t byte;         // the "next" 7-bit chunk, shifted (result from SHLD)
  int64_t res2, res3;    // accumulated result chunks
#define SHLD(n) byte = ((byte << (n * 7)) | (ones >> (64 - (n * 7))))

  int sign_bit;
#if defined(__GCC_ASM_FLAG_OUTPUTS__) && defined(__x86_64__)
  // For the first two rounds (ptr[1] and ptr[2]), micro benchmarks show a
  // substantial improvement from capturing the sign from the condition code
  // register on x86-64.
#define SHLD_SIGN(n)                  \
  asm("shldq %3, %2, %1"              \
      : "=@ccs"(sign_bit), "+r"(byte) \
      : "r"(ones), "i"(n * 7))
#else
  // Generic fallback:
#define SHLD_SIGN(n)                           \
  do {                                         \
    SHLD(n);                                   \
    sign_bit = static_cast<int64_t>(byte) < 0; \
  } while (0)
#endif

  byte = ptr[1];
  SHLD_SIGN(1);
  res2 = byte;
  if (!sign_bit) goto done2;
  byte = ptr[2];
  SHLD_SIGN(2);
  res3 = byte;
  if (!sign_bit) goto done3;

#undef SHLD_SIGN

  // For the remainder of the chunks, check the sign of the AND result.
  byte = ptr[3];
  SHLD(3);
  res1 &= byte;
  if (res1 >= 0) goto done4;
  byte = ptr[4];
  SHLD(4);
  res2 &= byte;
  if (res2 >= 0) goto done5;
  byte = ptr[5];
  SHLD(5);
  res3 &= byte;
  if (res3 >= 0) goto done6;
  byte = ptr[6];
  SHLD(6);
  res1 &= byte;
  if (res1 >= 0) goto done7;
  byte = ptr[7];
  SHLD(7);
  res2 &= byte;
  if (res2 >= 0) goto done8;
  byte = ptr[8];
  SHLD(8);
  res3 &= byte;
  if (res3 >= 0) goto done9;

#undef SHLD

  // For valid 64bit varints, the 10th byte/ptr[9] should be exactly 1. In this
  // case, the continuation bit of ptr[8] already set the top bit of res3
  // correctly, so all we have to do is check that the expected case is true.
  byte = ptr[9];
  if (PROTOBUF_PREDICT_TRUE(byte == 1)) goto done10;

  // A value of 0, however, represents an over-serialized varint. This case
  // should not happen, but if does (say, due to a nonconforming serializer),
  // deassert the continuation bit that came from ptr[8].
  if (byte == 0) {
    res3 ^= static_cast<uint64_t>(1) << 63;
    goto done10;
  }

  // If the 10th byte/ptr[9] itself has any other value, then it is too big to
  // fit in 64 bits. If the continue bit is set, it is an unterminated varint.
  return {nullptr, 0};

#define DONE(n) done##n : return {p + n, res1 & res2 & res3};
done2:
  return {p + 2, res1 & res2};
  DONE(3)
  DONE(4)
  DONE(5)
  DONE(6)
  DONE(7)
  DONE(8)
  DONE(9)
  DONE(10)
#undef DONE
}

inline PROTOBUF_ALWAYS_INLINE const char* ParseVarint(const char* p,
                                                      uint64_t* value) {
  int64_t byte = static_cast<int8_t>(*p);
  if (PROTOBUF_PREDICT_TRUE(byte >= 0)) {
    *value = byte;
    return p + 1;
  } else {
    auto tmp = Parse64FallbackPair(p, byte);
    if (PROTOBUF_PREDICT_TRUE(tmp.first)) *value = tmp.second;
    return tmp.first;
  }
}

template <typename FieldType,
          TcParser::VarintDecode = TcParser::VarintDecode::kNoConversion>
FieldType ZigZagDecodeHelper(uint64_t value) {
  return static_cast<FieldType>(value);
}

template <>
int32_t ZigZagDecodeHelper<int32_t, TcParser::VarintDecode::kZigZag>(
    uint64_t value) {
  return WireFormatLite::ZigZagDecode32(value);
}

template <>
int64_t ZigZagDecodeHelper<int64_t, TcParser::VarintDecode::kZigZag>(
    uint64_t value) {
  return WireFormatLite::ZigZagDecode64(value);
}

}  // namespace

template <typename FieldType, typename TagType, TcParser::VarintDecode zigzag>
const char* TcParser::SingularVarint(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    return table->fallback(PROTOBUF_TC_PARAM_PASS);
  }
  ptr += sizeof(TagType);  // Consume tag
  hasbits |= (uint64_t{1} << data.hasbit_idx());
  uint64_t tmp;
  ptr = ParseVarint(ptr, &tmp);
  if (ptr == nullptr) {
    return Error(PROTOBUF_TC_PARAM_PASS);
  }
  RefAt<FieldType>(msg, data.offset()) =
      ZigZagDecodeHelper<FieldType, zigzag>(tmp);
  PROTOBUF_MUSTTAIL return ToTagDispatch(PROTOBUF_TC_PARAM_PASS);
}

template <typename FieldType, typename TagType, TcParser::VarintDecode zigzag>
PROTOBUF_NOINLINE const char* TcParser::RepeatedVarint(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    // Try parsing as non-packed repeated:
    InvertPacked<WireFormatLite::WIRETYPE_VARINT>(data);
    if (data.coded_tag<TagType>() == 0) {
      return PackedVarint<FieldType, TagType, zigzag>(PROTOBUF_TC_PARAM_PASS);
    } else {
      return table->fallback(PROTOBUF_TC_PARAM_PASS);
    }
  }
  auto& field = RefAt<RepeatedField<FieldType>>(msg, data.offset());
  auto expected_tag = UnalignedLoad<TagType>(ptr);
  do {
    ptr += sizeof(TagType);
    uint64_t tmp;
    ptr = ParseVarint(ptr, &tmp);
    if (ptr == nullptr) {
      return Error(PROTOBUF_TC_PARAM_PASS);
    }
    field.Add(ZigZagDecodeHelper<FieldType, zigzag>(tmp));
    if (!ctx->DataAvailable(ptr)) {
      break;
    }
  } while (UnalignedLoad<TagType>(ptr) == expected_tag);
  return ToParseLoop(PROTOBUF_TC_PARAM_PASS);
}

template <typename FieldType, typename TagType, TcParser::VarintDecode zigzag>
PROTOBUF_NOINLINE const char* TcParser::PackedVarint(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    InvertPacked<WireFormatLite::WIRETYPE_VARINT>(data);
    if (data.coded_tag<TagType>() == 0) {
      return RepeatedVarint<FieldType, TagType, zigzag>(PROTOBUF_TC_PARAM_PASS);
    } else {
      return table->fallback(PROTOBUF_TC_PARAM_PASS);
    }
  }
  ptr += sizeof(TagType);
  // Since ctx->ReadPackedVarint does not use TailCall or Return, sync any
  // pending hasbits now:
  SyncHasbits(msg, hasbits, table);
  auto* field = &RefAt<RepeatedField<FieldType>>(msg, data.offset());
  return ctx->ReadPackedVarint(ptr, [field](uint64_t varint) {
    FieldType val;
    if (zigzag) {
      if (sizeof(FieldType) == 8) {
        val = WireFormatLite::ZigZagDecode64(varint);
      } else {
        val = WireFormatLite::ZigZagDecode32(varint);
      }
    } else {
      val = varint;
    }
    field->Add(val);
  });
}

//////////////////////////////////////////////////////////////////////////////
// String/bytes fields
//////////////////////////////////////////////////////////////////////////////

// Defined in wire_format_lite.cc
void PrintUTF8ErrorLog(const char* field_name, const char* operation_str,
                       bool emit_stacktrace);

namespace {

PROTOBUF_NOINLINE
const char* SingularStringParserFallback(ArenaStringPtr* s, const char* ptr,
                                         EpsCopyInputStream* stream) {
  int size = ReadSize(&ptr);
  if (!ptr) return nullptr;
  return stream->ReadString(
      ptr, size, s->MutableNoArenaNoDefault(&GetEmptyStringAlreadyInited()));
}

}  // namespace

template <typename TagType, TcParser::Utf8Type utf8>
const char* TcParser::SingularString(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    return table->fallback(PROTOBUF_TC_PARAM_PASS);
  }
  ptr += sizeof(TagType);
  hasbits |= (uint64_t{1} << data.hasbit_idx());
  auto& field = RefAt<ArenaStringPtr>(msg, data.offset());
  auto arena = ctx->data().arena;
  if (arena) {
    ptr = ctx->ReadArenaString(ptr, &field, arena);
  } else {
    ptr = SingularStringParserFallback(&field, ptr, ctx);
  }
  if (ptr == nullptr) return Error(PROTOBUF_TC_PARAM_PASS);
  switch (utf8) {
    case kNoUtf8:
#ifdef NDEBUG
    case kUtf8ValidateOnly:
#endif
      return ToParseLoop(PROTOBUF_TC_PARAM_PASS);
    default:
      if (PROTOBUF_PREDICT_TRUE(IsStructurallyValidUTF8(field.Get()))) {
        return ToParseLoop(PROTOBUF_TC_PARAM_PASS);
      }
      PrintUTF8ErrorLog("unknown", "parsing", false);
      return utf8 == kUtf8 ? Error(PROTOBUF_TC_PARAM_PASS)
                           : ToParseLoop(PROTOBUF_TC_PARAM_PASS);
  }
}

template <typename TagType, TcParser::Utf8Type utf8>
const char* TcParser::RepeatedString(PROTOBUF_TC_PARAM_DECL) {
  if (PROTOBUF_PREDICT_FALSE(data.coded_tag<TagType>() != 0)) {
    return table->fallback(PROTOBUF_TC_PARAM_PASS);
  }
  auto expected_tag = UnalignedLoad<TagType>(ptr);
  auto& field = RefAt<RepeatedPtrField<std::string>>(msg, data.offset());
  do {
    ptr += sizeof(TagType);
    std::string* str = field.Add();
    ptr = InlineGreedyStringParser(str, ptr, ctx);
    if (ptr == nullptr) {
      return Error(PROTOBUF_TC_PARAM_PASS);
    }
    if (utf8 != kNoUtf8) {
      if (PROTOBUF_PREDICT_FALSE(!IsStructurallyValidUTF8(*str))) {
        PrintUTF8ErrorLog("unknown", "parsing", false);
        if (utf8 == kUtf8) return Error(PROTOBUF_TC_PARAM_PASS);
      }
    }
    if (!ctx->DataAvailable(ptr)) break;
  } while (UnalignedLoad<TagType>(ptr) == expected_tag);
  return ToParseLoop(PROTOBUF_TC_PARAM_PASS);
}

#define PROTOBUF_TCT_SOURCE
#include <google/protobuf/generated_message_tctable_impl.inc>

}  // namespace internal
}  // namespace protobuf
}  // namespace google
