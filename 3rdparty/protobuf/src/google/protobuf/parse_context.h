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

#ifndef GOOGLE_PROTOBUF_PARSE_CONTEXT_H__
#define GOOGLE_PROTOBUF_PARSE_CONTEXT_H__

#include <cstdint>
#include <cstring>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/implicit_weak_message.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/port.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/stubs/strutil.h>

#include <google/protobuf/port_def.inc>


namespace google {
namespace protobuf {

class UnknownFieldSet;
class DescriptorPool;
class MessageFactory;

namespace internal {

// Template code below needs to know about the existence of these functions.
PROTOBUF_EXPORT void WriteVarint(uint32_t num, uint64_t val, std::string* s);
PROTOBUF_EXPORT void WriteLengthDelimited(uint32_t num, StringPiece val,
                                          std::string* s);
// Inline because it is just forwarding to s->WriteVarint
inline void WriteVarint(uint32_t num, uint64_t val, UnknownFieldSet* s);
inline void WriteLengthDelimited(uint32_t num, StringPiece val,
                                 UnknownFieldSet* s);


// The basic abstraction the parser is designed for is a slight modification
// of the ZeroCopyInputStream (ZCIS) abstraction. A ZCIS presents a serialized
// stream as a series of buffers that concatenate to the full stream.
// Pictorially a ZCIS presents a stream in chunks like so
// [---------------------------------------------------------------]
// [---------------------] chunk 1
//                      [----------------------------] chunk 2
//                                          chunk 3 [--------------]
//
// Where the '-' represent the bytes which are vertically lined up with the
// bytes of the stream. The proto parser requires its input to be presented
// similarly with the extra
// property that each chunk has kSlopBytes past its end that overlaps with the
// first kSlopBytes of the next chunk, or if there is no next chunk at least its
// still valid to read those bytes. Again, pictorially, we now have
//
// [---------------------------------------------------------------]
// [-------------------....] chunk 1
//                    [------------------------....] chunk 2
//                                    chunk 3 [------------------..**]
//                                                      chunk 4 [--****]
// Here '-' mean the bytes of the stream or chunk and '.' means bytes past the
// chunk that match up with the start of the next chunk. Above each chunk has
// 4 '.' after the chunk. In the case these 'overflow' bytes represents bytes
// past the stream, indicated by '*' above, their values are unspecified. It is
// still legal to read them (ie. should not segfault). Reading past the
// end should be detected by the user and indicated as an error.
//
// The reason for this, admittedly, unconventional invariant is to ruthlessly
// optimize the protobuf parser. Having an overlap helps in two important ways.
// Firstly it alleviates having to performing bounds checks if a piece of code
// is guaranteed to not read more than kSlopBytes. Secondly, and more
// importantly, the protobuf wireformat is such that reading a key/value pair is
// always less than 16 bytes. This removes the need to change to next buffer in
// the middle of reading primitive values. Hence there is no need to store and
// load the current position.

class PROTOBUF_EXPORT EpsCopyInputStream {
 public:
  enum { kSlopBytes = 16, kMaxCordBytesToCopy = 512 };

  explicit EpsCopyInputStream(bool enable_aliasing)
      : aliasing_(enable_aliasing ? kOnPatch : kNoAliasing) {}

  void BackUp(const char* ptr) {
    GOOGLE_DCHECK(ptr <= buffer_end_ + kSlopBytes);
    int count;
    if (next_chunk_ == buffer_) {
      count = static_cast<int>(buffer_end_ + kSlopBytes - ptr);
    } else {
      count = size_ + static_cast<int>(buffer_end_ - ptr);
    }
    if (count > 0) StreamBackUp(count);
  }

  // If return value is negative it's an error
  PROTOBUF_NODISCARD int PushLimit(const char* ptr, int limit) {
    GOOGLE_DCHECK(limit >= 0 && limit <= INT_MAX - kSlopBytes);
    // This add is safe due to the invariant above, because
    // ptr - buffer_end_ <= kSlopBytes.
    limit += static_cast<int>(ptr - buffer_end_);
    limit_end_ = buffer_end_ + (std::min)(0, limit);
    auto old_limit = limit_;
    limit_ = limit;
    return old_limit - limit;
  }

  PROTOBUF_NODISCARD bool PopLimit(int delta) {
    if (PROTOBUF_PREDICT_FALSE(!EndedAtLimit())) return false;
    limit_ = limit_ + delta;
    // TODO(gerbens) We could remove this line and hoist the code to
    // DoneFallback. Study the perf/bin-size effects.
    limit_end_ = buffer_end_ + (std::min)(0, limit_);
    return true;
  }

  PROTOBUF_NODISCARD const char* Skip(const char* ptr, int size) {
    if (size <= buffer_end_ + kSlopBytes - ptr) {
      return ptr + size;
    }
    return SkipFallback(ptr, size);
  }
  PROTOBUF_NODISCARD const char* ReadString(const char* ptr, int size,
                                            std::string* s) {
    if (size <= buffer_end_ + kSlopBytes - ptr) {
      s->assign(ptr, size);
      return ptr + size;
    }
    return ReadStringFallback(ptr, size, s);
  }
  PROTOBUF_NODISCARD const char* AppendString(const char* ptr, int size,
                                              std::string* s) {
    if (size <= buffer_end_ + kSlopBytes - ptr) {
      s->append(ptr, size);
      return ptr + size;
    }
    return AppendStringFallback(ptr, size, s);
  }
  // Implemented in arenastring.cc
  PROTOBUF_NODISCARD const char* ReadArenaString(const char* ptr,
                                                 ArenaStringPtr* s,
                                                 Arena* arena);

  template <typename Tag, typename T>
  PROTOBUF_NODISCARD const char* ReadRepeatedFixed(const char* ptr,
                                                   Tag expected_tag,
                                                   RepeatedField<T>* out);

  template <typename T>
  PROTOBUF_NODISCARD const char* ReadPackedFixed(const char* ptr, int size,
                                                 RepeatedField<T>* out);
  template <typename Add>
  PROTOBUF_NODISCARD const char* ReadPackedVarint(const char* ptr, Add add);

  uint32_t LastTag() const { return last_tag_minus_1_ + 1; }
  bool ConsumeEndGroup(uint32_t start_tag) {
    bool res = last_tag_minus_1_ == start_tag;
    last_tag_minus_1_ = 0;
    return res;
  }
  bool EndedAtLimit() const { return last_tag_minus_1_ == 0; }
  bool EndedAtEndOfStream() const { return last_tag_minus_1_ == 1; }
  void SetLastTag(uint32_t tag) { last_tag_minus_1_ = tag - 1; }
  void SetEndOfStream() { last_tag_minus_1_ = 1; }
  bool IsExceedingLimit(const char* ptr) {
    return ptr > limit_end_ &&
           (next_chunk_ == nullptr || ptr - buffer_end_ > limit_);
  }
  int BytesUntilLimit(const char* ptr) const {
    return limit_ + static_cast<int>(buffer_end_ - ptr);
  }
  // Returns true if more data is available, if false is returned one has to
  // call Done for further checks.
  bool DataAvailable(const char* ptr) { return ptr < limit_end_; }

 protected:
  // Returns true is limit (either an explicit limit or end of stream) is
  // reached. It aligns *ptr across buffer seams.
  // If limit is exceeded it returns true and ptr is set to null.
  bool DoneWithCheck(const char** ptr, int d) {
    GOOGLE_DCHECK(*ptr);
    if (PROTOBUF_PREDICT_TRUE(*ptr < limit_end_)) return false;
    int overrun = static_cast<int>(*ptr - buffer_end_);
    GOOGLE_DCHECK_LE(overrun, kSlopBytes);  // Guaranteed by parse loop.
    if (overrun ==
        limit_) {  //  No need to flip buffers if we ended on a limit.
      // If we actually overrun the buffer and next_chunk_ is null. It means
      // the stream ended and we passed the stream end.
      if (overrun > 0 && next_chunk_ == nullptr) *ptr = nullptr;
      return true;
    }
    auto res = DoneFallback(overrun, d);
    *ptr = res.first;
    return res.second;
  }

  const char* InitFrom(StringPiece flat) {
    overall_limit_ = 0;
    if (flat.size() > kSlopBytes) {
      limit_ = kSlopBytes;
      limit_end_ = buffer_end_ = flat.data() + flat.size() - kSlopBytes;
      next_chunk_ = buffer_;
      if (aliasing_ == kOnPatch) aliasing_ = kNoDelta;
      return flat.data();
    } else {
      std::memcpy(buffer_, flat.data(), flat.size());
      limit_ = 0;
      limit_end_ = buffer_end_ = buffer_ + flat.size();
      next_chunk_ = nullptr;
      if (aliasing_ == kOnPatch) {
        aliasing_ = reinterpret_cast<std::uintptr_t>(flat.data()) -
                    reinterpret_cast<std::uintptr_t>(buffer_);
      }
      return buffer_;
    }
  }

  const char* InitFrom(io::ZeroCopyInputStream* zcis);

  const char* InitFrom(io::ZeroCopyInputStream* zcis, int limit) {
    if (limit == -1) return InitFrom(zcis);
    overall_limit_ = limit;
    auto res = InitFrom(zcis);
    limit_ = limit - static_cast<int>(buffer_end_ - res);
    limit_end_ = buffer_end_ + (std::min)(0, limit_);
    return res;
  }

 private:
  const char* limit_end_;  // buffer_end_ + min(limit_, 0)
  const char* buffer_end_;
  const char* next_chunk_;
  int size_;
  int limit_;  // relative to buffer_end_;
  io::ZeroCopyInputStream* zcis_ = nullptr;
  char buffer_[2 * kSlopBytes] = {};
  enum { kNoAliasing = 0, kOnPatch = 1, kNoDelta = 2 };
  std::uintptr_t aliasing_ = kNoAliasing;
  // This variable is used to communicate how the parse ended, in order to
  // completely verify the parsed data. A wire-format parse can end because of
  // one of the following conditions:
  // 1) A parse can end on a pushed limit.
  // 2) A parse can end on End Of Stream (EOS).
  // 3) A parse can end on 0 tag (only valid for toplevel message).
  // 4) A parse can end on an end-group tag.
  // This variable should always be set to 0, which indicates case 1. If the
  // parse terminated due to EOS (case 2), it's set to 1. In case the parse
  // ended due to a terminating tag (case 3 and 4) it's set to (tag - 1).
  // This var doesn't really belong in EpsCopyInputStream and should be part of
  // the ParseContext, but case 2 is most easily and optimally implemented in
  // DoneFallback.
  uint32_t last_tag_minus_1_ = 0;
  int overall_limit_ = INT_MAX;  // Overall limit independent of pushed limits.
  // Pretty random large number that seems like a safe allocation on most
  // systems. TODO(gerbens) do we need to set this as build flag?
  enum { kSafeStringSize = 50000000 };

  // Advances to next buffer chunk returns a pointer to the same logical place
  // in the stream as set by overrun. Overrun indicates the position in the slop
  // region the parse was left (0 <= overrun <= kSlopBytes). Returns true if at
  // limit, at which point the returned pointer maybe null if there was an
  // error. The invariant of this function is that it's guaranteed that
  // kSlopBytes bytes can be accessed from the returned ptr. This function might
  // advance more buffers than one in the underlying ZeroCopyInputStream.
  std::pair<const char*, bool> DoneFallback(int overrun, int depth);
  // Advances to the next buffer, at most one call to Next() on the underlying
  // ZeroCopyInputStream is made. This function DOES NOT match the returned
  // pointer to where in the slop region the parse ends, hence no overrun
  // parameter. This is useful for string operations where you always copy
  // to the end of the buffer (including the slop region).
  const char* Next();
  // overrun is the location in the slop region the stream currently is
  // (0 <= overrun <= kSlopBytes). To prevent flipping to the next buffer of
  // the ZeroCopyInputStream in the case the parse will end in the last
  // kSlopBytes of the current buffer. depth is the current depth of nested
  // groups (or negative if the use case does not need careful tracking).
  inline const char* NextBuffer(int overrun, int depth);
  const char* SkipFallback(const char* ptr, int size);
  const char* AppendStringFallback(const char* ptr, int size, std::string* str);
  const char* ReadStringFallback(const char* ptr, int size, std::string* str);
  bool StreamNext(const void** data) {
    bool res = zcis_->Next(data, &size_);
    if (res) overall_limit_ -= size_;
    return res;
  }
  void StreamBackUp(int count) {
    zcis_->BackUp(count);
    overall_limit_ += count;
  }

  template <typename A>
  const char* AppendSize(const char* ptr, int size, const A& append) {
    int chunk_size = buffer_end_ + kSlopBytes - ptr;
    do {
      GOOGLE_DCHECK(size > chunk_size);
      if (next_chunk_ == nullptr) return nullptr;
      append(ptr, chunk_size);
      ptr += chunk_size;
      size -= chunk_size;
      // TODO(gerbens) Next calls NextBuffer which generates buffers with
      // overlap and thus incurs cost of copying the slop regions. This is not
      // necessary for reading strings. We should just call Next buffers.
      if (limit_ <= kSlopBytes) return nullptr;
      ptr = Next();
      if (ptr == nullptr) return nullptr;  // passed the limit
      ptr += kSlopBytes;
      chunk_size = buffer_end_ + kSlopBytes - ptr;
    } while (size > chunk_size);
    append(ptr, size);
    return ptr + size;
  }

  // AppendUntilEnd appends data until a limit (either a PushLimit or end of
  // stream. Normal payloads are from length delimited fields which have an
  // explicit size. Reading until limit only comes when the string takes
  // the place of a protobuf, ie RawMessage/StringRawMessage, lazy fields and
  // implicit weak messages. We keep these methods private and friend them.
  template <typename A>
  const char* AppendUntilEnd(const char* ptr, const A& append) {
    if (ptr - buffer_end_ > limit_) return nullptr;
    while (limit_ > kSlopBytes) {
      size_t chunk_size = buffer_end_ + kSlopBytes - ptr;
      append(ptr, chunk_size);
      ptr = Next();
      if (ptr == nullptr) return limit_end_;
      ptr += kSlopBytes;
    }
    auto end = buffer_end_ + limit_;
    GOOGLE_DCHECK(end >= ptr);
    append(ptr, end - ptr);
    return end;
  }

  PROTOBUF_NODISCARD const char* AppendString(const char* ptr,
                                              std::string* str) {
    return AppendUntilEnd(
        ptr, [str](const char* p, ptrdiff_t s) { str->append(p, s); });
  }
  friend class ImplicitWeakMessage;
};

// ParseContext holds all data that is global to the entire parse. Most
// importantly it contains the input stream, but also recursion depth and also
// stores the end group tag, in case a parser ended on a endgroup, to verify
// matching start/end group tags.
class PROTOBUF_EXPORT ParseContext : public EpsCopyInputStream {
 public:
  struct Data {
    const DescriptorPool* pool = nullptr;
    MessageFactory* factory = nullptr;
    Arena* arena = nullptr;
  };

  template <typename... T>
  ParseContext(int depth, bool aliasing, const char** start, T&&... args)
      : EpsCopyInputStream(aliasing), depth_(depth) {
    *start = InitFrom(std::forward<T>(args)...);
  }

  void TrackCorrectEnding() { group_depth_ = 0; }

  bool Done(const char** ptr) { return DoneWithCheck(ptr, group_depth_); }

  int depth() const { return depth_; }

  Data& data() { return data_; }
  const Data& data() const { return data_; }

  const char* ParseMessage(MessageLite* msg, const char* ptr);

  // This overload supports those few cases where ParseMessage is called
  // on a class that is not actually a proto message.
  // TODO(jorg): Eliminate this use case.
  template <typename T,
            typename std::enable_if<!std::is_base_of<MessageLite, T>::value,
                                    bool>::type = true>
  PROTOBUF_NODISCARD const char* ParseMessage(T* msg, const char* ptr);

  template <typename T>
  PROTOBUF_NODISCARD PROTOBUF_NDEBUG_INLINE const char* ParseGroup(
      T* msg, const char* ptr, uint32_t tag) {
    if (--depth_ < 0) return nullptr;
    group_depth_++;
    ptr = msg->_InternalParse(ptr, this);
    group_depth_--;
    depth_++;
    if (PROTOBUF_PREDICT_FALSE(!ConsumeEndGroup(tag))) return nullptr;
    return ptr;
  }

 private:
  // Out-of-line routine to save space in ParseContext::ParseMessage<T>
  //   int old;
  //   ptr = ReadSizeAndPushLimitAndDepth(ptr, &old)
  // is equivalent to:
  //   int size = ReadSize(&ptr);
  //   if (!ptr) return nullptr;
  //   int old = PushLimit(ptr, size);
  //   if (--depth_ < 0) return nullptr;
  PROTOBUF_NODISCARD const char* ReadSizeAndPushLimitAndDepth(const char* ptr,
                                                              int* old_limit);

  // The context keeps an internal stack to keep track of the recursive
  // part of the parse state.
  // Current depth of the active parser, depth counts down.
  // This is used to limit recursion depth (to prevent overflow on malicious
  // data), but is also used to index in stack_ to store the current state.
  int depth_;
  // Unfortunately necessary for the fringe case of ending on 0 or end-group tag
  // in the last kSlopBytes of a ZeroCopyInputStream chunk.
  int group_depth_ = INT_MIN;
  Data data_;
};

template <uint32_t tag>
bool ExpectTag(const char* ptr) {
  if (tag < 128) {
    return *ptr == static_cast<char>(tag);
  } else {
    static_assert(tag < 128 * 128, "We only expect tags for 1 or 2 bytes");
    char buf[2] = {static_cast<char>(tag | 0x80), static_cast<char>(tag >> 7)};
    return std::memcmp(ptr, buf, 2) == 0;
  }
}

template <int>
struct EndianHelper;

template <>
struct EndianHelper<1> {
  static uint8_t Load(const void* p) { return *static_cast<const uint8_t*>(p); }
};

template <>
struct EndianHelper<2> {
  static uint16_t Load(const void* p) {
    uint16_t tmp;
    std::memcpy(&tmp, p, 2);
#ifndef PROTOBUF_LITTLE_ENDIAN
    tmp = bswap_16(tmp);
#endif
    return tmp;
  }
};

template <>
struct EndianHelper<4> {
  static uint32_t Load(const void* p) {
    uint32_t tmp;
    std::memcpy(&tmp, p, 4);
#ifndef PROTOBUF_LITTLE_ENDIAN
    tmp = bswap_32(tmp);
#endif
    return tmp;
  }
};

template <>
struct EndianHelper<8> {
  static uint64_t Load(const void* p) {
    uint64_t tmp;
    std::memcpy(&tmp, p, 8);
#ifndef PROTOBUF_LITTLE_ENDIAN
    tmp = bswap_64(tmp);
#endif
    return tmp;
  }
};

template <typename T>
T UnalignedLoad(const char* p) {
  auto tmp = EndianHelper<sizeof(T)>::Load(p);
  T res;
  memcpy(&res, &tmp, sizeof(T));
  return res;
}

PROTOBUF_EXPORT
std::pair<const char*, uint32_t> VarintParseSlow32(const char* p, uint32_t res);
PROTOBUF_EXPORT
std::pair<const char*, uint64_t> VarintParseSlow64(const char* p, uint32_t res);

inline const char* VarintParseSlow(const char* p, uint32_t res, uint32_t* out) {
  auto tmp = VarintParseSlow32(p, res);
  *out = tmp.second;
  return tmp.first;
}

inline const char* VarintParseSlow(const char* p, uint32_t res, uint64_t* out) {
  auto tmp = VarintParseSlow64(p, res);
  *out = tmp.second;
  return tmp.first;
}

template <typename T>
PROTOBUF_NODISCARD const char* VarintParse(const char* p, T* out) {
  auto ptr = reinterpret_cast<const uint8_t*>(p);
  uint32_t res = ptr[0];
  if (!(res & 0x80)) {
    *out = res;
    return p + 1;
  }
  uint32_t byte = ptr[1];
  res += (byte - 1) << 7;
  if (!(byte & 0x80)) {
    *out = res;
    return p + 2;
  }
  return VarintParseSlow(p, res, out);
}

// Used for tags, could read up to 5 bytes which must be available.
// Caller must ensure its safe to call.

PROTOBUF_EXPORT
std::pair<const char*, uint32_t> ReadTagFallback(const char* p, uint32_t res);

// Same as ParseVarint but only accept 5 bytes at most.
inline const char* ReadTag(const char* p, uint32_t* out,
                           uint32_t /*max_tag*/ = 0) {
  uint32_t res = static_cast<uint8_t>(p[0]);
  if (res < 128) {
    *out = res;
    return p + 1;
  }
  uint32_t second = static_cast<uint8_t>(p[1]);
  res += (second - 1) << 7;
  if (second < 128) {
    *out = res;
    return p + 2;
  }
  auto tmp = ReadTagFallback(p, res);
  *out = tmp.second;
  return tmp.first;
}

// Decode 2 consecutive bytes of a varint and returns the value, shifted left
// by 1. It simultaneous updates *ptr to *ptr + 1 or *ptr + 2 depending if the
// first byte's continuation bit is set.
// If bit 15 of return value is set (equivalent to the continuation bits of both
// bytes being set) the varint continues, otherwise the parse is done. On x86
// movsx eax, dil
// add edi, eax
// adc [rsi], 1
// add eax, eax
// and eax, edi
inline uint32_t DecodeTwoBytes(const char** ptr) {
  uint32_t value = UnalignedLoad<uint16_t>(*ptr);
  // Sign extend the low byte continuation bit
  uint32_t x = static_cast<int8_t>(value);
  // This add is an amazing operation, it cancels the low byte continuation bit
  // from y transferring it to the carry. Simultaneously it also shifts the 7
  // LSB left by one tightly against high byte varint bits. Hence value now
  // contains the unpacked value shifted left by 1.
  value += x;
  // Use the carry to update the ptr appropriately.
  *ptr += value < x ? 2 : 1;
  return value & (x + x);  // Mask out the high byte iff no continuation
}

// More efficient varint parsing for big varints
inline const char* ParseBigVarint(const char* p, uint64_t* out) {
  auto pnew = p;
  auto tmp = DecodeTwoBytes(&pnew);
  uint64_t res = tmp >> 1;
  if (PROTOBUF_PREDICT_TRUE(static_cast<std::int16_t>(tmp) >= 0)) {
    *out = res;
    return pnew;
  }
  for (std::uint32_t i = 1; i < 5; i++) {
    pnew = p + 2 * i;
    tmp = DecodeTwoBytes(&pnew);
    res += (static_cast<std::uint64_t>(tmp) - 2) << (14 * i - 1);
    if (PROTOBUF_PREDICT_TRUE(static_cast<std::int16_t>(tmp) >= 0)) {
      *out = res;
      return pnew;
    }
  }
  return nullptr;
}

PROTOBUF_EXPORT
std::pair<const char*, int32_t> ReadSizeFallback(const char* p, uint32_t first);
// Used for tags, could read up to 5 bytes which must be available. Additionally
// it makes sure the unsigned value fits a int32_t, otherwise returns nullptr.
// Caller must ensure its safe to call.
inline uint32_t ReadSize(const char** pp) {
  auto p = *pp;
  uint32_t res = static_cast<uint8_t>(p[0]);
  if (res < 128) {
    *pp = p + 1;
    return res;
  }
  auto x = ReadSizeFallback(p, res);
  *pp = x.first;
  return x.second;
}

// Some convenience functions to simplify the generated parse loop code.
// Returning the value and updating the buffer pointer allows for nicer
// function composition. We rely on the compiler to inline this.
// Also in debug compiles having local scoped variables tend to generated
// stack frames that scale as O(num fields).
inline uint64_t ReadVarint64(const char** p) {
  uint64_t tmp;
  *p = VarintParse(*p, &tmp);
  return tmp;
}

inline uint32_t ReadVarint32(const char** p) {
  uint32_t tmp;
  *p = VarintParse(*p, &tmp);
  return tmp;
}

inline int64_t ReadVarintZigZag64(const char** p) {
  uint64_t tmp;
  *p = VarintParse(*p, &tmp);
  return WireFormatLite::ZigZagDecode64(tmp);
}

inline int32_t ReadVarintZigZag32(const char** p) {
  uint64_t tmp;
  *p = VarintParse(*p, &tmp);
  return WireFormatLite::ZigZagDecode32(static_cast<uint32_t>(tmp));
}

template <typename T, typename std::enable_if<
                          !std::is_base_of<MessageLite, T>::value, bool>::type>
PROTOBUF_NODISCARD const char* ParseContext::ParseMessage(T* msg,
                                                          const char* ptr) {
  int old;
  ptr = ReadSizeAndPushLimitAndDepth(ptr, &old);
  ptr = ptr ? msg->_InternalParse(ptr, this) : nullptr;
  depth_++;
  if (!PopLimit(old)) return nullptr;
  return ptr;
}

template <typename Tag, typename T>
const char* EpsCopyInputStream::ReadRepeatedFixed(const char* ptr,
                                                  Tag expected_tag,
                                                  RepeatedField<T>* out) {
  do {
    out->Add(UnalignedLoad<T>(ptr));
    ptr += sizeof(T);
    if (PROTOBUF_PREDICT_FALSE(ptr >= limit_end_)) return ptr;
  } while (UnalignedLoad<Tag>(ptr) == expected_tag && (ptr += sizeof(Tag)));
  return ptr;
}

// Add any of the following lines to debug which parse function is failing.

#define GOOGLE_PROTOBUF_ASSERT_RETURN(predicate, ret) \
  if (!(predicate)) {                                  \
    /*  ::raise(SIGINT);  */                           \
    /*  GOOGLE_LOG(ERROR) << "Parse failure";  */             \
    return ret;                                        \
  }

#define GOOGLE_PROTOBUF_PARSER_ASSERT(predicate) \
  GOOGLE_PROTOBUF_ASSERT_RETURN(predicate, nullptr)

template <typename T>
const char* EpsCopyInputStream::ReadPackedFixed(const char* ptr, int size,
                                                RepeatedField<T>* out) {
  GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
  int nbytes = buffer_end_ + kSlopBytes - ptr;
  while (size > nbytes) {
    int num = nbytes / sizeof(T);
    int old_entries = out->size();
    out->Reserve(old_entries + num);
    int block_size = num * sizeof(T);
    auto dst = out->AddNAlreadyReserved(num);
#ifdef PROTOBUF_LITTLE_ENDIAN
    std::memcpy(dst, ptr, block_size);
#else
    for (int i = 0; i < num; i++)
      dst[i] = UnalignedLoad<T>(ptr + i * sizeof(T));
#endif
    size -= block_size;
    if (limit_ <= kSlopBytes) return nullptr;
    ptr = Next();
    if (ptr == nullptr) return nullptr;
    ptr += kSlopBytes - (nbytes - block_size);
    nbytes = buffer_end_ + kSlopBytes - ptr;
  }
  int num = size / sizeof(T);
  int old_entries = out->size();
  out->Reserve(old_entries + num);
  int block_size = num * sizeof(T);
  auto dst = out->AddNAlreadyReserved(num);
#ifdef PROTOBUF_LITTLE_ENDIAN
  std::memcpy(dst, ptr, block_size);
#else
  for (int i = 0; i < num; i++) dst[i] = UnalignedLoad<T>(ptr + i * sizeof(T));
#endif
  ptr += block_size;
  if (size != block_size) return nullptr;
  return ptr;
}

template <typename Add>
const char* ReadPackedVarintArray(const char* ptr, const char* end, Add add) {
  while (ptr < end) {
    uint64_t varint;
    ptr = VarintParse(ptr, &varint);
    if (ptr == nullptr) return nullptr;
    add(varint);
  }
  return ptr;
}

template <typename Add>
const char* EpsCopyInputStream::ReadPackedVarint(const char* ptr, Add add) {
  int size = ReadSize(&ptr);
  GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
  int chunk_size = buffer_end_ - ptr;
  while (size > chunk_size) {
    ptr = ReadPackedVarintArray(ptr, buffer_end_, add);
    if (ptr == nullptr) return nullptr;
    int overrun = ptr - buffer_end_;
    GOOGLE_DCHECK(overrun >= 0 && overrun <= kSlopBytes);
    if (size - chunk_size <= kSlopBytes) {
      // The current buffer contains all the information needed, we don't need
      // to flip buffers. However we must parse from a buffer with enough space
      // so we are not prone to a buffer overflow.
      char buf[kSlopBytes + 10] = {};
      std::memcpy(buf, buffer_end_, kSlopBytes);
      GOOGLE_CHECK_LE(size - chunk_size, kSlopBytes);
      auto end = buf + (size - chunk_size);
      auto res = ReadPackedVarintArray(buf + overrun, end, add);
      if (res == nullptr || res != end) return nullptr;
      return buffer_end_ + (res - buf);
    }
    size -= overrun + chunk_size;
    GOOGLE_DCHECK_GT(size, 0);
    // We must flip buffers
    if (limit_ <= kSlopBytes) return nullptr;
    ptr = Next();
    if (ptr == nullptr) return nullptr;
    ptr += overrun;
    chunk_size = buffer_end_ - ptr;
  }
  auto end = ptr + size;
  ptr = ReadPackedVarintArray(ptr, end, add);
  return end == ptr ? ptr : nullptr;
}

// Helper for verification of utf8
PROTOBUF_EXPORT
bool VerifyUTF8(StringPiece s, const char* field_name);

inline bool VerifyUTF8(const std::string* s, const char* field_name) {
  return VerifyUTF8(*s, field_name);
}

// All the string parsers with or without UTF checking and for all CTypes.
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* InlineGreedyStringParser(
    std::string* s, const char* ptr, ParseContext* ctx);


template <typename T>
PROTOBUF_NODISCARD const char* FieldParser(uint64_t tag, T& field_parser,
                                           const char* ptr, ParseContext* ctx) {
  uint32_t number = tag >> 3;
  GOOGLE_PROTOBUF_PARSER_ASSERT(number != 0);
  using WireType = internal::WireFormatLite::WireType;
  switch (tag & 7) {
    case WireType::WIRETYPE_VARINT: {
      uint64_t value;
      ptr = VarintParse(ptr, &value);
      GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
      field_parser.AddVarint(number, value);
      break;
    }
    case WireType::WIRETYPE_FIXED64: {
      uint64_t value = UnalignedLoad<uint64_t>(ptr);
      ptr += 8;
      field_parser.AddFixed64(number, value);
      break;
    }
    case WireType::WIRETYPE_LENGTH_DELIMITED: {
      ptr = field_parser.ParseLengthDelimited(number, ptr, ctx);
      GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
      break;
    }
    case WireType::WIRETYPE_START_GROUP: {
      ptr = field_parser.ParseGroup(number, ptr, ctx);
      GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
      break;
    }
    case WireType::WIRETYPE_END_GROUP: {
      GOOGLE_LOG(FATAL) << "Can't happen";
      break;
    }
    case WireType::WIRETYPE_FIXED32: {
      uint32_t value = UnalignedLoad<uint32_t>(ptr);
      ptr += 4;
      field_parser.AddFixed32(number, value);
      break;
    }
    default:
      return nullptr;
  }
  return ptr;
}

template <typename T>
PROTOBUF_NODISCARD const char* WireFormatParser(T& field_parser,
                                                const char* ptr,
                                                ParseContext* ctx) {
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ReadTag(ptr, &tag);
    GOOGLE_PROTOBUF_PARSER_ASSERT(ptr != nullptr);
    if (tag == 0 || (tag & 7) == 4) {
      ctx->SetLastTag(tag);
      return ptr;
    }
    ptr = FieldParser(tag, field_parser, ptr, ctx);
    GOOGLE_PROTOBUF_PARSER_ASSERT(ptr != nullptr);
  }
  return ptr;
}

// The packed parsers parse repeated numeric primitives directly into  the
// corresponding field

// These are packed varints
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedInt32Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedUInt32Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedInt64Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedUInt64Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedSInt32Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedSInt64Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedEnumParser(
    void* object, const char* ptr, ParseContext* ctx);

template <typename T>
PROTOBUF_NODISCARD const char* PackedEnumParser(void* object, const char* ptr,
                                                ParseContext* ctx,
                                                bool (*is_valid)(int),
                                                InternalMetadata* metadata,
                                                int field_num) {
  return ctx->ReadPackedVarint(
      ptr, [object, is_valid, metadata, field_num](uint64_t val) {
        if (is_valid(val)) {
          static_cast<RepeatedField<int>*>(object)->Add(val);
        } else {
          WriteVarint(field_num, val, metadata->mutable_unknown_fields<T>());
        }
      });
}

template <typename T>
PROTOBUF_NODISCARD const char* PackedEnumParserArg(
    void* object, const char* ptr, ParseContext* ctx,
    bool (*is_valid)(const void*, int), const void* data,
    InternalMetadata* metadata, int field_num) {
  return ctx->ReadPackedVarint(
      ptr, [object, is_valid, data, metadata, field_num](uint64_t val) {
        if (is_valid(data, val)) {
          static_cast<RepeatedField<int>*>(object)->Add(val);
        } else {
          WriteVarint(field_num, val, metadata->mutable_unknown_fields<T>());
        }
      });
}

PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedBoolParser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedFixed32Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedSFixed32Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedFixed64Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedSFixed64Parser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedFloatParser(
    void* object, const char* ptr, ParseContext* ctx);
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* PackedDoubleParser(
    void* object, const char* ptr, ParseContext* ctx);

// This is the only recursive parser.
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* UnknownGroupLiteParse(
    std::string* unknown, const char* ptr, ParseContext* ctx);
// This is a helper to for the UnknownGroupLiteParse but is actually also
// useful in the generated code. It uses overload on std::string* vs
// UnknownFieldSet* to make the generated code isomorphic between full and lite.
PROTOBUF_EXPORT PROTOBUF_NODISCARD const char* UnknownFieldParse(
    uint32_t tag, std::string* unknown, const char* ptr, ParseContext* ctx);

}  // namespace internal
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_PARSE_CONTEXT_H__
