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
// This implementation is heavily optimized to make reads and writes
// of small values (especially varints) as fast as possible.  In
// particular, we optimize for the common case that a read or a write
// will not cross the end of the buffer, since we can avoid a lot
// of branching in this case.

#include <google/protobuf/io/coded_stream.h>

#include <limits.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/stubs/stl_util.h>


#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace io {

namespace {

static const int kMaxVarintBytes = 10;
static const int kMaxVarint32Bytes = 5;


inline bool NextNonEmpty(ZeroCopyInputStream* input, const void** data,
                         int* size) {
  bool success;
  do {
    success = input->Next(data, size);
  } while (success && *size == 0);
  return success;
}

}  // namespace

// CodedInputStream ==================================================

CodedInputStream::~CodedInputStream() {
  if (input_ != NULL) {
    BackUpInputToCurrentPosition();
  }
}

// Static.
int CodedInputStream::default_recursion_limit_ = 100;


void CodedInputStream::BackUpInputToCurrentPosition() {
  int backup_bytes = BufferSize() + buffer_size_after_limit_ + overflow_bytes_;
  if (backup_bytes > 0) {
    input_->BackUp(backup_bytes);

    // total_bytes_read_ doesn't include overflow_bytes_.
    total_bytes_read_ -= BufferSize() + buffer_size_after_limit_;
    buffer_end_ = buffer_;
    buffer_size_after_limit_ = 0;
    overflow_bytes_ = 0;
  }
}

inline void CodedInputStream::RecomputeBufferLimits() {
  buffer_end_ += buffer_size_after_limit_;
  int closest_limit = std::min(current_limit_, total_bytes_limit_);
  if (closest_limit < total_bytes_read_) {
    // The limit position is in the current buffer.  We must adjust
    // the buffer size accordingly.
    buffer_size_after_limit_ = total_bytes_read_ - closest_limit;
    buffer_end_ -= buffer_size_after_limit_;
  } else {
    buffer_size_after_limit_ = 0;
  }
}

CodedInputStream::Limit CodedInputStream::PushLimit(int byte_limit) {
  // Current position relative to the beginning of the stream.
  int current_position = CurrentPosition();

  Limit old_limit = current_limit_;

  // security: byte_limit is possibly evil, so check for negative values
  // and overflow. Also check that the new requested limit is before the
  // previous limit; otherwise we continue to enforce the previous limit.
  if (PROTOBUF_PREDICT_TRUE(byte_limit >= 0 &&
                            byte_limit <= INT_MAX - current_position &&
                            byte_limit < current_limit_ - current_position)) {
    current_limit_ = current_position + byte_limit;
    RecomputeBufferLimits();
  }

  return old_limit;
}

void CodedInputStream::PopLimit(Limit limit) {
  // The limit passed in is actually the *old* limit, which we returned from
  // PushLimit().
  current_limit_ = limit;
  RecomputeBufferLimits();

  // We may no longer be at a legitimate message end.  ReadTag() needs to be
  // called again to find out.
  legitimate_message_end_ = false;
}

std::pair<CodedInputStream::Limit, int>
CodedInputStream::IncrementRecursionDepthAndPushLimit(int byte_limit) {
  return std::make_pair(PushLimit(byte_limit), --recursion_budget_);
}

CodedInputStream::Limit CodedInputStream::ReadLengthAndPushLimit() {
  uint32_t length;
  return PushLimit(ReadVarint32(&length) ? length : 0);
}

bool CodedInputStream::DecrementRecursionDepthAndPopLimit(Limit limit) {
  bool result = ConsumedEntireMessage();
  PopLimit(limit);
  GOOGLE_DCHECK_LT(recursion_budget_, recursion_limit_);
  ++recursion_budget_;
  return result;
}

bool CodedInputStream::CheckEntireMessageConsumedAndPopLimit(Limit limit) {
  bool result = ConsumedEntireMessage();
  PopLimit(limit);
  return result;
}

int CodedInputStream::BytesUntilLimit() const {
  if (current_limit_ == INT_MAX) return -1;
  int current_position = CurrentPosition();

  return current_limit_ - current_position;
}

void CodedInputStream::SetTotalBytesLimit(int total_bytes_limit) {
  // Make sure the limit isn't already past, since this could confuse other
  // code.
  int current_position = CurrentPosition();
  total_bytes_limit_ = std::max(current_position, total_bytes_limit);
  RecomputeBufferLimits();
}

int CodedInputStream::BytesUntilTotalBytesLimit() const {
  if (total_bytes_limit_ == INT_MAX) return -1;
  return total_bytes_limit_ - CurrentPosition();
}

void CodedInputStream::PrintTotalBytesLimitError() {
  GOOGLE_LOG(ERROR)
      << "A protocol message was rejected because it was too "
         "big (more than "
      << total_bytes_limit_
      << " bytes).  To increase the limit (or to disable these "
         "warnings), see CodedInputStream::SetTotalBytesLimit() "
         "in third_party/protobuf/src/google/protobuf/io/coded_stream.h.";
}

bool CodedInputStream::SkipFallback(int count, int original_buffer_size) {
  if (buffer_size_after_limit_ > 0) {
    // We hit a limit inside this buffer.  Advance to the limit and fail.
    Advance(original_buffer_size);
    return false;
  }

  count -= original_buffer_size;
  buffer_ = NULL;
  buffer_end_ = buffer_;

  // Make sure this skip doesn't try to skip past the current limit.
  int closest_limit = std::min(current_limit_, total_bytes_limit_);
  int bytes_until_limit = closest_limit - total_bytes_read_;
  if (bytes_until_limit < count) {
    // We hit the limit.  Skip up to it then fail.
    if (bytes_until_limit > 0) {
      total_bytes_read_ = closest_limit;
      input_->Skip(bytes_until_limit);
    }
    return false;
  }

  if (!input_->Skip(count)) {
    total_bytes_read_ = input_->ByteCount();
    return false;
  }
  total_bytes_read_ += count;
  return true;
}

bool CodedInputStream::GetDirectBufferPointer(const void** data, int* size) {
  if (BufferSize() == 0 && !Refresh()) return false;

  *data = buffer_;
  *size = BufferSize();
  return true;
}

bool CodedInputStream::ReadRaw(void* buffer, int size) {
  int current_buffer_size;
  while ((current_buffer_size = BufferSize()) < size) {
    // Reading past end of buffer.  Copy what we have, then refresh.
    memcpy(buffer, buffer_, current_buffer_size);
    buffer = reinterpret_cast<uint8_t*>(buffer) + current_buffer_size;
    size -= current_buffer_size;
    Advance(current_buffer_size);
    if (!Refresh()) return false;
  }

  memcpy(buffer, buffer_, size);
  Advance(size);

  return true;
}

bool CodedInputStream::ReadString(std::string* buffer, int size) {
  if (size < 0) return false;  // security: size is often user-supplied

  if (BufferSize() >= size) {
    STLStringResizeUninitialized(buffer, size);
    std::pair<char*, bool> z = as_string_data(buffer);
    if (z.second) {
      // Oddly enough, memcpy() requires its first two args to be non-NULL even
      // if we copy 0 bytes.  So, we have ensured that z.first is non-NULL here.
      GOOGLE_DCHECK(z.first != NULL);
      memcpy(z.first, buffer_, size);
      Advance(size);
    }
    return true;
  }

  return ReadStringFallback(buffer, size);
}

bool CodedInputStream::ReadStringFallback(std::string* buffer, int size) {
  if (!buffer->empty()) {
    buffer->clear();
  }

  int closest_limit = std::min(current_limit_, total_bytes_limit_);
  if (closest_limit != INT_MAX) {
    int bytes_to_limit = closest_limit - CurrentPosition();
    if (bytes_to_limit > 0 && size > 0 && size <= bytes_to_limit) {
      buffer->reserve(size);
    }
  }

  int current_buffer_size;
  while ((current_buffer_size = BufferSize()) < size) {
    // Some STL implementations "helpfully" crash on buffer->append(NULL, 0).
    if (current_buffer_size != 0) {
      // Note:  string1.append(string2) is O(string2.size()) (as opposed to
      //   O(string1.size() + string2.size()), which would be bad).
      buffer->append(reinterpret_cast<const char*>(buffer_),
                     current_buffer_size);
    }
    size -= current_buffer_size;
    Advance(current_buffer_size);
    if (!Refresh()) return false;
  }

  buffer->append(reinterpret_cast<const char*>(buffer_), size);
  Advance(size);

  return true;
}


bool CodedInputStream::ReadLittleEndian32Fallback(uint32_t* value) {
  uint8_t bytes[sizeof(*value)];

  const uint8_t* ptr;
  if (BufferSize() >= static_cast<int64_t>(sizeof(*value))) {
    // Fast path:  Enough bytes in the buffer to read directly.
    ptr = buffer_;
    Advance(sizeof(*value));
  } else {
    // Slow path:  Had to read past the end of the buffer.
    if (!ReadRaw(bytes, sizeof(*value))) return false;
    ptr = bytes;
  }
  ReadLittleEndian32FromArray(ptr, value);
  return true;
}

bool CodedInputStream::ReadLittleEndian64Fallback(uint64_t* value) {
  uint8_t bytes[sizeof(*value)];

  const uint8_t* ptr;
  if (BufferSize() >= static_cast<int64_t>(sizeof(*value))) {
    // Fast path:  Enough bytes in the buffer to read directly.
    ptr = buffer_;
    Advance(sizeof(*value));
  } else {
    // Slow path:  Had to read past the end of the buffer.
    if (!ReadRaw(bytes, sizeof(*value))) return false;
    ptr = bytes;
  }
  ReadLittleEndian64FromArray(ptr, value);
  return true;
}

namespace {

// Decodes varint64 with known size, N, and returns next pointer. Knowing N at
// compile time, compiler can generate optimal code. For example, instead of
// subtracting 0x80 at each iteration, it subtracts properly shifted mask once.
template <size_t N>
const uint8_t* DecodeVarint64KnownSize(const uint8_t* buffer, uint64_t* value) {
  GOOGLE_DCHECK_GT(N, 0);
  uint64_t result = static_cast<uint64_t>(buffer[N - 1]) << (7 * (N - 1));
  for (size_t i = 0, offset = 0; i < N - 1; i++, offset += 7) {
    result += static_cast<uint64_t>(buffer[i] - 0x80) << offset;
  }
  *value = result;
  return buffer + N;
}

// Read a varint from the given buffer, write it to *value, and return a pair.
// The first part of the pair is true iff the read was successful.  The second
// part is buffer + (number of bytes read).  This function is always inlined,
// so returning a pair is costless.
PROTOBUF_ALWAYS_INLINE
::std::pair<bool, const uint8_t*> ReadVarint32FromArray(uint32_t first_byte,
                                                      const uint8_t* buffer,
                                                      uint32_t* value);
inline ::std::pair<bool, const uint8_t*> ReadVarint32FromArray(
    uint32_t first_byte, const uint8_t* buffer, uint32_t* value) {
  // Fast path:  We have enough bytes left in the buffer to guarantee that
  // this read won't cross the end, so we can skip the checks.
  GOOGLE_DCHECK_EQ(*buffer, first_byte);
  GOOGLE_DCHECK_EQ(first_byte & 0x80, 0x80) << first_byte;
  const uint8_t* ptr = buffer;
  uint32_t b;
  uint32_t result = first_byte - 0x80;
  ++ptr;  // We just processed the first byte.  Move on to the second.
  b = *(ptr++);
  result += b << 7;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 7;
  b = *(ptr++);
  result += b << 14;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 14;
  b = *(ptr++);
  result += b << 21;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 21;
  b = *(ptr++);
  result += b << 28;
  if (!(b & 0x80)) goto done;
  // "result -= 0x80 << 28" is irrevelant.

  // If the input is larger than 32 bits, we still need to read it all
  // and discard the high-order bits.
  for (int i = 0; i < kMaxVarintBytes - kMaxVarint32Bytes; i++) {
    b = *(ptr++);
    if (!(b & 0x80)) goto done;
  }

  // We have overrun the maximum size of a varint (10 bytes).  Assume
  // the data is corrupt.
  return std::make_pair(false, ptr);

done:
  *value = result;
  return std::make_pair(true, ptr);
}

PROTOBUF_ALWAYS_INLINE::std::pair<bool, const uint8_t*> ReadVarint64FromArray(
    const uint8_t* buffer, uint64_t* value);
inline ::std::pair<bool, const uint8_t*> ReadVarint64FromArray(
    const uint8_t* buffer, uint64_t* value) {
  // Assumes varint64 is at least 2 bytes.
  GOOGLE_DCHECK_GE(buffer[0], 128);

  const uint8_t* next;
  if (buffer[1] < 128) {
    next = DecodeVarint64KnownSize<2>(buffer, value);
  } else if (buffer[2] < 128) {
    next = DecodeVarint64KnownSize<3>(buffer, value);
  } else if (buffer[3] < 128) {
    next = DecodeVarint64KnownSize<4>(buffer, value);
  } else if (buffer[4] < 128) {
    next = DecodeVarint64KnownSize<5>(buffer, value);
  } else if (buffer[5] < 128) {
    next = DecodeVarint64KnownSize<6>(buffer, value);
  } else if (buffer[6] < 128) {
    next = DecodeVarint64KnownSize<7>(buffer, value);
  } else if (buffer[7] < 128) {
    next = DecodeVarint64KnownSize<8>(buffer, value);
  } else if (buffer[8] < 128) {
    next = DecodeVarint64KnownSize<9>(buffer, value);
  } else if (buffer[9] < 128) {
    next = DecodeVarint64KnownSize<10>(buffer, value);
  } else {
    // We have overrun the maximum size of a varint (10 bytes). Assume
    // the data is corrupt.
    return std::make_pair(false, buffer + 11);
  }

  return std::make_pair(true, next);
}

}  // namespace

bool CodedInputStream::ReadVarint32Slow(uint32_t* value) {
  // Directly invoke ReadVarint64Fallback, since we already tried to optimize
  // for one-byte varints.
  std::pair<uint64_t, bool> p = ReadVarint64Fallback();
  *value = static_cast<uint32_t>(p.first);
  return p.second;
}

int64_t CodedInputStream::ReadVarint32Fallback(uint32_t first_byte_or_zero) {
  if (BufferSize() >= kMaxVarintBytes ||
      // Optimization:  We're also safe if the buffer is non-empty and it ends
      // with a byte that would terminate a varint.
      (buffer_end_ > buffer_ && !(buffer_end_[-1] & 0x80))) {
    GOOGLE_DCHECK_NE(first_byte_or_zero, 0)
        << "Caller should provide us with *buffer_ when buffer is non-empty";
    uint32_t temp;
    ::std::pair<bool, const uint8_t*> p =
        ReadVarint32FromArray(first_byte_or_zero, buffer_, &temp);
    if (!p.first) return -1;
    buffer_ = p.second;
    return temp;
  } else {
    // Really slow case: we will incur the cost of an extra function call here,
    // but moving this out of line reduces the size of this function, which
    // improves the common case. In micro benchmarks, this is worth about 10-15%
    uint32_t temp;
    return ReadVarint32Slow(&temp) ? static_cast<int64_t>(temp) : -1;
  }
}

int CodedInputStream::ReadVarintSizeAsIntSlow() {
  // Directly invoke ReadVarint64Fallback, since we already tried to optimize
  // for one-byte varints.
  std::pair<uint64_t, bool> p = ReadVarint64Fallback();
  if (!p.second || p.first > static_cast<uint64_t>(INT_MAX)) return -1;
  return p.first;
}

int CodedInputStream::ReadVarintSizeAsIntFallback() {
  if (BufferSize() >= kMaxVarintBytes ||
      // Optimization:  We're also safe if the buffer is non-empty and it ends
      // with a byte that would terminate a varint.
      (buffer_end_ > buffer_ && !(buffer_end_[-1] & 0x80))) {
    uint64_t temp;
    ::std::pair<bool, const uint8_t*> p = ReadVarint64FromArray(buffer_, &temp);
    if (!p.first || temp > static_cast<uint64_t>(INT_MAX)) return -1;
    buffer_ = p.second;
    return temp;
  } else {
    // Really slow case: we will incur the cost of an extra function call here,
    // but moving this out of line reduces the size of this function, which
    // improves the common case. In micro benchmarks, this is worth about 10-15%
    return ReadVarintSizeAsIntSlow();
  }
}

uint32_t CodedInputStream::ReadTagSlow() {
  if (buffer_ == buffer_end_) {
    // Call refresh.
    if (!Refresh()) {
      // Refresh failed.  Make sure that it failed due to EOF, not because
      // we hit total_bytes_limit_, which, unlike normal limits, is not a
      // valid place to end a message.
      int current_position = total_bytes_read_ - buffer_size_after_limit_;
      if (current_position >= total_bytes_limit_) {
        // Hit total_bytes_limit_.  But if we also hit the normal limit,
        // we're still OK.
        legitimate_message_end_ = current_limit_ == total_bytes_limit_;
      } else {
        legitimate_message_end_ = true;
      }
      return 0;
    }
  }

  // For the slow path, just do a 64-bit read. Try to optimize for one-byte tags
  // again, since we have now refreshed the buffer.
  uint64_t result = 0;
  if (!ReadVarint64(&result)) return 0;
  return static_cast<uint32_t>(result);
}

uint32_t CodedInputStream::ReadTagFallback(uint32_t first_byte_or_zero) {
  const int buf_size = BufferSize();
  if (buf_size >= kMaxVarintBytes ||
      // Optimization:  We're also safe if the buffer is non-empty and it ends
      // with a byte that would terminate a varint.
      (buf_size > 0 && !(buffer_end_[-1] & 0x80))) {
    GOOGLE_DCHECK_EQ(first_byte_or_zero, buffer_[0]);
    if (first_byte_or_zero == 0) {
      ++buffer_;
      return 0;
    }
    uint32_t tag;
    ::std::pair<bool, const uint8_t*> p =
        ReadVarint32FromArray(first_byte_or_zero, buffer_, &tag);
    if (!p.first) {
      return 0;
    }
    buffer_ = p.second;
    return tag;
  } else {
    // We are commonly at a limit when attempting to read tags. Try to quickly
    // detect this case without making another function call.
    if ((buf_size == 0) &&
        ((buffer_size_after_limit_ > 0) ||
         (total_bytes_read_ == current_limit_)) &&
        // Make sure that the limit we hit is not total_bytes_limit_, since
        // in that case we still need to call Refresh() so that it prints an
        // error.
        total_bytes_read_ - buffer_size_after_limit_ < total_bytes_limit_) {
      // We hit a byte limit.
      legitimate_message_end_ = true;
      return 0;
    }
    return ReadTagSlow();
  }
}

bool CodedInputStream::ReadVarint64Slow(uint64_t* value) {
  // Slow path:  This read might cross the end of the buffer, so we
  // need to check and refresh the buffer if and when it does.

  uint64_t result = 0;
  int count = 0;
  uint32_t b;

  do {
    if (count == kMaxVarintBytes) {
      *value = 0;
      return false;
    }
    while (buffer_ == buffer_end_) {
      if (!Refresh()) {
        *value = 0;
        return false;
      }
    }
    b = *buffer_;
    result |= static_cast<uint64_t>(b & 0x7F) << (7 * count);
    Advance(1);
    ++count;
  } while (b & 0x80);

  *value = result;
  return true;
}

std::pair<uint64_t, bool> CodedInputStream::ReadVarint64Fallback() {
  if (BufferSize() >= kMaxVarintBytes ||
      // Optimization:  We're also safe if the buffer is non-empty and it ends
      // with a byte that would terminate a varint.
      (buffer_end_ > buffer_ && !(buffer_end_[-1] & 0x80))) {
    uint64_t temp;
    ::std::pair<bool, const uint8_t*> p = ReadVarint64FromArray(buffer_, &temp);
    if (!p.first) {
      return std::make_pair(0, false);
    }
    buffer_ = p.second;
    return std::make_pair(temp, true);
  } else {
    uint64_t temp;
    bool success = ReadVarint64Slow(&temp);
    return std::make_pair(temp, success);
  }
}

bool CodedInputStream::Refresh() {
  GOOGLE_DCHECK_EQ(0, BufferSize());

  if (buffer_size_after_limit_ > 0 || overflow_bytes_ > 0 ||
      total_bytes_read_ == current_limit_) {
    // We've hit a limit.  Stop.
    int current_position = total_bytes_read_ - buffer_size_after_limit_;

    if (current_position >= total_bytes_limit_ &&
        total_bytes_limit_ != current_limit_) {
      // Hit total_bytes_limit_.
      PrintTotalBytesLimitError();
    }

    return false;
  }

  const void* void_buffer;
  int buffer_size;
  if (NextNonEmpty(input_, &void_buffer, &buffer_size)) {
    buffer_ = reinterpret_cast<const uint8_t*>(void_buffer);
    buffer_end_ = buffer_ + buffer_size;
    GOOGLE_CHECK_GE(buffer_size, 0);

    if (total_bytes_read_ <= INT_MAX - buffer_size) {
      total_bytes_read_ += buffer_size;
    } else {
      // Overflow.  Reset buffer_end_ to not include the bytes beyond INT_MAX.
      // We can't get that far anyway, because total_bytes_limit_ is guaranteed
      // to be less than it.  We need to keep track of the number of bytes
      // we discarded, though, so that we can call input_->BackUp() to back
      // up over them on destruction.

      // The following line is equivalent to:
      //   overflow_bytes_ = total_bytes_read_ + buffer_size - INT_MAX;
      // except that it avoids overflows.  Signed integer overflow has
      // undefined results according to the C standard.
      overflow_bytes_ = total_bytes_read_ - (INT_MAX - buffer_size);
      buffer_end_ -= overflow_bytes_;
      total_bytes_read_ = INT_MAX;
    }

    RecomputeBufferLimits();
    return true;
  } else {
    buffer_ = NULL;
    buffer_end_ = NULL;
    return false;
  }
}

// CodedOutputStream =================================================

void EpsCopyOutputStream::EnableAliasing(bool enabled) {
  aliasing_enabled_ = enabled && stream_->AllowsAliasing();
}

int64_t EpsCopyOutputStream::ByteCount(uint8_t* ptr) const {
  // Calculate the current offset relative to the end of the stream buffer.
  int delta = (end_ - ptr) + (buffer_end_ ? 0 : kSlopBytes);
  return stream_->ByteCount() - delta;
}

// Flushes what's written out to the underlying ZeroCopyOutputStream buffers.
// Returns the size remaining in the buffer and sets buffer_end_ to the start
// of the remaining buffer, ie. [buffer_end_, buffer_end_ + return value)
int EpsCopyOutputStream::Flush(uint8_t* ptr) {
  while (buffer_end_ && ptr > end_) {
    int overrun = ptr - end_;
    GOOGLE_DCHECK(!had_error_);
    GOOGLE_DCHECK(overrun <= kSlopBytes);  // NOLINT
    ptr = Next() + overrun;
    if (had_error_) return 0;
  }
  int s;
  if (buffer_end_) {
    std::memcpy(buffer_end_, buffer_, ptr - buffer_);
    buffer_end_ += ptr - buffer_;
    s = end_ - ptr;
  } else {
    // The stream is writing directly in the ZeroCopyOutputStream buffer.
    s = end_ + kSlopBytes - ptr;
    buffer_end_ = ptr;
  }
  GOOGLE_DCHECK(s >= 0);  // NOLINT
  return s;
}

uint8_t* EpsCopyOutputStream::Trim(uint8_t* ptr) {
  if (had_error_) return ptr;
  int s = Flush(ptr);
  if (s) stream_->BackUp(s);
  // Reset to initial state (expecting new buffer)
  buffer_end_ = end_ = buffer_;
  return buffer_;
}


uint8_t* EpsCopyOutputStream::FlushAndResetBuffer(uint8_t* ptr) {
  if (had_error_) return buffer_;
  int s = Flush(ptr);
  if (had_error_) return buffer_;
  return SetInitialBuffer(buffer_end_, s);
}

bool EpsCopyOutputStream::Skip(int count, uint8_t** pp) {
  if (count < 0) return false;
  if (had_error_) {
    *pp = buffer_;
    return false;
  }
  int size = Flush(*pp);
  if (had_error_) {
    *pp = buffer_;
    return false;
  }
  void* data = buffer_end_;
  while (count > size) {
    count -= size;
    if (!stream_->Next(&data, &size)) {
      *pp = Error();
      return false;
    }
  }
  *pp = SetInitialBuffer(static_cast<uint8_t*>(data) + count, size - count);
  return true;
}

bool EpsCopyOutputStream::GetDirectBufferPointer(void** data, int* size,
                                                 uint8_t** pp) {
  if (had_error_) {
    *pp = buffer_;
    return false;
  }
  *size = Flush(*pp);
  if (had_error_) {
    *pp = buffer_;
    return false;
  }
  *data = buffer_end_;
  while (*size == 0) {
    if (!stream_->Next(data, size)) {
      *pp = Error();
      return false;
    }
  }
  *pp = SetInitialBuffer(*data, *size);
  return true;
}

uint8_t* EpsCopyOutputStream::GetDirectBufferForNBytesAndAdvance(int size,
                                                               uint8_t** pp) {
  if (had_error_) {
    *pp = buffer_;
    return nullptr;
  }
  int s = Flush(*pp);
  if (had_error_) {
    *pp = buffer_;
    return nullptr;
  }
  if (s >= size) {
    auto res = buffer_end_;
    *pp = SetInitialBuffer(buffer_end_ + size, s - size);
    return res;
  } else {
    *pp = SetInitialBuffer(buffer_end_, s);
    return nullptr;
  }
}

uint8_t* EpsCopyOutputStream::Next() {
  GOOGLE_DCHECK(!had_error_);  // NOLINT
  if (PROTOBUF_PREDICT_FALSE(stream_ == nullptr)) return Error();
  if (buffer_end_) {
    // We're in the patch buffer and need to fill up the previous buffer.
    std::memcpy(buffer_end_, buffer_, end_ - buffer_);
    uint8_t* ptr;
    int size;
    do {
      void* data;
      if (PROTOBUF_PREDICT_FALSE(!stream_->Next(&data, &size))) {
        // Stream has an error, we use the patch buffer to continue to be
        // able to write.
        return Error();
      }
      ptr = static_cast<uint8_t*>(data);
    } while (size == 0);
    if (PROTOBUF_PREDICT_TRUE(size > kSlopBytes)) {
      std::memcpy(ptr, end_, kSlopBytes);
      end_ = ptr + size - kSlopBytes;
      buffer_end_ = nullptr;
      return ptr;
    } else {
      GOOGLE_DCHECK(size > 0);  // NOLINT
      // Buffer to small
      std::memmove(buffer_, end_, kSlopBytes);
      buffer_end_ = ptr;
      end_ = buffer_ + size;
      return buffer_;
    }
  } else {
    std::memcpy(buffer_, end_, kSlopBytes);
    buffer_end_ = end_;
    end_ = buffer_ + kSlopBytes;
    return buffer_;
  }
}

uint8_t* EpsCopyOutputStream::EnsureSpaceFallback(uint8_t* ptr) {
  do {
    if (PROTOBUF_PREDICT_FALSE(had_error_)) return buffer_;
    int overrun = ptr - end_;
    GOOGLE_DCHECK(overrun >= 0);           // NOLINT
    GOOGLE_DCHECK(overrun <= kSlopBytes);  // NOLINT
    ptr = Next() + overrun;
  } while (ptr >= end_);
  GOOGLE_DCHECK(ptr < end_);  // NOLINT
  return ptr;
}

uint8_t* EpsCopyOutputStream::WriteRawFallback(const void* data, int size,
                                             uint8_t* ptr) {
  int s = GetSize(ptr);
  while (s < size) {
    std::memcpy(ptr, data, s);
    size -= s;
    data = static_cast<const uint8_t*>(data) + s;
    ptr = EnsureSpaceFallback(ptr + s);
    s = GetSize(ptr);
  }
  std::memcpy(ptr, data, size);
  return ptr + size;
}

uint8_t* EpsCopyOutputStream::WriteAliasedRaw(const void* data, int size,
                                            uint8_t* ptr) {
  if (size < GetSize(ptr)
  ) {
    return WriteRaw(data, size, ptr);
  } else {
    ptr = Trim(ptr);
    if (stream_->WriteAliasedRaw(data, size)) return ptr;
    return Error();
  }
}

#ifndef PROTOBUF_LITTLE_ENDIAN
uint8_t* EpsCopyOutputStream::WriteRawLittleEndian32(const void* data, int size,
                                                   uint8_t* ptr) {
  auto p = static_cast<const uint8_t*>(data);
  auto end = p + size;
  while (end - p >= kSlopBytes) {
    ptr = EnsureSpace(ptr);
    uint32_t buffer[4];
    static_assert(sizeof(buffer) == kSlopBytes, "Buffer must be kSlopBytes");
    std::memcpy(buffer, p, kSlopBytes);
    p += kSlopBytes;
    for (auto x : buffer)
      ptr = CodedOutputStream::WriteLittleEndian32ToArray(x, ptr);
  }
  while (p < end) {
    ptr = EnsureSpace(ptr);
    uint32_t buffer;
    std::memcpy(&buffer, p, 4);
    p += 4;
    ptr = CodedOutputStream::WriteLittleEndian32ToArray(buffer, ptr);
  }
  return ptr;
}

uint8_t* EpsCopyOutputStream::WriteRawLittleEndian64(const void* data, int size,
                                                   uint8_t* ptr) {
  auto p = static_cast<const uint8_t*>(data);
  auto end = p + size;
  while (end - p >= kSlopBytes) {
    ptr = EnsureSpace(ptr);
    uint64_t buffer[2];
    static_assert(sizeof(buffer) == kSlopBytes, "Buffer must be kSlopBytes");
    std::memcpy(buffer, p, kSlopBytes);
    p += kSlopBytes;
    for (auto x : buffer)
      ptr = CodedOutputStream::WriteLittleEndian64ToArray(x, ptr);
  }
  while (p < end) {
    ptr = EnsureSpace(ptr);
    uint64_t buffer;
    std::memcpy(&buffer, p, 8);
    p += 8;
    ptr = CodedOutputStream::WriteLittleEndian64ToArray(buffer, ptr);
  }
  return ptr;
}
#endif


uint8_t* EpsCopyOutputStream::WriteStringMaybeAliasedOutline(uint32_t num,
                                                           const std::string& s,
                                                           uint8_t* ptr) {
  ptr = EnsureSpace(ptr);
  uint32_t size = s.size();
  ptr = WriteLengthDelim(num, size, ptr);
  return WriteRawMaybeAliased(s.data(), size, ptr);
}

uint8_t* EpsCopyOutputStream::WriteStringOutline(uint32_t num, const std::string& s,
                                               uint8_t* ptr) {
  ptr = EnsureSpace(ptr);
  uint32_t size = s.size();
  ptr = WriteLengthDelim(num, size, ptr);
  return WriteRaw(s.data(), size, ptr);
}

std::atomic<bool> CodedOutputStream::default_serialization_deterministic_{
    false};

CodedOutputStream::CodedOutputStream(ZeroCopyOutputStream* stream,
                                     bool do_eager_refresh)
    : impl_(stream, IsDefaultSerializationDeterministic(), &cur_),
      start_count_(stream->ByteCount()) {
  if (do_eager_refresh) {
    void* data;
    int size;
    if (!stream->Next(&data, &size) || size == 0) return;
    cur_ = impl_.SetInitialBuffer(data, size);
  }
}

CodedOutputStream::~CodedOutputStream() { Trim(); }


uint8_t* CodedOutputStream::WriteStringWithSizeToArray(const std::string& str,
                                                     uint8_t* target) {
  GOOGLE_DCHECK_LE(str.size(), std::numeric_limits<uint32_t>::max());
  target = WriteVarint32ToArray(str.size(), target);
  return WriteStringToArray(str, target);
}

uint8_t* CodedOutputStream::WriteVarint32ToArrayOutOfLineHelper(uint32_t value,
                                                              uint8_t* target) {
  GOOGLE_DCHECK_GE(value, 0x80);
  target[0] |= static_cast<uint8_t>(0x80);
  value >>= 7;
  target[1] = static_cast<uint8_t>(value);
  if (value < 0x80) {
    return target + 2;
  }
  target += 2;
  do {
    // Turn on continuation bit in the byte we just wrote.
    target[-1] |= static_cast<uint8_t>(0x80);
    value >>= 7;
    *target = static_cast<uint8_t>(value);
    ++target;
  } while (value >= 0x80);
  return target;
}

}  // namespace io
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>
