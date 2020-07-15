// Adapted from the patch of kenton@google.com (Kenton Varda)
// See https://github.com/google/protobuf/pull/710 for details.

#ifndef GOOGLE_PROTOBUF_UTIL_DELIMITED_MESSAGE_UTIL_H__
#define GOOGLE_PROTOBUF_UTIL_DELIMITED_MESSAGE_UTIL_H__

#include <ostream>

#include <google/protobuf/message_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace google {
namespace protobuf {
namespace util {

// Write a single size-delimited message from the given stream. Delimited
// format allows a single file or stream to contain multiple messages,
// whereas normally writing multiple non-delimited messages to the same
// stream would cause them to be merged. A delimited message is a varint
// encoding the message size followed by a message of exactly that size.
//
// Note that if you want to *read* a delimited message from a file descriptor
// or istream, you will need to construct an io::FileInputStream or
// io::OstreamInputStream (implementations of io::ZeroCopyStream) and use the
// utility function ParseDelimitedFromZeroCopyStream(). You must then
// continue to use the same ZeroCopyInputStream to read all further data from
// the stream until EOF. This is because these ZeroCopyInputStream
// implementations are buffered: they read a big chunk of data at a time,
// then parse it. As a result, they may read past the end of the delimited
// message. There is no way for them to push the extra data back into the
// underlying source, so instead you must keep using the same stream object.
bool LIBPROTOBUF_EXPORT SerializeDelimitedToFileDescriptor(const MessageLite& message, int file_descriptor);

bool LIBPROTOBUF_EXPORT SerializeDelimitedToOstream(const MessageLite& message, ostream* output);

// Read a single size-delimited message from the given stream. Delimited
// format allows a single file or stream to contain multiple messages,
// whereas normally parsing consumes the entire input. A delimited message
// is a varint encoding the message size followed by a message of exactly
// that size.
//
// If |clean_eof| is not NULL, then it will be set to indicate whether the
// stream ended cleanly. That is, if the stream ends without this method
// having read any data at all from it, then *clean_eof will be set true,
// otherwise it will be set false. Note that these methods return false
// on EOF, but they also return false on other errors, so |clean_eof| is
// needed to distinguish a clean end from errors.
bool LIBPROTOBUF_EXPORT ParseDelimitedFromZeroCopyStream(MessageLite* message, io::ZeroCopyInputStream* input, bool* clean_eof);

bool LIBPROTOBUF_EXPORT ParseDelimitedFromCodedStream(MessageLite* message, io::CodedInputStream* input, bool* clean_eof);

// Write a single size-delimited message from the given stream. Delimited
// format allows a single file or stream to contain multiple messages,
// whereas normally writing multiple non-delimited messages to the same
// stream would cause them to be merged. A delimited message is a varint
// encoding the message size followed by a message of exactly that size.
bool LIBPROTOBUF_EXPORT SerializeDelimitedToZeroCopyStream(const MessageLite& message, io::ZeroCopyOutputStream* output);

bool LIBPROTOBUF_EXPORT SerializeDelimitedToCodedStream(const MessageLite& message, io::CodedOutputStream* output);

}  // namespace util
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_UTIL_DELIMITED_MESSAGE_UTIL_H__
