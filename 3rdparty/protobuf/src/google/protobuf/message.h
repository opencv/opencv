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
// Defines Message, the abstract interface implemented by non-lite
// protocol message objects.  Although it's possible to implement this
// interface manually, most users will use the protocol compiler to
// generate implementations.
//
// Example usage:
//
// Say you have a message defined as:
//
//   message Foo {
//     optional string text = 1;
//     repeated int32 numbers = 2;
//   }
//
// Then, if you used the protocol compiler to generate a class from the above
// definition, you could use it like so:
//
//   std::string data;  // Will store a serialized version of the message.
//
//   {
//     // Create a message and serialize it.
//     Foo foo;
//     foo.set_text("Hello World!");
//     foo.add_numbers(1);
//     foo.add_numbers(5);
//     foo.add_numbers(42);
//
//     foo.SerializeToString(&data);
//   }
//
//   {
//     // Parse the serialized message and check that it contains the
//     // correct data.
//     Foo foo;
//     foo.ParseFromString(data);
//
//     assert(foo.text() == "Hello World!");
//     assert(foo.numbers_size() == 3);
//     assert(foo.numbers(0) == 1);
//     assert(foo.numbers(1) == 5);
//     assert(foo.numbers(2) == 42);
//   }
//
//   {
//     // Same as the last block, but do it dynamically via the Message
//     // reflection interface.
//     Message* foo = new Foo;
//     const Descriptor* descriptor = foo->GetDescriptor();
//
//     // Get the descriptors for the fields we're interested in and verify
//     // their types.
//     const FieldDescriptor* text_field = descriptor->FindFieldByName("text");
//     assert(text_field != nullptr);
//     assert(text_field->type() == FieldDescriptor::TYPE_STRING);
//     assert(text_field->label() == FieldDescriptor::LABEL_OPTIONAL);
//     const FieldDescriptor* numbers_field = descriptor->
//                                            FindFieldByName("numbers");
//     assert(numbers_field != nullptr);
//     assert(numbers_field->type() == FieldDescriptor::TYPE_INT32);
//     assert(numbers_field->label() == FieldDescriptor::LABEL_REPEATED);
//
//     // Parse the message.
//     foo->ParseFromString(data);
//
//     // Use the reflection interface to examine the contents.
//     const Reflection* reflection = foo->GetReflection();
//     assert(reflection->GetString(*foo, text_field) == "Hello World!");
//     assert(reflection->FieldSize(*foo, numbers_field) == 3);
//     assert(reflection->GetRepeatedInt32(*foo, numbers_field, 0) == 1);
//     assert(reflection->GetRepeatedInt32(*foo, numbers_field, 1) == 5);
//     assert(reflection->GetRepeatedInt32(*foo, numbers_field, 2) == 42);
//
//     delete foo;
//   }

#ifndef GOOGLE_PROTOBUF_MESSAGE_H__
#define GOOGLE_PROTOBUF_MESSAGE_H__

#include <iosfwd>
#include <string>
#include <type_traits>
#include <vector>

#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/port.h>


#define GOOGLE_PROTOBUF_HAS_ONEOF
#define GOOGLE_PROTOBUF_HAS_ARENAS

#include <google/protobuf/port_def.inc>

#ifdef SWIG
#error "You cannot SWIG proto headers"
#endif

namespace google {
namespace protobuf {

// Defined in this file.
class Message;
class Reflection;
class MessageFactory;

// Defined in other files.
class AssignDescriptorsHelper;
class DynamicMessageFactory;
class DynamicMessageReflectionHelper;
class GeneratedMessageReflectionTestHelper;
class MapKey;
class MapValueConstRef;
class MapValueRef;
class MapIterator;
class MapReflectionTester;

namespace internal {
struct DescriptorTable;
class MapFieldBase;
class SwapFieldHelper;
class CachedSize;
}  // namespace internal
class UnknownFieldSet;  // unknown_field_set.h
namespace io {
class ZeroCopyInputStream;   // zero_copy_stream.h
class ZeroCopyOutputStream;  // zero_copy_stream.h
class CodedInputStream;      // coded_stream.h
class CodedOutputStream;     // coded_stream.h
}  // namespace io
namespace python {
class MapReflectionFriend;  // scalar_map_container.h
class MessageReflectionFriend;
}  // namespace python
namespace expr {
class CelMapReflectionFriend;  // field_backed_map_impl.cc
}

namespace internal {
class MapFieldPrinterHelper;  // text_format.cc
}
namespace util {
class MessageDifferencer;
}


namespace internal {
class ReflectionAccessor;      // message.cc
class ReflectionOps;           // reflection_ops.h
class MapKeySorter;            // wire_format.cc
class WireFormat;              // wire_format.h
class MapFieldReflectionTest;  // map_test.cc
}  // namespace internal

template <typename T>
class RepeatedField;  // repeated_field.h

template <typename T>
class RepeatedPtrField;  // repeated_field.h

// A container to hold message metadata.
struct Metadata {
  const Descriptor* descriptor;
  const Reflection* reflection;
};

namespace internal {
template <class To>
inline To* GetPointerAtOffset(Message* message, uint32_t offset) {
  return reinterpret_cast<To*>(reinterpret_cast<char*>(message) + offset);
}

template <class To>
const To* GetConstPointerAtOffset(const Message* message, uint32_t offset) {
  return reinterpret_cast<const To*>(reinterpret_cast<const char*>(message) +
                                     offset);
}

template <class To>
const To& GetConstRefAtOffset(const Message& message, uint32_t offset) {
  return *GetConstPointerAtOffset<To>(&message, offset);
}

bool CreateUnknownEnumValues(const FieldDescriptor* field);
}  // namespace internal

// Abstract interface for protocol messages.
//
// See also MessageLite, which contains most every-day operations.  Message
// adds descriptors and reflection on top of that.
//
// The methods of this class that are virtual but not pure-virtual have
// default implementations based on reflection.  Message classes which are
// optimized for speed will want to override these with faster implementations,
// but classes optimized for code size may be happy with keeping them.  See
// the optimize_for option in descriptor.proto.
//
// Users must not derive from this class. Only the protocol compiler and
// the internal library are allowed to create subclasses.
class PROTOBUF_EXPORT Message : public MessageLite {
 public:
  constexpr Message() {}

  // Basic Operations ------------------------------------------------

  // Construct a new instance of the same type.  Ownership is passed to the
  // caller.  (This is also defined in MessageLite, but is defined again here
  // for return-type covariance.)
  Message* New() const { return New(nullptr); }

  // Construct a new instance on the arena. Ownership is passed to the caller
  // if arena is a nullptr.
  Message* New(Arena* arena) const override = 0;

  // Make this message into a copy of the given message.  The given message
  // must have the same descriptor, but need not necessarily be the same class.
  // By default this is just implemented as "Clear(); MergeFrom(from);".
  virtual void CopyFrom(const Message& from);

  // Merge the fields from the given message into this message.  Singular
  // fields will be overwritten, if specified in from, except for embedded
  // messages which will be merged.  Repeated fields will be concatenated.
  // The given message must be of the same type as this message (i.e. the
  // exact same class).
  virtual void MergeFrom(const Message& from);

  // Verifies that IsInitialized() returns true.  GOOGLE_CHECK-fails otherwise, with
  // a nice error message.
  void CheckInitialized() const;

  // Slowly build a list of all required fields that are not set.
  // This is much, much slower than IsInitialized() as it is implemented
  // purely via reflection.  Generally, you should not call this unless you
  // have already determined that an error exists by calling IsInitialized().
  void FindInitializationErrors(std::vector<std::string>* errors) const;

  // Like FindInitializationErrors, but joins all the strings, delimited by
  // commas, and returns them.
  std::string InitializationErrorString() const override;

  // Clears all unknown fields from this message and all embedded messages.
  // Normally, if unknown tag numbers are encountered when parsing a message,
  // the tag and value are stored in the message's UnknownFieldSet and
  // then written back out when the message is serialized.  This allows servers
  // which simply route messages to other servers to pass through messages
  // that have new field definitions which they don't yet know about.  However,
  // this behavior can have security implications.  To avoid it, call this
  // method after parsing.
  //
  // See Reflection::GetUnknownFields() for more on unknown fields.
  void DiscardUnknownFields();

  // Computes (an estimate of) the total number of bytes currently used for
  // storing the message in memory.  The default implementation calls the
  // Reflection object's SpaceUsed() method.
  //
  // SpaceUsed() is noticeably slower than ByteSize(), as it is implemented
  // using reflection (rather than the generated code implementation for
  // ByteSize()). Like ByteSize(), its CPU time is linear in the number of
  // fields defined for the proto.
  virtual size_t SpaceUsedLong() const;

  PROTOBUF_DEPRECATED_MSG("Please use SpaceUsedLong() instead")
  int SpaceUsed() const { return internal::ToIntSize(SpaceUsedLong()); }

  // Debugging & Testing----------------------------------------------

  // Generates a human readable form of this message, useful for debugging
  // and other purposes.
  std::string DebugString() const;
  // Like DebugString(), but with less whitespace.
  std::string ShortDebugString() const;
  // Like DebugString(), but do not escape UTF-8 byte sequences.
  std::string Utf8DebugString() const;
  // Convenience function useful in GDB.  Prints DebugString() to stdout.
  void PrintDebugString() const;

  // Reflection-based methods ----------------------------------------
  // These methods are pure-virtual in MessageLite, but Message provides
  // reflection-based default implementations.

  std::string GetTypeName() const override;
  void Clear() override;

  // Returns whether all required fields have been set. Note that required
  // fields no longer exist starting in proto3.
  bool IsInitialized() const override;

  void CheckTypeAndMergeFrom(const MessageLite& other) override;
  // Reflective parser
  const char* _InternalParse(const char* ptr,
                             internal::ParseContext* ctx) override;
  size_t ByteSizeLong() const override;
  uint8_t* _InternalSerialize(uint8_t* target,
                              io::EpsCopyOutputStream* stream) const override;

 private:
  // This is called only by the default implementation of ByteSize(), to
  // update the cached size.  If you override ByteSize(), you do not need
  // to override this.  If you do not override ByteSize(), you MUST override
  // this; the default implementation will crash.
  //
  // The method is private because subclasses should never call it; only
  // override it.  Yes, C++ lets you do that.  Crazy, huh?
  virtual void SetCachedSize(int size) const;

 public:
  // Introspection ---------------------------------------------------


  // Get a non-owning pointer to a Descriptor for this message's type.  This
  // describes what fields the message contains, the types of those fields, etc.
  // This object remains property of the Message.
  const Descriptor* GetDescriptor() const { return GetMetadata().descriptor; }

  // Get a non-owning pointer to the Reflection interface for this Message,
  // which can be used to read and modify the fields of the Message dynamically
  // (in other words, without knowing the message type at compile time).  This
  // object remains property of the Message.
  const Reflection* GetReflection() const { return GetMetadata().reflection; }

 protected:
  // Get a struct containing the metadata for the Message, which is used in turn
  // to implement GetDescriptor() and GetReflection() above.
  virtual Metadata GetMetadata() const = 0;

  struct ClassData {
    // Note: The order of arguments (to, then from) is chosen so that the ABI
    // of this function is the same as the CopyFrom method.  That is, the
    // hidden "this" parameter comes first.
    void (*copy_to_from)(Message* to, const Message& from_msg);
    void (*merge_to_from)(Message* to, const Message& from_msg);
  };
  // GetClassData() returns a pointer to a ClassData struct which
  // exists in global memory and is unique to each subclass.  This uniqueness
  // property is used in order to quickly determine whether two messages are
  // of the same type.
  // TODO(jorg): change to pure virtual
  virtual const ClassData* GetClassData() const { return nullptr; }

  // CopyWithSizeCheck calls Clear() and then MergeFrom(), and in debug
  // builds, checks that calling Clear() on the destination message doesn't
  // alter the size of the source.  It assumes the messages are known to be
  // of the same type, and thus uses GetClassData().
  static void CopyWithSizeCheck(Message* to, const Message& from);

  inline explicit Message(Arena* arena, bool is_message_owned = false)
      : MessageLite(arena, is_message_owned) {}
  size_t ComputeUnknownFieldsSize(size_t total_size,
                                  internal::CachedSize* cached_size) const;
  size_t MaybeComputeUnknownFieldsSize(size_t total_size,
                                       internal::CachedSize* cached_size) const;


 protected:
  static uint64_t GetInvariantPerBuild(uint64_t salt);

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Message);
};

namespace internal {
// Forward-declare interfaces used to implement RepeatedFieldRef.
// These are protobuf internals that users shouldn't care about.
class RepeatedFieldAccessor;
}  // namespace internal

// Forward-declare RepeatedFieldRef templates. The second type parameter is
// used for SFINAE tricks. Users should ignore it.
template <typename T, typename Enable = void>
class RepeatedFieldRef;

template <typename T, typename Enable = void>
class MutableRepeatedFieldRef;

// This interface contains methods that can be used to dynamically access
// and modify the fields of a protocol message.  Their semantics are
// similar to the accessors the protocol compiler generates.
//
// To get the Reflection for a given Message, call Message::GetReflection().
//
// This interface is separate from Message only for efficiency reasons;
// the vast majority of implementations of Message will share the same
// implementation of Reflection (GeneratedMessageReflection,
// defined in generated_message.h), and all Messages of a particular class
// should share the same Reflection object (though you should not rely on
// the latter fact).
//
// There are several ways that these methods can be used incorrectly.  For
// example, any of the following conditions will lead to undefined
// results (probably assertion failures):
// - The FieldDescriptor is not a field of this message type.
// - The method called is not appropriate for the field's type.  For
//   each field type in FieldDescriptor::TYPE_*, there is only one
//   Get*() method, one Set*() method, and one Add*() method that is
//   valid for that type.  It should be obvious which (except maybe
//   for TYPE_BYTES, which are represented using strings in C++).
// - A Get*() or Set*() method for singular fields is called on a repeated
//   field.
// - GetRepeated*(), SetRepeated*(), or Add*() is called on a non-repeated
//   field.
// - The Message object passed to any method is not of the right type for
//   this Reflection object (i.e. message.GetReflection() != reflection).
//
// You might wonder why there is not any abstract representation for a field
// of arbitrary type.  E.g., why isn't there just a "GetField()" method that
// returns "const Field&", where "Field" is some class with accessors like
// "GetInt32Value()".  The problem is that someone would have to deal with
// allocating these Field objects.  For generated message classes, having to
// allocate space for an additional object to wrap every field would at least
// double the message's memory footprint, probably worse.  Allocating the
// objects on-demand, on the other hand, would be expensive and prone to
// memory leaks.  So, instead we ended up with this flat interface.
class PROTOBUF_EXPORT Reflection final {
 public:
  // Get the UnknownFieldSet for the message.  This contains fields which
  // were seen when the Message was parsed but were not recognized according
  // to the Message's definition.
  const UnknownFieldSet& GetUnknownFields(const Message& message) const;
  // Get a mutable pointer to the UnknownFieldSet for the message.  This
  // contains fields which were seen when the Message was parsed but were not
  // recognized according to the Message's definition.
  UnknownFieldSet* MutableUnknownFields(Message* message) const;

  // Estimate the amount of memory used by the message object.
  size_t SpaceUsedLong(const Message& message) const;

  PROTOBUF_DEPRECATED_MSG("Please use SpaceUsedLong() instead")
  int SpaceUsed(const Message& message) const {
    return internal::ToIntSize(SpaceUsedLong(message));
  }

  // Check if the given non-repeated field is set.
  bool HasField(const Message& message, const FieldDescriptor* field) const;

  // Get the number of elements of a repeated field.
  int FieldSize(const Message& message, const FieldDescriptor* field) const;

  // Clear the value of a field, so that HasField() returns false or
  // FieldSize() returns zero.
  void ClearField(Message* message, const FieldDescriptor* field) const;

  // Check if the oneof is set. Returns true if any field in oneof
  // is set, false otherwise.
  bool HasOneof(const Message& message,
                const OneofDescriptor* oneof_descriptor) const;

  void ClearOneof(Message* message,
                  const OneofDescriptor* oneof_descriptor) const;

  // Returns the field descriptor if the oneof is set. nullptr otherwise.
  const FieldDescriptor* GetOneofFieldDescriptor(
      const Message& message, const OneofDescriptor* oneof_descriptor) const;

  // Removes the last element of a repeated field.
  // We don't provide a way to remove any element other than the last
  // because it invites inefficient use, such as O(n^2) filtering loops
  // that should have been O(n).  If you want to remove an element other
  // than the last, the best way to do it is to re-arrange the elements
  // (using Swap()) so that the one you want removed is at the end, then
  // call RemoveLast().
  void RemoveLast(Message* message, const FieldDescriptor* field) const;
  // Removes the last element of a repeated message field, and returns the
  // pointer to the caller.  Caller takes ownership of the returned pointer.
  PROTOBUF_NODISCARD Message* ReleaseLast(Message* message,
                                          const FieldDescriptor* field) const;

  // Similar to ReleaseLast() without internal safety and ownershp checks. This
  // method should only be used when the objects are on the same arena or paired
  // with a call to `UnsafeArenaAddAllocatedMessage`.
  Message* UnsafeArenaReleaseLast(Message* message,
                                  const FieldDescriptor* field) const;

  // Swap the complete contents of two messages.
  void Swap(Message* message1, Message* message2) const;

  // Swap fields listed in fields vector of two messages.
  void SwapFields(Message* message1, Message* message2,
                  const std::vector<const FieldDescriptor*>& fields) const;

  // Swap two elements of a repeated field.
  void SwapElements(Message* message, const FieldDescriptor* field, int index1,
                    int index2) const;

  // Swap without internal safety and ownership checks. This method should only
  // be used when the objects are on the same arena.
  void UnsafeArenaSwap(Message* lhs, Message* rhs) const;

  // SwapFields without internal safety and ownership checks. This method should
  // only be used when the objects are on the same arena.
  void UnsafeArenaSwapFields(
      Message* lhs, Message* rhs,
      const std::vector<const FieldDescriptor*>& fields) const;

  // List all fields of the message which are currently set, except for unknown
  // fields, but including extension known to the parser (i.e. compiled in).
  // Singular fields will only be listed if HasField(field) would return true
  // and repeated fields will only be listed if FieldSize(field) would return
  // non-zero.  Fields (both normal fields and extension fields) will be listed
  // ordered by field number.
  // Use Reflection::GetUnknownFields() or message.unknown_fields() to also get
  // access to fields/extensions unknown to the parser.
  void ListFields(const Message& message,
                  std::vector<const FieldDescriptor*>* output) const;

  // Singular field getters ------------------------------------------
  // These get the value of a non-repeated field.  They return the default
  // value for fields that aren't set.

  int32_t GetInt32(const Message& message, const FieldDescriptor* field) const;
  int64_t GetInt64(const Message& message, const FieldDescriptor* field) const;
  uint32_t GetUInt32(const Message& message,
                     const FieldDescriptor* field) const;
  uint64_t GetUInt64(const Message& message,
                     const FieldDescriptor* field) const;
  float GetFloat(const Message& message, const FieldDescriptor* field) const;
  double GetDouble(const Message& message, const FieldDescriptor* field) const;
  bool GetBool(const Message& message, const FieldDescriptor* field) const;
  std::string GetString(const Message& message,
                        const FieldDescriptor* field) const;
  const EnumValueDescriptor* GetEnum(const Message& message,
                                     const FieldDescriptor* field) const;

  // GetEnumValue() returns an enum field's value as an integer rather than
  // an EnumValueDescriptor*. If the integer value does not correspond to a
  // known value descriptor, a new value descriptor is created. (Such a value
  // will only be present when the new unknown-enum-value semantics are enabled
  // for a message.)
  int GetEnumValue(const Message& message, const FieldDescriptor* field) const;

  // See MutableMessage() for the meaning of the "factory" parameter.
  const Message& GetMessage(const Message& message,
                            const FieldDescriptor* field,
                            MessageFactory* factory = nullptr) const;

  // Get a string value without copying, if possible.
  //
  // GetString() necessarily returns a copy of the string.  This can be
  // inefficient when the std::string is already stored in a std::string object
  // in the underlying message.  GetStringReference() will return a reference to
  // the underlying std::string in this case.  Otherwise, it will copy the
  // string into *scratch and return that.
  //
  // Note:  It is perfectly reasonable and useful to write code like:
  //     str = reflection->GetStringReference(message, field, &str);
  //   This line would ensure that only one copy of the string is made
  //   regardless of the field's underlying representation.  When initializing
  //   a newly-constructed string, though, it's just as fast and more
  //   readable to use code like:
  //     std::string str = reflection->GetString(message, field);
  const std::string& GetStringReference(const Message& message,
                                        const FieldDescriptor* field,
                                        std::string* scratch) const;


  // Singular field mutators -----------------------------------------
  // These mutate the value of a non-repeated field.

  void SetInt32(Message* message, const FieldDescriptor* field,
                int32_t value) const;
  void SetInt64(Message* message, const FieldDescriptor* field,
                int64_t value) const;
  void SetUInt32(Message* message, const FieldDescriptor* field,
                 uint32_t value) const;
  void SetUInt64(Message* message, const FieldDescriptor* field,
                 uint64_t value) const;
  void SetFloat(Message* message, const FieldDescriptor* field,
                float value) const;
  void SetDouble(Message* message, const FieldDescriptor* field,
                 double value) const;
  void SetBool(Message* message, const FieldDescriptor* field,
               bool value) const;
  void SetString(Message* message, const FieldDescriptor* field,
                 std::string value) const;
  void SetEnum(Message* message, const FieldDescriptor* field,
               const EnumValueDescriptor* value) const;
  // Set an enum field's value with an integer rather than EnumValueDescriptor.
  // For proto3 this is just setting the enum field to the value specified, for
  // proto2 it's more complicated. If value is a known enum value the field is
  // set as usual. If the value is unknown then it is added to the unknown field
  // set. Note this matches the behavior of parsing unknown enum values.
  // If multiple calls with unknown values happen than they are all added to the
  // unknown field set in order of the calls.
  void SetEnumValue(Message* message, const FieldDescriptor* field,
                    int value) const;

  // Get a mutable pointer to a field with a message type.  If a MessageFactory
  // is provided, it will be used to construct instances of the sub-message;
  // otherwise, the default factory is used.  If the field is an extension that
  // does not live in the same pool as the containing message's descriptor (e.g.
  // it lives in an overlay pool), then a MessageFactory must be provided.
  // If you have no idea what that meant, then you probably don't need to worry
  // about it (don't provide a MessageFactory).  WARNING:  If the
  // FieldDescriptor is for a compiled-in extension, then
  // factory->GetPrototype(field->message_type()) MUST return an instance of
  // the compiled-in class for this type, NOT DynamicMessage.
  Message* MutableMessage(Message* message, const FieldDescriptor* field,
                          MessageFactory* factory = nullptr) const;

  // Replaces the message specified by 'field' with the already-allocated object
  // sub_message, passing ownership to the message.  If the field contained a
  // message, that message is deleted.  If sub_message is nullptr, the field is
  // cleared.
  void SetAllocatedMessage(Message* message, Message* sub_message,
                           const FieldDescriptor* field) const;

  // Similar to `SetAllocatedMessage`, but omits all internal safety and
  // ownership checks.  This method should only be used when the objects are on
  // the same arena or paired with a call to `UnsafeArenaReleaseMessage`.
  void UnsafeArenaSetAllocatedMessage(Message* message, Message* sub_message,
                                      const FieldDescriptor* field) const;

  // Releases the message specified by 'field' and returns the pointer,
  // ReleaseMessage() will return the message the message object if it exists.
  // Otherwise, it may or may not return nullptr.  In any case, if the return
  // value is non-null, the caller takes ownership of the pointer.
  // If the field existed (HasField() is true), then the returned pointer will
  // be the same as the pointer returned by MutableMessage().
  // This function has the same effect as ClearField().
  PROTOBUF_NODISCARD Message* ReleaseMessage(
      Message* message, const FieldDescriptor* field,
      MessageFactory* factory = nullptr) const;

  // Similar to `ReleaseMessage`, but omits all internal safety and ownership
  // checks.  This method should only be used when the objects are on the same
  // arena or paired with a call to `UnsafeArenaSetAllocatedMessage`.
  Message* UnsafeArenaReleaseMessage(Message* message,
                                     const FieldDescriptor* field,
                                     MessageFactory* factory = nullptr) const;


  // Repeated field getters ------------------------------------------
  // These get the value of one element of a repeated field.

  int32_t GetRepeatedInt32(const Message& message, const FieldDescriptor* field,
                           int index) const;
  int64_t GetRepeatedInt64(const Message& message, const FieldDescriptor* field,
                           int index) const;
  uint32_t GetRepeatedUInt32(const Message& message,
                             const FieldDescriptor* field, int index) const;
  uint64_t GetRepeatedUInt64(const Message& message,
                             const FieldDescriptor* field, int index) const;
  float GetRepeatedFloat(const Message& message, const FieldDescriptor* field,
                         int index) const;
  double GetRepeatedDouble(const Message& message, const FieldDescriptor* field,
                           int index) const;
  bool GetRepeatedBool(const Message& message, const FieldDescriptor* field,
                       int index) const;
  std::string GetRepeatedString(const Message& message,
                                const FieldDescriptor* field, int index) const;
  const EnumValueDescriptor* GetRepeatedEnum(const Message& message,
                                             const FieldDescriptor* field,
                                             int index) const;
  // GetRepeatedEnumValue() returns an enum field's value as an integer rather
  // than an EnumValueDescriptor*. If the integer value does not correspond to a
  // known value descriptor, a new value descriptor is created. (Such a value
  // will only be present when the new unknown-enum-value semantics are enabled
  // for a message.)
  int GetRepeatedEnumValue(const Message& message, const FieldDescriptor* field,
                           int index) const;
  const Message& GetRepeatedMessage(const Message& message,
                                    const FieldDescriptor* field,
                                    int index) const;

  // See GetStringReference(), above.
  const std::string& GetRepeatedStringReference(const Message& message,
                                                const FieldDescriptor* field,
                                                int index,
                                                std::string* scratch) const;


  // Repeated field mutators -----------------------------------------
  // These mutate the value of one element of a repeated field.

  void SetRepeatedInt32(Message* message, const FieldDescriptor* field,
                        int index, int32_t value) const;
  void SetRepeatedInt64(Message* message, const FieldDescriptor* field,
                        int index, int64_t value) const;
  void SetRepeatedUInt32(Message* message, const FieldDescriptor* field,
                         int index, uint32_t value) const;
  void SetRepeatedUInt64(Message* message, const FieldDescriptor* field,
                         int index, uint64_t value) const;
  void SetRepeatedFloat(Message* message, const FieldDescriptor* field,
                        int index, float value) const;
  void SetRepeatedDouble(Message* message, const FieldDescriptor* field,
                         int index, double value) const;
  void SetRepeatedBool(Message* message, const FieldDescriptor* field,
                       int index, bool value) const;
  void SetRepeatedString(Message* message, const FieldDescriptor* field,
                         int index, std::string value) const;
  void SetRepeatedEnum(Message* message, const FieldDescriptor* field,
                       int index, const EnumValueDescriptor* value) const;
  // Set an enum field's value with an integer rather than EnumValueDescriptor.
  // For proto3 this is just setting the enum field to the value specified, for
  // proto2 it's more complicated. If value is a known enum value the field is
  // set as usual. If the value is unknown then it is added to the unknown field
  // set. Note this matches the behavior of parsing unknown enum values.
  // If multiple calls with unknown values happen than they are all added to the
  // unknown field set in order of the calls.
  void SetRepeatedEnumValue(Message* message, const FieldDescriptor* field,
                            int index, int value) const;
  // Get a mutable pointer to an element of a repeated field with a message
  // type.
  Message* MutableRepeatedMessage(Message* message,
                                  const FieldDescriptor* field,
                                  int index) const;


  // Repeated field adders -------------------------------------------
  // These add an element to a repeated field.

  void AddInt32(Message* message, const FieldDescriptor* field,
                int32_t value) const;
  void AddInt64(Message* message, const FieldDescriptor* field,
                int64_t value) const;
  void AddUInt32(Message* message, const FieldDescriptor* field,
                 uint32_t value) const;
  void AddUInt64(Message* message, const FieldDescriptor* field,
                 uint64_t value) const;
  void AddFloat(Message* message, const FieldDescriptor* field,
                float value) const;
  void AddDouble(Message* message, const FieldDescriptor* field,
                 double value) const;
  void AddBool(Message* message, const FieldDescriptor* field,
               bool value) const;
  void AddString(Message* message, const FieldDescriptor* field,
                 std::string value) const;
  void AddEnum(Message* message, const FieldDescriptor* field,
               const EnumValueDescriptor* value) const;
  // Add an integer value to a repeated enum field rather than
  // EnumValueDescriptor. For proto3 this is just setting the enum field to the
  // value specified, for proto2 it's more complicated. If value is a known enum
  // value the field is set as usual. If the value is unknown then it is added
  // to the unknown field set. Note this matches the behavior of parsing unknown
  // enum values. If multiple calls with unknown values happen than they are all
  // added to the unknown field set in order of the calls.
  void AddEnumValue(Message* message, const FieldDescriptor* field,
                    int value) const;
  // See MutableMessage() for comments on the "factory" parameter.
  Message* AddMessage(Message* message, const FieldDescriptor* field,
                      MessageFactory* factory = nullptr) const;

  // Appends an already-allocated object 'new_entry' to the repeated field
  // specified by 'field' passing ownership to the message.
  void AddAllocatedMessage(Message* message, const FieldDescriptor* field,
                           Message* new_entry) const;

  // Similar to AddAllocatedMessage() without internal safety and ownership
  // checks. This method should only be used when the objects are on the same
  // arena or paired with a call to `UnsafeArenaReleaseLast`.
  void UnsafeArenaAddAllocatedMessage(Message* message,
                                      const FieldDescriptor* field,
                                      Message* new_entry) const;


  // Get a RepeatedFieldRef object that can be used to read the underlying
  // repeated field. The type parameter T must be set according to the
  // field's cpp type. The following table shows the mapping from cpp type
  // to acceptable T.
  //
  //   field->cpp_type()      T
  //   CPPTYPE_INT32        int32_t
  //   CPPTYPE_UINT32       uint32_t
  //   CPPTYPE_INT64        int64_t
  //   CPPTYPE_UINT64       uint64_t
  //   CPPTYPE_DOUBLE       double
  //   CPPTYPE_FLOAT        float
  //   CPPTYPE_BOOL         bool
  //   CPPTYPE_ENUM         generated enum type or int32_t
  //   CPPTYPE_STRING       std::string
  //   CPPTYPE_MESSAGE      generated message type or google::protobuf::Message
  //
  // A RepeatedFieldRef object can be copied and the resulted object will point
  // to the same repeated field in the same message. The object can be used as
  // long as the message is not destroyed.
  //
  // Note that to use this method users need to include the header file
  // "reflection.h" (which defines the RepeatedFieldRef class templates).
  template <typename T>
  RepeatedFieldRef<T> GetRepeatedFieldRef(const Message& message,
                                          const FieldDescriptor* field) const;

  // Like GetRepeatedFieldRef() but return an object that can also be used
  // manipulate the underlying repeated field.
  template <typename T>
  MutableRepeatedFieldRef<T> GetMutableRepeatedFieldRef(
      Message* message, const FieldDescriptor* field) const;

  // DEPRECATED. Please use Get(Mutable)RepeatedFieldRef() for repeated field
  // access. The following repeated field accessors will be removed in the
  // future.
  //
  // Repeated field accessors  -------------------------------------------------
  // The methods above, e.g. GetRepeatedInt32(msg, fd, index), provide singular
  // access to the data in a RepeatedField.  The methods below provide aggregate
  // access by exposing the RepeatedField object itself with the Message.
  // Applying these templates to inappropriate types will lead to an undefined
  // reference at link time (e.g. GetRepeatedField<***double>), or possibly a
  // template matching error at compile time (e.g. GetRepeatedPtrField<File>).
  //
  // Usage example: my_doubs = refl->GetRepeatedField<double>(msg, fd);

  // DEPRECATED. Please use GetRepeatedFieldRef().
  //
  // for T = Cord and all protobuf scalar types except enums.
  template <typename T>
  PROTOBUF_DEPRECATED_MSG("Please use GetRepeatedFieldRef() instead")
  const RepeatedField<T>& GetRepeatedField(const Message& msg,
                                           const FieldDescriptor* d) const {
    return GetRepeatedFieldInternal<T>(msg, d);
  }

  // DEPRECATED. Please use GetMutableRepeatedFieldRef().
  //
  // for T = Cord and all protobuf scalar types except enums.
  template <typename T>
  PROTOBUF_DEPRECATED_MSG("Please use GetMutableRepeatedFieldRef() instead")
  RepeatedField<T>* MutableRepeatedField(Message* msg,
                                         const FieldDescriptor* d) const {
    return MutableRepeatedFieldInternal<T>(msg, d);
  }

  // DEPRECATED. Please use GetRepeatedFieldRef().
  //
  // for T = std::string, google::protobuf::internal::StringPieceField
  //         google::protobuf::Message & descendants.
  template <typename T>
  PROTOBUF_DEPRECATED_MSG("Please use GetRepeatedFieldRef() instead")
  const RepeatedPtrField<T>& GetRepeatedPtrField(
      const Message& msg, const FieldDescriptor* d) const {
    return GetRepeatedPtrFieldInternal<T>(msg, d);
  }

  // DEPRECATED. Please use GetMutableRepeatedFieldRef().
  //
  // for T = std::string, google::protobuf::internal::StringPieceField
  //         google::protobuf::Message & descendants.
  template <typename T>
  PROTOBUF_DEPRECATED_MSG("Please use GetMutableRepeatedFieldRef() instead")
  RepeatedPtrField<T>* MutableRepeatedPtrField(Message* msg,
                                               const FieldDescriptor* d) const {
    return MutableRepeatedPtrFieldInternal<T>(msg, d);
  }

  // Extensions ----------------------------------------------------------------

  // Try to find an extension of this message type by fully-qualified field
  // name.  Returns nullptr if no extension is known for this name or number.
  const FieldDescriptor* FindKnownExtensionByName(
      const std::string& name) const;

  // Try to find an extension of this message type by field number.
  // Returns nullptr if no extension is known for this name or number.
  const FieldDescriptor* FindKnownExtensionByNumber(int number) const;

  // Feature Flags -------------------------------------------------------------

  // Does this message support storing arbitrary integer values in enum fields?
  // If |true|, GetEnumValue/SetEnumValue and associated repeated-field versions
  // take arbitrary integer values, and the legacy GetEnum() getter will
  // dynamically create an EnumValueDescriptor for any integer value without
  // one. If |false|, setting an unknown enum value via the integer-based
  // setters results in undefined behavior (in practice, GOOGLE_DCHECK-fails).
  //
  // Generic code that uses reflection to handle messages with enum fields
  // should check this flag before using the integer-based setter, and either
  // downgrade to a compatible value or use the UnknownFieldSet if not. For
  // example:
  //
  //   int new_value = GetValueFromApplicationLogic();
  //   if (reflection->SupportsUnknownEnumValues()) {
  //     reflection->SetEnumValue(message, field, new_value);
  //   } else {
  //     if (field_descriptor->enum_type()->
  //             FindValueByNumber(new_value) != nullptr) {
  //       reflection->SetEnumValue(message, field, new_value);
  //     } else if (emit_unknown_enum_values) {
  //       reflection->MutableUnknownFields(message)->AddVarint(
  //           field->number(), new_value);
  //     } else {
  //       // convert value to a compatible/default value.
  //       new_value = CompatibleDowngrade(new_value);
  //       reflection->SetEnumValue(message, field, new_value);
  //     }
  //   }
  bool SupportsUnknownEnumValues() const;

  // Returns the MessageFactory associated with this message.  This can be
  // useful for determining if a message is a generated message or not, for
  // example:
  //   if (message->GetReflection()->GetMessageFactory() ==
  //       google::protobuf::MessageFactory::generated_factory()) {
  //     // This is a generated message.
  //   }
  // It can also be used to create more messages of this type, though
  // Message::New() is an easier way to accomplish this.
  MessageFactory* GetMessageFactory() const;

 private:
  template <typename T>
  const RepeatedField<T>& GetRepeatedFieldInternal(
      const Message& message, const FieldDescriptor* field) const;
  template <typename T>
  RepeatedField<T>* MutableRepeatedFieldInternal(
      Message* message, const FieldDescriptor* field) const;
  template <typename T>
  const RepeatedPtrField<T>& GetRepeatedPtrFieldInternal(
      const Message& message, const FieldDescriptor* field) const;
  template <typename T>
  RepeatedPtrField<T>* MutableRepeatedPtrFieldInternal(
      Message* message, const FieldDescriptor* field) const;
  // Obtain a pointer to a Repeated Field Structure and do some type checking:
  //   on field->cpp_type(),
  //   on field->field_option().ctype() (if ctype >= 0)
  //   of field->message_type() (if message_type != nullptr).
  // We use 2 routine rather than 4 (const vs mutable) x (scalar vs pointer).
  void* MutableRawRepeatedField(Message* message, const FieldDescriptor* field,
                                FieldDescriptor::CppType, int ctype,
                                const Descriptor* message_type) const;

  const void* GetRawRepeatedField(const Message& message,
                                  const FieldDescriptor* field,
                                  FieldDescriptor::CppType cpptype, int ctype,
                                  const Descriptor* message_type) const;

  // The following methods are used to implement (Mutable)RepeatedFieldRef.
  // A Ref object will store a raw pointer to the repeated field data (obtained
  // from RepeatedFieldData()) and a pointer to a Accessor (obtained from
  // RepeatedFieldAccessor) which will be used to access the raw data.

  // Returns a raw pointer to the repeated field
  //
  // "cpp_type" and "message_type" are deduced from the type parameter T passed
  // to Get(Mutable)RepeatedFieldRef. If T is a generated message type,
  // "message_type" should be set to its descriptor. Otherwise "message_type"
  // should be set to nullptr. Implementations of this method should check
  // whether "cpp_type"/"message_type" is consistent with the actual type of the
  // field. We use 1 routine rather than 2 (const vs mutable) because it is
  // protected and it doesn't change the message.
  void* RepeatedFieldData(Message* message, const FieldDescriptor* field,
                          FieldDescriptor::CppType cpp_type,
                          const Descriptor* message_type) const;

  // The returned pointer should point to a singleton instance which implements
  // the RepeatedFieldAccessor interface.
  const internal::RepeatedFieldAccessor* RepeatedFieldAccessor(
      const FieldDescriptor* field) const;

  // Lists all fields of the message which are currently set, except for unknown
  // fields and stripped fields. See ListFields for details.
  void ListFieldsOmitStripped(
      const Message& message,
      std::vector<const FieldDescriptor*>* output) const;

  bool IsMessageStripped(const Descriptor* descriptor) const {
    return schema_.IsMessageStripped(descriptor);
  }

  friend class TextFormat;

  void ListFieldsMayFailOnStripped(
      const Message& message, bool should_fail,
      std::vector<const FieldDescriptor*>* output) const;

  // Returns true if the message field is backed by a LazyField.
  //
  // A message field may be backed by a LazyField without the user annotation
  // ([lazy = true]). While the user-annotated LazyField is lazily verified on
  // first touch (i.e. failure on access rather than parsing if the LazyField is
  // not initialized), the inferred LazyField is eagerly verified to avoid lazy
  // parsing error at the cost of lower efficiency. When reflecting a message
  // field, use this API instead of checking field->options().lazy().
  bool IsLazyField(const FieldDescriptor* field) const {
    return IsLazilyVerifiedLazyField(field) ||
           IsEagerlyVerifiedLazyField(field);
  }

  // Returns true if the field is lazy extension. It is meant to allow python
  // reparse lazy field until b/157559327 is fixed.
  bool IsLazyExtension(const Message& message,
                       const FieldDescriptor* field) const;

  bool IsLazilyVerifiedLazyField(const FieldDescriptor* field) const;
  bool IsEagerlyVerifiedLazyField(const FieldDescriptor* field) const;

  friend class FastReflectionMessageMutator;

  const Descriptor* const descriptor_;
  const internal::ReflectionSchema schema_;
  const DescriptorPool* const descriptor_pool_;
  MessageFactory* const message_factory_;

  // Last non weak field index. This is an optimization when most weak fields
  // are at the end of the containing message. If a message proto doesn't
  // contain weak fields, then this field equals descriptor_->field_count().
  int last_non_weak_field_index_;

  template <typename T, typename Enable>
  friend class RepeatedFieldRef;
  template <typename T, typename Enable>
  friend class MutableRepeatedFieldRef;
  friend class ::PROTOBUF_NAMESPACE_ID::MessageLayoutInspector;
  friend class ::PROTOBUF_NAMESPACE_ID::AssignDescriptorsHelper;
  friend class DynamicMessageFactory;
  friend class DynamicMessageReflectionHelper;
  friend class GeneratedMessageReflectionTestHelper;
  friend class python::MapReflectionFriend;
  friend class python::MessageReflectionFriend;
  friend class util::MessageDifferencer;
#define GOOGLE_PROTOBUF_HAS_CEL_MAP_REFLECTION_FRIEND
  friend class expr::CelMapReflectionFriend;
  friend class internal::MapFieldReflectionTest;
  friend class internal::MapKeySorter;
  friend class internal::WireFormat;
  friend class internal::ReflectionOps;
  friend class internal::SwapFieldHelper;
  // Needed for implementing text format for map.
  friend class internal::MapFieldPrinterHelper;

  Reflection(const Descriptor* descriptor,
             const internal::ReflectionSchema& schema,
             const DescriptorPool* pool, MessageFactory* factory);

  // Special version for specialized implementations of string.  We can't
  // call MutableRawRepeatedField directly here because we don't have access to
  // FieldOptions::* which are defined in descriptor.pb.h.  Including that
  // file here is not possible because it would cause a circular include cycle.
  // We use 1 routine rather than 2 (const vs mutable) because it is private
  // and mutable a repeated string field doesn't change the message.
  void* MutableRawRepeatedString(Message* message, const FieldDescriptor* field,
                                 bool is_string) const;

  friend class MapReflectionTester;
  // Returns true if key is in map. Returns false if key is not in map field.
  bool ContainsMapKey(const Message& message, const FieldDescriptor* field,
                      const MapKey& key) const;

  // If key is in map field: Saves the value pointer to val and returns
  // false. If key in not in map field: Insert the key into map, saves
  // value pointer to val and returns true. Users are able to modify the
  // map value by MapValueRef.
  bool InsertOrLookupMapValue(Message* message, const FieldDescriptor* field,
                              const MapKey& key, MapValueRef* val) const;

  // If key is in map field: Saves the value pointer to val and returns true.
  // Returns false if key is not in map field. Users are NOT able to modify
  // the value by MapValueConstRef.
  bool LookupMapValue(const Message& message, const FieldDescriptor* field,
                      const MapKey& key, MapValueConstRef* val) const;
  bool LookupMapValue(const Message&, const FieldDescriptor*, const MapKey&,
                      MapValueRef*) const = delete;

  // Delete and returns true if key is in the map field. Returns false
  // otherwise.
  bool DeleteMapValue(Message* message, const FieldDescriptor* field,
                      const MapKey& key) const;

  // Returns a MapIterator referring to the first element in the map field.
  // If the map field is empty, this function returns the same as
  // reflection::MapEnd. Mutation to the field may invalidate the iterator.
  MapIterator MapBegin(Message* message, const FieldDescriptor* field) const;

  // Returns a MapIterator referring to the theoretical element that would
  // follow the last element in the map field. It does not point to any
  // real element. Mutation to the field may invalidate the iterator.
  MapIterator MapEnd(Message* message, const FieldDescriptor* field) const;

  // Get the number of <key, value> pair of a map field. The result may be
  // different from FieldSize which can have duplicate keys.
  int MapSize(const Message& message, const FieldDescriptor* field) const;

  // Help method for MapIterator.
  friend class MapIterator;
  friend class WireFormatForMapFieldTest;
  internal::MapFieldBase* MutableMapData(Message* message,
                                         const FieldDescriptor* field) const;

  const internal::MapFieldBase* GetMapData(const Message& message,
                                           const FieldDescriptor* field) const;

  template <class T>
  const T& GetRawNonOneof(const Message& message,
                          const FieldDescriptor* field) const;
  template <class T>
  T* MutableRawNonOneof(Message* message, const FieldDescriptor* field) const;

  template <typename Type>
  const Type& GetRaw(const Message& message,
                     const FieldDescriptor* field) const;
  template <typename Type>
  inline Type* MutableRaw(Message* message, const FieldDescriptor* field) const;
  template <typename Type>
  const Type& DefaultRaw(const FieldDescriptor* field) const;

  const Message* GetDefaultMessageInstance(const FieldDescriptor* field) const;

  inline const uint32_t* GetHasBits(const Message& message) const;
  inline uint32_t* MutableHasBits(Message* message) const;
  inline uint32_t GetOneofCase(const Message& message,
                               const OneofDescriptor* oneof_descriptor) const;
  inline uint32_t* MutableOneofCase(
      Message* message, const OneofDescriptor* oneof_descriptor) const;
  inline bool HasExtensionSet(const Message& /* message */) const {
    return schema_.HasExtensionSet();
  }
  const internal::ExtensionSet& GetExtensionSet(const Message& message) const;
  internal::ExtensionSet* MutableExtensionSet(Message* message) const;

  inline const internal::InternalMetadata& GetInternalMetadata(
      const Message& message) const;

  internal::InternalMetadata* MutableInternalMetadata(Message* message) const;

  inline bool IsInlined(const FieldDescriptor* field) const;

  inline bool HasBit(const Message& message,
                     const FieldDescriptor* field) const;
  inline void SetBit(Message* message, const FieldDescriptor* field) const;
  inline void ClearBit(Message* message, const FieldDescriptor* field) const;
  inline void SwapBit(Message* message1, Message* message2,
                      const FieldDescriptor* field) const;

  inline const uint32_t* GetInlinedStringDonatedArray(
      const Message& message) const;
  inline uint32_t* MutableInlinedStringDonatedArray(Message* message) const;
  inline bool IsInlinedStringDonated(const Message& message,
                                     const FieldDescriptor* field) const;

  // Shallow-swap fields listed in fields vector of two messages. It is the
  // caller's responsibility to make sure shallow swap is safe.
  void UnsafeShallowSwapFields(
      Message* message1, Message* message2,
      const std::vector<const FieldDescriptor*>& fields) const;

  // This function only swaps the field. Should swap corresponding has_bit
  // before or after using this function.
  void SwapField(Message* message1, Message* message2,
                 const FieldDescriptor* field) const;

  // Unsafe but shallow version of SwapField.
  void UnsafeShallowSwapField(Message* message1, Message* message2,
                              const FieldDescriptor* field) const;

  template <bool unsafe_shallow_swap>
  void SwapFieldsImpl(Message* message1, Message* message2,
                      const std::vector<const FieldDescriptor*>& fields) const;

  template <bool unsafe_shallow_swap>
  void SwapOneofField(Message* lhs, Message* rhs,
                      const OneofDescriptor* oneof_descriptor) const;

  inline bool HasOneofField(const Message& message,
                            const FieldDescriptor* field) const;
  inline void SetOneofCase(Message* message,
                           const FieldDescriptor* field) const;
  inline void ClearOneofField(Message* message,
                              const FieldDescriptor* field) const;

  template <typename Type>
  inline const Type& GetField(const Message& message,
                              const FieldDescriptor* field) const;
  template <typename Type>
  inline void SetField(Message* message, const FieldDescriptor* field,
                       const Type& value) const;
  template <typename Type>
  inline Type* MutableField(Message* message,
                            const FieldDescriptor* field) const;
  template <typename Type>
  inline const Type& GetRepeatedField(const Message& message,
                                      const FieldDescriptor* field,
                                      int index) const;
  template <typename Type>
  inline const Type& GetRepeatedPtrField(const Message& message,
                                         const FieldDescriptor* field,
                                         int index) const;
  template <typename Type>
  inline void SetRepeatedField(Message* message, const FieldDescriptor* field,
                               int index, Type value) const;
  template <typename Type>
  inline Type* MutableRepeatedField(Message* message,
                                    const FieldDescriptor* field,
                                    int index) const;
  template <typename Type>
  inline void AddField(Message* message, const FieldDescriptor* field,
                       const Type& value) const;
  template <typename Type>
  inline Type* AddField(Message* message, const FieldDescriptor* field) const;

  int GetExtensionNumberOrDie(const Descriptor* type) const;

  // Internal versions of EnumValue API perform no checking. Called after checks
  // by public methods.
  void SetEnumValueInternal(Message* message, const FieldDescriptor* field,
                            int value) const;
  void SetRepeatedEnumValueInternal(Message* message,
                                    const FieldDescriptor* field, int index,
                                    int value) const;
  void AddEnumValueInternal(Message* message, const FieldDescriptor* field,
                            int value) const;

  friend inline  // inline so nobody can call this function.
      void
      RegisterAllTypesInternal(const Metadata* file_level_metadata, int size);
  friend inline const char* ParseLenDelim(int field_number,
                                          const FieldDescriptor* field,
                                          Message* msg,
                                          const Reflection* reflection,
                                          const char* ptr,
                                          internal::ParseContext* ctx);
  friend inline const char* ParsePackedField(const FieldDescriptor* field,
                                             Message* msg,
                                             const Reflection* reflection,
                                             const char* ptr,
                                             internal::ParseContext* ctx);

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Reflection);
};

// Abstract interface for a factory for message objects.
class PROTOBUF_EXPORT MessageFactory {
 public:
  inline MessageFactory() {}
  virtual ~MessageFactory();

  // Given a Descriptor, gets or constructs the default (prototype) Message
  // of that type.  You can then call that message's New() method to construct
  // a mutable message of that type.
  //
  // Calling this method twice with the same Descriptor returns the same
  // object.  The returned object remains property of the factory.  Also, any
  // objects created by calling the prototype's New() method share some data
  // with the prototype, so these must be destroyed before the MessageFactory
  // is destroyed.
  //
  // The given descriptor must outlive the returned message, and hence must
  // outlive the MessageFactory.
  //
  // Some implementations do not support all types.  GetPrototype() will
  // return nullptr if the descriptor passed in is not supported.
  //
  // This method may or may not be thread-safe depending on the implementation.
  // Each implementation should document its own degree thread-safety.
  virtual const Message* GetPrototype(const Descriptor* type) = 0;

  // Gets a MessageFactory which supports all generated, compiled-in messages.
  // In other words, for any compiled-in type FooMessage, the following is true:
  //   MessageFactory::generated_factory()->GetPrototype(
  //     FooMessage::descriptor()) == FooMessage::default_instance()
  // This factory supports all types which are found in
  // DescriptorPool::generated_pool().  If given a descriptor from any other
  // pool, GetPrototype() will return nullptr.  (You can also check if a
  // descriptor is for a generated message by checking if
  // descriptor->file()->pool() == DescriptorPool::generated_pool().)
  //
  // This factory is 100% thread-safe; calling GetPrototype() does not modify
  // any shared data.
  //
  // This factory is a singleton.  The caller must not delete the object.
  static MessageFactory* generated_factory();

  // For internal use only:  Registers a .proto file at static initialization
  // time, to be placed in generated_factory.  The first time GetPrototype()
  // is called with a descriptor from this file, |register_messages| will be
  // called, with the file name as the parameter.  It must call
  // InternalRegisterGeneratedMessage() (below) to register each message type
  // in the file.  This strange mechanism is necessary because descriptors are
  // built lazily, so we can't register types by their descriptor until we
  // know that the descriptor exists.  |filename| must be a permanent string.
  static void InternalRegisterGeneratedFile(
      const google::protobuf::internal::DescriptorTable* table);

  // For internal use only:  Registers a message type.  Called only by the
  // functions which are registered with InternalRegisterGeneratedFile(),
  // above.
  static void InternalRegisterGeneratedMessage(const Descriptor* descriptor,
                                               const Message* prototype);


 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MessageFactory);
};

#define DECLARE_GET_REPEATED_FIELD(TYPE)                           \
  template <>                                                      \
  PROTOBUF_EXPORT const RepeatedField<TYPE>&                       \
  Reflection::GetRepeatedFieldInternal<TYPE>(                      \
      const Message& message, const FieldDescriptor* field) const; \
                                                                   \
  template <>                                                      \
  PROTOBUF_EXPORT RepeatedField<TYPE>*                             \
  Reflection::MutableRepeatedFieldInternal<TYPE>(                  \
      Message * message, const FieldDescriptor* field) const;

DECLARE_GET_REPEATED_FIELD(int32_t)
DECLARE_GET_REPEATED_FIELD(int64_t)
DECLARE_GET_REPEATED_FIELD(uint32_t)
DECLARE_GET_REPEATED_FIELD(uint64_t)
DECLARE_GET_REPEATED_FIELD(float)
DECLARE_GET_REPEATED_FIELD(double)
DECLARE_GET_REPEATED_FIELD(bool)

#undef DECLARE_GET_REPEATED_FIELD

// Tries to downcast this message to a generated message type.  Returns nullptr
// if this class is not an instance of T.  This works even if RTTI is disabled.
//
// This also has the effect of creating a strong reference to T that will
// prevent the linker from stripping it out at link time.  This can be important
// if you are using a DynamicMessageFactory that delegates to the generated
// factory.
template <typename T>
const T* DynamicCastToGenerated(const Message* from) {
  // Compile-time assert that T is a generated type that has a
  // default_instance() accessor, but avoid actually calling it.
  const T& (*get_default_instance)() = &T::default_instance;
  (void)get_default_instance;

  // Compile-time assert that T is a subclass of google::protobuf::Message.
  const Message* unused = static_cast<T*>(nullptr);
  (void)unused;

#if PROTOBUF_RTTI
  return dynamic_cast<const T*>(from);
#else
  bool ok = from != nullptr &&
            T::default_instance().GetReflection() == from->GetReflection();
  return ok ? down_cast<const T*>(from) : nullptr;
#endif
}

template <typename T>
T* DynamicCastToGenerated(Message* from) {
  const Message* message_const = from;
  return const_cast<T*>(DynamicCastToGenerated<T>(message_const));
}

// Call this function to ensure that this message's reflection is linked into
// the binary:
//
//   google::protobuf::LinkMessageReflection<FooMessage>();
//
// This will ensure that the following lookup will succeed:
//
//   DescriptorPool::generated_pool()->FindMessageTypeByName("FooMessage");
//
// As a side-effect, it will also guarantee that anything else from the same
// .proto file will also be available for lookup in the generated pool.
//
// This function does not actually register the message, so it does not need
// to be called before the lookup.  However it does need to occur in a function
// that cannot be stripped from the binary (ie. it must be reachable from main).
//
// Best practice is to call this function as close as possible to where the
// reflection is actually needed.  This function is very cheap to call, so you
// should not need to worry about its runtime overhead except in the tightest
// of loops (on x86-64 it compiles into two "mov" instructions).
template <typename T>
void LinkMessageReflection() {
  internal::StrongReference(T::default_instance);
}

// =============================================================================
// Implementation details for {Get,Mutable}RawRepeatedPtrField.  We provide
// specializations for <std::string>, <StringPieceField> and <Message> and
// handle everything else with the default template which will match any type
// having a method with signature "static const google::protobuf::Descriptor*
// descriptor()". Such a type presumably is a descendant of google::protobuf::Message.

template <>
inline const RepeatedPtrField<std::string>&
Reflection::GetRepeatedPtrFieldInternal<std::string>(
    const Message& message, const FieldDescriptor* field) const {
  return *static_cast<RepeatedPtrField<std::string>*>(
      MutableRawRepeatedString(const_cast<Message*>(&message), field, true));
}

template <>
inline RepeatedPtrField<std::string>*
Reflection::MutableRepeatedPtrFieldInternal<std::string>(
    Message* message, const FieldDescriptor* field) const {
  return static_cast<RepeatedPtrField<std::string>*>(
      MutableRawRepeatedString(message, field, true));
}


// -----

template <>
inline const RepeatedPtrField<Message>& Reflection::GetRepeatedPtrFieldInternal(
    const Message& message, const FieldDescriptor* field) const {
  return *static_cast<const RepeatedPtrField<Message>*>(GetRawRepeatedField(
      message, field, FieldDescriptor::CPPTYPE_MESSAGE, -1, nullptr));
}

template <>
inline RepeatedPtrField<Message>* Reflection::MutableRepeatedPtrFieldInternal(
    Message* message, const FieldDescriptor* field) const {
  return static_cast<RepeatedPtrField<Message>*>(MutableRawRepeatedField(
      message, field, FieldDescriptor::CPPTYPE_MESSAGE, -1, nullptr));
}

template <typename PB>
inline const RepeatedPtrField<PB>& Reflection::GetRepeatedPtrFieldInternal(
    const Message& message, const FieldDescriptor* field) const {
  return *static_cast<const RepeatedPtrField<PB>*>(
      GetRawRepeatedField(message, field, FieldDescriptor::CPPTYPE_MESSAGE, -1,
                          PB::default_instance().GetDescriptor()));
}

template <typename PB>
inline RepeatedPtrField<PB>* Reflection::MutableRepeatedPtrFieldInternal(
    Message* message, const FieldDescriptor* field) const {
  return static_cast<RepeatedPtrField<PB>*>(
      MutableRawRepeatedField(message, field, FieldDescriptor::CPPTYPE_MESSAGE,
                              -1, PB::default_instance().GetDescriptor()));
}

template <typename Type>
const Type& Reflection::DefaultRaw(const FieldDescriptor* field) const {
  return *reinterpret_cast<const Type*>(schema_.GetFieldDefault(field));
}

uint32_t Reflection::GetOneofCase(
    const Message& message, const OneofDescriptor* oneof_descriptor) const {
  GOOGLE_DCHECK(!oneof_descriptor->is_synthetic());
  return internal::GetConstRefAtOffset<uint32_t>(
      message, schema_.GetOneofCaseOffset(oneof_descriptor));
}

bool Reflection::HasOneofField(const Message& message,
                               const FieldDescriptor* field) const {
  return (GetOneofCase(message, field->containing_oneof()) ==
          static_cast<uint32_t>(field->number()));
}

template <typename Type>
const Type& Reflection::GetRaw(const Message& message,
                               const FieldDescriptor* field) const {
  GOOGLE_DCHECK(!schema_.InRealOneof(field) || HasOneofField(message, field))
      << "Field = " << field->full_name();
  return internal::GetConstRefAtOffset<Type>(message,
                                             schema_.GetFieldOffset(field));
}
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif  // GOOGLE_PROTOBUF_MESSAGE_H__
