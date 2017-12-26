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

#ifndef GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__
#define GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__

#include <assert.h>
#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/wire_format_lite_inl.h>

namespace google {
namespace protobuf {
class Arena;
namespace internal {
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapEntry;
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapFieldLite;
}  // namespace internal
}  // namespace protobuf

namespace protobuf {
namespace internal {

// MoveHelper::Move is used to set *dest.  It copies *src, or moves it (in
// the C++11 sense), or swaps it. *src is left in a sane state for
// subsequent destruction, but shouldn't be used for anything.
template <bool is_enum, bool is_message, bool is_stringlike, typename T>
struct MoveHelper {  // primitives
  static void Move(T* src, T* dest) { *dest = *src; }
};

template <bool is_message, bool is_stringlike, typename T>
struct MoveHelper<true, is_message, is_stringlike, T> {  // enums
  static void Move(T* src, T* dest) { *dest = *src; }
  // T is an enum here, so allow conversions to and from int.
  static void Move(T* src, int* dest) { *dest = static_cast<int>(*src); }
  static void Move(int* src, T* dest) { *dest = static_cast<T>(*src); }
};

template <bool is_stringlike, typename T>
struct MoveHelper<false, true, is_stringlike, T> {  // messages
  static void Move(T* src, T* dest) { dest->Swap(src); }
};

template <typename T>
struct MoveHelper<false, false, true, T> {  // strings and similar
  static void Move(T* src, T* dest) {
#if __cplusplus >= 201103L
    *dest = std::move(*src);
#else
    dest->swap(*src);
#endif
  }
};

// MapEntryLite is used to implement parsing and serialization of map for lite
// runtime.
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapEntryLite : public MessageLite {
  // Provide utilities to parse/serialize key/value.  Provide utilities to
  // manipulate internal stored type.
  typedef MapTypeHandler<kKeyFieldType, Key> KeyTypeHandler;
  typedef MapTypeHandler<kValueFieldType, Value> ValueTypeHandler;

  // Define internal memory layout. Strings and messages are stored as
  // pointers, while other types are stored as values.
  typedef typename KeyTypeHandler::TypeOnMemory KeyOnMemory;
  typedef typename ValueTypeHandler::TypeOnMemory ValueOnMemory;

  // Enum type cannot be used for MapTypeHandler::Read. Define a type
  // which will replace Enum with int.
  typedef typename KeyTypeHandler::MapEntryAccessorType KeyMapEntryAccessorType;
  typedef typename ValueTypeHandler::MapEntryAccessorType
      ValueMapEntryAccessorType;

  // Constants for field number.
  static const int kKeyFieldNumber = 1;
  static const int kValueFieldNumber = 2;

  // Constants for field tag.
  static const uint8 kKeyTag = GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(
      kKeyFieldNumber, KeyTypeHandler::kWireType);
  static const uint8 kValueTag = GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(
      kValueFieldNumber, ValueTypeHandler::kWireType);
  static const size_t kTagSize = 1;

 public:
  ~MapEntryLite() {
    if (this != default_instance_) {
      if (GetArenaNoVirtual() != NULL) return;
      KeyTypeHandler::DeleteNoArena(key_);
      ValueTypeHandler::DeleteNoArena(value_);
    }
  }

  // accessors ======================================================

  virtual inline const KeyMapEntryAccessorType& key() const {
    return KeyTypeHandler::GetExternalReference(key_);
  }
  virtual inline const ValueMapEntryAccessorType& value() const {
    GOOGLE_CHECK(default_instance_ != NULL);
    return ValueTypeHandler::DefaultIfNotInitialized(value_,
                                                    default_instance_->value_);
  }
  inline KeyMapEntryAccessorType* mutable_key() {
    set_has_key();
    return KeyTypeHandler::EnsureMutable(&key_, GetArenaNoVirtual());
  }
  inline ValueMapEntryAccessorType* mutable_value() {
    set_has_value();
    return ValueTypeHandler::EnsureMutable(&value_, GetArenaNoVirtual());
  }

  // implements MessageLite =========================================

  // MapEntryLite is for implementation only and this function isn't called
  // anywhere. Just provide a fake implementation here for MessageLite.
  string GetTypeName() const { return ""; }

  void CheckTypeAndMergeFrom(const MessageLite& other) {
    MergeFrom(*::google::protobuf::down_cast<const MapEntryLite*>(&other));
  }

  bool MergePartialFromCodedStream(::google::protobuf::io::CodedInputStream* input) {
    uint32 tag;

    for (;;) {
      // 1) corrupted data: return false;
      // 2) unknown field: skip without putting into unknown field set;
      // 3) unknown enum value: keep it in parsing. In proto2, caller should
      // check the value and put this entry into containing message's unknown
      // field set if the value is an unknown enum. In proto3, caller doesn't
      // need to care whether the value is unknown enum;
      // 4) missing key/value: missed key/value will have default value. caller
      // should take this entry as if key/value is set to default value.
      tag = input->ReadTag();
      switch (tag) {
        case kKeyTag:
          if (!KeyTypeHandler::Read(input, mutable_key())) {
            return false;
          }
          set_has_key();
          if (!input->ExpectTag(kValueTag)) break;
          GOOGLE_FALLTHROUGH_INTENDED;

        case kValueTag:
          if (!ValueTypeHandler::Read(input, mutable_value())) {
            return false;
          }
          set_has_value();
          if (input->ExpectAtEnd()) return true;
          break;

        default:
          if (tag == 0 ||
              WireFormatLite::GetTagWireType(tag) ==
              WireFormatLite::WIRETYPE_END_GROUP) {
            return true;
          }
          if (!WireFormatLite::SkipField(input, tag)) return false;
          break;
      }
    }
  }

  size_t ByteSizeLong() const {
    size_t size = 0;
    size += has_key() ? kTagSize + KeyTypeHandler::ByteSize(key()) : 0;
    size += has_value() ? kTagSize + ValueTypeHandler::ByteSize(value()) : 0;
    return size;
  }

  void SerializeWithCachedSizes(::google::protobuf::io::CodedOutputStream* output) const {
    KeyTypeHandler::Write(kKeyFieldNumber, key(), output);
    ValueTypeHandler::Write(kValueFieldNumber, value(), output);
  }

  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(bool deterministic,
                                                   ::google::protobuf::uint8* output) const {
    output = KeyTypeHandler::InternalWriteToArray(kKeyFieldNumber, key(),
                                                  deterministic, output);
    output = ValueTypeHandler::InternalWriteToArray(kValueFieldNumber, value(),
                                                    deterministic, output);
    return output;
  }
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }

  int GetCachedSize() const {
    int size = 0;
    size += has_key()
        ? kTagSize + KeyTypeHandler::GetCachedSize(key())
        : 0;
    size += has_value()
        ? kTagSize + ValueTypeHandler::GetCachedSize(
            value())
        : 0;
    return size;
  }

  bool IsInitialized() const { return ValueTypeHandler::IsInitialized(value_); }

  MessageLite* New() const {
    MapEntryLite* entry = new MapEntryLite;
    entry->default_instance_ = default_instance_;
    return entry;
  }

  MessageLite* New(Arena* arena) const {
    MapEntryLite* entry = Arena::CreateMessage<MapEntryLite>(arena);
    entry->default_instance_ = default_instance_;
    return entry;
  }

  int SpaceUsed() const {
    int size = sizeof(MapEntryLite);
    size += KeyTypeHandler::SpaceUsedInMapEntry(key_);
    size += ValueTypeHandler::SpaceUsedInMapEntry(value_);
    return size;
  }

  void MergeFrom(const MapEntryLite& from) {
    if (from._has_bits_[0]) {
      if (from.has_key()) {
        KeyTypeHandler::EnsureMutable(&key_, GetArenaNoVirtual());
        KeyTypeHandler::Merge(from.key(), &key_, GetArenaNoVirtual());
        set_has_key();
      }
      if (from.has_value()) {
        ValueTypeHandler::EnsureMutable(&value_, GetArenaNoVirtual());
        ValueTypeHandler::Merge(from.value(), &value_, GetArenaNoVirtual());
        set_has_value();
      }
    }
  }

  void Clear() {
    KeyTypeHandler::Clear(&key_, GetArenaNoVirtual());
    ValueTypeHandler::ClearMaybeByDefaultEnum(
        &value_, GetArenaNoVirtual(), default_enum_value);
    clear_has_key();
    clear_has_value();
  }

  void InitAsDefaultInstance() {
    KeyTypeHandler::AssignDefaultValue(&key_);
    ValueTypeHandler::AssignDefaultValue(&value_);
  }

  Arena* GetArena() const {
    return GetArenaNoVirtual();
  }

  // Create a MapEntryLite for given key and value from google::protobuf::Map in
  // serialization. This function is only called when value is enum. Enum is
  // treated differently because its type in MapEntry is int and its type in
  // google::protobuf::Map is enum. We cannot create a reference to int from an enum.
  static MapEntryLite* EnumWrap(const Key& key, const Value value,
                                Arena* arena) {
    return Arena::CreateMessage<MapEnumEntryWrapper<
        Key, Value, kKeyFieldType, kValueFieldType, default_enum_value> >(
        arena, key, value);
  }

  // Like above, but for all the other types. This avoids value copy to create
  // MapEntryLite from google::protobuf::Map in serialization.
  static MapEntryLite* Wrap(const Key& key, const Value& value, Arena* arena) {
    return Arena::CreateMessage<MapEntryWrapper<Key, Value, kKeyFieldType,
                                                kValueFieldType,
                                                default_enum_value> >(
        arena, key, value);
  }

  // Parsing using MergePartialFromCodedStream, above, is not as
  // efficient as it could be.  This helper class provides a speedier way.
  template <typename MapField, typename Map>
  class Parser {
   public:
    explicit Parser(MapField* mf) : mf_(mf), map_(mf->MutableMap()) {}

    // This does what the typical MergePartialFromCodedStream() is expected to
    // do, with the additional side-effect that if successful (i.e., if true is
    // going to be its return value) it inserts the key-value pair into map_.
    bool MergePartialFromCodedStream(::google::protobuf::io::CodedInputStream* input) {
      // Look for the expected thing: a key and then a value.  If it fails,
      // invoke the enclosing class's MergePartialFromCodedStream, or return
      // false if that would be pointless.
      if (input->ExpectTag(kKeyTag)) {
        if (!KeyTypeHandler::Read(input, &key_)) {
          return false;
        }
        // Peek at the next byte to see if it is kValueTag.  If not, bail out.
        const void* data;
        int size;
        input->GetDirectBufferPointerInline(&data, &size);
        // We could use memcmp here, but we don't bother. The tag is one byte.
        assert(kTagSize == 1);
        if (size > 0 && *reinterpret_cast<const char*>(data) == kValueTag) {
          typename Map::size_type size = map_->size();
          value_ptr_ = &(*map_)[key_];
          if (GOOGLE_PREDICT_TRUE(size != map_->size())) {
            // We created a new key-value pair.  Fill in the value.
            typedef
                typename MapIf<ValueTypeHandler::kIsEnum, int*, Value*>::type T;
            input->Skip(kTagSize);  // Skip kValueTag.
            if (!ValueTypeHandler::Read(input,
                                        reinterpret_cast<T>(value_ptr_))) {
              map_->erase(key_);  // Failure! Undo insertion.
              return false;
            }
            if (input->ExpectAtEnd()) return true;
            return ReadBeyondKeyValuePair(input);
          }
        }
      } else {
        key_ = Key();
      }

      entry_.reset(mf_->NewEntry());
      *entry_->mutable_key() = key_;
      if (!entry_->MergePartialFromCodedStream(input)) return false;
      return UseKeyAndValueFromEntry();
    }

    const Key& key() const { return key_; }
    const Value& value() const { return *value_ptr_; }

   private:
    bool UseKeyAndValueFromEntry() GOOGLE_ATTRIBUTE_COLD {
      // Update key_ in case we need it later (because key() is called).
      // This is potentially inefficient, especially if the key is
      // expensive to copy (e.g., a long string), but this is a cold
      // path, so it's not a big deal.
      key_ = entry_->key();
      value_ptr_ = &(*map_)[key_];
      MoveHelper<ValueTypeHandler::kIsEnum,
                 ValueTypeHandler::kIsMessage,
                 ValueTypeHandler::kWireType ==
                 WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
                 Value>::Move(entry_->mutable_value(), value_ptr_);
      if (entry_->GetArena() != NULL) entry_.release();
      return true;
    }

    // After reading a key and value successfully, and inserting that data
    // into map_, we are not at the end of the input.  This is unusual, but
    // allowed by the spec.
    bool ReadBeyondKeyValuePair(::google::protobuf::io::CodedInputStream* input)
        GOOGLE_ATTRIBUTE_COLD {
      typedef MoveHelper<KeyTypeHandler::kIsEnum,
                         KeyTypeHandler::kIsMessage,
                         KeyTypeHandler::kWireType ==
                         WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
                         Key> KeyMover;
      typedef MoveHelper<ValueTypeHandler::kIsEnum,
                         ValueTypeHandler::kIsMessage,
                         ValueTypeHandler::kWireType ==
                         WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
                         Value> ValueMover;
      entry_.reset(mf_->NewEntry());
      ValueMover::Move(value_ptr_, entry_->mutable_value());
      map_->erase(key_);
      KeyMover::Move(&key_, entry_->mutable_key());
      if (!entry_->MergePartialFromCodedStream(input)) return false;
      return UseKeyAndValueFromEntry();
    }

    MapField* const mf_;
    Map* const map_;
    Key key_;
    Value* value_ptr_;
    // On the fast path entry_ is not used.
    google::protobuf::scoped_ptr<MapEntryLite> entry_;
  };

 protected:
  void set_has_key() { _has_bits_[0] |= 0x00000001u; }
  bool has_key() const { return (_has_bits_[0] & 0x00000001u) != 0; }
  void clear_has_key() { _has_bits_[0] &= ~0x00000001u; }
  void set_has_value() { _has_bits_[0] |= 0x00000002u; }
  bool has_value() const { return (_has_bits_[0] & 0x00000002u) != 0; }
  void clear_has_value() { _has_bits_[0] &= ~0x00000002u; }

 private:
  // Serializing a generated message containing map field involves serializing
  // key-value pairs from google::protobuf::Map. The wire format of each key-value pair
  // after serialization should be the same as that of a MapEntry message
  // containing the same key and value inside it.  However, google::protobuf::Map doesn't
  // store key and value as MapEntry message, which disables us to use existing
  // code to serialize message. In order to use existing code to serialize
  // message, we need to construct a MapEntry from key-value pair. But it
  // involves copy of key and value to construct a MapEntry. In order to avoid
  // this copy in constructing a MapEntry, we need the following class which
  // only takes references of given key and value.
  template <typename K, typename V, WireFormatLite::FieldType k_wire_type,
            WireFormatLite::FieldType v_wire_type, int default_enum>
  class MapEntryWrapper
      : public MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> {
    typedef MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> Base;
    typedef typename Base::KeyMapEntryAccessorType KeyMapEntryAccessorType;
    typedef typename Base::ValueMapEntryAccessorType ValueMapEntryAccessorType;

   public:
    MapEntryWrapper(Arena* arena, const K& key, const V& value)
        : MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum>(arena),
          key_(key),
          value_(value) {
      Base::set_has_key();
      Base::set_has_value();
    }
    inline const KeyMapEntryAccessorType& key() const { return key_; }
    inline const ValueMapEntryAccessorType& value() const { return value_; }

   private:
    const Key& key_;
    const Value& value_;

    friend class ::google::protobuf::Arena;
    typedef void InternalArenaConstructable_;
    typedef void DestructorSkippable_;
  };

  // Like above, but for enum value only, which stores value instead of
  // reference of value field inside. This is needed because the type of value
  // field in constructor is an enum, while we need to store it as an int. If we
  // initialize a reference to int with a reference to enum, compiler will
  // generate a temporary int from enum and initialize the reference to int with
  // the temporary.
  template <typename K, typename V, WireFormatLite::FieldType k_wire_type,
            WireFormatLite::FieldType v_wire_type, int default_enum>
  class MapEnumEntryWrapper
      : public MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> {
    typedef MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> Base;
    typedef typename Base::KeyMapEntryAccessorType KeyMapEntryAccessorType;
    typedef typename Base::ValueMapEntryAccessorType ValueMapEntryAccessorType;

   public:
    MapEnumEntryWrapper(Arena* arena, const K& key, const V& value)
        : MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum>(arena),
          key_(key),
          value_(value) {
      Base::set_has_key();
      Base::set_has_value();
    }
    inline const KeyMapEntryAccessorType& key() const { return key_; }
    inline const ValueMapEntryAccessorType& value() const { return value_; }

   private:
    const KeyMapEntryAccessorType& key_;
    const ValueMapEntryAccessorType value_;

    friend class google::protobuf::Arena;
    typedef void DestructorSkippable_;
  };

  MapEntryLite() : default_instance_(NULL), arena_(NULL) {
    KeyTypeHandler::Initialize(&key_, NULL);
    ValueTypeHandler::InitializeMaybeByDefaultEnum(
        &value_, default_enum_value, NULL);
    _has_bits_[0] = 0;
  }

  explicit MapEntryLite(Arena* arena)
      : default_instance_(NULL), arena_(arena) {
    KeyTypeHandler::Initialize(&key_, arena);
    ValueTypeHandler::InitializeMaybeByDefaultEnum(
        &value_, default_enum_value, arena);
    _has_bits_[0] = 0;
  }

  inline Arena* GetArenaNoVirtual() const {
    return arena_;
  }

  void set_default_instance(MapEntryLite* default_instance) {
    default_instance_ = default_instance;
  }

  MapEntryLite* default_instance_;

  KeyOnMemory key_;
  ValueOnMemory value_;
  Arena* arena_;
  uint32 _has_bits_[1];

  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  template <typename K, typename V, WireFormatLite::FieldType,
            WireFormatLite::FieldType, int>
  friend class internal::MapEntry;
  template <typename K, typename V, WireFormatLite::FieldType,
            WireFormatLite::FieldType, int>
  friend class internal::MapFieldLite;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MapEntryLite);
};

// Helpers for deterministic serialization =============================

// This struct can be used with any generic sorting algorithm.  If the Key
// type is relatively small and easy to copy then copying Keys into an
// array of SortItems can be beneficial.  Then all the data the sorting
// algorithm needs to touch is in that one array.
template <typename Key, typename PtrToKeyValuePair> struct SortItem {
  SortItem() {}
  explicit SortItem(PtrToKeyValuePair p) : first(p->first), second(p) {}

  Key first;
  PtrToKeyValuePair second;
};

template <typename T> struct CompareByFirstField {
  bool operator()(const T& a, const T& b) const {
    return a.first < b.first;
  }
};

template <typename T> struct CompareByDerefFirst {
  bool operator()(const T& a, const T& b) const {
    return a->first < b->first;
  }
};

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__
