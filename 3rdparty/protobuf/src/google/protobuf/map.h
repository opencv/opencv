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

// This file defines the map container and its helpers to support protobuf maps.
//
// The Map and MapIterator types are provided by this header file.
// Please avoid using other types defined here, unless they are public
// types within Map or MapIterator, such as Map::value_type.

#ifndef GOOGLE_PROTOBUF_MAP_H__
#define GOOGLE_PROTOBUF_MAP_H__

#include <google/protobuf/stubs/hash.h>
#include <iterator>
#include <limits>  // To support Visual Studio 2008
#include <set>
#include <utility>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/generated_enum_util.h>
#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#if __cpp_exceptions && LANG_CXX11
#include <random>
#endif

namespace google {
namespace protobuf {

template <typename Key, typename T>
class Map;

class MapIterator;

template <typename Enum> struct is_proto_enum;

namespace internal {
template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
class MapFieldLite;

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
class MapField;

template <typename Key, typename T>
class TypeDefinedMapFieldBase;

class DynamicMapField;

class GeneratedMessageReflection;
}  // namespace internal

#define TYPE_CHECK(EXPECTEDTYPE, METHOD)                        \
  if (type() != EXPECTEDTYPE) {                                 \
    GOOGLE_LOG(FATAL)                                                  \
        << "Protocol Buffer map usage error:\n"                 \
        << METHOD << " type does not match\n"                   \
        << "  Expected : "                                      \
        << FieldDescriptor::CppTypeName(EXPECTEDTYPE) << "\n"   \
        << "  Actual   : "                                      \
        << FieldDescriptor::CppTypeName(type());                \
  }

// MapKey is an union type for representing any possible
// map key.
class LIBPROTOBUF_EXPORT MapKey {
 public:
  MapKey() : type_(0) {
  }
  MapKey(const MapKey& other) : type_(0) {
    CopyFrom(other);
  }

  ~MapKey() {
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      delete val_.string_value_;
    }
  }

  FieldDescriptor::CppType type() const {
    if (type_ == 0) {
      GOOGLE_LOG(FATAL)
          << "Protocol Buffer map usage error:\n"
          << "MapKey::type MapKey is not initialized. "
          << "Call set methods to initialize MapKey.";
    }
    return (FieldDescriptor::CppType)type_;
  }

  void SetInt64Value(int64 value) {
    SetType(FieldDescriptor::CPPTYPE_INT64);
    val_.int64_value_ = value;
  }
  void SetUInt64Value(uint64 value) {
    SetType(FieldDescriptor::CPPTYPE_UINT64);
    val_.uint64_value_ = value;
  }
  void SetInt32Value(int32 value) {
    SetType(FieldDescriptor::CPPTYPE_INT32);
    val_.int32_value_ = value;
  }
  void SetUInt32Value(uint32 value) {
    SetType(FieldDescriptor::CPPTYPE_UINT32);
    val_.uint32_value_ = value;
  }
  void SetBoolValue(bool value) {
    SetType(FieldDescriptor::CPPTYPE_BOOL);
    val_.bool_value_ = value;
  }
  void SetStringValue(const string& val) {
    SetType(FieldDescriptor::CPPTYPE_STRING);
    *val_.string_value_ = val;
  }

  int64 GetInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapKey::GetInt64Value");
    return val_.int64_value_;
  }
  uint64 GetUInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapKey::GetUInt64Value");
    return val_.uint64_value_;
  }
  int32 GetInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapKey::GetInt32Value");
    return val_.int32_value_;
  }
  uint32 GetUInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapKey::GetUInt32Value");
    return val_.uint32_value_;
  }
  bool GetBoolValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapKey::GetBoolValue");
    return val_.bool_value_;
  }
  const string& GetStringValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapKey::GetStringValue");
    return *val_.string_value_;
  }

  bool operator<(const MapKey& other) const {
    if (type_ != other.type_) {
      // We could define a total order that handles this case, but
      // there currently no need.  So, for now, fail.
      GOOGLE_LOG(FATAL) << "Unsupported: type mismatch";
    }
    switch (type()) {
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_ENUM:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Unsupported";
        return false;
      case FieldDescriptor::CPPTYPE_STRING:
        return *val_.string_value_ < *other.val_.string_value_;
      case FieldDescriptor::CPPTYPE_INT64:
        return val_.int64_value_ < other.val_.int64_value_;
      case FieldDescriptor::CPPTYPE_INT32:
        return val_.int32_value_ < other.val_.int32_value_;
      case FieldDescriptor::CPPTYPE_UINT64:
        return val_.uint64_value_ < other.val_.uint64_value_;
      case FieldDescriptor::CPPTYPE_UINT32:
        return val_.uint32_value_ < other.val_.uint32_value_;
      case FieldDescriptor::CPPTYPE_BOOL:
        return val_.bool_value_ < other.val_.bool_value_;
    }
    return false;
  }

  bool operator==(const MapKey& other) const {
    if (type_ != other.type_) {
      // To be consistent with operator<, we don't allow this either.
      GOOGLE_LOG(FATAL) << "Unsupported: type mismatch";
    }
    switch (type()) {
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_ENUM:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Unsupported";
        break;
      case FieldDescriptor::CPPTYPE_STRING:
        return *val_.string_value_ == *other.val_.string_value_;
      case FieldDescriptor::CPPTYPE_INT64:
        return val_.int64_value_ == other.val_.int64_value_;
      case FieldDescriptor::CPPTYPE_INT32:
        return val_.int32_value_ == other.val_.int32_value_;
      case FieldDescriptor::CPPTYPE_UINT64:
        return val_.uint64_value_ == other.val_.uint64_value_;
      case FieldDescriptor::CPPTYPE_UINT32:
        return val_.uint32_value_ == other.val_.uint32_value_;
      case FieldDescriptor::CPPTYPE_BOOL:
        return val_.bool_value_ == other.val_.bool_value_;
    }
    GOOGLE_LOG(FATAL) << "Can't get here.";
    return false;
  }

  void CopyFrom(const MapKey& other) {
    SetType(other.type());
    switch (type_) {
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_ENUM:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Unsupported";
        break;
      case FieldDescriptor::CPPTYPE_STRING:
        *val_.string_value_ = *other.val_.string_value_;
        break;
      case FieldDescriptor::CPPTYPE_INT64:
        val_.int64_value_ = other.val_.int64_value_;
        break;
      case FieldDescriptor::CPPTYPE_INT32:
        val_.int32_value_ = other.val_.int32_value_;
        break;
      case FieldDescriptor::CPPTYPE_UINT64:
        val_.uint64_value_ = other.val_.uint64_value_;
        break;
      case FieldDescriptor::CPPTYPE_UINT32:
        val_.uint32_value_ = other.val_.uint32_value_;
        break;
      case FieldDescriptor::CPPTYPE_BOOL:
        val_.bool_value_ = other.val_.bool_value_;
        break;
    }
  }

 private:
  template <typename K, typename V>
  friend class internal::TypeDefinedMapFieldBase;
  friend class MapIterator;
  friend class internal::DynamicMapField;

  union KeyValue {
    KeyValue() {}
    string* string_value_;
    int64 int64_value_;
    int32 int32_value_;
    uint64 uint64_value_;
    uint32 uint32_value_;
    bool bool_value_;
  } val_;

  void SetType(FieldDescriptor::CppType type) {
    if (type_ == type) return;
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      delete val_.string_value_;
    }
    type_ = type;
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      val_.string_value_ = new string;
    }
  }

  // type_ is 0 or a valid FieldDescriptor::CppType.
  int type_;
};

// MapValueRef points to a map value.
class LIBPROTOBUF_EXPORT MapValueRef {
 public:
  MapValueRef() : data_(NULL), type_(0) {}

  void SetInt64Value(int64 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapValueRef::SetInt64Value");
    *reinterpret_cast<int64*>(data_) = value;
  }
  void SetUInt64Value(uint64 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapValueRef::SetUInt64Value");
    *reinterpret_cast<uint64*>(data_) = value;
  }
  void SetInt32Value(int32 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapValueRef::SetInt32Value");
    *reinterpret_cast<int32*>(data_) = value;
  }
  void SetUInt32Value(uint32 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapValueRef::SetUInt32Value");
    *reinterpret_cast<uint32*>(data_) = value;
  }
  void SetBoolValue(bool value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapValueRef::SetBoolValue");
    *reinterpret_cast<bool*>(data_) = value;
  }
  // TODO(jieluo) - Checks that enum is member.
  void SetEnumValue(int value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_ENUM,
               "MapValueRef::SetEnumValue");
    *reinterpret_cast<int*>(data_) = value;
  }
  void SetStringValue(const string& value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapValueRef::SetStringValue");
    *reinterpret_cast<string*>(data_) = value;
  }
  void SetFloatValue(float value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_FLOAT,
               "MapValueRef::SetFloatValue");
    *reinterpret_cast<float*>(data_) = value;
  }
  void SetDoubleValue(double value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_DOUBLE,
               "MapValueRef::SetDoubleValue");
    *reinterpret_cast<double*>(data_) = value;
  }

  int64 GetInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapValueRef::GetInt64Value");
    return *reinterpret_cast<int64*>(data_);
  }
  uint64 GetUInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapValueRef::GetUInt64Value");
    return *reinterpret_cast<uint64*>(data_);
  }
  int32 GetInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapValueRef::GetInt32Value");
    return *reinterpret_cast<int32*>(data_);
  }
  uint32 GetUInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapValueRef::GetUInt32Value");
    return *reinterpret_cast<uint32*>(data_);
  }
  bool GetBoolValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapValueRef::GetBoolValue");
    return *reinterpret_cast<bool*>(data_);
  }
  int GetEnumValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_ENUM,
               "MapValueRef::GetEnumValue");
    return *reinterpret_cast<int*>(data_);
  }
  const string& GetStringValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapValueRef::GetStringValue");
    return *reinterpret_cast<string*>(data_);
  }
  float GetFloatValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_FLOAT,
               "MapValueRef::GetFloatValue");
    return *reinterpret_cast<float*>(data_);
  }
  double GetDoubleValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_DOUBLE,
               "MapValueRef::GetDoubleValue");
    return *reinterpret_cast<double*>(data_);
  }

  const Message& GetMessageValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_MESSAGE,
               "MapValueRef::GetMessageValue");
    return *reinterpret_cast<Message*>(data_);
  }

  Message* MutableMessageValue() {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_MESSAGE,
               "MapValueRef::MutableMessageValue");
    return reinterpret_cast<Message*>(data_);
  }

 private:
  template <typename K, typename V,
            internal::WireFormatLite::FieldType key_wire_type,
            internal::WireFormatLite::FieldType value_wire_type,
            int default_enum_value>
  friend class internal::MapField;
  template <typename K, typename V>
  friend class internal::TypeDefinedMapFieldBase;
  friend class MapIterator;
  friend class internal::GeneratedMessageReflection;
  friend class internal::DynamicMapField;

  void SetType(FieldDescriptor::CppType type) {
    type_ = type;
  }

  FieldDescriptor::CppType type() const {
    if (type_ == 0 || data_ == NULL) {
      GOOGLE_LOG(FATAL)
          << "Protocol Buffer map usage error:\n"
          << "MapValueRef::type MapValueRef is not initialized.";
    }
    return (FieldDescriptor::CppType)type_;
  }
  void SetValue(const void* val) {
    data_ = const_cast<void*>(val);
  }
  void CopyFrom(const MapValueRef& other) {
    type_ = other.type_;
    data_ = other.data_;
  }
  // Only used in DynamicMapField
  void DeleteData() {
    switch (type_) {
#define HANDLE_TYPE(CPPTYPE, TYPE)                              \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: {        \
        delete reinterpret_cast<TYPE*>(data_);                  \
        break;                                                  \
      }
      HANDLE_TYPE(INT32, int32);
      HANDLE_TYPE(INT64, int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(ENUM, int32);
      HANDLE_TYPE(MESSAGE, Message);
#undef HANDLE_TYPE
    }
  }
  // data_ point to a map value. MapValueRef does not
  // own this value.
  void* data_;
  // type_ is 0 or a valid FieldDescriptor::CppType.
  int type_;
};

#undef TYPE_CHECK

// This is the class for google::protobuf::Map's internal value_type. Instead of using
// std::pair as value_type, we use this class which provides us more control of
// its process of construction and destruction.
template <typename Key, typename T>
class MapPair {
 public:
  typedef const Key first_type;
  typedef T second_type;

  MapPair(const Key& other_first, const T& other_second)
      : first(other_first), second(other_second) {}
  explicit MapPair(const Key& other_first) : first(other_first), second() {}
  MapPair(const MapPair& other)
      : first(other.first), second(other.second) {}

  ~MapPair() {}

  // Implicitly convertible to std::pair of compatible types.
  template <typename T1, typename T2>
  operator std::pair<T1, T2>() const {
    return std::pair<T1, T2>(first, second);
  }

  const Key first;
  T second;

 private:
  friend class ::google::protobuf::Arena;
  friend class Map<Key, T>;
};

// google::protobuf::Map is an associative container type used to store protobuf map
// fields.  Each Map instance may or may not use a different hash function, a
// different iteration order, and so on.  E.g., please don't examine
// implementation details to decide if the following would work:
//  Map<int, int> m0, m1;
//  m0[0] = m1[0] = m0[1] = m1[1] = 0;
//  assert(m0.begin()->first == m1.begin()->first);  // Bug!
//
// Map's interface is similar to std::unordered_map, except that Map is not
// designed to play well with exceptions.
template <typename Key, typename T>
class Map {
 public:
  typedef Key key_type;
  typedef T mapped_type;
  typedef MapPair<Key, T> value_type;

  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  typedef size_t size_type;
  typedef hash<Key> hasher;

  explicit Map(bool old_style = true)
      : arena_(NULL),
        default_enum_value_(0),
        old_style_(old_style) {
    Init();
  }
  explicit Map(Arena* arena, bool old_style = true)
      : arena_(arena),
        default_enum_value_(0),
        old_style_(old_style) {
    Init();
  }
  Map(const Map& other)
      : arena_(NULL),
        default_enum_value_(other.default_enum_value_),
        old_style_(other.old_style_) {
    Init();
    insert(other.begin(), other.end());
  }
  template <class InputIt>
  Map(const InputIt& first, const InputIt& last, bool old_style = true)
      : arena_(NULL),
        default_enum_value_(0),
        old_style_(old_style) {
    Init();
    insert(first, last);
  }

  ~Map() {
    clear();
    if (arena_ == NULL) {
      if (old_style_)
        delete deprecated_elements_;
      else
        delete elements_;
    }
  }

 private:
  void Init() {
    if (old_style_)
      deprecated_elements_ = Arena::Create<DeprecatedInnerMap>(
          arena_, 0, hasher(), std::equal_to<Key>(),
          MapAllocator<std::pair<const Key, MapPair<Key, T>*> >(arena_));
    else
      elements_ =
          Arena::Create<InnerMap>(arena_, 0, hasher(), Allocator(arena_));
  }

  // re-implement std::allocator to use arena allocator for memory allocation.
  // Used for google::protobuf::Map implementation. Users should not use this class
  // directly.
  template <typename U>
  class MapAllocator {
   public:
    typedef U value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    MapAllocator() : arena_(NULL) {}
    explicit MapAllocator(Arena* arena) : arena_(arena) {}
    template <typename X>
    MapAllocator(const MapAllocator<X>& allocator)
        : arena_(allocator.arena()) {}

    pointer allocate(size_type n, const_pointer hint = 0) {
      // If arena is not given, malloc needs to be called which doesn't
      // construct element object.
      if (arena_ == NULL) {
        return static_cast<pointer>(::operator new(n * sizeof(value_type)));
      } else {
        return reinterpret_cast<pointer>(
            Arena::CreateArray<uint8>(arena_, n * sizeof(value_type)));
      }
    }

    void deallocate(pointer p, size_type n) {
      if (arena_ == NULL) {
#if defined(__GXX_DELETE_WITH_SIZE__) || defined(__cpp_sized_deallocation)
        ::operator delete(p, n * sizeof(value_type));
#else
        ::operator delete(p);
#endif
      }
    }

#if __cplusplus >= 201103L && !defined(GOOGLE_PROTOBUF_OS_APPLE) && \
    !defined(GOOGLE_PROTOBUF_OS_NACL) &&                            \
    !defined(GOOGLE_PROTOBUF_OS_ANDROID) &&                         \
    !defined(GOOGLE_PROTOBUF_OS_EMSCRIPTEN)
    template<class NodeType, class... Args>
    void construct(NodeType* p, Args&&... args) {
      // Clang 3.6 doesn't compile static casting to void* directly. (Issue
      // #1266) According C++ standard 5.2.9/1: "The static_cast operator shall
      // not cast away constness". So first the maybe const pointer is casted to
      // const void* and after the const void* is const casted.
      new (const_cast<void*>(static_cast<const void*>(p)))
          NodeType(std::forward<Args>(args)...);
    }

    template<class NodeType>
    void destroy(NodeType* p) {
      p->~NodeType();
    }
#else
    void construct(pointer p, const_reference t) { new (p) value_type(t); }

    void destroy(pointer p) { p->~value_type(); }
#endif

    template <typename X>
    struct rebind {
      typedef MapAllocator<X> other;
    };

    template <typename X>
    bool operator==(const MapAllocator<X>& other) const {
      return arena_ == other.arena_;
    }

    template <typename X>
    bool operator!=(const MapAllocator<X>& other) const {
      return arena_ != other.arena_;
    }

    // To support Visual Studio 2008
    size_type max_size() const {
      return std::numeric_limits<size_type>::max();
    }

    // To support gcc-4.4, which does not properly
    // support templated friend classes
    Arena* arena() const {
      return arena_;
    }

   private:
    typedef void DestructorSkippable_;
    Arena* const arena_;
  };

  // InnerMap's key type is Key and its value type is value_type*.  We use a
  // custom class here and for Node, below, to ensure that k_ is at offset 0,
  // allowing safe conversion from pointer to Node to pointer to Key, and vice
  // versa when appropriate.
  class KeyValuePair {
   public:
    KeyValuePair(const Key& k, value_type* v) : k_(k), v_(v) {}

    const Key& key() const { return k_; }
    Key& key() { return k_; }
    value_type* const value() const { return v_; }
    value_type*& value() { return v_; }

   private:
    Key k_;
    value_type* v_;
  };

  typedef MapAllocator<KeyValuePair> Allocator;

  // InnerMap is a generic hash-based map.  It doesn't contain any
  // protocol-buffer-specific logic.  It is a chaining hash map with the
  // additional feature that some buckets can be converted to use an ordered
  // container.  This ensures O(lg n) bounds on find, insert, and erase, while
  // avoiding the overheads of ordered containers most of the time.
  //
  // The implementation doesn't need the full generality of unordered_map,
  // and it doesn't have it.  More bells and whistles can be added as needed.
  // Some implementation details:
  // 1. The hash function has type hasher and the equality function
  //    equal_to<Key>.  We inherit from hasher to save space
  //    (empty-base-class optimization).
  // 2. The number of buckets is a power of two.
  // 3. Buckets are converted to trees in pairs: if we convert bucket b then
  //    buckets b and b^1 will share a tree.  Invariant: buckets b and b^1 have
  //    the same non-NULL value iff they are sharing a tree.  (An alternative
  //    implementation strategy would be to have a tag bit per bucket.)
  // 4. As is typical for hash_map and such, the Keys and Values are always
  //    stored in linked list nodes.  Pointers to elements are never invalidated
  //    until the element is deleted.
  // 5. The trees' payload type is pointer to linked-list node.  Tree-converting
  //    a bucket doesn't copy Key-Value pairs.
  // 6. Once we've tree-converted a bucket, it is never converted back. However,
  //    the items a tree contains may wind up assigned to trees or lists upon a
  //    rehash.
  // 7. The code requires no C++ features from C++11 or later.
  // 8. Mutations to a map do not invalidate the map's iterators, pointers to
  //    elements, or references to elements.
  // 9. Except for erase(iterator), any non-const method can reorder iterators.
  class InnerMap : private hasher {
   public:
    typedef value_type* Value;

    InnerMap(size_type n, hasher h, Allocator alloc)
        : hasher(h),
          num_elements_(0),
          seed_(Seed()),
          table_(NULL),
          alloc_(alloc) {
      n = TableSize(n);
      table_ = CreateEmptyTable(n);
      num_buckets_ = index_of_first_non_null_ = n;
    }

    ~InnerMap() {
      if (table_ != NULL) {
        clear();
        Dealloc<void*>(table_, num_buckets_);
      }
    }

   private:
    enum { kMinTableSize = 8 };

    // Linked-list nodes, as one would expect for a chaining hash table.
    struct Node {
      KeyValuePair kv;
      Node* next;
    };

    // This is safe only if the given pointer is known to point to a Key that is
    // part of a Node.
    static Node* NodePtrFromKeyPtr(Key* k) {
      return reinterpret_cast<Node*>(k);
    }

    static Key* KeyPtrFromNodePtr(Node* node) { return &node->kv.key(); }

    // Trees.  The payload type is pointer to Key, so that we can query the tree
    // with Keys that are not in any particular data structure.  When we insert,
    // though, the pointer is always pointing to a Key that is inside a Node.
    struct KeyCompare {
      bool operator()(const Key* n0, const Key* n1) const { return *n0 < *n1; }
    };
    typedef typename Allocator::template rebind<Key*>::other KeyPtrAllocator;
    typedef std::set<Key*, KeyCompare, KeyPtrAllocator> Tree;

    // iterator and const_iterator are instantiations of iterator_base.
    template <typename KeyValueType>
    class iterator_base {
     public:
      typedef KeyValueType& reference;
      typedef KeyValueType* pointer;
      typedef typename Tree::iterator TreeIterator;

      // Invariants:
      // node_ is always correct. This is handy because the most common
      // operations are operator* and operator-> and they only use node_.
      // When node_ is set to a non-NULL value, all the other non-const fields
      // are updated to be correct also, but those fields can become stale
      // if the underlying map is modified.  When those fields are needed they
      // are rechecked, and updated if necessary.
      iterator_base() : node_(NULL) {}

      explicit iterator_base(const InnerMap* m) : m_(m) {
        SearchFrom(m->index_of_first_non_null_);
      }

      // Any iterator_base can convert to any other.  This is overkill, and we
      // rely on the enclosing class to use it wisely.  The standard "iterator
      // can convert to const_iterator" is OK but the reverse direction is not.
      template <typename U>
      explicit iterator_base(const iterator_base<U>& it)
          : node_(it.node_),
            m_(it.m_),
            bucket_index_(it.bucket_index_),
            tree_it_(it.tree_it_) {}

      iterator_base(Node* n, const InnerMap* m, size_type index)
          : node_(n),
            m_(m),
            bucket_index_(index) {}

      iterator_base(TreeIterator tree_it, const InnerMap* m, size_type index)
          : node_(NodePtrFromKeyPtr(*tree_it)),
            m_(m),
            bucket_index_(index),
            tree_it_(tree_it) {
        // Invariant: iterators that use tree_it_ have an even bucket_index_.
        GOOGLE_DCHECK_EQ(bucket_index_ % 2, 0);
      }

      // Advance through buckets, looking for the first that isn't empty.
      // If nothing non-empty is found then leave node_ == NULL.
      void SearchFrom(size_type start_bucket) {
        GOOGLE_DCHECK(m_->index_of_first_non_null_ == m_->num_buckets_ ||
               m_->table_[m_->index_of_first_non_null_] != NULL);
        node_ = NULL;
        for (bucket_index_ = start_bucket; bucket_index_ < m_->num_buckets_;
             bucket_index_++) {
          if (m_->TableEntryIsNonEmptyList(bucket_index_)) {
            node_ = static_cast<Node*>(m_->table_[bucket_index_]);
            break;
          } else if (m_->TableEntryIsTree(bucket_index_)) {
            Tree* tree = static_cast<Tree*>(m_->table_[bucket_index_]);
            GOOGLE_DCHECK(!tree->empty());
            tree_it_ = tree->begin();
            node_ = NodePtrFromKeyPtr(*tree_it_);
            break;
          }
        }
      }

      reference operator*() const { return node_->kv; }
      pointer operator->() const { return &(operator*()); }

      friend bool operator==(const iterator_base& a, const iterator_base& b) {
        return a.node_ == b.node_;
      }
      friend bool operator!=(const iterator_base& a, const iterator_base& b) {
        return a.node_ != b.node_;
      }

      iterator_base& operator++() {
        if (node_->next == NULL) {
          const bool is_list = revalidate_if_necessary();
          if (is_list) {
            SearchFrom(bucket_index_ + 1);
          } else {
            GOOGLE_DCHECK_EQ(bucket_index_ & 1, 0);
            Tree* tree = static_cast<Tree*>(m_->table_[bucket_index_]);
            if (++tree_it_ == tree->end()) {
              SearchFrom(bucket_index_ + 2);
            } else {
              node_ = NodePtrFromKeyPtr(*tree_it_);
            }
          }
        } else {
          node_ = node_->next;
        }
        return *this;
      }

      iterator_base operator++(int /* unused */) {
        iterator_base tmp = *this;
        ++*this;
        return tmp;
      }

      // Assumes node_ and m_ are correct and non-NULL, but other fields may be
      // stale.  Fix them as needed.  Then return true iff node_ points to a
      // Node in a list.
      bool revalidate_if_necessary() {
        GOOGLE_DCHECK(node_ != NULL && m_ != NULL);
        // Force bucket_index_ to be in range.
        bucket_index_ &= (m_->num_buckets_ - 1);
        // Common case: the bucket we think is relevant points to node_.
        if (m_->table_[bucket_index_] == static_cast<void*>(node_))
          return true;
        // Less common: the bucket is a linked list with node_ somewhere in it,
        // but not at the head.
        if (m_->TableEntryIsNonEmptyList(bucket_index_)) {
          Node* l = static_cast<Node*>(m_->table_[bucket_index_]);
          while ((l = l->next) != NULL) {
            if (l == node_) {
              return true;
            }
          }
        }
        // Well, bucket_index_ still might be correct, but probably
        // not.  Revalidate just to be sure.  This case is rare enough that we
        // don't worry about potential optimizations, such as having a custom
        // find-like method that compares Node* instead of const Key&.
        iterator_base i(m_->find(*KeyPtrFromNodePtr(node_)));
        bucket_index_ = i.bucket_index_;
        tree_it_ = i.tree_it_;
        return m_->TableEntryIsList(bucket_index_);
      }

      Node* node_;
      const InnerMap* m_;
      size_type bucket_index_;
      TreeIterator tree_it_;
    };

   public:
    typedef iterator_base<KeyValuePair> iterator;
    typedef iterator_base<const KeyValuePair> const_iterator;

    iterator begin() { return iterator(this); }
    iterator end() { return iterator(); }
    const_iterator begin() const { return const_iterator(this); }
    const_iterator end() const { return const_iterator(); }

    void clear() {
      for (size_type b = 0; b < num_buckets_; b++) {
        if (TableEntryIsNonEmptyList(b)) {
          Node* node = static_cast<Node*>(table_[b]);
          table_[b] = NULL;
          do {
            Node* next = node->next;
            DestroyNode(node);
            node = next;
          } while (node != NULL);
        } else if (TableEntryIsTree(b)) {
          Tree* tree = static_cast<Tree*>(table_[b]);
          GOOGLE_DCHECK(table_[b] == table_[b + 1] && (b & 1) == 0);
          table_[b] = table_[b + 1] = NULL;
          typename Tree::iterator tree_it = tree->begin();
          do {
            Node* node = NodePtrFromKeyPtr(*tree_it);
            typename Tree::iterator next = tree_it;
            ++next;
            tree->erase(tree_it);
            DestroyNode(node);
            tree_it = next;
          } while (tree_it != tree->end());
          DestroyTree(tree);
          b++;
        }
      }
      num_elements_ = 0;
      index_of_first_non_null_ = num_buckets_;
    }

    const hasher& hash_function() const { return *this; }

    static size_type max_size() {
      return static_cast<size_type>(1) << (sizeof(void**) >= 8 ? 60 : 28);
    }
    size_type size() const { return num_elements_; }
    bool empty() const { return size() == 0; }

    iterator find(const Key& k) { return iterator(FindHelper(k).first); }
    const_iterator find(const Key& k) const { return FindHelper(k).first; }

    // In traditional C++ style, this performs "insert if not present."
    std::pair<iterator, bool> insert(const KeyValuePair& kv) {
      std::pair<const_iterator, size_type> p = FindHelper(kv.key());
      // Case 1: key was already present.
      if (p.first.node_ != NULL)
        return std::make_pair(iterator(p.first), false);
      // Case 2: insert.
      if (ResizeIfLoadIsOutOfRange(num_elements_ + 1)) {
        p = FindHelper(kv.key());
      }
      const size_type b = p.second;  // bucket number
      Node* node = Alloc<Node>(1);
      alloc_.construct(&node->kv, kv);
      iterator result = InsertUnique(b, node);
      ++num_elements_;
      return std::make_pair(result, true);
    }

    // The same, but if an insertion is necessary then the value portion of the
    // inserted key-value pair is left uninitialized.
    std::pair<iterator, bool> insert(const Key& k) {
      std::pair<const_iterator, size_type> p = FindHelper(k);
      // Case 1: key was already present.
      if (p.first.node_ != NULL)
        return std::make_pair(iterator(p.first), false);
      // Case 2: insert.
      if (ResizeIfLoadIsOutOfRange(num_elements_ + 1)) {
        p = FindHelper(k);
      }
      const size_type b = p.second;  // bucket number
      Node* node = Alloc<Node>(1);
      typedef typename Allocator::template rebind<Key>::other KeyAllocator;
      KeyAllocator(alloc_).construct(&node->kv.key(), k);
      iterator result = InsertUnique(b, node);
      ++num_elements_;
      return std::make_pair(result, true);
    }

    Value& operator[](const Key& k) {
      KeyValuePair kv(k, Value());
      return insert(kv).first->value();
    }

    void erase(iterator it) {
      GOOGLE_DCHECK_EQ(it.m_, this);
      const bool is_list = it.revalidate_if_necessary();
      size_type b = it.bucket_index_;
      Node* const item = it.node_;
      if (is_list) {
        GOOGLE_DCHECK(TableEntryIsNonEmptyList(b));
        Node* head = static_cast<Node*>(table_[b]);
        head = EraseFromLinkedList(item, head);
        table_[b] = static_cast<void*>(head);
      } else {
        GOOGLE_DCHECK(TableEntryIsTree(b));
        Tree* tree = static_cast<Tree*>(table_[b]);
        tree->erase(it.tree_it_);
        if (tree->empty()) {
          // Force b to be the minimum of b and b ^ 1.  This is important
          // only because we want index_of_first_non_null_ to be correct.
          b &= ~static_cast<size_type>(1);
          DestroyTree(tree);
          table_[b] = table_[b + 1] = NULL;
        }
      }
      DestroyNode(item);
      --num_elements_;
      if (GOOGLE_PREDICT_FALSE(b == index_of_first_non_null_)) {
        while (index_of_first_non_null_ < num_buckets_ &&
               table_[index_of_first_non_null_] == NULL) {
          ++index_of_first_non_null_;
        }
      }
    }

   private:
    std::pair<const_iterator, size_type> FindHelper(const Key& k) const {
      size_type b = BucketNumber(k);
      if (TableEntryIsNonEmptyList(b)) {
        Node* node = static_cast<Node*>(table_[b]);
        do {
          if (IsMatch(*KeyPtrFromNodePtr(node), k)) {
            return std::make_pair(const_iterator(node, this, b), b);
          } else {
            node = node->next;
          }
        } while (node != NULL);
      } else if (TableEntryIsTree(b)) {
        GOOGLE_DCHECK_EQ(table_[b], table_[b ^ 1]);
        b &= ~static_cast<size_t>(1);
        Tree* tree = static_cast<Tree*>(table_[b]);
        Key* key = const_cast<Key*>(&k);
        typename Tree::iterator tree_it = tree->find(key);
        if (tree_it != tree->end()) {
          return std::make_pair(const_iterator(tree_it, this, b), b);
        }
      }
      return std::make_pair(end(), b);
    }

    // Insert the given Node in bucket b.  If that would make bucket b too big,
    // and bucket b is not a tree, create a tree for buckets b and b^1 to share.
    // Requires count(*KeyPtrFromNodePtr(node)) == 0 and that b is the correct
    // bucket.  num_elements_ is not modified.
    iterator InsertUnique(size_type b, Node* node) {
      GOOGLE_DCHECK(index_of_first_non_null_ == num_buckets_ ||
             table_[index_of_first_non_null_] != NULL);
      // In practice, the code that led to this point may have already
      // determined whether we are inserting into an empty list, a short list,
      // or whatever.  But it's probably cheap enough to recompute that here;
      // it's likely that we're inserting into an empty or short list.
      iterator result;
      GOOGLE_DCHECK(find(*KeyPtrFromNodePtr(node)) == end());
      if (TableEntryIsEmpty(b)) {
        result = InsertUniqueInList(b, node);
      } else if (TableEntryIsNonEmptyList(b)) {
        if (GOOGLE_PREDICT_FALSE(TableEntryIsTooLong(b))) {
          TreeConvert(b);
          result = InsertUniqueInTree(b, node);
          GOOGLE_DCHECK_EQ(result.bucket_index_, b & ~static_cast<size_type>(1));
        } else {
          // Insert into a pre-existing list.  This case cannot modify
          // index_of_first_non_null_, so we skip the code to update it.
          return InsertUniqueInList(b, node);
        }
      } else {
        // Insert into a pre-existing tree.  This case cannot modify
        // index_of_first_non_null_, so we skip the code to update it.
        return InsertUniqueInTree(b, node);
      }
      index_of_first_non_null_ =
          std::min(index_of_first_non_null_, result.bucket_index_);
      return result;
    }

    // Helper for InsertUnique.  Handles the case where bucket b is a
    // not-too-long linked list.
    iterator InsertUniqueInList(size_type b, Node* node) {
      node->next = static_cast<Node*>(table_[b]);
      table_[b] = static_cast<void*>(node);
      return iterator(node, this, b);
    }

    // Helper for InsertUnique.  Handles the case where bucket b points to a
    // Tree.
    iterator InsertUniqueInTree(size_type b, Node* node) {
      GOOGLE_DCHECK_EQ(table_[b], table_[b ^ 1]);
      // Maintain the invariant that node->next is NULL for all Nodes in Trees.
      node->next = NULL;
      return iterator(static_cast<Tree*>(table_[b])
                      ->insert(KeyPtrFromNodePtr(node))
                      .first,
                      this, b & ~static_cast<size_t>(1));
    }

    // Returns whether it did resize.  Currently this is only used when
    // num_elements_ increases, though it could be used in other situations.
    // It checks for load too low as well as load too high: because any number
    // of erases can occur between inserts, the load could be as low as 0 here.
    // Resizing to a lower size is not always helpful, but failing to do so can
    // destroy the expected big-O bounds for some operations. By having the
    // policy that sometimes we resize down as well as up, clients can easily
    // keep O(size()) = O(number of buckets) if they want that.
    bool ResizeIfLoadIsOutOfRange(size_type new_size) {
      const size_type kMaxMapLoadTimes16 = 12;  // controls RAM vs CPU tradeoff
      const size_type hi_cutoff = num_buckets_ * kMaxMapLoadTimes16 / 16;
      const size_type lo_cutoff = hi_cutoff / 4;
      // We don't care how many elements are in trees.  If a lot are,
      // we may resize even though there are many empty buckets.  In
      // practice, this seems fine.
      if (GOOGLE_PREDICT_FALSE(new_size >= hi_cutoff)) {
        if (num_buckets_ <= max_size() / 2) {
          Resize(num_buckets_ * 2);
          return true;
        }
      } else if (GOOGLE_PREDICT_FALSE(new_size <= lo_cutoff &&
                               num_buckets_ > kMinTableSize)) {
        size_type lg2_of_size_reduction_factor = 1;
        // It's possible we want to shrink a lot here... size() could even be 0.
        // So, estimate how much to shrink by making sure we don't shrink so
        // much that we would need to grow the table after a few inserts.
        const size_type hypothetical_size = new_size * 5 / 4 + 1;
        while ((hypothetical_size << lg2_of_size_reduction_factor) <
               hi_cutoff) {
          ++lg2_of_size_reduction_factor;
        }
        size_type new_num_buckets = std::max<size_type>(
            kMinTableSize, num_buckets_ >> lg2_of_size_reduction_factor);
        if (new_num_buckets != num_buckets_) {
          Resize(new_num_buckets);
          return true;
        }
      }
      return false;
    }

    // Resize to the given number of buckets.
    void Resize(size_t new_num_buckets) {
      GOOGLE_DCHECK_GE(new_num_buckets, kMinTableSize);
      void** const old_table = table_;
      const size_type old_table_size = num_buckets_;
      num_buckets_ = new_num_buckets;
      table_ = CreateEmptyTable(num_buckets_);
      const size_type start = index_of_first_non_null_;
      index_of_first_non_null_ = num_buckets_;
      for (size_type i = start; i < old_table_size; i++) {
        if (TableEntryIsNonEmptyList(old_table, i)) {
          TransferList(old_table, i);
        } else if (TableEntryIsTree(old_table, i)) {
          TransferTree(old_table, i++);
        }
      }
      Dealloc<void*>(old_table, old_table_size);
    }

    void TransferList(void* const* table, size_type index) {
      Node* node = static_cast<Node*>(table[index]);
      do {
        Node* next = node->next;
        InsertUnique(BucketNumber(*KeyPtrFromNodePtr(node)), node);
        node = next;
      } while (node != NULL);
    }

    void TransferTree(void* const* table, size_type index) {
      Tree* tree = static_cast<Tree*>(table[index]);
      typename Tree::iterator tree_it = tree->begin();
      do {
        Node* node = NodePtrFromKeyPtr(*tree_it);
        InsertUnique(BucketNumber(**tree_it), node);
      } while (++tree_it != tree->end());
      DestroyTree(tree);
    }

    Node* EraseFromLinkedList(Node* item, Node* head) {
      if (head == item) {
        return head->next;
      } else {
        head->next = EraseFromLinkedList(item, head->next);
        return head;
      }
    }

    bool TableEntryIsEmpty(size_type b) const {
      return TableEntryIsEmpty(table_, b);
    }
    bool TableEntryIsNonEmptyList(size_type b) const {
      return TableEntryIsNonEmptyList(table_, b);
    }
    bool TableEntryIsTree(size_type b) const {
      return TableEntryIsTree(table_, b);
    }
    bool TableEntryIsList(size_type b) const {
      return TableEntryIsList(table_, b);
    }
    static bool TableEntryIsEmpty(void* const* table, size_type b) {
      return table[b] == NULL;
    }
    static bool TableEntryIsNonEmptyList(void* const* table, size_type b) {
      return table[b] != NULL && table[b] != table[b ^ 1];
    }
    static bool TableEntryIsTree(void* const* table, size_type b) {
      return !TableEntryIsEmpty(table, b) &&
          !TableEntryIsNonEmptyList(table, b);
    }
    static bool TableEntryIsList(void* const* table, size_type b) {
      return !TableEntryIsTree(table, b);
    }

    void TreeConvert(size_type b) {
      GOOGLE_DCHECK(!TableEntryIsTree(b) && !TableEntryIsTree(b ^ 1));
      typename Allocator::template rebind<Tree>::other tree_allocator(alloc_);
      Tree* tree = tree_allocator.allocate(1);
      // We want to use the three-arg form of construct, if it exists, but we
      // create a temporary and use the two-arg construct that's known to exist.
      // It's clunky, but the compiler should be able to generate more-or-less
      // the same code.
      tree_allocator.construct(tree,
                               Tree(KeyCompare(), KeyPtrAllocator(alloc_)));
      // Now the tree is ready to use.
      size_type count = CopyListToTree(b, tree) + CopyListToTree(b ^ 1, tree);
      GOOGLE_DCHECK_EQ(count, tree->size());
      table_[b] = table_[b ^ 1] = static_cast<void*>(tree);
    }

    // Copy a linked list in the given bucket to a tree.
    // Returns the number of things it copied.
    size_type CopyListToTree(size_type b, Tree* tree) {
      size_type count = 0;
      Node* node = static_cast<Node*>(table_[b]);
      while (node != NULL) {
        tree->insert(KeyPtrFromNodePtr(node));
        ++count;
        Node* next = node->next;
        node->next = NULL;
        node = next;
      }
      return count;
    }

    // Return whether table_[b] is a linked list that seems awfully long.
    // Requires table_[b] to point to a non-empty linked list.
    bool TableEntryIsTooLong(size_type b) {
      const size_type kMaxLength = 8;
      size_type count = 0;
      Node* node = static_cast<Node*>(table_[b]);
      do {
        ++count;
        node = node->next;
      } while (node != NULL);
      // Invariant: no linked list ever is more than kMaxLength in length.
      GOOGLE_DCHECK_LE(count, kMaxLength);
      return count >= kMaxLength;
    }

    size_type BucketNumber(const Key& k) const {
      // We inherit from hasher, so one-arg operator() provides a hash function.
      size_type h = (*const_cast<InnerMap*>(this))(k);
      // To help prevent people from making assumptions about the hash function,
      // we use the seed differently depending on NDEBUG.  The default hash
      // function, the seeding, etc., are all likely to change in the future.
#ifndef NDEBUG
      return (h * (seed_ | 1)) & (num_buckets_ - 1);
#else
      return (h + seed_) & (num_buckets_ - 1);
#endif
    }

    bool IsMatch(const Key& k0, const Key& k1) const {
      return std::equal_to<Key>()(k0, k1);
    }

    // Return a power of two no less than max(kMinTableSize, n).
    // Assumes either n < kMinTableSize or n is a power of two.
    size_type TableSize(size_type n) {
      return n < kMinTableSize ? kMinTableSize : n;
    }

    // Use alloc_ to allocate an array of n objects of type U.
    template <typename U>
    U* Alloc(size_type n) {
      typedef typename Allocator::template rebind<U>::other alloc_type;
      return alloc_type(alloc_).allocate(n);
    }

    // Use alloc_ to deallocate an array of n objects of type U.
    template <typename U>
    void Dealloc(U* t, size_type n) {
      typedef typename Allocator::template rebind<U>::other alloc_type;
      alloc_type(alloc_).deallocate(t, n);
    }

    void DestroyNode(Node* node) {
      alloc_.destroy(&node->kv);
      Dealloc<Node>(node, 1);
    }

    void DestroyTree(Tree* tree) {
      typename Allocator::template rebind<Tree>::other tree_allocator(alloc_);
      tree_allocator.destroy(tree);
      tree_allocator.deallocate(tree, 1);
    }

    void** CreateEmptyTable(size_type n) {
      GOOGLE_DCHECK(n >= kMinTableSize);
      GOOGLE_DCHECK_EQ(n & (n - 1), 0);
      void** result = Alloc<void*>(n);
      memset(result, 0, n * sizeof(result[0]));
      return result;
    }

    // Return a randomish value.
    size_type Seed() const {
      // random_device can throw, so avoid it unless we are compiling with
      // exceptions enabled.
#if __cpp_exceptions && LANG_CXX11
      try {
        std::random_device rd;
        std::knuth_b knuth(rd());
        std::uniform_int_distribution<size_type> u;
        return u(knuth);
      } catch (...) { }
#endif
      size_type s = static_cast<size_type>(reinterpret_cast<uintptr_t>(this));
#if defined(__x86_64__) && defined(__GNUC__)
      uint32 hi, lo;
      asm("rdtsc" : "=a" (lo), "=d" (hi));
      s += ((static_cast<uint64>(hi) << 32) | lo);
#endif
      return s;
    }

    size_type num_elements_;
    size_type num_buckets_;
    size_type seed_;
    size_type index_of_first_non_null_;
    void** table_;  // an array with num_buckets_ entries
    Allocator alloc_;
    GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(InnerMap);
  };  // end of class InnerMap

  typedef hash_map<Key, value_type*, hash<Key>, std::equal_to<Key>,
                   MapAllocator<std::pair<const Key, MapPair<Key, T>*> > >
      DeprecatedInnerMap;

 public:
  // Iterators
  class iterator_base {
   public:
    // We support "old style" and "new style" iterators for now. This is
    // temporary.  Also, for "iterator()" we have an unknown category.
    // TODO(gpike): get rid of this.
    enum IteratorStyle { kUnknown, kOld, kNew };
    explicit iterator_base(IteratorStyle style) : iterator_style_(style) {}

    bool OldStyle() const {
      GOOGLE_DCHECK_NE(iterator_style_, kUnknown);
      return iterator_style_ == kOld;
    }
    bool UnknownStyle() const {
      return iterator_style_ == kUnknown;
    }
    bool SameStyle(const iterator_base& other) const {
      return iterator_style_ == other.iterator_style_;
    }

   private:
    IteratorStyle iterator_style_;
  };

  class const_iterator
      : private iterator_base,
        public std::iterator<std::forward_iterator_tag, value_type, ptrdiff_t,
                             const value_type*, const value_type&> {
    typedef typename InnerMap::const_iterator InnerIt;
    typedef typename DeprecatedInnerMap::const_iterator DeprecatedInnerIt;

   public:
    const_iterator() : iterator_base(iterator_base::kUnknown) {}
    explicit const_iterator(const DeprecatedInnerIt& dit)
        : iterator_base(iterator_base::kOld), dit_(dit) {}
    explicit const_iterator(const InnerIt& it)
        : iterator_base(iterator_base::kNew), it_(it) {}

    const_iterator(const const_iterator& other)
        : iterator_base(other), it_(other.it_), dit_(other.dit_) {}

    const_reference operator*() const {
      return this->OldStyle() ? *dit_->second : *it_->value();
    }
    const_pointer operator->() const { return &(operator*()); }

    const_iterator& operator++() {
      if (this->OldStyle())
        ++dit_;
      else
        ++it_;
      return *this;
    }
    const_iterator operator++(int) {
      return this->OldStyle() ? const_iterator(dit_++) : const_iterator(it_++);
    }

    friend bool operator==(const const_iterator& a, const const_iterator& b) {
      if (!a.SameStyle(b)) return false;
      if (a.UnknownStyle()) return true;
      return a.OldStyle() ? (a.dit_ == b.dit_) : (a.it_ == b.it_);
    }
    friend bool operator!=(const const_iterator& a, const const_iterator& b) {
      return !(a == b);
    }

   private:
    InnerIt it_;
    DeprecatedInnerIt dit_;
  };

  class iterator : private iterator_base,
                   public std::iterator<std::forward_iterator_tag, value_type> {
    typedef typename InnerMap::iterator InnerIt;
    typedef typename DeprecatedInnerMap::iterator DeprecatedInnerIt;

   public:
    iterator() : iterator_base(iterator_base::kUnknown) {}
    explicit iterator(const DeprecatedInnerIt& dit)
        : iterator_base(iterator_base::kOld), dit_(dit) {}
    explicit iterator(const InnerIt& it)
        : iterator_base(iterator_base::kNew), it_(it) {}

    reference operator*() const {
      return this->OldStyle() ? *dit_->second : *it_->value();
    }
    pointer operator->() const { return &(operator*()); }

    iterator& operator++() {
      if (this->OldStyle())
        ++dit_;
      else
        ++it_;
      return *this;
    }
    iterator operator++(int) {
      return this->OldStyle() ? iterator(dit_++) : iterator(it_++);
    }

    // Allow implicit conversion to const_iterator.
    operator const_iterator() const {
      return this->OldStyle() ?
          const_iterator(typename DeprecatedInnerMap::const_iterator(dit_)) :
          const_iterator(typename InnerMap::const_iterator(it_));
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      if (!a.SameStyle(b)) return false;
      if (a.UnknownStyle()) return true;
      return a.OldStyle() ? a.dit_ == b.dit_ : a.it_ == b.it_;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

   private:
    friend class Map;

    InnerIt it_;
    DeprecatedInnerIt dit_;
  };

  iterator begin() {
    return old_style_ ? iterator(deprecated_elements_->begin())
                      : iterator(elements_->begin());
  }
  iterator end() {
    return old_style_ ? iterator(deprecated_elements_->end())
                      : iterator(elements_->end());
  }
  const_iterator begin() const {
    return old_style_ ? const_iterator(deprecated_elements_->begin())
                      : const_iterator(iterator(elements_->begin()));
  }
  const_iterator end() const {
    return old_style_ ? const_iterator(deprecated_elements_->end())
                      : const_iterator(iterator(elements_->end()));
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Capacity
  size_type size() const {
    return old_style_ ? deprecated_elements_->size() : elements_->size();
  }
  bool empty() const { return size() == 0; }

  // Element access
  T& operator[](const key_type& key) {
    value_type** value =
        old_style_ ? &(*deprecated_elements_)[key] : &(*elements_)[key];
    if (*value == NULL) {
      *value = CreateValueTypeInternal(key);
      internal::MapValueInitializer<google::protobuf::is_proto_enum<T>::value,
                                    T>::Initialize((*value)->second,
                                                   default_enum_value_);
    }
    return (*value)->second;
  }
  const T& at(const key_type& key) const {
    const_iterator it = find(key);
    GOOGLE_CHECK(it != end());
    return it->second;
  }
  T& at(const key_type& key) {
    iterator it = find(key);
    GOOGLE_CHECK(it != end());
    return it->second;
  }

  // Lookup
  size_type count(const key_type& key) const {
    if (find(key) != end()) assert(key == find(key)->first);
    return find(key) == end() ? 0 : 1;
  }
  const_iterator find(const key_type& key) const {
    return old_style_ ? const_iterator(deprecated_elements_->find(key))
        : const_iterator(iterator(elements_->find(key)));
  }
  iterator find(const key_type& key) {
    return old_style_ ? iterator(deprecated_elements_->find(key))
                      : iterator(elements_->find(key));
  }
  std::pair<const_iterator, const_iterator> equal_range(
      const key_type& key) const {
    const_iterator it = find(key);
    if (it == end()) {
      return std::pair<const_iterator, const_iterator>(it, it);
    } else {
      const_iterator begin = it++;
      return std::pair<const_iterator, const_iterator>(begin, it);
    }
  }
  std::pair<iterator, iterator> equal_range(const key_type& key) {
    iterator it = find(key);
    if (it == end()) {
      return std::pair<iterator, iterator>(it, it);
    } else {
      iterator begin = it++;
      return std::pair<iterator, iterator>(begin, it);
    }
  }

  // insert
  std::pair<iterator, bool> insert(const value_type& value) {
    if (old_style_) {
      iterator it = find(value.first);
      if (it != end()) {
        return std::pair<iterator, bool>(it, false);
      } else {
        return std::pair<iterator, bool>(
            iterator(deprecated_elements_->insert(std::pair<Key, value_type*>(
                value.first, CreateValueTypeInternal(value))).first), true);
      }
    } else {
      std::pair<typename InnerMap::iterator, bool> p =
          elements_->insert(value.first);
      if (p.second) {
        p.first->value() = CreateValueTypeInternal(value);
      }
      return std::pair<iterator, bool>(iterator(p.first), p.second);
    }
  }
  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    for (InputIt it = first; it != last; ++it) {
      iterator exist_it = find(it->first);
      if (exist_it == end()) {
        operator[](it->first) = it->second;
      }
    }
  }

  // Erase and clear
  size_type erase(const key_type& key) {
    iterator it = find(key);
    if (it == end()) {
      return 0;
    } else {
      erase(it);
      return 1;
    }
  }
  iterator erase(iterator pos) {
    if (arena_ == NULL) delete pos.operator->();
    iterator i = pos++;
    if (old_style_)
      deprecated_elements_->erase(i.dit_);
    else
      elements_->erase(i.it_);
    return pos;
  }
  void erase(iterator first, iterator last) {
    while (first != last) {
      first = erase(first);
    }
  }
  void clear() { erase(begin(), end()); }

  // Assign
  Map& operator=(const Map& other) {
    if (this != &other) {
      clear();
      insert(other.begin(), other.end());
    }
    return *this;
  }

  void swap(Map& other) {
    if (arena_ == other.arena_ && old_style_ == other.old_style_) {
      std::swap(default_enum_value_, other.default_enum_value_);
      if (old_style_) {
        std::swap(deprecated_elements_, other.deprecated_elements_);
      } else {
        std::swap(elements_, other.elements_);
      }
    } else {
      // TODO(zuguang): optimize this. The temporary copy can be allocated
      // in the same arena as the other message, and the "other = copy" can
      // be replaced with the fast-path swap above.
      Map copy = *this;
      *this = other;
      other = copy;
    }
  }

  // Access to hasher.  Currently this returns a copy, but it may
  // be modified to return a const reference in the future.
  hasher hash_function() const {
    return old_style_ ? deprecated_elements_->hash_function()
                      : elements_->hash_function();
  }

 private:
  // Set default enum value only for proto2 map field whose value is enum type.
  void SetDefaultEnumValue(int default_enum_value) {
    default_enum_value_ = default_enum_value;
  }

  value_type* CreateValueTypeInternal(const Key& key) {
    if (arena_ == NULL) {
      return new value_type(key);
    } else {
      value_type* value = reinterpret_cast<value_type*>(
          Arena::CreateArray<uint8>(arena_, sizeof(value_type)));
      Arena::CreateInArenaStorage(const_cast<Key*>(&value->first), arena_);
      Arena::CreateInArenaStorage(&value->second, arena_);
      const_cast<Key&>(value->first) = key;
      return value;
    }
  }

  value_type* CreateValueTypeInternal(const value_type& value) {
    if (arena_ == NULL) {
      return new value_type(value);
    } else {
      value_type* p = reinterpret_cast<value_type*>(
          Arena::CreateArray<uint8>(arena_, sizeof(value_type)));
      Arena::CreateInArenaStorage(const_cast<Key*>(&p->first), arena_);
      Arena::CreateInArenaStorage(&p->second, arena_);
      const_cast<Key&>(p->first) = value.first;
      p->second = value.second;
      return p;
    }
  }

  Arena* arena_;
  int default_enum_value_;
  // The following is a tagged union because we support two map styles
  // for now.
  // TODO(gpike): get rid of the old style.
  const bool old_style_;
  union {
    InnerMap* elements_;
    DeprecatedInnerMap* deprecated_elements_;
  };

  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  template <typename K, typename V,
            internal::WireFormatLite::FieldType key_wire_type,
            internal::WireFormatLite::FieldType value_wire_type,
            int default_enum_value>
  friend class internal::MapFieldLite;
};

}  // namespace protobuf
}  // namespace google

GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_START
template<>
struct hash<google::protobuf::MapKey> {
  size_t
  operator()(const google::protobuf::MapKey& map_key) const {
    switch (map_key.type()) {
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Unsupported";
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
        return hash<string>()(map_key.GetStringValue());
      case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
        return hash< ::google::protobuf::int64>()(map_key.GetInt64Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
        return hash< ::google::protobuf::int32>()(map_key.GetInt32Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
        return hash< ::google::protobuf::uint64>()(map_key.GetUInt64Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
        return hash< ::google::protobuf::uint32>()(map_key.GetUInt32Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        return hash<bool>()(map_key.GetBoolValue());
    }
    GOOGLE_LOG(FATAL) << "Can't get here.";
    return 0;
  }
  bool
  operator()(const google::protobuf::MapKey& map_key1,
             const google::protobuf::MapKey& map_key2) const {
    return map_key1 < map_key2;
  }
};
GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_END

#endif  // GOOGLE_PROTOBUF_MAP_H__
