#ifndef OPENCV_MAP_HPP_
#define OPENCV_MAP_HPP_

#if __cplusplus >= 201103L

 // If we have C++11 support, we just want to use unordered_map 
#include <unordered_map>
template <typename KeyType, typename ValueType>
using Map = std::unordered_map<KeyType, ValueType>;

#else

// If we don't have C++11 support, we wrap another map implementation
// in the same public API as unordered_map
#include <map>
#include <stdexcept>

template <typename KeyType, typename ValueType>
class Map {
private:
  std::map<KeyType, ValueType> map_;
public:
  // map[key] = val;
  ValueType& operator[] (const KeyType& k) {
    return map_[k];
  }

  // map.at(key) = val (throws)
  ValueType& at(const KeyType& k) {
    typename std::map<KeyType, ValueType>::iterator it;
    it = map_.find(k);
    if (it == map_.end()) throw std::out_of_range("Key not found");
    return *it;
  }

  // val = map.at(key)  (throws, const)
  const ValueType& at(const KeyType& k) const {
    typename std::map<KeyType, ValueType>::const_iterator it;
    it = map_.find(k);
    if (it == map_.end()) throw std::out_of_range("Key not found");
    return *it;
  }
};

#endif
#endif
