////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef OPENCV_MAP_HPP_
#define OPENCV_MAP_HPP_

namespace matlab {
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

} // namespace matlab

#endif
#endif
