// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_COMMON_BIT_ARRAY_HPP__
#define __ZXING_COMMON_BIT_ARRAY_HPP__

/*
 *  Copyright 2010 ZXing authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../zxing.hpp"
#include "counted.hpp"
#include "illegal_argument_exception.hpp"
#include "array.hpp"
#include <vector>
#include <limits>
#include <iostream>
#include <cstring>
#include "../error_handler.hpp"

namespace zxing {

class BitArray : public Counted {
private: 
    int size;
    ArrayRef<unsigned char> bits;
    ArrayRef<int> nextSets;
    ArrayRef<int> nextUnSets;
    
public:
    BitArray(int size);
    ~BitArray();
    int getSize() const;
    
    
    bool get(int i) const{
        return bits[i] != 0;
    }
    void set(int i){
        bits[i] = true;
    }
    void setOneRow(unsigned char* rowBits, int length)
    {
        unsigned char* dst = bits->data();
        memcpy(dst, rowBits, length);
    }
    
    bool* getRowBoolPtr()
    {
        return (bool*)bits->data();
    }
    
    // init for next sets and unsets to speed up
    // By Valiantliu
    void initAllNextSets();
    void initAllNextSetsFromCounters(std::vector<int> counters);
    
    int getNextSet(int from);
    int getNextUnset(int from);
    
    void setUnchar(int i, unsigned char newBist);
    
    void clear();
    bool isRange(int start, int end, bool value, ErrorHandler & err_handler);
    
    void reverse();
    
    class Reverse {
    private:
        Ref<BitArray> array;
    public:
        Reverse(Ref<BitArray> array);
        ~Reverse();
    };
    
    void appendBit(bool value);
    int getSizeInBytes() const;
    void appendBits(int value, int numberOfBits, ErrorHandler & err_handler);
    void appendBitArray(const BitArray& array);
    void toBytes(int bitOffset, ArrayRef<int>& array, int offset, int numBytes);
    void bitXOR(const BitArray& other, ErrorHandler & err_handler);
    
#ifndef USE_BYTE_FOR_BIT
private:
    static int makeArraySize(int size);
#endif
};

std::ostream& operator << (std::ostream&, BitArray const&);

}  // namespace zxing

#endif  // __ZXING_COMMON_BIT_ARRAY_HPP__
