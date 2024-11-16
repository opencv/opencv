// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __BIT_MATRIX_H__
#define __BIT_MATRIX_H__

/*
 *  BitMatrix.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
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

#include "counted.hpp"
#include "bit_array.hpp"
#include "array.hpp"
#include <limits>
#include <iostream>
#include "../error_handler.hpp"

namespace zxing {

class BitMatrix : public Counted {
public:
    static const int bitsPerWord = std::numeric_limits<unsigned int>::digits;
    
private:
    int width;
    int height;
    int rowBitsSize;
    
    std::vector<COUNTER_TYPE> row_counters;
    std::vector<COUNTER_TYPE> row_counters_offset;
    std::vector<bool> row_counters_recorded;
    std::vector<COUNTER_TYPE> row_counter_offset_end;
    std::vector<COUNTER_TYPE> row_point_offset;
    
    std::vector<COUNTER_TYPE> cols_counters;
    std::vector<COUNTER_TYPE> cols_counters_offset;
    std::vector<bool> cols_counters_recorded;
    std::vector<COUNTER_TYPE> cols_counter_offset_end;
    std::vector<COUNTER_TYPE> cols_point_offset;
    
    ArrayRef<unsigned char> bits;
    ArrayRef<int> rowOffsets;
    
public:
    BitMatrix(int width, int height, bool* bitsPtr, ErrorHandler & err_handler);
    BitMatrix(int dimension, ErrorHandler &err_handler);
    BitMatrix(int width, int height, ErrorHandler & err_handler);
    
    void copyOf(Ref<BitMatrix> bits_, ErrorHandler &err_handler);
    bool copyOf2(Ref<BitMatrix> bits_);
    void xxor(Ref<BitMatrix> bits_);
    
    void fillHoles();
    
    ~BitMatrix();
    
    bool get(int x, int y)const{
        return (bool)bits[width * y + x];
    }
    
    void set(int x, int y){
        bits[rowOffsets[y] + x] = true;
    }
    
    void set(int x, int y, bool value){
        bits[rowOffsets[y] + x] = value;
    }
    
    void swap(int srcX, int srcY, int dstX, int dstY)
    {
        bool temp = bits[width * srcY + srcX];
        bits[width * srcY + srcX] = bits[width * dstY + dstX];
        bits[width * dstY + dstX] = temp;
    }
    
    void getRowBool(int y,bool* row);
    bool* getRowBoolPtr(int y);
    void setRowBool(int y, bool* row);
    int getRowBitsSize() {return rowBitsSize;}
    unsigned char* getPtr(){return bits->data();}
    
    void flip(int x, int y);
    void flipAll();
    void clear();
    void setRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler);
    void flipRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler);
    void randomFlipRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler);
    Ref<BitArray> getRow(int y, Ref<BitArray> row);
    
    int getWidth() const;
    int getHeight() const;
    
    ArrayRef<int> getTopLeftOnBit() const;
    ArrayRef<int> getTopLeftOnBitNew() const;
    ArrayRef<int> getBottomRightOnBit() const;
    ArrayRef<int> getBottomRightOnBitNew() const;
    ArrayRef<int> getTopRightOnBitNew() const;
    ArrayRef<int> getBottomLeftOnBitNew() const;
    
    ArrayRef<int> getPointOnRight(int x, int y, int, bool);
    ArrayRef<int> getPointOnLeft(int x, int y, int, bool);
    ArrayRef<int> getPointOnTop(int x, int y, int, bool);
    ArrayRef<int> getPointOnBottom(int x, int y, int, bool);
    
    bool isInitRowCounters;
    void initRowCounters();
    COUNTER_TYPE* getRowRecords(int y);
    COUNTER_TYPE* getRowRecordsOffset(int y);
    bool getRowFirstIsWhite(int y);
    COUNTER_TYPE getRowCounterOffsetEnd(int y);
    bool getRowLastIsWhite(int y);
    COUNTER_TYPE* getRowPointInRecords(int y);
    
    bool isInitColsCounters;
    void initColsCounters();
    COUNTER_TYPE* getColsRecords(int x);
    COUNTER_TYPE* getColsRecordsOffset(int x);
    COUNTER_TYPE* getColsPointInRecords(int x);
    COUNTER_TYPE getColsCounterOffsetEnd(int x);
    
private:
    inline void init(int, int , ErrorHandler & err_handler);
    inline void init(int width_, int height_, bool* bitsPtr, ErrorHandler & err_handler);
    
    void setRowRecords(int y);
    void setColsRecords(int x);
    
    BitMatrix(const BitMatrix&, ErrorHandler & err_handler);
};

}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_COMMON_BITMATRIX_H_ 
