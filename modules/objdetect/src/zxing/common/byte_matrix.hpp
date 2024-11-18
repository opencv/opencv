// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_BYTE_MATRIX_HPP__
#define __ZXING_COMMON_BYTE_MATRIX_HPP__

#include "counted.hpp"
#include "bit_array.hpp"
#include "array.hpp"
#include <limits>
#include "../error_handler.hpp"

namespace zxing {

class ByteMatrix : public Counted {
public:
    ByteMatrix(int dimension);
    ByteMatrix(int width, int height);
    ByteMatrix(int width, int height, ArrayRef<char> source);
    ~ByteMatrix();
    
    char get(int x, int y) const {
        int offset =row_offsets[y] + x;
        return bytes[offset];
    }
    
    void set(int x, int y, char char_value){
        int offset=row_offsets[y]+x;
        bytes[offset]=char_value&0XFF;
    }
    
    unsigned char* getByteRow(int y, ErrorHandler & err_handler);
    
    int getWidth() const { return width; }
    int getHeight() const {return height;}
    
    unsigned char* bytes;
    
private:
    int width;
    int height;
    
    int* row_offsets;
    
private:
    inline void init(int, int);
    ByteMatrix(const ByteMatrix&);
    ByteMatrix& operator =(const ByteMatrix&);
};

}

#endif // __ZXING_COMMON_BYTE_MATRIX_HPP__

