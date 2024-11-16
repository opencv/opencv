#ifndef __BYTE_MATRIX_H__
#define __BYTE_MATRIX_H__

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

#endif

