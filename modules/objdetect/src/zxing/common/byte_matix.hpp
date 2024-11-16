#ifndef __BYTE_MATRIX_H__
#define __BYTE_MATRIX_H__

#include "counted.hpp"
#include "bit_array.hpp"
#include "array.hpp"
#include <limits>

namespace zxing {

	class ByteMatix : public Counted {
	public:
		ByteMatix(int dimension);
		ByteMatix(int width, int height);
		ByteMatix(int width, int height, ArrayRef<char> source);
		~ByteMatix();

		char get(int x, int y) const {
			int offset =row_offsets[y] + x;
			return bytes[offset] & 0xFF;
		}
		
		void set(int x, int y, char char_value){
			int offset=row_offsets[y]+x;
			bytes[offset]=char_value& 0xFF;
		}

		ArrayRef<char> getRow(int y, ArrayRef<char> row);

		int getWidth() const { return width; }
		int getHeight() const {return height;}

	private:
		int width;
		int height;
		ArrayRef<char> bytes;
		ArrayRef<int> row_offsets;

	private:
	    inline void init(int, int);
		ByteMatix(const ByteMatix&);
		ByteMatix& operator =(const ByteMatix&);
	};

}

#endif

