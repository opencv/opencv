/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_GPU_BORDER_INTERPOLATE_HPP__
#define __OPENCV_GPU_BORDER_INTERPOLATE_HPP__

#include "saturate_cast.hpp"
#include "vec_traits.hpp"

namespace cv { namespace gpu { namespace device
{
    //////////////////////////////////////////////////////////////
    // BrdConstant

    template <typename D> struct BrdRowConstant
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdRowConstant(int width_, const D& val_ = VecTraits<D>::all(0)) : width(width_), val(val_) {}

        template <typename T> __device__ __forceinline__ D at_low(int x, const T* data) const 
        {
            return x >= 0 ? saturate_cast<D>(data[x]) : val;
        }

        template <typename T> __device__ __forceinline__ D at_high(int x, const T* data) const 
        {
            return x < width ? saturate_cast<D>(data[x]) : val;
        }

        template <typename T> __device__ __forceinline__ D at(int x, const T* data) const 
        {
            return (x >= 0 && x < width) ? saturate_cast<D>(data[x]) : val;
        }

        __host__ __device__ __forceinline__ bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

        const int width;
        const D val;
    };

    template <typename D> struct BrdColConstant
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdColConstant(int height_, const D& val_ = VecTraits<D>::all(0)) : height(height_), val(val_) {}

        template <typename T> __device__ __forceinline__ D at_low(int y, const T* data, size_t step) const 
        {
            return y >= 0 ? saturate_cast<D>(*(const T*)((const char*)data + y * step)) : val;
        }

        template <typename T> __device__ __forceinline__ D at_high(int y, const T* data, size_t step) const 
        {
            return y < height ? saturate_cast<D>(*(const T*)((const char*)data + y * step)) : val;
        }

        template <typename T> __device__ __forceinline__ D at(int y, const T* data, size_t step) const 
        {
            return (y >= 0 && y < height) ? saturate_cast<D>(*(const T*)((const char*)data + y * step)) : val;
        }

        __host__ __device__ __forceinline__ bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

        const int height;
        const D val;
    };

    template <typename D> struct BrdConstant
    {
        typedef D result_type;

        __host__ __device__ __forceinline__ BrdConstant(int height_, int width_, const D& val_ = VecTraits<D>::all(0)) : 
            height(height_), width(width_), val(val_) 
        {
        }

        template <typename T> __device__ __forceinline__ D at(int y, int x, const T* data, size_t step) const
        {
            return (x >= 0 && x < width && y >= 0 && y < height) ? saturate_cast<D>(((const T*)((const uchar*)data + y * step))[x]) : val;
        }

        template <typename Ptr2D> __device__ __forceinline__ D at(typename Ptr2D::index_type y, typename Ptr2D::index_type x, const Ptr2D& src) const
        {
            return (x >= 0 && x < width && y >= 0 && y < height) ? saturate_cast<D>(src(y, x)) : val;
        }

        const int height;
        const int width;
        const D val;
    };

    //////////////////////////////////////////////////////////////
    // BrdReplicate

    template <typename D> struct BrdRowReplicate
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdRowReplicate(int width) : last_col(width - 1) {}
        template <typename U> __host__ __device__ __forceinline__ BrdRowReplicate(int width, U) : last_col(width - 1) {}

        __device__ __forceinline__ int idx_col_low(int x) const
        {
            return ::max(x, 0);
        }

        __device__ __forceinline__ int idx_col_high(int x) const 
        {
            return ::min(x, last_col);
        }

        __device__ __forceinline__ int idx_col(int x) const
        {
            return idx_col_low(idx_col_high(x));
        }

        template <typename T> __device__ __forceinline__ D at_low(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col_low(x)]);
        }

        template <typename T> __device__ __forceinline__ D at_high(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col_high(x)]);
        }

        template <typename T> __device__ __forceinline__ D at(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col(x)]);
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

        const int last_col;
    };

    template <typename D> struct BrdColReplicate
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdColReplicate(int height) : last_row(height - 1) {}
        template <typename U> __host__ __device__ __forceinline__ BrdColReplicate(int height, U) : last_row(height - 1) {}

        __device__ __forceinline__ int idx_row_low(int y) const
        {
            return ::max(y, 0);
        }

        __device__ __forceinline__ int idx_row_high(int y) const 
        {
            return ::min(y, last_row);
        }

        __device__ __forceinline__ int idx_row(int y) const
        {
            return idx_row_low(idx_row_high(y));
        }

        template <typename T> __device__ __forceinline__ D at_low(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const T*)((const char*)data + idx_row_low(y) * step));
        }

        template <typename T> __device__ __forceinline__ D at_high(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const T*)((const char*)data + idx_row_high(y) * step));
        }

        template <typename T> __device__ __forceinline__ D at(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const T*)((const char*)data + idx_row(y) * step));
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

        const int last_row;
    };

    template <typename D> struct BrdReplicate
    {
        typedef D result_type;

        __host__ __device__ __forceinline__ BrdReplicate(int height, int width) : 
            last_row(height - 1), last_col(width - 1) 
        {
        }
        template <typename U> 
        __host__ __device__ __forceinline__ BrdReplicate(int height, int width, U) : 
            last_row(height - 1), last_col(width - 1) 
        {
        }

        __device__ __forceinline__ int idx_row_low(int y) const
        {
            return ::max(y, 0);
        }
        __device__ __forceinline__ float idx_row_low(float y) const
        {
            return ::fmax(y, 0.0f);
        }

        __device__ __forceinline__ int idx_row_high(int y) const 
        {
            return ::min(y, last_row);
        }
        __device__ __forceinline__ float idx_row_high(float y) const 
        {
            return ::fmin(y, last_row);
        }

        __device__ __forceinline__ int idx_row(int y) const
        {
            return idx_row_low(idx_row_high(y));
        }
        __device__ __forceinline__ float idx_row(float y) const
        {
            return idx_row_low(idx_row_high(y));
        }

        __device__ __forceinline__ int idx_col_low(int x) const
        {
            return ::max(x, 0);
        }
        __device__ __forceinline__ float idx_col_low(float x) const
        {
            return ::fmax(x, 0);
        }

        __device__ __forceinline__ int idx_col_high(int x) const 
        {
            return ::min(x, last_col);
        }
        __device__ __forceinline__ float idx_col_high(float x) const 
        {
            return ::fmin(x, last_col);
        }

        __device__ __forceinline__ int idx_col(int x) const
        {
            return idx_col_low(idx_col_high(x));
        }
        __device__ __forceinline__ float idx_col(float x) const
        {
            return idx_col_low(idx_col_high(x));
        }

        template <typename T> __device__ __forceinline__ D at(int y, int x, const T* data, size_t step) const 
        {
            return saturate_cast<D>(((const T*)((const char*)data + idx_row(y) * step))[idx_col(x)]);
        }

        template <typename Ptr2D> __device__ __forceinline__ D at(typename Ptr2D::index_type y, typename Ptr2D::index_type x, const Ptr2D& src) const 
        {
            return saturate_cast<D>(src(idx_row(y), idx_col(x)));
        }

        const int last_row;
        const int last_col;
    };

    //////////////////////////////////////////////////////////////
    // BrdReflect101

    template <typename D> struct BrdRowReflect101
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdRowReflect101(int width) : last_col(width - 1) {}
        template <typename U> __host__ __device__ __forceinline__ BrdRowReflect101(int width, U) : last_col(width - 1) {}

        __device__ __forceinline__ int idx_col_low(int x) const
        {
            return ::abs(x);
        }

        __device__ __forceinline__ int idx_col_high(int x) const 
        {
            return last_col - ::abs(last_col - x);
        }

        __device__ __forceinline__ int idx_col(int x) const
        {
            return idx_col_low(idx_col_high(x));
        }

        template <typename T> __device__ __forceinline__ D at_low(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col_low(x)]);
        }

        template <typename T> __device__ __forceinline__ D at_high(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col_high(x)]);
        }

        template <typename T> __device__ __forceinline__ D at(int x, const T* data) const 
        {
            return saturate_cast<D>(data[idx_col(x)]);
        }

        __host__ __device__ __forceinline__ bool is_range_safe(int mini, int maxi) const 
        {
            return -last_col <= mini && maxi <= 2 * last_col;
        }

        const int last_col;
    };

    template <typename D> struct BrdColReflect101
    {
        typedef D result_type;

        explicit __host__ __device__ __forceinline__ BrdColReflect101(int height) : last_row(height - 1) {}
        template <typename U> __host__ __device__ __forceinline__ BrdColReflect101(int height, U) : last_row(height - 1) {}

        __device__ __forceinline__ int idx_row_low(int y) const
        {
            return ::abs(y);
        }

        __device__ __forceinline__ int idx_row_high(int y) const 
        {
            return last_row - ::abs(last_row - y);
        }

        __device__ __forceinline__ int idx_row(int y) const
        {
            return idx_row_low(idx_row_high(y));
        }

        template <typename T> __device__ __forceinline__ D at_low(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const D*)((const char*)data + idx_row_low(y) * step));
        }

        template <typename T> __device__ __forceinline__ D at_high(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const D*)((const char*)data + idx_row_high(y) * step));
        }

        template <typename T> __device__ __forceinline__ D at(int y, const T* data, size_t step) const 
        {
            return saturate_cast<D>(*(const D*)((const char*)data + idx_row(y) * step));
        }

        __host__ __device__ __forceinline__ bool is_range_safe(int mini, int maxi) const 
        {
            return -last_row <= mini && maxi <= 2 * last_row;
        }

        const int last_row;
    };

    template <typename D> struct BrdReflect101
    {
        typedef D result_type;

        __host__ __device__ __forceinline__ BrdReflect101(int height, int width) : 
            last_row(height - 1), last_col(width - 1) 
        {
        }
        template <typename U> 
        __host__ __device__ __forceinline__ BrdReflect101(int height, int width, U) : 
            last_row(height - 1), last_col(width - 1) 
        {
        }

        __device__ __forceinline__ int idx_row_low(int y) const
        {
            return ::abs(y);
        }
        __device__ __forceinline__ float idx_row_low(float y) const
        {
            return ::fabs(y);
        }

        __device__ __forceinline__ int idx_row_high(int y) const 
        {
            return last_row - ::abs(last_row - y);
        }
        __device__ __forceinline__ float idx_row_high(float y) const 
        {
            return last_row - ::fabs(last_row - y);
        }

        __device__ __forceinline__ int idx_row(int y) const
        {
            return idx_row_low(idx_row_high(y));
        }
        __device__ __forceinline__ float idx_row(float y) const
        {
            return idx_row_low(idx_row_high(y));
        }

        __device__ __forceinline__ int idx_col_low(int x) const
        {
            return ::abs(x);
        }
        __device__ __forceinline__ float idx_col_low(float x) const
        {
            return ::fabs(x);
        }

        __device__ __forceinline__ int idx_col_high(int x) const 
        {
            return last_col - ::abs(last_col - x);
        }
        __device__ __forceinline__ float idx_col_high(float x) const 
        {
            return last_col - ::fabs(last_col - x);
        }

        __device__ __forceinline__ int idx_col(int x) const
        {
            return idx_col_low(idx_col_high(x));
        }
        __device__ __forceinline__ float idx_col(float x) const
        {
            return idx_col_low(idx_col_high(x));
        }

        template <typename T> __device__ __forceinline__ D at(int y, int x, const T* data, size_t step) const 
        {
            return saturate_cast<D>(((const T*)((const char*)data + idx_row(y) * step))[idx_col(x)]);
        }

        template <typename Ptr2D> __device__ __forceinline__ D at(typename Ptr2D::index_type y, typename Ptr2D::index_type x, const Ptr2D& src) const 
        {
            return saturate_cast<D>(src(idx_row(y), idx_col(x)));
        }

        const int last_row;
        const int last_col;
    };

    //////////////////////////////////////////////////////////////
    // BorderReader

    template <typename Ptr2D, typename B> struct BorderReader
    {
        typedef typename B::result_type elem_type;
        typedef typename Ptr2D::index_type index_type;

        __host__ __device__ __forceinline__ BorderReader(const Ptr2D& ptr_, const B& b_) : ptr(ptr_), b(b_) {}

        __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const
        {
            return b.at(y, x, ptr);
        }

        const Ptr2D ptr;
        const B b;
    };
}}}

#endif // __OPENCV_GPU_BORDER_INTERPOLATE_HPP__
