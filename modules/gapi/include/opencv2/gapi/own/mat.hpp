// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_MAT_HPP
#define OPENCV_GAPI_OWN_MAT_HPP

#include "opencv2/gapi/opencv_includes.hpp"
#include "opencv2/gapi/own/types.hpp"
#include "opencv2/gapi/own/scalar.hpp"
#include "opencv2/gapi/own/saturate.hpp"
#include "opencv2/gapi/own/assert.hpp"

#include <memory>                   //std::shared_ptr
#include <cstring>                  //std::memcpy
#include "opencv2/gapi/util/throw.hpp"

namespace cv { namespace gapi { namespace own {
    namespace detail {
        template <typename T, unsigned char channels>
        void assign_row(void* ptr, int cols, Scalar const& s)
        {
            auto p = static_cast<T*>(ptr);
            for (int c = 0; c < cols; c++)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    p[c * channels + ch] = saturate<T>(s[ch], roundd);
                }
            }
        }

        inline size_t default_step(int type, int cols)
        {
            return CV_ELEM_SIZE(type) * cols;
        }
        //Matrix header, i.e. fields that are unique to each Mat object.
        //Devoted class is needed to implement custom behavior on move (erasing state of moved from object)
        struct MatHeader{
            enum { AUTO_STEP = 0};
            enum { TYPE_MASK = 0x00000FFF  };

            MatHeader() = default;

            MatHeader(int _rows, int _cols, int type, void* _data, size_t _step)
            : flags((type & TYPE_MASK)), rows(_rows), cols(_cols), data((uchar*)_data), step(_step == AUTO_STEP ? detail::default_step(type, _cols) : _step)
            {}

            MatHeader(const MatHeader& ) = default;
            MatHeader(MatHeader&& src) : MatHeader(src) // reuse copy constructor here
            {
                MatHeader empty; //give it a name to call copy(not move) assignment below
                src = empty;
            }
            MatHeader& operator=(const MatHeader& ) = default;
            MatHeader& operator=(MatHeader&& src)
            {
                *this = src; //calling a copy assignment here, not move one
                MatHeader empty; //give it a name to call copy(not move) assignment below
                src = empty;
                return *this;
            }
            /*! includes several bit-fields:
                 - depth
                 - number of channels
             */
            int flags = 0;

            //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
            int rows = 0, cols = 0;
            //! pointer to the data
            uchar* data = nullptr;
            size_t step = 0;
        };
    }
    //concise version of cv::Mat suitable for GAPI needs (used when no dependence on OpenCV is required)
    class Mat : public detail::MatHeader{
    public:

        Mat() = default;

        /** @overload
        @param _rows Number of rows in a 2D array.
        @param _cols Number of columns in a 2D array.
        @param _type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
        CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
        @param _data Pointer to the user data. Matrix constructors that take data and step parameters do not
        allocate matrix data. Instead, they just initialize the matrix header that points to the specified
        data, which means that no data is copied. This operation is very efficient and can be used to
        process external data using OpenCV functions. The external data is not automatically deallocated, so
        you should take care of it.
        @param _step Number of bytes each matrix row occupies. The value should include the padding bytes at
        the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
        and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
        */
        Mat(int _rows, int _cols, int _type, void* _data, size_t _step = AUTO_STEP)
        : MatHeader (_rows, _cols, _type, _data, _step)
        {}

        Mat(Mat const& src, const Rect& roi )
        : Mat(src)
        {
           rows = roi.height;
           cols = roi.width;
           data = ptr(roi.y, roi.x);
        }

        Mat(Mat const& src) = default;
        Mat(Mat&& src) = default;

        Mat& operator=(Mat const& src) = default;
        Mat& operator=(Mat&& src) = default;

        /** @brief Sets all or some of the array elements to the specified value.
        @param s Assigned scalar converted to the actual array type.
        */
        Mat& operator = (const Scalar& s)
        {
            constexpr unsigned max_channels = 4; //Scalar can't fit more than 4
            const auto channels = static_cast<unsigned int>(this->channels());
            GAPI_Assert(channels <= max_channels);

            using func_p_t = void (*)(void*, int, Scalar const&);
            using detail::assign_row;
            #define TABLE_ENTRY(type)  {assign_row<type, 1>, assign_row<type, 2>, assign_row<type, 3>, assign_row<type, 4>}
            static constexpr func_p_t func_tbl[][max_channels] = {
                    TABLE_ENTRY(uchar),
                    TABLE_ENTRY(schar),
                    TABLE_ENTRY(ushort),
                    TABLE_ENTRY(short),
                    TABLE_ENTRY(int),
                    TABLE_ENTRY(float),
                    TABLE_ENTRY(double)
            };
            #undef TABLE_ENTRY

            static_assert(CV_8U == 0 && CV_8S == 1  && CV_16U == 2 && CV_16S == 3
                       && CV_32S == 4 && CV_32F == 5 && CV_64F == 6,
                       "OCV type ids used as indexes to array, thus exact numbers are important!"
            );

            const auto depth = static_cast<unsigned int>(this->depth());
            GAPI_Assert(depth < sizeof(func_tbl)/sizeof(func_tbl[0]));

            for (int r = 0; r < rows; ++r)
            {
                auto* f = func_tbl[depth][channels -1];
                (*f)(static_cast<void *>(ptr(r)), cols, s );
            }
            return *this;
        }

        /** @brief Returns the matrix element size in bytes.

        The method returns the matrix element size in bytes. For example, if the matrix type is CV_16SC3 ,
        the method returns 3\*sizeof(short) or 6.
         */
        size_t elemSize() const
        {
            return CV_ELEM_SIZE(type());
        }
        /** @brief Returns the type of a matrix element.

        The method returns a matrix element type. This is an identifier compatible with the CvMat type
        system, like CV_16SC3 or 16-bit signed 3-channel array, and so on.
         */
        int type() const            {return CV_MAT_TYPE(flags);}

        /** @brief Returns the depth of a matrix element.

        The method returns the identifier of the matrix element depth (the type of each individual channel).
        For example, for a 16-bit signed element array, the method returns CV_16S . A complete list of
        matrix types contains the following values:
        -   CV_8U - 8-bit unsigned integers ( 0..255 )
        -   CV_8S - 8-bit signed integers ( -128..127 )
        -   CV_16U - 16-bit unsigned integers ( 0..65535 )
        -   CV_16S - 16-bit signed integers ( -32768..32767 )
        -   CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
        -   CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
        -   CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
         */
        int depth() const           {return CV_MAT_DEPTH(flags);}

        /** @brief Returns the number of matrix channels.

        The method returns the number of matrix channels.
         */
        int channels() const        {return CV_MAT_CN(flags);}

        /**
        @param _rows New number of rows.
        @param _cols New number of columns.
        @param _type New matrix type.
         */
        void create(int _rows, int _cols, int _type)
        {
            create({_cols, _rows}, _type);
        }
        /** @overload
        @param _size Alternative new matrix size specification: Size(cols, rows)
        @param _type New matrix type.
        */
        void create(Size _size, int _type)
        {
            if (_size != Size{cols, rows} )
            {
                Mat tmp{_size.height, _size.width, _type, nullptr};
                tmp.memory.reset(new uchar[ tmp.step * tmp.rows], [](uchar * p){delete[] p;});
                tmp.data = tmp.memory.get();

                *this = std::move(tmp);
            }
        }

        /** @brief Copies the matrix to another one.

        The method copies the matrix data to another matrix. Before copying the data, the method invokes :
        @code
            m.create(this->size(), this->type());
        @endcode
        so that the destination matrix is reallocated if needed. While m.copyTo(m); works flawlessly, the
        function does not handle the case of a partial overlap between the source and the destination
        matrices.
         */
        void copyTo(Mat& dst) const
        {
            dst.create(rows, cols, type());
            for (int r = 0; r < rows; ++r)
            {
                std::copy_n(ptr(r), detail::default_step(type(),cols), dst.ptr(r));
            }
        }

        /** @brief Returns true if the array has no elements.

        The method returns true if Mat::total() is 0 or if Mat::data is NULL. Because of pop_back() and
        resize() methods `M.total() == 0` does not imply that `M.data == NULL`.
         */
        bool empty() const;

        /** @brief Returns the total number of array elements.

        The method returns the number of array elements (a number of pixels if the array represents an
        image).
         */
        size_t total() const
        {
            return static_cast<size_t>(rows * cols);
        }


        /** @overload
        @param roi Extracted submatrix specified as a rectangle.
        */
        Mat operator()( const Rect& roi ) const
        {
            return Mat{*this, roi};
        }


        /** @brief Returns a pointer to the specified matrix row.

        The methods return `uchar*` or typed pointer to the specified matrix row. See the sample in
        Mat::isContinuous to know how to use these methods.
        @param row Index along the dimension 0
        @param col Index along the dimension 1
        */
        uchar* ptr(int row, int col = 0)
        {
            return const_cast<uchar*>(const_cast<const Mat*>(this)->ptr(row,col));
        }
        /** @overload */
        const uchar* ptr(int row, int col = 0) const
        {
            return data + step * row + CV_ELEM_SIZE(type()) * col;
        }


    private:
        //actual memory allocated for storage, or nullptr if object is non owning view to over memory
        std::shared_ptr<uchar> memory;
    };

} //namespace own
} //namespace gapi
} //namespace cv

#endif /* OPENCV_GAPI_OWN_MAT_HPP */
