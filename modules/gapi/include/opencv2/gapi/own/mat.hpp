// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef INCLUDE_OPENCV2_GAPI_OWN_MAT_HPP
#define INCLUDE_OPENCV2_GAPI_OWN_MAT_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/gapi/own/types.hpp"
#include <memory>                   //std::shared_ptr

namespace cv { namespace gapi { namespace own {
    namespace detail {
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

        Mat(Mat const& src) = default;
        Mat(Mat&& src) = default;

        Mat& operator=(Mat const& src) = default;
        Mat& operator=(Mat&& src) = default;

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

        /** @overload
        @param _size Alternative new matrix size specification: Size(cols, rows)
        @param _type New matrix type.
        */
        void create(cv::gapi::own::Size _size, int _type)
        {
            if (_size != cv::gapi::own::Size{cols, rows} )
            {
                Mat tmp{_size.height, _size.width, _type, nullptr};
                tmp.memory.reset(new uchar[ tmp.step * tmp.rows], [](uchar * p){delete[] p;});
                tmp.data = tmp.memory.get();

                *this = std::move(tmp);
            }
        }
    private:
        //actual memory allocated for storage, or nullptr if object is non owning view to over memory
        std::shared_ptr<uchar> memory;
    };

} //namespace own
} //namespace gapi
} //namespace cv

#endif /* INCLUDE_OPENCV2_GAPI_OWN_MAT_HPP */
