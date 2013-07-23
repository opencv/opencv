/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// any express or implied warranties, including, but not limited to, the implied
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

cv::cuda::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(rows_), cols(cols_),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert( step >= minstep );

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }

    dataend += step * (rows - 1) + minstep;
}

cv::cuda::GpuMat::GpuMat(Size size_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(size_.height), cols(size_.width),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert( step >= minstep );

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::cuda::GpuMat::GpuMat(const GpuMat& m, Range rowRange_, Range colRange_)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (rowRange_ == Range::all())
    {
        rows = m.rows;
    }
    else
    {
        CV_Assert( 0 <= rowRange_.start && rowRange_.start <= rowRange_.end && rowRange_.end <= m.rows );

        rows = rowRange_.size();
        data += step*rowRange_.start;
    }

    if (colRange_ == Range::all())
    {
        cols = m.cols;
    }
    else
    {
        CV_Assert( 0 <= colRange_.start && colRange_.start <= colRange_.end && colRange_.end <= m.cols );

        cols = colRange_.size();
        data += colRange_.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if (rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::cuda::GpuMat::GpuMat(const GpuMat& m, Rect roi) :
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x * elemSize();

    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

GpuMat cv::cuda::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(cv::Error::BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(cv::Error::StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(cv::Error::StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(cv::Error::BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;
}

void cv::cuda::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    CV_DbgAssert( step > 0 );

    size_t esz = elemSize();
    ptrdiff_t delta1 = data - datastart;
    ptrdiff_t delta2 = dataend - datastart;

    if (delta1 == 0)
    {
        ofs.x = ofs.y = 0;
    }
    else
    {
        ofs.y = static_cast<int>(delta1 / step);
        ofs.x = static_cast<int>((delta1 - step * ofs.y) / esz);

        CV_DbgAssert( data == datastart + ofs.y * step + ofs.x * esz );
    }

    size_t minstep = (ofs.x + cols) * esz;

    wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / step + 1), ofs.y + rows);
    wholeSize.width = std::max(static_cast<int>((delta2 - step * (wholeSize.height - 1)) / esz), ofs.x + cols);
}

GpuMat& cv::cuda::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize;
    Point ofs;
    locateROI(wholeSize, ofs);

    size_t esz = elemSize();

    int row1 = std::max(ofs.y - dtop, 0);
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);

    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright, wholeSize.width);

    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1;
    cols = col2 - col1;

    if (esz * cols == step || rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;

    return *this;
}

namespace
{
    template <class ObjType>
    void createContinuousImpl(int rows, int cols, int type, ObjType& obj)
    {
        const int area = rows * cols;

        if (obj.empty() || obj.type() != type || !obj.isContinuous() || obj.size().area() < area)
            obj.create(1, area, type);

        obj = obj.reshape(obj.channels(), rows);
    }
}

void cv::cuda::createContinuous(int rows, int cols, int type, OutputArray arr)
{
    switch (arr.kind())
    {
    case _InputArray::MAT:
        ::createContinuousImpl(rows, cols, type, arr.getMatRef());
        break;

    case _InputArray::GPU_MAT:
        ::createContinuousImpl(rows, cols, type, arr.getGpuMatRef());
        break;

    case _InputArray::CUDA_MEM:
        ::createContinuousImpl(rows, cols, type, arr.getCudaMemRef());
        break;

    default:
        arr.create(rows, cols, type);
    }
}

namespace
{
    template <class ObjType>
    void ensureSizeIsEnoughImpl(int rows, int cols, int type, ObjType& obj)
    {
        if (obj.empty() || obj.type() != type || obj.data != obj.datastart)
        {
            obj.create(rows, cols, type);
        }
        else
        {
            const size_t esz = obj.elemSize();
            const ptrdiff_t delta2 = obj.dataend - obj.datastart;

            const size_t minstep = obj.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / static_cast<size_t>(obj.step) + 1), obj.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - static_cast<size_t>(obj.step) * (wholeSize.height - 1)) / esz), obj.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
            {
                obj.create(rows, cols, type);
            }
            else
            {
                obj.cols = cols;
                obj.rows = rows;
            }
        }
    }
}

void cv::cuda::ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr)
{
    switch (arr.kind())
    {
    case _InputArray::MAT:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getMatRef());
        break;

    case _InputArray::GPU_MAT:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getGpuMatRef());
        break;

    case _InputArray::CUDA_MEM:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getCudaMemRef());
        break;

    default:
        arr.create(rows, cols, type);
    }
}

GpuMat cv::cuda::allocMatFromBuf(int rows, int cols, int type, GpuMat& mat)
{
    if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
        return mat(Rect(0, 0, cols, rows));

    return mat = GpuMat(rows, cols, type);
}

#ifndef HAVE_CUDA

void cv::cuda::GpuMat::create(int _rows, int _cols, int _type)
{
    (void) _rows;
    (void) _cols;
    (void) _type;
    throw_no_cuda();
}

void cv::cuda::GpuMat::release()
{
}

void cv::cuda::GpuMat::upload(InputArray arr)
{
    (void) arr;
    throw_no_cuda();
}

void cv::cuda::GpuMat::upload(InputArray arr, Stream& _stream)
{
    (void) arr;
    (void) _stream;
    throw_no_cuda();
}

void cv::cuda::GpuMat::download(OutputArray _dst) const
{
    (void) _dst;
    throw_no_cuda();
}

void cv::cuda::GpuMat::download(OutputArray _dst, Stream& _stream) const
{
    (void) _dst;
    (void) _stream;
    throw_no_cuda();
}

void cv::cuda::GpuMat::copyTo(OutputArray _dst) const
{
    (void) _dst;
    throw_no_cuda();
}

void cv::cuda::GpuMat::copyTo(OutputArray _dst, Stream& _stream) const
{
    (void) _dst;
    (void) _stream;
    throw_no_cuda();
}

void cv::cuda::GpuMat::copyTo(OutputArray _dst, InputArray _mask, Stream& _stream) const
{
    (void) _dst;
    (void) _mask;
    (void) _stream;
    throw_no_cuda();
}

GpuMat& cv::cuda::GpuMat::setTo(Scalar s, Stream& _stream)
{
    (void) s;
    (void) _stream;
    throw_no_cuda();
    return *this;
}

GpuMat& cv::cuda::GpuMat::setTo(Scalar s, InputArray _mask, Stream& _stream)
{
    (void) s;
    (void) _mask;
    (void) _stream;
    throw_no_cuda();
    return *this;
}

void cv::cuda::GpuMat::convertTo(OutputArray _dst, int rtype, Stream& _stream) const
{
    (void) _dst;
    (void) rtype;
    (void) _stream;
    throw_no_cuda();
}

void cv::cuda::GpuMat::convertTo(OutputArray _dst, int rtype, double alpha, double beta, Stream& _stream) const
{
    (void) _dst;
    (void) rtype;
    (void) alpha;
    (void) beta;
    (void) _stream;
    throw_no_cuda();
}

#endif
