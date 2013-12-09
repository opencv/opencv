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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
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

#ifndef __OPENCV_OCL_MATRIX_OPERATIONS_HPP__
#define __OPENCV_OCL_MATRIX_OPERATIONS_HPP__

#include "opencv2/ocl.hpp"

namespace cv
{

    namespace ocl
    {

        enum
        {
            MAT_ADD = 1,
            MAT_SUB,
            MAT_MUL,
            MAT_DIV,
            MAT_NOT,
            MAT_AND,
            MAT_OR,
            MAT_XOR
        };

        class CV_EXPORTS oclMatExpr
        {
            public:
                oclMatExpr() : a(oclMat()), b(oclMat()), op(0) {}
                oclMatExpr(const oclMat& _a, const oclMat& _b, int _op)
                    : a(_a), b(_b), op(_op) {}
                operator oclMat() const;
                void assign(oclMat& m) const;

            protected:
                oclMat a, b;
                int op;
        };
        ////////////////////////////////////////////////////////////////////////
        //////////////////////////////// oclMat ////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        inline oclMat::oclMat() : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0) {}

        inline oclMat::oclMat(int _rows, int _cols, int _type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            if( _rows > 0 && _cols > 0 )
                create( _rows, _cols, _type );
        }

        inline oclMat::oclMat(Size _size, int _type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            if( _size.height > 0 && _size.width > 0 )
                create( _size.height, _size.width, _type );
        }

        inline oclMat::oclMat(int _rows, int _cols, int _type, const Scalar &_s)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            if(_rows > 0 && _cols > 0)
            {
                create(_rows, _cols, _type);
                *this = _s;
            }
        }

        inline oclMat::oclMat(Size _size, int _type, const Scalar &_s)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            if( _size.height > 0 && _size.width > 0 )
            {
                create( _size.height, _size.width, _type );
                *this = _s;
            }
        }

        inline oclMat::oclMat(const oclMat &m)
            : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data),
              refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), clCxt(m.clCxt), offset(m.offset), wholerows(m.wholerows), wholecols(m.wholecols)
        {
            if( refcount )
                CV_XADD(refcount, 1);
        }

        inline oclMat::oclMat(int _rows, int _cols, int _type, void *_data, size_t _step)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0),
              datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            cv::Mat m(_rows, _cols, _type, _data, _step);
            upload(m);
            //size_t minstep = cols * elemSize();
            //if( step == Mat::AUTO_STEP )
            //{
            //    step = minstep;
            //    flags |= Mat::CONTINUOUS_FLAG;
            //}
            //else
            //{
            //    if( rows == 1 ) step = minstep;
            //    CV_DbgAssert( step >= minstep );
            //    flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
            //}
            //dataend += step * (rows - 1) + minstep;
        }

        inline oclMat::oclMat(Size _size, int _type, void *_data, size_t _step)
            : flags(0), rows(0), cols(0),
              step(0), data(0), refcount(0),
              datastart(0), dataend(0), offset(0), wholerows(0), wholecols(0)
        {
            cv::Mat m(_size, _type, _data, _step);
            upload(m);
            //size_t minstep = cols * elemSize();
            //if( step == Mat::AUTO_STEP )
            //{
            //    step = minstep;
            //    flags |= Mat::CONTINUOUS_FLAG;
            //}
            //else
            //{
            //    if( rows == 1 ) step = minstep;
            //    CV_DbgAssert( step >= minstep );
            //    flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
            //}
            //dataend += step * (rows - 1) + minstep;
        }


        inline oclMat::oclMat(const oclMat &m, const Range &rRange, const Range &cRange)
        {
            flags = m.flags;
            step = m.step;
            refcount = m.refcount;
            data = m.data;
            datastart = m.datastart;
            dataend = m.dataend;
            wholerows = m.wholerows;
            wholecols = m.wholecols;
            offset = m.offset;
            if( rRange == Range::all() )
                rows = m.rows;
            else
            {
                CV_Assert( 0 <= rRange.start && rRange.start <= rRange.end && rRange.end <= m.rows );
                rows = rRange.size();
                offset += step * rRange.start;
            }

            if( cRange == Range::all() )
                cols = m.cols;
            else
            {
                CV_Assert( 0 <= cRange.start && cRange.start <= cRange.end && cRange.end <= m.cols );
                cols = cRange.size();
                offset += cRange.start * elemSize();
                flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
            }

            if( rows == 1 )
                flags |= Mat::CONTINUOUS_FLAG;

            if( refcount )
                CV_XADD(refcount, 1);
            if( rows <= 0 || cols <= 0 )
                rows = cols = 0;
        }

        inline oclMat::oclMat(const oclMat &m, const Rect &roi)
            : flags(m.flags), rows(roi.height), cols(roi.width),
              step(m.step), data(m.data), refcount(m.refcount),
              datastart(m.datastart), dataend(m.dataend), clCxt(m.clCxt), offset(m.offset), wholerows(m.wholerows), wholecols(m.wholecols)
        {
            flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
            offset += roi.y * step + roi.x * elemSize();
            CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.wholecols &&
                       0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.wholerows );
            if( refcount )
                CV_XADD(refcount, 1);
            if( rows <= 0 || cols <= 0 )
                rows = cols = 0;
        }

        inline oclMat::oclMat(const Mat &m)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) , offset(0), wholerows(0), wholecols(0)
        {
            //clCxt = Context::getContext();
            upload(m);
        }

        inline oclMat::~oclMat()
        {
            release();
        }

        inline oclMat &oclMat::operator = (const oclMat &m)
        {
            if( this != &m )
            {
                if( m.refcount )
                    CV_XADD(m.refcount, 1);
                release();
                clCxt = m.clCxt;
                flags = m.flags;
                rows = m.rows;
                cols = m.cols;
                step = m.step;
                data = m.data;
                datastart = m.datastart;
                dataend = m.dataend;
                offset = m.offset;
                wholerows = m.wholerows;
                wholecols = m.wholecols;
                refcount = m.refcount;
            }
            return *this;
        }

        inline oclMat &oclMat::operator = (const Mat &m)
        {
            //clCxt = Context::getContext();
            upload(m);
            return *this;
        }

        inline oclMat& oclMat::operator = (const oclMatExpr& expr)
        {
            expr.assign(*this);
            return *this;
        }

        /* Fixme! To be supported in OpenCL later. */
#if 0
        template <class T> inline oclMat::operator DevMem2D_<T>() const
        {
            return DevMem2D_<T>(rows, cols, (T *)data, step);
        }
        template <class T> inline oclMat::operator PtrStep_<T>() const
        {
            return PtrStep_<T>(static_cast< DevMem2D_<T> >(*this));
        }
#endif

        //CPP: void oclMat::upload(const Mat& m);

        inline oclMat::operator Mat() const
        {
            Mat m;
            download(m);
            return m;
        }

        //CPP void oclMat::download(cv::Mat& m) const;

        inline oclMat oclMat::row(int y) const
        {
            return oclMat(*this, Range(y, y + 1), Range::all());
        }
        inline oclMat oclMat::col(int x) const
        {
            return oclMat(*this, Range::all(), Range(x, x + 1));
        }
        inline oclMat oclMat::rowRange(int startrow, int endrow) const
        {
            return oclMat(*this, Range(startrow, endrow), Range::all());
        }
        inline oclMat oclMat::rowRange(const Range &r) const
        {
            return oclMat(*this, r, Range::all());
        }
        inline oclMat oclMat::colRange(int startcol, int endcol) const
        {
            return oclMat(*this, Range::all(), Range(startcol, endcol));
        }
        inline oclMat oclMat::colRange(const Range &r) const
        {
            return oclMat(*this, Range::all(), r);
        }

        inline oclMat oclMat::clone() const
        {
            oclMat m;
            copyTo(m);
            return m;
        }

        //CPP void oclMat::copyTo( oclMat& m ) const;
        //CPP void oclMat::copyTo( oclMat& m, const oclMat& mask  ) const;
        //CPP void oclMat::convertTo( oclMat& m, int rtype, double alpha=1, double beta=0 ) const;

        inline void oclMat::assignTo( oclMat &m, int mtype ) const
        {
            if( mtype < 0 )
                m = *this;
            else
                convertTo(m, mtype);
        }

        //CPP oclMat& oclMat::operator = (const Scalar& s);
        //CPP oclMat& oclMat::setTo(const Scalar& s, const oclMat& mask=oclMat());
        //CPP oclMat oclMat::reshape(int _cn, int _rows=0) const;
        inline void oclMat::create(Size _size, int _type)
        {
            create(_size.height, _size.width, _type);
        }
        //CPP void oclMat::create(int _rows, int _cols, int _type);
        //CPP void oclMat::release();

        inline void oclMat::swap(oclMat &b)
        {
            std::swap( flags, b.flags );
            std::swap( rows, b.rows );
            std::swap( cols, b.cols );
            std::swap( step, b.step );
            std::swap( data, b.data );
            std::swap( datastart, b.datastart );
            std::swap( dataend, b.dataend );
            std::swap( refcount, b.refcount );
            std::swap( offset, b.offset );
            std::swap( clCxt,  b.clCxt );
            std::swap( wholerows, b.wholerows );
            std::swap( wholecols, b.wholecols );
        }

        inline void oclMat::locateROI( Size &wholeSize, Point &ofs ) const
        {
            size_t esz = elemSize();//, minstep;
            //ptrdiff_t delta1 = offset;//, delta2 = dataend - datastart;
            CV_DbgAssert( step > 0 );
            if( offset == 0 )
                ofs.x = ofs.y = 0;
            else
            {
                ofs.y = (int)(offset / step);
                ofs.x = (int)((offset - step * ofs.y) / esz);
                //CV_DbgAssert( data == datastart + ofs.y*step + ofs.x*esz );
            }
            //minstep = (ofs.x + cols)*esz;
            //wholeSize.height = (int)((delta2 - minstep)/step + 1);
            //wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
            //wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
            //wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
            wholeSize.height = wholerows;
            wholeSize.width = wholecols;
        }

        inline oclMat &oclMat::adjustROI( int dtop, int dbottom, int dleft, int dright )
        {
            Size wholeSize;
            Point ofs;
            size_t esz = elemSize();
            locateROI( wholeSize, ofs );
            int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
            int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
            offset += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
            rows = row2 - row1;
            cols = col2 - col1;
            if( esz * cols == step || rows == 1 )
                flags |= Mat::CONTINUOUS_FLAG;
            else
                flags &= ~Mat::CONTINUOUS_FLAG;
            return *this;
        }

        inline oclMat oclMat::operator()( Range rRange, Range cRange ) const
        {
            return oclMat(*this, rRange, cRange);
        }
        inline oclMat oclMat::operator()( const Rect &roi ) const
        {
            return oclMat(*this, roi);
        }

        inline bool oclMat::isContinuous() const
        {
            return (flags & Mat::CONTINUOUS_FLAG) != 0;
        }
        inline size_t oclMat::elemSize() const
        {
            return CV_ELEM_SIZE((CV_MAKE_TYPE(type(), oclchannels())));
        }
        inline size_t oclMat::elemSize1() const
        {
            return CV_ELEM_SIZE1(flags);
        }
        inline int oclMat::type() const
        {
            return CV_MAT_TYPE(flags);
        }
        inline int oclMat::ocltype() const
        {
            return CV_MAKE_TYPE(depth(), oclchannels());
        }
        inline int oclMat::depth() const
        {
            return CV_MAT_DEPTH(flags);
        }
        inline int oclMat::channels() const
        {
            return CV_MAT_CN(flags);
        }
        inline int oclMat::oclchannels() const
        {
            return (CV_MAT_CN(flags)) == 3 ? 4 : (CV_MAT_CN(flags));
        }
        inline size_t oclMat::step1() const
        {
            return step / elemSize1();
        }
        inline Size oclMat::size() const
        {
            return Size(cols, rows);
        }
        inline bool oclMat::empty() const
        {
            return data == 0;
        }

        inline oclMat oclMat::t() const
        {
            oclMat tmp;
            transpose(*this, tmp);
            return tmp;
        }

        static inline void swap( oclMat &a, oclMat &b )
        {
            a.swap(b);
        }

        inline void ensureSizeIsEnough(int rows, int cols, int type, oclMat &m)
        {
            if (m.type() == type && m.rows >= rows && m.cols >= cols)
                m = m(Rect(0, 0, cols, rows));
            else
                m.create(rows, cols, type);
        }

        inline void ensureSizeIsEnough(Size size, int type, oclMat &m)
        {
            ensureSizeIsEnough(size.height, size.width, type, m);
        }


    } /* end of namespace ocl */

} /* end of namespace cv */

#endif /* __OPENCV_OCL_MATRIX_OPERATIONS_HPP__ */
