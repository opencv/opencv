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

namespace
{
    class FormattedImpl : public cv::Formatted
    {
        enum { STATE_PROLOGUE, STATE_EPILOGUE, STATE_ROW_OPEN, STATE_ROW_CLOSE, STATE_CN_OPEN, STATE_CN_CLOSE, STATE_VALUE, STATE_FINISHED,
               STATE_LINE_SEPARATOR, STATE_CN_SEPARATOR, STATE_VALUE_SEPARATOR };
        enum {BRACE_ROW_OPEN = 0, BRACE_ROW_CLOSE = 1, BRACE_ROW_SEP=2, BRACE_CN_OPEN=3, BRACE_CN_CLOSE=4 };

        char floatFormat[8];
        char buf[32];   // enough for double with precision up to 20

        cv::Mat mtx;
        int mcn; // == mtx.channels()
        bool singleLine;

        int state;
        int row;
        int col;
        int cn;

        cv::String prologue;
        cv::String epilogue;
        char braces[5];

        void (FormattedImpl::*valueToStr)();
        void valueToStr8u()  { sprintf(buf, "%3d", (int)mtx.ptr<uchar>(row, col)[cn]); }
        void valueToStr8s()  { sprintf(buf, "%3d", (int)mtx.ptr<schar>(row, col)[cn]); }
        void valueToStr16u() { sprintf(buf, "%d", (int)mtx.ptr<ushort>(row, col)[cn]); }
        void valueToStr16s() { sprintf(buf, "%d", (int)mtx.ptr<short>(row, col)[cn]); }
        void valueToStr32s() { sprintf(buf, "%d", mtx.ptr<int>(row, col)[cn]); }
        void valueToStr32f() { sprintf(buf, floatFormat, mtx.ptr<float>(row, col)[cn]); }
        void valueToStr64f() { sprintf(buf, floatFormat, mtx.ptr<double>(row, col)[cn]); }
        void valueToStrOther() { buf[0] = 0; }

    public:

        FormattedImpl(cv::String pl, cv::String el, cv::Mat m, char br[5], bool sLine, int precision)
        {
            prologue = pl;
            epilogue = el;
            mtx = m;
            mcn = m.channels();
            memcpy(braces, br, 5);
            state = STATE_PROLOGUE;
            singleLine = sLine;

            if (precision < 0)
            {
                floatFormat[0] = '%';
                floatFormat[1] = 'a';
                floatFormat[2] = 0;
            }
            else
            {
                sprintf(floatFormat, "%%.%dg", std::min(precision, 20));
            }

            switch(mtx.depth())
            {
                case CV_8U:  valueToStr = &FormattedImpl::valueToStr8u; break;
                case CV_8S:  valueToStr = &FormattedImpl::valueToStr8s; break;
                case CV_16U: valueToStr = &FormattedImpl::valueToStr16u; break;
                case CV_16S: valueToStr = &FormattedImpl::valueToStr16s; break;
                case CV_32S: valueToStr = &FormattedImpl::valueToStr32s; break;
                case CV_32F: valueToStr = &FormattedImpl::valueToStr32f; break;
                case CV_64F: valueToStr = &FormattedImpl::valueToStr64f; break;
                default:     valueToStr = &FormattedImpl::valueToStrOther; break;
            }
        }

        void reset()
        {
            state = STATE_PROLOGUE;
        }

        const char* next()
        {
            switch(state)
            {
                case STATE_PROLOGUE:
                    row = 0;
                    if (mtx.empty())
                        state = STATE_EPILOGUE;
                    else
                        state = STATE_ROW_OPEN;
                    return prologue.c_str();
                case STATE_EPILOGUE:
                    state = STATE_FINISHED;
                    return epilogue.c_str();
                case STATE_ROW_OPEN:
                    col = 0;
                    state = STATE_CN_OPEN;
                    {
                        size_t pos = 0;
                        if (row > 0)
                            while(pos < prologue.size() && pos < sizeof(buf) - 2)
                                buf[pos++] = ' ';
                        if (braces[BRACE_ROW_OPEN])
                            buf[pos++] = braces[BRACE_ROW_OPEN];
                        if(!pos)
                            return next();
                        buf[pos] = 0;
                    }
                    return buf;
                case STATE_ROW_CLOSE:
                    state = STATE_LINE_SEPARATOR;
                    ++row;
                    if (braces[BRACE_ROW_CLOSE])
                    {
                        buf[0] = braces[BRACE_ROW_CLOSE];
                        buf[1] = row < mtx.rows ? ',' : '\0';
                        buf[2] = 0;
                        return buf;
                    }
                    else if(braces[BRACE_ROW_SEP] && row < mtx.rows)
                    {
                        buf[0] = braces[BRACE_ROW_SEP];
                        buf[1] = 0;
                        return buf;
                    }
                    return next();
                case STATE_CN_OPEN:
                    cn = 0;
                    state = STATE_VALUE;
                    if (mcn > 1 && braces[BRACE_CN_OPEN])
                    {
                        buf[0] = braces[BRACE_CN_OPEN];
                        buf[1] = 0;
                        return buf;
                    }
                    return next();
                case STATE_CN_CLOSE:
                    ++col;
                    if (col >= mtx.cols)
                        state = STATE_ROW_CLOSE;
                    else
                        state = STATE_CN_SEPARATOR;
                    if (mcn > 1 && braces[BRACE_CN_CLOSE])
                    {
                        buf[0] = braces[BRACE_CN_CLOSE];
                        buf[1] = 0;
                        return buf;
                    }
                    return next();
                case STATE_VALUE:
                    (this->*valueToStr)();
                    if (++cn >= mcn)
                        state = STATE_CN_CLOSE;
                    else
                        state = STATE_VALUE_SEPARATOR;
                    return buf;
                case STATE_FINISHED:
                    return 0;
                case STATE_LINE_SEPARATOR:
                    if (row >= mtx.rows)
                    {
                        state = STATE_EPILOGUE;
                        return next();
                    }
                    state = STATE_ROW_OPEN;
                    buf[0] = singleLine ? ' ' : '\n';
                    buf[1] = 0;
                    return buf;
                case STATE_CN_SEPARATOR:
                    state = STATE_CN_OPEN;
                    buf[0] = ',';
                    buf[1] = ' ';
                    buf[2] = 0;
                    return buf;
                case STATE_VALUE_SEPARATOR:
                    state = STATE_VALUE;
                    buf[0] = ',';
                    buf[1] = ' ';
                    buf[2] = 0;
                    return buf;
            }
            return 0;
        }
    };

    class FormatterBase : public cv::Formatter
    {
    public:
        FormatterBase() : prec32f(8), prec64f(16), multiline(true) {}

        void set32fPrecision(int p)
        {
            prec32f = p;
        }

        void set64fPrecision(int p)
        {
            prec64f = p;
        }

        void setMultiline(bool ml)
        {
            multiline = ml;
        }

    protected:
        int prec32f;
        int prec64f;
        int multiline;
    };

    class MatlabFormatter : public FormatterBase
    {
    public:

        cv::Ptr<cv::Formatted> format(const cv::Mat& mtx) const
        {
            char braces[5] = {'\0', '\0', ';', '\0', '\0'};
            return cv::makePtr<FormattedImpl>("[", "]", mtx, &*braces,
                mtx.cols == 1 || !multiline, mtx.depth() == CV_64F ? prec64f : prec32f );
        }
    };

    class PythonFormatter : public FormatterBase
    {
    public:

        cv::Ptr<cv::Formatted> format(const cv::Mat& mtx) const
        {
            char braces[5] = {'[', ']', '\0', '[', ']'};
            if (mtx.cols == 1)
                braces[0] = braces[1] = '\0';
            return cv::makePtr<FormattedImpl>("[", "]", mtx, &*braces,
                mtx.cols*mtx.channels() == 1 || !multiline, mtx.depth() == CV_64F ? prec64f : prec32f );
        }
    };

    class NumpyFormatter : public FormatterBase
    {
    public:

        cv::Ptr<cv::Formatted> format(const cv::Mat& mtx) const
        {
            static const char* numpyTypes[] =
            {
                "uint8", "int8", "uint16", "int16", "int32", "float32", "float64", "uint64"
            };
            char braces[5] = {'[', ']', '\0', '[', ']'};
            if (mtx.cols == 1)
                braces[0] = braces[1] = '\0';
            return cv::makePtr<FormattedImpl>("array([",
                cv::format("], type='%s')", numpyTypes[mtx.depth()]), mtx, &*braces,
                mtx.cols*mtx.channels() == 1 || !multiline, mtx.depth() == CV_64F ? prec64f : prec32f );
        }
    };

    class CSVFormatter : public FormatterBase
    {
    public:

        cv::Ptr<cv::Formatted> format(const cv::Mat& mtx) const
        {
            char braces[5] = {'\0', '\0', '\0', '\0', '\0'};
            return cv::makePtr<FormattedImpl>(cv::String(),
                mtx.rows > 1 ? cv::String("\n") : cv::String(), mtx, &*braces,
                mtx.cols*mtx.channels() == 1 || !multiline, mtx.depth() == CV_64F ? prec64f : prec32f );
        }
    };

    class CFormatter : public FormatterBase
    {
    public:

        cv::Ptr<cv::Formatted> format(const cv::Mat& mtx) const
        {
            char braces[5] = {'\0', '\0', ',', '\0', '\0'};
            return cv::makePtr<FormattedImpl>("{", "}", mtx, &*braces,
                mtx.cols == 1 || !multiline, mtx.depth() == CV_64F ? prec64f : prec32f );
        }
    };

} // namespace


namespace cv
{
    Formatted::~Formatted() {}
    Formatter::~Formatter() {}

    Ptr<Formatter> Formatter::get(int fmt)
    {
        switch(fmt)
        {
            case FMT_MATLAB:
                return makePtr<MatlabFormatter>();
            case FMT_CSV:
                return makePtr<CSVFormatter>();
            case FMT_PYTHON:
                return makePtr<PythonFormatter>();
            case FMT_NUMPY:
                return makePtr<NumpyFormatter>();
            case FMT_C:
                return makePtr<CFormatter>();
        }
        return makePtr<MatlabFormatter>();
    }
} // cv
