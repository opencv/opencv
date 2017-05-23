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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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
#include <iterator>

namespace cv
{

class DefaultFormatter : public Formatter
{
public:
    DefaultFormatter() : Formatter(SQUARE_BRACKET_OPEN, NO_BRACKET,
                                    NO_BRACKET, NO_BRACKET,
                                    SEMICOLON_SEPARATOR, COMMA_SEPARATOR, COMMA_SEPARATOR) {}
    virtual ~DefaultFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << "\n";
    }
};

class MatlabFormatter : public Formatter
{
public:
    MatlabFormatter() : Formatter(NO_BRACKET, NO_BRACKET,
                                    NO_BRACKET, NO_BRACKET,
                                    SPACE_SEPARATOR, SPACE_SEPARATOR, SPACE_SEPARATOR) {}
    virtual ~MatlabFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << "\n";
    }

protected:
    void writeRow(std::ostream& out, const Mat& m) const
    {
        CV_Assert(m.dims <= 2);
        for( int cn = 0; cn < m.channels(); cn++ )
        {
            if (m.channels() == 1)
                out << "(:,:) = \n";
            else
                out << "(:,:," << cn << ") = \n";

            for (int row = 0; row < m.rows; row++ )
            {
                out << getBracketString(rowOpen);
                for( int col = 0; col < m.cols; col++ )
                {
                    out << getBracketString(colOpen) << getBracketString(valueOpen);
                    if( m.data ) {
                        Formatter::writeValue(out, m, row, col, cn);
                    }

                    out << getBracketString(getCloseBracket(valueOpen)) << getBracketString(getCloseBracket(colOpen));

                    if (col+1 < m.cols)
                        out << (char)colsep;
                }

                // close row bracket, row separator, and new line feed
                out << getBracketString(getCloseBracket(rowOpen));
                if (row+1 < m.rows)
                    out << (char)rowsep << "\n";
            }
            if (cn+1 < m.channels())
                out << "\n";
        }
    }
};

class PythonFormatter : public Formatter
{
public:
    PythonFormatter() : Formatter(SQUARE_BRACKET_OPEN, SQUARE_BRACKET_OPEN,
                                    SQUARE_BRACKET_OPEN, NO_BRACKET,
                                    COMMA_SEPARATOR, COMMA_SEPARATOR, COMMA_SEPARATOR) {}
    virtual ~PythonFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << "\n";
    }
};


class NumpyFormatter : public Formatter
{
public:
    NumpyFormatter() : Formatter(SQUARE_BRACKET_OPEN, SQUARE_BRACKET_OPEN,
                                    SQUARE_BRACKET_OPEN, NO_BRACKET,
                                    COMMA_SEPARATOR, COMMA_SEPARATOR, COMMA_SEPARATOR) {}
    virtual ~NumpyFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        static const char* numpyTypes[] =
        {
            "uint8", "int8", "uint16", "int16", "int32", "float32", "float64", "uint64"
        };

        out << "array(";
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << ", type='" << numpyTypes[m.depth()] << "')" << "\n";
    }
};


class CSVFormatter : public Formatter
{
public:
    CSVFormatter() : Formatter(NO_BRACKET, NO_BRACKET,
                                NO_BRACKET, NO_BRACKET,
                                SPACE_SEPARATOR, COMMA_SEPARATOR, COMMA_SEPARATOR) {}
    virtual ~CSVFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << "\n";
    }
};


class CFormatter : public Formatter
{
public:
    CFormatter() : Formatter(PARENTHESES_OPEN, NO_BRACKET,
                                NO_BRACKET, NO_BRACKET,
                                COMMA_SEPARATOR, COMMA_SEPARATOR, COMMA_SEPARATOR) {}
    virtual ~CFormatter() {}
    void write(std::ostream& out, const Mat& m, const int*, int) const
    {
        out << getBracketString(matrixOpen);
        writeRow(out, m);
        out << getBracketString(getCloseBracket(matrixOpen)) << "\n";
    }
};


static DefaultFormatter defaultFormatter;
static MatlabFormatter matlabFormatter;
static PythonFormatter pythonFormatter;
static NumpyFormatter numpyFormatter;
static CSVFormatter csvFormatter;
static CFormatter cFormatter;

static const Formatter* g_defaultFormatter0 = &defaultFormatter;
static const Formatter* g_defaultFormatter = &defaultFormatter;

static bool my_streq(const char* a, const char* b)
{
    size_t i, alen = strlen(a), blen = strlen(b);
    if( alen != blen )
        return false;
    for( i = 0; i < alen; i++ )
        if( a[i] != b[i] && a[i] - 32 != b[i] )
            return false;
    return true;
}

const Formatter* Formatter::get(const char* fmt)
{
    if(!fmt || my_streq(fmt, ""))
        return g_defaultFormatter;
    if( my_streq(fmt, "MATLAB"))
        return &matlabFormatter;
    if( my_streq(fmt, "CSV"))
        return &csvFormatter;
    if( my_streq(fmt, "PYTHON"))
        return &pythonFormatter;
    if( my_streq(fmt, "NUMPY"))
        return &numpyFormatter;
    if( my_streq(fmt, "C"))
        return &cFormatter;
    CV_Error(CV_StsBadArg, "Unknown formatter");
    return g_defaultFormatter;
}

const Formatter* Formatter::setDefault(const Formatter* fmt)
{
    const Formatter* prevFmt = g_defaultFormatter;
    if(!fmt)
        fmt = g_defaultFormatter0;
    g_defaultFormatter = fmt;
    return prevFmt;
}

Formatted::Formatted(const Mat& _m, const Formatter* _fmt,
                     const vector<int>& _params)
{
    mtx = _m;
    fmt = _fmt ? _fmt : Formatter::get();
    std::copy(_params.begin(), _params.end(), back_inserter(params));
}

Formatted::Formatted(const Mat& _m, const Formatter* _fmt, const int* _params)
{
    mtx = _m;
    fmt = _fmt ? _fmt : Formatter::get();

    if( _params )
    {
        int i, maxParams = 100;
        for(i = 0; i < maxParams && _params[i] != 0; i+=2)
            ;
        std::copy(_params, _params + i, back_inserter(params));
    }
}

}
