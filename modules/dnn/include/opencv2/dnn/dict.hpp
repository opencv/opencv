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

#include <opencv2/core.hpp>
#include <map>
#include <ostream>

#include <opencv2/dnn/dnn.hpp>

#ifndef OPENCV_DNN_DNN_DICT_HPP
#define OPENCV_DNN_DNN_DICT_HPP

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN
//! @addtogroup dnn
//! @{

/** @brief This struct stores the scalar value (or array) of one of the following type: double, cv::String or int64.
 *  @todo Maybe int64 is useless because double type exactly stores at least 2^52 integers.
 */
struct CV_EXPORTS_W DictValue
{
    DictValue(const DictValue &r);
    DictValue(int64 i = 0)      : type(Param::INT), pi(new AutoBuffer<int64,1>) { (*pi)[0] = i; }       //!< Constructs integer scalar
    CV_WRAP DictValue(int i)            : type(Param::INT), pi(new AutoBuffer<int64,1>) { (*pi)[0] = i; }       //!< Constructs integer scalar
    DictValue(unsigned p)       : type(Param::INT), pi(new AutoBuffer<int64,1>) { (*pi)[0] = p; }       //!< Constructs integer scalar
    CV_WRAP DictValue(double p)         : type(Param::REAL), pd(new AutoBuffer<double,1>) { (*pd)[0] = p; }     //!< Constructs floating point scalar
    CV_WRAP DictValue(const String &s)  : type(Param::STRING), ps(new AutoBuffer<String,1>) { (*ps)[0] = s; }   //!< Constructs string scalar
    DictValue(const char *s)    : type(Param::STRING), ps(new AutoBuffer<String,1>) { (*ps)[0] = s; }   //!< @overload

    template<typename TypeIter>
    static DictValue arrayInt(TypeIter begin, int size);    //!< Constructs integer array
    template<typename TypeIter>
    static DictValue arrayReal(TypeIter begin, int size);   //!< Constructs floating point array
    template<typename TypeIter>
    static DictValue arrayString(TypeIter begin, int size); //!< Constructs array of strings

    template<typename T>
    T get(int idx = -1) const; //!< Tries to convert array element with specified index to requested type and returns its.

    int size() const;

    CV_WRAP bool isInt() const;
    CV_WRAP bool isString() const;
    CV_WRAP bool isReal() const;

    CV_WRAP int getIntValue(int idx = -1) const;
    CV_WRAP double getRealValue(int idx = -1) const;
    CV_WRAP String getStringValue(int idx = -1) const;

    DictValue &operator=(const DictValue &r);

    friend std::ostream &operator<<(std::ostream &stream, const DictValue &dictv);

    ~DictValue();

private:

    int type;

    union
    {
        AutoBuffer<int64, 1> *pi;
        AutoBuffer<double, 1> *pd;
        AutoBuffer<String, 1> *ps;
        void *pv;
    };

    DictValue(int _type, void *_p) : type(_type), pv(_p) {}
    void release();
};

/** @brief This class implements name-value dictionary, values are instances of DictValue. */
class CV_EXPORTS Dict
{
    typedef std::map<String, DictValue> _Dict;
    _Dict dict;

public:

    //! Checks a presence of the @p key in the dictionary.
    bool has(const String &key) const;

    //! If the @p key in the dictionary then returns pointer to its value, else returns NULL.
    DictValue *ptr(const String &key);

    /** @overload */
    const DictValue *ptr(const String &key) const;

    //! If the @p key in the dictionary then returns its value, else an error will be generated.
    const DictValue &get(const String &key) const;

    /** @overload */
    template <typename T>
    T get(const String &key) const;

    //! If the @p key in the dictionary then returns its value, else returns @p defaultValue.
    template <typename T>
    T get(const String &key, const T &defaultValue) const;

    //! Sets new @p value for the @p key, or adds new key-value pair into the dictionary.
    template<typename T>
    const T &set(const String &key, const T &value);

    friend std::ostream &operator<<(std::ostream &stream, const Dict &dict);

    std::map<String, DictValue>::const_iterator begin() const;

    std::map<String, DictValue>::const_iterator end() const;
};

//! @}
CV__DNN_EXPERIMENTAL_NS_END
}
}

#endif
