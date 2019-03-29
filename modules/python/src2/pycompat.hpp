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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// Defines for Python 2/3 compatibility.
#ifndef __PYCOMPAT_HPP__
#define __PYCOMPAT_HPP__

#if PY_MAJOR_VERSION >= 3

// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Int PyNumber_Long


#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize

#endif // PY_MAJOR >=3

static inline bool getUnicodeString(PyObject * obj, std::string &str)
{
    bool res = false;
    if (PyUnicode_Check(obj))
    {
        PyObject * bytes = PyUnicode_AsUTF8String(obj);
        if (PyBytes_Check(bytes))
        {
            const char * raw = PyBytes_AsString(bytes);
            if (raw)
            {
                str = std::string(raw);
                res = true;
            }
        }
        Py_XDECREF(bytes);
    }
#if PY_MAJOR_VERSION < 3
    else if (PyString_Check(obj))
    {
        const char * raw = PyString_AsString(obj);
        if (raw)
        {
            str = std::string(raw);
            res = true;
        }
    }
#endif
    return res;
}

//==================================================================================================

#if PY_MAJOR_VERSION >= 3
#define CVPY_TYPE_HEAD PyVarObject_HEAD_INIT(&PyType_Type, 0)
#define CVPY_TYPE_INCREF(T) Py_INCREF(&T)
#else
#define CVPY_TYPE_HEAD PyObject_HEAD_INIT(&PyType_Type) 0,
#define CVPY_TYPE_INCREF(T) _Py_INC_REFTOTAL _Py_REF_DEBUG_COMMA (&T)->ob_refcnt++
#endif


#define CVPY_TYPE_DECLARE(name, name2) \
    struct pyopencv_##name##_t \
    { \
        PyObject_HEAD \
        name2 v; \
    }; \

#define CVPY_TYPE_REGISTER_STATIC(name) \
    static PyTypeObject pyopencv_##name##_Type = \
    { \
        CVPY_TYPE_HEAD \
        MODULESTR"."#name, \
        sizeof(pyopencv_##name##_t), \
    };

#define CVPY_TYPE_INIT_STATIC(name, err_code) \
    { \
        pyopencv_##name##_specials(); \
        pyopencv_##name##_Type.tp_alloc = PyType_GenericAlloc; \
        pyopencv_##name##_Type.tp_new = PyType_GenericNew; \
        pyopencv_##name##_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE; \
        if (PyType_Ready(&pyopencv_##name##_Type) != 0) \
        { \
            err_code; \
        } \
        CVPY_TYPE_INCREF(pyopencv_##name##_Type); \
        PyModule_AddObject(m, #name, (PyObject *)&pyopencv_##name##_Type); \
    }


#endif // END HEADER GUARD
