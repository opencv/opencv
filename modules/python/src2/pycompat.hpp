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

#include <string>

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

#define CV_PY_FN_WITH_KW_(fn, flags) (PyCFunction)(void*)(PyCFunctionWithKeywords)(fn), (flags) | METH_VARARGS | METH_KEYWORDS
#define CV_PY_FN_NOARGS_(fn, flags) (PyCFunction)(fn), (flags) | METH_NOARGS

#define CV_PY_FN_WITH_KW(fn) CV_PY_FN_WITH_KW_(fn, 0)
#define CV_PY_FN_NOARGS(fn) CV_PY_FN_NOARGS_(fn, 0)

#define CV_PY_TO_CLASS(TYPE)                                                                          \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE& src, const ArgInfo& info)                                       \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    Ptr<TYPE> ptr;                                                                                    \
                                                                                                      \
    if (!pyopencv_to(dst, ptr, info)) return false;                                                   \
    src = *ptr;                                                                                       \
    return true;                                                                                      \
}

#define CV_PY_FROM_CLASS(TYPE)                                                                        \
template<>                                                                                            \
PyObject* pyopencv_from(const TYPE& src)                                                              \
{                                                                                                     \
    Ptr<TYPE> ptr(new TYPE());                                                                        \
                                                                                                      \
    *ptr = src;                                                                                       \
    return pyopencv_from(ptr);                                                                        \
}

#define CV_PY_TO_CLASS_PTR(TYPE)                                                                      \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE*& src, const ArgInfo& info)                                      \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    Ptr<TYPE> ptr;                                                                                    \
                                                                                                      \
    if (!pyopencv_to(dst, ptr, info)) return false;                                                   \
    src = ptr;                                                                                        \
    return true;                                                                                      \
}

#define CV_PY_FROM_CLASS_PTR(TYPE)                                                                    \
static PyObject* pyopencv_from(TYPE*& src)                                                            \
{                                                                                                     \
    return pyopencv_from(Ptr<TYPE>(src));                                                             \
}

#define CV_PY_TO_ENUM(TYPE)                                                                           \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE& src, const ArgInfo& info)                                       \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    int underlying = 0;                                                  \
                                                                                                      \
    if (!pyopencv_to(dst, underlying, info)) return false;                                            \
    src = static_cast<TYPE>(underlying);                                                              \
    return true;                                                                                      \
}

#define CV_PY_FROM_ENUM(TYPE)                                                                         \
template<>                                                                                            \
PyObject* pyopencv_from(const TYPE& src)                                                              \
{                                                                                                     \
    return pyopencv_from(static_cast<int>(src));                         \
}

//==================================================================================================

#if PY_MAJOR_VERSION >= 3
#define CVPY_TYPE_HEAD PyVarObject_HEAD_INIT(&PyType_Type, 0)
#define CVPY_TYPE_INCREF(T) Py_INCREF(T)
#else
#define CVPY_TYPE_HEAD PyObject_HEAD_INIT(&PyType_Type) 0,
#define CVPY_TYPE_INCREF(T) _Py_INC_REFTOTAL _Py_REF_DEBUG_COMMA (T)->ob_refcnt++
#endif


#define CVPY_TYPE_DECLARE(WNAME, NAME, STORAGE, SNAME) \
    struct pyopencv_##NAME##_t \
    { \
        PyObject_HEAD \
        STORAGE v; \
    }; \
    static PyTypeObject pyopencv_##NAME##_TypeXXX = \
    { \
        CVPY_TYPE_HEAD \
        MODULESTR"."#WNAME, \
        sizeof(pyopencv_##NAME##_t), \
    }; \
    static PyTypeObject * pyopencv_##NAME##_TypePtr = &pyopencv_##NAME##_TypeXXX; \
    static bool pyopencv_##NAME##_getp(PyObject * self, STORAGE * & dst) \
    { \
        if (PyObject_TypeCheck(self, pyopencv_##NAME##_TypePtr)) \
        { \
            dst = &(((pyopencv_##NAME##_t*)self)->v); \
            return true; \
        } \
        return false; \
    } \
    static PyObject * pyopencv_##NAME##_Instance(const STORAGE &r) \
    { \
        pyopencv_##NAME##_t *m = PyObject_NEW(pyopencv_##NAME##_t, pyopencv_##NAME##_TypePtr); \
        new (&(m->v)) STORAGE(r); \
        return (PyObject*)m; \
    } \
    static void pyopencv_##NAME##_dealloc(PyObject* self) \
    { \
        ((pyopencv_##NAME##_t*)self)->v.STORAGE::~SNAME(); \
        PyObject_Del(self); \
    } \
    static PyObject* pyopencv_##NAME##_repr(PyObject* self) \
    { \
        char str[1000]; \
        sprintf(str, "<"#WNAME" %p>", self); \
        return PyString_FromString(str); \
    }


#define CVPY_TYPE_INIT_STATIC(WNAME, NAME, ERROR_HANDLER, BASE, CONSTRUCTOR) \
    { \
        pyopencv_##NAME##_TypePtr->tp_base = pyopencv_##BASE##_TypePtr; \
        pyopencv_##NAME##_TypePtr->tp_dealloc = pyopencv_##NAME##_dealloc; \
        pyopencv_##NAME##_TypePtr->tp_repr = pyopencv_##NAME##_repr; \
        pyopencv_##NAME##_TypePtr->tp_getset = pyopencv_##NAME##_getseters; \
        pyopencv_##NAME##_TypePtr->tp_init = (initproc) CONSTRUCTOR; \
        pyopencv_##NAME##_TypePtr->tp_methods = pyopencv_##NAME##_methods; \
        pyopencv_##NAME##_TypePtr->tp_alloc = PyType_GenericAlloc; \
        pyopencv_##NAME##_TypePtr->tp_new = PyType_GenericNew; \
        pyopencv_##NAME##_TypePtr->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE; \
        if (PyType_Ready(pyopencv_##NAME##_TypePtr) != 0) \
        { \
            ERROR_HANDLER; \
        } \
        CVPY_TYPE_INCREF(pyopencv_##NAME##_TypePtr); \
        if (PyModule_AddObject(m, #WNAME, (PyObject *)pyopencv_##NAME##_TypePtr) < 0) \
        { \
            printf("Failed to register a new type: " #WNAME  ", base (" #BASE ")\n"); \
            Py_DECREF(pyopencv_##NAME##_TypePtr); \
            ERROR_HANDLER; \
        } \
    }

//==================================================================================================

#define CVPY_TYPE_DECLARE_DYNAMIC(WNAME, NAME, STORAGE, SNAME) \
    struct pyopencv_##NAME##_t \
    { \
        PyObject_HEAD \
        STORAGE v; \
    }; \
    static PyObject * pyopencv_##NAME##_TypePtr = 0; \
    static bool pyopencv_##NAME##_getp(PyObject * self, STORAGE * & dst) \
    { \
        if (PyObject_TypeCheck(self, (PyTypeObject*)pyopencv_##NAME##_TypePtr)) \
        { \
            dst = &(((pyopencv_##NAME##_t*)self)->v); \
            return true; \
        } \
        return false; \
    } \
    static PyObject * pyopencv_##NAME##_Instance(const STORAGE &r) \
    { \
        pyopencv_##NAME##_t *m = PyObject_New(pyopencv_##NAME##_t, (PyTypeObject*)pyopencv_##NAME##_TypePtr); \
        new (&(m->v)) STORAGE(r); \
        return (PyObject*)m; \
    } \
    static void pyopencv_##NAME##_dealloc(PyObject* self) \
    { \
        ((pyopencv_##NAME##_t*)self)->v.STORAGE::~SNAME(); \
        PyObject_Del(self); \
    } \
    static PyObject* pyopencv_##NAME##_repr(PyObject* self) \
    { \
        char str[1000]; \
        sprintf(str, "<"#WNAME" %p>", self); \
        return PyString_FromString(str); \
    } \
    static PyType_Slot pyopencv_##NAME##_Slots[] =  \
    { \
        {Py_tp_dealloc, 0}, \
        {Py_tp_repr, 0}, \
        {Py_tp_getset, 0}, \
        {Py_tp_init, 0}, \
        {Py_tp_methods, 0}, \
        {Py_tp_alloc, 0}, \
        {Py_tp_new, 0}, \
        {0, 0} \
    }; \
    static PyType_Spec pyopencv_##NAME##_Spec = \
    { \
        MODULESTR"."#WNAME, \
        sizeof(pyopencv_##NAME##_t), \
        0, \
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, \
        pyopencv_##NAME##_Slots  \
    };

#define CVPY_TYPE_INIT_DYNAMIC(WNAME, NAME, ERROR_HANDLER, BASE, CONSTRUCTOR) \
    { \
        pyopencv_##NAME##_Slots[0].pfunc /*tp_dealloc*/ = (void*)pyopencv_##NAME##_dealloc; \
        pyopencv_##NAME##_Slots[1].pfunc /*tp_repr*/ = (void*)pyopencv_##NAME##_repr; \
        pyopencv_##NAME##_Slots[2].pfunc /*tp_getset*/ = (void*)pyopencv_##NAME##_getseters; \
        pyopencv_##NAME##_Slots[3].pfunc /*tp_init*/ = (void*) CONSTRUCTOR; \
        pyopencv_##NAME##_Slots[4].pfunc /*tp_methods*/ = pyopencv_##NAME##_methods; \
        pyopencv_##NAME##_Slots[5].pfunc /*tp_alloc*/ = (void*)PyType_GenericAlloc; \
        pyopencv_##NAME##_Slots[6].pfunc /*tp_new*/ = (void*)PyType_GenericNew; \
        PyObject * bases = 0; \
        if (pyopencv_##BASE##_TypePtr) \
            bases = PyTuple_Pack(1, pyopencv_##BASE##_TypePtr); \
        pyopencv_##NAME##_TypePtr = PyType_FromSpecWithBases(&pyopencv_##NAME##_Spec, bases); \
        if (!pyopencv_##NAME##_TypePtr) \
        { \
            printf("Failed to create type from spec: " #WNAME ", base (" #BASE ")\n"); \
            ERROR_HANDLER; \
        } \
        if (PyModule_AddObject(m, #WNAME, (PyObject *)pyopencv_##NAME##_TypePtr) < 0) \
        { \
            printf("Failed to register a new type: " #WNAME  ", base (" #BASE ")\n"); \
            Py_DECREF(pyopencv_##NAME##_TypePtr); \
            ERROR_HANDLER; \
        } \
    }

// Debug module load:
//
// else \
// { \
//     printf("Init: " #NAME ", base (" #BASE ") -> %p" "\n", pyopencv_##NAME##_TypePtr); \
// } \


#endif // END HEADER GUARD
