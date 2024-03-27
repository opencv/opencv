// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef HAVE_OPENCV_CORE

// see https://github.com/opencv/opencv/issues/24057
// see https://github.com/opencv/opencv/issues/25165
/*
  cv2.add(numeric, numeric) -> Supported, but it works same strange behavious to convert CV_64F as C++.
  cv2.add(numeric, array)   -> Supported.
  cv2.add(numeric, tuple)   -> Not Supported
  cv2.add(array,   numeric) -> Supported.
  cv2.add(array,   array)   -> Supported.
  cv2.add(array,   tuple)   -> Supported
  cv2.add(tuple,   numeric) -> Not Supported.
  cv2.add(tuple,   array)   -> Supported.
  cv2.add(tuple,   tuple)   -> Not Supported.
*/

#define COND(x,y) ((x << 8)+y)

enum ConvRes
{
    None,
    AsMat,
    AsUMat,
    AsScalar,
    AsLong,
    AsDouble,
};

struct ArgPyToCv
{
    ConvRes  cres;
    Mat      asMat;
    UMat     asUMat;
    Scalar   asScalar;
    long int asLong;
    double   asDouble;
};

static void convert_pyobj_to_cv( PyObject* py_src1, ArgPyToCv& cv_src1,
                                 PyObject* py_src2, ArgPyToCv& cv_src2, bool needUMat)
{
    cv_src1.cres = ConvRes::None;
    cv_src2.cres = ConvRes::None;

    // If one is tuple, others must be array.
    if( PyTuple_Check(py_src1) )
    {
        // py_src1 is numpy.tuple -> cv_src1 is scalar
        if( pyopencv_to(py_src1, cv_src1.asScalar, ArgInfo("src1", 0) ) ) {
            cv_src1.cres = ConvRes::AsScalar;
        }
        // py_src2 is numpy.array -> cv_src2 is Mat or UMat
        if( needUMat ){
            if( pyopencv_to(py_src2, cv_src2.asUMat, ArgInfo("src2", 0) ) ) {
                cv_src2.cres = ConvRes::AsUMat;
            }
        }else{
            if( pyopencv_to(py_src2, cv_src2.asMat, ArgInfo("src2", 0) ) ) {
                cv_src2.cres = ConvRes::AsMat;
            }
        }
        return;
    }
    if( PyTuple_Check(py_src2) )
    {
        // py_src1 is numpy.array -> cv_src1 is Mat or UMat
        if( needUMat ){
            if( pyopencv_to(py_src1, cv_src1.asUMat, ArgInfo("src1", 0) ) ) {
                cv_src1.cres = ConvRes::AsUMat;
            }
        }else{
            if( pyopencv_to(py_src1, cv_src1.asMat, ArgInfo("src1", 0) ) ) {
                cv_src1.cres = ConvRes::AsMat;
            }
        }
        // py_src2 is numpy.tuple -> cv_src2 is scalar
        if( pyopencv_to(py_src2, cv_src2.asScalar, ArgInfo("src2", 0) ) ) {
            cv_src2.cres = ConvRes::AsScalar;
        }
        return;
    }

    if( PyInt_Check(py_src1) ) {
        cv_src1.asLong = PyInt_AsLong((PyObject*)py_src1);
        cv_src1.cres = ConvRes::AsLong;
    }else if( PyFloat_Check(py_src1) ) {
        cv_src1.asDouble = PyFloat_AsDouble((PyObject*)py_src1);
        cv_src1.cres = ConvRes::AsDouble;
    }else{
        if( needUMat ) {
            if( pyopencv_to(py_src1, cv_src1.asUMat, ArgInfo("src1", 0) ) ) {
                cv_src1.cres = ConvRes::AsUMat;
            }
        }else{
            if( pyopencv_to(py_src1, cv_src1.asMat, ArgInfo("src1", 0) ) ) {
                cv_src1.cres = ConvRes::AsMat;
            }
        }
    }

    if( PyInt_Check(py_src2) ) {
        cv_src2.asLong = PyInt_AsLong((PyObject*)py_src2);
        cv_src2.cres = ConvRes::AsLong;
    }else if( PyFloat_Check(py_src2) ) {
        cv_src2.asDouble = PyFloat_AsDouble((PyObject*)py_src2);
        cv_src2.cres = ConvRes::AsDouble;
    }else{
        if( needUMat ) {
            if( pyopencv_to(py_src2, cv_src2.asUMat, ArgInfo("src2", 0) ) ) {
                cv_src2.cres = ConvRes::AsUMat;
            }
        }else{
            if( pyopencv_to(py_src2, cv_src2.asMat, ArgInfo("src2", 0) ) ) {
                cv_src2.cres = ConvRes::AsMat;
            }
        }
    }
}

static PyObject* pyopencv_cv_add(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

#define PROC(x,y) \
            if (isUMat) { \
                ERRWRAP2(cv::add(x, y, dstU, mask, dtype)) \
            } else { \
                ERRWRAP2(cv::add(x, y, dst,  mask, dtype)) \
            }
#define PROCS(a,b) \
            case COND(a,    ConvRes::AsMat):    PROC(b, cv_src2.asMat);       break; \
            case COND(a,    ConvRes::AsUMat):   PROC(b, cv_src2.asUMat);      break; \
            case COND(a,    ConvRes::AsScalar): PROC(b, cv_src2.asScalar);    break; \
            case COND(a,    ConvRes::AsLong):   PROC(b, cv_src2.asLong);      break; \
            case COND(a,    ConvRes::AsDouble): PROC(b, cv_src2.asDouble);    break;

    pyPrepareArgumentConversionErrorsStorage(2);

    const bool isUMatList[] = { false /* Mat */, true /* UMat */ };
    for(bool isUMat : isUMatList )
    {
        PyObject* pyobj_src1 = NULL;
        ArgPyToCv cv_src1;
        PyObject* pyobj_src2 = NULL;
        ArgPyToCv cv_src2;
        PyObject* pyobj_dst = NULL;
        Mat dst;
        UMat dstU;
        PyObject* pyobj_mask = NULL;
        Mat mask;
        PyObject* pyobj_dtype = NULL;
        int dtype=-1;

        const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
        if( PyArg_ParseTupleAndKeywords(py_args, kw, "OO|OOO:add",
                                        (char**)keywords,
                                        &pyobj_src1,
                                        &pyobj_src2,
                                        &pyobj_dst,
                                        &pyobj_mask,
                                        &pyobj_dtype) &&
            (isUMat)?( pyopencv_to_safe(pyobj_dst, dstU, ArgInfo("dst", 1))):
                     ( pyopencv_to_safe(pyobj_dst, dst,  ArgInfo("dst", 1))) &&
            pyopencv_to_safe(pyobj_mask, mask, ArgInfo("mask", 0)) &&
            pyopencv_to_safe(pyobj_dtype, dtype, ArgInfo("dtype", 0))
        ){
            bool ret = true;
            convert_pyobj_to_cv( pyobj_src1, cv_src1,
                                 pyobj_src2, cv_src2, isUMat);
            switch( COND(cv_src1.cres, cv_src2.cres) )
            {
            PROCS(ConvRes::AsMat,    cv_src1.asMat    );
            PROCS(ConvRes::AsUMat,   cv_src1.asUMat   );
            PROCS(ConvRes::AsScalar, cv_src1.asScalar );
            PROCS(ConvRes::AsLong,   cv_src1.asLong   );
            PROCS(ConvRes::AsDouble, cv_src1.asDouble );
            default: ret = false; break;
            }
            if( ret ) {
                return (isUMat) ? pyopencv_from(dstU) : pyopencv_from(dst);
            }
        }
        pyPopulateArgumentConversionErrors();
    }

    pyRaiseCVOverloadException("add");
#undef PROC
#undef PROCS
    return NULL;
}

static PyObject* pyopencv_cv_subtract(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

#define PROC(x,y) \
            if (isUMat) { \
                ERRWRAP2(cv::subtract(x, y, dstU, mask, dtype)) \
            } else { \
                ERRWRAP2(cv::subtract(x, y, dst, mask, dtype)) \
            }
#define PROCS(a,b) \
            case COND(a,    ConvRes::AsMat):    PROC(b, cv_src2.asMat);       break; \
            case COND(a,    ConvRes::AsUMat):   PROC(b, cv_src2.asUMat);      break; \
            case COND(a,    ConvRes::AsScalar): PROC(b, cv_src2.asScalar);    break; \
            case COND(a,    ConvRes::AsLong):   PROC(b, cv_src2.asLong);      break; \
            case COND(a,    ConvRes::AsDouble): PROC(b, cv_src2.asDouble);    break;

    pyPrepareArgumentConversionErrorsStorage(2);

    const bool isUMatList[] = { false /* Mat */, true /* UMat */ };
    for(bool isUMat : isUMatList )
    {
        PyObject* pyobj_src1 = NULL;
        ArgPyToCv cv_src1;
        PyObject* pyobj_src2 = NULL;
        ArgPyToCv cv_src2;
        PyObject* pyobj_dst = NULL;
        Mat dst;
        UMat dstU;
        PyObject* pyobj_mask = NULL;
        Mat mask;
        PyObject* pyobj_dtype = NULL;
        int dtype=-1;

        const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
        if( PyArg_ParseTupleAndKeywords(py_args, kw, "OO|OOO:subtract",
                                        (char**)keywords,
                                        &pyobj_src1,
                                        &pyobj_src2,
                                        &pyobj_dst,
                                        &pyobj_mask,
                                        &pyobj_dtype) &&
            (isUMat)?( pyopencv_to_safe(pyobj_dst, dstU, ArgInfo("dst", 1))):
                     ( pyopencv_to_safe(pyobj_dst, dst,  ArgInfo("dst", 1))) &&
            pyopencv_to_safe(pyobj_mask, mask, ArgInfo("mask", 0)) &&
            pyopencv_to_safe(pyobj_dtype, dtype, ArgInfo("dtype", 0))
        ){
            bool ret = true;
            convert_pyobj_to_cv( pyobj_src1, cv_src1,
                                 pyobj_src2, cv_src2, isUMat );
            switch( COND(cv_src1.cres, cv_src2.cres) )
            {
            PROCS(ConvRes::AsMat,    cv_src1.asMat    );
            PROCS(ConvRes::AsUMat,   cv_src1.asUMat   );
            PROCS(ConvRes::AsScalar, cv_src1.asScalar );
            PROCS(ConvRes::AsLong,   cv_src1.asLong   );
            PROCS(ConvRes::AsDouble, cv_src1.asDouble );
            default: ret = false; break;
            }
            if( ret ) {
                return (isUMat) ? pyopencv_from(dstU) : pyopencv_from(dst);
            }
        }
        pyPopulateArgumentConversionErrors();
    }

    pyRaiseCVOverloadException("subtract");
#undef PROC
#undef PROCS
    return NULL;
}

static PyObject* pyopencv_cv_absdiff(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

#define PROC(x,y) \
            if (isUMat) { \
                ERRWRAP2(cv::absdiff(x, y, dstU)) \
            } else { \
                ERRWRAP2(cv::absdiff(x, y, dst)) \
            }
#define PROCS(a,b) \
            case COND(a,    ConvRes::AsMat):    PROC(b, cv_src2.asMat);       break; \
            case COND(a,    ConvRes::AsUMat):   PROC(b, cv_src2.asUMat);      break; \
            case COND(a,    ConvRes::AsScalar): PROC(b, cv_src2.asScalar);    break; \
            case COND(a,    ConvRes::AsLong):   PROC(b, cv_src2.asLong);      break; \
            case COND(a,    ConvRes::AsDouble): PROC(b, cv_src2.asDouble);    break;

    pyPrepareArgumentConversionErrorsStorage(2);

    const bool isUMatList[] = { false /* Mat */, true /* UMat */ };
    for(bool isUMat : isUMatList )
    {
        PyObject* pyobj_src1 = NULL;
        ArgPyToCv cv_src1;
        PyObject* pyobj_src2 = NULL;
        ArgPyToCv cv_src2;
        PyObject* pyobj_dst = NULL;
        Mat dst;
        UMat dstU;

        const char* keywords[] = { "src1", "src2", "dst", NULL };
        if( PyArg_ParseTupleAndKeywords(py_args, kw, "OO|O:absdiff",
                                        (char**)keywords,
                                        &pyobj_src1,
                                        &pyobj_src2,
                                        &pyobj_dst) &&
            (isUMat)?( pyopencv_to_safe(pyobj_dst, dstU, ArgInfo("dst", 1))):
                     ( pyopencv_to_safe(pyobj_dst, dst,  ArgInfo("dst", 1)))
        ){
            bool ret = true;
            convert_pyobj_to_cv( pyobj_src1, cv_src1,
                                 pyobj_src2, cv_src2, isUMat );
            switch( COND(cv_src1.cres, cv_src2.cres) )
            {
            PROCS(ConvRes::AsMat,    cv_src1.asMat    );
            PROCS(ConvRes::AsUMat,   cv_src1.asUMat   );
            PROCS(ConvRes::AsScalar, cv_src1.asScalar );
            PROCS(ConvRes::AsLong,   cv_src1.asLong   );
            PROCS(ConvRes::AsDouble, cv_src1.asDouble );
            default: ret = false; break;
            }
            if( ret ) {
                return (isUMat) ? pyopencv_from(dstU) : pyopencv_from(dst);
            }
        }
        pyPopulateArgumentConversionErrors();
    }

    pyRaiseCVOverloadException("absdiff");
#undef PROC
#undef PROCS
    return NULL;
}

static PyObject* pyopencv_cv_divide(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

#define PROC(x,y) \
            if (isUMat) { \
                ERRWRAP2(cv::divide(x, y, dstU, scale, dtype)) \
            } else { \
                ERRWRAP2(cv::divide(x, y, dst, scale, dtype)) \
            }
#define PROCS(a,b) \
            case COND(a,    ConvRes::AsMat):    PROC(b, cv_src2.asMat);       break; \
            case COND(a,    ConvRes::AsUMat):   PROC(b, cv_src2.asUMat);      break; \
            case COND(a,    ConvRes::AsScalar): PROC(b, cv_src2.asScalar);    break; \
            case COND(a,    ConvRes::AsLong):   PROC(b, cv_src2.asLong);      break; \
            case COND(a,    ConvRes::AsDouble): PROC(b, cv_src2.asDouble);    break;

    pyPrepareArgumentConversionErrorsStorage(2);

    const bool isUMatList[] = { false /* Mat */, true /* UMat */ };
    for(bool isUMat : isUMatList )
    {
        PyObject* pyobj_src1 = NULL;
        ArgPyToCv cv_src1;
        PyObject* pyobj_src2 = NULL;
        ArgPyToCv cv_src2;
        PyObject* pyobj_dst = NULL;
        Mat dst;
        UMat dstU;
        PyObject* pyobj_scale = NULL;
        double scale=1;
        PyObject* pyobj_dtype = NULL;
        int dtype=-1;

        const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
        if( PyArg_ParseTupleAndKeywords(py_args, kw, "OO|OOO:divide",
                                        (char**)keywords,
                                        &pyobj_src1,
                                        &pyobj_src2,
                                        &pyobj_dst,
                                        &pyobj_scale,
                                        &pyobj_dtype) &&
            (isUMat)?( pyopencv_to_safe(pyobj_dst, dstU, ArgInfo("dst", 1))):
                     ( pyopencv_to_safe(pyobj_dst, dst,  ArgInfo("dst", 1))) &&
            pyopencv_to_safe(pyobj_scale, scale, ArgInfo("scale", 0)) &&
            pyopencv_to_safe(pyobj_dtype, dtype, ArgInfo("dtype", 0))
        ){
            bool ret = true;
            convert_pyobj_to_cv( pyobj_src1, cv_src1,
                                 pyobj_src2, cv_src2, isUMat );
            switch( COND(cv_src1.cres, cv_src2.cres) )
            {
            PROCS(ConvRes::AsMat,    cv_src1.asMat    );
            PROCS(ConvRes::AsUMat,   cv_src1.asUMat   );
            PROCS(ConvRes::AsScalar, cv_src1.asScalar );
            PROCS(ConvRes::AsLong,   cv_src1.asLong   );
            PROCS(ConvRes::AsDouble, cv_src1.asDouble );
            default: ret = false; break;
            }
            if( ret ) {
                return (isUMat) ? pyopencv_from(dstU) : pyopencv_from(dst);
            }
        }
        pyPopulateArgumentConversionErrors();
    }

    pyRaiseCVOverloadException("divide");
#undef PROC
#undef PROCS
    return NULL;
}

static PyObject* pyopencv_cv_multiply(PyObject*, PyObject* py_args, PyObject* kw)
{
    using namespace cv;

#define PROC(x,y) \
            if (isUMat) { \
                ERRWRAP2(cv::multiply(x, y, dstU, scale, dtype)) \
            } else { \
                ERRWRAP2(cv::multiply(x, y, dst, scale, dtype)) \
            }
#define PROCS(a,b) \
            case COND(a,    ConvRes::AsMat):    PROC(b, cv_src2.asMat);       break; \
            case COND(a,    ConvRes::AsUMat):   PROC(b, cv_src2.asUMat);      break; \
            case COND(a,    ConvRes::AsScalar): PROC(b, cv_src2.asScalar);    break; \
            case COND(a,    ConvRes::AsLong):   PROC(b, cv_src2.asLong);      break; \
            case COND(a,    ConvRes::AsDouble): PROC(b, cv_src2.asDouble);    break;

    pyPrepareArgumentConversionErrorsStorage(2);

    const bool isUMatList[] = { false /* Mat */, true /* UMat */ };
    for(bool isUMat : isUMatList )
    {
        PyObject* pyobj_src1 = NULL;
        ArgPyToCv cv_src1;
        PyObject* pyobj_src2 = NULL;
        ArgPyToCv cv_src2;
        PyObject* pyobj_dst = NULL;
        Mat dst;
        UMat dstU;
        PyObject* pyobj_scale = NULL;
        double scale=1;
        PyObject* pyobj_dtype = NULL;
        int dtype=-1;

        const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
        if( PyArg_ParseTupleAndKeywords(py_args, kw, "OO|OOO:multiply",
                                        (char**)keywords,
                                        &pyobj_src1,
                                        &pyobj_src2,
                                        &pyobj_dst,
                                        &pyobj_scale,
                                        &pyobj_dtype) &&
            (isUMat)?( pyopencv_to_safe(pyobj_dst, dstU, ArgInfo("dst", 1))):
                     ( pyopencv_to_safe(pyobj_dst, dst,  ArgInfo("dst", 1))) &&
            pyopencv_to_safe(pyobj_scale, scale, ArgInfo("scale", 0)) &&
            pyopencv_to_safe(pyobj_dtype, dtype, ArgInfo("dtype", 0))
        ){
            bool ret = true;
            convert_pyobj_to_cv( pyobj_src1, cv_src1,
                                 pyobj_src2, cv_src2, isUMat );
            switch( COND(cv_src1.cres, cv_src2.cres) )
            {
            PROCS(ConvRes::AsMat,    cv_src1.asMat    );
            PROCS(ConvRes::AsUMat,   cv_src1.asUMat   );
            PROCS(ConvRes::AsScalar, cv_src1.asScalar );
            PROCS(ConvRes::AsLong,   cv_src1.asLong   );
            PROCS(ConvRes::AsDouble, cv_src1.asDouble );
            default: ret = false; break;
            }
            if( ret ) {
                return (isUMat) ? pyopencv_from(dstU) : pyopencv_from(dst);
            }
        }
        pyPopulateArgumentConversionErrors();
    }

    pyRaiseCVOverloadException("multiply");
#undef PROC
#undef PROCS
    return NULL;
}
#undef COND

#endif // HAVE_OPENCV_CORE
