#include <Python.h>

#define MODULESTR "cv2"

#include "numpy/ndarrayobject.h"
#include "cv2.hpp"

using cv::flann::IndexParams;
using cv::flann::SearchParams;
using namespace cv;

typedef cv::softcascade::ChannelFeatureBuilder softcascade_ChannelFeatureBuilder;

typedef std::vector<KeyPoint> vector_KeyPoint;
typedef std::vector<DMatch> vector_DMatch;
typedef std::vector<std::vector<DMatch> > vector_vector_DMatch;

typedef Ptr<Algorithm> Ptr_Algorithm;
typedef Ptr<FeatureDetector> Ptr_FeatureDetector;
typedef Ptr<DescriptorExtractor> Ptr_DescriptorExtractor;
typedef Ptr<Feature2D> Ptr_Feature2D;
typedef Ptr<DescriptorMatcher> Ptr_DescriptorMatcher;
typedef Ptr<BackgroundSubtractor> Ptr_BackgroundSubtractor;
typedef Ptr<BackgroundSubtractorMOG> Ptr_BackgroundSubtractorMOG;
typedef Ptr<BackgroundSubtractorMOG2> Ptr_BackgroundSubtractorMOG2;
typedef Ptr<BackgroundSubtractorGMG> Ptr_BackgroundSubtractorGMG;

typedef Ptr<StereoMatcher> Ptr_StereoMatcher;
typedef Ptr<StereoBM> Ptr_StereoBM;
typedef Ptr<StereoSGBM> Ptr_StereoSGBM;

typedef Ptr<Tonemap> Ptr_Tonemap;
typedef Ptr<TonemapDrago> Ptr_TonemapDrago;
typedef Ptr<TonemapReinhard> Ptr_TonemapReinhard;
typedef Ptr<TonemapDurand> Ptr_TonemapDurand;
typedef Ptr<TonemapMantiuk> Ptr_TonemapMantiuk;
typedef Ptr<AlignMTB> Ptr_AlignMTB;
typedef Ptr<CalibrateDebevec> Ptr_CalibrateDebevec;
typedef Ptr<CalibrateRobertson> Ptr_CalibrateRobertson;
typedef Ptr<MergeDebevec> Ptr_MergeDebevec;
typedef Ptr<MergeRobertson> Ptr_MergeRobertson;
typedef Ptr<MergeMertens> Ptr_MergeMertens;
typedef Ptr<MergeRobertson> Ptr_MergeRobertson;

typedef Ptr<cv::softcascade::ChannelFeatureBuilder> Ptr_ChannelFeatureBuilder;
typedef Ptr<CLAHE> Ptr_CLAHE;
typedef Ptr<LineSegmentDetector > Ptr_LineSegmentDetector;

typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;

typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;
typedef Ptr<flann::IndexParams> Ptr_flann_IndexParams;
typedef Ptr<flann::SearchParams> Ptr_flann_SearchParams;

typedef Ptr<FaceRecognizer> Ptr_FaceRecognizer;

#include "cv2support.cpp"

bool pyopencv_coerce(PyObject *obj, cv::TermCriteria& dst);
bool pyopencv_coerce(PyObject *obj, CvTermCriteria& dst);

template<>
PyObject* pyopencv_from(const cvflann_flann_algorithm_t& value)
{
    return PyInt_FromLong(int(value));
}

template<>
PyObject* pyopencv_from(const cvflann_flann_distance_t& value)
{
    return PyInt_FromLong(int(value));
}

template<>
bool pyopencv_to(PyObject* obj, Range& r, const ArgInfo info)
{
    (void)info.name;
    if(!obj || obj == Py_None)
        return true;
    if(PyObject_Size(obj) == 0)
    {
        r = Range::all();
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start, &r.end) > 0;
}

template<>
PyObject* pyopencv_from(const Range& r)
{
    return Py_BuildValue("(ii)", r.start, r.end);
}

template<>
bool pyopencv_to(PyObject* obj, CvSlice& r, const ArgInfo info)
{
    (void)info.name;
    if(!obj || obj == Py_None)
        return true;
    if(PyObject_Size(obj) == 0)
    {
        r = CV_WHOLE_SEQ;
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start_index, &r.end_index) > 0;
}

template<>
PyObject* pyopencv_from(const CvSlice& r)
{
    return Py_BuildValue("(ii)", r.start_index, r.end_index);
}

template<> struct pyopencvVecConverter<KeyPoint>
{
    static bool to(PyObject* obj, std::vector<KeyPoint>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<KeyPoint>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<DMatch>
{
    static bool to(PyObject* obj, std::vector<DMatch>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<DMatch>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

bool pyopencv_coerce(PyObject *obj, CvTermCriteria& dst)
{
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.max_iter, &dst.epsilon) > 0;
}

bool pyopencv_coerce(PyObject *obj, TermCriteria& dst)
{
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.maxCount, &dst.epsilon) > 0;
}

template<>
bool pyopencv_to(PyObject *obj, TermCriteria& dst, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, dst);
}

template<>
PyObject* pyopencv_from(const TermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.maxCount, src.epsilon);
}

template<>
bool pyopencv_to(PyObject *o, cv::flann::IndexParams& p, const ArgInfo info)
{
    (void)info.name;
    bool ok = false;
    PyObject* keys = PyObject_CallMethod(o,(char*)"keys",0);
    PyObject* values = PyObject_CallMethod(o,(char*)"values",0);

    if( keys && values )
    {
        int i, n = (int)PyList_GET_SIZE(keys);
        for( i = 0; i < n; i++ )
        {
            PyObject* key = PyList_GET_ITEM(keys, i);
            PyObject* item = PyList_GET_ITEM(values, i);
            if( !PyString_Check(key) )
                break;
            String k = PyString_AsString(key);
            if( PyString_Check(item) )
            {
                const char* value = PyString_AsString(item);
                p.setString(k, value);
            }
            else if( !!PyBool_Check(item) )
                p.setBool(k, item == Py_True);
            else if( PyInt_Check(item) )
            {
                int value = (int)PyInt_AsLong(item);
                if( strcmp(k.c_str(), "algorithm") == 0 )
                    p.setAlgorithm(value);
                else
                    p.setInt(k, value);
            }
            else if( PyFloat_Check(item) )
            {
                double value = PyFloat_AsDouble(item);
                p.setDouble(k, value);
            }
            else
                break;
        }
        ok = i == n && !PyErr_Occurred();
    }

    Py_XDECREF(keys);
    Py_XDECREF(values);
    return ok;
}

template<>
bool pyopencv_to(PyObject* obj, cv::flann::SearchParams & value, const ArgInfo info)
{
    return pyopencv_to<cv::flann::IndexParams>(obj, value, info);
}

template <typename T>
bool pyopencv_to(PyObject *o, Ptr<T>& p, const ArgInfo info)
{
    p = makePtr<T>();
    return pyopencv_to(o, *p, info);
}

template<>
bool pyopencv_to(PyObject *o, cvflann::flann_distance_t& dist, const ArgInfo info)
{
    int d = (int)dist;
    bool ok = pyopencv_to(o, d, info);
    dist = (cvflann::flann_distance_t)d;
    return ok;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: REMOVE used only by ml wrapper

template<>
bool pyopencv_to(PyObject *obj, CvTermCriteria& dst, const ArgInfo info)
{
    (void)info.name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.max_iter, &dst.epsilon) > 0;
}

template<>
PyObject* pyopencv_from(CvDTreeNode* const & node)
{
    double value = node->value;
    int ivalue = cvRound(value);
    return value == ivalue ? PyInt_FromLong(ivalue) : PyFloat_FromDouble(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void OnMouse(int event, int x, int y, int flags, void* param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("iiiiO", event, x, y, flags, PyTuple_GetItem(o, 1));

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "window_name", "on_mouse", "param", NULL };
    char* name;
    PyObject *on_mouse;
    PyObject *param = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &name, &on_mouse, &param))
        return NULL;
    if (!PyCallable_Check(on_mouse)) {
        PyErr_SetString(PyExc_TypeError, "on_mouse must be callable");
        return NULL;
    }
    if (param == NULL) {
        param = Py_None;
    }
    ERRWRAP2(setMouseCallback(name, OnMouse, Py_BuildValue("OO", on_mouse, param)));
    Py_RETURN_NONE;
}

static void OnChange(int pos, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("(i)", pos);
    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int *value = new int;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, value, &count, &on_change))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "on_change must be callable");
        return NULL;
    }
    ERRWRAP2(createTrackbar(trackbar_name, window_name, value, count, OnChange, Py_BuildValue("OO", on_change, Py_None)));
    Py_RETURN_NONE;
}

///////////////////////////////////////////////////////////////////////////////////////

static int convert_to_char(PyObject *o, char *dst, const ArgInfo info)
{
  if (PyString_Check(o) && PyString_Size(o) == 1) {
    *dst = PyString_AsString(o)[0];
    return 1;
  } else {
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", info.name);
  }
}

#include "cv_generated_types.h"
#include "cv_generated_funcs.h"

static PyMethodDef methods[] = {

#include "cv_generated_func_tab.h"
  {"createTrackbar", pycvCreateTrackbar, METH_VARARGS, "createTrackbar(trackbarName, windowName, value, count, onChange) -> None"},
  {"setMouseCallback", (PyCFunction)pycvSetMouseCallback, METH_VARARGS | METH_KEYWORDS, "setMouseCallback(windowName, onMouse [, param]) -> None"},
  {NULL, NULL},
};

/************************************************************************/
/* Module init */


#if PY_MAJOR_VERSION >= 3
extern "C" CV_EXPORTS PyObject* PyInit_cv2();
static struct PyModuleDef cv2_moduledef =
{
    PyModuleDef_HEAD_INIT,
    MODULESTR,
    "Python wrapper for OpenCV.",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    methods
};

PyObject* PyInit_cv2()
#else
extern "C" CV_EXPORTS void initcv2();

void initcv2()
#endif
{
  import_array();

#include "cv_generated_type_reg.h"

#if PY_MAJOR_VERSION >= 3
  PyObject* m = PyModule_Create(&cv2_moduledef);
#else
  PyObject* m = Py_InitModule(MODULESTR, methods);
#endif
  PyObject* d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "__version__", PyString_FromString(CV_VERSION));

  opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", opencv_error);

#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
//#define PUBLISHU(I) PyDict_SetItemString(d, #I, PyLong_FromUnsignedLong(I))
#define PUBLISH2(I, value) PyDict_SetItemString(d, #I, PyLong_FromLong(value))

  PUBLISH(CV_8U);
  PUBLISH(CV_8UC1);
  PUBLISH(CV_8UC2);
  PUBLISH(CV_8UC3);
  PUBLISH(CV_8UC4);
  PUBLISH(CV_8S);
  PUBLISH(CV_8SC1);
  PUBLISH(CV_8SC2);
  PUBLISH(CV_8SC3);
  PUBLISH(CV_8SC4);
  PUBLISH(CV_16U);
  PUBLISH(CV_16UC1);
  PUBLISH(CV_16UC2);
  PUBLISH(CV_16UC3);
  PUBLISH(CV_16UC4);
  PUBLISH(CV_16S);
  PUBLISH(CV_16SC1);
  PUBLISH(CV_16SC2);
  PUBLISH(CV_16SC3);
  PUBLISH(CV_16SC4);
  PUBLISH(CV_32S);
  PUBLISH(CV_32SC1);
  PUBLISH(CV_32SC2);
  PUBLISH(CV_32SC3);
  PUBLISH(CV_32SC4);
  PUBLISH(CV_32F);
  PUBLISH(CV_32FC1);
  PUBLISH(CV_32FC2);
  PUBLISH(CV_32FC3);
  PUBLISH(CV_32FC4);
  PUBLISH(CV_64F);
  PUBLISH(CV_64FC1);
  PUBLISH(CV_64FC2);
  PUBLISH(CV_64FC3);
  PUBLISH(CV_64FC4);

#include "cv_generated_const_reg.h"
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
