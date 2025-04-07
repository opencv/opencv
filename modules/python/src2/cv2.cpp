// must be defined before importing numpy headers
// https://numpy.org/doc/1.17/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL opencv_ARRAY_API

#include "cv2.hpp"

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "pyopencv_generated_include.h"
#include "opencv2/core/types_c.h"


#include "cv2_util.hpp"
#include "cv2_numpy.hpp"
#include "cv2_convert.hpp"
#include "cv2_highgui.hpp"

using namespace cv;

typedef std::vector<uchar> vector_uchar;
typedef std::vector<char> vector_char;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<double> vector_double;
typedef std::vector<size_t> vector_size_t;
typedef std::vector<Point> vector_Point;
typedef std::vector<Point2f> vector_Point2f;
typedef std::vector<Point3f> vector_Point3f;
typedef std::vector<Size> vector_Size;
typedef std::vector<Vec2f> vector_Vec2f;
typedef std::vector<Vec3f> vector_Vec3f;
typedef std::vector<Vec4f> vector_Vec4f;
typedef std::vector<Vec6f> vector_Vec6f;
typedef std::vector<Vec4i> vector_Vec4i;
typedef std::vector<Rect> vector_Rect;
typedef std::vector<Rect2d> vector_Rect2d;
typedef std::vector<RotatedRect> vector_RotatedRect;
typedef std::vector<KeyPoint> vector_KeyPoint;
typedef std::vector<Mat> vector_Mat;
typedef std::vector<std::vector<Mat> > vector_vector_Mat;
typedef std::vector<UMat> vector_UMat;
typedef std::vector<DMatch> vector_DMatch;
typedef std::vector<String> vector_String;
typedef std::vector<std::string> vector_string;
typedef std::vector<Scalar> vector_Scalar;
typedef std::vector<aruco::Dictionary> vector_Dictionary;

typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<Point> > vector_vector_Point;
typedef std::vector<std::vector<Point2f> > vector_vector_Point2f;
typedef std::vector<std::vector<Point3f> > vector_vector_Point3f;
typedef std::vector<std::vector<DMatch> > vector_vector_DMatch;
typedef std::vector<std::vector<KeyPoint> > vector_vector_KeyPoint;

// enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };


///////////////////////////////////////////////////////////////////////////////////////

static int convert_to_char(PyObject *o, char *dst, const ArgInfo& info)
{
    std::string str;
    if (getUnicodeString(o, str))
    {
        *dst = str[0];
        return 1;
    }
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", info.name);
}

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif


#include "pyopencv_generated_enums.h"

#ifdef CVPY_DYNAMIC_INIT
#define CVPY_TYPE(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, _1, _2, SCOPE) CVPY_TYPE_DECLARE_DYNAMIC(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, SCOPE)
#else
#define CVPY_TYPE(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, _1, _2, SCOPE) CVPY_TYPE_DECLARE(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, SCOPE)
#endif
#include "pyopencv_generated_types.h"
#undef CVPY_TYPE
#include "pyopencv_custom_headers.h"

#include "pyopencv_generated_types_content.h"
#include "pyopencv_generated_funcs.h"

static PyObject* pycvRegisterMatType(PyObject *self, PyObject *value)
{
    CV_LOG_DEBUG(NULL, cv::format("pycvRegisterMatType %p %p\n", self, value));

    if (0 == PyType_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Type argument is expected");
        return NULL;
    }

    Py_INCREF(value);
    pyopencv_Mat_TypePtr = (PyTypeObject*)value;

    Py_RETURN_NONE;
}

static PyMethodDef special_methods[] = {
  {"_registerMatType", (PyCFunction)(pycvRegisterMatType), METH_O, "_registerMatType(cv.Mat) -> None (Internal)"},
  {"redirectError", CV_PY_FN_WITH_KW(pycvRedirectError), "redirectError(onError) -> None"},
#ifdef HAVE_OPENCV_HIGHGUI
  {"createTrackbar", (PyCFunction)pycvCreateTrackbar, METH_VARARGS, "createTrackbar(trackbarName, windowName, value, count, onChange) -> None"},
  {"createButton", CV_PY_FN_WITH_KW(pycvCreateButton), "createButton(buttonName, onChange [, userData, buttonType, initialButtonState]) -> None"},
  {"setMouseCallback", CV_PY_FN_WITH_KW(pycvSetMouseCallback), "setMouseCallback(windowName, onMouse [, param]) -> None"},
#endif
#ifdef HAVE_OPENCV_DNN
  {"dnn_registerLayer", CV_PY_FN_WITH_KW(pyopencv_cv_dnn_registerLayer), "registerLayer(type, class) -> None"},
  {"dnn_unregisterLayer", CV_PY_FN_WITH_KW(pyopencv_cv_dnn_unregisterLayer), "unregisterLayer(type) -> None"},
#endif
  {NULL, NULL},
};

/************************************************************************/
/* Module init */

struct ConstDef
{
    const char * name;
    long long val;
};

static inline bool strStartsWith(const std::string& str, const std::string& prefix) {
    return prefix.empty() || \
        (str.size() >= prefix.size() && std::memcmp(str.data(), prefix.data(), prefix.size()) == 0);
}

static inline bool strEndsWith(const std::string& str, char symbol) {
    return !str.empty() && str[str.size() - 1] == symbol;
}

/**
 * \brief Creates a submodule of the `root`. Missing parents submodules
 * are created as needed. If name equals to parent module name than
 * borrowed reference to parent module is returned (no reference counting
 * are done).
 * Submodule lifetime is managed by the parent module.
 * If nested submodules are created than the lifetime is managed by the
 * predecessor submodule in a list.
 *
 * \param parent_module Parent module object.
 * \param name Submodule name.
 * \return borrowed reference to the created submodule.
 *         If any of submodules can't be created than NULL is returned.
 */
static PyObject* createSubmodule(PyObject* parent_module, const std::string& name)
{
    if (!parent_module)
    {
        return PyErr_Format(PyExc_ImportError,
            "Bindings generation error. "
            "Parent module is NULL during the submodule '%s' creation",
            name.c_str()
        );
    }
    if (strEndsWith(name, '.'))
    {
        return PyErr_Format(PyExc_ImportError,
            "Bindings generation error. "
            "Submodule can't end with a dot. Got: %s", name.c_str()
        );
    }

    const std::string parent_name = PyModule_GetName(parent_module);

    /// Special case handling when caller tries to register a submodule of the parent module with
    /// the same name
    if (name == parent_name) {
        return parent_module;
    }

    if (!strStartsWith(name, parent_name))
    {
        return PyErr_Format(PyExc_ImportError,
            "Bindings generation error. "
            "Submodule name should always start with a parent module name. "
            "Parent name: %s. Submodule name: %s", parent_name.c_str(),
            name.c_str()
        );
    }

    size_t submodule_name_end = name.find('.', parent_name.size() + 1);
    /// There is no intermediate submodules in the provided name
    if (submodule_name_end == std::string::npos)
    {
        submodule_name_end = name.size();
    }

    PyObject* submodule = parent_module;

    for (size_t submodule_name_start = parent_name.size() + 1;
         submodule_name_start < name.size(); )
    {
        const std::string submodule_name = name.substr(submodule_name_start,
                                                       submodule_name_end - submodule_name_start);

        const std::string full_submodule_name = name.substr(0, submodule_name_end);


        PyObject* parent_module_dict = PyModule_GetDict(submodule);
        /// If submodule already exists it can be found in the parent module dictionary,
        /// otherwise it should be added to it.
        submodule = PyDict_GetItemString(parent_module_dict,
                                         submodule_name.c_str());
        if (!submodule)
        {
            /// Populates global modules dictionary and returns borrowed reference to it
            submodule = PyImport_AddModule(full_submodule_name.c_str());
            if (!submodule)
            {
                /// Return `PyImport_AddModule` NULL with an exception set on failure.
                return NULL;
            }
            /// Populates parent module dictionary. Submodule lifetime should be managed
            /// by the global modules dictionary and parent module dictionary, so Py_DECREF after
            /// successfull call to the `PyDict_SetItemString` is redundant.
            if (PyDict_SetItemString(parent_module_dict, submodule_name.c_str(), submodule) < 0) {
                return PyErr_Format(PyExc_ImportError,
                    "Can't register a submodule '%s' (full name: '%s')",
                    submodule_name.c_str(), full_submodule_name.c_str()
                );
            }
        }

        submodule_name_start = submodule_name_end + 1;

        submodule_name_end = name.find('.', submodule_name_start);
        if (submodule_name_end == std::string::npos) {
            submodule_name_end = name.size();
        }
    }
    return submodule;
}

static bool init_submodule(PyObject * root, const char * name, PyMethodDef * methods, ConstDef * consts)
{
    // traverse and create nested submodules
    PyObject* submodule = createSubmodule(root, name);
    if (!submodule)
    {
        return false;
    }
    // populate module's dict
    PyObject * d = PyModule_GetDict(submodule);
    for (PyMethodDef * m = methods; m->ml_name != NULL; ++m)
    {
        PyObject * method_obj = PyCFunction_NewEx(m, NULL, NULL);
        if (PyDict_SetItemString(d, m->ml_name, method_obj) < 0)
        {
            PyErr_Format(PyExc_ImportError,
                "Can't register function %s in module: %s", m->ml_name, name
            );
            Py_CLEAR(method_obj);
            return false;
        }
        Py_DECREF(method_obj);
    }
    for (ConstDef * c = consts; c->name != NULL; ++c)
    {
        PyObject* const_obj = PyLong_FromLongLong(c->val);
        if (PyDict_SetItemString(d, c->name, const_obj) < 0)
        {
            PyErr_Format(PyExc_ImportError,
                "Can't register constant %s in module %s", c->name, name
            );
            Py_CLEAR(const_obj);
            return false;
        }
        Py_DECREF(const_obj);
    }
    return true;
}

static inline
bool registerTypeInModuleScope(PyObject* module, const char* type_name, PyObject* type_obj)
{
    Py_INCREF(type_obj); /// Give PyModule_AddObject a reference to steal.
    if (PyModule_AddObject(module, type_name, type_obj) < 0)
    {
        PyErr_Format(PyExc_ImportError,
            "Failed to register type '%s' in module scope '%s'",
            type_name, PyModule_GetName(module)
        );
        Py_DECREF(type_obj);
        return false;
    }
    return true;
}

static inline
bool registerTypeInClassScope(PyObject* cls, const char* type_name, PyObject* type_obj)
{
    if (!PyType_CheckExact(cls)) {
        PyErr_Format(PyExc_ImportError,
            "Failed to register type '%s' in class scope. "
            "Scope class object has a wrong type", type_name
        );
        return false;
    }
    if (PyObject_SetAttrString(cls, type_name, type_obj) < 0)
    {
        #ifndef Py_LIMITED_API
            PyObject* cls_dict = reinterpret_cast<PyTypeObject*>(cls)->tp_dict;
            if (PyDict_SetItemString(cls_dict, type_name, type_obj) >= 0) {
                /// Clearing the error set by PyObject_SetAttrString:
                /// TypeError: can't set attributes of built-in/extension type NAME
                PyErr_Clear();
                return true;
            }
        #endif
        const std::string cls_name = getPyObjectNameAttr(cls);
        PyErr_Format(PyExc_ImportError,
            "Failed to register type '%s' in '%s' class scope. Can't update scope dictionary",
            type_name, cls_name.c_str()
        );
        return false;
    }
    return true;
}

static inline
PyObject* getScopeFromTypeObject(PyObject* obj, const std::string& scope_name)
{
    if (!PyType_CheckExact(obj)) {
        const std::string type_name = getPyObjectNameAttr(obj);
        return PyErr_Format(PyExc_ImportError,
            "Failed to get scope from type '%s' "
            "Scope class object has a wrong type", type_name.c_str()
        );
    }
    /// When using LIMITED API all classes are registered in the heap
#if defined(Py_LIMITED_API)
    return PyObject_GetAttrString(obj, scope_name.c_str());
#else
    /// Otherwise classes may be registed on the stack or heap
    PyObject* type_dict = reinterpret_cast<PyTypeObject*>(obj)->tp_dict;
    if (!type_dict) {
        const std::string type_name = getPyObjectNameAttr(obj);
        return PyErr_Format(PyExc_ImportError,
            "Failed to get scope from type '%s' "
            "Type dictionary is not available", type_name.c_str()
        );
    }
    return PyDict_GetItemString(type_dict, scope_name.c_str());
#endif // Py_LIMITED_API
}

static inline
PyObject* findTypeScope(PyObject* root_module, const std::string& scope_name)
{
    PyObject* scope = root_module;
    if (scope_name.empty())
    {
        return scope;
    }
    /// Starting with 1 to omit leading dot in the scope name
    size_t name_end = scope_name.find('.', 1);
    if (name_end == std::string::npos)
    {
        name_end = scope_name.size();
    }
    for (size_t name_start = 1; name_start < scope_name.size() && scope; )
    {
        const std::string current_scope_name = scope_name.substr(name_start,
                                                                 name_end - name_start);

        if (PyModule_CheckExact(scope))
        {
            PyObject* scope_dict = PyModule_GetDict(scope);
            if (!scope_dict)
            {
                return PyErr_Format(PyExc_ImportError,
                    "Scope '%s' dictionary is not available during the search for "
                    " the '%s' scope object", current_scope_name.c_str(),
                    scope_name.c_str()
                );
            }

            scope = PyDict_GetItemString(scope_dict, current_scope_name.c_str());
        }
        else if (PyType_CheckExact(scope))
        {
            scope = getScopeFromTypeObject(scope, current_scope_name);
        }
        else
        {
            return PyErr_Format(PyExc_ImportError,
                "Can't find scope '%s'. '%s' doesn't reference a module or a class",
                 scope_name.c_str(), current_scope_name.c_str()
            );
        }


        name_start = name_end + 1;
        name_end = scope_name.find('.', name_start);
        if (name_end == std::string::npos)
        {
            name_end = scope_name.size();
        }
    }
    if (!scope)
    {
        return PyErr_Format(PyExc_ImportError,
            "Module or class with name '%s' can't be found in '%s' module",
            scope_name.c_str(), PyModule_GetName(root_module)
        );
    }
    return scope;
}

static bool registerNewType(PyObject* root_module, const char* type_name,
                            PyObject* type_obj, const std::string& scope_name)
{
    PyObject* scope = findTypeScope(root_module, scope_name);

    /// If scope can't be found it means that there is an error during
    /// bindings generation
    if (!scope) {
        return false;
    }

    if (PyModule_CheckExact(scope))
    {
        if (!registerTypeInModuleScope(scope, type_name, type_obj))
        {
            return false;
        }
    }
    else
    {
        /// In Python 2 it is disallowed to register an inner classes
        /// via modifing dictionary of the built-in type.
        if (!registerTypeInClassScope(scope, type_name, type_obj))
        {
            return false;
        }
    }

    /// Expose all classes that are defined in the submodules as aliases in the
    /// root module for backward compatibility
    /// If submodule and root module are same than no aliases registration are
    /// required
    if (scope != root_module)
    {
        std::string type_name_str(type_name);

        std::string alias_name;
        alias_name.reserve(scope_name.size() + type_name_str.size());
        std::replace_copy(scope_name.begin() + 1, scope_name.end(), std::back_inserter(alias_name), '.', '_');
        alias_name += '_';
        alias_name += type_name_str;

        return registerTypeInModuleScope(root_module, alias_name.c_str(), type_obj);
    }
    return true;
}

#include "pyopencv_generated_modules_content.h"

static bool init_body(PyObject * m)
{
#define CVPY_MODULE(NAMESTR, NAME) \
    if (!init_submodule(m, MODULESTR NAMESTR, methods_##NAME, consts_##NAME)) \
    { \
        return false; \
    }
    #include "pyopencv_generated_modules.h"
#undef CVPY_MODULE

#ifdef CVPY_DYNAMIC_INIT
#define CVPY_TYPE(EXPORT_NAME, CLASS_ID, _1, _2, BASE, CONSTRUCTOR, SCOPE) CVPY_TYPE_INIT_DYNAMIC(EXPORT_NAME, CLASS_ID, return false, BASE, CONSTRUCTOR, SCOPE)
    PyObject * pyopencv_NoBase_TypePtr = NULL;
#else
#define CVPY_TYPE(EXPORT_NAME, CLASS_ID, _1, _2, BASE, CONSTRUCTOR, SCOPE) CVPY_TYPE_INIT_STATIC(EXPORT_NAME, CLASS_ID, return false, BASE, CONSTRUCTOR, SCOPE)
    PyTypeObject * pyopencv_NoBase_TypePtr = NULL;
#endif
    #include "pyopencv_generated_types.h"
#undef CVPY_TYPE

    PyObject* d = PyModule_GetDict(m);


    PyObject* version_obj = PyString_FromString(CV_VERSION);
    if (PyDict_SetItemString(d, "__version__", version_obj) < 0) {
        PyErr_SetString(PyExc_ImportError, "Can't update module version");
        Py_CLEAR(version_obj);
        return false;
    }
    Py_DECREF(version_obj);

    PyObject *opencv_error_dict = PyDict_New();
    PyDict_SetItemString(opencv_error_dict, "file", Py_None);
    PyDict_SetItemString(opencv_error_dict, "func", Py_None);
    PyDict_SetItemString(opencv_error_dict, "line", Py_None);
    PyDict_SetItemString(opencv_error_dict, "code", Py_None);
    PyDict_SetItemString(opencv_error_dict, "msg", Py_None);
    PyDict_SetItemString(opencv_error_dict, "err", Py_None);
    opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, opencv_error_dict);
    Py_DECREF(opencv_error_dict);
    PyDict_SetItemString(d, "error", opencv_error);


#define PUBLISH_(I, var_name, type_obj) \
    PyObject* type_obj = PyInt_FromLong(I); \
    if (PyDict_SetItemString(d, var_name, type_obj) < 0) \
    { \
        PyErr_SetString(PyExc_ImportError, "Can't register "  var_name " constant"); \
        Py_CLEAR(type_obj); \
        return false; \
    } \
    Py_DECREF(type_obj);

#define PUBLISH(I) PUBLISH_(I, #I, I ## _obj)

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
    PUBLISH(CV_16F);
    PUBLISH(CV_16FC1);
    PUBLISH(CV_16FC2);
    PUBLISH(CV_16FC3);
    PUBLISH(CV_16FC4);
#undef PUBLISH_
#undef PUBLISH

    return true;
}

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#if defined(CV_PYTHON_3)
// === Python 3

static struct PyModuleDef cv2_moduledef =
{
    PyModuleDef_HEAD_INIT,
    MODULESTR,
    "Python wrapper for OpenCV.",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    special_methods
};

PyMODINIT_FUNC PyInit_cv2();
PyObject* PyInit_cv2()
{
    import_array(); // from numpy
    PyObject* m = PyModule_Create(&cv2_moduledef);
    if (!init_body(m))
        return NULL;
    return m;
}

#else
// === Python 2
PyMODINIT_FUNC initcv2();
void initcv2()
{
    import_array(); // from numpy
    PyObject* m = Py_InitModule(MODULESTR, special_methods);
    init_body(m);
}

#endif
