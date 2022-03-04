#include "cv2_highgui.hpp"

#ifdef HAVE_OPENCV_HIGHGUI

#include "cv2_util.hpp"
#include "opencv2/highgui.hpp"
#include <map>

using namespace cv;

//======================================================================================================================

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

PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw)
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
    PyObject* py_callback_info = Py_BuildValue("OO", on_mouse, param);
    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(std::string(name), py_callback_info));
    }
    ERRWRAP2(setMouseCallback(name, OnMouse, py_callback_info));
    Py_RETURN_NONE;
}

//======================================================================================================================

static void OnChange(int pos, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("(i)", pos);
    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

// workaround for #20408, use nullptr, set value later
static int _createTrackbar(const String &trackbar_name, const String &window_name, int value, int count,
                    TrackbarCallback onChange, PyObject* py_callback_info)
{
    int n = createTrackbar(trackbar_name, window_name, NULL, count, onChange, py_callback_info);
    setTrackbarPos(trackbar_name, window_name, value);
    return n;
}

PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int value;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, &value, &count, &on_change))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "on_change must be callable");
        return NULL;
    }
    PyObject* py_callback_info = Py_BuildValue("OO", on_change, Py_None);
    std::string name = std::string(window_name) + ":" + std::string(trackbar_name);
    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(name, py_callback_info));
    }
    ERRWRAP2(_createTrackbar(trackbar_name, window_name, value, count, OnChange, py_callback_info));
    Py_RETURN_NONE;
}

//======================================================================================================================

static void OnButtonChange(int state, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args;
    if(PyTuple_GetItem(o, 1) != NULL)
    {
        args = Py_BuildValue("(iO)", state, PyTuple_GetItem(o,1));
    }
    else
    {
        args = Py_BuildValue("(i)", state);
    }

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

PyObject *pycvCreateButton(PyObject*, PyObject *args, PyObject *kw)
{
    const char* keywords[] = {"buttonName", "onChange", "userData", "buttonType", "initialButtonState", NULL};
    PyObject *on_change;
    PyObject *userdata = NULL;
    char* button_name;
    int button_type = 0;
    int initial_button_state = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|Oii", (char**)keywords, &button_name, &on_change, &userdata, &button_type, &initial_button_state))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "onChange must be callable");
        return NULL;
    }
    if (userdata == NULL) {
        userdata = Py_None;
    }

    PyObject* py_callback_info = Py_BuildValue("OO", on_change, userdata);
    std::string name(button_name);

    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(name, py_callback_info));
    }
    ERRWRAP2(createButton(button_name, OnButtonChange, py_callback_info, button_type, initial_button_state != 0));
    Py_RETURN_NONE;
}

#endif // HAVE_OPENCV_HIGHGUI
