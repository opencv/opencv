#ifndef OPENCV_CORE_PYOPENCV_CORE_HPP
#define OPENCV_CORE_PYOPENCV_CORE_HPP

#ifdef HAVE_OPENCV_CORE

#include "dlpack/dlpack.h"

static PyObject* pycvMakeType(PyObject* , PyObject* args, PyObject* kw) {
    const char *keywords[] = { "depth", "channels", NULL };

    int depth, channels;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "ii", (char**)keywords, &depth, &channels))
        return NULL;

    int type = CV_MAKETYPE(depth, channels);
    return PyInt_FromLong(type);
}

template <int depth>
static PyObject* pycvMakeTypeCh(PyObject*, PyObject *value) {
    int channels = (int)PyLong_AsLong(value);
    return PyInt_FromLong(CV_MAKETYPE(depth, channels));
}

#define CV_DLPACK_CAPSULE_NAME "dltensor"
#define CV_DLPACK_USED_CAPSULE_NAME "used_dltensor"

template<typename T>
bool fillDLPackTensor(const T& src, DLManagedTensor* tensor, const DLDevice& device);

template<typename T>
bool parseDLPackTensor(DLManagedTensor* tensor, T& obj, bool copy);

template<typename T>
int GetNumDims(const T& src);

// source: https://github.com/dmlc/dlpack/blob/7f393bbb86a0ddd71fde3e700fc2affa5cdce72d/docs/source/python_spec.rst#L110
static void dlpack_capsule_deleter(PyObject *self){
   if (PyCapsule_IsValid(self, CV_DLPACK_USED_CAPSULE_NAME)) {
      return;
   }

   DLManagedTensor *managed = (DLManagedTensor *)PyCapsule_GetPointer(self, CV_DLPACK_CAPSULE_NAME);
   if (managed == NULL) {
      PyErr_WriteUnraisable(self);
      return;
   }

   if (managed->deleter) {
      managed->deleter(managed);
   }
}

static void array_dlpack_deleter(DLManagedTensor *self)
{
   if (!Py_IsInitialized()) {
      return;
   }

   PyGILState_STATE state = PyGILState_Ensure();

   PyObject *array = (PyObject *)self->manager_ctx;
   PyMem_Free(self);
   Py_XDECREF(array);

   PyGILState_Release(state);
}

template<typename T>
static PyObject* to_dlpack(const T& src, PyObject* self, PyObject* py_args, PyObject* kw)
{
    int stream = 0;
    PyObject* maxVersion = nullptr;
    PyObject* dlDevice = nullptr;
    bool copy = false;
    const char* keywords[] = { "stream", "max_version", "dl_device", "copy", NULL };
    if (!PyArg_ParseTupleAndKeywords(py_args, kw, "|iOOp:__dlpack__", (char**)keywords, &stream, &maxVersion, &dlDevice, &copy))
        return nullptr;

    DLDevice device = {(DLDeviceType)-1, 0};
    if (dlDevice && dlDevice != Py_None && PyTuple_Check(dlDevice))
    {
        device.device_type = static_cast<DLDeviceType>(PyLong_AsLong(PyTuple_GetItem(dlDevice, 0)));
        device.device_id = PyLong_AsLong(PyTuple_GetItem(dlDevice, 1));
    }

    int ndim = GetNumDims(src);
    void* ptr = PyMem_Malloc(sizeof(DLManagedTensor) + sizeof(int64_t) * ndim * 2);
    if (!ptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    DLManagedTensor* tensor = reinterpret_cast<DLManagedTensor*>(ptr);
    tensor->manager_ctx = self;
    tensor->deleter = array_dlpack_deleter;
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.shape = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(ptr) + sizeof(DLManagedTensor));
    tensor->dl_tensor.strides = tensor->dl_tensor.shape + ndim;
    fillDLPackTensor(src, tensor, device);

    PyObject* capsule = PyCapsule_New(ptr, CV_DLPACK_CAPSULE_NAME, dlpack_capsule_deleter);
    if (!capsule) {
        PyMem_Free(ptr);
        return nullptr;
    }

    // the capsule holds a reference
    Py_INCREF(self);

    return capsule;
}

template<typename T>
static PyObject* from_dlpack(PyObject* py_args, PyObject* kw)
{
    PyObject* arr = nullptr;
    PyObject* device = nullptr;
    bool copy = false;
    const char* keywords[] = { "device", "copy", NULL };
    if (!PyArg_ParseTupleAndKeywords(py_args, kw, "O|Op:from_dlpack", (char**)keywords, &arr, &device, &copy))
        return nullptr;

    PyObject* capsule = nullptr;
    if (PyCapsule_CheckExact(arr))
    {
        capsule = arr;
    }
    else
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        capsule = PyObject_CallMethodObjArgs(arr, PyString_FromString("__dlpack__"), NULL);
        PyGILState_Release(gstate);
    }

    DLManagedTensor* tensor = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, CV_DLPACK_CAPSULE_NAME));
    if (tensor == nullptr)
    {
        if (capsule != arr)
            Py_DECREF(capsule);
        return nullptr;
    }

    T retval;
    bool success = parseDLPackTensor(tensor, retval, copy);
    if (success)
    {
        PyCapsule_SetName(capsule, CV_DLPACK_USED_CAPSULE_NAME);
    }
    if (capsule != arr)
        Py_DECREF(capsule);

    return success ? pyopencv_from(retval) : nullptr;
}

static DLDataType GetDLPackType(size_t elemSize1, int depth) {
    DLDataType dtype;
    dtype.bits = static_cast<uint8_t>(8 * elemSize1);
    dtype.lanes = 1;
    switch (depth)
    {
        case CV_8S: case CV_16S: case CV_32S: dtype.code = kDLInt; break;
        case CV_8U: case CV_16U: dtype.code = kDLUInt; break;
        case CV_16F: case CV_32F: case CV_64F: dtype.code = kDLFloat; break;
        default:
            CV_Error(Error::StsNotImplemented, "__dlpack__ data type");
    }
    return dtype;
}

static int DLPackTypeToCVType(const DLDataType& dtype, int channels) {
    if (dtype.code == kDLInt)
    {
        switch (dtype.bits)
        {
            case 8: return CV_8SC(channels);
            case 16: return CV_16SC(channels);
            case 32: return CV_32SC(channels);
            default:
            {
                PyErr_SetString(PyExc_BufferError,
                                format("Unsupported int dlpack depth: %d", dtype.bits).c_str());
                return -1;
            }
        }
    }
    if (dtype.code == kDLUInt)
    {
        switch (dtype.bits)
        {
            case 8: return CV_8UC(channels);
            case 16: return CV_16UC(channels);
            default:
            {
                PyErr_SetString(PyExc_BufferError,
                                format("Unsupported uint dlpack depth: %d", dtype.bits).c_str());
                return -1;
            }
        }
    }
    if (dtype.code == kDLFloat)
    {
        switch (dtype.bits)
        {
            case 16: return CV_16FC(channels);
            case 32: return CV_32FC(channels);
            case 64: return CV_64FC(channels);
            default:
            {
                PyErr_SetString(PyExc_BufferError,
                                format("Unsupported float dlpack depth: %d", dtype.bits).c_str());
                return -1;
            }
        }
    }
    PyErr_SetString(PyExc_BufferError, format("Unsupported dlpack data type: %d", dtype.code).c_str());
    return -1;
}

#define PYOPENCV_EXTRA_METHODS_CV \
  {"CV_MAKETYPE", CV_PY_FN_WITH_KW(pycvMakeType), "CV_MAKETYPE(depth, channels) -> retval"}, \
  {"CV_8UC", (PyCFunction)(pycvMakeTypeCh<CV_8U>), METH_O, "CV_8UC(channels) -> retval"}, \
  {"CV_8SC", (PyCFunction)(pycvMakeTypeCh<CV_8S>), METH_O, "CV_8SC(channels) -> retval"}, \
  {"CV_16UC", (PyCFunction)(pycvMakeTypeCh<CV_16U>), METH_O, "CV_16UC(channels) -> retval"}, \
  {"CV_16SC", (PyCFunction)(pycvMakeTypeCh<CV_16S>), METH_O, "CV_16SC(channels) -> retval"}, \
  {"CV_32SC", (PyCFunction)(pycvMakeTypeCh<CV_32S>), METH_O, "CV_32SC(channels) -> retval"}, \
  {"CV_32FC", (PyCFunction)(pycvMakeTypeCh<CV_32F>), METH_O, "CV_32FC(channels) -> retval"}, \
  {"CV_64FC", (PyCFunction)(pycvMakeTypeCh<CV_64F>), METH_O, "CV_64FC(channels) -> retval"}, \
  {"CV_16FC", (PyCFunction)(pycvMakeTypeCh<CV_16F>), METH_O, "CV_16FC(channels) -> retval"},

#endif  // HAVE_OPENCV_CORE
#endif  // OPENCV_CORE_PYOPENCV_CORE_HPP
