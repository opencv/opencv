#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/cuda.hpp"
#include "dlpack/dlpack.h"

typedef std::vector<cuda::GpuMat> vector_GpuMat;
typedef cuda::GpuMat::Allocator GpuMat_Allocator;
typedef cuda::HostMem::AllocType HostMem_AllocType;
typedef cuda::Event::CreateFlags Event_CreateFlags;

template<> struct pyopencvVecConverter<cuda::GpuMat>
{
    static bool to(PyObject* obj, std::vector<cuda::GpuMat>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cuda::GpuMat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

CV_PY_TO_CLASS(cuda::GpuMat)
CV_PY_TO_CLASS(cuda::Stream)
CV_PY_TO_CLASS(cuda::Event)
CV_PY_TO_CLASS(cuda::HostMem)

CV_PY_TO_CLASS_PTR(cuda::GpuMat)
CV_PY_TO_CLASS_PTR(cuda::GpuMat::Allocator)

CV_PY_FROM_CLASS(cuda::GpuMat)
CV_PY_FROM_CLASS(cuda::Stream)
CV_PY_FROM_CLASS(cuda::HostMem)

CV_PY_FROM_CLASS_PTR(cuda::GpuMat::Allocator)

static DLDataType GetDLDataType(Ptr<cv::cuda::GpuMat> m) {
    DLDataType dtype;
    dtype.bits = 8 * m->elemSize1();
    dtype.lanes = 1;
    switch (m->depth())
    {
        case CV_8S: case CV_16S: case CV_32S: /*case CV_64S:*/ dtype.code = kDLInt; break;
        case CV_8U: case CV_16U: /*case CV_32U: case CV_64U:*/ dtype.code = kDLUInt; break;
        case CV_16F: case CV_32F: case CV_64F: dtype.code = kDLFloat; break;
        default:
            CV_Error(Error::StsNotImplemented, "__dlpack__ data type");
        // TODO: bool
    }
    return dtype;
}

static PyObject* pyGpuMatDLPack(PyObject* self, PyObject* py_args, PyObject* kw) {
    Ptr<cv::cuda::GpuMat> * self1 = 0;
    if (!pyopencv_cuda_GpuMat_getp(self, self1))
        return failmsgp("Incorrect type of self (must be 'cuda_GpuMat' or its derivative)");
    Ptr<cv::cuda::GpuMat> _self_ = *(self1);

    int stream = 0;
    PyObject* maxVersion = nullptr;
    PyObject* dlDevice = nullptr;
    bool copy = false;
    const char* keywords[] = { "stream", "max_version", "dl_device", "copy", NULL };
    if (!PyArg_ParseTupleAndKeywords(py_args, kw, "|iOOp:cuda_GpuMat.__dlpack__", (char**)keywords, &stream, &maxVersion, &dlDevice, &copy))
        return nullptr;

    if (dlDevice && dlDevice != Py_None && PyTuple_Check(dlDevice))
    {
        // TODO: check for device type
    }

    void* ptr = PyMem_Malloc(sizeof(DLManagedTensor));
    if (!ptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    DLManagedTensor* tensor = reinterpret_cast<DLManagedTensor*>(ptr);
    tensor->manager_ctx = 0;
    tensor->deleter = 0;
    tensor->dl_tensor.data = _self_->cudaPtr();
    tensor->dl_tensor.device.device_type = kDLCUDA;
    tensor->dl_tensor.device.device_id = 0;  // TODO: which id?
    tensor->dl_tensor.ndim = 3;
    tensor->dl_tensor.dtype = GetDLDataType(_self_);
    tensor->dl_tensor.shape = new int64_t[3];
    tensor->dl_tensor.shape[0] = _self_->rows;
    tensor->dl_tensor.shape[1] = _self_->cols;
    tensor->dl_tensor.shape[2] = _self_->channels();
    tensor->dl_tensor.strides = new int64_t[3];
    tensor->dl_tensor.strides[0] = _self_->step1();
    tensor->dl_tensor.strides[1] = _self_->channels();
    tensor->dl_tensor.strides[2] = 1;
    tensor->dl_tensor.byte_offset = 0;

    PyObject* capsule = PyCapsule_New(ptr, "dltensor", nullptr);
    if (!capsule) {
        PyMem_Free(ptr);
        return nullptr;
    }

    // the capsule holds a reference
    Py_INCREF(self);

    return capsule;
}

static PyObject* pyGpuMatDLPackDevice() {
    return pyopencv_from(std::tuple<int, int>(kDLCUDA, 0));
}

#define PYOPENCV_EXTRA_METHODS_cuda_GpuMat \
  {"__dlpack__", CV_PY_FN_WITH_KW(pyGpuMatDLPack), ""}, \
  {"__dlpack_device__", CV_PY_FN_WITH_KW(pyGpuMatDLPackDevice), ""}, \

#endif
