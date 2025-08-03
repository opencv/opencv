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

template<>
void fillDLPackTensor(const Ptr<cv::cuda::GpuMat>& src, DLManagedTensor* tensor)
{
    tensor->dl_tensor.data = src->cudaPtr();
    tensor->dl_tensor.device.device_type = kDLCUDA;
    tensor->dl_tensor.device.device_id = 0;  // TODO: which id?
    tensor->dl_tensor.dtype = GetDLPackType(src->elemSize1(), src->depth());
    tensor->dl_tensor.shape[0] = src->rows;
    tensor->dl_tensor.shape[1] = src->cols;
    tensor->dl_tensor.shape[2] = src->channels();
    tensor->dl_tensor.strides[0] = src->step1();
    tensor->dl_tensor.strides[1] = src->channels();
    tensor->dl_tensor.strides[2] = 1;
    tensor->dl_tensor.byte_offset = 0;
}

template<>
void fillDLPackTensor(const Ptr<cv::cuda::GpuMatND>& src, DLManagedTensor* tensor)
{
    tensor->dl_tensor.data = src->getDevicePtr();
    tensor->dl_tensor.device.device_type = kDLCUDA;
    tensor->dl_tensor.device.device_id = 0;  // TODO: which id?
    tensor->dl_tensor.dtype = GetDLPackType(src->elemSize1(), CV_MAT_DEPTH(src->flags));
    for (int i = 0; i < src->dims; ++i)
        tensor->dl_tensor.shape[i] = src->size[i];
    for (int i = 0; i < src->dims; ++i)
        tensor->dl_tensor.strides[i] = src->step[i];
    tensor->dl_tensor.byte_offset = 0;
}

template<>
bool parseDLPackTensor(DLManagedTensor* tensor, cv::cuda::GpuMat& obj)
{
    if (tensor->dl_tensor.byte_offset != 0)
    {
        PyErr_SetString(PyExc_BufferError, "Unimplemented from_dlpack for GpuMat with memory offset");
        return false;
    }
    if (tensor->dl_tensor.ndim != 3)
    {
        PyErr_SetString(PyExc_BufferError, "cuda_GpuMat.from_dlpack expects a 3D tensor. Use cuda_GpuMatND.from_dlpack instead");
        return false;
    }
    if (tensor->dl_tensor.device.device_type != kDLCUDA)
    {
        PyErr_SetString(PyExc_BufferError, "cuda_GpuMat.from_dlpack expects a tensor on CUDA device");
        return false;
    }
    if (tensor->dl_tensor.strides[1] != tensor->dl_tensor.shape[2] ||
        tensor->dl_tensor.strides[2] != 1)
    {
        PyErr_SetString(PyExc_BufferError, "Unexpected strides for image. Try use GpuMatND");
        return false;
    }
    int type = DLPackTypeToCVType(tensor->dl_tensor.dtype, tensor->dl_tensor.shape[2]);
    if (type == -1)
        return false;

    obj = cv::cuda::GpuMat(
        tensor->dl_tensor.shape[0],
        tensor->dl_tensor.shape[1],
        type,
        tensor->dl_tensor.data,
        tensor->dl_tensor.strides[0] * tensor->dl_tensor.dtype.bits / 8
    );
    return true;
}

template<>
bool parseDLPackTensor(DLManagedTensor* tensor, Ptr<cv::cuda::GpuMatND>& obj)
{
    if (tensor->dl_tensor.byte_offset != 0)
    {
        PyErr_SetString(PyExc_BufferError, "Unimplemented from_dlpack for GpuMat with memory offset");
        return false;
    }
    if (tensor->dl_tensor.device.device_type != kDLCUDA)
    {
        PyErr_SetString(PyExc_BufferError, "cuda_GpuMat.from_dlpack expects a tensor on CUDA device");
        return false;
    }
    int type = DLPackTypeToCVType(tensor->dl_tensor.dtype, tensor->dl_tensor.shape[2]);
    if (type == -1)
        return false;

    std::vector<size_t> steps(tensor->dl_tensor.ndim - 1);
    for (int i = 0; i < tensor->dl_tensor.ndim - 1; ++i)
    {
        steps[i] = tensor->dl_tensor.strides[i] * tensor->dl_tensor.dtype.bits / 8;
    }
    obj.reset(new cv::cuda::GpuMatND(
        std::vector<int>(&tensor->dl_tensor.shape[0], tensor->dl_tensor.shape + tensor->dl_tensor.ndim),
        type, tensor->dl_tensor.data, steps
    ));
    return true;
}

template<>
int GetNumDims(const Ptr<cv::cuda::GpuMat>& src) { return 3; }

template<>
int GetNumDims(const Ptr<cv::cuda::GpuMatND>& src) { return src->dims; }

static PyObject* pyDLPackGpuMat(PyObject* self, PyObject* py_args, PyObject* kw) {
    Ptr<cv::cuda::GpuMat> * self1 = 0;
    if (!pyopencv_cuda_GpuMat_getp(self, self1))
        return failmsgp("Incorrect type of self (must be 'cuda_GpuMat' or its derivative)");
    return to_dlpack(*(self1), self, py_args, kw);
}

static PyObject* pyDLPackGpuMatND(PyObject* self, PyObject* py_args, PyObject* kw) {
    Ptr<cv::cuda::GpuMatND> * self1 = 0;
    if (!pyopencv_cuda_GpuMatND_getp(self, self1))
        return failmsgp("Incorrect type of self (must be 'cuda_GpuMatND' or its derivative)");
    return to_dlpack(*(self1), self, py_args, kw);
}

static PyObject* pyDLPackDeviceCUDA(PyObject* , PyObject*, PyObject*) {
    return pyopencv_from(std::tuple<int, int>(kDLCUDA, 0));
}

static PyObject* pyGpuMatFromDLPack(PyObject*, PyObject* py_args, PyObject* kw) {
    return from_dlpack<cv::cuda::GpuMat>(py_args, kw);
}

static PyObject* pyGpuMatNDFromDLPack(PyObject*, PyObject* py_args, PyObject* kw) {
    return from_dlpack<Ptr<cv::cuda::GpuMatND> >(py_args, kw);
}

#define PYOPENCV_EXTRA_METHODS_cuda_GpuMat \
  {"__dlpack__", CV_PY_FN_WITH_KW(pyDLPackGpuMat), ""}, \
  {"__dlpack_device__", CV_PY_FN_WITH_KW(pyDLPackDeviceCUDA), ""}, \
  {"from_dlpack", CV_PY_FN_WITH_KW_(pyGpuMatFromDLPack, METH_STATIC), ""}, \

#define PYOPENCV_EXTRA_METHODS_cuda_GpuMatND \
  {"__dlpack__", CV_PY_FN_WITH_KW(pyDLPackGpuMatND), ""}, \
  {"__dlpack_device__", CV_PY_FN_WITH_KW(pyDLPackDeviceCUDA), ""}, \
  {"from_dlpack", CV_PY_FN_WITH_KW_(pyGpuMatNDFromDLPack, METH_STATIC), ""}, \

#endif
