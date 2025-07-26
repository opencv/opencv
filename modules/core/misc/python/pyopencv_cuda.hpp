#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/cuda.hpp"

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


static PyObject* pyGpuMatDLPack(PyObject* self, PyObject* py_args, PyObject* kw) {
    printf("__dlpack__\n");
    int stream = 0;
    PyObject* maxVersion = nullptr;
    PyObject* dlDevice = nullptr;
    bool copy = false;
    const char* keywords[] = { "stream", "max_version", "dl_device", "copy", NULL };
    if (!PyArg_ParseTupleAndKeywords(py_args, kw, "|iOOp:cuda_GpuMat.__dlpack__", (char**)keywords, &stream, &maxVersion, &dlDevice, &copy))
        return nullptr;

    std::cout << stream << std::endl;
    if (maxVersion != Py_None)
        std::cout << "maxVersion " << maxVersion << std::endl;
    if (dlDevice != Py_None)
        std::cout << "dlDevice " << dlDevice << std::endl;
    std::cout << copy << std::endl;
    return nullptr;
}

static PyObject* pyGpuMatDLPackDevice() {
    printf("__dlpack_device__\n");
    return nullptr;
}

#define PYOPENCV_EXTRA_METHODS_cuda_GpuMat \
  {"__dlpack__", CV_PY_FN_WITH_KW(pyGpuMatDLPack), ""}, \
  {"__dlpack_device__", CV_PY_FN_WITH_KW(pyGpuMatDLPackDevice), ""}, \

#endif
