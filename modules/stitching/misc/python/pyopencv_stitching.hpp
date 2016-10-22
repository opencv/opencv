#ifdef HAVE_OPENCV_STITCHING
typedef Stitcher::Status Status;

template<>
PyObject* pyopencv_from(const Status& value)
{
    return PyInt_FromLong(value);
}
#endif