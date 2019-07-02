template<>
bool pyopencv_to(PyObject *obj, CvTermCriteria& dst, const char *name)
{
    CV_UNUSED(name);
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.max_iter, &dst.epsilon) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, CvSlice& r, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(PyObject_Size(obj) == 0)
    {
        r = CV_WHOLE_SEQ;
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start_index, &r.end_index) > 0;
}