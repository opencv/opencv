using gapi_GKernelPackage = cv::gapi::GKernelPackage;
using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

template<>
bool pyopencv_to(PyObject* obj, std::vector<GCompileArg>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template<>
PyObject* pyopencv_from(const std::vector<GCompileArg>& value)
{
    return pyopencv_from_generic_vec(value);
}

template<>
bool pyopencv_to(PyObject* obj, GRunArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

//template<>
//PyObject* pyopencv_from(const GRunArgs& value)
//{
    //return pyopencv_from_generic_vec(value);
//}

PyObject* from_grunarg(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<cv::Mat>():
        {
            const auto& m = util::get<cv::Mat>(v);
            return pyopencv_from(m);
        }

        default:
            GAPI_Assert(false);
    }
    return NULL;
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        PyObject* item = from_grunarg(value[i]);
        if(!item)
            break;
        PyList_SetItem(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}

//static cv::GProtoArgs parse_gin_gout_args(PyObject* py_args, PyObject* kw)
//{
    //using namespace cv;

    //GProtoArgs args;
    //GMat* gmat;
    //GScalar* gscalar;

    //Py_ssize_t size = PyTuple_Size(py_args);
    //std::cout << "SIZE = " << size << std::endl;
    //// (void*)pyopencv_GScalar_getp;
    //for (int i = 0; i < size; ++i) {
        //PyObject* item = PyTuple_GetItem(py_args, i);
        //checkPtr(GScalar, item, gscalar);
        ////std::cout << "HERE" << std::endl;
        ////if (PyObject_TypeCheck(self, (PyTypeObject*)pyopencv_GScalar_TypePtr)) {
            ////std::cout << "GSCALAR " << std::endl;
            ////args.emplace_back(*gscalar);
        ////} else {
            ////std::cout << "ERROR" << std::enld;
        ////}
        ////if (pyopencv_to(item, gmat, ArgInfo("", i))) {
            ////std::cout << "GMAt " << std::endl;
            ////args.emplace_back(gmat);
        ////} else if (pyopencv_to(item, gscalar, ArgInfo("", i))) {
            ////std::cout << "GSCALAR " << std::endl;
            ////// args.emplace_back(gscalar);
        ////} else {
            ////// PyErr_SetString(PyExc_TypeError, "cv.GIn() supports only cv.GMat and cv.GScalar");
            ////throw std::runtime_error("cv.GIn() supports only cv.GMat and cv.GScalar");
            ////// throw "error";
            ////// return NULL;
        ////}
    //}
    //std::cout <<" end " << std::endl;

    //return args;
//}

//static PyObject* pyopencv_cv_GIn(PyObject* , PyObject* py_args, PyObject* kw)
//{
    //auto args = parse_gin_gout_args(py_args, kw);
    //GProtoInputArgs in_args{std::move(args)};
    //return pyopencv_from<GProtoInputArgs>(in_args);
//}

//static PyObject* pyopencv_cv_GOut(PyObject* , PyObject* py_args, PyObject* kw)
//{
    //auto args = parse_gin_gout_args(py_args, kw);
    //GProtoOutputArgs out_args{std::move(args)};
    //return pyopencv_from<GProtoOutputArgs>(out_args);
//}
