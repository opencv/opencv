#ifdef HAVE_OPENCV_FLANN
typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;

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
bool pyopencv_to(PyObject *o, cv::flann::IndexParams& p, const ArgInfo& info)
{
    if (!o || o == Py_None)
    {
        return true;
    }

    if(!PyDict_Check(o))
    {
        failmsg("Argument '%s' is not a dictionary", info.name);
        return false;
    }

    PyObject* key_obj = NULL;
    PyObject* value_obj = NULL;
    Py_ssize_t key_pos = 0;

    while(PyDict_Next(o, &key_pos, &key_obj, &value_obj))
    {
        // get key
        std::string key;
        if (!getUnicodeString(key_obj, key))
        {
            failmsg("Key at pos %lld is not a string", static_cast<int64_t>(key_pos));
            return false;
        }
        // key_arg_info.name is bound to key lifetime
        const ArgInfo key_arg_info(key.c_str(), false);

        // get value
        if (isBool(value_obj))
        {
            npy_bool npy_value = NPY_FALSE;
            if (PyArray_BoolConverter(value_obj, &npy_value) >= 0)
            {
                p.setBool(key, npy_value == NPY_TRUE);
                continue;
            }
            PyErr_Clear();
        }

        int int_value = 0;
        if (pyopencv_to(value_obj, int_value, key_arg_info))
        {
            if (key == "algorithm")
            {
                p.setAlgorithm(int_value);
            }
            else
            {
                p.setInt(key, int_value);
            }
            continue;
        }
        PyErr_Clear();

        double flt_value = 0.0;
        if (pyopencv_to(value_obj, flt_value, key_arg_info))
        {
            if (key == "eps")
            {
                p.setFloat(key, static_cast<float>(flt_value));
            }
            else
            {
                p.setDouble(key, flt_value);
            }
            continue;
        }
        PyErr_Clear();

        std::string str_value;
        if (getUnicodeString(value_obj, str_value))
        {
            p.setString(key, str_value);
            continue;
        }
        PyErr_Clear();
        // All conversions are failed
        failmsg("Failed to parse IndexParam with key '%s'. "
                "Supported types: [bool, int, float, str]", key.c_str());
        return false;

    }
    return true;
}

template<>
bool pyopencv_to(PyObject* obj, cv::flann::SearchParams & value, const ArgInfo& info)
{
    return pyopencv_to<cv::flann::IndexParams>(obj, value, info);
}

template<>
bool pyopencv_to(PyObject *o, cvflann::flann_distance_t& dist, const ArgInfo& info)
{
    int d = (int)dist;
    bool ok = pyopencv_to(o, d, info);
    dist = (cvflann::flann_distance_t)d;
    return ok;
}
#endif
