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
    CV_UNUSED(info);
    bool ok = true;
    PyObject* key = NULL;
    PyObject* item = NULL;
    Py_ssize_t pos = 0;

    if (!o || o == Py_None)
        return true;

    if(PyDict_Check(o)) {
        while(PyDict_Next(o, &pos, &key, &item))
        {
            // get key
            std::string k;
            if (!getUnicodeString(key, k))
            {
                ok = false;
                break;
            }
            // get value
            if( !!PyBool_Check(item) )
            {
                p.setBool(k, item == Py_True);
            }
            else if( PyInt_Check(item) )
            {
                int value = (int)PyInt_AsLong(item);
                if( strcmp(k.c_str(), "algorithm") == 0 )
                    p.setAlgorithm(value);
                else
                    p.setInt(k, value);
            }
            else if( PyFloat_Check(item) )
            {
                double value = PyFloat_AsDouble(item);
                p.setDouble(k, value);
            }
            else
            {
                std::string val_str;
                if (!getUnicodeString(item, val_str))
                {
                    ok = false;
                    break;
                }
                p.setString(k, val_str);
            }
        }
    }

    return ok && !PyErr_Occurred();
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
