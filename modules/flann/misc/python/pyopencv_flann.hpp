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
bool pyopencv_to(PyObject *o, cv::flann::IndexParams& p, const char *name)
{
    (void)name;
    bool ok = true;
    PyObject* key = NULL;
    PyObject* item = NULL;
    Py_ssize_t pos = 0;

    if(PyDict_Check(o)) {
        while(PyDict_Next(o, &pos, &key, &item)) {
            if( !PyString_Check(key) ) {
                ok = false;
                break;
            }

            String k = PyString_AsString(key);
            if( PyString_Check(item) )
            {
                const char* value = PyString_AsString(item);
                p.setString(k, value);
            }
            else if( !!PyBool_Check(item) )
                p.setBool(k, item == Py_True);
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
                ok = false;
                break;
            }
        }
    }

    return ok && !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, cv::flann::SearchParams & value, const char * name)
{
    return pyopencv_to<cv::flann::IndexParams>(obj, value, name);
}

template<>
bool pyopencv_to(PyObject *o, cvflann::flann_distance_t& dist, const char *name)
{
    int d = (int)dist;
    bool ok = pyopencv_to(o, d, name);
    dist = (cvflann::flann_distance_t)d;
    return ok;
}
#endif