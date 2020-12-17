#ifdef HAVE_OPENCV_RGBD
#include "opencv2/core/saturate.hpp"

template<> struct pyopencvVecConverter<linemod::Match>
{
    static bool to(PyObject* obj, std::vector<linemod::Match>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<linemod::Match>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<linemod::Template>
{
    static bool to(PyObject* obj, std::vector<linemod::Template>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<linemod::Template>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<linemod::Feature>
{
    static bool to(PyObject* obj, std::vector<linemod::Feature>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<linemod::Feature>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<Ptr<linemod::Modality> >
{
    static bool to(PyObject* obj, std::vector<Ptr<linemod::Modality> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<Ptr<linemod::Modality> >& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

typedef std::vector<linemod::Match> vector_Match;
typedef std::vector<linemod::Template> vector_Template;
typedef std::vector<linemod::Feature> vector_Feature;
typedef std::vector<Ptr<linemod::Modality> > vector_Ptr_Modality;
#endif
