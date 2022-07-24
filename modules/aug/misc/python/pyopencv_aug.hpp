#ifdef HAVE_OPENCV_AUG
typedef std::vector<Ptr<Transform> > vector_Ptr_Transform;

//template<>
//bool pyopencv_to(PyObject *o, std::vector<Ptr<cv::Transform> > &value, const ArgInfo& info){
//    return pyopencv_to_generic_vec(o, value, info);
//}
template<> struct pyopencvVecConverter<Ptr<Transform> >
{
    static bool to(PyObject* obj, std::vector<Ptr<Transform> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

};

#endif