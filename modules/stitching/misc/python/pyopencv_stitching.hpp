#ifdef HAVE_OPENCV_STITCHING

typedef Stitcher::Status Status;
typedef Stitcher::Mode Mode;

typedef std::vector<detail::ImageFeatures> vector_ImageFeatures;
typedef std::vector<detail::MatchesInfo> vector_MatchesInfo;
typedef std::vector<detail::CameraParams> vector_CameraParams;

template<> struct pyopencvVecConverter<detail::ImageFeatures>
{
    static bool to(PyObject* obj, std::vector<detail::ImageFeatures>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<detail::ImageFeatures>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<detail::MatchesInfo>
{
    static bool to(PyObject* obj, std::vector<detail::MatchesInfo>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<detail::MatchesInfo>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<detail::CameraParams>
{
    static bool to(PyObject* obj, std::vector<detail::CameraParams>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<detail::CameraParams>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

#endif
