#ifndef __OPENCV_PYTHON_HPP__
#define __OPENCV_PYTHON_HPP__

#include <Python.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/contrib.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/softcascade.hpp"
#include "opencv2/video.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE
#  include "opencv2/nonfree.hpp"
#endif

typedef std::vector<uchar> vector_uchar;
typedef std::vector<char> vector_char;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<double> vector_double;
typedef std::vector<cv::Point> vector_Point;
typedef std::vector<cv::Point2f> vector_Point2f;
typedef std::vector<cv::Vec2f> vector_Vec2f;
typedef std::vector<cv::Vec3f> vector_Vec3f;
typedef std::vector<cv::Vec4f> vector_Vec4f;
typedef std::vector<cv::Vec6f> vector_Vec6f;
typedef std::vector<cv::Vec4i> vector_Vec4i;
typedef std::vector<cv::Rect> vector_Rect;
typedef std::vector<cv::Mat> vector_Mat;
typedef std::vector<std::string> vector_string;
typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<cv::Point> > vector_vector_Point;
typedef std::vector<std::vector<cv::Point2f> > vector_vector_Point2f;
typedef std::vector<std::vector<cv::Point3f> > vector_vector_Point3f;
typedef std::vector<cv::Scalar> vector_Scalar;
typedef std::vector<cv::String> vector_String;

static PyObject* opencv_error = 0;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary
    ArgInfo(const char * name_, bool outputarg_);

    // to match with older pyopencv_to function signature
    // operator const char *() const;
};

class PyAllowThreads
{
public:
    PyAllowThreads();
    ~PyAllowThreads();
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL();
    ~PyEnsureGIL();
private:
    PyGILState_STATE _state;
};

class NumpyAllocator : public cv::MatAllocator
{
    const MatAllocator* stdAllocator;
public:
    NumpyAllocator();
    ~NumpyAllocator();

    cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const;
    cv::UMatData* allocate(int dims0, const int* sizes, int type, size_t* step) const;
    bool allocate(cv::UMatData* u, int accessFlags) const;
    void deallocate(cv::UMatData* u) const;
    void map(cv::UMatData*, int) const;
    void unmap(cv::UMatData* u) const;
    void download(cv::UMatData* u, void* dstptr,
              int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dststep[]) const;
    void upload(cv::UMatData* u, const void* srcptr, int dims, const size_t sz[],
            const size_t dstofs[], const size_t dststep[],
            const size_t srcstep[]) const;
    void copy(cv::UMatData* usrc, cv::UMatData* udst, int dims, const size_t sz[],
          const size_t srcofs[], const size_t srcstep[],
          const size_t dstofs[], const size_t dststep[], bool sync) const;
};

static inline PyObject* pyObjectFromRefcount(const int* refcount);
static inline int* refcountFromPyObject(const PyObject* obj);


// definition of generic transformation functions
template<typename T>
bool pyopencv_to(PyObject* obj, T& p, const ArgInfo info=ArgInfo("<unknown>", 0));

template<typename T> static
PyObject* pyopencv_from(const T& src);

template<> PyObject* pyopencv_from(const cv::Mat& m);
template<> bool pyopencv_to(PyObject* o, cv::Mat& m, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Scalar& src);
bool pyopencv_coerce(PyObject* obj, cv::Scalar& value, const ArgInfo info);
template<> bool pyopencv_to(PyObject *o, cv::Scalar& s, const ArgInfo info);

template<> PyObject* pyopencv_from(const bool& value);
bool pyopencv_coerce(PyObject* obj, bool& value);
template<> bool pyopencv_to(PyObject* obj, bool& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const size_t& value);
bool pyopencv_coerce(PyObject* obj, size_t& value);
template<> bool pyopencv_to(PyObject* obj, size_t& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const int& value);
bool pyopencv_coerce(PyObject* obj, int& value);
template<> bool pyopencv_to(PyObject* obj, int& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const int64& value);
template<> bool pyopencv_to(PyObject* obj, int64& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const uchar& value);
bool pyopencv_coerce(PyObject* obj, uchar& value);
template<> bool pyopencv_to(PyObject* obj, uchar& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const double& value);
bool pyopencv_coerce(PyObject* obj, double& value);
template<> bool pyopencv_to(PyObject* obj, double& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const float& value);
bool pyopencv_coerce(PyObject* obj, float& value);
template<> bool pyopencv_to(PyObject* obj, float& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const std::string& value);
bool pyopencv_coerce(PyObject* obj, std::string& value);
template<> bool pyopencv_to(PyObject* obj, std::string& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::String& value);
bool pyopencv_coerce(PyObject* obj, cv::String& value);
template<> bool pyopencv_to(PyObject* obj, cv::String& value, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Rect& r);
template<> bool pyopencv_to(PyObject* obj, cv::Rect& r, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Point& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point& p, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Point2f& p);
bool pyopencv_coerce(PyObject* obj, cv::Point2f& value);
template<> bool pyopencv_to(PyObject* obj, cv::Point2f& p, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Vec2d& v);

template<> PyObject* pyopencv_from(const cv::Vec3d& v);
template<> bool pyopencv_to(PyObject* obj, cv::Vec3d& v, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Point2d& p);
template<> bool pyopencv_to(PyObject* obj, cv::Point2d& v, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::RotatedRect& src);
template<> bool pyopencv_to(PyObject *obj, cv::RotatedRect& dst, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Size& sz);
bool pyopencv_coerce(PyObject* obj, cv::Size& value);
template<> bool pyopencv_to(PyObject* obj, cv::Size& sz, const ArgInfo info);

template<> PyObject* pyopencv_from(const cv::Moments& m);

#endif
