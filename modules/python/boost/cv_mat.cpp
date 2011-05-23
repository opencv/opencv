#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>

#include <opencv2/core/core.hpp>

namespace bp = boost::python;

namespace
{

  template<typename T>
  inline   void mat_set_t(cv::Mat&m, bp::object o)
    {

      int length = bp::len(o);
      if (m.size().area() != length || m.depth() != cv::DataType<T>::depth)
      {
        m.create(length, 1, cv::DataType<T>::type);
      }
      bp::stl_input_iterator<T> begin(o), end;
      typename cv::Mat_<T>::iterator it = m.begin<T> (), itEnd = m.end<T> ();
      for (; it != itEnd; ++it)
        *it = *(begin++);
    }

  inline void mat_set(cv::Mat& m, bp::object o, int type)
  {
    //switch on the given type and use this type as the cv::Mat element type
    switch (CV_MAT_DEPTH(type))
    {
      case CV_8U:
        mat_set_t<unsigned char> (m, o);
        break;
      case CV_8S:
        mat_set_t<signed char> (m, o);
        break;
      case CV_16U:
        mat_set_t<uint16_t> (m, o);
        break;
      case CV_16S:
        mat_set_t<int16_t> (m, o);
        break;
      case CV_32S:
        mat_set_t<int32_t> (m, o);
        break;
      case CV_32F:
        mat_set_t<float_t> (m, o);
        break;
      case CV_64F:
        mat_set_t<double_t> (m, o);
        break;
      default:
        throw std::logic_error("Given type not supported.");
    }
  }
  inline cv::Size mat_size(cv::Mat& m)
  {
    return m.size();
  }

  inline int mat_type(cv::Mat& m)
  {
    return m.type();
  }
  inline void mat_set(cv::Mat& m, bp::object o)
  {
    if (m.empty())
      throw std::logic_error("The matrix is empty, can not deduce type.");
    //use the m.type and implicitly assume that o is of this type
    mat_set(m, o, m.type());
  }

  inline cv::Mat mat_mat_star(cv::Mat& m, cv::Mat& m2)
  {
    return m * m2;
  }

  inline cv::Mat mat_scalar_star(cv::Mat& m, double s)
  {
    return m * s;
  }

  inline  cv::Mat mat_scalar_plus(cv::Mat& m, double s)
  {
    return m + cv::Scalar::all(s);
  }

  inline cv::Mat mat_scalar_plus2(cv::Mat& m, cv::Scalar s)
  {
    return m + s;
  }


  inline cv::Mat mat_scalar_sub(cv::Mat& m, double s)
  {
    return m - cv::Scalar::all(s);
  }

  inline cv::Mat mat_scalar_sub2(cv::Mat& m, cv::Scalar s)
  {
    return m - s;
  }

  inline cv::Mat mat_scalar_div(cv::Mat& m, double s)
  {
    return m / s;
  }

  inline cv::Mat mat_mat_plus(cv::Mat& m, cv::Mat& m2)
  {
    return m + m2;
  }

  inline cv::Mat mat_mat_sub(cv::Mat& m, cv::Mat& m2)
  {
    return m - m2;
  }
  inline cv::Mat mat_mat_div(cv::Mat& m, cv::Mat& m2)
  {
    return m/m2;
  }

  inline cv::Mat roi(cv::Mat& m, cv::Rect region)
  {
    return m(region);
  }

  //overloaded function pointers
  void (*mat_set_p2)(cv::Mat&, bp::object) = mat_set;
  void (*mat_set_p3)(cv::Mat&, bp::object, int) = mat_set;

}

namespace opencv_wrappers
{
  void wrap_mat()
  {
    typedef std::vector<uchar> buffer_t;
    bp::class_<std::vector<uchar> > ("buffer")
        .def(bp::vector_indexing_suite<std::vector<uchar>, false>() );

    bp::class_<cv::InputArray>("InputArray");
    bp::class_<cv::OutputArray>("OuputArray");
    bp::implicitly_convertible<cv::Mat,cv::InputArray>();
    bp::implicitly_convertible<cv::Mat,cv::OutputArray>();

    //mat definition
    bp::class_<cv::Mat> Mat_("Mat");
    Mat_.def(bp::init<>());
    Mat_.def(bp::init<int, int, int>());
    Mat_.def(bp::init<cv::Size, int>());
    Mat_.def(bp::init<buffer_t>());
    Mat_.def_readonly("rows", &cv::Mat::rows, "the number of rows");
    Mat_.def_readonly("cols", &cv::Mat::cols, "the number of columns");
    Mat_.def("row", &cv::Mat::row, "get the row at index");
    Mat_.def("col", &cv::Mat::col, "get the column at index");
    Mat_.def("fromarray", mat_set_p2, "Set a Matrix from a python iterable. Assumes the type of the Mat "
      "while setting. If the size of the Matrix will not accommodate "
      "the given python iterable length, then the matrix will be allocated "
      "as a single channel, Nx1 vector where N = len(list)");
    Mat_.def("fromarray", mat_set_p3, "Set a Matrix from a python array. Explicitly give "
      "the type of the array. If the size of the Matrix will not accommodate "
      "the given python iterable length, then the matrix will be allocated "
      "as a single channel, Nx1 vector where N = len(list)");
    Mat_.def("size", mat_size);
    Mat_.def("type", mat_type);
    Mat_.def("convertTo",&cv::Mat::convertTo);
    Mat_.def("clone", &cv::Mat::clone);
    Mat_.def("t",&cv::Mat::t);
    Mat_.def("roi",roi);
    Mat_.def("__mul__", mat_mat_star);
    Mat_.def("__mul__", mat_scalar_star);
    Mat_.def("__add__",mat_mat_plus);
    Mat_.def("__add__",mat_scalar_plus);
    Mat_.def("__add__",mat_scalar_plus2);
    Mat_.def("__sub__",mat_mat_sub);
    Mat_.def("__sub__",mat_scalar_sub);
    Mat_.def("__sub__",mat_scalar_sub2);
    Mat_.def("__div__",mat_mat_div);
    Mat_.def("__div__",mat_scalar_div);



  }
}
