#include <boost/python.hpp>

#include <string>

#include <opencv2/core/core.hpp>

namespace bp = boost::python;

namespace
{

template<typename T>
  void wrap_point(const std::string& name)
  {
    typedef cv::Point_<T> Point_t;
    bp::class_<Point_t> Point_(name.c_str());
    Point_.def(bp::init<>());
    Point_.def(bp::init<T, T>());
    Point_.def(bp::init<Point_t>());
    Point_.def_readwrite("x", &Point_t::x);
    Point_.def_readwrite("y", &Point_t::y);
    Point_.def_readwrite("dot", &Point_t::dot);
    Point_.def_readwrite("inside", &Point_t::inside);
  }

template<typename T>
  void wrap_rect(const std::string& name)
  {
    typedef cv::Rect_<T> Rect_t;
    bp::class_<Rect_t> c_(name.c_str());
    c_.def(bp::init<>());
    c_.def(bp::init<T, T, T, T>());
    c_.def(bp::init<cv::Point_<T>, cv::Point_<T> >());
    c_.def(bp::init<cv::Point_<T>, cv::Size_<T> >());

    c_.def(bp::init<Rect_t>());
    c_.def_readwrite("x", &Rect_t::x);
    c_.def_readwrite("y", &Rect_t::y);
    c_.def_readwrite("width", &Rect_t::width);
    c_.def_readwrite("height", &Rect_t::height);
    c_.def("tl", &Rect_t::tl);
    c_.def("br", &Rect_t::br);
    c_.def("size", &Rect_t::size);
    c_.def("area", &Rect_t::area);
    c_.def("contains", &Rect_t::contains);
  }
}

namespace opencv_wrappers
{
  void wrap_points()
  {
    bp::class_<cv::Size> Size_("Size");
    Size_.def(bp::init<int, int>());
    Size_.def_readwrite("width", &cv::Size::width);
    Size_.def_readwrite("height", &cv::Size::height);
    Size_.def("area", &cv::Size::area);

    wrap_point<int> ("Point");
    wrap_point<float> ("Point2f");
    wrap_point<double> ("Point2d");

    wrap_rect<int> ("Rect");
    wrap_rect<float> ("Rectf");
    wrap_rect<double> ("Rectd");

  }
}
