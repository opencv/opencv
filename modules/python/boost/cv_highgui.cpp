#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/tuple.hpp>

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

namespace bp = boost::python;

namespace
{
  BOOST_PYTHON_FUNCTION_OVERLOADS(imread_overloads,cv::imread,1,2)
  ;

  BOOST_PYTHON_FUNCTION_OVERLOADS(imwrite_overloads,cv::imwrite,2,3)
  ;
  BOOST_PYTHON_FUNCTION_OVERLOADS(imencode_overloads,cv::imencode,3,4)
  ;
  struct PyMCallBackData
  {
    bp::object cb,udata;

    static void callback_fn(int event,int x, int y, int flags, void* param)
    {
      PyMCallBackData* d = static_cast<PyMCallBackData*>(param);
      d->cb(event,x,y,flags,d->udata);
    }
    static std::map<std::string,PyMCallBackData*> callbacks_;
  };
  
  std::map<std::string,PyMCallBackData*> PyMCallBackData::callbacks_;
  //typedef void (*MouseCallback )(int event, int x, int y, int flags, void* param);
  //CV_EXPORTS void setMouseCallback( const string& windowName, MouseCallback onMouse, void* param=0)
  void setMouseCallback_(const std::string& windowName, bp::object callback, bp::object userdata)
  {
    if(callback == bp::object())
    {
        std::cout << "Clearing callback" << std::endl;
        PyMCallBackData::callbacks_[windowName] = NULL;
        cv::setMouseCallback(windowName,NULL,NULL);
        return;
    }
    //FIXME get rid of this leak...
    PyMCallBackData* d = new PyMCallBackData;
    d->cb = callback;
    d->udata = userdata;
    PyMCallBackData::callbacks_[windowName] = d;
    cv::setMouseCallback(windowName,&PyMCallBackData::callback_fn,d);
  }
}
namespace opencv_wrappers
{
  void wrap_highgui_defines();
  void wrap_video_capture()
  {
    bp::class_<cv::VideoCapture> VideoCapture_("VideoCapture");
    VideoCapture_.def(bp::init<>());
    VideoCapture_.def(bp::init<std::string>());
    VideoCapture_.def(bp::init<int>());
    typedef bool(cv::VideoCapture::*open_1)(const std::string&);
    typedef bool(cv::VideoCapture::*open_2)(int);
    VideoCapture_.def("open", open_1(&cv::VideoCapture::open));
    VideoCapture_.def("open", open_2(&cv::VideoCapture::open));
    VideoCapture_.def("isOpened", &cv::VideoCapture::isOpened);
    VideoCapture_.def("release", &cv::VideoCapture::release);
    VideoCapture_.def("grab", &cv::VideoCapture::grab);
    VideoCapture_.def("retrieve", &cv::VideoCapture::retrieve);
    VideoCapture_.def("read", &cv::VideoCapture::read);
    VideoCapture_.def("set", &cv::VideoCapture::set);
    VideoCapture_.def("get", &cv::VideoCapture::get);
  }

  void wrap_video_writer()
  {
    bp::class_<cv::VideoWriter> VideoWriter_("VideoWriter");
    VideoWriter_.def(bp::init<>());
    VideoWriter_.def(bp::init<const std::string&, int, double, cv::Size, bool>());
    VideoWriter_.def("open", &cv::VideoWriter::open);
    VideoWriter_.def("isOpened", &cv::VideoWriter::isOpened);
    VideoWriter_.def("write", &cv::VideoWriter::write);
  }

  void wrap_highgui()
  {
    wrap_highgui_defines();
    //video stuff.
    wrap_video_capture();
    wrap_video_writer();

    //image windows
    bp::def("imshow", cv::imshow);
    bp::def("waitKey", cv::waitKey);
    bp::def("namedWindow", cv::namedWindow);
//CV_EXPORTS void setMouseCallback( const string& windowName, MouseCallback onMouse, void* param=0);
    bp::def("setMouseCallback", setMouseCallback_);
    //image io
    bp::def("imread", cv::imread, imread_overloads());
    bp::def("imwrite", cv::imwrite, imwrite_overloads());
    bp::def("imdecode", cv::imdecode);
    bp::def("imencode", cv::imencode, imencode_overloads());
  }
}
