////////////////////////////////////////////////////////////////////////////////
// AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015, University of California, Irvine
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the UC Irvine.
// 4. Neither the name of the UC Irvine nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"

#include <emscripten/bind.h>

using namespace emscripten;
using namespace cv;
using namespace cv::flann;
using namespace cv::ml;

namespace Utils{

    template<typename T>
    emscripten::val data(const cv::Mat& mat) {
        return emscripten::val(emscripten::memory_view<T>( (mat.total()*mat.elemSize())/sizeof(T), (T*) mat.data));
    }

    emscripten::val matPtrI(const cv::Mat& mat, int i) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(0), mat.ptr(i)));
    }

    emscripten::val matPtrII(const cv::Mat& mat, int i, int j) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(1), mat.ptr(i,j)));
    }

    emscripten::val  matFromArray(const emscripten::val& object, int type) {
        int w=  object["width"].as<unsigned>();
        int h=  object["height"].as<unsigned>();
        std::string str = object["data"]["buffer"].as<std::string>();

        cv::Mat mat(h, w, type);
        memcpy(mat.data, str.data(), mat.total()*mat.elemSize());
        
        return emscripten::val(mat);
    }

    cv::Mat* createMat(int rows, int cols, int type, intptr_t data, size_t step) {
        return new cv::Mat(rows, cols, type, reinterpret_cast<void*>(data), step);
    }


    cv::Mat* createMat2(const std::vector<unsigned char>& vector) {
        return new cv::Mat(vector, false);
    }

    // returning MatSize
    static std::vector<int> getMatSize(const cv::Mat& mat)
    {
      std::vector<int> size;
      for (int i = 0; i < mat.dims; i++) {
        size.push_back(mat.size[i]);
      }
      return size;
    }

    static Mat eye(int rows, int cols, int type) {
      return Mat(cv::Mat::eye(rows, cols, type));
    }

    static Mat eye(Size size, int type) {
      return Mat(cv::Mat::eye(size, type));
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta) {
        obj.convertTo(m, rtype, alpha, beta);
    }

    void convertTo_1(const Mat& obj, Mat& m, int rtype) {
        obj.convertTo(m, rtype);
    }

    void convertTo_2(const Mat& obj, Mat& m, int rtype, double alpha) {
        obj.convertTo(m, rtype, alpha);
    }

    Size matSize(const cv::Mat& mat) {
        return  mat.size();
    }
    cv::Mat mat_zeros_iii(int arg0, int arg1, int arg2) {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }
    cv::Mat mat_zeros_Si(cv::Size arg0, int arg1) {
        return cv::Mat::zeros(arg0,arg1);
    }
    cv::Mat mat_zeros_ipii(int arg0, const int* arg1, int arg2) {
        return cv::Mat::zeros(arg0,arg1,arg2);
    }
    cv::Mat mat_ones_iii(int arg0, int arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat mat_ones_ipii(int arg0, const int* arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat mat_ones_Si(cv::Size arg0, int arg1) {
        return cv::Mat::ones(arg0, arg1);
    }

    double matDot(const cv::Mat& obj, const Mat& mat) {
        return  obj.dot(mat);
    }
    Mat matMul(const cv::Mat& obj, const Mat& mat, double scale) {
        return  Mat(obj.mul(mat, scale));
    }
    Mat matT(const cv::Mat& obj) {
        return  Mat(obj.t());
    }
    Mat matInv(const cv::Mat& obj, int type) {
        return  Mat(obj.inv(type));
    }

    void matCopyTo(const cv::Mat& obj, cv::Mat& mat) {
        return obj.copyTo(mat);
    }

    void matCopyTo_1(const cv::Mat& obj, cv::Mat& mat, const cv::Mat& mask) {
        return obj.copyTo(mat, mask);
    }

    void rotatedRectPoints(const cv::RotatedRect& obj, std::vector<cv::Point2f>& points) {
        points.resize(4);
        return obj.points(points.data());
    }

    int cvMatDepth(int flags) {
        return CV_MAT_DEPTH(flags);
    }

    class MinMaxLocResult {
      public:
        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;
    };

    void minMaxLoc(const cv::Mat& src, MinMaxLocResult& result, const cv::Mat& mask) {
        return cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc, mask);
    }

    void minMaxLoc_1(const cv::Mat& src, MinMaxLocResult& result) {
        return cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc);
    }

    class Circle {
      public:
        Point2f center;
        float radius;
    };

    void minEnclosingCircle(const cv::Mat& points, Circle& circle) {
        return cv::minEnclosingCircle(points, circle.center, circle.radius);
    }
}

EMSCRIPTEN_BINDINGS(Utils) {

    register_vector<int>("IntVector");
    register_vector<char>("CharVector");
    register_vector<unsigned>("UnsignedVector");
    register_vector<unsigned char>("UCharVector");
    register_vector<std::string>("StrVector");
    register_vector<emscripten::val>("EmvalVector");
    register_vector<float>("FloatVector");
    register_vector<std::vector<int>>("IntVectorVector");
    register_vector<std::vector<Point>>("PointVectorVector");
    register_vector<cv::Point>("PointVector");
    register_vector<cv::Vec4i>("Vec4iVector");
    register_vector<cv::Mat>("MatVector");
    register_vector<cv::KeyPoint>("KeyPointVector");
    register_vector<cv::Rect>("RectVector");
    register_vector<cv::Point2f>("Point2fVector");

    emscripten::class_<cv::TermCriteria>("TermCriteria")
        .constructor<>()
        .constructor<int, int, double>()
        .property("type", &cv::TermCriteria::type)
        .property("maxCount", &cv::TermCriteria::maxCount)
        .property("epsilon", &cv::TermCriteria::epsilon);

    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        //.constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor<int, int, int, const Scalar&>()
        .constructor(&Utils::createMat, allow_raw_pointers())
        .constructor(&Utils::createMat2, allow_raw_pointers())
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        //.function("assignTo", select_overload<void(Mat&, int)const>(&cv::Mat::assignTo))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int, double, double)>(&Utils::convertTo))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int)>(&Utils::convertTo_1))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int, double)>(&Utils::convertTo_2))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .class_function("eye",select_overload<Mat(int, int, int)>(&Utils::eye))
        .class_function("eye",select_overload<Mat(Size, int)>(&Utils::eye))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))

        .function("copyTo", select_overload<void(const Mat&, Mat&)>(&Utils::matCopyTo))
        .function("copyTo", select_overload<void(const Mat&, Mat&, const Mat&)>(&Utils::matCopyTo_1))
        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))

        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .class_function("ones",select_overload<Mat(int, int, int)>(&Utils::mat_ones_iii))
        .class_function("ones",select_overload<Mat(Size, int)>(&Utils::mat_ones_Si))
        .class_function("zeros",select_overload<Mat(int, int, int)>(&Utils::mat_zeros_iii))
        .class_function("zeros",select_overload<Mat(Size, int)>(&Utils::mat_zeros_Si))
        .function("depth", select_overload<int()const>(&cv::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&cv::Mat::col))

        .function("dot", select_overload<double(const Mat&, const Mat&)>(&Utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&Utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&Utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&Utils::matT))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)

        .function("data", &Utils::data<unsigned char>)
        .function("data8S", &Utils::data<char>)
        .function("data16u", &Utils::data<unsigned short>)
        .function("data16s", &Utils::data<short>)
        .function("data32s", &Utils::data<int>)
        .function("data32f", &Utils::data<float>)
        .function("data64f", &Utils::data<double>)

        .function("ptr", select_overload<val(const Mat&, int)>(&Utils::matPtrI))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&Utils::matPtrII))

        .function("size" , &Utils::getMatSize)
        .function("get_uchar_at" , select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_ushort_at", select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_int_at" , select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("get_double_at", select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int, int)>(&cv::Mat::at<double>))
        .function("get_float_at", select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function( "getROI_Rect", select_overload<Mat(const Rect&)const>(&cv::Mat::operator()));

    emscripten::class_<cv::Vec<int,4>>("Vec4i")
        .constructor<>()
        .constructor<int, int, int, int>();

    emscripten::class_<cv::RNG> ("RNG");

#define EMSCRIPTEN_CV_SIZE(type) \
    value_array<type>("#type") \
        .element(&type::height) \
        .element(&type::width);

    EMSCRIPTEN_CV_SIZE(Size)
    EMSCRIPTEN_CV_SIZE(Size2f)

#define EMSCRIPTEN_CV_POINT(type) \
    value_array<type>("#type") \
        .element(&type::x) \
        .element(&type::y); \

    EMSCRIPTEN_CV_POINT(Point)
    EMSCRIPTEN_CV_POINT(Point2f)

#define EMSCRIPTEN_CV_RECT(type, name) \
    emscripten::class_<cv::Rect_<type>> (name) \
        .constructor<>() \
        .constructor<const cv::Point_<type>&, const cv::Size_<type>&>() \
        .constructor<type, type, type, type>() \
        .constructor<const cv::Rect_<type>&>() \
        .property("x", &cv::Rect_<type>::x) \
        .property("y", &cv::Rect_<type>::y) \
        .property("width", &cv::Rect_<type>::width) \
        .property("height", &cv::Rect_<type>::height);

    EMSCRIPTEN_CV_RECT(int, "Rect")
    EMSCRIPTEN_CV_RECT(float, "Rect2f")

    emscripten::class_<cv::RotatedRect>("RotatedRect")
        .constructor<>()
        .constructor<const Point2f&, const Size2f&, float>()
        .property("center", &cv::RotatedRect::center)
        .property("size", &cv::RotatedRect::size)
        .property("angle", &cv::RotatedRect::angle)
        .function("points", select_overload<void(const cv::RotatedRect&, std::vector<Point2f>&)>(&Utils::rotatedRectPoints))
        .function("boundingRect", select_overload<Rect()const>(&cv::RotatedRect::boundingRect))
        .function("boundingRect2f", select_overload<Rect2f()const>(&cv::RotatedRect::boundingRect2f));

    emscripten::class_<cv::Scalar_<double>> ("Scalar")
        .constructor<>()
        .constructor<double>()
        .constructor<double, double>()
        .constructor<double, double, double>()
        .constructor<double, double, double, double>()
        .class_function("all", &cv::Scalar_<double>::all)
        .function("isReal", select_overload<bool()const>(&cv::Scalar_<double>::isReal));

    emscripten::class_<Utils::MinMaxLocResult>("MinMaxLocResult")
        .constructor<>()
        .property("minVal", &Utils::MinMaxLocResult::minVal)
        .property("maxVal", &Utils::MinMaxLocResult::maxVal)
        .property("minLoc", &Utils::MinMaxLocResult::minLoc)
        .property("maxLoc", &Utils::MinMaxLocResult::maxLoc);

    emscripten::class_<Utils::Circle>("Circle")
        .constructor<>()
        .property("center", &Utils::Circle::center)
        .property("radius", &Utils::Circle::radius);

    function("minEnclosingCircle", select_overload<void(const cv::Mat&, Utils::Circle&)>(&Utils::minEnclosingCircle));

    function("minMaxLoc", select_overload<void(const cv::Mat&, Utils::MinMaxLocResult&, const cv::Mat&)>(&Utils::minMaxLoc));

    function("minMaxLoc", select_overload<void(const cv::Mat&, Utils::MinMaxLocResult&)>(&Utils::minMaxLoc_1));

    function("matFromArray", &Utils::matFromArray);

    function("morphologyDefaultBorderValue", &cv::morphologyDefaultBorderValue);

    function("CV_MAT_DEPTH", &Utils::cvMatDepth);

    constant("CV_8UC1", CV_8UC1) ;
    constant("CV_8UC2", CV_8UC2) ;
    constant("CV_8UC3", CV_8UC3) ;
    constant("CV_8UC4", CV_8UC4) ;

    constant("CV_8SC1", CV_8SC1) ;
    constant("CV_8SC2", CV_8SC2) ;
    constant("CV_8SC3", CV_8SC3) ;
    constant("CV_8SC4", CV_8SC4) ;

    constant("CV_16UC1", CV_16UC1) ;
    constant("CV_16UC2", CV_16UC2) ;
    constant("CV_16UC3", CV_16UC3) ;
    constant("CV_16UC4", CV_16UC4) ;

    constant("CV_16SC1", CV_16SC1) ;
    constant("CV_16SC2", CV_16SC2) ;
    constant("CV_16SC3", CV_16SC3) ;
    constant("CV_16SC4", CV_16SC4) ;

    constant("CV_32SC1", CV_32SC1) ;
    constant("CV_32SC2", CV_32SC2) ;
    constant("CV_32SC3", CV_32SC3) ;
    constant("CV_32SC4", CV_32SC4) ;

    constant("CV_32FC1", CV_32FC1) ;
    constant("CV_32FC2", CV_32FC2) ;
    constant("CV_32FC3", CV_32FC3) ;
    constant("CV_32FC4", CV_32FC4) ;

    constant("CV_64FC1", CV_64FC1) ;
    constant("CV_64FC2", CV_64FC2) ;
    constant("CV_64FC3", CV_64FC3) ;
    constant("CV_64FC4", CV_64FC4) ;

    constant("CV_8U", CV_8U);
    constant("CV_8S", CV_8S);
    constant("CV_16U", CV_16U);
    constant("CV_16S", CV_16S);
    constant("CV_32S",  CV_32S);
    constant("CV_32F", CV_32F);
    constant("CV_64F", CV_64F);

    constant("INT_MIN", INT_MIN);
    constant("INT_MAX", INT_MAX);
}
