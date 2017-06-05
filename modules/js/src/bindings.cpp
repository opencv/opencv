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

        return emscripten::val(cv::Mat(h, w, type, (void*)str.data(), 0));
    }
    cv::Mat* createMat(Size size, int type, intptr_t data, size_t step) {
        return new cv::Mat(size, type, reinterpret_cast<void*>(data), step);
    }
    cv::Mat* createMat2(const std::vector<unsigned char>& vector) {
        return new cv::Mat(vector, false);
    }
    static std::vector<int> getMatSize(const cv::Mat& mat)
    {
      std::vector<int> size;
      for (int i = 0; i < mat.dims; i++) {
        size.push_back(mat.size[i]);
      }
      return size;
    }
    static Mat eye(int rows, int cols, int type) {
      return cv::Mat::eye(rows, cols, type);
    }
    static Mat eye(Size size, int type) {
      return cv::Mat::eye(size, type);
    }
    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta) {
        obj.convertTo(m, rtype, alpha, beta);
    }
    Size matSize(const cv::Mat& mat) {
        return  mat.size();
    }
    cv::Mat zeros(int arg0, int arg1, int arg2) {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }
    cv::Mat zeros(cv::Size arg0, int arg1) {
        return cv::Mat::zeros(arg0,arg1);
    }
    cv::Mat zeros(int arg0, const int* arg1, int arg2) {
        return cv::Mat::zeros(arg0,arg1,arg2);
    }
    cv::Mat ones(int arg0, int arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat ones(int arg0, const int* arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat ones(cv::Size arg0, int arg1) {
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
        .constructor(&Utils::createMat, allow_raw_pointers())
        .constructor(&Utils::createMat2, allow_raw_pointers())
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        //.function("assignTo", select_overload<void(Mat&, int)const>(&cv::Mat::assignTo))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int, double, double)>(&Utils::convertTo))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .class_function("eye",select_overload<Mat(int, int, int)>(&Utils::eye))
        .class_function("eye",select_overload<Mat(Size, int)>(&Utils::eye))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))

        .function("copyTo", select_overload<void(OutputArray)const>(&cv::Mat::copyTo))
        .function("copyTo", select_overload<void(OutputArray, InputArray)const>(&cv::Mat::copyTo))
        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))

        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .class_function("ones",select_overload<Mat(int, int, int)>(&Utils::ones))
        .class_function("ones",select_overload<Mat(Size, int)>(&Utils::ones))
        .class_function("zeros",select_overload<Mat(int, int, int)>(&Utils::zeros))
        .class_function("zeros",select_overload<Mat(Size, int)>(&Utils::zeros))
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

    value_array<Size>("Size")
        .element(&Size::height)
        .element(&Size::width);


    value_array<Point>("Point")
        .element(&Point::x)
        .element(&Point::y);

    value_array<Point2f>("Point2f")
        .element(&Point2f::x)
        .element(&Point2f::y);

    emscripten::class_<cv::Rect_<int>> ("Rect")
        .constructor<>()
        .constructor<const cv::Point_<int>&, const cv::Size_<int>&>()
        .constructor<int, int, int, int>()
        .constructor<const cv::Rect_<int>&>()
        .property("x", &cv::Rect_<int>::x)
        .property("y", &cv::Rect_<int>::y)
        .property("width", &cv::Rect_<int>::width)
        .property("height", &cv::Rect_<int>::height);

    emscripten::class_<cv::Scalar_<double>> ("Scalar")
        .constructor<>()
        .constructor<double>()
        .constructor<double, double>()
        .constructor<double, double, double>()
        .constructor<double, double, double, double>()
        .class_function("all", &cv::Scalar_<double>::all)
        .function("isReal", select_overload<bool()const>(&cv::Scalar_<double>::isReal));

    function("matFromArray", &Utils::matFromArray);

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
    constant("CV_32F", CV_32F);


    constant("BORDER_CONSTANT", +cv::BorderTypes::BORDER_CONSTANT);
    constant("BORDER_REPLICATE", +cv::BorderTypes::BORDER_REPLICATE);
    constant("BORDER_REFLECT", +cv::BorderTypes::BORDER_REFLECT);
    constant("BORDER_WRAP", +cv::BorderTypes::BORDER_WRAP);
    constant("BORDER_REFLECT_101", +cv::BorderTypes::BORDER_REFLECT_101);
    constant("BORDER_TRANSPARENT", +cv::BorderTypes::BORDER_TRANSPARENT);
    constant("BORDER_REFLECT101", +cv::BorderTypes::BORDER_REFLECT101);
    constant("BORDER_DEFAULT", +cv::BorderTypes::BORDER_DEFAULT);
    constant("BORDER_ISOLATED", +cv::BorderTypes::BORDER_ISOLATED);

    constant("NORM_INF", +cv::NormTypes::NORM_INF);
    constant("NORM_L1", +cv::NormTypes::NORM_L1);
    constant("NORM_L2", +cv::NormTypes::NORM_L2);
    constant("NORM_L2SQR", +cv::NormTypes::NORM_L2SQR);
    constant("NORM_HAMMING", +cv::NormTypes::NORM_HAMMING);
    constant("NORM_HAMMING2", +cv::NormTypes::NORM_HAMMING2);
    constant("NORM_TYPE_MASK", +cv::NormTypes::NORM_TYPE_MASK);
    constant("NORM_RELATIVE", +cv::NormTypes::NORM_RELATIVE);
    constant("NORM_MINMAX", +cv::NormTypes::NORM_MINMAX);

    constant("INPAINT_NS", +cv::INPAINT_NS);
    constant("INPAINT_TELEA", +cv::INPAINT_TELEA);

}
namespace Wrappers {
    RotatedRect CamShift_wrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3) {
        return cv::CamShift(arg1, arg2, arg3);
    }

    void Canny_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, bool arg6) {
        return cv::Canny(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void GaussianBlur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, double arg4, double arg5, int arg6) {
        return cv::GaussianBlur(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void HoughCircles_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, double arg6, double arg7, int arg8, int arg9) {
        return cv::HoughCircles(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void HoughLines_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, double arg6, double arg7, double arg8, double arg9) {
        return cv::HoughLines(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void HoughLinesP_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, double arg6, double arg7) {
        return cv::HoughLinesP(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void HuMoments_wrapper(const Moments& arg1, cv::Mat& arg2) {
        return cv::HuMoments(arg1, arg2);
    }

    void LUT_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::LUT(arg1, arg2, arg3);
    }

    void Laplacian_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, double arg5, double arg6, int arg7) {
        return cv::Laplacian(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    double Mahalanobis_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::Mahalanobis(arg1, arg2, arg3);
    }

    void PCABackProject_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::PCABackProject(arg1, arg2, arg3, arg4);
    }

    void PCACompute_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::PCACompute(arg1, arg2, arg3, arg4);
    }

    void PCACompute_wrapper1(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, double arg4) {
        return cv::PCACompute(arg1, arg2, arg3, arg4);
    }

    void PCAProject_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::PCAProject(arg1, arg2, arg3, arg4);
    }

    double PSNR_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::PSNR(arg1, arg2);
    }

    void SVBackSubst_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, cv::Mat& arg5) {
        return cv::SVBackSubst(arg1, arg2, arg3, arg4, arg5);
    }

    void SVDecomp_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5) {
        return cv::SVDecomp(arg1, arg2, arg3, arg4, arg5);
    }

    void Scharr_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, double arg6, double arg7, int arg8) {
        return cv::Scharr(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void Sobel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, int arg6, double arg7, double arg8, int arg9) {
        return cv::Sobel(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void absdiff_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::absdiff(arg1, arg2, arg3);
    }

    void accumulate_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::accumulate(arg1, arg2, arg3);
    }

    void accumulateProduct_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::accumulateProduct(arg1, arg2, arg3, arg4);
    }

    void accumulateSquare_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::accumulateSquare(arg1, arg2, arg3);
    }

    void accumulateWeighted_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, const cv::Mat& arg4) {
        return cv::accumulateWeighted(arg1, arg2, arg3, arg4);
    }

    void adaptiveThreshold_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, int arg4, int arg5, int arg6, double arg7) {
        return cv::adaptiveThreshold(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void add_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4, int arg5) {
        return cv::add(arg1, arg2, arg3, arg4, arg5);
    }

    void addWeighted_wrapper(const cv::Mat& arg1, double arg2, const cv::Mat& arg3, double arg4, double arg5, cv::Mat& arg6, int arg7) {
        return cv::addWeighted(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void applyColorMap_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::applyColorMap(arg1, arg2, arg3);
    }

    void approxPolyDP_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, bool arg4) {
        return cv::approxPolyDP(arg1, arg2, arg3, arg4);
    }

    double arcLength_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::arcLength(arg1, arg2);
    }

    void arrowedLine_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7, double arg8) {
        return cv::arrowedLine(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void batchDistance_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, cv::Mat& arg5, int arg6, int arg7, const cv::Mat& arg8, int arg9, bool arg10) {
        return cv::batchDistance(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void bilateralFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, int arg6) {
        return cv::bilateralFilter(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void bitwise_and_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_and(arg1, arg2, arg3, arg4);
    }

    void bitwise_not_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::bitwise_not(arg1, arg2, arg3);
    }

    void bitwise_or_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_or(arg1, arg2, arg3, arg4);
    }

    void bitwise_xor_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_xor(arg1, arg2, arg3, arg4);
    }

    void blur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, Point arg4, int arg5) {
        return cv::blur(arg1, arg2, arg3, arg4, arg5);
    }

    int borderInterpolate_wrapper(int arg1, int arg2, int arg3) {
        return cv::borderInterpolate(arg1, arg2, arg3);
    }

    Rect boundingRect_wrapper(const cv::Mat& arg1) {
        return cv::boundingRect(arg1);
    }

    void boxFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, Size arg4, Point arg5, bool arg6, int arg7) {
        return cv::boxFilter(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void boxPoints_wrapper(RotatedRect arg1, cv::Mat& arg2) {
        return cv::boxPoints(arg1, arg2);
    }

    int buildOpticalFlowPyramid_wrapper(const cv::Mat& arg1, std::vector<cv::Mat>& arg2, Size arg3, int arg4, bool arg5, int arg6, int arg7, bool arg8) {
        return cv::buildOpticalFlowPyramid(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void calcBackProject_wrapper(const std::vector<cv::Mat>& arg1, const std::vector<int>& arg2, const cv::Mat& arg3, cv::Mat& arg4, const std::vector<float>& arg5, double arg6) {
        return cv::calcBackProject(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void calcCovarMatrix_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::calcCovarMatrix(arg1, arg2, arg3, arg4, arg5);
    }

    void calcHist_wrapper(const std::vector<cv::Mat>& arg1, const std::vector<int>& arg2, const cv::Mat& arg3, cv::Mat& arg4, const std::vector<int>& arg5, const std::vector<float>& arg6, bool arg7) {
        return cv::calcHist(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void calcOpticalFlowFarneback_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5, int arg6, int arg7, int arg8, double arg9, int arg10) {
        return cv::calcOpticalFlowFarneback(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void calcOpticalFlowPyrLK_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4, cv::Mat& arg5, cv::Mat& arg6, Size arg7, int arg8, TermCriteria arg9, int arg10, double arg11) {
        return cv::calcOpticalFlowPyrLK(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    void cartToPolar_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, bool arg5) {
        return cv::cartToPolar(arg1, arg2, arg3, arg4, arg5);
    }

    void circle_wrapper(cv::Mat& arg1, Point arg2, int arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::circle(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    bool clipLine_wrapper(Rect arg1, Point& arg2, Point& arg3) {
        return cv::clipLine(arg1, arg2, arg3);
    }

    void colorChange_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, float arg4, float arg5, float arg6) {
        return cv::colorChange(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void compare_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::compare(arg1, arg2, arg3, arg4);
    }

    double compareHist_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, int arg3) {
        return cv::compareHist(arg1, arg2, arg3);
    }

    void completeSymm_wrapper(cv::Mat& arg1, bool arg2) {
        return cv::completeSymm(arg1, arg2);
    }

    int connectedComponents_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::connectedComponents(arg1, arg2, arg3, arg4);
    }

    int connectedComponentsWithStats_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, int arg6) {
        return cv::connectedComponentsWithStats(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    double contourArea_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::contourArea(arg1, arg2);
    }

    void convertMaps_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, bool arg6) {
        return cv::convertMaps(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void convertScaleAbs_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4) {
        return cv::convertScaleAbs(arg1, arg2, arg3, arg4);
    }

    void convexHull_wrapper(const cv::Mat& arg1, cv::Mat& arg2, bool arg3, bool arg4) {
        return cv::convexHull(arg1, arg2, arg3, arg4);
    }

    void convexityDefects_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::convexityDefects(arg1, arg2, arg3);
    }

    void copyMakeBorder_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, int arg6, int arg7, const Scalar& arg8) {
        return cv::copyMakeBorder(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void cornerEigenValsAndVecs_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::cornerEigenValsAndVecs(arg1, arg2, arg3, arg4, arg5);
    }

    void cornerHarris_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, double arg5, int arg6) {
        return cv::cornerHarris(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void cornerMinEigenVal_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::cornerMinEigenVal(arg1, arg2, arg3, arg4, arg5);
    }

    void cornerSubPix_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, Size arg4, TermCriteria arg5) {
        return cv::cornerSubPix(arg1, arg2, arg3, arg4, arg5);
    }

    int countNonZero_wrapper(const cv::Mat& arg1) {
        return cv::countNonZero(arg1);
    }

    void createHanningWindow_wrapper(cv::Mat& arg1, Size arg2, int arg3) {
        return cv::createHanningWindow(arg1, arg2, arg3);
    }

    void cvtColor_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::cvtColor(arg1, arg2, arg3, arg4);
    }

    void dct_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::dct(arg1, arg2, arg3);
    }

    void decolor_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3) {
        return cv::decolor(arg1, arg2, arg3);
    }

    void demosaicing_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::demosaicing(arg1, arg2, arg3, arg4);
    }

    void denoise_TVL1_wrapper(const std::vector<Mat>& arg1, Mat& arg2, double arg3, int arg4) {
        return cv::denoise_TVL1(arg1, arg2, arg3, arg4);
    }

    void detailEnhance_wrapper(const cv::Mat& arg1, cv::Mat& arg2, float arg3, float arg4) {
        return cv::detailEnhance(arg1, arg2, arg3, arg4);
    }

    double determinant_wrapper(const cv::Mat& arg1) {
        return cv::determinant(arg1);
    }

    void dft_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::dft(arg1, arg2, arg3, arg4);
    }

    void dilate_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Point arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::dilate(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void distanceTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::distanceTransform(arg1, arg2, arg3, arg4, arg5);
    }

    void distanceTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5, int arg6) {
        return cv::distanceTransform(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void divide_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5) {
        return cv::divide(arg1, arg2, arg3, arg4, arg5);
    }

    void divide_wrapper1(double arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::divide(arg1, arg2, arg3, arg4);
    }

    void drawContours_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, int arg3, const Scalar& arg4, int arg5, int arg6, const cv::Mat& arg7, int arg8, Point arg9) {
        return cv::drawContours(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void drawKeypoints_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, cv::Mat& arg3, const Scalar& arg4, int arg5) {
        return cv::drawKeypoints(arg1, arg2, arg3, arg4, arg5);
    }

    void drawMarker_wrapper(Mat& arg1, Point arg2, const Scalar& arg3, int arg4, int arg5, int arg6, int arg7) {
        return cv::drawMarker(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void drawMatches_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, const cv::Mat& arg3, const std::vector<KeyPoint>& arg4, const std::vector<DMatch>& arg5, cv::Mat& arg6, const Scalar& arg7, const Scalar& arg8, const std::vector<char>& arg9, int arg10) {
        return cv::drawMatches(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void drawMatches_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, const cv::Mat& arg3, const std::vector<KeyPoint>& arg4, const std::vector<std::vector<DMatch> >& arg5, cv::Mat& arg6, const Scalar& arg7, const Scalar& arg8, const std::vector<std::vector<char> >& arg9, int arg10) {
        return cv::drawMatches(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void edgePreservingFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, float arg4, float arg5) {
        return cv::edgePreservingFilter(arg1, arg2, arg3, arg4, arg5);
    }

    bool eigen_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3) {
        return cv::eigen(arg1, arg2, arg3);
    }

    void ellipse_wrapper(cv::Mat& arg1, Point arg2, Size arg3, double arg4, double arg5, double arg6, const Scalar& arg7, int arg8, int arg9, int arg10) {
        return cv::ellipse(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void ellipse_wrapper1(cv::Mat& arg1, const RotatedRect& arg2, const Scalar& arg3, int arg4, int arg5) {
        return cv::ellipse(arg1, arg2, arg3, arg4, arg5);
    }

    void ellipse2Poly_wrapper(Point arg1, Size arg2, int arg3, int arg4, int arg5, int arg6, std::vector<Point>& arg7) {
        return cv::ellipse2Poly(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void equalizeHist_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::equalizeHist(arg1, arg2);
    }

    void erode_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Point arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::erode(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    Mat estimateRigidTransform_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, bool arg3) {
        return cv::estimateRigidTransform(arg1, arg2, arg3);
    }

    void exp_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::exp(arg1, arg2);
    }

    void extractChannel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::extractChannel(arg1, arg2, arg3);
    }

    void fastNlMeansDenoising_wrapper(const cv::Mat& arg1, cv::Mat& arg2, float arg3, int arg4, int arg5) {
        return cv::fastNlMeansDenoising(arg1, arg2, arg3, arg4, arg5);
    }

    void fastNlMeansDenoising_wrapper1(const cv::Mat& arg1, cv::Mat& arg2, const std::vector<float>& arg3, int arg4, int arg5, int arg6) {
        return cv::fastNlMeansDenoising(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void fastNlMeansDenoisingColored_wrapper(const cv::Mat& arg1, cv::Mat& arg2, float arg3, float arg4, int arg5, int arg6) {
        return cv::fastNlMeansDenoisingColored(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void fastNlMeansDenoisingColoredMulti_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2, int arg3, int arg4, float arg5, float arg6, int arg7, int arg8) {
        return cv::fastNlMeansDenoisingColoredMulti(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void fastNlMeansDenoisingMulti_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2, int arg3, int arg4, float arg5, int arg6, int arg7) {
        return cv::fastNlMeansDenoisingMulti(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void fastNlMeansDenoisingMulti_wrapper1(const std::vector<cv::Mat>& arg1, cv::Mat& arg2, int arg3, int arg4, const std::vector<float>& arg5, int arg6, int arg7, int arg8) {
        return cv::fastNlMeansDenoisingMulti(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void fillConvexPoly_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const Scalar& arg3, int arg4, int arg5) {
        return cv::fillConvexPoly(arg1, arg2, arg3, arg4, arg5);
    }

    void fillPoly_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, const Scalar& arg3, int arg4, int arg5, Point arg6) {
        return cv::fillPoly(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void filter2D_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, Point arg5, double arg6, int arg7) {
        return cv::filter2D(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void findContours_wrapper(cv::Mat& arg1, std::vector<cv::Mat>& arg2, cv::Mat& arg3, int arg4, int arg5, Point arg6) {
        return cv::findContours(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void findNonZero_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::findNonZero(arg1, arg2);
    }

    double findTransformECC_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, TermCriteria arg5, const cv::Mat& arg6) {
        return cv::findTransformECC(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    RotatedRect fitEllipse_wrapper(const cv::Mat& arg1) {
        return cv::fitEllipse(arg1);
    }

    void fitLine_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, double arg6) {
        return cv::fitLine(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void flip_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::flip(arg1, arg2, arg3);
    }

    void gemm_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, double arg3, const cv::Mat& arg4, double arg5, cv::Mat& arg6, int arg7) {
        return cv::gemm(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    Mat getAffineTransform_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::getAffineTransform(arg1, arg2);
    }

    Mat getDefaultNewCameraMatrix_wrapper(const cv::Mat& arg1, Size arg2, bool arg3) {
        return cv::getDefaultNewCameraMatrix(arg1, arg2, arg3);
    }

    void getDerivKernels_wrapper(cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, bool arg6, int arg7) {
        return cv::getDerivKernels(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    Mat getGaborKernel_wrapper(Size arg1, double arg2, double arg3, double arg4, double arg5, double arg6, int arg7) {
        return cv::getGaborKernel(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    Mat getGaussianKernel_wrapper(int arg1, double arg2, int arg3) {
        return cv::getGaussianKernel(arg1, arg2, arg3);
    }

    int getOptimalDFTSize_wrapper(int arg1) {
        return cv::getOptimalDFTSize(arg1);
    }

    Mat getPerspectiveTransform_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::getPerspectiveTransform(arg1, arg2);
    }

    void getRectSubPix_wrapper(const cv::Mat& arg1, Size arg2, Point2f arg3, cv::Mat& arg4, int arg5) {
        return cv::getRectSubPix(arg1, arg2, arg3, arg4, arg5);
    }

    Mat getRotationMatrix2D_wrapper(Point2f arg1, double arg2, double arg3) {
        return cv::getRotationMatrix2D(arg1, arg2, arg3);
    }

    Mat getStructuringElement_wrapper(int arg1, Size arg2, Point arg3) {
        return cv::getStructuringElement(arg1, arg2, arg3);
    }

    Size getTextSize_wrapper(const std::string& arg1, int arg2, double arg3, int arg4, int* arg5) {
        return cv::getTextSize(arg1, arg2, arg3, arg4, arg5);
    }

    void goodFeaturesToTrack_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, const cv::Mat& arg6, int arg7, bool arg8, double arg9) {
        return cv::goodFeaturesToTrack(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void grabCut_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Rect arg3, cv::Mat& arg4, cv::Mat& arg5, int arg6, int arg7) {
        return cv::grabCut(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void groupRectangles_wrapper(std::vector<Rect>& arg1, std::vector<int>& arg2, int arg3, double arg4) {
        return cv::groupRectangles(arg1, arg2, arg3, arg4);
    }

    void hconcat_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::hconcat(arg1, arg2);
    }

    void idct_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::idct(arg1, arg2, arg3);
    }

    void idft_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::idft(arg1, arg2, arg3, arg4);
    }

    void illuminationChange_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, float arg4, float arg5) {
        return cv::illuminationChange(arg1, arg2, arg3, arg4, arg5);
    }

    Mat imdecode_wrapper(const cv::Mat& arg1, int arg2) {
        return cv::imdecode(arg1, arg2);
    }

    bool imencode_wrapper(const std::string& arg1, const cv::Mat& arg2, std::vector<uchar>& arg3, const std::vector<int>& arg4) {
        return cv::imencode(arg1, arg2, arg3, arg4);
    }

    Mat imread_wrapper(const std::string& arg1, int arg2) {
        return cv::imread(arg1, arg2);
    }

    bool imreadmulti_wrapper(const std::string& arg1, std::vector<Mat>& arg2, int arg3) {
        return cv::imreadmulti(arg1, arg2, arg3);
    }

    bool imwrite_wrapper(const std::string& arg1, const cv::Mat& arg2, const std::vector<int>& arg3) {
        return cv::imwrite(arg1, arg2, arg3);
    }

    void inRange_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::inRange(arg1, arg2, arg3, arg4);
    }

    void initUndistortRectifyMap_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, Size arg5, int arg6, cv::Mat& arg7, cv::Mat& arg8) {
        return cv::initUndistortRectifyMap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    float initWideAngleProjMap_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, Size arg3, int arg4, int arg5, cv::Mat& arg6, cv::Mat& arg7, int arg8, double arg9) {
        return cv::initWideAngleProjMap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void inpaint_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5) {
        return cv::inpaint(arg1, arg2, arg3, arg4, arg5);
    }

    void insertChannel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::insertChannel(arg1, arg2, arg3);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::integral(arg1, arg2, arg3);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, int arg6) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    float intersectConvexConvex_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, bool arg4) {
        return cv::intersectConvexConvex(arg1, arg2, arg3, arg4);
    }

    double invert_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::invert(arg1, arg2, arg3);
    }

    void invertAffineTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::invertAffineTransform(arg1, arg2);
    }

    bool isContourConvex_wrapper(const cv::Mat& arg1) {
        return cv::isContourConvex(arg1);
    }

    double kmeans_wrapper(const cv::Mat& arg1, int arg2, cv::Mat& arg3, TermCriteria arg4, int arg5, int arg6, cv::Mat& arg7) {
        return cv::kmeans(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void line_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::line(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void linearPolar_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Point2f arg3, double arg4, int arg5) {
        return cv::linearPolar(arg1, arg2, arg3, arg4, arg5);
    }

    void log_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::log(arg1, arg2);
    }

    void logPolar_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Point2f arg3, double arg4, int arg5) {
        return cv::logPolar(arg1, arg2, arg3, arg4, arg5);
    }

    void magnitude_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::magnitude(arg1, arg2, arg3);
    }

    double matchShapes_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, int arg3, double arg4) {
        return cv::matchShapes(arg1, arg2, arg3, arg4);
    }

    void matchTemplate_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, const cv::Mat& arg5) {
        return cv::matchTemplate(arg1, arg2, arg3, arg4, arg5);
    }

    void max_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::max(arg1, arg2, arg3);
    }

    Scalar mean_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::mean(arg1, arg2);
    }

    int meanShift_wrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3) {
        return cv::meanShift(arg1, arg2, arg3);
    }

    void meanStdDev_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::meanStdDev(arg1, arg2, arg3, arg4);
    }

    void medianBlur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::medianBlur(arg1, arg2, arg3);
    }

    void merge_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::merge(arg1, arg2);
    }

    void min_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::min(arg1, arg2, arg3);
    }

    RotatedRect minAreaRect_wrapper(const cv::Mat& arg1) {
        return cv::minAreaRect(arg1);
    }

    double minEnclosingTriangle_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::minEnclosingTriangle(arg1, arg2);
    }

    void mixChannels_wrapper(const std::vector<cv::Mat>& arg1, InputOutputArrayOfArrays arg2, const std::vector<int>& arg3) {
        return cv::mixChannels(arg1, arg2, arg3);
    }

    Moments moments_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::moments(arg1, arg2);
    }

    void morphologyEx_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, Point arg5, int arg6, int arg7, const Scalar& arg8) {
        return cv::morphologyEx(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void mulSpectrums_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, bool arg5) {
        return cv::mulSpectrums(arg1, arg2, arg3, arg4, arg5);
    }

    void mulTransposed_wrapper(const cv::Mat& arg1, cv::Mat& arg2, bool arg3, const cv::Mat& arg4, double arg5, int arg6) {
        return cv::mulTransposed(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void multiply_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5) {
        return cv::multiply(arg1, arg2, arg3, arg4, arg5);
    }

    double norm_wrapper(const cv::Mat& arg1, int arg2, const cv::Mat& arg3) {
        return cv::norm(arg1, arg2, arg3);
    }

    double norm_wrapper1(const cv::Mat& arg1, const cv::Mat& arg2, int arg3, const cv::Mat& arg4) {
        return cv::norm(arg1, arg2, arg3, arg4);
    }

    void normalize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, int arg6, const cv::Mat& arg7) {
        return cv::normalize(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void patchNaNs_wrapper(cv::Mat& arg1, double arg2) {
        return cv::patchNaNs(arg1, arg2);
    }

    void pencilSketch_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, float arg4, float arg5, float arg6) {
        return cv::pencilSketch(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void perspectiveTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::perspectiveTransform(arg1, arg2, arg3);
    }

    void phase_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, bool arg4) {
        return cv::phase(arg1, arg2, arg3, arg4);
    }

    double pointPolygonTest_wrapper(const cv::Mat& arg1, Point2f arg2, bool arg3) {
        return cv::pointPolygonTest(arg1, arg2, arg3);
    }

    void polarToCart_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, bool arg5) {
        return cv::polarToCart(arg1, arg2, arg3, arg4, arg5);
    }

    void polylines_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, bool arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::polylines(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void pow_wrapper(const cv::Mat& arg1, double arg2, cv::Mat& arg3) {
        return cv::pow(arg1, arg2, arg3);
    }

    void preCornerDetect_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::preCornerDetect(arg1, arg2, arg3, arg4);
    }

    void putText_wrapper(cv::Mat& arg1, const std::string& arg2, Point arg3, int arg4, double arg5, Scalar arg6, int arg7, int arg8, bool arg9) {
        return cv::putText(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void pyrDown_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const Size& arg3, int arg4) {
        return cv::pyrDown(arg1, arg2, arg3, arg4);
    }

    void pyrMeanShiftFiltering_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, TermCriteria arg6) {
        return cv::pyrMeanShiftFiltering(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void pyrUp_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const Size& arg3, int arg4) {
        return cv::pyrUp(arg1, arg2, arg3, arg4);
    }

    void randn_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::randn(arg1, arg2, arg3);
    }

    void randu_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::randu(arg1, arg2, arg3);
    }

    void rectangle_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::rectangle(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void reduce_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::reduce(arg1, arg2, arg3, arg4, arg5);
    }

    void remap_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::remap(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void repeat_wrapper(const cv::Mat& arg1, int arg2, int arg3, cv::Mat& arg4) {
        return cv::repeat(arg1, arg2, arg3, arg4);
    }

    void resize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, double arg4, double arg5, int arg6) {
        return cv::resize(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    int rotatedRectangleIntersection_wrapper(const RotatedRect& arg1, const RotatedRect& arg2, cv::Mat& arg3) {
        return cv::rotatedRectangleIntersection(arg1, arg2, arg3);
    }

    void scaleAdd_wrapper(const cv::Mat& arg1, double arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::scaleAdd(arg1, arg2, arg3, arg4);
    }

    void seamlessClone_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, Point arg4, cv::Mat& arg5, int arg6) {
        return cv::seamlessClone(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void sepFilter2D_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, const cv::Mat& arg5, Point arg6, double arg7, int arg8) {
        return cv::sepFilter2D(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void setIdentity_wrapper(cv::Mat& arg1, const Scalar& arg2) {
        return cv::setIdentity(arg1, arg2);
    }

    bool solve_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::solve(arg1, arg2, arg3, arg4);
    }

    int solveCubic_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::solveCubic(arg1, arg2);
    }

    double solvePoly_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::solvePoly(arg1, arg2, arg3);
    }

    void sort_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::sort(arg1, arg2, arg3);
    }

    void sortIdx_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::sortIdx(arg1, arg2, arg3);
    }

    void spatialGradient_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::spatialGradient(arg1, arg2, arg3, arg4, arg5);
    }

    void split_wrapper(const cv::Mat& arg1, std::vector<cv::Mat>& arg2) {
        return cv::split(arg1, arg2);
    }

    void sqrBoxFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, Size arg4, Point arg5, bool arg6, int arg7) {
        return cv::sqrBoxFilter(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void sqrt_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::sqrt(arg1, arg2);
    }

    void stylization_wrapper(const cv::Mat& arg1, cv::Mat& arg2, float arg3, float arg4) {
        return cv::stylization(arg1, arg2, arg3, arg4);
    }

    void subtract_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4, int arg5) {
        return cv::subtract(arg1, arg2, arg3, arg4, arg5);
    }

    Scalar sum_wrapper(const cv::Mat& arg1) {
        return cv::sum(arg1);
    }

    void textureFlattening_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, float arg4, float arg5, int arg6) {
        return cv::textureFlattening(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    double threshold_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5) {
        return cv::threshold(arg1, arg2, arg3, arg4, arg5);
    }

    Scalar trace_wrapper(const cv::Mat& arg1) {
        return cv::trace(arg1);
    }

    void transform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::transform(arg1, arg2, arg3);
    }

    void transpose_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::transpose(arg1, arg2);
    }

    void undistort_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, const cv::Mat& arg5) {
        return cv::undistort(arg1, arg2, arg3, arg4, arg5);
    }

    void undistortPoints_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, const cv::Mat& arg5, const cv::Mat& arg6) {
        return cv::undistortPoints(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void vconcat_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::vconcat(arg1, arg2);
    }

    void warpAffine_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Size arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::warpAffine(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void warpPerspective_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Size arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::warpPerspective(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void watershed_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::watershed(arg1, arg2);
    }

    void finish_wrapper() {
        return cv::ocl::finish();
    }

    bool haveAmdBlas_wrapper() {
        return cv::ocl::haveAmdBlas();
    }

    bool haveAmdFft_wrapper() {
        return cv::ocl::haveAmdFft();
    }

    bool haveOpenCL_wrapper() {
        return cv::ocl::haveOpenCL();
    }

    void setUseOpenCL_wrapper(bool arg1) {
        return cv::ocl::setUseOpenCL(arg1);
    }

    bool useOpenCL_wrapper() {
        return cv::ocl::useOpenCL();
    }

    void MergeMertens_setExposureWeight_wrapper(cv::MergeMertens& arg0 , float arg1) {
        return arg0.setExposureWeight(arg1);
    }

    void MergeMertens_process_wrapper(cv::MergeMertens& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    void MergeMertens_process_wrapper1(cv::MergeMertens& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return arg0.process(arg1, arg2);
    }

    void MergeMertens_setSaturationWeight_wrapper(cv::MergeMertens& arg0 , float arg1) {
        return arg0.setSaturationWeight(arg1);
    }

    Ptr<MergeMertens> _createMergeMertens_wrapper(float arg1, float arg2, float arg3) {
        return cv::createMergeMertens(arg1, arg2, arg3);
    }

    void MergeMertens_setContrastWeight_wrapper(cv::MergeMertens& arg0 , float arg1) {
        return arg0.setContrastWeight(arg1);
    }

    Ptr<BackgroundSubtractorMOG2> _createBackgroundSubtractorMOG2_wrapper(int arg1, double arg2, bool arg3) {
        return cv::createBackgroundSubtractorMOG2(arg1, arg2, arg3);
    }

    void BackgroundSubtractorMOG2_setVarMax_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setVarMax(arg1);
    }

    void BackgroundSubtractorMOG2_setBackgroundRatio_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setBackgroundRatio(arg1);
    }

    void BackgroundSubtractorMOG2_setNMixtures_wrapper(cv::BackgroundSubtractorMOG2& arg0 , int arg1) {
        return arg0.setNMixtures(arg1);
    }

    void BackgroundSubtractorMOG2_setVarInit_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setVarInit(arg1);
    }

    void BackgroundSubtractorMOG2_setDetectShadows_wrapper(cv::BackgroundSubtractorMOG2& arg0 , bool arg1) {
        return arg0.setDetectShadows(arg1);
    }

    void BackgroundSubtractorMOG2_setVarMin_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setVarMin(arg1);
    }

    void BackgroundSubtractorMOG2_setHistory_wrapper(cv::BackgroundSubtractorMOG2& arg0 , int arg1) {
        return arg0.setHistory(arg1);
    }

    void BackgroundSubtractorMOG2_setVarThreshold_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setVarThreshold(arg1);
    }

    void BackgroundSubtractorMOG2_setShadowThreshold_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setShadowThreshold(arg1);
    }

    void BackgroundSubtractorMOG2_setVarThresholdGen_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setVarThresholdGen(arg1);
    }

    void BackgroundSubtractorMOG2_setShadowValue_wrapper(cv::BackgroundSubtractorMOG2& arg0 , int arg1) {
        return arg0.setShadowValue(arg1);
    }

    void BackgroundSubtractorMOG2_setComplexityReductionThreshold_wrapper(cv::BackgroundSubtractorMOG2& arg0 , double arg1) {
        return arg0.setComplexityReductionThreshold(arg1);
    }

    void CalibrateRobertson_setThreshold_wrapper(cv::CalibrateRobertson& arg0 , float arg1) {
        return arg0.setThreshold(arg1);
    }

    void CalibrateRobertson_setMaxIter_wrapper(cv::CalibrateRobertson& arg0 , int arg1) {
        return arg0.setMaxIter(arg1);
    }

    Ptr<CalibrateRobertson> _createCalibrateRobertson_wrapper(int arg1, float arg2) {
        return cv::createCalibrateRobertson(arg1, arg2);
    }

    bool HOGDescriptor_load_wrapper(cv::HOGDescriptor& arg0 , const std::string& arg1, const std::string& arg2) {
        return arg0.load(arg1, arg2);
    }

    void HOGDescriptor_detect_wrapper(cv::HOGDescriptor& arg0 , const Mat& arg1, std::vector<Point>& arg2, std::vector<double>& arg3, double arg4, Size arg5, Size arg6, const std::vector<Point>& arg7) {
        return arg0.detect(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void HOGDescriptor_computeGradient_wrapper(cv::HOGDescriptor& arg0 , const Mat& arg1, Mat& arg2, Mat& arg3, Size arg4, Size arg5) {
        return arg0.computeGradient(arg1, arg2, arg3, arg4, arg5);
    }

    void HOGDescriptor_compute_wrapper(cv::HOGDescriptor& arg0 , const cv::Mat& arg1, std::vector<float>& arg2, Size arg3, Size arg4, const std::vector<Point>& arg5) {
        return arg0.compute(arg1, arg2, arg3, arg4, arg5);
    }

    void HOGDescriptor_save_wrapper(cv::HOGDescriptor& arg0 , const std::string& arg1, const std::string& arg2) {
        return arg0.save(arg1, arg2);
    }

    void HOGDescriptor_setSVMDetector_wrapper(cv::HOGDescriptor& arg0 , const cv::Mat& arg1) {
        return arg0.setSVMDetector(arg1);
    }

    void HOGDescriptor_detectMultiScale_wrapper(cv::HOGDescriptor& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, std::vector<double>& arg3, double arg4, Size arg5, Size arg6, double arg7, double arg8, bool arg9) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void ml_LogisticRegression_setTrainMethod_wrapper(cv::ml::LogisticRegression& arg0 , int arg1) {
        return arg0.setTrainMethod(arg1);
    }

    float ml_LogisticRegression_predict_wrapper(cv::ml::LogisticRegression& arg0 , const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return arg0.predict(arg1, arg2, arg3);
    }

    void ml_LogisticRegression_setTermCriteria_wrapper(cv::ml::LogisticRegression& arg0 , TermCriteria arg1) {
        return arg0.setTermCriteria(arg1);
    }

    void ml_LogisticRegression_setMiniBatchSize_wrapper(cv::ml::LogisticRegression& arg0 , int arg1) {
        return arg0.setMiniBatchSize(arg1);
    }

    void ml_LogisticRegression_setIterations_wrapper(cv::ml::LogisticRegression& arg0 , int arg1) {
        return arg0.setIterations(arg1);
    }

    void ml_LogisticRegression_setLearningRate_wrapper(cv::ml::LogisticRegression& arg0 , double arg1) {
        return arg0.setLearningRate(arg1);
    }

    void ml_LogisticRegression_setRegularization_wrapper(cv::ml::LogisticRegression& arg0 , int arg1) {
        return arg0.setRegularization(arg1);
    }

    Ptr<HistogramCostExtractor> _createNormHistogramCostExtractor_wrapper(int arg1, int arg2, float arg3) {
        return cv::createNormHistogramCostExtractor(arg1, arg2, arg3);
    }

    void NormHistogramCostExtractor_setNormFlag_wrapper(cv::NormHistogramCostExtractor& arg0 , int arg1) {
        return arg0.setNormFlag(arg1);
    }

    Ptr<AgastFeatureDetector> AgastFeatureDetector_create_wrapper(int arg1, bool arg2, int arg3) {
        return cv::AgastFeatureDetector::create(arg1, arg2, arg3);
    }

    void AgastFeatureDetector_setNonmaxSuppression_wrapper(cv::AgastFeatureDetector& arg0 , bool arg1) {
        return arg0.setNonmaxSuppression(arg1);
    }

    void AgastFeatureDetector_setThreshold_wrapper(cv::AgastFeatureDetector& arg0 , int arg1) {
        return arg0.setThreshold(arg1);
    }

    void AgastFeatureDetector_setType_wrapper(cv::AgastFeatureDetector& arg0 , int arg1) {
        return arg0.setType(arg1);
    }

    void ml_Boost_setWeakCount_wrapper(cv::ml::Boost& arg0 , int arg1) {
        return arg0.setWeakCount(arg1);
    }

    void ml_Boost_setWeightTrimRate_wrapper(cv::ml::Boost& arg0 , double arg1) {
        return arg0.setWeightTrimRate(arg1);
    }

    void ml_Boost_setBoostType_wrapper(cv::ml::Boost& arg0 , int arg1) {
        return arg0.setBoostType(arg1);
    }

    void DenseOpticalFlow_calc_wrapper(cv::DenseOpticalFlow& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return arg0.calc(arg1, arg2, arg3);
    }

    int LineSegmentDetector_compareSegments_wrapper(cv::LineSegmentDetector& arg0 , const Size& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return arg0.compareSegments(arg1, arg2, arg3, arg4);
    }

    void LineSegmentDetector_detect_wrapper(cv::LineSegmentDetector& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, cv::Mat& arg5) {
        return arg0.detect(arg1, arg2, arg3, arg4, arg5);
    }

    void LineSegmentDetector_drawSegments_wrapper(cv::LineSegmentDetector& arg0 , cv::Mat& arg1, const cv::Mat& arg2) {
        return arg0.drawSegments(arg1, arg2);
    }

    Ptr<LineSegmentDetector> _createLineSegmentDetector_wrapper(int arg1, double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, int arg8) {
        return cv::createLineSegmentDetector(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void BackgroundSubtractor_apply_wrapper(cv::BackgroundSubtractor& arg0 , const cv::Mat& arg1, cv::Mat& arg2, double arg3) {
        return arg0.apply(arg1, arg2, arg3);
    }

    void BackgroundSubtractor_getBackgroundImage_wrapper(cv::BackgroundSubtractor& arg0 , cv::Mat& arg1) {
        return arg0.getBackgroundImage(arg1);
    }

    Ptr<HistogramCostExtractor> _createChiHistogramCostExtractor_wrapper(int arg1, float arg2) {
        return cv::createChiHistogramCostExtractor(arg1, arg2);
    }

    void KeyPoint_convert_wrapper(cv::KeyPoint& arg0 , const std::vector<KeyPoint>& arg1, std::vector<Point2f>& arg2, const std::vector<int>& arg3) {
        return arg0.convert(arg1, arg2, arg3);
    }

    void KeyPoint_convert_wrapper1(cv::KeyPoint& arg0 , const std::vector<Point2f>& arg1, std::vector<KeyPoint>& arg2, float arg3, float arg4, int arg5, int arg6) {
        return arg0.convert(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    float KeyPoint_overlap_wrapper(cv::KeyPoint& arg0 , const KeyPoint& arg1, const KeyPoint& arg2) {
        return arg0.overlap(arg1, arg2);
    }

    void AlignMTB_setCut_wrapper(cv::AlignMTB& arg0 , bool arg1) {
        return arg0.setCut(arg1);
    }

    void AlignMTB_computeBitmaps_wrapper(cv::AlignMTB& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3) {
        return arg0.computeBitmaps(arg1, arg2, arg3);
    }

    void AlignMTB_process_wrapper(cv::AlignMTB& arg0 , const std::vector<cv::Mat>& arg1, std::vector<Mat>& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    void AlignMTB_process_wrapper1(cv::AlignMTB& arg0 , const std::vector<cv::Mat>& arg1, std::vector<Mat>& arg2) {
        return arg0.process(arg1, arg2);
    }

    Ptr<AlignMTB> _createAlignMTB_wrapper(int arg1, int arg2, bool arg3) {
        return cv::createAlignMTB(arg1, arg2, arg3);
    }

    void AlignMTB_setExcludeRange_wrapper(cv::AlignMTB& arg0 , int arg1) {
        return arg0.setExcludeRange(arg1);
    }

    void AlignMTB_setMaxBits_wrapper(cv::AlignMTB& arg0 , int arg1) {
        return arg0.setMaxBits(arg1);
    }

    void AlignMTB_shiftMat_wrapper(cv::AlignMTB& arg0 , const cv::Mat& arg1, cv::Mat& arg2, const Point arg3) {
        return arg0.shiftMat(arg1, arg2, arg3);
    }

    Point AlignMTB_calculateShift_wrapper(cv::AlignMTB& arg0 , const cv::Mat& arg1, const cv::Mat& arg2) {
        return arg0.calculateShift(arg1, arg2);
    }

    Ptr<AffineTransformer> _createAffineTransformer_wrapper(bool arg1) {
        return cv::createAffineTransformer(arg1);
    }

    void AffineTransformer_setFullAffine_wrapper(cv::AffineTransformer& arg0 , bool arg1) {
        return arg0.setFullAffine(arg1);
    }

    void MergeDebevec_process_wrapper(cv::MergeDebevec& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    void MergeDebevec_process_wrapper1(cv::MergeDebevec& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return arg0.process(arg1, arg2, arg3);
    }

    void HistogramCostExtractor_buildCostMatrix_wrapper(cv::HistogramCostExtractor& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return arg0.buildCostMatrix(arg1, arg2, arg3);
    }

    void HistogramCostExtractor_setNDummies_wrapper(cv::HistogramCostExtractor& arg0 , int arg1) {
        return arg0.setNDummies(arg1);
    }

    void HistogramCostExtractor_setDefaultCost_wrapper(cv::HistogramCostExtractor& arg0 , float arg1) {
        return arg0.setDefaultCost(arg1);
    }

    void ml_DTrees_setTruncatePrunedTree_wrapper(cv::ml::DTrees& arg0 , bool arg1) {
        return arg0.setTruncatePrunedTree(arg1);
    }

    void ml_DTrees_setMaxDepth_wrapper(cv::ml::DTrees& arg0 , int arg1) {
        return arg0.setMaxDepth(arg1);
    }

    void ml_DTrees_setMinSampleCount_wrapper(cv::ml::DTrees& arg0 , int arg1) {
        return arg0.setMinSampleCount(arg1);
    }

    void ml_DTrees_setMaxCategories_wrapper(cv::ml::DTrees& arg0 , int arg1) {
        return arg0.setMaxCategories(arg1);
    }

    void ml_DTrees_setCVFolds_wrapper(cv::ml::DTrees& arg0 , int arg1) {
        return arg0.setCVFolds(arg1);
    }

    void ml_DTrees_setUseSurrogates_wrapper(cv::ml::DTrees& arg0 , bool arg1) {
        return arg0.setUseSurrogates(arg1);
    }

    void ml_DTrees_setUse1SERule_wrapper(cv::ml::DTrees& arg0 , bool arg1) {
        return arg0.setUse1SERule(arg1);
    }

    void ml_DTrees_setRegressionAccuracy_wrapper(cv::ml::DTrees& arg0 , float arg1) {
        return arg0.setRegressionAccuracy(arg1);
    }

    void ml_DTrees_setPriors_wrapper(cv::ml::DTrees& arg0 , const cv::Mat & arg1) {
        return arg0.setPriors(arg1);
    }

    Mat BOWKMeansTrainer_cluster_wrapper(cv::BOWKMeansTrainer& arg0 ) {
        return arg0.cluster();
    }

    Mat BOWKMeansTrainer_cluster_wrapper1(cv::BOWKMeansTrainer& arg0 , const Mat& arg1) {
        return arg0.cluster(arg1);
    }

    Ptr<TonemapMantiuk> _createTonemapMantiuk_wrapper(float arg1, float arg2, float arg3) {
        return cv::createTonemapMantiuk(arg1, arg2, arg3);
    }

    void TonemapMantiuk_setSaturation_wrapper(cv::TonemapMantiuk& arg0 , float arg1) {
        return arg0.setSaturation(arg1);
    }

    void TonemapMantiuk_setScale_wrapper(cv::TonemapMantiuk& arg0 , float arg1) {
        return arg0.setScale(arg1);
    }

    float ml_StatModel_predict_wrapper(cv::ml::StatModel& arg0 , const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return arg0.predict(arg1, arg2, arg3);
    }

    float ml_StatModel_calcError_wrapper(cv::ml::StatModel& arg0 , const Ptr<TrainData>& arg1, bool arg2, cv::Mat& arg3) {
        return arg0.calcError(arg1, arg2, arg3);
    }

    bool ml_StatModel_train_wrapper(cv::ml::StatModel& arg0 , const Ptr<TrainData>& arg1, int arg2) {
        return arg0.train(arg1, arg2);
    }

    bool ml_StatModel_train_wrapper1(cv::ml::StatModel& arg0 , const cv::Mat& arg1, int arg2, const cv::Mat& arg3) {
        return arg0.train(arg1, arg2, arg3);
    }

    void Tonemap_process_wrapper(cv::Tonemap& arg0 , const cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.process(arg1, arg2);
    }

    void Tonemap_setGamma_wrapper(cv::Tonemap& arg0 , float arg1) {
        return arg0.setGamma(arg1);
    }

    Ptr<Tonemap> _createTonemap_wrapper(float arg1) {
        return cv::createTonemap(arg1);
    }

    void ml_KNearest_setIsClassifier_wrapper(cv::ml::KNearest& arg0 , bool arg1) {
        return arg0.setIsClassifier(arg1);
    }

    void ml_KNearest_setAlgorithmType_wrapper(cv::ml::KNearest& arg0 , int arg1) {
        return arg0.setAlgorithmType(arg1);
    }

    void ml_KNearest_setEmax_wrapper(cv::ml::KNearest& arg0 , int arg1) {
        return arg0.setEmax(arg1);
    }

    void ml_KNearest_setDefaultK_wrapper(cv::ml::KNearest& arg0 , int arg1) {
        return arg0.setDefaultK(arg1);
    }

    float ml_KNearest_findNearest_wrapper(cv::ml::KNearest& arg0 , const cv::Mat& arg1, int arg2, cv::Mat& arg3, cv::Mat& arg4, cv::Mat& arg5) {
        return arg0.findNearest(arg1, arg2, arg3, arg4, arg5);
    }

    Ptr<FastFeatureDetector> FastFeatureDetector_create_wrapper(int arg1, bool arg2, int arg3) {
        return cv::FastFeatureDetector::create(arg1, arg2, arg3);
    }

    void FastFeatureDetector_setNonmaxSuppression_wrapper(cv::FastFeatureDetector& arg0 , bool arg1) {
        return arg0.setNonmaxSuppression(arg1);
    }

    void FastFeatureDetector_setThreshold_wrapper(cv::FastFeatureDetector& arg0 , int arg1) {
        return arg0.setThreshold(arg1);
    }

    void FastFeatureDetector_setType_wrapper(cv::FastFeatureDetector& arg0 , int arg1) {
        return arg0.setType(arg1);
    }

    int Subdiv2D_insert_wrapper(cv::Subdiv2D& arg0 , Point2f arg1) {
        return arg0.insert(arg1);
    }

    void Subdiv2D_insert_wrapper1(cv::Subdiv2D& arg0 , const std::vector<Point2f>& arg1) {
        return arg0.insert(arg1);
    }

    int Subdiv2D_edgeOrg_wrapper(cv::Subdiv2D& arg0 , int arg1, Point2f* arg2) {
        return arg0.edgeOrg(arg1, arg2);
    }

    int Subdiv2D_rotateEdge_wrapper(cv::Subdiv2D& arg0 , int arg1, int arg2) {
        return arg0.rotateEdge(arg1, arg2);
    }

    void Subdiv2D_initDelaunay_wrapper(cv::Subdiv2D& arg0 , Rect arg1) {
        return arg0.initDelaunay(arg1);
    }

    int Subdiv2D_getEdge_wrapper(cv::Subdiv2D& arg0 , int arg1, int arg2) {
        return arg0.getEdge(arg1, arg2);
    }

    void Subdiv2D_getTriangleList_wrapper(cv::Subdiv2D& arg0 , std::vector<Vec6f>& arg1) {
        return arg0.getTriangleList(arg1);
    }

    int Subdiv2D_nextEdge_wrapper(cv::Subdiv2D& arg0 , int arg1) {
        return arg0.nextEdge(arg1);
    }

    int Subdiv2D_edgeDst_wrapper(cv::Subdiv2D& arg0 , int arg1, Point2f* arg2) {
        return arg0.edgeDst(arg1, arg2);
    }

    void Subdiv2D_getEdgeList_wrapper(cv::Subdiv2D& arg0 , std::vector<Vec4f>& arg1) {
        return arg0.getEdgeList(arg1);
    }

    Point2f Subdiv2D_getVertex_wrapper(cv::Subdiv2D& arg0 , int arg1, int* arg2) {
        return arg0.getVertex(arg1, arg2);
    }

    void Subdiv2D_getVoronoiFacetList_wrapper(cv::Subdiv2D& arg0 , const std::vector<int>& arg1, std::vector<std::vector<Point2f> >& arg2, std::vector<Point2f>& arg3) {
        return arg0.getVoronoiFacetList(arg1, arg2, arg3);
    }

    int Subdiv2D_symEdge_wrapper(cv::Subdiv2D& arg0 , int arg1) {
        return arg0.symEdge(arg1);
    }

    int Subdiv2D_findNearest_wrapper(cv::Subdiv2D& arg0 , Point2f arg1, Point2f* arg2) {
        return arg0.findNearest(arg1, arg2);
    }

    Ptr<BRISK> BRISK_create_wrapper(int arg1, int arg2, float arg3) {
        return cv::BRISK::create(arg1, arg2, arg3);
    }

    Ptr<BRISK> BRISK_create_wrapper1(const std::vector<float> & arg1, const std::vector<int> & arg2, float arg3, float arg4, const std::vector<int>& arg5) {
        return cv::BRISK::create(arg1, arg2, arg3, arg4, arg5);
    }

    const Mat& KalmanFilter_predict_wrapper(cv::KalmanFilter& arg0 , const Mat& arg1) {
        return arg0.predict(arg1);
    }

    const Mat& KalmanFilter_correct_wrapper(cv::KalmanFilter& arg0 , const Mat& arg1) {
        return arg0.correct(arg1);
    }

    Ptr<HistogramCostExtractor> _createEMDHistogramCostExtractor_wrapper(int arg1, int arg2, float arg3) {
        return cv::createEMDHistogramCostExtractor(arg1, arg2, arg3);
    }

    void EMDHistogramCostExtractor_setNormFlag_wrapper(cv::EMDHistogramCostExtractor& arg0 , int arg1) {
        return arg0.setNormFlag(arg1);
    }

    void ThinPlateSplineShapeTransformer_setRegularizationParameter_wrapper(cv::ThinPlateSplineShapeTransformer& arg0 , double arg1) {
        return arg0.setRegularizationParameter(arg1);
    }

    Ptr<ThinPlateSplineShapeTransformer> _createThinPlateSplineShapeTransformer_wrapper(double arg1) {
        return cv::createThinPlateSplineShapeTransformer(arg1);
    }

    void ShapeContextDistanceExtractor_setAngularBins_wrapper(cv::ShapeContextDistanceExtractor& arg0 , int arg1) {
        return arg0.setAngularBins(arg1);
    }

    void ShapeContextDistanceExtractor_setTransformAlgorithm_wrapper(cv::ShapeContextDistanceExtractor& arg0 , Ptr<ShapeTransformer> arg1) {
        return arg0.setTransformAlgorithm(arg1);
    }

    void ShapeContextDistanceExtractor_setShapeContextWeight_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setShapeContextWeight(arg1);
    }

    void ShapeContextDistanceExtractor_setImageAppearanceWeight_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setImageAppearanceWeight(arg1);
    }

    void ShapeContextDistanceExtractor_setRotationInvariant_wrapper(cv::ShapeContextDistanceExtractor& arg0 , bool arg1) {
        return arg0.setRotationInvariant(arg1);
    }

    void ShapeContextDistanceExtractor_setBendingEnergyWeight_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setBendingEnergyWeight(arg1);
    }

    void ShapeContextDistanceExtractor_setStdDev_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setStdDev(arg1);
    }

    void ShapeContextDistanceExtractor_setCostExtractor_wrapper(cv::ShapeContextDistanceExtractor& arg0 , Ptr<HistogramCostExtractor> arg1) {
        return arg0.setCostExtractor(arg1);
    }

    void ShapeContextDistanceExtractor_setOuterRadius_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setOuterRadius(arg1);
    }

    void ShapeContextDistanceExtractor_getImages_wrapper(cv::ShapeContextDistanceExtractor& arg0 , cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.getImages(arg1, arg2);
    }

    Ptr<ShapeContextDistanceExtractor> _createShapeContextDistanceExtractor_wrapper(int arg1, int arg2, float arg3, float arg4, int arg5, const Ptr<HistogramCostExtractor> & arg6, const Ptr<ShapeTransformer> & arg7) {
        return cv::createShapeContextDistanceExtractor(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void ShapeContextDistanceExtractor_setInnerRadius_wrapper(cv::ShapeContextDistanceExtractor& arg0 , float arg1) {
        return arg0.setInnerRadius(arg1);
    }

    void ShapeContextDistanceExtractor_setIterations_wrapper(cv::ShapeContextDistanceExtractor& arg0 , int arg1) {
        return arg0.setIterations(arg1);
    }

    void ShapeContextDistanceExtractor_setImages_wrapper(cv::ShapeContextDistanceExtractor& arg0 , const cv::Mat& arg1, const cv::Mat& arg2) {
        return arg0.setImages(arg1, arg2);
    }

    void ShapeContextDistanceExtractor_setRadialBins_wrapper(cv::ShapeContextDistanceExtractor& arg0 , int arg1) {
        return arg0.setRadialBins(arg1);
    }

    void KAZE_setExtended_wrapper(cv::KAZE& arg0 , bool arg1) {
        return arg0.setExtended(arg1);
    }

    void KAZE_setNOctaveLayers_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setNOctaveLayers(arg1);
    }

    void KAZE_setNOctaves_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setNOctaves(arg1);
    }

    Ptr<KAZE> KAZE_create_wrapper(bool arg1, bool arg2, float arg3, int arg4, int arg5, int arg6) {
        return cv::KAZE::create(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void KAZE_setUpright_wrapper(cv::KAZE& arg0 , bool arg1) {
        return arg0.setUpright(arg1);
    }

    void KAZE_setDiffusivity_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setDiffusivity(arg1);
    }

    void KAZE_setThreshold_wrapper(cv::KAZE& arg0 , double arg1) {
        return arg0.setThreshold(arg1);
    }

    void CalibrateCRF_process_wrapper(cv::CalibrateCRF& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return arg0.process(arg1, arg2, arg3);
    }

    void AKAZE_setNOctaveLayers_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setNOctaveLayers(arg1);
    }

    void AKAZE_setDescriptorType_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorType(arg1);
    }

    void AKAZE_setNOctaves_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setNOctaves(arg1);
    }

    Ptr<AKAZE> AKAZE_create_wrapper(int arg1, int arg2, int arg3, float arg4, int arg5, int arg6, int arg7) {
        return cv::AKAZE::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void AKAZE_setDescriptorChannels_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorChannels(arg1);
    }

    void AKAZE_setThreshold_wrapper(cv::AKAZE& arg0 , double arg1) {
        return arg0.setThreshold(arg1);
    }

    void AKAZE_setDescriptorSize_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorSize(arg1);
    }

    void AKAZE_setDiffusivity_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDiffusivity(arg1);
    }

    void Feature2D_detect_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, std::vector<KeyPoint>& arg2, const cv::Mat& arg3) {
        return arg0.detect(arg1, arg2, arg3);
    }

    void Feature2D_compute_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, std::vector<KeyPoint>& arg2, cv::Mat& arg3) {
        return arg0.compute(arg1, arg2, arg3);
    }

    void Feature2D_detectAndCompute_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<KeyPoint>& arg3, cv::Mat& arg4, bool arg5) {
        return arg0.detectAndCompute(arg1, arg2, arg3, arg4, arg5);
    }

    bool flann_Index_load_wrapper(cv::flann::Index& arg0 , const cv::Mat& arg1, const std::string& arg2) {
        return arg0.load(arg1, arg2);
    }

    int flann_Index_radiusSearch_wrapper(cv::flann::Index& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5, const SearchParams& arg6) {
        return arg0.radiusSearch(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void flann_Index_build_wrapper(cv::flann::Index& arg0 , const cv::Mat& arg1, const IndexParams& arg2, cvflann::flann_distance_t arg3) {
        return arg0.build(arg1, arg2, arg3);
    }

    void flann_Index_save_wrapper(cv::flann::Index& arg0 , const std::string& arg1) {
        return arg0.save(arg1);
    }

    void flann_Index_knnSearch_wrapper(cv::flann::Index& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, const SearchParams& arg5) {
        return arg0.knnSearch(arg1, arg2, arg3, arg4, arg5);
    }

    float ml_NormalBayesClassifier_predictProb_wrapper(cv::ml::NormalBayesClassifier& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return arg0.predictProb(arg1, arg2, arg3, arg4);
    }

    void MergeExposures_process_wrapper(cv::MergeExposures& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    void MergeRobertson_process_wrapper(cv::MergeRobertson& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    void MergeRobertson_process_wrapper1(cv::MergeRobertson& arg0 , const std::vector<cv::Mat>& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return arg0.process(arg1, arg2, arg3);
    }

    void CalibrateDebevec_setRandom_wrapper(cv::CalibrateDebevec& arg0 , bool arg1) {
        return arg0.setRandom(arg1);
    }

    Ptr<CalibrateDebevec> _createCalibrateDebevec_wrapper(int arg1, float arg2, bool arg3) {
        return cv::createCalibrateDebevec(arg1, arg2, arg3);
    }

    void CalibrateDebevec_setLambda_wrapper(cv::CalibrateDebevec& arg0 , float arg1) {
        return arg0.setLambda(arg1);
    }

    void CalibrateDebevec_setSamples_wrapper(cv::CalibrateDebevec& arg0 , int arg1) {
        return arg0.setSamples(arg1);
    }

    void BOWImgDescriptorExtractor_setVocabulary_wrapper(cv::BOWImgDescriptorExtractor& arg0 , const Mat& arg1) {
        return arg0.setVocabulary(arg1);
    }

    void BOWImgDescriptorExtractor_compute2_wrapper(cv::BOWImgDescriptorExtractor& arg0 , const Mat& arg1, std::vector<KeyPoint>& arg2, Mat& arg3) {
        return arg0.compute2(arg1, arg2, arg3);
    }

    void AlignExposures_process_wrapper(cv::AlignExposures& arg0 , const std::vector<cv::Mat>& arg1, std::vector<Mat>& arg2, const cv::Mat& arg3, const cv::Mat& arg4) {
        return arg0.process(arg1, arg2, arg3, arg4);
    }

    std::string Algorithm_getDefaultName_wrapper(cv::Algorithm& arg0 ) {
        return arg0.getDefaultName();
    }

    void Algorithm_save_wrapper(cv::Algorithm& arg0 , const std::string& arg1) {
        return arg0.save(arg1);
    }

    void ShapeTransformer_warpImage_wrapper(cv::ShapeTransformer& arg0 , const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, const Scalar& arg5) {
        return arg0.warpImage(arg1, arg2, arg3, arg4, arg5);
    }

    void ShapeTransformer_estimateTransformation_wrapper(cv::ShapeTransformer& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<DMatch>& arg3) {
        return arg0.estimateTransformation(arg1, arg2, arg3);
    }

    float ShapeTransformer_applyTransformation_wrapper(cv::ShapeTransformer& arg0 , const cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.applyTransformation(arg1, arg2);
    }

    void ml_ANN_MLP_setRpropDWMinus_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setRpropDWMinus(arg1);
    }

    void ml_ANN_MLP_setBackpropWeightScale_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setBackpropWeightScale(arg1);
    }

    void ml_ANN_MLP_setRpropDWMin_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setRpropDWMin(arg1);
    }

    void ml_ANN_MLP_setActivationFunction_wrapper(cv::ml::ANN_MLP& arg0 , int arg1, double arg2, double arg3) {
        return arg0.setActivationFunction(arg1, arg2, arg3);
    }

    void ml_ANN_MLP_setRpropDWMax_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setRpropDWMax(arg1);
    }

    void ml_ANN_MLP_setRpropDW0_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setRpropDW0(arg1);
    }

    void ml_ANN_MLP_setBackpropMomentumScale_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setBackpropMomentumScale(arg1);
    }

    void ml_ANN_MLP_setRpropDWPlus_wrapper(cv::ml::ANN_MLP& arg0 , double arg1) {
        return arg0.setRpropDWPlus(arg1);
    }

    void ml_ANN_MLP_setTrainMethod_wrapper(cv::ml::ANN_MLP& arg0 , int arg1, double arg2, double arg3) {
        return arg0.setTrainMethod(arg1, arg2, arg3);
    }

    void ml_ANN_MLP_setTermCriteria_wrapper(cv::ml::ANN_MLP& arg0 , TermCriteria arg1) {
        return arg0.setTermCriteria(arg1);
    }

    void ml_ANN_MLP_setLayerSizes_wrapper(cv::ml::ANN_MLP& arg0 , const cv::Mat& arg1) {
        return arg0.setLayerSizes(arg1);
    }

    Mat ml_ANN_MLP_getWeights_wrapper(cv::ml::ANN_MLP& arg0 , int arg1) {
        return arg0.getWeights(arg1);
    }

    void ml_RTrees_setCalculateVarImportance_wrapper(cv::ml::RTrees& arg0 , bool arg1) {
        return arg0.setCalculateVarImportance(arg1);
    }

    void ml_RTrees_setTermCriteria_wrapper(cv::ml::RTrees& arg0 , const TermCriteria & arg1) {
        return arg0.setTermCriteria(arg1);
    }

    void ml_RTrees_setActiveVarCount_wrapper(cv::ml::RTrees& arg0 , int arg1) {
        return arg0.setActiveVarCount(arg1);
    }

    float ShapeDistanceExtractor_computeDistance_wrapper(cv::ShapeDistanceExtractor& arg0 , const cv::Mat& arg1, const cv::Mat& arg2) {
        return arg0.computeDistance(arg1, arg2);
    }

    void GFTTDetector_setHarrisDetector_wrapper(cv::GFTTDetector& arg0 , bool arg1) {
        return arg0.setHarrisDetector(arg1);
    }

    void GFTTDetector_setBlockSize_wrapper(cv::GFTTDetector& arg0 , int arg1) {
        return arg0.setBlockSize(arg1);
    }

    Ptr<GFTTDetector> GFTTDetector_create_wrapper(int arg1, double arg2, double arg3, int arg4, bool arg5, double arg6) {
        return cv::GFTTDetector::create(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void GFTTDetector_setQualityLevel_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setQualityLevel(arg1);
    }

    void GFTTDetector_setMaxFeatures_wrapper(cv::GFTTDetector& arg0 , int arg1) {
        return arg0.setMaxFeatures(arg1);
    }

    void GFTTDetector_setK_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setK(arg1);
    }

    void GFTTDetector_setMinDistance_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setMinDistance(arg1);
    }

    Ptr<DescriptorMatcher> DescriptorMatcher_create_wrapper(const std::string& arg1) {
        return cv::DescriptorMatcher::create(arg1);
    }

    void DescriptorMatcher_knnMatch_wrapper(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<std::vector<DMatch> >& arg3, int arg4, const cv::Mat& arg5, bool arg6) {
        return arg0.knnMatch(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void DescriptorMatcher_knnMatch_wrapper1(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, std::vector<std::vector<DMatch> >& arg2, int arg3, const std::vector<cv::Mat>& arg4, bool arg5) {
        return arg0.knnMatch(arg1, arg2, arg3, arg4, arg5);
    }

    void DescriptorMatcher_add_wrapper(cv::DescriptorMatcher& arg0 , const std::vector<cv::Mat>& arg1) {
        return arg0.add(arg1);
    }

    void DescriptorMatcher_match_wrapper(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<DMatch>& arg3, const cv::Mat& arg4) {
        return arg0.match(arg1, arg2, arg3, arg4);
    }

    void DescriptorMatcher_match_wrapper1(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, std::vector<DMatch>& arg2, const std::vector<cv::Mat>& arg3) {
        return arg0.match(arg1, arg2, arg3);
    }

    void TonemapDurand_setSigmaSpace_wrapper(cv::TonemapDurand& arg0 , float arg1) {
        return arg0.setSigmaSpace(arg1);
    }

    void TonemapDurand_setSigmaColor_wrapper(cv::TonemapDurand& arg0 , float arg1) {
        return arg0.setSigmaColor(arg1);
    }

    void TonemapDurand_setSaturation_wrapper(cv::TonemapDurand& arg0 , float arg1) {
        return arg0.setSaturation(arg1);
    }

    Ptr<TonemapDurand> _createTonemapDurand_wrapper(float arg1, float arg2, float arg3, float arg4, float arg5) {
        return cv::createTonemapDurand(arg1, arg2, arg3, arg4, arg5);
    }

    void TonemapDurand_setContrast_wrapper(cv::TonemapDurand& arg0 , float arg1) {
        return arg0.setContrast(arg1);
    }

    void ml_TrainData_setTrainTestSplit_wrapper(cv::ml::TrainData& arg0 , int arg1, bool arg2) {
        return arg0.setTrainTestSplit(arg1, arg2);
    }

    void ml_TrainData_setTrainTestSplitRatio_wrapper(cv::ml::TrainData& arg0 , double arg1, bool arg2) {
        return arg0.setTrainTestSplitRatio(arg1, arg2);
    }

    void ml_TrainData_getValues_wrapper(cv::ml::TrainData& arg0 , int arg1, const cv::Mat& arg2, float* arg3) {
        return arg0.getValues(arg1, arg2, arg3);
    }

    Ptr<TrainData> ml_TrainData_create_wrapper(const cv::Mat& arg1, int arg2, const cv::Mat& arg3, const cv::Mat& arg4, const cv::Mat& arg5, const cv::Mat& arg6, const cv::Mat& arg7) {
        return cv::ml::TrainData::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void ml_TrainData_getSample_wrapper(cv::ml::TrainData& arg0 , const cv::Mat& arg1, int arg2, float* arg3) {
        return arg0.getSample(arg1, arg2, arg3);
    }

    int ml_TrainData_getCatCount_wrapper(cv::ml::TrainData& arg0 , int arg1) {
        return arg0.getCatCount(arg1);
    }

    Mat ml_TrainData_getSubVector_wrapper(cv::ml::TrainData& arg0 , const Mat& arg1, const Mat& arg2) {
        return arg0.getSubVector(arg1, arg2);
    }

    Mat ml_TrainData_getTrainSamples_wrapper(cv::ml::TrainData& arg0 , int arg1, bool arg2, bool arg3) {
        return arg0.getTrainSamples(arg1, arg2, arg3);
    }

    void MSER_setMinArea_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setMinArea(arg1);
    }

    Ptr<MSER> MSER_create_wrapper(int arg1, int arg2, int arg3, double arg4, double arg5, int arg6, double arg7, double arg8, int arg9) {
        return cv::MSER::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void MSER_setMaxArea_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setMaxArea(arg1);
    }

    void MSER_setPass2Only_wrapper(cv::MSER& arg0 , bool arg1) {
        return arg0.setPass2Only(arg1);
    }

    void MSER_detectRegions_wrapper(cv::MSER& arg0 , const cv::Mat& arg1, std::vector<std::vector<Point> >& arg2, std::vector<Rect>& arg3) {
        return arg0.detectRegions(arg1, arg2, arg3);
    }

    void MSER_setDelta_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setDelta(arg1);
    }

    Ptr<BackgroundSubtractorKNN> _createBackgroundSubtractorKNN_wrapper(int arg1, double arg2, bool arg3) {
        return cv::createBackgroundSubtractorKNN(arg1, arg2, arg3);
    }

    void BackgroundSubtractorKNN_setDetectShadows_wrapper(cv::BackgroundSubtractorKNN& arg0 , bool arg1) {
        return arg0.setDetectShadows(arg1);
    }

    void BackgroundSubtractorKNN_setDist2Threshold_wrapper(cv::BackgroundSubtractorKNN& arg0 , double arg1) {
        return arg0.setDist2Threshold(arg1);
    }

    void BackgroundSubtractorKNN_setkNNSamples_wrapper(cv::BackgroundSubtractorKNN& arg0 , int arg1) {
        return arg0.setkNNSamples(arg1);
    }

    void BackgroundSubtractorKNN_setShadowValue_wrapper(cv::BackgroundSubtractorKNN& arg0 , int arg1) {
        return arg0.setShadowValue(arg1);
    }

    void BackgroundSubtractorKNN_setHistory_wrapper(cv::BackgroundSubtractorKNN& arg0 , int arg1) {
        return arg0.setHistory(arg1);
    }

    void BackgroundSubtractorKNN_setNSamples_wrapper(cv::BackgroundSubtractorKNN& arg0 , int arg1) {
        return arg0.setNSamples(arg1);
    }

    void BackgroundSubtractorKNN_setShadowThreshold_wrapper(cv::BackgroundSubtractorKNN& arg0 , double arg1) {
        return arg0.setShadowThreshold(arg1);
    }

    void CLAHE_setTilesGridSize_wrapper(cv::CLAHE& arg0 , Size arg1) {
        return arg0.setTilesGridSize(arg1);
    }

    Ptr<CLAHE> _createCLAHE_wrapper(double arg1, Size arg2) {
        return cv::createCLAHE(arg1, arg2);
    }

    void CLAHE_setClipLimit_wrapper(cv::CLAHE& arg0 , double arg1) {
        return arg0.setClipLimit(arg1);
    }

    void CLAHE_apply_wrapper(cv::CLAHE& arg0 , const cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.apply(arg1, arg2);
    }

    Ptr<SimpleBlobDetector> SimpleBlobDetector_create_wrapper(const SimpleBlobDetector::Params & arg1) {
        return cv::SimpleBlobDetector::create(arg1);
    }

    void ml_SVM_setCoef0_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setCoef0(arg1);
    }

    void ml_SVM_setNu_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setNu(arg1);
    }

    void ml_SVM_setType_wrapper(cv::ml::SVM& arg0 , int arg1) {
        return arg0.setType(arg1);
    }

    void ml_SVM_setDegree_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setDegree(arg1);
    }

    void ml_SVM_setC_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setC(arg1);
    }

    void ml_SVM_setTermCriteria_wrapper(cv::ml::SVM& arg0 , const cv::TermCriteria & arg1) {
        return arg0.setTermCriteria(arg1);
    }

    void ml_SVM_setGamma_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setGamma(arg1);
    }

    void ml_SVM_setP_wrapper(cv::ml::SVM& arg0 , double arg1) {
        return arg0.setP(arg1);
    }

    double ml_SVM_getDecisionFunction_wrapper(cv::ml::SVM& arg0 , int arg1, cv::Mat& arg2, cv::Mat& arg3) {
        return arg0.getDecisionFunction(arg1, arg2, arg3);
    }

    void ml_SVM_setKernel_wrapper(cv::ml::SVM& arg0 , int arg1) {
        return arg0.setKernel(arg1);
    }

    void ml_SVM_setClassWeights_wrapper(cv::ml::SVM& arg0 , const cv::Mat & arg1) {
        return arg0.setClassWeights(arg1);
    }

    Ptr<HistogramCostExtractor> _createEMDL1HistogramCostExtractor_wrapper(int arg1, float arg2) {
        return cv::createEMDL1HistogramCostExtractor(arg1, arg2);
    }

    bool CascadeClassifier_load_wrapper(cv::CascadeClassifier& arg0 , const std::string& arg1) {
        return arg0.load(arg1);
    }

    bool CascadeClassifier_convert_wrapper(cv::CascadeClassifier& arg0 , const std::string& arg1, const std::string& arg2) {
        return arg0.convert(arg1, arg2);
    }

    bool CascadeClassifier_read_wrapper(cv::CascadeClassifier& arg0 , const FileNode& arg1) {
        return arg0.read(arg1);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, std::vector<int>& arg3, double arg4, int arg5, int arg6, Size arg7, Size arg8) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, std::vector<int>& arg3, std::vector<double>& arg4, double arg5, int arg6, int arg7, Size arg8, Size arg9, bool arg10) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, double arg3, int arg4, int arg5, Size arg6, Size arg7) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void TonemapReinhard_setColorAdaptation_wrapper(cv::TonemapReinhard& arg0 , float arg1) {
        return arg0.setColorAdaptation(arg1);
    }

    void TonemapReinhard_setLightAdaptation_wrapper(cv::TonemapReinhard& arg0 , float arg1) {
        return arg0.setLightAdaptation(arg1);
    }

    Ptr<TonemapReinhard> _createTonemapReinhard_wrapper(float arg1, float arg2, float arg3, float arg4) {
        return cv::createTonemapReinhard(arg1, arg2, arg3, arg4);
    }

    void TonemapReinhard_setIntensity_wrapper(cv::TonemapReinhard& arg0 , float arg1) {
        return arg0.setIntensity(arg1);
    }

    void TonemapDrago_setBias_wrapper(cv::TonemapDrago& arg0 , float arg1) {
        return arg0.setBias(arg1);
    }

    Ptr<TonemapDrago> _createTonemapDrago_wrapper(float arg1, float arg2, float arg3) {
        return cv::createTonemapDrago(arg1, arg2, arg3);
    }

    void TonemapDrago_setSaturation_wrapper(cv::TonemapDrago& arg0 , float arg1) {
        return arg0.setSaturation(arg1);
    }

    void HausdorffDistanceExtractor_setDistanceFlag_wrapper(cv::HausdorffDistanceExtractor& arg0 , int arg1) {
        return arg0.setDistanceFlag(arg1);
    }

    void HausdorffDistanceExtractor_setRankProportion_wrapper(cv::HausdorffDistanceExtractor& arg0 , float arg1) {
        return arg0.setRankProportion(arg1);
    }

    Ptr<HausdorffDistanceExtractor> _createHausdorffDistanceExtractor_wrapper(int arg1, float arg2) {
        return cv::createHausdorffDistanceExtractor(arg1, arg2);
    }

    Mat BOWTrainer_cluster_wrapper(cv::BOWTrainer& arg0 ) {
        return arg0.cluster();
    }

    Mat BOWTrainer_cluster_wrapper1(cv::BOWTrainer& arg0 , const Mat& arg1) {
        return arg0.cluster(arg1);
    }

    void BOWTrainer_add_wrapper(cv::BOWTrainer& arg0 , const Mat& arg1) {
        return arg0.add(arg1);
    }

    void ORB_setEdgeThreshold_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setEdgeThreshold(arg1);
    }

    void ORB_setFirstLevel_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setFirstLevel(arg1);
    }

    Ptr<ORB> ORB_create_wrapper(int arg1, float arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int arg9) {
        return cv::ORB::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    void ORB_setMaxFeatures_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setMaxFeatures(arg1);
    }

    void ORB_setNLevels_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setNLevels(arg1);
    }

    void ORB_setFastThreshold_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setFastThreshold(arg1);
    }

    void ORB_setPatchSize_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setPatchSize(arg1);
    }

    void ORB_setWTA_K_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setWTA_K(arg1);
    }

    void ORB_setScaleFactor_wrapper(cv::ORB& arg0 , double arg1) {
        return arg0.setScaleFactor(arg1);
    }

    void ORB_setScoreType_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setScoreType(arg1);
    }

    bool ml_EM_trainE_wrapper(cv::ml::EM& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, cv::Mat& arg5, cv::Mat& arg6, cv::Mat& arg7) {
        return arg0.trainE(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    void ml_EM_getCovs_wrapper(cv::ml::EM& arg0 , std::vector<Mat>& arg1) {
        return arg0.getCovs(arg1);
    }

    void ml_EM_setClustersNumber_wrapper(cv::ml::EM& arg0 , int arg1) {
        return arg0.setClustersNumber(arg1);
    }

    bool ml_EM_trainM_wrapper(cv::ml::EM& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, cv::Mat& arg5) {
        return arg0.trainM(arg1, arg2, arg3, arg4, arg5);
    }

    Vec2d ml_EM_predict2_wrapper(cv::ml::EM& arg0 , const cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.predict2(arg1, arg2);
    }

    void ml_EM_setTermCriteria_wrapper(cv::ml::EM& arg0 , const TermCriteria & arg1) {
        return arg0.setTermCriteria(arg1);
    }

    void ml_EM_setCovarianceMatrixType_wrapper(cv::ml::EM& arg0 , int arg1) {
        return arg0.setCovarianceMatrixType(arg1);
    }

    bool ml_EM_trainEM_wrapper(cv::ml::EM& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4) {
        return arg0.trainEM(arg1, arg2, arg3, arg4);
    }

}

EMSCRIPTEN_BINDINGS(testBinding) {
    function("CamShift", select_overload<RotatedRect(const cv::Mat&, Rect&, TermCriteria)>(&Wrappers::CamShift_wrapper));

    function("Canny", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, bool)>(&Wrappers::Canny_wrapper));

    function("GaussianBlur", select_overload<void(const cv::Mat&, cv::Mat&, Size, double, double, int)>(&Wrappers::GaussianBlur_wrapper));

    function("HoughCircles", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, double, double, int, int)>(&Wrappers::HoughCircles_wrapper));

    function("HoughLines", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, double, double, double, double)>(&Wrappers::HoughLines_wrapper));

    function("HoughLinesP", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, double, double)>(&Wrappers::HoughLinesP_wrapper));

    function("HuMoments", select_overload<void(const Moments&, cv::Mat&)>(&Wrappers::HuMoments_wrapper));

    function("LUT", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::LUT_wrapper));

    function("Laplacian", select_overload<void(const cv::Mat&, cv::Mat&, int, int, double, double, int)>(&Wrappers::Laplacian_wrapper));

    function("Mahalanobis", select_overload<double(const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::Mahalanobis_wrapper));

    function("PCABackProject", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::PCABackProject_wrapper));

    function("PCACompute", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int)>(&Wrappers::PCACompute_wrapper));

    function("PCACompute1", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, double)>(&Wrappers::PCACompute_wrapper1));

    function("PCAProject", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::PCAProject_wrapper));

    function("PSNR", select_overload<double(const cv::Mat&, const cv::Mat&)>(&Wrappers::PSNR_wrapper));

    function("SVBackSubst", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::SVBackSubst_wrapper));

    function("SVDecomp", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int)>(&Wrappers::SVDecomp_wrapper));

    function("Scharr", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, double, double, int)>(&Wrappers::Scharr_wrapper));

    function("Sobel", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, int, double, double, int)>(&Wrappers::Sobel_wrapper));

    function("absdiff", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::absdiff_wrapper));

    function("accumulate", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::accumulate_wrapper));

    function("accumulateProduct", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::accumulateProduct_wrapper));

    function("accumulateSquare", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::accumulateSquare_wrapper));

    function("accumulateWeighted", select_overload<void(const cv::Mat&, cv::Mat&, double, const cv::Mat&)>(&Wrappers::accumulateWeighted_wrapper));

    function("adaptiveThreshold", select_overload<void(const cv::Mat&, cv::Mat&, double, int, int, int, double)>(&Wrappers::adaptiveThreshold_wrapper));

    function("add", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&, int)>(&Wrappers::add_wrapper));

    function("addWeighted", select_overload<void(const cv::Mat&, double, const cv::Mat&, double, double, cv::Mat&, int)>(&Wrappers::addWeighted_wrapper));

    function("applyColorMap", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::applyColorMap_wrapper));

    function("approxPolyDP", select_overload<void(const cv::Mat&, cv::Mat&, double, bool)>(&Wrappers::approxPolyDP_wrapper));

    function("arcLength", select_overload<double(const cv::Mat&, bool)>(&Wrappers::arcLength_wrapper));

    function("arrowedLine", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int, double)>(&Wrappers::arrowedLine_wrapper));

    function("batchDistance", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, cv::Mat&, int, int, const cv::Mat&, int, bool)>(&Wrappers::batchDistance_wrapper));

    function("bilateralFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, int)>(&Wrappers::bilateralFilter_wrapper));

    function("bitwise_and", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::bitwise_and_wrapper));

    function("bitwise_not", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::bitwise_not_wrapper));

    function("bitwise_or", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::bitwise_or_wrapper));

    function("bitwise_xor", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::bitwise_xor_wrapper));

    function("blur", select_overload<void(const cv::Mat&, cv::Mat&, Size, Point, int)>(&Wrappers::blur_wrapper));

    function("borderInterpolate", select_overload<int(int, int, int)>(&Wrappers::borderInterpolate_wrapper));

    function("boundingRect", select_overload<Rect(const cv::Mat&)>(&Wrappers::boundingRect_wrapper));

    function("boxFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, Size, Point, bool, int)>(&Wrappers::boxFilter_wrapper));

    function("boxPoints", select_overload<void(RotatedRect, cv::Mat&)>(&Wrappers::boxPoints_wrapper));

    function("buildOpticalFlowPyramid", select_overload<int(const cv::Mat&, std::vector<cv::Mat>&, Size, int, bool, int, int, bool)>(&Wrappers::buildOpticalFlowPyramid_wrapper));

    function("calcBackProject", select_overload<void(const std::vector<cv::Mat>&, const std::vector<int>&, const cv::Mat&, cv::Mat&, const std::vector<float>&, double)>(&Wrappers::calcBackProject_wrapper));

    function("calcCovarMatrix", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::calcCovarMatrix_wrapper));

    function("calcHist", select_overload<void(const std::vector<cv::Mat>&, const std::vector<int>&, const cv::Mat&, cv::Mat&, const std::vector<int>&, const std::vector<float>&, bool)>(&Wrappers::calcHist_wrapper));

    function("calcOpticalFlowFarneback", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int, int, int, int, double, int)>(&Wrappers::calcOpticalFlowFarneback_wrapper));

    function("calcOpticalFlowPyrLK", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, Size, int, TermCriteria, int, double)>(&Wrappers::calcOpticalFlowPyrLK_wrapper));

    function("cartToPolar", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, bool)>(&Wrappers::cartToPolar_wrapper));

    function("circle", select_overload<void(cv::Mat&, Point, int, const Scalar&, int, int, int)>(&Wrappers::circle_wrapper));

    function("clipLine", select_overload<bool(Rect, Point&, Point&)>(&Wrappers::clipLine_wrapper));

    function("colorChange", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, float, float, float)>(&Wrappers::colorChange_wrapper));

    function("compare", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int)>(&Wrappers::compare_wrapper));

    function("compareHist", select_overload<double(const cv::Mat&, const cv::Mat&, int)>(&Wrappers::compareHist_wrapper));

    function("completeSymm", select_overload<void(cv::Mat&, bool)>(&Wrappers::completeSymm_wrapper));

    function("connectedComponents", select_overload<int(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::connectedComponents_wrapper));

    function("connectedComponentsWithStats", select_overload<int(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::connectedComponentsWithStats_wrapper));

    function("contourArea", select_overload<double(const cv::Mat&, bool)>(&Wrappers::contourArea_wrapper));

    function("convertMaps", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, int, bool)>(&Wrappers::convertMaps_wrapper));

    function("convertScaleAbs", select_overload<void(const cv::Mat&, cv::Mat&, double, double)>(&Wrappers::convertScaleAbs_wrapper));

    function("convexHull", select_overload<void(const cv::Mat&, cv::Mat&, bool, bool)>(&Wrappers::convexHull_wrapper));

    function("convexityDefects", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::convexityDefects_wrapper));

    function("copyMakeBorder", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, int, int, const Scalar&)>(&Wrappers::copyMakeBorder_wrapper));

    function("cornerEigenValsAndVecs", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::cornerEigenValsAndVecs_wrapper));

    function("cornerHarris", select_overload<void(const cv::Mat&, cv::Mat&, int, int, double, int)>(&Wrappers::cornerHarris_wrapper));

    function("cornerMinEigenVal", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::cornerMinEigenVal_wrapper));

    function("cornerSubPix", select_overload<void(const cv::Mat&, cv::Mat&, Size, Size, TermCriteria)>(&Wrappers::cornerSubPix_wrapper));

    function("countNonZero", select_overload<int(const cv::Mat&)>(&Wrappers::countNonZero_wrapper));

    function("createHanningWindow", select_overload<void(cv::Mat&, Size, int)>(&Wrappers::createHanningWindow_wrapper));

    function("cvtColor", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::cvtColor_wrapper));

    function("dct", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::dct_wrapper));

    function("decolor", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&)>(&Wrappers::decolor_wrapper));

    function("demosaicing", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::demosaicing_wrapper));

    function("denoise_TVL1", select_overload<void(const std::vector<Mat>&, Mat&, double, int)>(&Wrappers::denoise_TVL1_wrapper));

    function("detailEnhance", select_overload<void(const cv::Mat&, cv::Mat&, float, float)>(&Wrappers::detailEnhance_wrapper));

    function("determinant", select_overload<double(const cv::Mat&)>(&Wrappers::determinant_wrapper));

    function("dft", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::dft_wrapper));

    function("dilate", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Point, int, int, const Scalar&)>(&Wrappers::dilate_wrapper));

    function("distanceTransform", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::distanceTransform_wrapper));

    function("distanceTransformWithLabels", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::distanceTransform_wrapper));

    function("divide", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int)>(&Wrappers::divide_wrapper));

    function("divide1", select_overload<void(double, const cv::Mat&, cv::Mat&, int)>(&Wrappers::divide_wrapper1));

    function("drawContours", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, int, const Scalar&, int, int, const cv::Mat&, int, Point)>(&Wrappers::drawContours_wrapper));

    function("drawKeypoints", select_overload<void(const cv::Mat&, const std::vector<KeyPoint>&, cv::Mat&, const Scalar&, int)>(&Wrappers::drawKeypoints_wrapper));

    function("drawMarker", select_overload<void(Mat&, Point, const Scalar&, int, int, int, int)>(&Wrappers::drawMarker_wrapper));

    function("drawMatches", select_overload<void(const cv::Mat&, const std::vector<KeyPoint>&, const cv::Mat&, const std::vector<KeyPoint>&, const std::vector<DMatch>&, cv::Mat&, const Scalar&, const Scalar&, const std::vector<char>&, int)>(&Wrappers::drawMatches_wrapper));

    function("drawMatchesKnn", select_overload<void(const cv::Mat&, const std::vector<KeyPoint>&, const cv::Mat&, const std::vector<KeyPoint>&, const std::vector<std::vector<DMatch> >&, cv::Mat&, const Scalar&, const Scalar&, const std::vector<std::vector<char> >&, int)>(&Wrappers::drawMatches_wrapper));

    function("edgePreservingFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, float, float)>(&Wrappers::edgePreservingFilter_wrapper));

    function("eigen", select_overload<bool(const cv::Mat&, cv::Mat&, cv::Mat&)>(&Wrappers::eigen_wrapper));

    function("ellipse", select_overload<void(cv::Mat&, Point, Size, double, double, double, const Scalar&, int, int, int)>(&Wrappers::ellipse_wrapper));

    function("ellipse1", select_overload<void(cv::Mat&, const RotatedRect&, const Scalar&, int, int)>(&Wrappers::ellipse_wrapper1));

    function("ellipse2Poly", select_overload<void(Point, Size, int, int, int, int, std::vector<Point>&)>(&Wrappers::ellipse2Poly_wrapper));

    function("equalizeHist", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::equalizeHist_wrapper));

    function("erode", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Point, int, int, const Scalar&)>(&Wrappers::erode_wrapper));

    function("estimateRigidTransform", select_overload<Mat(const cv::Mat&, const cv::Mat&, bool)>(&Wrappers::estimateRigidTransform_wrapper));

    function("exp", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::exp_wrapper));

    function("extractChannel", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::extractChannel_wrapper));

    function("fastNlMeansDenoising", select_overload<void(const cv::Mat&, cv::Mat&, float, int, int)>(&Wrappers::fastNlMeansDenoising_wrapper));

    function("fastNlMeansDenoising1", select_overload<void(const cv::Mat&, cv::Mat&, const std::vector<float>&, int, int, int)>(&Wrappers::fastNlMeansDenoising_wrapper1));

    function("fastNlMeansDenoisingColored", select_overload<void(const cv::Mat&, cv::Mat&, float, float, int, int)>(&Wrappers::fastNlMeansDenoisingColored_wrapper));

    function("fastNlMeansDenoisingColoredMulti", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&, int, int, float, float, int, int)>(&Wrappers::fastNlMeansDenoisingColoredMulti_wrapper));

    function("fastNlMeansDenoisingMulti", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&, int, int, float, int, int)>(&Wrappers::fastNlMeansDenoisingMulti_wrapper));

    function("fastNlMeansDenoisingMulti1", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&, int, int, const std::vector<float>&, int, int, int)>(&Wrappers::fastNlMeansDenoisingMulti_wrapper1));

    function("fillConvexPoly", select_overload<void(cv::Mat&, const cv::Mat&, const Scalar&, int, int)>(&Wrappers::fillConvexPoly_wrapper));

    function("fillPoly", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, const Scalar&, int, int, Point)>(&Wrappers::fillPoly_wrapper));

    function("filter2D", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, Point, double, int)>(&Wrappers::filter2D_wrapper));

    function("findContours", select_overload<void(cv::Mat&, std::vector<cv::Mat>&, cv::Mat&, int, int, Point)>(&Wrappers::findContours_wrapper));

    function("findNonZero", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::findNonZero_wrapper));

    function("findTransformECC", select_overload<double(const cv::Mat&, const cv::Mat&, cv::Mat&, int, TermCriteria, const cv::Mat&)>(&Wrappers::findTransformECC_wrapper));

    function("fitEllipse", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::fitEllipse_wrapper));

    function("fitLine", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, double)>(&Wrappers::fitLine_wrapper));

    function("flip", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::flip_wrapper));

    function("gemm", select_overload<void(const cv::Mat&, const cv::Mat&, double, const cv::Mat&, double, cv::Mat&, int)>(&Wrappers::gemm_wrapper));

    function("getAffineTransform", select_overload<Mat(const cv::Mat&, const cv::Mat&)>(&Wrappers::getAffineTransform_wrapper));

    function("getDefaultNewCameraMatrix", select_overload<Mat(const cv::Mat&, Size, bool)>(&Wrappers::getDefaultNewCameraMatrix_wrapper));

    function("getDerivKernels", select_overload<void(cv::Mat&, cv::Mat&, int, int, int, bool, int)>(&Wrappers::getDerivKernels_wrapper));

    function("getGaborKernel", select_overload<Mat(Size, double, double, double, double, double, int)>(&Wrappers::getGaborKernel_wrapper));

    function("getGaussianKernel", select_overload<Mat(int, double, int)>(&Wrappers::getGaussianKernel_wrapper));

    function("getOptimalDFTSize", select_overload<int(int)>(&Wrappers::getOptimalDFTSize_wrapper));

    function("getPerspectiveTransform", select_overload<Mat(const cv::Mat&, const cv::Mat&)>(&Wrappers::getPerspectiveTransform_wrapper));

    function("getRectSubPix", select_overload<void(const cv::Mat&, Size, Point2f, cv::Mat&, int)>(&Wrappers::getRectSubPix_wrapper));

    function("getRotationMatrix2D", select_overload<Mat(Point2f, double, double)>(&Wrappers::getRotationMatrix2D_wrapper));

    function("getStructuringElement", select_overload<Mat(int, Size, Point)>(&Wrappers::getStructuringElement_wrapper));

    function("getTextSize", select_overload<Size(const std::string&, int, double, int, int*)>(&Wrappers::getTextSize_wrapper), allow_raw_pointers());

    function("goodFeaturesToTrack", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, const cv::Mat&, int, bool, double)>(&Wrappers::goodFeaturesToTrack_wrapper));

    function("grabCut", select_overload<void(const cv::Mat&, cv::Mat&, Rect, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::grabCut_wrapper));

    function("groupRectangles", select_overload<void(std::vector<Rect>&, std::vector<int>&, int, double)>(&Wrappers::groupRectangles_wrapper));

    function("hconcat", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&)>(&Wrappers::hconcat_wrapper));

    function("idct", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::idct_wrapper));

    function("idft", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::idft_wrapper));

    function("illuminationChange", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, float, float)>(&Wrappers::illuminationChange_wrapper));

    function("imdecode", select_overload<Mat(const cv::Mat&, int)>(&Wrappers::imdecode_wrapper));

    function("imencode", select_overload<bool(const std::string&, const cv::Mat&, std::vector<uchar>&, const std::vector<int>&)>(&Wrappers::imencode_wrapper));

    function("imread", select_overload<Mat(const std::string&, int)>(&Wrappers::imread_wrapper));

    function("imreadmulti", select_overload<bool(const std::string&, std::vector<Mat>&, int)>(&Wrappers::imreadmulti_wrapper));

    function("imwrite", select_overload<bool(const std::string&, const cv::Mat&, const std::vector<int>&)>(&Wrappers::imwrite_wrapper));

    function("inRange", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::inRange_wrapper));

    function("initUndistortRectifyMap", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, Size, int, cv::Mat&, cv::Mat&)>(&Wrappers::initUndistortRectifyMap_wrapper));

    function("initWideAngleProjMap", select_overload<float(const cv::Mat&, const cv::Mat&, Size, int, int, cv::Mat&, cv::Mat&, int, double)>(&Wrappers::initWideAngleProjMap_wrapper));

    function("inpaint", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int)>(&Wrappers::inpaint_wrapper));

    function("insertChannel", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::insertChannel_wrapper));

    function("integral", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::integral_wrapper));

    function("integral2", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    function("integral3", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    function("intersectConvexConvex", select_overload<float(const cv::Mat&, const cv::Mat&, cv::Mat&, bool)>(&Wrappers::intersectConvexConvex_wrapper));

    function("invert", select_overload<double(const cv::Mat&, cv::Mat&, int)>(&Wrappers::invert_wrapper));

    function("invertAffineTransform", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::invertAffineTransform_wrapper));

    function("isContourConvex", select_overload<bool(const cv::Mat&)>(&Wrappers::isContourConvex_wrapper));

    function("kmeans", select_overload<double(const cv::Mat&, int, cv::Mat&, TermCriteria, int, int, cv::Mat&)>(&Wrappers::kmeans_wrapper));

    function("line", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int)>(&Wrappers::line_wrapper));

    function("linearPolar", select_overload<void(const cv::Mat&, cv::Mat&, Point2f, double, int)>(&Wrappers::linearPolar_wrapper));

    function("log", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::log_wrapper));

    function("logPolar", select_overload<void(const cv::Mat&, cv::Mat&, Point2f, double, int)>(&Wrappers::logPolar_wrapper));

    function("magnitude", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::magnitude_wrapper));

    function("matchShapes", select_overload<double(const cv::Mat&, const cv::Mat&, int, double)>(&Wrappers::matchShapes_wrapper));

    function("matchTemplate", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, const cv::Mat&)>(&Wrappers::matchTemplate_wrapper));

    function("max", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::max_wrapper));

    function("mean", select_overload<Scalar(const cv::Mat&, const cv::Mat&)>(&Wrappers::mean_wrapper));

    function("meanShift", select_overload<int(const cv::Mat&, Rect&, TermCriteria)>(&Wrappers::meanShift_wrapper));

    function("meanStdDev", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::meanStdDev_wrapper));

    function("medianBlur", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::medianBlur_wrapper));

    function("merge", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&)>(&Wrappers::merge_wrapper));

    function("min", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::min_wrapper));

    function("minAreaRect", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::minAreaRect_wrapper));

    function("minEnclosingTriangle", select_overload<double(const cv::Mat&, cv::Mat&)>(&Wrappers::minEnclosingTriangle_wrapper));

    function("mixChannels", select_overload<void(const std::vector<cv::Mat>&, InputOutputArrayOfArrays, const std::vector<int>&)>(&Wrappers::mixChannels_wrapper));

    function("moments", select_overload<Moments(const cv::Mat&, bool)>(&Wrappers::moments_wrapper));

    function("morphologyEx", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, Point, int, int, const Scalar&)>(&Wrappers::morphologyEx_wrapper));

    function("mulSpectrums", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, bool)>(&Wrappers::mulSpectrums_wrapper));

    function("mulTransposed", select_overload<void(const cv::Mat&, cv::Mat&, bool, const cv::Mat&, double, int)>(&Wrappers::mulTransposed_wrapper));

    function("multiply", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int)>(&Wrappers::multiply_wrapper));

    function("norm", select_overload<double(const cv::Mat&, int, const cv::Mat&)>(&Wrappers::norm_wrapper));

    function("norm1", select_overload<double(const cv::Mat&, const cv::Mat&, int, const cv::Mat&)>(&Wrappers::norm_wrapper1));

    function("normalize", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, int, const cv::Mat&)>(&Wrappers::normalize_wrapper));

    function("patchNaNs", select_overload<void(cv::Mat&, double)>(&Wrappers::patchNaNs_wrapper));

    function("pencilSketch", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, float, float, float)>(&Wrappers::pencilSketch_wrapper));

    function("perspectiveTransform", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::perspectiveTransform_wrapper));

    function("phase", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, bool)>(&Wrappers::phase_wrapper));

    function("pointPolygonTest", select_overload<double(const cv::Mat&, Point2f, bool)>(&Wrappers::pointPolygonTest_wrapper));

    function("polarToCart", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, bool)>(&Wrappers::polarToCart_wrapper));

    function("polylines", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, bool, const Scalar&, int, int, int)>(&Wrappers::polylines_wrapper));

    function("pow", select_overload<void(const cv::Mat&, double, cv::Mat&)>(&Wrappers::pow_wrapper));

    function("preCornerDetect", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::preCornerDetect_wrapper));

    function("putText", select_overload<void(cv::Mat&, const std::string&, Point, int, double, Scalar, int, int, bool)>(&Wrappers::putText_wrapper));

    function("pyrDown", select_overload<void(const cv::Mat&, cv::Mat&, const Size&, int)>(&Wrappers::pyrDown_wrapper));

    function("pyrMeanShiftFiltering", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, TermCriteria)>(&Wrappers::pyrMeanShiftFiltering_wrapper));

    function("pyrUp", select_overload<void(const cv::Mat&, cv::Mat&, const Size&, int)>(&Wrappers::pyrUp_wrapper));

    function("randn", select_overload<void(cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::randn_wrapper));

    function("randu", select_overload<void(cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::randu_wrapper));

    function("rectangle", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int)>(&Wrappers::rectangle_wrapper));

    function("reduce", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::reduce_wrapper));

    function("remap", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, int, int, const Scalar&)>(&Wrappers::remap_wrapper));

    function("repeat", select_overload<void(const cv::Mat&, int, int, cv::Mat&)>(&Wrappers::repeat_wrapper));

    function("resize", select_overload<void(const cv::Mat&, cv::Mat&, Size, double, double, int)>(&Wrappers::resize_wrapper));

    function("rotatedRectangleIntersection", select_overload<int(const RotatedRect&, const RotatedRect&, cv::Mat&)>(&Wrappers::rotatedRectangleIntersection_wrapper));

    function("scaleAdd", select_overload<void(const cv::Mat&, double, const cv::Mat&, cv::Mat&)>(&Wrappers::scaleAdd_wrapper));

    function("seamlessClone", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, Point, cv::Mat&, int)>(&Wrappers::seamlessClone_wrapper));

    function("sepFilter2D", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, const cv::Mat&, Point, double, int)>(&Wrappers::sepFilter2D_wrapper));

    function("setIdentity", select_overload<void(cv::Mat&, const Scalar&)>(&Wrappers::setIdentity_wrapper));

    function("solve", select_overload<bool(const cv::Mat&, const cv::Mat&, cv::Mat&, int)>(&Wrappers::solve_wrapper));

    function("solveCubic", select_overload<int(const cv::Mat&, cv::Mat&)>(&Wrappers::solveCubic_wrapper));

    function("solvePoly", select_overload<double(const cv::Mat&, cv::Mat&, int)>(&Wrappers::solvePoly_wrapper));

    function("sort", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::sort_wrapper));

    function("sortIdx", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::sortIdx_wrapper));

    function("spatialGradient", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::spatialGradient_wrapper));

    function("split", select_overload<void(const cv::Mat&, std::vector<cv::Mat>&)>(&Wrappers::split_wrapper));

    function("sqrBoxFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, Size, Point, bool, int)>(&Wrappers::sqrBoxFilter_wrapper));

    function("sqrt", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::sqrt_wrapper));

    function("stylization", select_overload<void(const cv::Mat&, cv::Mat&, float, float)>(&Wrappers::stylization_wrapper));

    function("subtract", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&, int)>(&Wrappers::subtract_wrapper));

    function("sumElems", select_overload<Scalar(const cv::Mat&)>(&Wrappers::sum_wrapper));

    function("textureFlattening", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, float, float, int)>(&Wrappers::textureFlattening_wrapper));

    function("threshold", select_overload<double(const cv::Mat&, cv::Mat&, double, double, int)>(&Wrappers::threshold_wrapper));

    function("trace", select_overload<Scalar(const cv::Mat&)>(&Wrappers::trace_wrapper));

    function("transform", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::transform_wrapper));

    function("transpose", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::transpose_wrapper));

    function("undistort", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::undistort_wrapper));

    function("undistortPoints", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::undistortPoints_wrapper));

    function("vconcat", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&)>(&Wrappers::vconcat_wrapper));

    function("warpAffine", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Size, int, int, const Scalar&)>(&Wrappers::warpAffine_wrapper));

    function("warpPerspective", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Size, int, int, const Scalar&)>(&Wrappers::warpPerspective_wrapper));

    function("watershed", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::watershed_wrapper));

    function("finish", select_overload<void()>(&Wrappers::finish_wrapper));

    function("haveAmdBlas", select_overload<bool()>(&Wrappers::haveAmdBlas_wrapper));

    function("haveAmdFft", select_overload<bool()>(&Wrappers::haveAmdFft_wrapper));

    function("haveOpenCL", select_overload<bool()>(&Wrappers::haveOpenCL_wrapper));

    function("setUseOpenCL", select_overload<void(bool)>(&Wrappers::setUseOpenCL_wrapper));

    function("useOpenCL", select_overload<bool()>(&Wrappers::useOpenCL_wrapper));

    emscripten::class_<cv::MergeMertens ,base<MergeExposures>>("MergeMertens")
        .function("setExposureWeight", select_overload<void(cv::MergeMertens&,float)>(&Wrappers::MergeMertens_setExposureWeight_wrapper), pure_virtual())
        .function("getContrastWeight", select_overload<float()const>(&cv::MergeMertens::getContrastWeight), pure_virtual())
        .function("getExposureWeight", select_overload<float()const>(&cv::MergeMertens::getExposureWeight), pure_virtual())
        .function("process", select_overload<void(cv::MergeMertens&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::MergeMertens_process_wrapper), pure_virtual())
        .function("process1", select_overload<void(cv::MergeMertens&,const std::vector<cv::Mat>&,cv::Mat&)>(&Wrappers::MergeMertens_process_wrapper1), pure_virtual())
        .function("getSaturationWeight", select_overload<float()const>(&cv::MergeMertens::getSaturationWeight), pure_virtual())
        .function("setSaturationWeight", select_overload<void(cv::MergeMertens&,float)>(&Wrappers::MergeMertens_setSaturationWeight_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<MergeMertens>(float,float,float)>(&Wrappers::_createMergeMertens_wrapper))
        .function("setContrastWeight", select_overload<void(cv::MergeMertens&,float)>(&Wrappers::MergeMertens_setContrastWeight_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::MergeMertens>>("Ptr<MergeMertens>")
;

    emscripten::class_<cv::BackgroundSubtractorMOG2 ,base<BackgroundSubtractor>>("BackgroundSubtractorMOG2")
        .function("getDetectShadows", select_overload<bool()const>(&cv::BackgroundSubtractorMOG2::getDetectShadows), pure_virtual())
        .function("getVarInit", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getVarInit), pure_virtual())
        .constructor(select_overload<Ptr<BackgroundSubtractorMOG2>(int,double,bool)>(&Wrappers::_createBackgroundSubtractorMOG2_wrapper))
        .function("setVarMax", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setVarMax_wrapper), pure_virtual())
        .function("setBackgroundRatio", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setBackgroundRatio_wrapper), pure_virtual())
        .function("getShadowThreshold", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getShadowThreshold), pure_virtual())
        .function("getNMixtures", select_overload<int()const>(&cv::BackgroundSubtractorMOG2::getNMixtures), pure_virtual())
        .function("getComplexityReductionThreshold", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getComplexityReductionThreshold), pure_virtual())
        .function("setNMixtures", select_overload<void(cv::BackgroundSubtractorMOG2&,int)>(&Wrappers::BackgroundSubtractorMOG2_setNMixtures_wrapper), pure_virtual())
        .function("setVarInit", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setVarInit_wrapper), pure_virtual())
        .function("getShadowValue", select_overload<int()const>(&cv::BackgroundSubtractorMOG2::getShadowValue), pure_virtual())
        .function("setDetectShadows", select_overload<void(cv::BackgroundSubtractorMOG2&,bool)>(&Wrappers::BackgroundSubtractorMOG2_setDetectShadows_wrapper), pure_virtual())
        .function("getVarMin", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getVarMin), pure_virtual())
        .function("setVarMin", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setVarMin_wrapper), pure_virtual())
        .function("setHistory", select_overload<void(cv::BackgroundSubtractorMOG2&,int)>(&Wrappers::BackgroundSubtractorMOG2_setHistory_wrapper), pure_virtual())
        .function("setVarThreshold", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setVarThreshold_wrapper), pure_virtual())
        .function("getVarThresholdGen", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getVarThresholdGen), pure_virtual())
        .function("setShadowThreshold", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setShadowThreshold_wrapper), pure_virtual())
        .function("getVarMax", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getVarMax), pure_virtual())
        .function("getBackgroundRatio", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getBackgroundRatio), pure_virtual())
        .function("setVarThresholdGen", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setVarThresholdGen_wrapper), pure_virtual())
        .function("getHistory", select_overload<int()const>(&cv::BackgroundSubtractorMOG2::getHistory), pure_virtual())
        .function("setShadowValue", select_overload<void(cv::BackgroundSubtractorMOG2&,int)>(&Wrappers::BackgroundSubtractorMOG2_setShadowValue_wrapper), pure_virtual())
        .function("setComplexityReductionThreshold", select_overload<void(cv::BackgroundSubtractorMOG2&,double)>(&Wrappers::BackgroundSubtractorMOG2_setComplexityReductionThreshold_wrapper), pure_virtual())
        .function("getVarThreshold", select_overload<double()const>(&cv::BackgroundSubtractorMOG2::getVarThreshold), pure_virtual())
        .smart_ptr<Ptr<cv::BackgroundSubtractorMOG2>>("Ptr<BackgroundSubtractorMOG2>")
;

    emscripten::class_<cv::CalibrateRobertson ,base<CalibrateCRF>>("CalibrateRobertson")
        .function("getThreshold", select_overload<float()const>(&cv::CalibrateRobertson::getThreshold), pure_virtual())
        .function("getRadiance", select_overload<Mat()const>(&cv::CalibrateRobertson::getRadiance), pure_virtual())
        .function("setThreshold", select_overload<void(cv::CalibrateRobertson&,float)>(&Wrappers::CalibrateRobertson_setThreshold_wrapper), pure_virtual())
        .function("setMaxIter", select_overload<void(cv::CalibrateRobertson&,int)>(&Wrappers::CalibrateRobertson_setMaxIter_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<CalibrateRobertson>(int,float)>(&Wrappers::_createCalibrateRobertson_wrapper))
        .function("getMaxIter", select_overload<int()const>(&cv::CalibrateRobertson::getMaxIter), pure_virtual())
        .smart_ptr<Ptr<cv::CalibrateRobertson>>("Ptr<CalibrateRobertson>")
;

    emscripten::class_<cv::HOGDescriptor >("HOGDescriptor")
        .function("load", select_overload<bool(cv::HOGDescriptor&,const std::string&,const std::string&)>(&Wrappers::HOGDescriptor_load_wrapper))
        .function("detect", select_overload<void(cv::HOGDescriptor&,const Mat&,std::vector<Point>&,std::vector<double>&,double,Size,Size,const std::vector<Point>&)>(&Wrappers::HOGDescriptor_detect_wrapper))
        .constructor<>()
        .constructor<Size, Size, Size, Size, int, int, double, int, double, bool, int, bool>()
        .constructor<const String&>()
        .class_function("getDefaultPeopleDetector", select_overload<std::vector<float>()>(&cv::HOGDescriptor::getDefaultPeopleDetector))
        .function("computeGradient", select_overload<void(cv::HOGDescriptor&,const Mat&,Mat&,Mat&,Size,Size)>(&Wrappers::HOGDescriptor_computeGradient_wrapper))
        .class_function("getDaimlerPeopleDetector", select_overload<std::vector<float>()>(&cv::HOGDescriptor::getDaimlerPeopleDetector))
        .function("checkDetectorSize", select_overload<bool()const>(&cv::HOGDescriptor::checkDetectorSize))
        .function("getWinSigma", select_overload<double()const>(&cv::HOGDescriptor::getWinSigma))
        .function("compute", select_overload<void(cv::HOGDescriptor&,const cv::Mat&,std::vector<float>&,Size,Size,const std::vector<Point>&)>(&Wrappers::HOGDescriptor_compute_wrapper))
        .function("save", select_overload<void(cv::HOGDescriptor&,const std::string&,const std::string&)>(&Wrappers::HOGDescriptor_save_wrapper))
        .function("setSVMDetector", select_overload<void(cv::HOGDescriptor&,const cv::Mat&)>(&Wrappers::HOGDescriptor_setSVMDetector_wrapper))
        .function("getDescriptorSize", select_overload<size_t()const>(&cv::HOGDescriptor::getDescriptorSize))
        .function("detectMultiScale", select_overload<void(cv::HOGDescriptor&,const cv::Mat&,std::vector<Rect>&,std::vector<double>&,double,Size,Size,double,double,bool)>(&Wrappers::HOGDescriptor_detectMultiScale_wrapper))
        .property("winSize", &cv::HOGDescriptor::winSize)
        .property("blockSize", &cv::HOGDescriptor::blockSize)
        .property("blockStride", &cv::HOGDescriptor::blockStride)
        .property("cellSize", &cv::HOGDescriptor::cellSize)
        .property("nbins", &cv::HOGDescriptor::nbins)
        .property("derivAperture", &cv::HOGDescriptor::derivAperture)
        .property("winSigma", &cv::HOGDescriptor::winSigma)
        .property("histogramNormType", &cv::HOGDescriptor::histogramNormType)
        .property("L2HysThreshold", &cv::HOGDescriptor::L2HysThreshold)
        .property("gammaCorrection", &cv::HOGDescriptor::gammaCorrection)
        .property("svmDetector", &cv::HOGDescriptor::svmDetector)
        .property("nlevels", &cv::HOGDescriptor::nlevels)
        .property("signedGradient", &cv::HOGDescriptor::signedGradient);

    emscripten::class_<cv::ml::LogisticRegression ,base<ml::StatModel>>("ml_LogisticRegression")
        .function("getTermCriteria", select_overload<TermCriteria()const>(&cv::ml::LogisticRegression::getTermCriteria), pure_virtual())
        .function("setTrainMethod", select_overload<void(cv::ml::LogisticRegression&,int)>(&Wrappers::ml_LogisticRegression_setTrainMethod_wrapper), pure_virtual())
        .function("getIterations", select_overload<int()const>(&cv::ml::LogisticRegression::getIterations), pure_virtual())
        .function("getTrainMethod", select_overload<int()const>(&cv::ml::LogisticRegression::getTrainMethod), pure_virtual())
        .function("predict", select_overload<float(cv::ml::LogisticRegression&,const cv::Mat&,cv::Mat&,int)>(&Wrappers::ml_LogisticRegression_predict_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<LogisticRegression>()>(&cv::ml::LogisticRegression::create))
        .function("getLearningRate", select_overload<double()const>(&cv::ml::LogisticRegression::getLearningRate), pure_virtual())
        .function("setTermCriteria", select_overload<void(cv::ml::LogisticRegression&,TermCriteria)>(&Wrappers::ml_LogisticRegression_setTermCriteria_wrapper), pure_virtual())
        .function("setMiniBatchSize", select_overload<void(cv::ml::LogisticRegression&,int)>(&Wrappers::ml_LogisticRegression_setMiniBatchSize_wrapper), pure_virtual())
        .function("setIterations", select_overload<void(cv::ml::LogisticRegression&,int)>(&Wrappers::ml_LogisticRegression_setIterations_wrapper), pure_virtual())
        .function("setLearningRate", select_overload<void(cv::ml::LogisticRegression&,double)>(&Wrappers::ml_LogisticRegression_setLearningRate_wrapper), pure_virtual())
        .function("getRegularization", select_overload<int()const>(&cv::ml::LogisticRegression::getRegularization), pure_virtual())
        .function("getMiniBatchSize", select_overload<int()const>(&cv::ml::LogisticRegression::getMiniBatchSize), pure_virtual())
        .function("setRegularization", select_overload<void(cv::ml::LogisticRegression&,int)>(&Wrappers::ml_LogisticRegression_setRegularization_wrapper), pure_virtual())
        .function("get_learnt_thetas", select_overload<Mat()const>(&cv::ml::LogisticRegression::get_learnt_thetas), pure_virtual())
        .smart_ptr<Ptr<cv::ml::LogisticRegression>>("Ptr<ml_LogisticRegression>")
;

    emscripten::class_<cv::NormHistogramCostExtractor ,base<HistogramCostExtractor>>("NormHistogramCostExtractor")
        .constructor(select_overload<Ptr<HistogramCostExtractor>(int,int,float)>(&Wrappers::_createNormHistogramCostExtractor_wrapper))
        .function("setNormFlag", select_overload<void(cv::NormHistogramCostExtractor&,int)>(&Wrappers::NormHistogramCostExtractor_setNormFlag_wrapper), pure_virtual())
        .function("getNormFlag", select_overload<int()const>(&cv::NormHistogramCostExtractor::getNormFlag), pure_virtual());

    emscripten::class_<cv::AgastFeatureDetector ,base<Feature2D>>("AgastFeatureDetector")
        .function("getNonmaxSuppression", select_overload<bool()const>(&cv::AgastFeatureDetector::getNonmaxSuppression), pure_virtual())
        .function("getThreshold", select_overload<int()const>(&cv::AgastFeatureDetector::getThreshold), pure_virtual())
        .constructor(select_overload<Ptr<AgastFeatureDetector>(int,bool,int)>(&Wrappers::AgastFeatureDetector_create_wrapper))
        .function("getType", select_overload<int()const>(&cv::AgastFeatureDetector::getType), pure_virtual())
        .function("setNonmaxSuppression", select_overload<void(cv::AgastFeatureDetector&,bool)>(&Wrappers::AgastFeatureDetector_setNonmaxSuppression_wrapper), pure_virtual())
        .function("setThreshold", select_overload<void(cv::AgastFeatureDetector&,int)>(&Wrappers::AgastFeatureDetector_setThreshold_wrapper), pure_virtual())
        .function("setType", select_overload<void(cv::AgastFeatureDetector&,int)>(&Wrappers::AgastFeatureDetector_setType_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::AgastFeatureDetector>>("Ptr<AgastFeatureDetector>")
;

    emscripten::class_<cv::ml::Boost ,base<ml::DTrees>>("ml_Boost")
        .constructor(select_overload<Ptr<Boost>()>(&cv::ml::Boost::create))
        .function("getWeightTrimRate", select_overload<double()const>(&cv::ml::Boost::getWeightTrimRate), pure_virtual())
        .function("setWeakCount", select_overload<void(cv::ml::Boost&,int)>(&Wrappers::ml_Boost_setWeakCount_wrapper), pure_virtual())
        .function("getWeakCount", select_overload<int()const>(&cv::ml::Boost::getWeakCount), pure_virtual())
        .function("getBoostType", select_overload<int()const>(&cv::ml::Boost::getBoostType), pure_virtual())
        .function("setWeightTrimRate", select_overload<void(cv::ml::Boost&,double)>(&Wrappers::ml_Boost_setWeightTrimRate_wrapper), pure_virtual())
        .function("setBoostType", select_overload<void(cv::ml::Boost&,int)>(&Wrappers::ml_Boost_setBoostType_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ml::Boost>>("Ptr<ml_Boost>")
;

    emscripten::class_<cv::DenseOpticalFlow ,base<Algorithm>>("DenseOpticalFlow")
        .function("calc", select_overload<void(cv::DenseOpticalFlow&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::DenseOpticalFlow_calc_wrapper), pure_virtual())
        .function("collectGarbage", select_overload<void()>(&cv::DenseOpticalFlow::collectGarbage), pure_virtual());

    emscripten::class_<cv::LineSegmentDetector ,base<Algorithm>>("LineSegmentDetector")
        .function("compareSegments", select_overload<int(cv::LineSegmentDetector&,const Size&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::LineSegmentDetector_compareSegments_wrapper), pure_virtual())
        .function("detect", select_overload<void(cv::LineSegmentDetector&,const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::LineSegmentDetector_detect_wrapper), pure_virtual())
        .function("drawSegments", select_overload<void(cv::LineSegmentDetector&,cv::Mat&,const cv::Mat&)>(&Wrappers::LineSegmentDetector_drawSegments_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<LineSegmentDetector>(int,double,double,double,double,double,double,int)>(&Wrappers::_createLineSegmentDetector_wrapper))
        .smart_ptr<Ptr<cv::LineSegmentDetector>>("Ptr<LineSegmentDetector>")
;

    emscripten::class_<cv::BackgroundSubtractor ,base<Algorithm>>("BackgroundSubtractor")
        .function("apply", select_overload<void(cv::BackgroundSubtractor&,const cv::Mat&,cv::Mat&,double)>(&Wrappers::BackgroundSubtractor_apply_wrapper), pure_virtual())
        .function("getBackgroundImage", select_overload<void(cv::BackgroundSubtractor&,cv::Mat&)>(&Wrappers::BackgroundSubtractor_getBackgroundImage_wrapper), pure_virtual());

    emscripten::class_<cv::ChiHistogramCostExtractor ,base<HistogramCostExtractor>>("ChiHistogramCostExtractor")
        .constructor(select_overload<Ptr<HistogramCostExtractor>(int,float)>(&Wrappers::_createChiHistogramCostExtractor_wrapper));

    emscripten::class_<cv::KeyPoint >("KeyPoint")
        .class_function("convert", select_overload<void(cv::KeyPoint&,const std::vector<KeyPoint>&,std::vector<Point2f>&,const std::vector<int>&)>(&Wrappers::KeyPoint_convert_wrapper))
        .class_function("convert1", select_overload<void(cv::KeyPoint&,const std::vector<Point2f>&,std::vector<KeyPoint>&,float,float,int,int)>(&Wrappers::KeyPoint_convert_wrapper1))
        .constructor<>()
        .constructor<float, float, float, float, float, int, int>()
        .class_function("overlap", select_overload<float(cv::KeyPoint&,const KeyPoint&,const KeyPoint&)>(&Wrappers::KeyPoint_overlap_wrapper))
        .property("pt", &cv::KeyPoint::pt)
        .property("size", &cv::KeyPoint::size)
        .property("angle", &cv::KeyPoint::angle)
        .property("response", &cv::KeyPoint::response)
        .property("octave", &cv::KeyPoint::octave)
        .property("class_id", &cv::KeyPoint::class_id);

    emscripten::class_<cv::AlignMTB ,base<AlignExposures>>("AlignMTB")
        .function("setCut", select_overload<void(cv::AlignMTB&,bool)>(&Wrappers::AlignMTB_setCut_wrapper), pure_virtual())
        .function("computeBitmaps", select_overload<void(cv::AlignMTB&,const cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::AlignMTB_computeBitmaps_wrapper), pure_virtual())
        .function("process", select_overload<void(cv::AlignMTB&,const std::vector<cv::Mat>&,std::vector<Mat>&,const cv::Mat&,const cv::Mat&)>(&Wrappers::AlignMTB_process_wrapper), pure_virtual())
        .function("process1", select_overload<void(cv::AlignMTB&,const std::vector<cv::Mat>&,std::vector<Mat>&)>(&Wrappers::AlignMTB_process_wrapper1), pure_virtual())
        .constructor(select_overload<Ptr<AlignMTB>(int,int,bool)>(&Wrappers::_createAlignMTB_wrapper))
        .function("setExcludeRange", select_overload<void(cv::AlignMTB&,int)>(&Wrappers::AlignMTB_setExcludeRange_wrapper), pure_virtual())
        .function("getExcludeRange", select_overload<int()const>(&cv::AlignMTB::getExcludeRange), pure_virtual())
        .function("setMaxBits", select_overload<void(cv::AlignMTB&,int)>(&Wrappers::AlignMTB_setMaxBits_wrapper), pure_virtual())
        .function("shiftMat", select_overload<void(cv::AlignMTB&,const cv::Mat&,cv::Mat&,const Point)>(&Wrappers::AlignMTB_shiftMat_wrapper), pure_virtual())
        .function("getCut", select_overload<bool()const>(&cv::AlignMTB::getCut), pure_virtual())
        .function("getMaxBits", select_overload<int()const>(&cv::AlignMTB::getMaxBits), pure_virtual())
        .function("calculateShift", select_overload<Point(cv::AlignMTB&,const cv::Mat&,const cv::Mat&)>(&Wrappers::AlignMTB_calculateShift_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::AlignMTB>>("Ptr<AlignMTB>")
;

    emscripten::class_<cv::AffineTransformer ,base<ShapeTransformer>>("AffineTransformer")
        .constructor(select_overload<Ptr<AffineTransformer>(bool)>(&Wrappers::_createAffineTransformer_wrapper))
        .function("getFullAffine", select_overload<bool()const>(&cv::AffineTransformer::getFullAffine), pure_virtual())
        .function("setFullAffine", select_overload<void(cv::AffineTransformer&,bool)>(&Wrappers::AffineTransformer_setFullAffine_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::AffineTransformer>>("Ptr<AffineTransformer>")
;

    emscripten::class_<cv::MergeDebevec ,base<MergeExposures>>("MergeDebevec")
        .function("process", select_overload<void(cv::MergeDebevec&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::MergeDebevec_process_wrapper), pure_virtual())
        .function("process1", select_overload<void(cv::MergeDebevec&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&)>(&Wrappers::MergeDebevec_process_wrapper1), pure_virtual())
        .constructor(select_overload<Ptr<MergeDebevec>()>(&cv::createMergeDebevec))
        .smart_ptr<Ptr<cv::MergeDebevec>>("Ptr<MergeDebevec>")
;

    emscripten::class_<cv::HistogramCostExtractor ,base<Algorithm>>("HistogramCostExtractor")
        .function("getNDummies", select_overload<int()const>(&cv::HistogramCostExtractor::getNDummies), pure_virtual())
        .function("getDefaultCost", select_overload<float()const>(&cv::HistogramCostExtractor::getDefaultCost), pure_virtual())
        .function("buildCostMatrix", select_overload<void(cv::HistogramCostExtractor&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::HistogramCostExtractor_buildCostMatrix_wrapper), pure_virtual())
        .function("setNDummies", select_overload<void(cv::HistogramCostExtractor&,int)>(&Wrappers::HistogramCostExtractor_setNDummies_wrapper), pure_virtual())
        .function("setDefaultCost", select_overload<void(cv::HistogramCostExtractor&,float)>(&Wrappers::HistogramCostExtractor_setDefaultCost_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::HistogramCostExtractor>>("Ptr<HistogramCostExtractor>")
;

    emscripten::class_<cv::ml::DTrees ,base<ml::StatModel>>("ml_DTrees")
        .function("getUse1SERule", select_overload<bool()const>(&cv::ml::DTrees::getUse1SERule), pure_virtual())
        .function("getMinSampleCount", select_overload<int()const>(&cv::ml::DTrees::getMinSampleCount), pure_virtual())
        .function("setTruncatePrunedTree", select_overload<void(cv::ml::DTrees&,bool)>(&Wrappers::ml_DTrees_setTruncatePrunedTree_wrapper), pure_virtual())
        .function("setMaxDepth", select_overload<void(cv::ml::DTrees&,int)>(&Wrappers::ml_DTrees_setMaxDepth_wrapper), pure_virtual())
        .function("setMinSampleCount", select_overload<void(cv::ml::DTrees&,int)>(&Wrappers::ml_DTrees_setMinSampleCount_wrapper), pure_virtual())
        .function("getTruncatePrunedTree", select_overload<bool()const>(&cv::ml::DTrees::getTruncatePrunedTree), pure_virtual())
        .function("setMaxCategories", select_overload<void(cv::ml::DTrees&,int)>(&Wrappers::ml_DTrees_setMaxCategories_wrapper), pure_virtual())
        .function("setCVFolds", select_overload<void(cv::ml::DTrees&,int)>(&Wrappers::ml_DTrees_setCVFolds_wrapper), pure_virtual())
        .function("getPriors", select_overload<cv::Mat()const>(&cv::ml::DTrees::getPriors), pure_virtual())
        .function("getCVFolds", select_overload<int()const>(&cv::ml::DTrees::getCVFolds), pure_virtual())
        .function("setUseSurrogates", select_overload<void(cv::ml::DTrees&,bool)>(&Wrappers::ml_DTrees_setUseSurrogates_wrapper), pure_virtual())
        .function("setUse1SERule", select_overload<void(cv::ml::DTrees&,bool)>(&Wrappers::ml_DTrees_setUse1SERule_wrapper), pure_virtual())
        .function("getMaxCategories", select_overload<int()const>(&cv::ml::DTrees::getMaxCategories), pure_virtual())
        .function("getUseSurrogates", select_overload<bool()const>(&cv::ml::DTrees::getUseSurrogates), pure_virtual())
        .function("setRegressionAccuracy", select_overload<void(cv::ml::DTrees&,float)>(&Wrappers::ml_DTrees_setRegressionAccuracy_wrapper), pure_virtual())
        .function("getMaxDepth", select_overload<int()const>(&cv::ml::DTrees::getMaxDepth), pure_virtual())
        .function("setPriors", select_overload<void(cv::ml::DTrees&,const cv::Mat &)>(&Wrappers::ml_DTrees_setPriors_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<DTrees>()>(&cv::ml::DTrees::create))
        .function("getRegressionAccuracy", select_overload<float()const>(&cv::ml::DTrees::getRegressionAccuracy), pure_virtual())
        .smart_ptr<Ptr<cv::ml::DTrees>>("Ptr<ml_DTrees>")
;

    emscripten::class_<cv::BOWKMeansTrainer ,base<BOWTrainer>>("BOWKMeansTrainer")
        .function("cluster", select_overload<Mat(cv::BOWKMeansTrainer&)>(&Wrappers::BOWKMeansTrainer_cluster_wrapper))
        .function("cluster1", select_overload<Mat(cv::BOWKMeansTrainer&,const Mat&)>(&Wrappers::BOWKMeansTrainer_cluster_wrapper1))
        .constructor<int, const TermCriteria&, int, int>();

    emscripten::class_<cv::FlannBasedMatcher ,base<DescriptorMatcher>>("FlannBasedMatcher")
        .constructor<const Ptr<flann::IndexParams>&, const Ptr<flann::SearchParams>&>();

    emscripten::class_<cv::TonemapMantiuk ,base<Tonemap>>("TonemapMantiuk")
        .function("getSaturation", select_overload<float()const>(&cv::TonemapMantiuk::getSaturation), pure_virtual())
        .function("getScale", select_overload<float()const>(&cv::TonemapMantiuk::getScale), pure_virtual())
        .constructor(select_overload<Ptr<TonemapMantiuk>(float,float,float)>(&Wrappers::_createTonemapMantiuk_wrapper))
        .function("setSaturation", select_overload<void(cv::TonemapMantiuk&,float)>(&Wrappers::TonemapMantiuk_setSaturation_wrapper), pure_virtual())
        .function("setScale", select_overload<void(cv::TonemapMantiuk&,float)>(&Wrappers::TonemapMantiuk_setScale_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::TonemapMantiuk>>("Ptr<TonemapMantiuk>")
;

    emscripten::class_<cv::ml::StatModel ,base<Algorithm>>("ml_StatModel")
        .function("isTrained", select_overload<bool()const>(&cv::ml::StatModel::isTrained), pure_virtual())
        .function("isClassifier", select_overload<bool()const>(&cv::ml::StatModel::isClassifier), pure_virtual())
        .function("predict", select_overload<float(cv::ml::StatModel&,const cv::Mat&,cv::Mat&,int)>(&Wrappers::ml_StatModel_predict_wrapper), pure_virtual())
        .function("calcError", select_overload<float(cv::ml::StatModel&,const Ptr<TrainData>&,bool,cv::Mat&)>(&Wrappers::ml_StatModel_calcError_wrapper))
        .function("getVarCount", select_overload<int()const>(&cv::ml::StatModel::getVarCount), pure_virtual())
        .function("train", select_overload<bool(cv::ml::StatModel&,const Ptr<TrainData>&,int)>(&Wrappers::ml_StatModel_train_wrapper))
        .function("train1", select_overload<bool(cv::ml::StatModel&,const cv::Mat&,int,const cv::Mat&)>(&Wrappers::ml_StatModel_train_wrapper1))
        .function("empty", select_overload<bool()const>(&cv::ml::StatModel::empty));

    emscripten::class_<cv::Tonemap ,base<Algorithm>>("Tonemap")
        .function("process", select_overload<void(cv::Tonemap&,const cv::Mat&,cv::Mat&)>(&Wrappers::Tonemap_process_wrapper), pure_virtual())
        .function("setGamma", select_overload<void(cv::Tonemap&,float)>(&Wrappers::Tonemap_setGamma_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<Tonemap>(float)>(&Wrappers::_createTonemap_wrapper))
        .function("getGamma", select_overload<float()const>(&cv::Tonemap::getGamma), pure_virtual())
        .smart_ptr<Ptr<cv::Tonemap>>("Ptr<Tonemap>")
;

    emscripten::class_<cv::ml::KNearest ,base<ml::StatModel>>("ml_KNearest")
        .function("setIsClassifier", select_overload<void(cv::ml::KNearest&,bool)>(&Wrappers::ml_KNearest_setIsClassifier_wrapper), pure_virtual())
        .function("getIsClassifier", select_overload<bool()const>(&cv::ml::KNearest::getIsClassifier), pure_virtual())
        .function("setAlgorithmType", select_overload<void(cv::ml::KNearest&,int)>(&Wrappers::ml_KNearest_setAlgorithmType_wrapper), pure_virtual())
        .function("setEmax", select_overload<void(cv::ml::KNearest&,int)>(&Wrappers::ml_KNearest_setEmax_wrapper), pure_virtual())
        .function("setDefaultK", select_overload<void(cv::ml::KNearest&,int)>(&Wrappers::ml_KNearest_setDefaultK_wrapper), pure_virtual())
        .function("getDefaultK", select_overload<int()const>(&cv::ml::KNearest::getDefaultK), pure_virtual())
        .function("getEmax", select_overload<int()const>(&cv::ml::KNearest::getEmax), pure_virtual())
        .function("getAlgorithmType", select_overload<int()const>(&cv::ml::KNearest::getAlgorithmType), pure_virtual())
        .constructor(select_overload<Ptr<KNearest>()>(&cv::ml::KNearest::create))
        .function("findNearest", select_overload<float(cv::ml::KNearest&,const cv::Mat&,int,cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::ml_KNearest_findNearest_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ml::KNearest>>("Ptr<ml_KNearest>")
;

    emscripten::class_<cv::FastFeatureDetector ,base<Feature2D>>("FastFeatureDetector")
        .function("getNonmaxSuppression", select_overload<bool()const>(&cv::FastFeatureDetector::getNonmaxSuppression), pure_virtual())
        .function("getThreshold", select_overload<int()const>(&cv::FastFeatureDetector::getThreshold), pure_virtual())
        .constructor(select_overload<Ptr<FastFeatureDetector>(int,bool,int)>(&Wrappers::FastFeatureDetector_create_wrapper))
        .function("getType", select_overload<int()const>(&cv::FastFeatureDetector::getType), pure_virtual())
        .function("setNonmaxSuppression", select_overload<void(cv::FastFeatureDetector&,bool)>(&Wrappers::FastFeatureDetector_setNonmaxSuppression_wrapper), pure_virtual())
        .function("setThreshold", select_overload<void(cv::FastFeatureDetector&,int)>(&Wrappers::FastFeatureDetector_setThreshold_wrapper), pure_virtual())
        .function("setType", select_overload<void(cv::FastFeatureDetector&,int)>(&Wrappers::FastFeatureDetector_setType_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::FastFeatureDetector>>("Ptr<FastFeatureDetector>")
;

    emscripten::class_<cv::Subdiv2D >("Subdiv2D")
        .function("insert", select_overload<int(cv::Subdiv2D&,Point2f)>(&Wrappers::Subdiv2D_insert_wrapper))
        .function("insert1", select_overload<void(cv::Subdiv2D&,const std::vector<Point2f>&)>(&Wrappers::Subdiv2D_insert_wrapper1))
        .function("edgeOrg", select_overload<int(cv::Subdiv2D&,int,Point2f*)>(&Wrappers::Subdiv2D_edgeOrg_wrapper), allow_raw_pointers())
        .function("rotateEdge", select_overload<int(cv::Subdiv2D&,int,int)>(&Wrappers::Subdiv2D_rotateEdge_wrapper))
        .function("initDelaunay", select_overload<void(cv::Subdiv2D&,Rect)>(&Wrappers::Subdiv2D_initDelaunay_wrapper))
        .constructor<>()
        .constructor<Rect>()
        .function("getEdge", select_overload<int(cv::Subdiv2D&,int,int)>(&Wrappers::Subdiv2D_getEdge_wrapper))
        .function("getTriangleList", select_overload<void(cv::Subdiv2D&,std::vector<Vec6f>&)>(&Wrappers::Subdiv2D_getTriangleList_wrapper))
        .function("nextEdge", select_overload<int(cv::Subdiv2D&,int)>(&Wrappers::Subdiv2D_nextEdge_wrapper))
        .function("edgeDst", select_overload<int(cv::Subdiv2D&,int,Point2f*)>(&Wrappers::Subdiv2D_edgeDst_wrapper), allow_raw_pointers())
        .function("getEdgeList", select_overload<void(cv::Subdiv2D&,std::vector<Vec4f>&)>(&Wrappers::Subdiv2D_getEdgeList_wrapper))
        .function("getVertex", select_overload<Point2f(cv::Subdiv2D&,int,int*)>(&Wrappers::Subdiv2D_getVertex_wrapper), allow_raw_pointers())
        .function("getVoronoiFacetList", select_overload<void(cv::Subdiv2D&,const std::vector<int>&,std::vector<std::vector<Point2f> >&,std::vector<Point2f>&)>(&Wrappers::Subdiv2D_getVoronoiFacetList_wrapper))
        .function("symEdge", select_overload<int(cv::Subdiv2D&,int)>(&Wrappers::Subdiv2D_symEdge_wrapper))
        .function("findNearest", select_overload<int(cv::Subdiv2D&,Point2f,Point2f*)>(&Wrappers::Subdiv2D_findNearest_wrapper), allow_raw_pointers());

    emscripten::class_<cv::BRISK ,base<Feature2D>>("BRISK")
        .constructor(select_overload<Ptr<BRISK>(int,int,float)>(&Wrappers::BRISK_create_wrapper))
        .constructor(select_overload<Ptr<BRISK>(const std::vector<float> &,const std::vector<int> &,float,float,const std::vector<int>&)>(&Wrappers::BRISK_create_wrapper1))
        .smart_ptr<Ptr<cv::BRISK>>("Ptr<BRISK>")
;

    emscripten::class_<cv::KalmanFilter >("KalmanFilter")
        .function("predict", select_overload<const Mat&(cv::KalmanFilter&,const Mat&)>(&Wrappers::KalmanFilter_predict_wrapper))
        .constructor<>()
        .constructor<int, int, int, int>()
        .function("correct", select_overload<const Mat&(cv::KalmanFilter&,const Mat&)>(&Wrappers::KalmanFilter_correct_wrapper))
        .property("statePre", &cv::KalmanFilter::statePre)
        .property("statePost", &cv::KalmanFilter::statePost)
        .property("transitionMatrix", &cv::KalmanFilter::transitionMatrix)
        .property("controlMatrix", &cv::KalmanFilter::controlMatrix)
        .property("measurementMatrix", &cv::KalmanFilter::measurementMatrix)
        .property("processNoiseCov", &cv::KalmanFilter::processNoiseCov)
        .property("measurementNoiseCov", &cv::KalmanFilter::measurementNoiseCov)
        .property("errorCovPre", &cv::KalmanFilter::errorCovPre)
        .property("gain", &cv::KalmanFilter::gain)
        .property("errorCovPost", &cv::KalmanFilter::errorCovPost);

    emscripten::class_<cv::EMDHistogramCostExtractor ,base<HistogramCostExtractor>>("EMDHistogramCostExtractor")
        .constructor(select_overload<Ptr<HistogramCostExtractor>(int,int,float)>(&Wrappers::_createEMDHistogramCostExtractor_wrapper))
        .function("setNormFlag", select_overload<void(cv::EMDHistogramCostExtractor&,int)>(&Wrappers::EMDHistogramCostExtractor_setNormFlag_wrapper), pure_virtual())
        .function("getNormFlag", select_overload<int()const>(&cv::EMDHistogramCostExtractor::getNormFlag), pure_virtual());

    emscripten::class_<cv::ThinPlateSplineShapeTransformer ,base<ShapeTransformer>>("ThinPlateSplineShapeTransformer")
        .function("setRegularizationParameter", select_overload<void(cv::ThinPlateSplineShapeTransformer&,double)>(&Wrappers::ThinPlateSplineShapeTransformer_setRegularizationParameter_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<ThinPlateSplineShapeTransformer>(double)>(&Wrappers::_createThinPlateSplineShapeTransformer_wrapper))
        .function("getRegularizationParameter", select_overload<double()const>(&cv::ThinPlateSplineShapeTransformer::getRegularizationParameter), pure_virtual())
        .smart_ptr<Ptr<cv::ThinPlateSplineShapeTransformer>>("Ptr<ThinPlateSplineShapeTransformer>")
;

    emscripten::class_<cv::ShapeContextDistanceExtractor ,base<ShapeDistanceExtractor>>("ShapeContextDistanceExtractor")
        .function("getBendingEnergyWeight", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getBendingEnergyWeight), pure_virtual())
        .function("setAngularBins", select_overload<void(cv::ShapeContextDistanceExtractor&,int)>(&Wrappers::ShapeContextDistanceExtractor_setAngularBins_wrapper), pure_virtual())
        .function("getCostExtractor", select_overload<Ptr<HistogramCostExtractor>()const>(&cv::ShapeContextDistanceExtractor::getCostExtractor), pure_virtual())
        .function("getTransformAlgorithm", select_overload<Ptr<ShapeTransformer>()const>(&cv::ShapeContextDistanceExtractor::getTransformAlgorithm), pure_virtual())
        .function("setTransformAlgorithm", select_overload<void(cv::ShapeContextDistanceExtractor&,Ptr<ShapeTransformer>)>(&Wrappers::ShapeContextDistanceExtractor_setTransformAlgorithm_wrapper), pure_virtual())
        .function("getIterations", select_overload<int()const>(&cv::ShapeContextDistanceExtractor::getIterations), pure_virtual())
        .function("setShapeContextWeight", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setShapeContextWeight_wrapper), pure_virtual())
        .function("getInnerRadius", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getInnerRadius), pure_virtual())
        .function("setImageAppearanceWeight", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setImageAppearanceWeight_wrapper), pure_virtual())
        .function("setRotationInvariant", select_overload<void(cv::ShapeContextDistanceExtractor&,bool)>(&Wrappers::ShapeContextDistanceExtractor_setRotationInvariant_wrapper), pure_virtual())
        .function("setBendingEnergyWeight", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setBendingEnergyWeight_wrapper), pure_virtual())
        .function("setStdDev", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setStdDev_wrapper), pure_virtual())
        .function("setCostExtractor", select_overload<void(cv::ShapeContextDistanceExtractor&,Ptr<HistogramCostExtractor>)>(&Wrappers::ShapeContextDistanceExtractor_setCostExtractor_wrapper), pure_virtual())
        .function("setOuterRadius", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setOuterRadius_wrapper), pure_virtual())
        .function("getShapeContextWeight", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getShapeContextWeight), pure_virtual())
        .function("getRotationInvariant", select_overload<bool()const>(&cv::ShapeContextDistanceExtractor::getRotationInvariant), pure_virtual())
        .function("getAngularBins", select_overload<int()const>(&cv::ShapeContextDistanceExtractor::getAngularBins), pure_virtual())
        .function("getImages", select_overload<void(cv::ShapeContextDistanceExtractor&,cv::Mat&,cv::Mat&)>(&Wrappers::ShapeContextDistanceExtractor_getImages_wrapper), pure_virtual())
        .function("getImageAppearanceWeight", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getImageAppearanceWeight), pure_virtual())
        .constructor(select_overload<Ptr<ShapeContextDistanceExtractor>(int,int,float,float,int,const Ptr<HistogramCostExtractor> &,const Ptr<ShapeTransformer> &)>(&Wrappers::_createShapeContextDistanceExtractor_wrapper))
        .function("getOuterRadius", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getOuterRadius), pure_virtual())
        .function("setInnerRadius", select_overload<void(cv::ShapeContextDistanceExtractor&,float)>(&Wrappers::ShapeContextDistanceExtractor_setInnerRadius_wrapper), pure_virtual())
        .function("setIterations", select_overload<void(cv::ShapeContextDistanceExtractor&,int)>(&Wrappers::ShapeContextDistanceExtractor_setIterations_wrapper), pure_virtual())
        .function("getRadialBins", select_overload<int()const>(&cv::ShapeContextDistanceExtractor::getRadialBins), pure_virtual())
        .function("getStdDev", select_overload<float()const>(&cv::ShapeContextDistanceExtractor::getStdDev), pure_virtual())
        .function("setImages", select_overload<void(cv::ShapeContextDistanceExtractor&,const cv::Mat&,const cv::Mat&)>(&Wrappers::ShapeContextDistanceExtractor_setImages_wrapper), pure_virtual())
        .function("setRadialBins", select_overload<void(cv::ShapeContextDistanceExtractor&,int)>(&Wrappers::ShapeContextDistanceExtractor_setRadialBins_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ShapeContextDistanceExtractor>>("Ptr<ShapeContextDistanceExtractor>")
;

    emscripten::class_<cv::KAZE ,base<Feature2D>>("KAZE")
        .function("setExtended", select_overload<void(cv::KAZE&,bool)>(&Wrappers::KAZE_setExtended_wrapper), pure_virtual())
        .function("setNOctaveLayers", select_overload<void(cv::KAZE&,int)>(&Wrappers::KAZE_setNOctaveLayers_wrapper), pure_virtual())
        .function("getNOctaves", select_overload<int()const>(&cv::KAZE::getNOctaves), pure_virtual())
        .function("getNOctaveLayers", select_overload<int()const>(&cv::KAZE::getNOctaveLayers), pure_virtual())
        .function("setNOctaves", select_overload<void(cv::KAZE&,int)>(&Wrappers::KAZE_setNOctaves_wrapper), pure_virtual())
        .function("getUpright", select_overload<bool()const>(&cv::KAZE::getUpright), pure_virtual())
        .constructor(select_overload<Ptr<KAZE>(bool,bool,float,int,int,int)>(&Wrappers::KAZE_create_wrapper))
        .function("getExtended", select_overload<bool()const>(&cv::KAZE::getExtended), pure_virtual())
        .function("setUpright", select_overload<void(cv::KAZE&,bool)>(&Wrappers::KAZE_setUpright_wrapper), pure_virtual())
        .function("setDiffusivity", select_overload<void(cv::KAZE&,int)>(&Wrappers::KAZE_setDiffusivity_wrapper), pure_virtual())
        .function("getThreshold", select_overload<double()const>(&cv::KAZE::getThreshold), pure_virtual())
        .function("getDiffusivity", select_overload<int()const>(&cv::KAZE::getDiffusivity), pure_virtual())
        .function("setThreshold", select_overload<void(cv::KAZE&,double)>(&Wrappers::KAZE_setThreshold_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::KAZE>>("Ptr<KAZE>")
;

    emscripten::class_<cv::CalibrateCRF ,base<Algorithm>>("CalibrateCRF")
        .function("process", select_overload<void(cv::CalibrateCRF&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&)>(&Wrappers::CalibrateCRF_process_wrapper), pure_virtual());

    emscripten::class_<cv::AKAZE ,base<Feature2D>>("AKAZE")
        .function("setNOctaveLayers", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setNOctaveLayers_wrapper), pure_virtual())
        .function("setDescriptorType", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setDescriptorType_wrapper), pure_virtual())
        .function("getNOctaveLayers", select_overload<int()const>(&cv::AKAZE::getNOctaveLayers), pure_virtual())
        .function("setNOctaves", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setNOctaves_wrapper), pure_virtual())
        .function("getDescriptorType", select_overload<int()const>(&cv::AKAZE::getDescriptorType), pure_virtual())
        .function("getThreshold", select_overload<double()const>(&cv::AKAZE::getThreshold), pure_virtual())
        .constructor(select_overload<Ptr<AKAZE>(int,int,int,float,int,int,int)>(&Wrappers::AKAZE_create_wrapper))
        .function("getNOctaves", select_overload<int()const>(&cv::AKAZE::getNOctaves), pure_virtual())
        .function("setDescriptorChannels", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setDescriptorChannels_wrapper), pure_virtual())
        .function("setThreshold", select_overload<void(cv::AKAZE&,double)>(&Wrappers::AKAZE_setThreshold_wrapper), pure_virtual())
        .function("getDescriptorChannels", select_overload<int()const>(&cv::AKAZE::getDescriptorChannels), pure_virtual())
        .function("setDescriptorSize", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setDescriptorSize_wrapper), pure_virtual())
        .function("setDiffusivity", select_overload<void(cv::AKAZE&,int)>(&Wrappers::AKAZE_setDiffusivity_wrapper), pure_virtual())
        .function("getDiffusivity", select_overload<int()const>(&cv::AKAZE::getDiffusivity), pure_virtual())
        .function("getDescriptorSize", select_overload<int()const>(&cv::AKAZE::getDescriptorSize), pure_virtual())
        .smart_ptr<Ptr<cv::AKAZE>>("Ptr<AKAZE>")
;

    emscripten::class_<cv::Feature2D ,base<Algorithm ,true>>("Feature2D")
        .function("detect", select_overload<void(cv::Feature2D&,const cv::Mat&,std::vector<KeyPoint>&,const cv::Mat&)>(&Wrappers::Feature2D_detect_wrapper))
        .function("compute", select_overload<void(cv::Feature2D&,const cv::Mat&,std::vector<KeyPoint>&,cv::Mat&)>(&Wrappers::Feature2D_compute_wrapper))
        .function("defaultNorm", select_overload<int()const>(&cv::Feature2D::defaultNorm))
        .function("detectAndCompute", select_overload<void(cv::Feature2D&,const cv::Mat&,const cv::Mat&,std::vector<KeyPoint>&,cv::Mat&,bool)>(&Wrappers::Feature2D_detectAndCompute_wrapper))
        .function("descriptorSize", select_overload<int()const>(&cv::Feature2D::descriptorSize))
        .function("descriptorType", select_overload<int()const>(&cv::Feature2D::descriptorType))
        .function("empty", select_overload<bool()const>(&cv::Feature2D::empty));

    emscripten::class_<cv::flann::Index >("flann_Index")
        .function("load", select_overload<bool(cv::flann::Index&,const cv::Mat&,const std::string&)>(&Wrappers::flann_Index_load_wrapper))
        .constructor<>()
        .constructor<InputArray, const IndexParams&, cvflann::flann_distance_t>()
        .function("radiusSearch", select_overload<int(cv::flann::Index&,const cv::Mat&,cv::Mat&,cv::Mat&,double,int,const SearchParams&)>(&Wrappers::flann_Index_radiusSearch_wrapper))
        .function("getAlgorithm", select_overload<cvflann::flann_algorithm_t()const>(&cv::flann::Index::getAlgorithm))
        .function("build", select_overload<void(cv::flann::Index&,const cv::Mat&,const IndexParams&,cvflann::flann_distance_t)>(&Wrappers::flann_Index_build_wrapper))
        .function("getDistance", select_overload<cvflann::flann_distance_t()const>(&cv::flann::Index::getDistance))
        .function("release", select_overload<void()>(&cv::flann::Index::release))
        .function("save", select_overload<void(cv::flann::Index&,const std::string&)>(&Wrappers::flann_Index_save_wrapper))
        .function("knnSearch", select_overload<void(cv::flann::Index&,const cv::Mat&,cv::Mat&,cv::Mat&,int,const SearchParams&)>(&Wrappers::flann_Index_knnSearch_wrapper));

    emscripten::class_<cv::ml::NormalBayesClassifier ,base<ml::StatModel>>("ml_NormalBayesClassifier")
        .function("predictProb", select_overload<float(cv::ml::NormalBayesClassifier&,const cv::Mat&,cv::Mat&,cv::Mat&,int)>(&Wrappers::ml_NormalBayesClassifier_predictProb_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<NormalBayesClassifier>()>(&cv::ml::NormalBayesClassifier::create))
        .smart_ptr<Ptr<cv::ml::NormalBayesClassifier>>("Ptr<ml_NormalBayesClassifier>")
;

    emscripten::class_<cv::MergeExposures ,base<Algorithm>>("MergeExposures")
        .function("process", select_overload<void(cv::MergeExposures&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::MergeExposures_process_wrapper), pure_virtual());

    emscripten::class_<cv::MergeRobertson ,base<MergeExposures>>("MergeRobertson")
        .function("process", select_overload<void(cv::MergeRobertson&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::MergeRobertson_process_wrapper), pure_virtual())
        .function("process1", select_overload<void(cv::MergeRobertson&,const std::vector<cv::Mat>&,cv::Mat&,const cv::Mat&)>(&Wrappers::MergeRobertson_process_wrapper1), pure_virtual())
        .constructor(select_overload<Ptr<MergeRobertson>()>(&cv::createMergeRobertson))
        .smart_ptr<Ptr<cv::MergeRobertson>>("Ptr<MergeRobertson>")
;

    emscripten::class_<cv::CalibrateDebevec ,base<CalibrateCRF>>("CalibrateDebevec")
        .function("setRandom", select_overload<void(cv::CalibrateDebevec&,bool)>(&Wrappers::CalibrateDebevec_setRandom_wrapper), pure_virtual())
        .function("getRandom", select_overload<bool()const>(&cv::CalibrateDebevec::getRandom), pure_virtual())
        .constructor(select_overload<Ptr<CalibrateDebevec>(int,float,bool)>(&Wrappers::_createCalibrateDebevec_wrapper))
        .function("getSamples", select_overload<int()const>(&cv::CalibrateDebevec::getSamples), pure_virtual())
        .function("setLambda", select_overload<void(cv::CalibrateDebevec&,float)>(&Wrappers::CalibrateDebevec_setLambda_wrapper), pure_virtual())
        .function("setSamples", select_overload<void(cv::CalibrateDebevec&,int)>(&Wrappers::CalibrateDebevec_setSamples_wrapper), pure_virtual())
        .function("getLambda", select_overload<float()const>(&cv::CalibrateDebevec::getLambda), pure_virtual())
        .smart_ptr<Ptr<cv::CalibrateDebevec>>("Ptr<CalibrateDebevec>")
;

    emscripten::class_<cv::BOWImgDescriptorExtractor >("BOWImgDescriptorExtractor")
        .function("setVocabulary", select_overload<void(cv::BOWImgDescriptorExtractor&,const Mat&)>(&Wrappers::BOWImgDescriptorExtractor_setVocabulary_wrapper))
        .function("compute", select_overload<void(cv::BOWImgDescriptorExtractor&,const Mat&,std::vector<KeyPoint>&,Mat&)>(&Wrappers::BOWImgDescriptorExtractor_compute2_wrapper))
        .constructor<const Ptr<DescriptorExtractor>&, const Ptr<DescriptorMatcher>&>()
        .function("getVocabulary", select_overload<const Mat&()const>(&cv::BOWImgDescriptorExtractor::getVocabulary))
        .function("descriptorSize", select_overload<int()const>(&cv::BOWImgDescriptorExtractor::descriptorSize))
        .function("descriptorType", select_overload<int()const>(&cv::BOWImgDescriptorExtractor::descriptorType));

    emscripten::class_<cv::AlignExposures ,base<Algorithm>>("AlignExposures")
        .function("process", select_overload<void(cv::AlignExposures&,const std::vector<cv::Mat>&,std::vector<Mat>&,const cv::Mat&,const cv::Mat&)>(&Wrappers::AlignExposures_process_wrapper), pure_virtual());

    emscripten::class_<cv::Algorithm >("Algorithm")
        .function("getDefaultName", select_overload<std::string(cv::Algorithm&)>(&Wrappers::Algorithm_getDefaultName_wrapper))
        .function("clear", select_overload<void()>(&cv::Algorithm::clear))
        .function("save", select_overload<void(cv::Algorithm&,const std::string&)>(&Wrappers::Algorithm_save_wrapper));

    emscripten::class_<cv::ShapeTransformer ,base<Algorithm>>("ShapeTransformer")
        .function("warpImage", select_overload<void(cv::ShapeTransformer&,const cv::Mat&,cv::Mat&,int,int,const Scalar&)>(&Wrappers::ShapeTransformer_warpImage_wrapper), pure_virtual())
        .function("estimateTransformation", select_overload<void(cv::ShapeTransformer&,const cv::Mat&,const cv::Mat&,std::vector<DMatch>&)>(&Wrappers::ShapeTransformer_estimateTransformation_wrapper), pure_virtual())
        .function("applyTransformation", select_overload<float(cv::ShapeTransformer&,const cv::Mat&,cv::Mat&)>(&Wrappers::ShapeTransformer_applyTransformation_wrapper), pure_virtual());

    emscripten::class_<cv::ml::ANN_MLP ,base<ml::StatModel>>("ml_ANN_MLP")
        .function("getTrainMethod", select_overload<int()const>(&cv::ml::ANN_MLP::getTrainMethod), pure_virtual())
        .function("getRpropDWMinus", select_overload<double()const>(&cv::ml::ANN_MLP::getRpropDWMinus), pure_virtual())
        .function("getRpropDWMin", select_overload<double()const>(&cv::ml::ANN_MLP::getRpropDWMin), pure_virtual())
        .function("setRpropDWMinus", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setRpropDWMinus_wrapper), pure_virtual())
        .function("getRpropDWPlus", select_overload<double()const>(&cv::ml::ANN_MLP::getRpropDWPlus), pure_virtual())
        .function("setBackpropWeightScale", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setBackpropWeightScale_wrapper), pure_virtual())
        .function("setRpropDWMin", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setRpropDWMin_wrapper), pure_virtual())
        .function("setActivationFunction", select_overload<void(cv::ml::ANN_MLP&,int,double,double)>(&Wrappers::ml_ANN_MLP_setActivationFunction_wrapper), pure_virtual())
        .function("getBackpropMomentumScale", select_overload<double()const>(&cv::ml::ANN_MLP::getBackpropMomentumScale), pure_virtual())
        .function("getRpropDWMax", select_overload<double()const>(&cv::ml::ANN_MLP::getRpropDWMax), pure_virtual())
        .function("setRpropDWMax", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setRpropDWMax_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<ANN_MLP>()>(&cv::ml::ANN_MLP::create))
        .function("setRpropDW0", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setRpropDW0_wrapper), pure_virtual())
        .function("setBackpropMomentumScale", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setBackpropMomentumScale_wrapper), pure_virtual())
        .function("setRpropDWPlus", select_overload<void(cv::ml::ANN_MLP&,double)>(&Wrappers::ml_ANN_MLP_setRpropDWPlus_wrapper), pure_virtual())
        .function("setTrainMethod", select_overload<void(cv::ml::ANN_MLP&,int,double,double)>(&Wrappers::ml_ANN_MLP_setTrainMethod_wrapper), pure_virtual())
        .function("setTermCriteria", select_overload<void(cv::ml::ANN_MLP&,TermCriteria)>(&Wrappers::ml_ANN_MLP_setTermCriteria_wrapper), pure_virtual())
        .function("setLayerSizes", select_overload<void(cv::ml::ANN_MLP&,const cv::Mat&)>(&Wrappers::ml_ANN_MLP_setLayerSizes_wrapper), pure_virtual())
        .function("getLayerSizes", select_overload<cv::Mat()const>(&cv::ml::ANN_MLP::getLayerSizes), pure_virtual())
        .function("getBackpropWeightScale", select_overload<double()const>(&cv::ml::ANN_MLP::getBackpropWeightScale), pure_virtual())
        .function("getTermCriteria", select_overload<TermCriteria()const>(&cv::ml::ANN_MLP::getTermCriteria), pure_virtual())
        .function("getWeights", select_overload<Mat(cv::ml::ANN_MLP&,int)>(&Wrappers::ml_ANN_MLP_getWeights_wrapper), pure_virtual())
        .function("getRpropDW0", select_overload<double()const>(&cv::ml::ANN_MLP::getRpropDW0), pure_virtual())
        .smart_ptr<Ptr<cv::ml::ANN_MLP>>("Ptr<ml_ANN_MLP>")
;

    emscripten::class_<cv::ml::RTrees ,base<ml::DTrees>>("ml_RTrees")
        .function("getTermCriteria", select_overload<TermCriteria()const>(&cv::ml::RTrees::getTermCriteria), pure_virtual())
        .function("getCalculateVarImportance", select_overload<bool()const>(&cv::ml::RTrees::getCalculateVarImportance), pure_virtual())
        .constructor(select_overload<Ptr<RTrees>()>(&cv::ml::RTrees::create))
        .function("setCalculateVarImportance", select_overload<void(cv::ml::RTrees&,bool)>(&Wrappers::ml_RTrees_setCalculateVarImportance_wrapper), pure_virtual())
        .function("getVarImportance", select_overload<Mat()const>(&cv::ml::RTrees::getVarImportance), pure_virtual())
        .function("setTermCriteria", select_overload<void(cv::ml::RTrees&,const TermCriteria &)>(&Wrappers::ml_RTrees_setTermCriteria_wrapper), pure_virtual())
        .function("getActiveVarCount", select_overload<int()const>(&cv::ml::RTrees::getActiveVarCount), pure_virtual())
        .function("setActiveVarCount", select_overload<void(cv::ml::RTrees&,int)>(&Wrappers::ml_RTrees_setActiveVarCount_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ml::RTrees>>("Ptr<ml_RTrees>")
;

    emscripten::class_<cv::ShapeDistanceExtractor ,base<Algorithm>>("ShapeDistanceExtractor")
        .function("computeDistance", select_overload<float(cv::ShapeDistanceExtractor&,const cv::Mat&,const cv::Mat&)>(&Wrappers::ShapeDistanceExtractor_computeDistance_wrapper), pure_virtual());

    emscripten::class_<cv::GFTTDetector ,base<Feature2D>>("GFTTDetector")
        .function("getBlockSize", select_overload<int()const>(&cv::GFTTDetector::getBlockSize), pure_virtual())
        .function("getK", select_overload<double()const>(&cv::GFTTDetector::getK), pure_virtual())
        .function("setHarrisDetector", select_overload<void(cv::GFTTDetector&,bool)>(&Wrappers::GFTTDetector_setHarrisDetector_wrapper), pure_virtual())
        .function("setBlockSize", select_overload<void(cv::GFTTDetector&,int)>(&Wrappers::GFTTDetector_setBlockSize_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<GFTTDetector>(int,double,double,int,bool,double)>(&Wrappers::GFTTDetector_create_wrapper))
        .function("setQualityLevel", select_overload<void(cv::GFTTDetector&,double)>(&Wrappers::GFTTDetector_setQualityLevel_wrapper), pure_virtual())
        .function("setMaxFeatures", select_overload<void(cv::GFTTDetector&,int)>(&Wrappers::GFTTDetector_setMaxFeatures_wrapper), pure_virtual())
        .function("setK", select_overload<void(cv::GFTTDetector&,double)>(&Wrappers::GFTTDetector_setK_wrapper), pure_virtual())
        .function("getMaxFeatures", select_overload<int()const>(&cv::GFTTDetector::getMaxFeatures), pure_virtual())
        .function("setMinDistance", select_overload<void(cv::GFTTDetector&,double)>(&Wrappers::GFTTDetector_setMinDistance_wrapper), pure_virtual())
        .function("getMinDistance", select_overload<double()const>(&cv::GFTTDetector::getMinDistance), pure_virtual())
        .function("getQualityLevel", select_overload<double()const>(&cv::GFTTDetector::getQualityLevel), pure_virtual())
        .function("getHarrisDetector", select_overload<bool()const>(&cv::GFTTDetector::getHarrisDetector), pure_virtual())
        .smart_ptr<Ptr<cv::GFTTDetector>>("Ptr<GFTTDetector>")
;

    emscripten::class_<cv::DescriptorMatcher ,base<Algorithm>>("DescriptorMatcher")
        .constructor(select_overload<Ptr<DescriptorMatcher>(const std::string&)>(&Wrappers::DescriptorMatcher_create_wrapper))
        .function("clear", select_overload<void()>(&cv::DescriptorMatcher::clear))
        .function("knnMatch", select_overload<void(cv::DescriptorMatcher&,const cv::Mat&,const cv::Mat&,std::vector<std::vector<DMatch> >&,int,const cv::Mat&,bool)>(&Wrappers::DescriptorMatcher_knnMatch_wrapper))
        .function("knnMatch1", select_overload<void(cv::DescriptorMatcher&,const cv::Mat&,std::vector<std::vector<DMatch> >&,int,const std::vector<cv::Mat>&,bool)>(&Wrappers::DescriptorMatcher_knnMatch_wrapper1))
        .function("add", select_overload<void(cv::DescriptorMatcher&,const std::vector<cv::Mat>&)>(&Wrappers::DescriptorMatcher_add_wrapper))
        .function("train", select_overload<void()>(&cv::DescriptorMatcher::train))
        .function("match", select_overload<void(cv::DescriptorMatcher&,const cv::Mat&,const cv::Mat&,std::vector<DMatch>&,const cv::Mat&)>(&Wrappers::DescriptorMatcher_match_wrapper))
        .function("match1", select_overload<void(cv::DescriptorMatcher&,const cv::Mat&,std::vector<DMatch>&,const std::vector<cv::Mat>&)>(&Wrappers::DescriptorMatcher_match_wrapper1))
        .function("getTrainDescriptors", select_overload<const std::vector<Mat>&()const>(&cv::DescriptorMatcher::getTrainDescriptors))
        .function("isMaskSupported", select_overload<bool()const>(&cv::DescriptorMatcher::isMaskSupported), pure_virtual())
        .function("empty", select_overload<bool()const>(&cv::DescriptorMatcher::empty))
        .smart_ptr<Ptr<cv::DescriptorMatcher>>("Ptr<DescriptorMatcher>")
;

    emscripten::class_<cv::DMatch >("DMatch")
        .constructor<>()
        .constructor<int, int, float>()
        .constructor<int, int, int, float>()
        .property("queryIdx", &cv::DMatch::queryIdx)
        .property("trainIdx", &cv::DMatch::trainIdx)
        .property("imgIdx", &cv::DMatch::imgIdx)
        .property("distance", &cv::DMatch::distance);

    emscripten::class_<cv::TonemapDurand ,base<Tonemap>>("TonemapDurand")
        .function("setSigmaSpace", select_overload<void(cv::TonemapDurand&,float)>(&Wrappers::TonemapDurand_setSigmaSpace_wrapper), pure_virtual())
        .function("setSigmaColor", select_overload<void(cv::TonemapDurand&,float)>(&Wrappers::TonemapDurand_setSigmaColor_wrapper), pure_virtual())
        .function("setSaturation", select_overload<void(cv::TonemapDurand&,float)>(&Wrappers::TonemapDurand_setSaturation_wrapper), pure_virtual())
        .function("getContrast", select_overload<float()const>(&cv::TonemapDurand::getContrast), pure_virtual())
        .function("getSigmaSpace", select_overload<float()const>(&cv::TonemapDurand::getSigmaSpace), pure_virtual())
        .function("getSigmaColor", select_overload<float()const>(&cv::TonemapDurand::getSigmaColor), pure_virtual())
        .function("getSaturation", select_overload<float()const>(&cv::TonemapDurand::getSaturation), pure_virtual())
        .constructor(select_overload<Ptr<TonemapDurand>(float,float,float,float,float)>(&Wrappers::_createTonemapDurand_wrapper))
        .function("setContrast", select_overload<void(cv::TonemapDurand&,float)>(&Wrappers::TonemapDurand_setContrast_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::TonemapDurand>>("Ptr<TonemapDurand>")
;

    emscripten::class_<cv::BFMatcher ,base<DescriptorMatcher>>("BFMatcher")
        .constructor<int, bool>();

    emscripten::class_<cv::BaseCascadeClassifier ,base<Algorithm>>("BaseCascadeClassifier");

    emscripten::class_<cv::ml::TrainData >("ml_TrainData")
        .function("setTrainTestSplit", select_overload<void(cv::ml::TrainData&,int,bool)>(&Wrappers::ml_TrainData_setTrainTestSplit_wrapper), pure_virtual())
        .function("getTrainResponses", select_overload<Mat()const>(&cv::ml::TrainData::getTrainResponses), pure_virtual())
        .function("setTrainTestSplitRatio", select_overload<void(cv::ml::TrainData&,double,bool)>(&Wrappers::ml_TrainData_setTrainTestSplitRatio_wrapper), pure_virtual())
        .function("getValues", select_overload<void(cv::ml::TrainData&,int,const cv::Mat&,float*)>(&Wrappers::ml_TrainData_getValues_wrapper), allow_raw_pointers(), pure_virtual())
        .function("getNTestSamples", select_overload<int()const>(&cv::ml::TrainData::getNTestSamples), pure_virtual())
        .function("getTestResponses", select_overload<Mat()const>(&cv::ml::TrainData::getTestResponses), pure_virtual())
        .function("getNTrainSamples", select_overload<int()const>(&cv::ml::TrainData::getNTrainSamples), pure_virtual())
        .function("getVarIdx", select_overload<Mat()const>(&cv::ml::TrainData::getVarIdx), pure_virtual())
        .function("getTestNormCatResponses", select_overload<Mat()const>(&cv::ml::TrainData::getTestNormCatResponses), pure_virtual())
        .function("getSampleWeights", select_overload<Mat()const>(&cv::ml::TrainData::getSampleWeights), pure_virtual())
        .function("getClassLabels", select_overload<Mat()const>(&cv::ml::TrainData::getClassLabels), pure_virtual())
        .function("getNAllVars", select_overload<int()const>(&cv::ml::TrainData::getNAllVars), pure_virtual())
        .constructor(select_overload<Ptr<TrainData>(const cv::Mat&,int,const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::ml_TrainData_create_wrapper))
        .function("getResponses", select_overload<Mat()const>(&cv::ml::TrainData::getResponses), pure_virtual())
        .function("getNVars", select_overload<int()const>(&cv::ml::TrainData::getNVars), pure_virtual())
        .function("getSample", select_overload<void(cv::ml::TrainData&,const cv::Mat&,int,float*)>(&Wrappers::ml_TrainData_getSample_wrapper), allow_raw_pointers(), pure_virtual())
        .function("getCatCount", select_overload<int(cv::ml::TrainData&,int)>(&Wrappers::ml_TrainData_getCatCount_wrapper), pure_virtual())
        .function("getTestSampleIdx", select_overload<Mat()const>(&cv::ml::TrainData::getTestSampleIdx), pure_virtual())
        .function("getTrainNormCatResponses", select_overload<Mat()const>(&cv::ml::TrainData::getTrainNormCatResponses), pure_virtual())
        .function("getTrainSampleWeights", select_overload<Mat()const>(&cv::ml::TrainData::getTrainSampleWeights), pure_virtual())
        .class_function("getSubVector", select_overload<Mat(cv::ml::TrainData&,const Mat&,const Mat&)>(&Wrappers::ml_TrainData_getSubVector_wrapper))
        .function("getTestSampleWeights", select_overload<Mat()const>(&cv::ml::TrainData::getTestSampleWeights), pure_virtual())
        .function("shuffleTrainTest", select_overload<void()>(&cv::ml::TrainData::shuffleTrainTest), pure_virtual())
        .function("getTrainSamples", select_overload<Mat(cv::ml::TrainData&,int,bool,bool)>(&Wrappers::ml_TrainData_getTrainSamples_wrapper), pure_virtual())
        .function("getVarType", select_overload<Mat()const>(&cv::ml::TrainData::getVarType), pure_virtual())
        .function("getTrainSampleIdx", select_overload<Mat()const>(&cv::ml::TrainData::getTrainSampleIdx), pure_virtual())
        .function("getMissing", select_overload<Mat()const>(&cv::ml::TrainData::getMissing), pure_virtual())
        .function("getDefaultSubstValues", select_overload<Mat()const>(&cv::ml::TrainData::getDefaultSubstValues), pure_virtual())
        .function("getCatMap", select_overload<Mat()const>(&cv::ml::TrainData::getCatMap), pure_virtual())
        .function("getCatOfs", select_overload<Mat()const>(&cv::ml::TrainData::getCatOfs), pure_virtual())
        .function("getNSamples", select_overload<int()const>(&cv::ml::TrainData::getNSamples), pure_virtual())
        .function("getSamples", select_overload<Mat()const>(&cv::ml::TrainData::getSamples), pure_virtual())
        .function("getNormCatResponses", select_overload<Mat()const>(&cv::ml::TrainData::getNormCatResponses), pure_virtual())
        .function("getResponseType", select_overload<int()const>(&cv::ml::TrainData::getResponseType), pure_virtual())
        .function("getLayout", select_overload<int()const>(&cv::ml::TrainData::getLayout), pure_virtual())
        .smart_ptr<Ptr<cv::ml::TrainData>>("Ptr<ml_TrainData>")
;

    emscripten::class_<cv::MSER ,base<Feature2D>>("MSER")
        .function("getPass2Only", select_overload<bool()const>(&cv::MSER::getPass2Only), pure_virtual())
        .function("setMinArea", select_overload<void(cv::MSER&,int)>(&Wrappers::MSER_setMinArea_wrapper), pure_virtual())
        .function("getDelta", select_overload<int()const>(&cv::MSER::getDelta), pure_virtual())
        .constructor(select_overload<Ptr<MSER>(int,int,int,double,double,int,double,double,int)>(&Wrappers::MSER_create_wrapper))
        .function("getMaxArea", select_overload<int()const>(&cv::MSER::getMaxArea), pure_virtual())
        .function("setMaxArea", select_overload<void(cv::MSER&,int)>(&Wrappers::MSER_setMaxArea_wrapper), pure_virtual())
        .function("setPass2Only", select_overload<void(cv::MSER&,bool)>(&Wrappers::MSER_setPass2Only_wrapper), pure_virtual())
        .function("getMinArea", select_overload<int()const>(&cv::MSER::getMinArea), pure_virtual())
        .function("detectRegions", select_overload<void(cv::MSER&,const cv::Mat&,std::vector<std::vector<Point> >&,std::vector<Rect>&)>(&Wrappers::MSER_detectRegions_wrapper), pure_virtual())
        .function("setDelta", select_overload<void(cv::MSER&,int)>(&Wrappers::MSER_setDelta_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::MSER>>("Ptr<MSER>")
;

    emscripten::class_<cv::BackgroundSubtractorKNN ,base<BackgroundSubtractor>>("BackgroundSubtractorKNN")
        .function("getDetectShadows", select_overload<bool()const>(&cv::BackgroundSubtractorKNN::getDetectShadows), pure_virtual())
        .function("getkNNSamples", select_overload<int()const>(&cv::BackgroundSubtractorKNN::getkNNSamples), pure_virtual())
        .constructor(select_overload<Ptr<BackgroundSubtractorKNN>(int,double,bool)>(&Wrappers::_createBackgroundSubtractorKNN_wrapper))
        .function("setDetectShadows", select_overload<void(cv::BackgroundSubtractorKNN&,bool)>(&Wrappers::BackgroundSubtractorKNN_setDetectShadows_wrapper), pure_virtual())
        .function("getDist2Threshold", select_overload<double()const>(&cv::BackgroundSubtractorKNN::getDist2Threshold), pure_virtual())
        .function("setDist2Threshold", select_overload<void(cv::BackgroundSubtractorKNN&,double)>(&Wrappers::BackgroundSubtractorKNN_setDist2Threshold_wrapper), pure_virtual())
        .function("getNSamples", select_overload<int()const>(&cv::BackgroundSubtractorKNN::getNSamples), pure_virtual())
        .function("setkNNSamples", select_overload<void(cv::BackgroundSubtractorKNN&,int)>(&Wrappers::BackgroundSubtractorKNN_setkNNSamples_wrapper), pure_virtual())
        .function("getHistory", select_overload<int()const>(&cv::BackgroundSubtractorKNN::getHistory), pure_virtual())
        .function("setShadowValue", select_overload<void(cv::BackgroundSubtractorKNN&,int)>(&Wrappers::BackgroundSubtractorKNN_setShadowValue_wrapper), pure_virtual())
        .function("setHistory", select_overload<void(cv::BackgroundSubtractorKNN&,int)>(&Wrappers::BackgroundSubtractorKNN_setHistory_wrapper), pure_virtual())
        .function("setNSamples", select_overload<void(cv::BackgroundSubtractorKNN&,int)>(&Wrappers::BackgroundSubtractorKNN_setNSamples_wrapper), pure_virtual())
        .function("getShadowThreshold", select_overload<double()const>(&cv::BackgroundSubtractorKNN::getShadowThreshold), pure_virtual())
        .function("getShadowValue", select_overload<int()const>(&cv::BackgroundSubtractorKNN::getShadowValue), pure_virtual())
        .function("setShadowThreshold", select_overload<void(cv::BackgroundSubtractorKNN&,double)>(&Wrappers::BackgroundSubtractorKNN_setShadowThreshold_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::BackgroundSubtractorKNN>>("Ptr<BackgroundSubtractorKNN>")
;

    emscripten::class_<cv::DualTVL1OpticalFlow ,base<DenseOpticalFlow>>("DualTVL1OpticalFlow")
        .constructor(select_overload<Ptr<DualTVL1OpticalFlow>()>(&cv::createOptFlow_DualTVL1))
        .smart_ptr<Ptr<cv::DualTVL1OpticalFlow>>("Ptr<DualTVL1OpticalFlow>")
;

    emscripten::class_<cv::CLAHE ,base<Algorithm>>("CLAHE")
        .function("setTilesGridSize", select_overload<void(cv::CLAHE&,Size)>(&Wrappers::CLAHE_setTilesGridSize_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<CLAHE>(double,Size)>(&Wrappers::_createCLAHE_wrapper))
        .function("collectGarbage", select_overload<void()>(&cv::CLAHE::collectGarbage), pure_virtual())
        .function("setClipLimit", select_overload<void(cv::CLAHE&,double)>(&Wrappers::CLAHE_setClipLimit_wrapper), pure_virtual())
        .function("getTilesGridSize", select_overload<Size()const>(&cv::CLAHE::getTilesGridSize), pure_virtual())
        .function("getClipLimit", select_overload<double()const>(&cv::CLAHE::getClipLimit), pure_virtual())
        .function("apply", select_overload<void(cv::CLAHE&,const cv::Mat&,cv::Mat&)>(&Wrappers::CLAHE_apply_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::CLAHE>>("Ptr<CLAHE>")
;

    emscripten::class_<cv::SimpleBlobDetector::Params >("SimpleBlobDetector_Params")
        .constructor<>()
        .property("thresholdStep", &cv::SimpleBlobDetector::Params::thresholdStep)
        .property("minThreshold", &cv::SimpleBlobDetector::Params::minThreshold)
        .property("maxThreshold", &cv::SimpleBlobDetector::Params::maxThreshold)
        .property("minRepeatability", &cv::SimpleBlobDetector::Params::minRepeatability)
        .property("minDistBetweenBlobs", &cv::SimpleBlobDetector::Params::minDistBetweenBlobs)
        .property("filterByColor", &cv::SimpleBlobDetector::Params::filterByColor)
        .property("blobColor", &cv::SimpleBlobDetector::Params::blobColor)
        .property("filterByArea", &cv::SimpleBlobDetector::Params::filterByArea)
        .property("minArea", &cv::SimpleBlobDetector::Params::minArea)
        .property("maxArea", &cv::SimpleBlobDetector::Params::maxArea)
        .property("filterByCircularity", &cv::SimpleBlobDetector::Params::filterByCircularity)
        .property("minCircularity", &cv::SimpleBlobDetector::Params::minCircularity)
        .property("maxCircularity", &cv::SimpleBlobDetector::Params::maxCircularity)
        .property("filterByInertia", &cv::SimpleBlobDetector::Params::filterByInertia)
        .property("minInertiaRatio", &cv::SimpleBlobDetector::Params::minInertiaRatio)
        .property("maxInertiaRatio", &cv::SimpleBlobDetector::Params::maxInertiaRatio)
        .property("filterByConvexity", &cv::SimpleBlobDetector::Params::filterByConvexity)
        .property("minConvexity", &cv::SimpleBlobDetector::Params::minConvexity)
        .property("maxConvexity", &cv::SimpleBlobDetector::Params::maxConvexity);

    emscripten::class_<cv::Moments >("Moments")
        .property("m00", &cv::Moments::m00)
        .property("m10", &cv::Moments::m10)
        .property("m01", &cv::Moments::m01)
        .property("m20", &cv::Moments::m20)
        .property("m11", &cv::Moments::m11)
        .property("m02", &cv::Moments::m02)
        .property("m30", &cv::Moments::m30)
        .property("m21", &cv::Moments::m21)
        .property("m12", &cv::Moments::m12)
        .property("m03", &cv::Moments::m03)
        .property("mu20", &cv::Moments::mu20)
        .property("mu11", &cv::Moments::mu11)
        .property("mu02", &cv::Moments::mu02)
        .property("mu30", &cv::Moments::mu30)
        .property("mu21", &cv::Moments::mu21)
        .property("mu12", &cv::Moments::mu12)
        .property("mu03", &cv::Moments::mu03)
        .property("nu20", &cv::Moments::nu20)
        .property("nu11", &cv::Moments::nu11)
        .property("nu02", &cv::Moments::nu02)
        .property("nu30", &cv::Moments::nu30)
        .property("nu21", &cv::Moments::nu21)
        .property("nu12", &cv::Moments::nu12)
        .property("nu03", &cv::Moments::nu03);

    emscripten::class_<cv::SimpleBlobDetector ,base<Feature2D>>("SimpleBlobDetector")
        .constructor(select_overload<Ptr<SimpleBlobDetector>(const SimpleBlobDetector::Params &)>(&Wrappers::SimpleBlobDetector_create_wrapper))
        .smart_ptr<Ptr<cv::SimpleBlobDetector>>("Ptr<SimpleBlobDetector>")
;

    emscripten::class_<cv::ml::SVM ,base<ml::StatModel>>("ml_SVM")
        .function("getDegree", select_overload<double()const>(&cv::ml::SVM::getDegree), pure_virtual())
        .function("getC", select_overload<double()const>(&cv::ml::SVM::getC), pure_virtual())
        .function("setCoef0", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setCoef0_wrapper), pure_virtual())
        .function("getP", select_overload<double()const>(&cv::ml::SVM::getP), pure_virtual())
        .function("setNu", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setNu_wrapper), pure_virtual())
        .function("setType", select_overload<void(cv::ml::SVM&,int)>(&Wrappers::ml_SVM_setType_wrapper), pure_virtual())
        .function("getSupportVectors", select_overload<Mat()const>(&cv::ml::SVM::getSupportVectors), pure_virtual())
        .function("setDegree", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setDegree_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<SVM>()>(&cv::ml::SVM::create))
        .function("setC", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setC_wrapper), pure_virtual())
        .function("getType", select_overload<int()const>(&cv::ml::SVM::getType), pure_virtual())
        .function("setTermCriteria", select_overload<void(cv::ml::SVM&,const cv::TermCriteria &)>(&Wrappers::ml_SVM_setTermCriteria_wrapper), pure_virtual())
        .function("setGamma", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setGamma_wrapper), pure_virtual())
        .function("setP", select_overload<void(cv::ml::SVM&,double)>(&Wrappers::ml_SVM_setP_wrapper), pure_virtual())
        .function("getCoef0", select_overload<double()const>(&cv::ml::SVM::getCoef0), pure_virtual())
        .function("getGamma", select_overload<double()const>(&cv::ml::SVM::getGamma), pure_virtual())
        .function("getKernelType", select_overload<int()const>(&cv::ml::SVM::getKernelType), pure_virtual())
        .function("getTermCriteria", select_overload<cv::TermCriteria()const>(&cv::ml::SVM::getTermCriteria), pure_virtual())
        .function("getDecisionFunction", select_overload<double(cv::ml::SVM&,int,cv::Mat&,cv::Mat&)>(&Wrappers::ml_SVM_getDecisionFunction_wrapper), pure_virtual())
        .function("getClassWeights", select_overload<cv::Mat()const>(&cv::ml::SVM::getClassWeights), pure_virtual())
        .function("setKernel", select_overload<void(cv::ml::SVM&,int)>(&Wrappers::ml_SVM_setKernel_wrapper), pure_virtual())
        .function("setClassWeights", select_overload<void(cv::ml::SVM&,const cv::Mat &)>(&Wrappers::ml_SVM_setClassWeights_wrapper), pure_virtual())
        .function("getNu", select_overload<double()const>(&cv::ml::SVM::getNu), pure_virtual())
        .function("getUncompressedSupportVectors", select_overload<Mat()const>(&cv::ml::SVM::getUncompressedSupportVectors))
        .smart_ptr<Ptr<cv::ml::SVM>>("Ptr<ml_SVM>")
;

    emscripten::class_<cv::EMDL1HistogramCostExtractor ,base<HistogramCostExtractor>>("EMDL1HistogramCostExtractor")
        .constructor(select_overload<Ptr<HistogramCostExtractor>(int,float)>(&Wrappers::_createEMDL1HistogramCostExtractor_wrapper));

    emscripten::class_<cv::CascadeClassifier >("CascadeClassifier")
        .function("load", select_overload<bool(cv::CascadeClassifier&,const std::string&)>(&Wrappers::CascadeClassifier_load_wrapper))
        .function("getFeatureType", select_overload<int()const>(&cv::CascadeClassifier::getFeatureType))
        .class_function("convert", select_overload<bool(cv::CascadeClassifier&,const std::string&,const std::string&)>(&Wrappers::CascadeClassifier_convert_wrapper))
        .function("read", select_overload<bool(cv::CascadeClassifier&,const FileNode&)>(&Wrappers::CascadeClassifier_read_wrapper))
        .function("detectMultiScale2", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,std::vector<int>&,double,int,int,Size,Size)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper))
        .function("isOldFormatCascade", select_overload<bool()const>(&cv::CascadeClassifier::isOldFormatCascade))
        .constructor<>()
        .constructor<const String&>()
        .function("detectMultiScale3", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,std::vector<int>&,std::vector<double>&,double,int,int,Size,Size,bool)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper))
        .function("getOriginalWindowSize", select_overload<Size()const>(&cv::CascadeClassifier::getOriginalWindowSize))
        .function("empty", select_overload<bool()const>(&cv::CascadeClassifier::empty))
        .function("detectMultiScale", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,double,int,int,Size,Size)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper));

    emscripten::class_<cv::TonemapReinhard ,base<Tonemap>>("TonemapReinhard")
        .function("setColorAdaptation", select_overload<void(cv::TonemapReinhard&,float)>(&Wrappers::TonemapReinhard_setColorAdaptation_wrapper), pure_virtual())
        .function("setLightAdaptation", select_overload<void(cv::TonemapReinhard&,float)>(&Wrappers::TonemapReinhard_setLightAdaptation_wrapper), pure_virtual())
        .function("getIntensity", select_overload<float()const>(&cv::TonemapReinhard::getIntensity), pure_virtual())
        .function("getLightAdaptation", select_overload<float()const>(&cv::TonemapReinhard::getLightAdaptation), pure_virtual())
        .constructor(select_overload<Ptr<TonemapReinhard>(float,float,float,float)>(&Wrappers::_createTonemapReinhard_wrapper))
        .function("setIntensity", select_overload<void(cv::TonemapReinhard&,float)>(&Wrappers::TonemapReinhard_setIntensity_wrapper), pure_virtual())
        .function("getColorAdaptation", select_overload<float()const>(&cv::TonemapReinhard::getColorAdaptation), pure_virtual())
        .smart_ptr<Ptr<cv::TonemapReinhard>>("Ptr<TonemapReinhard>")
;

    emscripten::class_<cv::TonemapDrago ,base<Tonemap>>("TonemapDrago")
        .function("setBias", select_overload<void(cv::TonemapDrago&,float)>(&Wrappers::TonemapDrago_setBias_wrapper), pure_virtual())
        .function("getSaturation", select_overload<float()const>(&cv::TonemapDrago::getSaturation), pure_virtual())
        .constructor(select_overload<Ptr<TonemapDrago>(float,float,float)>(&Wrappers::_createTonemapDrago_wrapper))
        .function("getBias", select_overload<float()const>(&cv::TonemapDrago::getBias), pure_virtual())
        .function("setSaturation", select_overload<void(cv::TonemapDrago&,float)>(&Wrappers::TonemapDrago_setSaturation_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::TonemapDrago>>("Ptr<TonemapDrago>")
;

    emscripten::class_<cv::HausdorffDistanceExtractor ,base<ShapeDistanceExtractor>>("HausdorffDistanceExtractor")
        .function("setDistanceFlag", select_overload<void(cv::HausdorffDistanceExtractor&,int)>(&Wrappers::HausdorffDistanceExtractor_setDistanceFlag_wrapper), pure_virtual())
        .function("setRankProportion", select_overload<void(cv::HausdorffDistanceExtractor&,float)>(&Wrappers::HausdorffDistanceExtractor_setRankProportion_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<HausdorffDistanceExtractor>(int,float)>(&Wrappers::_createHausdorffDistanceExtractor_wrapper))
        .function("getDistanceFlag", select_overload<int()const>(&cv::HausdorffDistanceExtractor::getDistanceFlag), pure_virtual())
        .function("getRankProportion", select_overload<float()const>(&cv::HausdorffDistanceExtractor::getRankProportion), pure_virtual())
        .smart_ptr<Ptr<cv::HausdorffDistanceExtractor>>("Ptr<HausdorffDistanceExtractor>")
;

    emscripten::class_<cv::BOWTrainer >("BOWTrainer")
        .function("getDescriptors", select_overload<const std::vector<Mat>&()const>(&cv::BOWTrainer::getDescriptors))
        .function("cluster", select_overload<Mat(cv::BOWTrainer&)>(&Wrappers::BOWTrainer_cluster_wrapper), pure_virtual())
        .function("cluster1", select_overload<Mat(cv::BOWTrainer&,const Mat&)>(&Wrappers::BOWTrainer_cluster_wrapper1), pure_virtual())
        .function("add", select_overload<void(cv::BOWTrainer&,const Mat&)>(&Wrappers::BOWTrainer_add_wrapper))
        .function("clear", select_overload<void()>(&cv::BOWTrainer::clear))
        .function("descriptorsCount", select_overload<int()const>(&cv::BOWTrainer::descriptorsCount));

    emscripten::class_<cv::ORB ,base<Feature2D>>("ORB")
        .function("getMaxFeatures", select_overload<int()const>(&cv::ORB::getMaxFeatures), pure_virtual())
        .function("setEdgeThreshold", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setEdgeThreshold_wrapper), pure_virtual())
        .function("setFirstLevel", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setFirstLevel_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<ORB>(int,float,int,int,int,int,int,int,int)>(&Wrappers::ORB_create_wrapper))
        .function("getFastThreshold", select_overload<int()const>(&cv::ORB::getFastThreshold), pure_virtual())
        .function("setMaxFeatures", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setMaxFeatures_wrapper), pure_virtual())
        .function("getWTA_K", select_overload<int()const>(&cv::ORB::getWTA_K), pure_virtual())
        .function("getNLevels", select_overload<int()const>(&cv::ORB::getNLevels), pure_virtual())
        .function("setNLevels", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setNLevels_wrapper), pure_virtual())
        .function("setFastThreshold", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setFastThreshold_wrapper), pure_virtual())
        .function("getScaleFactor", select_overload<double()const>(&cv::ORB::getScaleFactor), pure_virtual())
        .function("setPatchSize", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setPatchSize_wrapper), pure_virtual())
        .function("setWTA_K", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setWTA_K_wrapper), pure_virtual())
        .function("setScaleFactor", select_overload<void(cv::ORB&,double)>(&Wrappers::ORB_setScaleFactor_wrapper), pure_virtual())
        .function("getPatchSize", select_overload<int()const>(&cv::ORB::getPatchSize), pure_virtual())
        .function("getEdgeThreshold", select_overload<int()const>(&cv::ORB::getEdgeThreshold), pure_virtual())
        .function("getFirstLevel", select_overload<int()const>(&cv::ORB::getFirstLevel), pure_virtual())
        .function("getScoreType", select_overload<int()const>(&cv::ORB::getScoreType), pure_virtual())
        .function("setScoreType", select_overload<void(cv::ORB&,int)>(&Wrappers::ORB_setScoreType_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ORB>>("Ptr<ORB>")
;

    emscripten::class_<cv::ml::EM ,base<ml::StatModel>>("ml_EM")
        .function("trainE", select_overload<bool(cv::ml::EM&,const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::ml_EM_trainE_wrapper), pure_virtual())
        .function("getTermCriteria", select_overload<TermCriteria()const>(&cv::ml::EM::getTermCriteria), pure_virtual())
        .function("getCovs", select_overload<void(cv::ml::EM&,std::vector<Mat>&)>(&Wrappers::ml_EM_getCovs_wrapper), pure_virtual())
        .function("getWeights", select_overload<Mat()const>(&cv::ml::EM::getWeights), pure_virtual())
        .function("setClustersNumber", select_overload<void(cv::ml::EM&,int)>(&Wrappers::ml_EM_setClustersNumber_wrapper), pure_virtual())
        .function("trainM", select_overload<bool(cv::ml::EM&,const cv::Mat&,const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::ml_EM_trainM_wrapper), pure_virtual())
        .function("predict2", select_overload<Vec2d(cv::ml::EM&,const cv::Mat&,cv::Mat&)>(&Wrappers::ml_EM_predict2_wrapper), pure_virtual())
        .function("getClustersNumber", select_overload<int()const>(&cv::ml::EM::getClustersNumber), pure_virtual())
        .function("setTermCriteria", select_overload<void(cv::ml::EM&,const TermCriteria &)>(&Wrappers::ml_EM_setTermCriteria_wrapper), pure_virtual())
        .function("getCovarianceMatrixType", select_overload<int()const>(&cv::ml::EM::getCovarianceMatrixType), pure_virtual())
        .function("setCovarianceMatrixType", select_overload<void(cv::ml::EM&,int)>(&Wrappers::ml_EM_setCovarianceMatrixType_wrapper), pure_virtual())
        .constructor(select_overload<Ptr<EM>()>(&cv::ml::EM::create))
        .function("getMeans", select_overload<Mat()const>(&cv::ml::EM::getMeans), pure_virtual())
        .function("trainEM", select_overload<bool(cv::ml::EM&,const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::ml_EM_trainEM_wrapper), pure_virtual())
        .smart_ptr<Ptr<cv::ml::EM>>("Ptr<ml_EM>")
;

    emscripten::enum_<AdaptiveThresholdTypes>("AdaptiveThresholdTypes")
        .value("ADAPTIVE_THRESH_MEAN_C", AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C)
        .value("ADAPTIVE_THRESH_GAUSSIAN_C", AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C);

    emscripten::enum_<ColorConversionCodes>("ColorConversionCodes")
        .value("COLOR_BGR2BGRA", ColorConversionCodes::COLOR_BGR2BGRA)
        .value("COLOR_RGB2RGBA", ColorConversionCodes::COLOR_RGB2RGBA)
        .value("COLOR_BGRA2BGR", ColorConversionCodes::COLOR_BGRA2BGR)
        .value("COLOR_RGBA2RGB", ColorConversionCodes::COLOR_RGBA2RGB)
        .value("COLOR_BGR2RGBA", ColorConversionCodes::COLOR_BGR2RGBA)
        .value("COLOR_RGB2BGRA", ColorConversionCodes::COLOR_RGB2BGRA)
        .value("COLOR_RGBA2BGR", ColorConversionCodes::COLOR_RGBA2BGR)
        .value("COLOR_BGRA2RGB", ColorConversionCodes::COLOR_BGRA2RGB)
        .value("COLOR_BGR2RGB", ColorConversionCodes::COLOR_BGR2RGB)
        .value("COLOR_RGB2BGR", ColorConversionCodes::COLOR_RGB2BGR)
        .value("COLOR_BGRA2RGBA", ColorConversionCodes::COLOR_BGRA2RGBA)
        .value("COLOR_RGBA2BGRA", ColorConversionCodes::COLOR_RGBA2BGRA)
        .value("COLOR_BGR2GRAY", ColorConversionCodes::COLOR_BGR2GRAY)
        .value("COLOR_RGB2GRAY", ColorConversionCodes::COLOR_RGB2GRAY)
        .value("COLOR_GRAY2BGR", ColorConversionCodes::COLOR_GRAY2BGR)
        .value("COLOR_GRAY2RGB", ColorConversionCodes::COLOR_GRAY2RGB)
        .value("COLOR_GRAY2BGRA", ColorConversionCodes::COLOR_GRAY2BGRA)
        .value("COLOR_GRAY2RGBA", ColorConversionCodes::COLOR_GRAY2RGBA)
        .value("COLOR_BGRA2GRAY", ColorConversionCodes::COLOR_BGRA2GRAY)
        .value("COLOR_RGBA2GRAY", ColorConversionCodes::COLOR_RGBA2GRAY)
        .value("COLOR_BGR2BGR565", ColorConversionCodes::COLOR_BGR2BGR565)
        .value("COLOR_RGB2BGR565", ColorConversionCodes::COLOR_RGB2BGR565)
        .value("COLOR_BGR5652BGR", ColorConversionCodes::COLOR_BGR5652BGR)
        .value("COLOR_BGR5652RGB", ColorConversionCodes::COLOR_BGR5652RGB)
        .value("COLOR_BGRA2BGR565", ColorConversionCodes::COLOR_BGRA2BGR565)
        .value("COLOR_RGBA2BGR565", ColorConversionCodes::COLOR_RGBA2BGR565)
        .value("COLOR_BGR5652BGRA", ColorConversionCodes::COLOR_BGR5652BGRA)
        .value("COLOR_BGR5652RGBA", ColorConversionCodes::COLOR_BGR5652RGBA)
        .value("COLOR_GRAY2BGR565", ColorConversionCodes::COLOR_GRAY2BGR565)
        .value("COLOR_BGR5652GRAY", ColorConversionCodes::COLOR_BGR5652GRAY)
        .value("COLOR_BGR2BGR555", ColorConversionCodes::COLOR_BGR2BGR555)
        .value("COLOR_RGB2BGR555", ColorConversionCodes::COLOR_RGB2BGR555)
        .value("COLOR_BGR5552BGR", ColorConversionCodes::COLOR_BGR5552BGR)
        .value("COLOR_BGR5552RGB", ColorConversionCodes::COLOR_BGR5552RGB)
        .value("COLOR_BGRA2BGR555", ColorConversionCodes::COLOR_BGRA2BGR555)
        .value("COLOR_RGBA2BGR555", ColorConversionCodes::COLOR_RGBA2BGR555)
        .value("COLOR_BGR5552BGRA", ColorConversionCodes::COLOR_BGR5552BGRA)
        .value("COLOR_BGR5552RGBA", ColorConversionCodes::COLOR_BGR5552RGBA)
        .value("COLOR_GRAY2BGR555", ColorConversionCodes::COLOR_GRAY2BGR555)
        .value("COLOR_BGR5552GRAY", ColorConversionCodes::COLOR_BGR5552GRAY)
        .value("COLOR_BGR2XYZ", ColorConversionCodes::COLOR_BGR2XYZ)
        .value("COLOR_RGB2XYZ", ColorConversionCodes::COLOR_RGB2XYZ)
        .value("COLOR_XYZ2BGR", ColorConversionCodes::COLOR_XYZ2BGR)
        .value("COLOR_XYZ2RGB", ColorConversionCodes::COLOR_XYZ2RGB)
        .value("COLOR_BGR2YCrCb", ColorConversionCodes::COLOR_BGR2YCrCb)
        .value("COLOR_RGB2YCrCb", ColorConversionCodes::COLOR_RGB2YCrCb)
        .value("COLOR_YCrCb2BGR", ColorConversionCodes::COLOR_YCrCb2BGR)
        .value("COLOR_YCrCb2RGB", ColorConversionCodes::COLOR_YCrCb2RGB)
        .value("COLOR_BGR2HSV", ColorConversionCodes::COLOR_BGR2HSV)
        .value("COLOR_RGB2HSV", ColorConversionCodes::COLOR_RGB2HSV)
        .value("COLOR_BGR2Lab", ColorConversionCodes::COLOR_BGR2Lab)
        .value("COLOR_RGB2Lab", ColorConversionCodes::COLOR_RGB2Lab)
        .value("COLOR_BGR2Luv", ColorConversionCodes::COLOR_BGR2Luv)
        .value("COLOR_RGB2Luv", ColorConversionCodes::COLOR_RGB2Luv)
        .value("COLOR_BGR2HLS", ColorConversionCodes::COLOR_BGR2HLS)
        .value("COLOR_RGB2HLS", ColorConversionCodes::COLOR_RGB2HLS)
        .value("COLOR_HSV2BGR", ColorConversionCodes::COLOR_HSV2BGR)
        .value("COLOR_HSV2RGB", ColorConversionCodes::COLOR_HSV2RGB)
        .value("COLOR_Lab2BGR", ColorConversionCodes::COLOR_Lab2BGR)
        .value("COLOR_Lab2RGB", ColorConversionCodes::COLOR_Lab2RGB)
        .value("COLOR_Luv2BGR", ColorConversionCodes::COLOR_Luv2BGR)
        .value("COLOR_Luv2RGB", ColorConversionCodes::COLOR_Luv2RGB)
        .value("COLOR_HLS2BGR", ColorConversionCodes::COLOR_HLS2BGR)
        .value("COLOR_HLS2RGB", ColorConversionCodes::COLOR_HLS2RGB)
        .value("COLOR_BGR2HSV_FULL", ColorConversionCodes::COLOR_BGR2HSV_FULL)
        .value("COLOR_RGB2HSV_FULL", ColorConversionCodes::COLOR_RGB2HSV_FULL)
        .value("COLOR_BGR2HLS_FULL", ColorConversionCodes::COLOR_BGR2HLS_FULL)
        .value("COLOR_RGB2HLS_FULL", ColorConversionCodes::COLOR_RGB2HLS_FULL)
        .value("COLOR_HSV2BGR_FULL", ColorConversionCodes::COLOR_HSV2BGR_FULL)
        .value("COLOR_HSV2RGB_FULL", ColorConversionCodes::COLOR_HSV2RGB_FULL)
        .value("COLOR_HLS2BGR_FULL", ColorConversionCodes::COLOR_HLS2BGR_FULL)
        .value("COLOR_HLS2RGB_FULL", ColorConversionCodes::COLOR_HLS2RGB_FULL)
        .value("COLOR_LBGR2Lab", ColorConversionCodes::COLOR_LBGR2Lab)
        .value("COLOR_LRGB2Lab", ColorConversionCodes::COLOR_LRGB2Lab)
        .value("COLOR_LBGR2Luv", ColorConversionCodes::COLOR_LBGR2Luv)
        .value("COLOR_LRGB2Luv", ColorConversionCodes::COLOR_LRGB2Luv)
        .value("COLOR_Lab2LBGR", ColorConversionCodes::COLOR_Lab2LBGR)
        .value("COLOR_Lab2LRGB", ColorConversionCodes::COLOR_Lab2LRGB)
        .value("COLOR_Luv2LBGR", ColorConversionCodes::COLOR_Luv2LBGR)
        .value("COLOR_Luv2LRGB", ColorConversionCodes::COLOR_Luv2LRGB)
        .value("COLOR_BGR2YUV", ColorConversionCodes::COLOR_BGR2YUV)
        .value("COLOR_RGB2YUV", ColorConversionCodes::COLOR_RGB2YUV)
        .value("COLOR_YUV2BGR", ColorConversionCodes::COLOR_YUV2BGR)
        .value("COLOR_YUV2RGB", ColorConversionCodes::COLOR_YUV2RGB)
        .value("COLOR_YUV2RGB_NV12", ColorConversionCodes::COLOR_YUV2RGB_NV12)
        .value("COLOR_YUV2BGR_NV12", ColorConversionCodes::COLOR_YUV2BGR_NV12)
        .value("COLOR_YUV2RGB_NV21", ColorConversionCodes::COLOR_YUV2RGB_NV21)
        .value("COLOR_YUV2BGR_NV21", ColorConversionCodes::COLOR_YUV2BGR_NV21)
        .value("COLOR_YUV420sp2RGB", ColorConversionCodes::COLOR_YUV420sp2RGB)
        .value("COLOR_YUV420sp2BGR", ColorConversionCodes::COLOR_YUV420sp2BGR)
        .value("COLOR_YUV2RGBA_NV12", ColorConversionCodes::COLOR_YUV2RGBA_NV12)
        .value("COLOR_YUV2BGRA_NV12", ColorConversionCodes::COLOR_YUV2BGRA_NV12)
        .value("COLOR_YUV2RGBA_NV21", ColorConversionCodes::COLOR_YUV2RGBA_NV21)
        .value("COLOR_YUV2BGRA_NV21", ColorConversionCodes::COLOR_YUV2BGRA_NV21)
        .value("COLOR_YUV420sp2RGBA", ColorConversionCodes::COLOR_YUV420sp2RGBA)
        .value("COLOR_YUV420sp2BGRA", ColorConversionCodes::COLOR_YUV420sp2BGRA)
        .value("COLOR_YUV2RGB_YV12", ColorConversionCodes::COLOR_YUV2RGB_YV12)
        .value("COLOR_YUV2BGR_YV12", ColorConversionCodes::COLOR_YUV2BGR_YV12)
        .value("COLOR_YUV2RGB_IYUV", ColorConversionCodes::COLOR_YUV2RGB_IYUV)
        .value("COLOR_YUV2BGR_IYUV", ColorConversionCodes::COLOR_YUV2BGR_IYUV)
        .value("COLOR_YUV2RGB_I420", ColorConversionCodes::COLOR_YUV2RGB_I420)
        .value("COLOR_YUV2BGR_I420", ColorConversionCodes::COLOR_YUV2BGR_I420)
        .value("COLOR_YUV420p2RGB", ColorConversionCodes::COLOR_YUV420p2RGB)
        .value("COLOR_YUV420p2BGR", ColorConversionCodes::COLOR_YUV420p2BGR)
        .value("COLOR_YUV2RGBA_YV12", ColorConversionCodes::COLOR_YUV2RGBA_YV12)
        .value("COLOR_YUV2BGRA_YV12", ColorConversionCodes::COLOR_YUV2BGRA_YV12)
        .value("COLOR_YUV2RGBA_IYUV", ColorConversionCodes::COLOR_YUV2RGBA_IYUV)
        .value("COLOR_YUV2BGRA_IYUV", ColorConversionCodes::COLOR_YUV2BGRA_IYUV)
        .value("COLOR_YUV2RGBA_I420", ColorConversionCodes::COLOR_YUV2RGBA_I420)
        .value("COLOR_YUV2BGRA_I420", ColorConversionCodes::COLOR_YUV2BGRA_I420)
        .value("COLOR_YUV420p2RGBA", ColorConversionCodes::COLOR_YUV420p2RGBA)
        .value("COLOR_YUV420p2BGRA", ColorConversionCodes::COLOR_YUV420p2BGRA)
        .value("COLOR_YUV2GRAY_420", ColorConversionCodes::COLOR_YUV2GRAY_420)
        .value("COLOR_YUV2GRAY_NV21", ColorConversionCodes::COLOR_YUV2GRAY_NV21)
        .value("COLOR_YUV2GRAY_NV12", ColorConversionCodes::COLOR_YUV2GRAY_NV12)
        .value("COLOR_YUV2GRAY_YV12", ColorConversionCodes::COLOR_YUV2GRAY_YV12)
        .value("COLOR_YUV2GRAY_IYUV", ColorConversionCodes::COLOR_YUV2GRAY_IYUV)
        .value("COLOR_YUV2GRAY_I420", ColorConversionCodes::COLOR_YUV2GRAY_I420)
        .value("COLOR_YUV420sp2GRAY", ColorConversionCodes::COLOR_YUV420sp2GRAY)
        .value("COLOR_YUV420p2GRAY", ColorConversionCodes::COLOR_YUV420p2GRAY)
        .value("COLOR_YUV2RGB_UYVY", ColorConversionCodes::COLOR_YUV2RGB_UYVY)
        .value("COLOR_YUV2BGR_UYVY", ColorConversionCodes::COLOR_YUV2BGR_UYVY)
        .value("COLOR_YUV2RGB_Y422", ColorConversionCodes::COLOR_YUV2RGB_Y422)
        .value("COLOR_YUV2BGR_Y422", ColorConversionCodes::COLOR_YUV2BGR_Y422)
        .value("COLOR_YUV2RGB_UYNV", ColorConversionCodes::COLOR_YUV2RGB_UYNV)
        .value("COLOR_YUV2BGR_UYNV", ColorConversionCodes::COLOR_YUV2BGR_UYNV)
        .value("COLOR_YUV2RGBA_UYVY", ColorConversionCodes::COLOR_YUV2RGBA_UYVY)
        .value("COLOR_YUV2BGRA_UYVY", ColorConversionCodes::COLOR_YUV2BGRA_UYVY)
        .value("COLOR_YUV2RGBA_Y422", ColorConversionCodes::COLOR_YUV2RGBA_Y422)
        .value("COLOR_YUV2BGRA_Y422", ColorConversionCodes::COLOR_YUV2BGRA_Y422)
        .value("COLOR_YUV2RGBA_UYNV", ColorConversionCodes::COLOR_YUV2RGBA_UYNV)
        .value("COLOR_YUV2BGRA_UYNV", ColorConversionCodes::COLOR_YUV2BGRA_UYNV)
        .value("COLOR_YUV2RGB_YUY2", ColorConversionCodes::COLOR_YUV2RGB_YUY2)
        .value("COLOR_YUV2BGR_YUY2", ColorConversionCodes::COLOR_YUV2BGR_YUY2)
        .value("COLOR_YUV2RGB_YVYU", ColorConversionCodes::COLOR_YUV2RGB_YVYU)
        .value("COLOR_YUV2BGR_YVYU", ColorConversionCodes::COLOR_YUV2BGR_YVYU)
        .value("COLOR_YUV2RGB_YUYV", ColorConversionCodes::COLOR_YUV2RGB_YUYV)
        .value("COLOR_YUV2BGR_YUYV", ColorConversionCodes::COLOR_YUV2BGR_YUYV)
        .value("COLOR_YUV2RGB_YUNV", ColorConversionCodes::COLOR_YUV2RGB_YUNV)
        .value("COLOR_YUV2BGR_YUNV", ColorConversionCodes::COLOR_YUV2BGR_YUNV)
        .value("COLOR_YUV2RGBA_YUY2", ColorConversionCodes::COLOR_YUV2RGBA_YUY2)
        .value("COLOR_YUV2BGRA_YUY2", ColorConversionCodes::COLOR_YUV2BGRA_YUY2)
        .value("COLOR_YUV2RGBA_YVYU", ColorConversionCodes::COLOR_YUV2RGBA_YVYU)
        .value("COLOR_YUV2BGRA_YVYU", ColorConversionCodes::COLOR_YUV2BGRA_YVYU)
        .value("COLOR_YUV2RGBA_YUYV", ColorConversionCodes::COLOR_YUV2RGBA_YUYV)
        .value("COLOR_YUV2BGRA_YUYV", ColorConversionCodes::COLOR_YUV2BGRA_YUYV)
        .value("COLOR_YUV2RGBA_YUNV", ColorConversionCodes::COLOR_YUV2RGBA_YUNV)
        .value("COLOR_YUV2BGRA_YUNV", ColorConversionCodes::COLOR_YUV2BGRA_YUNV)
        .value("COLOR_YUV2GRAY_UYVY", ColorConversionCodes::COLOR_YUV2GRAY_UYVY)
        .value("COLOR_YUV2GRAY_YUY2", ColorConversionCodes::COLOR_YUV2GRAY_YUY2)
        .value("COLOR_YUV2GRAY_Y422", ColorConversionCodes::COLOR_YUV2GRAY_Y422)
        .value("COLOR_YUV2GRAY_UYNV", ColorConversionCodes::COLOR_YUV2GRAY_UYNV)
        .value("COLOR_YUV2GRAY_YVYU", ColorConversionCodes::COLOR_YUV2GRAY_YVYU)
        .value("COLOR_YUV2GRAY_YUYV", ColorConversionCodes::COLOR_YUV2GRAY_YUYV)
        .value("COLOR_YUV2GRAY_YUNV", ColorConversionCodes::COLOR_YUV2GRAY_YUNV)
        .value("COLOR_RGBA2mRGBA", ColorConversionCodes::COLOR_RGBA2mRGBA)
        .value("COLOR_mRGBA2RGBA", ColorConversionCodes::COLOR_mRGBA2RGBA)
        .value("COLOR_RGB2YUV_I420", ColorConversionCodes::COLOR_RGB2YUV_I420)
        .value("COLOR_BGR2YUV_I420", ColorConversionCodes::COLOR_BGR2YUV_I420)
        .value("COLOR_RGB2YUV_IYUV", ColorConversionCodes::COLOR_RGB2YUV_IYUV)
        .value("COLOR_BGR2YUV_IYUV", ColorConversionCodes::COLOR_BGR2YUV_IYUV)
        .value("COLOR_RGBA2YUV_I420", ColorConversionCodes::COLOR_RGBA2YUV_I420)
        .value("COLOR_BGRA2YUV_I420", ColorConversionCodes::COLOR_BGRA2YUV_I420)
        .value("COLOR_RGBA2YUV_IYUV", ColorConversionCodes::COLOR_RGBA2YUV_IYUV)
        .value("COLOR_BGRA2YUV_IYUV", ColorConversionCodes::COLOR_BGRA2YUV_IYUV)
        .value("COLOR_RGB2YUV_YV12", ColorConversionCodes::COLOR_RGB2YUV_YV12)
        .value("COLOR_BGR2YUV_YV12", ColorConversionCodes::COLOR_BGR2YUV_YV12)
        .value("COLOR_RGBA2YUV_YV12", ColorConversionCodes::COLOR_RGBA2YUV_YV12)
        .value("COLOR_BGRA2YUV_YV12", ColorConversionCodes::COLOR_BGRA2YUV_YV12)
        .value("COLOR_BayerBG2BGR", ColorConversionCodes::COLOR_BayerBG2BGR)
        .value("COLOR_BayerGB2BGR", ColorConversionCodes::COLOR_BayerGB2BGR)
        .value("COLOR_BayerRG2BGR", ColorConversionCodes::COLOR_BayerRG2BGR)
        .value("COLOR_BayerGR2BGR", ColorConversionCodes::COLOR_BayerGR2BGR)
        .value("COLOR_BayerBG2RGB", ColorConversionCodes::COLOR_BayerBG2RGB)
        .value("COLOR_BayerGB2RGB", ColorConversionCodes::COLOR_BayerGB2RGB)
        .value("COLOR_BayerRG2RGB", ColorConversionCodes::COLOR_BayerRG2RGB)
        .value("COLOR_BayerGR2RGB", ColorConversionCodes::COLOR_BayerGR2RGB)
        .value("COLOR_BayerBG2GRAY", ColorConversionCodes::COLOR_BayerBG2GRAY)
        .value("COLOR_BayerGB2GRAY", ColorConversionCodes::COLOR_BayerGB2GRAY)
        .value("COLOR_BayerRG2GRAY", ColorConversionCodes::COLOR_BayerRG2GRAY)
        .value("COLOR_BayerGR2GRAY", ColorConversionCodes::COLOR_BayerGR2GRAY)
        .value("COLOR_BayerBG2BGR_VNG", ColorConversionCodes::COLOR_BayerBG2BGR_VNG)
        .value("COLOR_BayerGB2BGR_VNG", ColorConversionCodes::COLOR_BayerGB2BGR_VNG)
        .value("COLOR_BayerRG2BGR_VNG", ColorConversionCodes::COLOR_BayerRG2BGR_VNG)
        .value("COLOR_BayerGR2BGR_VNG", ColorConversionCodes::COLOR_BayerGR2BGR_VNG)
        .value("COLOR_BayerBG2RGB_VNG", ColorConversionCodes::COLOR_BayerBG2RGB_VNG)
        .value("COLOR_BayerGB2RGB_VNG", ColorConversionCodes::COLOR_BayerGB2RGB_VNG)
        .value("COLOR_BayerRG2RGB_VNG", ColorConversionCodes::COLOR_BayerRG2RGB_VNG)
        .value("COLOR_BayerGR2RGB_VNG", ColorConversionCodes::COLOR_BayerGR2RGB_VNG)
        .value("COLOR_BayerBG2BGR_EA", ColorConversionCodes::COLOR_BayerBG2BGR_EA)
        .value("COLOR_BayerGB2BGR_EA", ColorConversionCodes::COLOR_BayerGB2BGR_EA)
        .value("COLOR_BayerRG2BGR_EA", ColorConversionCodes::COLOR_BayerRG2BGR_EA)
        .value("COLOR_BayerGR2BGR_EA", ColorConversionCodes::COLOR_BayerGR2BGR_EA)
        .value("COLOR_BayerBG2RGB_EA", ColorConversionCodes::COLOR_BayerBG2RGB_EA)
        .value("COLOR_BayerGB2RGB_EA", ColorConversionCodes::COLOR_BayerGB2RGB_EA)
        .value("COLOR_BayerRG2RGB_EA", ColorConversionCodes::COLOR_BayerRG2RGB_EA)
        .value("COLOR_BayerGR2RGB_EA", ColorConversionCodes::COLOR_BayerGR2RGB_EA)
        .value("COLOR_COLORCVT_MAX", ColorConversionCodes::COLOR_COLORCVT_MAX);

    emscripten::enum_<ColormapTypes>("ColormapTypes")
        .value("COLORMAP_AUTUMN", ColormapTypes::COLORMAP_AUTUMN)
        .value("COLORMAP_BONE", ColormapTypes::COLORMAP_BONE)
        .value("COLORMAP_JET", ColormapTypes::COLORMAP_JET)
        .value("COLORMAP_WINTER", ColormapTypes::COLORMAP_WINTER)
        .value("COLORMAP_RAINBOW", ColormapTypes::COLORMAP_RAINBOW)
        .value("COLORMAP_OCEAN", ColormapTypes::COLORMAP_OCEAN)
        .value("COLORMAP_SUMMER", ColormapTypes::COLORMAP_SUMMER)
        .value("COLORMAP_SPRING", ColormapTypes::COLORMAP_SPRING)
        .value("COLORMAP_COOL", ColormapTypes::COLORMAP_COOL)
        .value("COLORMAP_HSV", ColormapTypes::COLORMAP_HSV)
        .value("COLORMAP_PINK", ColormapTypes::COLORMAP_PINK)
        .value("COLORMAP_HOT", ColormapTypes::COLORMAP_HOT)
        .value("COLORMAP_PARULA", ColormapTypes::COLORMAP_PARULA);

    emscripten::enum_<ConnectedComponentsTypes>("ConnectedComponentsTypes")
        .value("CC_STAT_LEFT", ConnectedComponentsTypes::CC_STAT_LEFT)
        .value("CC_STAT_TOP", ConnectedComponentsTypes::CC_STAT_TOP)
        .value("CC_STAT_WIDTH", ConnectedComponentsTypes::CC_STAT_WIDTH)
        .value("CC_STAT_HEIGHT", ConnectedComponentsTypes::CC_STAT_HEIGHT)
        .value("CC_STAT_AREA", ConnectedComponentsTypes::CC_STAT_AREA)
        .value("CC_STAT_MAX", ConnectedComponentsTypes::CC_STAT_MAX);

    emscripten::enum_<ContourApproximationModes>("ContourApproximationModes")
        .value("CHAIN_APPROX_NONE", ContourApproximationModes::CHAIN_APPROX_NONE)
        .value("CHAIN_APPROX_SIMPLE", ContourApproximationModes::CHAIN_APPROX_SIMPLE)
        .value("CHAIN_APPROX_TC89_L1", ContourApproximationModes::CHAIN_APPROX_TC89_L1)
        .value("CHAIN_APPROX_TC89_KCOS", ContourApproximationModes::CHAIN_APPROX_TC89_KCOS);

    emscripten::enum_<CovarFlags>("CovarFlags")
        .value("COVAR_SCRAMBLED", CovarFlags::COVAR_SCRAMBLED)
        .value("COVAR_NORMAL", CovarFlags::COVAR_NORMAL)
        .value("COVAR_USE_AVG", CovarFlags::COVAR_USE_AVG)
        .value("COVAR_SCALE", CovarFlags::COVAR_SCALE)
        .value("COVAR_ROWS", CovarFlags::COVAR_ROWS)
        .value("COVAR_COLS", CovarFlags::COVAR_COLS);

    emscripten::enum_<DistanceTransformLabelTypes>("DistanceTransformLabelTypes")
        .value("DIST_LABEL_CCOMP", DistanceTransformLabelTypes::DIST_LABEL_CCOMP)
        .value("DIST_LABEL_PIXEL", DistanceTransformLabelTypes::DIST_LABEL_PIXEL);

    emscripten::enum_<DistanceTransformMasks>("DistanceTransformMasks")
        .value("DIST_MASK_3", DistanceTransformMasks::DIST_MASK_3)
        .value("DIST_MASK_5", DistanceTransformMasks::DIST_MASK_5)
        .value("DIST_MASK_PRECISE", DistanceTransformMasks::DIST_MASK_PRECISE);

    emscripten::enum_<DistanceTypes>("DistanceTypes")
        .value("DIST_USER", DistanceTypes::DIST_USER)
        .value("DIST_L1", DistanceTypes::DIST_L1)
        .value("DIST_L2", DistanceTypes::DIST_L2)
        .value("DIST_C", DistanceTypes::DIST_C)
        .value("DIST_L12", DistanceTypes::DIST_L12)
        .value("DIST_FAIR", DistanceTypes::DIST_FAIR)
        .value("DIST_WELSCH", DistanceTypes::DIST_WELSCH)
        .value("DIST_HUBER", DistanceTypes::DIST_HUBER);

    emscripten::enum_<FloodFillFlags>("FloodFillFlags")
        .value("FLOODFILL_FIXED_RANGE", FloodFillFlags::FLOODFILL_FIXED_RANGE)
        .value("FLOODFILL_MASK_ONLY", FloodFillFlags::FLOODFILL_MASK_ONLY);

    emscripten::enum_<GrabCutClasses>("GrabCutClasses")
        .value("GC_BGD", GrabCutClasses::GC_BGD)
        .value("GC_FGD", GrabCutClasses::GC_FGD)
        .value("GC_PR_BGD", GrabCutClasses::GC_PR_BGD)
        .value("GC_PR_FGD", GrabCutClasses::GC_PR_FGD);

    emscripten::enum_<GrabCutModes>("GrabCutModes")
        .value("GC_INIT_WITH_RECT", GrabCutModes::GC_INIT_WITH_RECT)
        .value("GC_INIT_WITH_MASK", GrabCutModes::GC_INIT_WITH_MASK)
        .value("GC_EVAL", GrabCutModes::GC_EVAL);

    emscripten::enum_<HersheyFonts>("HersheyFonts")
        .value("FONT_HERSHEY_SIMPLEX", HersheyFonts::FONT_HERSHEY_SIMPLEX)
        .value("FONT_HERSHEY_PLAIN", HersheyFonts::FONT_HERSHEY_PLAIN)
        .value("FONT_HERSHEY_DUPLEX", HersheyFonts::FONT_HERSHEY_DUPLEX)
        .value("FONT_HERSHEY_COMPLEX", HersheyFonts::FONT_HERSHEY_COMPLEX)
        .value("FONT_HERSHEY_TRIPLEX", HersheyFonts::FONT_HERSHEY_TRIPLEX)
        .value("FONT_HERSHEY_COMPLEX_SMALL", HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL)
        .value("FONT_HERSHEY_SCRIPT_SIMPLEX", HersheyFonts::FONT_HERSHEY_SCRIPT_SIMPLEX)
        .value("FONT_HERSHEY_SCRIPT_COMPLEX", HersheyFonts::FONT_HERSHEY_SCRIPT_COMPLEX)
        .value("FONT_ITALIC", HersheyFonts::FONT_ITALIC);

    emscripten::enum_<HistCompMethods>("HistCompMethods")
        .value("HISTCMP_CORREL", HistCompMethods::HISTCMP_CORREL)
        .value("HISTCMP_CHISQR", HistCompMethods::HISTCMP_CHISQR)
        .value("HISTCMP_INTERSECT", HistCompMethods::HISTCMP_INTERSECT)
        .value("HISTCMP_BHATTACHARYYA", HistCompMethods::HISTCMP_BHATTACHARYYA)
        .value("HISTCMP_HELLINGER", HistCompMethods::HISTCMP_HELLINGER)
        .value("HISTCMP_CHISQR_ALT", HistCompMethods::HISTCMP_CHISQR_ALT)
        .value("HISTCMP_KL_DIV", HistCompMethods::HISTCMP_KL_DIV);

    emscripten::enum_<HoughModes>("HoughModes")
        .value("HOUGH_STANDARD", HoughModes::HOUGH_STANDARD)
        .value("HOUGH_PROBABILISTIC", HoughModes::HOUGH_PROBABILISTIC)
        .value("HOUGH_MULTI_SCALE", HoughModes::HOUGH_MULTI_SCALE)
        .value("HOUGH_GRADIENT", HoughModes::HOUGH_GRADIENT);

    emscripten::enum_<ImreadModes>("ImreadModes")
        .value("IMREAD_UNCHANGED", ImreadModes::IMREAD_UNCHANGED)
        .value("IMREAD_GRAYSCALE", ImreadModes::IMREAD_GRAYSCALE)
        .value("IMREAD_COLOR", ImreadModes::IMREAD_COLOR)
        .value("IMREAD_ANYDEPTH", ImreadModes::IMREAD_ANYDEPTH)
        .value("IMREAD_ANYCOLOR", ImreadModes::IMREAD_ANYCOLOR)
        .value("IMREAD_LOAD_GDAL", ImreadModes::IMREAD_LOAD_GDAL)
        .value("IMREAD_REDUCED_GRAYSCALE_2", ImreadModes::IMREAD_REDUCED_GRAYSCALE_2)
        .value("IMREAD_REDUCED_COLOR_2", ImreadModes::IMREAD_REDUCED_COLOR_2)
        .value("IMREAD_REDUCED_GRAYSCALE_4", ImreadModes::IMREAD_REDUCED_GRAYSCALE_4)
        .value("IMREAD_REDUCED_COLOR_4", ImreadModes::IMREAD_REDUCED_COLOR_4)
        .value("IMREAD_REDUCED_GRAYSCALE_8", ImreadModes::IMREAD_REDUCED_GRAYSCALE_8)
        .value("IMREAD_REDUCED_COLOR_8", ImreadModes::IMREAD_REDUCED_COLOR_8);

    emscripten::enum_<ImwriteFlags>("ImwriteFlags")
        .value("IMWRITE_JPEG_QUALITY", ImwriteFlags::IMWRITE_JPEG_QUALITY)
        .value("IMWRITE_JPEG_PROGRESSIVE", ImwriteFlags::IMWRITE_JPEG_PROGRESSIVE)
        .value("IMWRITE_JPEG_OPTIMIZE", ImwriteFlags::IMWRITE_JPEG_OPTIMIZE)
        .value("IMWRITE_JPEG_RST_INTERVAL", ImwriteFlags::IMWRITE_JPEG_RST_INTERVAL)
        .value("IMWRITE_JPEG_LUMA_QUALITY", ImwriteFlags::IMWRITE_JPEG_LUMA_QUALITY)
        .value("IMWRITE_JPEG_CHROMA_QUALITY", ImwriteFlags::IMWRITE_JPEG_CHROMA_QUALITY)
        .value("IMWRITE_PNG_COMPRESSION", ImwriteFlags::IMWRITE_PNG_COMPRESSION)
        .value("IMWRITE_PNG_STRATEGY", ImwriteFlags::IMWRITE_PNG_STRATEGY)
        .value("IMWRITE_PNG_BILEVEL", ImwriteFlags::IMWRITE_PNG_BILEVEL)
        .value("IMWRITE_PXM_BINARY", ImwriteFlags::IMWRITE_PXM_BINARY)
        .value("IMWRITE_WEBP_QUALITY", ImwriteFlags::IMWRITE_WEBP_QUALITY);

    emscripten::enum_<ImwritePNGFlags>("ImwritePNGFlags")
        .value("IMWRITE_PNG_STRATEGY_DEFAULT", ImwritePNGFlags::IMWRITE_PNG_STRATEGY_DEFAULT)
        .value("IMWRITE_PNG_STRATEGY_FILTERED", ImwritePNGFlags::IMWRITE_PNG_STRATEGY_FILTERED)
        .value("IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY", ImwritePNGFlags::IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY)
        .value("IMWRITE_PNG_STRATEGY_RLE", ImwritePNGFlags::IMWRITE_PNG_STRATEGY_RLE)
        .value("IMWRITE_PNG_STRATEGY_FIXED", ImwritePNGFlags::IMWRITE_PNG_STRATEGY_FIXED);

    emscripten::enum_<InterpolationFlags>("InterpolationFlags")
        .value("INTER_NEAREST", InterpolationFlags::INTER_NEAREST)
        .value("INTER_LINEAR", InterpolationFlags::INTER_LINEAR)
        .value("INTER_CUBIC", InterpolationFlags::INTER_CUBIC)
        .value("INTER_AREA", InterpolationFlags::INTER_AREA)
        .value("INTER_LANCZOS4", InterpolationFlags::INTER_LANCZOS4)
        .value("INTER_MAX", InterpolationFlags::INTER_MAX)
        .value("WARP_FILL_OUTLIERS", InterpolationFlags::WARP_FILL_OUTLIERS)
        .value("WARP_INVERSE_MAP", InterpolationFlags::WARP_INVERSE_MAP);

    emscripten::enum_<InterpolationMasks>("InterpolationMasks")
        .value("INTER_BITS", InterpolationMasks::INTER_BITS)
        .value("INTER_BITS2", InterpolationMasks::INTER_BITS2)
        .value("INTER_TAB_SIZE", InterpolationMasks::INTER_TAB_SIZE)
        .value("INTER_TAB_SIZE2", InterpolationMasks::INTER_TAB_SIZE2);

    emscripten::enum_<KmeansFlags>("KmeansFlags")
        .value("KMEANS_RANDOM_CENTERS", KmeansFlags::KMEANS_RANDOM_CENTERS)
        .value("KMEANS_PP_CENTERS", KmeansFlags::KMEANS_PP_CENTERS)
        .value("KMEANS_USE_INITIAL_LABELS", KmeansFlags::KMEANS_USE_INITIAL_LABELS);

    emscripten::enum_<LineSegmentDetectorModes>("LineSegmentDetectorModes")
        .value("LSD_REFINE_NONE", LineSegmentDetectorModes::LSD_REFINE_NONE)
        .value("LSD_REFINE_STD", LineSegmentDetectorModes::LSD_REFINE_STD)
        .value("LSD_REFINE_ADV", LineSegmentDetectorModes::LSD_REFINE_ADV);

    emscripten::enum_<LineTypes>("LineTypes")
        .value("FILLED", LineTypes::FILLED)
        .value("LINE_4", LineTypes::LINE_4)
        .value("LINE_8", LineTypes::LINE_8)
        .value("LINE_AA", LineTypes::LINE_AA);

    emscripten::enum_<MarkerTypes>("MarkerTypes")
        .value("MARKER_CROSS", MarkerTypes::MARKER_CROSS)
        .value("MARKER_TILTED_CROSS", MarkerTypes::MARKER_TILTED_CROSS)
        .value("MARKER_STAR", MarkerTypes::MARKER_STAR)
        .value("MARKER_DIAMOND", MarkerTypes::MARKER_DIAMOND)
        .value("MARKER_SQUARE", MarkerTypes::MARKER_SQUARE)
        .value("MARKER_TRIANGLE_UP", MarkerTypes::MARKER_TRIANGLE_UP)
        .value("MARKER_TRIANGLE_DOWN", MarkerTypes::MARKER_TRIANGLE_DOWN);

    emscripten::enum_<MorphShapes>("MorphShapes")
        .value("MORPH_RECT", MorphShapes::MORPH_RECT)
        .value("MORPH_CROSS", MorphShapes::MORPH_CROSS)
        .value("MORPH_ELLIPSE", MorphShapes::MORPH_ELLIPSE);

    emscripten::enum_<MorphTypes>("MorphTypes")
        .value("MORPH_ERODE", MorphTypes::MORPH_ERODE)
        .value("MORPH_DILATE", MorphTypes::MORPH_DILATE)
        .value("MORPH_OPEN", MorphTypes::MORPH_OPEN)
        .value("MORPH_CLOSE", MorphTypes::MORPH_CLOSE)
        .value("MORPH_GRADIENT", MorphTypes::MORPH_GRADIENT)
        .value("MORPH_TOPHAT", MorphTypes::MORPH_TOPHAT)
        .value("MORPH_BLACKHAT", MorphTypes::MORPH_BLACKHAT)
        .value("MORPH_HITMISS", MorphTypes::MORPH_HITMISS);

    emscripten::enum_<PCA::Flags>("PCA_Flags")
        .value("DATA_AS_ROW", PCA::Flags::DATA_AS_ROW)
        .value("DATA_AS_COL", PCA::Flags::DATA_AS_COL)
        .value("USE_AVG", PCA::Flags::USE_AVG);

    emscripten::enum_<RectanglesIntersectTypes>("RectanglesIntersectTypes")
        .value("INTERSECT_NONE", RectanglesIntersectTypes::INTERSECT_NONE)
        .value("INTERSECT_PARTIAL", RectanglesIntersectTypes::INTERSECT_PARTIAL)
        .value("INTERSECT_FULL", RectanglesIntersectTypes::INTERSECT_FULL);

    emscripten::enum_<ReduceTypes>("ReduceTypes")
        .value("REDUCE_SUM", ReduceTypes::REDUCE_SUM)
        .value("REDUCE_AVG", ReduceTypes::REDUCE_AVG)
        .value("REDUCE_MAX", ReduceTypes::REDUCE_MAX)
        .value("REDUCE_MIN", ReduceTypes::REDUCE_MIN);

    emscripten::enum_<RetrievalModes>("RetrievalModes")
        .value("RETR_EXTERNAL", RetrievalModes::RETR_EXTERNAL)
        .value("RETR_LIST", RetrievalModes::RETR_LIST)
        .value("RETR_CCOMP", RetrievalModes::RETR_CCOMP)
        .value("RETR_TREE", RetrievalModes::RETR_TREE)
        .value("RETR_FLOODFILL", RetrievalModes::RETR_FLOODFILL);

    emscripten::enum_<SVD::Flags>("SVD_Flags")
        .value("MODIFY_A", SVD::Flags::MODIFY_A)
        .value("NO_UV", SVD::Flags::NO_UV)
        .value("FULL_UV", SVD::Flags::FULL_UV);

    emscripten::enum_<SortFlags>("SortFlags")
        .value("SORT_EVERY_ROW", SortFlags::SORT_EVERY_ROW)
        .value("SORT_EVERY_COLUMN", SortFlags::SORT_EVERY_COLUMN)
        .value("SORT_ASCENDING", SortFlags::SORT_ASCENDING)
        .value("SORT_DESCENDING", SortFlags::SORT_DESCENDING);

    emscripten::enum_<TemplateMatchModes>("TemplateMatchModes")
        .value("TM_SQDIFF", TemplateMatchModes::TM_SQDIFF)
        .value("TM_SQDIFF_NORMED", TemplateMatchModes::TM_SQDIFF_NORMED)
        .value("TM_CCORR", TemplateMatchModes::TM_CCORR)
        .value("TM_CCORR_NORMED", TemplateMatchModes::TM_CCORR_NORMED)
        .value("TM_CCOEFF", TemplateMatchModes::TM_CCOEFF)
        .value("TM_CCOEFF_NORMED", TemplateMatchModes::TM_CCOEFF_NORMED);

    emscripten::enum_<TermCriteria::Type>("TermCriteria_Type")
        .value("COUNT", TermCriteria::Type::COUNT)
        .value("MAX_ITER", TermCriteria::Type::MAX_ITER)
        .value("EPS", TermCriteria::Type::EPS);

    emscripten::enum_<ThresholdTypes>("ThresholdTypes")
        .value("THRESH_BINARY", ThresholdTypes::THRESH_BINARY)
        .value("THRESH_BINARY_INV", ThresholdTypes::THRESH_BINARY_INV)
        .value("THRESH_TRUNC", ThresholdTypes::THRESH_TRUNC)
        .value("THRESH_TOZERO", ThresholdTypes::THRESH_TOZERO)
        .value("THRESH_TOZERO_INV", ThresholdTypes::THRESH_TOZERO_INV)
        .value("THRESH_MASK", ThresholdTypes::THRESH_MASK)
        .value("THRESH_OTSU", ThresholdTypes::THRESH_OTSU)
        .value("THRESH_TRIANGLE", ThresholdTypes::THRESH_TRIANGLE);

    emscripten::enum_<UMatUsageFlags>("UMatUsageFlags")
        .value("USAGE_DEFAULT", UMatUsageFlags::USAGE_DEFAULT)
        .value("USAGE_ALLOCATE_HOST_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_HOST_MEMORY)
        .value("USAGE_ALLOCATE_DEVICE_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)
        .value("USAGE_ALLOCATE_SHARED_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_SHARED_MEMORY)
        .value("__UMAT_USAGE_FLAGS_32BIT", UMatUsageFlags::__UMAT_USAGE_FLAGS_32BIT);

    emscripten::enum_<UndistortTypes>("UndistortTypes")
        .value("PROJ_SPHERICAL_ORTHO", UndistortTypes::PROJ_SPHERICAL_ORTHO)
        .value("PROJ_SPHERICAL_EQRECT", UndistortTypes::PROJ_SPHERICAL_EQRECT);

    emscripten::enum_<ml::ANN_MLP::ActivationFunctions>("ml_ANN_MLP_ActivationFunctions")
        .value("IDENTITY", ml::ANN_MLP::ActivationFunctions::IDENTITY)
        .value("SIGMOID_SYM", ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM)
        .value("GAUSSIAN", ml::ANN_MLP::ActivationFunctions::GAUSSIAN);

    emscripten::enum_<ml::ANN_MLP::TrainFlags>("ml_ANN_MLP_TrainFlags")
        .value("UPDATE_WEIGHTS", ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS)
        .value("NO_INPUT_SCALE", ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE)
        .value("NO_OUTPUT_SCALE", ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);

    emscripten::enum_<ml::ANN_MLP::TrainingMethods>("ml_ANN_MLP_TrainingMethods")
        .value("BACKPROP", ml::ANN_MLP::TrainingMethods::BACKPROP)
        .value("RPROP", ml::ANN_MLP::TrainingMethods::RPROP);

    emscripten::enum_<ml::Boost::Types>("ml_Boost_Types")
        .value("DISCRETE", ml::Boost::Types::DISCRETE)
        .value("REAL", ml::Boost::Types::REAL)
        .value("LOGIT", ml::Boost::Types::LOGIT)
        .value("GENTLE", ml::Boost::Types::GENTLE);

    emscripten::enum_<ml::DTrees::Flags>("ml_DTrees_Flags")
        .value("PREDICT_AUTO", ml::DTrees::Flags::PREDICT_AUTO)
        .value("PREDICT_SUM", ml::DTrees::Flags::PREDICT_SUM)
        .value("PREDICT_MAX_VOTE", ml::DTrees::Flags::PREDICT_MAX_VOTE)
        .value("PREDICT_MASK", ml::DTrees::Flags::PREDICT_MASK);

    emscripten::enum_<ml::EM::Types>("ml_EM_Types")
        .value("COV_MAT_SPHERICAL", ml::EM::Types::COV_MAT_SPHERICAL)
        .value("COV_MAT_DIAGONAL", ml::EM::Types::COV_MAT_DIAGONAL)
        .value("COV_MAT_GENERIC", ml::EM::Types::COV_MAT_GENERIC)
        .value("COV_MAT_DEFAULT", ml::EM::Types::COV_MAT_DEFAULT);

    emscripten::enum_<ml::ErrorTypes>("ml_ErrorTypes")
        .value("TEST_ERROR", ml::ErrorTypes::TEST_ERROR)
        .value("TRAIN_ERROR", ml::ErrorTypes::TRAIN_ERROR);

    emscripten::enum_<ml::KNearest::Types>("ml_KNearest_Types")
        .value("BRUTE_FORCE", ml::KNearest::Types::BRUTE_FORCE)
        .value("KDTREE", ml::KNearest::Types::KDTREE);

    emscripten::enum_<ml::LogisticRegression::Methods>("ml_LogisticRegression_Methods")
        .value("BATCH", ml::LogisticRegression::Methods::BATCH)
        .value("MINI_BATCH", ml::LogisticRegression::Methods::MINI_BATCH);

    emscripten::enum_<ml::LogisticRegression::RegKinds>("ml_LogisticRegression_RegKinds")
        .value("REG_DISABLE", ml::LogisticRegression::RegKinds::REG_DISABLE)
        .value("REG_L1", ml::LogisticRegression::RegKinds::REG_L1)
        .value("REG_L2", ml::LogisticRegression::RegKinds::REG_L2);

    emscripten::enum_<ml::SVM::KernelTypes>("ml_SVM_KernelTypes")
        .value("CUSTOM", ml::SVM::KernelTypes::CUSTOM)
        .value("LINEAR", ml::SVM::KernelTypes::LINEAR)
        .value("POLY", ml::SVM::KernelTypes::POLY)
        .value("RBF", ml::SVM::KernelTypes::RBF)
        .value("SIGMOID", ml::SVM::KernelTypes::SIGMOID)
        .value("CHI2", ml::SVM::KernelTypes::CHI2)
        .value("INTER", ml::SVM::KernelTypes::INTER);

    emscripten::enum_<ml::SVM::ParamTypes>("ml_SVM_ParamTypes")
        .value("C", ml::SVM::ParamTypes::C)
        .value("GAMMA", ml::SVM::ParamTypes::GAMMA)
        .value("P", ml::SVM::ParamTypes::P)
        .value("NU", ml::SVM::ParamTypes::NU)
        .value("COEF", ml::SVM::ParamTypes::COEF)
        .value("DEGREE", ml::SVM::ParamTypes::DEGREE);

    emscripten::enum_<ml::SVM::Types>("ml_SVM_Types")
        .value("C_SVC", ml::SVM::Types::C_SVC)
        .value("NU_SVC", ml::SVM::Types::NU_SVC)
        .value("ONE_CLASS", ml::SVM::Types::ONE_CLASS)
        .value("EPS_SVR", ml::SVM::Types::EPS_SVR)
        .value("NU_SVR", ml::SVM::Types::NU_SVR);

    emscripten::enum_<ml::SampleTypes>("ml_SampleTypes")
        .value("ROW_SAMPLE", ml::SampleTypes::ROW_SAMPLE)
        .value("COL_SAMPLE", ml::SampleTypes::COL_SAMPLE);

    emscripten::enum_<ml::StatModel::Flags>("ml_StatModel_Flags")
        .value("UPDATE_MODEL", ml::StatModel::Flags::UPDATE_MODEL)
        .value("RAW_OUTPUT", ml::StatModel::Flags::RAW_OUTPUT)
        .value("COMPRESSED_INPUT", ml::StatModel::Flags::COMPRESSED_INPUT)
        .value("PREPROCESSED_INPUT", ml::StatModel::Flags::PREPROCESSED_INPUT);

    emscripten::enum_<ml::VariableTypes>("ml_VariableTypes")
        .value("VAR_NUMERICAL", ml::VariableTypes::VAR_NUMERICAL)
        .value("VAR_ORDERED", ml::VariableTypes::VAR_ORDERED)
        .value("VAR_CATEGORICAL", ml::VariableTypes::VAR_CATEGORICAL);

    emscripten::enum_<ocl::OclVectorStrategy>("ocl_OclVectorStrategy")
        .value("OCL_VECTOR_OWN", ocl::OclVectorStrategy::OCL_VECTOR_OWN)
        .value("OCL_VECTOR_MAX", ocl::OclVectorStrategy::OCL_VECTOR_MAX)
        .value("OCL_VECTOR_DEFAULT", ocl::OclVectorStrategy::OCL_VECTOR_DEFAULT);

}
