/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// Author: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015 The Regents of the University of California (Regents)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//M*/

#include <emscripten/bind.h>

@INCLUDES@

using namespace emscripten;
using namespace cv;
using namespace dnn;

namespace binding_utils
{
    template<typename T>
    emscripten::val matData(const cv::Mat& mat)
    {
        return emscripten::val(emscripten::memory_view<T>((mat.total()*mat.elemSize())/sizeof(T),
                               (T*)mat.data));
    }

    template<typename T>
    emscripten::val matPtr(const cv::Mat& mat, int i)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(0), mat.ptr<T>(i)));
    }

    template<typename T>
    emscripten::val matPtr(const cv::Mat& mat, int i, int j)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(1), mat.ptr<T>(i,j)));
    }

    cv::Mat* createMat(int rows, int cols, int type, intptr_t data, size_t step)
    {
        return new cv::Mat(rows, cols, type, reinterpret_cast<void*>(data), step);
    }

    static emscripten::val getMatSize(const cv::Mat& mat)
    {
        emscripten::val size = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            size.call<void>("push", mat.size[i]);
        }
        return size;
    }

    static emscripten::val getMatStep(const cv::Mat& mat)
    {
        emscripten::val step = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            step.call<void>("push", mat.step[i]);
        }
        return step;
    }

    static Mat matEye(int rows, int cols, int type)
    {
        return Mat(cv::Mat::eye(rows, cols, type));
    }

    static Mat matEye(Size size, int type)
    {
        return Mat(cv::Mat::eye(size, type));
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta)
    {
        obj.convertTo(m, rtype, alpha, beta);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype)
    {
        obj.convertTo(m, rtype);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha)
    {
        obj.convertTo(m, rtype, alpha);
    }

    Size matSize(const cv::Mat& mat)
    {
        return mat.size();
    }

    cv::Mat matZeros(int arg0, int arg1, int arg2)
    {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }

    cv::Mat matZeros(cv::Size arg0, int arg1)
    {
        return cv::Mat::zeros(arg0,arg1);
    }

    cv::Mat matOnes(int arg0, int arg1, int arg2)
    {
        return cv::Mat::ones(arg0, arg1, arg2);
    }

    cv::Mat matOnes(cv::Size arg0, int arg1)
    {
        return cv::Mat::ones(arg0, arg1);
    }

    double matDot(const cv::Mat& obj, const Mat& mat)
    {
        return  obj.dot(mat);
    }

    Mat matMul(const cv::Mat& obj, const Mat& mat, double scale)
    {
        return  Mat(obj.mul(mat, scale));
    }

    Mat matT(const cv::Mat& obj)
    {
        return  Mat(obj.t());
    }

    Mat matInv(const cv::Mat& obj, int type)
    {
        return  Mat(obj.inv(type));
    }

    void matCopyTo(const cv::Mat& obj, cv::Mat& mat)
    {
        return obj.copyTo(mat);
    }

    void matCopyTo(const cv::Mat& obj, cv::Mat& mat, const cv::Mat& mask)
    {
        return obj.copyTo(mat, mask);
    }

    Mat matDiag(const cv::Mat& obj, int d)
    {
        return obj.diag(d);
    }

    Mat matDiag(const cv::Mat& obj)
    {
        return obj.diag();
    }

    void matSetTo(cv::Mat& obj, const cv::Scalar& s)
    {
        obj.setTo(s);
    }

    void matSetTo(cv::Mat& obj, const cv::Scalar& s, const cv::Mat& mask)
    {
        obj.setTo(s, mask);
    }

    emscripten::val rotatedRectPoints(const cv::RotatedRect& obj)
    {
        cv::Point2f points[4];
        obj.points(points);
        emscripten::val pointsArray = emscripten::val::array();
        for (int i = 0; i < 4; i++) {
            pointsArray.call<void>("push", points[i]);
        }
        return pointsArray;
    }

    Rect rotatedRectBoundingRect(const cv::RotatedRect& obj)
    {
        return obj.boundingRect();
    }

    Rect2f rotatedRectBoundingRect2f(const cv::RotatedRect& obj)
    {
        return obj.boundingRect2f();
    }

    int cvMatDepth(int flags)
    {
        return CV_MAT_DEPTH(flags);
    }

    class MinMaxLoc
    {
    public:
        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;
    };

    MinMaxLoc minMaxLoc(const cv::Mat& src, const cv::Mat& mask)
    {
        MinMaxLoc result;
        cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc, mask);
        return result;
    }

    MinMaxLoc minMaxLoc_1(const cv::Mat& src)
    {
        MinMaxLoc result;
        cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc);
        return result;
    }

    class Circle
    {
    public:
        Point2f center;
        float radius;
    };

    Circle minEnclosingCircle(const cv::Mat& points)
    {
        Circle circle;
        cv::minEnclosingCircle(points, circle.center, circle.radius);
        return circle;
    }

    emscripten::val CamShiftWrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        RotatedRect rotatedRect = cv::CamShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", rotatedRect);
        result.call<void>("push", arg2);
        return result;
    }

    emscripten::val meanShiftWrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        int n = cv::meanShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", n);
        result.call<void>("push", arg2);
        return result;
    }

    std::string getExceptionMsg(const cv::Exception& e) {
        return e.msg;
    }

    void setExceptionMsg(cv::Exception& e, std::string msg) {
        e.msg = msg;
        return;
    }

    cv::Exception exceptionFromPtr(intptr_t ptr) {
        return *reinterpret_cast<cv::Exception*>(ptr);
    }

    std::string getBuildInformation() {
        return cv::getBuildInformation();
    }
}

EMSCRIPTEN_BINDINGS(binding_utils)
{
    register_vector<int>("IntVector");
    register_vector<float>("FloatVector");
    register_vector<double>("DoubleVector");
    register_vector<cv::Point>("PointVector");
    register_vector<cv::Mat>("MatVector");
    register_vector<cv::Rect>("RectVector");
    register_vector<cv::KeyPoint>("KeyPointVector");

    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        .constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor<int, int, int, const Scalar&>()
        .constructor(&binding_utils::createMat, allow_raw_pointers())

        .class_function("eye", select_overload<Mat(Size, int)>(&binding_utils::matEye))
        .class_function("eye", select_overload<Mat(int, int, int)>(&binding_utils::matEye))
        .class_function("ones", select_overload<Mat(Size, int)>(&binding_utils::matOnes))
        .class_function("ones", select_overload<Mat(int, int, int)>(&binding_utils::matOnes))
        .class_function("zeros", select_overload<Mat(Size, int)>(&binding_utils::matZeros))
        .class_function("zeros", select_overload<Mat(int, int, int)>(&binding_utils::matZeros))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)
        .property("matSize", &binding_utils::getMatSize)
        .property("step", &binding_utils::getMatStep)
        .property("data", &binding_utils::matData<unsigned char>)
        .property("data8S", &binding_utils::matData<char>)
        .property("data16U", &binding_utils::matData<unsigned short>)
        .property("data16S", &binding_utils::matData<short>)
        .property("data32S", &binding_utils::matData<int>)
        .property("data32F", &binding_utils::matData<float>)
        .property("data64F", &binding_utils::matData<double>)

        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double, double)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double)>(&binding_utils::convertTo))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))
        .function("copyTo", select_overload<void(const Mat&, Mat&)>(&binding_utils::matCopyTo))
        .function("copyTo", select_overload<void(const Mat&, Mat&, const Mat&)>(&binding_utils::matCopyTo))
        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .function("depth", select_overload<int()const>(&cv::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&cv::Mat::col))
        .function("dot", select_overload<double(const Mat&, const Mat&)>(&binding_utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&binding_utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&binding_utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&binding_utils::matT))
        .function("roi", select_overload<Mat(const Rect&)const>(&cv::Mat::operator()))
        .function("diag", select_overload<Mat(const Mat&, int)>(&binding_utils::matDiag))
        .function("diag", select_overload<Mat(const Mat&)>(&binding_utils::matDiag))
        .function("isContinuous", select_overload<bool()const>(&cv::Mat::isContinuous))
        .function("setTo", select_overload<void(Mat&, const Scalar&)>(&binding_utils::matSetTo))
        .function("setTo", select_overload<void(Mat&, const Scalar&, const Mat&)>(&binding_utils::matSetTo))
        .function("size", select_overload<Size(const Mat&)>(&binding_utils::matSize))

        .function("ptr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("charPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<char>))
        .function("charPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<char>))
        .function("shortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<short>))
        .function("shortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<short>))
        .function("ushortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned short>))
        .function("ushortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned short>))
        .function("intPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<int>))
        .function("intPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<int>))
        .function("floatPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<float>))
        .function("floatPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<float>))
        .function("doublePtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<double>))
        .function("doublePtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<double>))

        .function("charAt", select_overload<char&(int)>(&cv::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int)>(&cv::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int, int)>(&cv::Mat::at<char>))
        .function("ucharAt", select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("shortAt", select_overload<short&(int)>(&cv::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int)>(&cv::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int, int)>(&cv::Mat::at<short>))
        .function("ushortAt", select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("intAt", select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("floatAt", select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function("doubleAt", select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int, int)>(&cv::Mat::at<double>));

    emscripten::value_object<cv::Range>("Range")
        .field("start", &cv::Range::start)
        .field("end", &cv::Range::end);

    emscripten::value_object<cv::TermCriteria>("TermCriteria")
        .field("type", &cv::TermCriteria::type)
        .field("maxCount", &cv::TermCriteria::maxCount)
        .field("epsilon", &cv::TermCriteria::epsilon);

#define EMSCRIPTEN_CV_SIZE(type) \
    emscripten::value_object<type>("#type") \
        .field("width", &type::width) \
        .field("height", &type::height);

    EMSCRIPTEN_CV_SIZE(Size)
    EMSCRIPTEN_CV_SIZE(Size2f)

#define EMSCRIPTEN_CV_POINT(type) \
    emscripten::value_object<type>("#type") \
        .field("x", &type::x) \
        .field("y", &type::y); \

    EMSCRIPTEN_CV_POINT(Point)
    EMSCRIPTEN_CV_POINT(Point2f)

#define EMSCRIPTEN_CV_RECT(type, name) \
    emscripten::value_object<cv::Rect_<type>> (name) \
        .field("x", &cv::Rect_<type>::x) \
        .field("y", &cv::Rect_<type>::y) \
        .field("width", &cv::Rect_<type>::width) \
        .field("height", &cv::Rect_<type>::height);

    EMSCRIPTEN_CV_RECT(int, "Rect")
    EMSCRIPTEN_CV_RECT(float, "Rect2f")

    emscripten::value_object<cv::RotatedRect>("RotatedRect")
        .field("center", &cv::RotatedRect::center)
        .field("size", &cv::RotatedRect::size)
        .field("angle", &cv::RotatedRect::angle);

    function("rotatedRectPoints", select_overload<emscripten::val(const cv::RotatedRect&)>(&binding_utils::rotatedRectPoints));
    function("rotatedRectBoundingRect", select_overload<Rect(const cv::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect));
    function("rotatedRectBoundingRect2f", select_overload<Rect2f(const cv::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect2f));

    emscripten::value_object<cv::KeyPoint>("KeyPoint")
        .field("angle", &cv::KeyPoint::angle)
        .field("class_id", &cv::KeyPoint::class_id)
        .field("octave", &cv::KeyPoint::octave)
        .field("pt", &cv::KeyPoint::pt)
        .field("response", &cv::KeyPoint::response)
        .field("size", &cv::KeyPoint::size);

    emscripten::value_array<cv::Scalar_<double>> ("Scalar")
        .element(index<0>())
        .element(index<1>())
        .element(index<2>())
        .element(index<3>());

    emscripten::value_object<binding_utils::MinMaxLoc>("MinMaxLoc")
        .field("minVal", &binding_utils::MinMaxLoc::minVal)
        .field("maxVal", &binding_utils::MinMaxLoc::maxVal)
        .field("minLoc", &binding_utils::MinMaxLoc::minLoc)
        .field("maxLoc", &binding_utils::MinMaxLoc::maxLoc);

    emscripten::value_object<binding_utils::Circle>("Circle")
        .field("center", &binding_utils::Circle::center)
        .field("radius", &binding_utils::Circle::radius);

    emscripten::value_object<cv::Moments >("Moments")
        .field("m00", &cv::Moments::m00)
        .field("m10", &cv::Moments::m10)
        .field("m01", &cv::Moments::m01)
        .field("m20", &cv::Moments::m20)
        .field("m11", &cv::Moments::m11)
        .field("m02", &cv::Moments::m02)
        .field("m30", &cv::Moments::m30)
        .field("m21", &cv::Moments::m21)
        .field("m12", &cv::Moments::m12)
        .field("m03", &cv::Moments::m03)
        .field("mu20", &cv::Moments::mu20)
        .field("mu11", &cv::Moments::mu11)
        .field("mu02", &cv::Moments::mu02)
        .field("mu30", &cv::Moments::mu30)
        .field("mu21", &cv::Moments::mu21)
        .field("mu12", &cv::Moments::mu12)
        .field("mu03", &cv::Moments::mu03)
        .field("nu20", &cv::Moments::nu20)
        .field("nu11", &cv::Moments::nu11)
        .field("nu02", &cv::Moments::nu02)
        .field("nu30", &cv::Moments::nu30)
        .field("nu21", &cv::Moments::nu21)
        .field("nu12", &cv::Moments::nu12)
        .field("nu03", &cv::Moments::nu03);

    emscripten::value_object<cv::Exception>("Exception")
        .field("code", &cv::Exception::code)
        .field("msg", &binding_utils::getExceptionMsg, &binding_utils::setExceptionMsg);

    function("exceptionFromPtr", &binding_utils::exceptionFromPtr, allow_raw_pointers());

    function("minEnclosingCircle", select_overload<binding_utils::Circle(const cv::Mat&)>(&binding_utils::minEnclosingCircle));

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const cv::Mat&, const cv::Mat&)>(&binding_utils::minMaxLoc));

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const cv::Mat&)>(&binding_utils::minMaxLoc_1));

    function("morphologyDefaultBorderValue", &cv::morphologyDefaultBorderValue);

    function("CV_MAT_DEPTH", &binding_utils::cvMatDepth);

    function("CamShift", select_overload<emscripten::val(const cv::Mat&, Rect&, TermCriteria)>(&binding_utils::CamShiftWrapper));

    function("meanShift", select_overload<emscripten::val(const cv::Mat&, Rect&, TermCriteria)>(&binding_utils::meanShiftWrapper));

    function("getBuildInformation", &binding_utils::getBuildInformation);

    constant("CV_8UC1", CV_8UC1);
    constant("CV_8UC2", CV_8UC2);
    constant("CV_8UC3", CV_8UC3);
    constant("CV_8UC4", CV_8UC4);

    constant("CV_8SC1", CV_8SC1);
    constant("CV_8SC2", CV_8SC2);
    constant("CV_8SC3", CV_8SC3);
    constant("CV_8SC4", CV_8SC4);

    constant("CV_16UC1", CV_16UC1);
    constant("CV_16UC2", CV_16UC2);
    constant("CV_16UC3", CV_16UC3);
    constant("CV_16UC4", CV_16UC4);

    constant("CV_16SC1", CV_16SC1);
    constant("CV_16SC2", CV_16SC2);
    constant("CV_16SC3", CV_16SC3);
    constant("CV_16SC4", CV_16SC4);

    constant("CV_32SC1", CV_32SC1);
    constant("CV_32SC2", CV_32SC2);
    constant("CV_32SC3", CV_32SC3);
    constant("CV_32SC4", CV_32SC4);

    constant("CV_32FC1", CV_32FC1);
    constant("CV_32FC2", CV_32FC2);
    constant("CV_32FC3", CV_32FC3);
    constant("CV_32FC4", CV_32FC4);

    constant("CV_64FC1", CV_64FC1);
    constant("CV_64FC2", CV_64FC2);
    constant("CV_64FC3", CV_64FC3);
    constant("CV_64FC4", CV_64FC4);

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
