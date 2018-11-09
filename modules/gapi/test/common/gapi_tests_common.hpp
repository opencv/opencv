// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include <iostream>

#include "opencv2/ts.hpp"
#include "opencv2/gapi.hpp"

namespace
{
    inline std::ostream& operator<<(std::ostream& o, const cv::GCompileArg& arg)
    {
        return o << (arg.tag.empty() ? "empty" : arg.tag);
    }
}

namespace opencv_test
{

class TestFunctional
{
public:
    cv::Mat in_mat1;
    cv::Mat in_mat2;
    cv::Mat out_mat_gapi;
    cv::Mat out_mat_ocv;

    cv::Scalar sc;

    cv::Scalar initScalarRandU(unsigned upper)
    {
        auto& rng = cv::theRNG();
        double s1 = rng(upper);
        double s2 = rng(upper);
        double s3 = rng(upper);
        double s4 = rng(upper);
        return cv::Scalar(s1, s2, s3, s4);
    }

    void initMatsRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);
        in_mat2 = cv::Mat(sz_in, type);

        sc = initScalarRandU(100);
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

        if (createOutputMatrices && dtype != -1)
        {
            out_mat_gapi = cv::Mat (sz_in, dtype);
            out_mat_ocv = cv::Mat (sz_in, dtype);
        }
    }

    void initMatrixRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);

        sc = initScalarRandU(100);

        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

        if (createOutputMatrices && dtype != -1)
        {
            out_mat_gapi = cv::Mat (sz_in, dtype);
            out_mat_ocv = cv::Mat (sz_in, dtype);
        }
    }

    void initMatsRandN(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1  = cv::Mat(sz_in, type);
        cv::randn(in_mat1, cv::Scalar::all(127), cv::Scalar::all(40.f));

        if (createOutputMatrices  && dtype != -1)
        {
            out_mat_gapi = cv::Mat(sz_in, dtype);
            out_mat_ocv = cv::Mat(sz_in, dtype);
        }
    }

    static cv::Mat nonZeroPixels(const cv::Mat& mat)
    {
        int channels = mat.channels();
        std::vector<cv::Mat> split(channels);
        cv::split(mat, split);
        cv::Mat result;
        for (int c=0; c < channels; c++)
        {
            if (c == 0)
                result = split[c] != 0;
            else
                result = result | (split[c] != 0);
        }
        return result;
    }

    static int countNonZeroPixels(const cv::Mat& mat)
    {
        return cv::countNonZero( nonZeroPixels(mat) );
    }

};

template<class T>
class TestParams: public TestFunctional, public TestWithParam<T>{};

template<class T>
class TestPerfParams: public TestFunctional, public perf::TestBaseWithParam<T>{};

using compare_f = std::function<bool(const cv::Mat &a, const cv::Mat &b)>;

template<typename T>
struct Wrappable
{
    compare_f to_compare_f()
    {
        T t = *static_cast<T*const>(this);
        return [t](const cv::Mat &a, const cv::Mat &b)
        {
            return t(a, b);
        };
    }
};

class AbsExact : public Wrappable<AbsExact>
{
public:
    AbsExact() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (cv::countNonZero(in1 != in2) != 0)
        {
            std::cout << "AbsExact error: G-API output and reference output matrixes are not bitexact equal."  << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
private:
};

class AbsTolerance : public Wrappable<AbsTolerance>
{
public:
    AbsTolerance(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
        if(cv::countNonZero(absDiff > _tol))
        {
            std::cout << "AbsTolerance error: Number of different pixels in " << std::endl;
            std::cout << "G-API output and reference output matrixes exceeds " << _tol << " pixels threshold." << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
private:
    double _tol;
};

class AbsTolerance_Float_Int : public Wrappable<AbsTolerance_Float_Int>
{
public:
    AbsTolerance_Float_Int(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F)
        {
            if (cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2)))
            {
                std::cout << "AbsTolerance_Float_Int error (Float): One or more of pixels in" << std::endl;
                std::cout << "G-API output exceeds relative threshold value defined by reference_pixel_value * tolerance" << std::endl;
                std::cout << "for tolerance " << _tol << std::endl;
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            if (cv::countNonZero(in1 != in2) <= (_tol8u)* in2.total())
            {
                return true;
            }
            else
            {
                std::cout << "AbsTolerance_Float_Int error (Integer): Number of different pixels in" << std::endl;
                std::cout << "G-API output and reference output matrixes exceeds relative threshold value" << std::endl;
                std::cout << "defined by reference_total_pixels_number * tolerance" << std::endl;
                std::cout << "for tolerance " << _tol8u << std::endl;
                return false;
            }
        }
    }
private:
    double _tol;
    double _tol8u;
};

class AbsToleranceSepFilter : public Wrappable<AbsToleranceSepFilter>
{
public:
    AbsToleranceSepFilter(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)* cv::abs(in2)) <= 0.01 * in2.total()))
        {
            return true;
        }
        else
        {
            std::cout << "AbsToleranceSepFilter error: Number of different pixels in" << std::endl;
            std::cout << "G-API output and reference output matrixes which exceeds relative threshold value" << std::endl;
            std::cout << "defined by reference_pixel_value * tolerance" << std::endl;
            std::cout << "for tolerance " << _tol << " is more then 1% of total number of pixels in the reference matrix." << std::endl;
            return false;
        }
    }
private:
    double _tol;
};

class AbsToleranceGaussianBlur_Float_Int : public Wrappable<AbsToleranceGaussianBlur_Float_Int>
{
public:
    AbsToleranceGaussianBlur_Float_Int(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F || CV_MAT_DEPTH(in1.type()) == CV_64F)
        {
            if (cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2)))
            {
                std::cout << "AbsToleranceGaussianBlur_Float_Int error (Float): Number of different pixels in" << std::endl;
                std::cout << "G-API output and reference output matrixes which exceeds relative threshold value" << std::endl;
                std::cout << "defined by reference_pixel_value * tolerance" << std::endl;
                std::cout << "for tolerance " << _tol << " is more then 0." << std::endl;
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            if (CV_MAT_DEPTH(in1.type()) == CV_8U)
            {
                bool a = (cv::countNonZero(cv::abs(in1 - in2) > 1) <= _tol8u * in2.total());
                if (((a == 1 ? 0 : 1) && ((cv::countNonZero(cv::abs(in1 - in2) > 2) <= 0) == 1 ? 0 : 1)) == 1)
                {
                    std::cout << "AbsToleranceGaussianBlur_Float_Int error (8U): Number of pixels in" << std::endl;
                    std::cout << "G-API output and reference output matrixes with absolute difference which is more than 1 but less than 3" << std::endl;
                    std::cout << "exceeds relative threshold value defined by reference_total_pixels_number * tolerance" << std::endl;
                    std::cout << "for tolerance " << _tol8u << std::endl;
                    return false;
                }
                else
                {
                    return true;
                }
            }
            else
            {
                if (cv::countNonZero(in1 != in2) != 0)
                {
                    std::cout << "AbsToleranceGaussianBlur_Float_Int error: G-API output and reference output matrixes are not bitexact equal." << std::endl;
                    return false;
                }
                else
                {
                    return true;
                }
            }
        }
    }
private:
    double _tol;
    double _tol8u;
};

class ToleranceRGBBGR : public Wrappable<ToleranceRGBBGR>
{
public:
    ToleranceRGBBGR(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        bool a = (cv::countNonZero((in1 - in2) > 0) <= _tol * in2.total());
        if (((a == 1 ? 0 : 1) && ((cv::countNonZero((in1 - in2) > 1) <= 0) == 1 ? 0 : 1)) == 1)
        {
            std::cout << "ToleranceRGBBGR error: Number of pixels in" << std::endl;
            std::cout << "G-API output and reference output matrixes with difference which is more than 0 but no more than 1" << std::endl;
            std::cout << "exceeds relative threshold value defined by reference_total_pixels_number * tolerance" << std::endl;
            std::cout << "for tolerance " << _tol << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
private:
    double _tol;
};

class ToleranceTriple: public Wrappable<ToleranceTriple>
{
public:
    ToleranceTriple(double tol1, double tol2, double tol3) : _tol1(tol1), _tol2(tol2), _tol3(tol3) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        bool a = (cv::countNonZero((in1 - in2) > 0) <= _tol1 * in2.total());
        if ((((a == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 1) <= _tol2 * in2.total()) == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 2) <= _tol3 * in2.total()) == 1 ? 0 : 1))) == 1)
        {
            std::cout << "ToleranceTriple error: Number of pixels in" << std::endl;
            std::cout << "G-API output and reference output matrixes with difference which is more than 0 but no more than 1" << std::endl;
            std::cout << "exceeds relative threshold value defined by reference_total_pixels_number * tolerance1" << std::endl;
            std::cout << "for tolerance1 " << _tol1 << std::endl;
            std::cout << "AND with difference which is more than 1 but no more than 2" << std::endl;
            std::cout << "exceeds relative threshold value defined by reference_total_pixels_number * tolerance2" << std::endl;
            std::cout << "for tolerance2 " << _tol2 << std::endl;
            std::cout << "AND with difference which is more than 2" << std::endl;
            std::cout << "exceeds relative threshold value defined by reference_total_pixels_number * tolerance3" << std::endl;
            std::cout << "for tolerance3 " << _tol3 << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
private:
    double _tol1, _tol2, _tol3;
};

class AbsToleranceSobel : public Wrappable<AbsToleranceSobel>
{
public:
    AbsToleranceSobel(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat diff, a1, a2, b, base;
        cv::absdiff(in1, in2, diff);
        a1 = cv::abs(in1);
        a2 = cv::abs(in2);
        cv::max(a1, a2, b);
        cv::max(1, b, base);  // base = max{1, |in1|, |in2|}

        if(cv::countNonZero(diff > _tol*base) != 0)
        {
            std::cout << "AbsToleranceSobel error: Number of pixels in" << std::endl;
            std::cout << "G-API output and reference output matrixes with absolute difference which is more than relative threshold defined by tolerance * max{1, |in1|, |in2|}" << std::endl;
            std::cout << "relative threshold defined by tolerance * max{1, |in1|, |in2|} exceeds 0"<< std::endl;
            std::cout << "for tolerance " << _tol << std::endl;
            return false;
        }
        else
        {
            return true;
        }

    }
private:
    double _tol;
};
} // namespace opencv_test

namespace
{
    inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_f&)
    {
        return os << "compare_f";
    }
}
