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

using compare_scalar_f = std::function<bool(const cv::Scalar &a, const cv::Scalar &b)>;


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

template<typename T>
struct WrappableScalar
{
    compare_scalar_f to_compare_f()
    {
        T t = *static_cast<T*const>(this);
        return [t](const cv::Scalar &a, const cv::Scalar &b)
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
        if (cv::norm(in1, in2, NORM_INF) != 0)
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
        if (cv::norm(in1, in2, NORM_INF) > _tol)
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

class Tolerance_FloatRel_IntAbs : public Wrappable<Tolerance_FloatRel_IntAbs>
{
public:
    Tolerance_FloatRel_IntAbs(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        int depth = CV_MAT_DEPTH(in1.type());
        {
            double err = depth >= CV_32F ? cv::norm(in1, in2, NORM_L1 | NORM_RELATIVE)
                                                     : cv::norm(in1, in2, NORM_INF);
            double tolerance = depth >= CV_32F ? _tol : _tol8u;
            if (err > tolerance)
            {
                std::cout << "Tolerance_FloatRel_IntAbs error: err=" << err
                          << "  tolerance=" << tolerance
                          << "  depth=" << cv::typeToString(depth) << std::endl;
                return false;
            }
            else
            {
                return true;
            }
        }
    }
private:
    double _tol;
    double _tol8u;
};


class AbsSimilarPoints : public Wrappable<AbsSimilarPoints>
{
public:
    AbsSimilarPoints(double tol, double percent) : _tol(tol), _percent(percent) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        Mat diff;
        cv::absdiff(in1, in2, diff);
        Mat err_mask = diff > _tol;
        int err_points = cv::countNonZero(err_mask.reshape(1));
        double max_err_points = _percent * std::max((size_t)1000, in1.total());
        if (err_points > max_err_points)
        {
            std::cout << "AbsSimilarPoints error: err_points=" << err_points
                      << "  max_err_points=" << max_err_points << " (total=" << in1.total() << ")"
                      << "  diff_tolerance=" << _tol << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
private:
    double _tol;
    double _percent;
};


class ToleranceFilter : public Wrappable<ToleranceFilter>
{
public:
    ToleranceFilter(double tol, double tol8u, double inf_tol = 2.0) : _tol(tol), _tol8u(tol8u), _inf_tol(inf_tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        int depth = CV_MAT_DEPTH(in1.type());
        {
            double err_Inf = cv::norm(in1, in2, NORM_INF);
            if (err_Inf > _inf_tol)
            {
                std::cout << "ToleranceFilter error: err_Inf=" << err_Inf << "  tolerance=" << _inf_tol << std::endl;
                return false;
            }
            double err = cv::norm(in1, in2, NORM_L2 | NORM_RELATIVE);
            double tolerance = depth >= CV_32F ? _tol : _tol8u;
            if (err > tolerance)
            {
                std::cout << "ToleranceFilter error: err=" << err << "  tolerance=" << tolerance
                          << "  depth=" << cv::depthToString(depth)
                          << std::endl;
                return false;
            }
        }
        return true;
    }
private:
    double _tol;
    double _tol8u;
    double _inf_tol;
};

class ToleranceColor : public Wrappable<ToleranceColor>
{
public:
    ToleranceColor(double tol, double inf_tol = 2.0) : _tol(tol), _inf_tol(inf_tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        {
            double err_Inf = cv::norm(in1, in2, NORM_INF);
            if (err_Inf > _inf_tol)
            {
                std::cout << "ToleranceColor error: err_Inf=" << err_Inf << "  tolerance=" << _inf_tol << std::endl;;
                return false;
            }
            double err = cv::norm(in1, in2, NORM_L1 | NORM_RELATIVE);
            if (err > _tol)
            {
                std::cout << "ToleranceColor error: err=" << err << "  tolerance=" << _tol << std::endl;;
                return false;
            }
        }
        return true;
    }
private:
    double _tol;
    double _inf_tol;
};

class AbsToleranceScalar : public WrappableScalar<AbsToleranceScalar>
{
public:
    AbsToleranceScalar(double tol) : _tol(tol) {}
    bool operator() (const cv::Scalar& in1, const cv::Scalar& in2) const
    {
        double abs_err = std::abs(in1[0] - in2[0]) / std::max(1.0, std::abs(in2[0]));
        if (abs_err > _tol)
        {
            std::cout << "AbsToleranceScalar error: abs_err=" << abs_err << "  tolerance=" << _tol << " in1[0]" << in1[0] << " in2[0]" << in2[0] << std::endl;;
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

namespace
{
    inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_scalar_f&)
    {
        return os << "compare_scalar_f";
    }
}
