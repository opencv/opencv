/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_CORE_AFFINE3_HPP
#define OPENCV_CORE_AFFINE3_HPP

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

//! @addtogroup core
//! @{

    /** @brief Affine transform
      @todo document
     */
    template<typename T>
    class Affine3
    {
    public:
        typedef T float_type;
        typedef Matx<float_type, 3, 3> Mat3;
        typedef Matx<float_type, 4, 4> Mat4;
        typedef Vec<float_type, 3> Vec3;

        Affine3();

        //! Augmented affine matrix
        Affine3(const Mat4& affine);

        //! Rotation matrix
        Affine3(const Mat3& R, const Vec3& t = Vec3::all(0));

        //! Rodrigues vector
        Affine3(const Vec3& rvec, const Vec3& t = Vec3::all(0));

        //! Combines all contructors above. Supports 4x4, 4x3, 3x3, 1x3, 3x1 sizes of data matrix
        explicit Affine3(const Mat& data, const Vec3& t = Vec3::all(0));

        //! From 16th element array
        explicit Affine3(const float_type* vals);

        //! Create identity transform
        static Affine3 Identity();

        //! Rotation matrix
        void rotation(const Mat3& R);

        //! Rodrigues vector
        void rotation(const Vec3& rvec);

        //! Combines rotation methods above. Suports 3x3, 1x3, 3x1 sizes of data matrix;
        void rotation(const Mat& data);

        void linear(const Mat3& L);
        void translation(const Vec3& t);

        Mat3 rotation() const;
        Mat3 linear() const;
        Vec3 translation() const;

        //! Rodrigues vector
        Vec3 rvec() const;

        Affine3 inv(int method = cv::DECOMP_SVD) const;

        //! a.rotate(R) is equivalent to Affine(R, 0) * a;
        Affine3 rotate(const Mat3& R) const;

        //! a.rotate(rvec) is equivalent to Affine(rvec, 0) * a;
        Affine3 rotate(const Vec3& rvec) const;

        //! a.translate(t) is equivalent to Affine(E, t) * a;
        Affine3 translate(const Vec3& t) const;

        //! a.concatenate(affine) is equivalent to affine * a;
        Affine3 concatenate(const Affine3& affine) const;

        template <typename Y> operator Affine3<Y>() const;

        template <typename Y> Affine3<Y> cast() const;

        Mat4 matrix;

#if defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H
        Affine3(const Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>& affine);
        Affine3(const Eigen::Transform<T, 3, Eigen::Affine>& affine);
        operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>() const;
        operator Eigen::Transform<T, 3, Eigen::Affine>() const;
#endif
    };

    template<typename T> static
    Affine3<T> operator*(const Affine3<T>& affine1, const Affine3<T>& affine2);

    template<typename T, typename V> static
    V operator*(const Affine3<T>& affine, const V& vector);

    typedef Affine3<float> Affine3f;
    typedef Affine3<double> Affine3d;

    static Vec3f operator*(const Affine3f& affine, const Vec3f& vector);
    static Vec3d operator*(const Affine3d& affine, const Vec3d& vector);

    template<typename _Tp> class DataType< Affine3<_Tp> >
    {
    public:
        typedef Affine3<_Tp>                               value_type;
        typedef Affine3<typename DataType<_Tp>::work_type> work_type;
        typedef _Tp                                        channel_type;

        enum { generic_type = 0,
               depth        = DataType<channel_type>::depth,
               channels     = 16,
               fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
               type         = CV_MAKETYPE(depth, channels)
             };

        typedef Vec<channel_type, channels> vec_type;
    };

//! @} core

}

//! @cond IGNORED

///////////////////////////////////////////////////////////////////////////////////
// Implementaiton

template<typename T> inline
cv::Affine3<T>::Affine3()
    : matrix(Mat4::eye())
{}

template<typename T> inline
cv::Affine3<T>::Affine3(const Mat4& affine)
    : matrix(affine)
{}

template<typename T> inline
cv::Affine3<T>::Affine3(const Mat3& R, const Vec3& t)
{
    rotation(R);
    translation(t);
    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const Vec3& _rvec, const Vec3& t)
{
    rotation(_rvec);
    translation(t);
    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const cv::Mat& data, const Vec3& t)
{
    CV_Assert(data.type() == cv::DataType<T>::type);

    if (data.cols == 4 && data.rows == 4)
    {
        data.copyTo(matrix);
        return;
    }
    else if (data.cols == 4 && data.rows == 3)
    {
        rotation(data(Rect(0, 0, 3, 3)));
        translation(data(Rect(3, 0, 1, 3)));
        return;
    }

    rotation(data);
    translation(t);
    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const float_type* vals) : matrix(vals)
{}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::Identity()
{
    return Affine3<T>(cv::Affine3<T>::Mat4::eye());
}

template<typename T> inline
void cv::Affine3<T>::rotation(const Mat3& R)
{
    linear(R);
}

template<typename T> inline
void cv::Affine3<T>::rotation(const Vec3& _rvec)
{
    double theta = norm(_rvec);

    if (theta < DBL_EPSILON)
        rotation(Mat3::eye());
    else
    {
        double c = std::cos(theta);
        double s = std::sin(theta);
        double c1 = 1. - c;
        double itheta = (theta != 0) ? 1./theta : 0.;

        Point3_<T> r = _rvec*itheta;

        Mat3 rrt( r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z );
        Mat3 r_x( 0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0 );

        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        // where [r_x] is [0 -rz ry; rz 0 -rx; -ry rx 0]
        Mat3 R = c*Mat3::eye() + c1*rrt + s*r_x;

        rotation(R);
    }
}

//Combines rotation methods above. Suports 3x3, 1x3, 3x1 sizes of data matrix;
template<typename T> inline
void cv::Affine3<T>::rotation(const cv::Mat& data)
{
    CV_Assert(data.type() == cv::DataType<T>::type);

    if (data.cols == 3 && data.rows == 3)
    {
        Mat3 R;
        data.copyTo(R);
        rotation(R);
    }
    else if ((data.cols == 3 && data.rows == 1) || (data.cols == 1 && data.rows == 3))
    {
        Vec3 _rvec;
        data.reshape(1, 3).copyTo(_rvec);
        rotation(_rvec);
    }
    else
        CV_Assert(!"Input marix can be 3x3, 1x3 or 3x1");
}

template<typename T> inline
void cv::Affine3<T>::linear(const Mat3& L)
{
    matrix.val[0] = L.val[0]; matrix.val[1] = L.val[1];  matrix.val[ 2] = L.val[2];
    matrix.val[4] = L.val[3]; matrix.val[5] = L.val[4];  matrix.val[ 6] = L.val[5];
    matrix.val[8] = L.val[6]; matrix.val[9] = L.val[7];  matrix.val[10] = L.val[8];
}

template<typename T> inline
void cv::Affine3<T>::translation(const Vec3& t)
{
    matrix.val[3] = t[0]; matrix.val[7] = t[1]; matrix.val[11] = t[2];
}

template<typename T> inline
typename cv::Affine3<T>::Mat3 cv::Affine3<T>::rotation() const
{
    return linear();
}

template<typename T> inline
typename cv::Affine3<T>::Mat3 cv::Affine3<T>::linear() const
{
    typename cv::Affine3<T>::Mat3 R;
    R.val[0] = matrix.val[0];  R.val[1] = matrix.val[1];  R.val[2] = matrix.val[ 2];
    R.val[3] = matrix.val[4];  R.val[4] = matrix.val[5];  R.val[5] = matrix.val[ 6];
    R.val[6] = matrix.val[8];  R.val[7] = matrix.val[9];  R.val[8] = matrix.val[10];
    return R;
}

template<typename T> inline
typename cv::Affine3<T>::Vec3 cv::Affine3<T>::translation() const
{
    return Vec3(matrix.val[3], matrix.val[7], matrix.val[11]);
}

template<typename T> inline
typename cv::Affine3<T>::Vec3 cv::Affine3<T>::rvec() const
{
    cv::Vec3d w;
    cv::Matx33d u, vt, R = rotation();
    cv::SVD::compute(R, w, u, vt, cv::SVD::FULL_UV + cv::SVD::MODIFY_A);
    R = u * vt;

    double rx = R.val[7] - R.val[5];
    double ry = R.val[2] - R.val[6];
    double rz = R.val[3] - R.val[1];

    double s = std::sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.val[0] + R.val[4] + R.val[8] - 1) * 0.5;
    c = c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c;
    double theta = acos(c);

    if( s < 1e-5 )
    {
        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            double t;
            t = (R.val[0] + 1) * 0.5;
            rx = std::sqrt(std::max(t, 0.0));
            t = (R.val[4] + 1) * 0.5;
            ry = std::sqrt(std::max(t, 0.0)) * (R.val[1] < 0 ? -1.0 : 1.0);
            t = (R.val[8] + 1) * 0.5;
            rz = std::sqrt(std::max(t, 0.0)) * (R.val[2] < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R.val[5] > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= std::sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }

    return cv::Vec3d(rx, ry, rz);
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::inv(int method) const
{
    return matrix.inv(method);
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::rotate(const Mat3& R) const
{
    Mat3 Lc = linear();
    Vec3 tc = translation();
    Mat4 result;
    result.val[12] = result.val[13] = result.val[14] = 0;
    result.val[15] = 1;

    for(int j = 0; j < 3; ++j)
    {
        for(int i = 0; i < 3; ++i)
        {
            float_type value = 0;
            for(int k = 0; k < 3; ++k)
                value += R(j, k) * Lc(k, i);
            result(j, i) = value;
        }

        result(j, 3) = R.row(j).dot(tc.t());
    }
    return result;
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::rotate(const Vec3& _rvec) const
{
    return rotate(Affine3f(_rvec).rotation());
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::translate(const Vec3& t) const
{
    Mat4 m = matrix;
    m.val[ 3] += t[0];
    m.val[ 7] += t[1];
    m.val[11] += t[2];
    return m;
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::concatenate(const Affine3<T>& affine) const
{
    return (*this).rotate(affine.rotation()).translate(affine.translation());
}

template<typename T> template <typename Y> inline
cv::Affine3<T>::operator Affine3<Y>() const
{
    return Affine3<Y>(matrix);
}

template<typename T> template <typename Y> inline
cv::Affine3<Y> cv::Affine3<T>::cast() const
{
    return Affine3<Y>(matrix);
}

template<typename T> inline
cv::Affine3<T> cv::operator*(const cv::Affine3<T>& affine1, const cv::Affine3<T>& affine2)
{
    return affine2.concatenate(affine1);
}

template<typename T, typename V> inline
V cv::operator*(const cv::Affine3<T>& affine, const V& v)
{
    const typename Affine3<T>::Mat4& m = affine.matrix;

    V r;
    r.x = m.val[0] * v.x + m.val[1] * v.y + m.val[ 2] * v.z + m.val[ 3];
    r.y = m.val[4] * v.x + m.val[5] * v.y + m.val[ 6] * v.z + m.val[ 7];
    r.z = m.val[8] * v.x + m.val[9] * v.y + m.val[10] * v.z + m.val[11];
    return r;
}

static inline
cv::Vec3f cv::operator*(const cv::Affine3f& affine, const cv::Vec3f& v)
{
    const cv::Matx44f& m = affine.matrix;
    cv::Vec3f r;
    r.val[0] = m.val[0] * v[0] + m.val[1] * v[1] + m.val[ 2] * v[2] + m.val[ 3];
    r.val[1] = m.val[4] * v[0] + m.val[5] * v[1] + m.val[ 6] * v[2] + m.val[ 7];
    r.val[2] = m.val[8] * v[0] + m.val[9] * v[1] + m.val[10] * v[2] + m.val[11];
    return r;
}

static inline
cv::Vec3d cv::operator*(const cv::Affine3d& affine, const cv::Vec3d& v)
{
    const cv::Matx44d& m = affine.matrix;
    cv::Vec3d r;
    r.val[0] = m.val[0] * v[0] + m.val[1] * v[1] + m.val[ 2] * v[2] + m.val[ 3];
    r.val[1] = m.val[4] * v[0] + m.val[5] * v[1] + m.val[ 6] * v[2] + m.val[ 7];
    r.val[2] = m.val[8] * v[0] + m.val[9] * v[1] + m.val[10] * v[2] + m.val[11];
    return r;
}



#if defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H

template<typename T> inline
cv::Affine3<T>::Affine3(const Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>& affine)
{
    cv::Mat(4, 4, cv::DataType<T>::type, affine.matrix().data()).copyTo(matrix);
}

template<typename T> inline
cv::Affine3<T>::Affine3(const Eigen::Transform<T, 3, Eigen::Affine>& affine)
{
    Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)> a = affine;
    cv::Mat(4, 4, cv::DataType<T>::type, a.matrix().data()).copyTo(matrix);
}

template<typename T> inline
cv::Affine3<T>::operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>() const
{
    Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)> r;
    cv::Mat hdr(4, 4, cv::DataType<T>::type, r.matrix().data());
    cv::Mat(matrix, false).copyTo(hdr);
    return r;
}

template<typename T> inline
cv::Affine3<T>::operator Eigen::Transform<T, 3, Eigen::Affine>() const
{
    return this->operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>();
}

#endif /* defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H */

//! @endcond

#endif /* __cplusplus */

#endif /* OPENCV_CORE_AFFINE3_HPP */
