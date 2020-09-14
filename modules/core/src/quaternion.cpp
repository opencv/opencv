// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.  
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2020, Huawei Technologies Co., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and 
// limitations under the License. 
//
// Author: Longbu Wang <riskiest@gmail.com>
//         Liangqian Kong <chargerKong@126.com>:


#include "precomp.hpp"
#include "opencv2/core/quaternion.hpp"
#define EPS 0.0001
#include <vector>
using namespace cv;
template <typename T>
Quat<T>::Quat(const cv::Vec<T, 4> &coeff):w(coeff[0]), x(coeff[1]), y(coeff[2]), z(coeff[3]){}


template <typename T>
Quat<T>::Quat(const T qw, const T qx, const T qy, const T qz):w(qw), x(qx), y(qy), z(qz){}

template <typename T>
Quat<T>::Quat(const T angle, const cv::Vec<T, 3> &axis)
{ 	
    T vNorm = cv::sqrt(axis.dot(axis));
    w = std::cos(angle / 2);
    x = std::sin(angle / 2) * (axis[0] / vNorm);
    y = std::sin(angle / 2) * (axis[1] / vNorm);
    z = std::sin(angle / 2) * (axis[2] / vNorm);
}

template <typename T>
Quat<T>::Quat(const cv::Mat &R)
{
    assert(R.rows == 3 && R.cols == 3);
    T S;
    T trace = R.at<T>(0, 0) + R.at<T>(1, 1) + R.at<T>(2, 2);
    if (trace > 0)
    {
        S = sqrt(trace + 1) * 2;
        x = (R.at<T>(1, 2) - R.at<T>(2, 1)) / S;
        y = (R.at<T>(2, 0) - R.at<T>(0, 2)) / S;
        z = (R.at<T>(0, 1) - R.at<T>(1, 0)) / S;
        w = 0.25 * S;
    }
    else if (R.at<T>(0, 0) > R.at<T>(1, 1) && R.at<T>(0, 0) > R.at<T>(2, 2))
    {

        S = sqrt(1.0 + R.at<T>(0, 0) - R.at<T>(1, 1) - R.at<T>(2, 2)) * 2;
        x = 0.25 * S;
        y = (R.at<T>(1, 0) + R.at<T>(0, 1)) / S;
        z = (R.at<T>(0, 2) + R.at<T>(2, 0)) / S;
        w = (R.at<T>(1, 2) - R.at<T>(2, 1)) / S;
    }
    else if (R.at<T>(1, 1) > R.at<T>(2, 2))
    {
        S = sqrt(1.0 - R.at<T>(0, 0) + R.at<T>(1, 1) - R.at<T>(2, 2)) * 2;
        x = (R.at<T>(0, 1) + R.at<T>(1, 0)) / S;
        y = 0.25 * S;
        z = (R.at<T>(1, 2) + R.at<T>(2, 1)) / S;
        w = (R.at<T>(0, 2) - R.at<T>(2, 0)) / S;
    }
    else
    {
        S = sqrt(1.0 - R.at<T>(0, 0) - R.at<T>(1, 1) + R.at<T>(2, 2)) * 2;
        x = (R.at<T>(0, 2) + R.at<T>(2, 0)) / S;
        y = (R.at<T>(1, 2) + R.at<T>(2, 1)) / S;
        z = 0.25 * S;
        w = (R.at<T>(0, 1) - R.at<T>(1, 0)) / S;
    }
}


template <typename T>
inline Quat<T> Quat<T>::operator-() const
{
    return Quat<T>(-w, -x, -y, -z);
}


template <typename T>
inline bool Quat<T>::operator==(const Quat<T> &q) const
{
    return (w == q.w && x == q.x && y == q.y && z == q.z);
}

template <typename T>
inline Quat<T> Quat<T>::operator+(const Quat<T> &q1) const
{
    return Quat<T>(w + q1.w, x + q1.x, y + q1.y, z + q1.z);
}

template <typename T>
inline Quat<T> Quat<T>::operator-(const Quat<T> &q1) const
{
    return Quat<T>(w - q1.w, x - q1.x, y - q1.y, z - q1.z);
}

template <typename T>
inline Quat<T>& Quat<T>::operator+=(const Quat<T> &q1)
{
    w += q1.w;
    x += q1.x;
    y += q1.y;
    z += q1.z;
    return *this;
}

template <typename T>
inline Quat<T>& Quat<T>::operator-=(const Quat<T> &q1)
{
    w -= q1.w;
    x -= q1.x;
    y -= q1.y;
    z -= q1.z;
    return *this;
}

template <typename T>
inline Quat<T> Quat<T>::operator*(const Quat<T> &q1) const
{
    cv::Vec<T, 4> q{w, x, y, z};
    cv::Vec<T, 4> q2{q1.w, q1.x, q1.y, q1.z};
    return Quat<T>(q * q2);
}


template <typename T>
Quat<T> cv::operator*(const Quat<T> &q1, const T a)
{
    return Quat<T>(a * q1.w, a * q1.x, a * q1.y, a * q1.z);
}

template <typename T>
Quat<T> cv::operator*(const T a, const Quat<T> &q1)
{
    return Quat<T>(a * q1.w, a * q1.x, a * q1.y, a * q1.z);
}

template <typename T>
inline Quat<T>& Quat<T>::operator*=(const Quat<T> &q1)
{
    T qw, qx, qy, qz;
    qw = w * q1.w - x * q1.x - y * q1.y - z * q1.z;
    qx = x * q1.w + w * q1.x + y * q1.z - z * q1.y;
    qy = y * q1.w + w * q1.y + z * q1.x - x * q1.z;
    qz = z * q1.w + w * q1.z + x * q1.y - y * q1.x;
    w = qw;
    x = qx;
    y = qy;
    z = qz;
    return *this;
}

template <typename T>
Quat<T>& Quat<T>::operator*=(const T &q1)
{
    w *= q1;
    x *= q1;
    y *= q1;
    z *= q1;
    return *this;
}

template <typename T>
inline Quat<T>& Quat<T>::operator/=(const T &a)
{
    w /= a;
    x /= a;
    y /= a;
    z /= a;
    return *this;
}

template <typename T>
inline Quat<T> Quat<T>::operator/(const T &a) const
{
    return Quat<T>(w / a, x / a, y / a, z / a);
}

template <typename T>
inline Quat<T> Quat<T>::operator/(const Quat<T> &q) const
{
    return *this * q.inv();
}

template <typename T>
inline const T& Quat<T>::operator[](std::size_t n) const
{
    switch (n) {
        case 0:
            return w;
            break;
        case 1:
            return x;
            break;
        case 2:
            return y;
            break;
        case 3:
            return z;
            break;
        default:
            throw ("subscript exceeds the index range");
    }
}

template <typename T>
inline T& Quat<T>::operator[](std::size_t n)
{	
    switch (n) {
        case 0:
            return w;
            break;
        case 1:
            return x;
            break;
        case 2:
            return y;
            break;
        case 3:
            return z;
            break;
        default:
            throw ("subscript exceeds the index range");
    }
}

template <typename T>
std::ostream & cv::operator<<(std::ostream &os, const Quat<T> &q)
{
    os << "Quat " << cv::Vec<T, 4>{q.w, q.x, q.y, q.z};
    return os;
}

template <typename T>
inline Quat<T> Quat<T>::conjugate() const
{
    return Quat<T>(w, -x, -y, -z);
}

template <typename T>
inline T Quat<T>::norm() const
{
    return std::sqrt(dot(*this));
}

/*
template <typename T>
Quat<T> cv::exp(const Quat<T> &q)
{
    return q.exp();
}
*/

template <typename T>
Quat<T> Quat<T>::exp() const
{
    cv::Vec<T, 3> v{x, y, z};
    T normV = std::sqrt(v.dot(v));
    T k = normV < EPS ? 1 : std::sin(normV) / normV;
    return std::exp(w) * Quat<T>(std::cos(normV), v[0] * k, v[1] * k, v[2] * k);
}

/*
template <typename T>
Quat<T> cv::log(const Quat<T> &q)
{
    return q.log();
}
*/		

template <typename T>
Quat<T> Quat<T>::log() const
{
    cv::Vec<T, 3> v{x, y, z};
    T vNorm = std::sqrt(v.dot(v));
    T qNorm = norm();
    T k = vNorm < EPS ? 1 : std::acos(w / qNorm) / vNorm;
    return Quat<T>(std::log(qNorm), v[0] * k, v[1] * k, v[2] *k);
}

/*
template <typename T>
inline Quat<T> cv::power(const Quat<T> &q1, T x)
{
    return q1.power(x);
}
*/

template <typename T>
inline Quat<T> Quat<T>::power(T x1) const
{
    T angle = getAngle();
    cv::Vec<T, 3> axis = getAxis();
    return std::pow(norm(), x1) * Quat<T>(x1 * angle, axis);
}


template <typename T>
inline Quat<T> sqrt(Quat<T> &q)
{
    return q.sqrt();
}

template <typename T>
inline Quat<T> Quat<T>::sqrt() const
{
    return power(0.5);
}

/*
template <typename T>
inline Quat<T> cv::power(const Quat<T> &p, const Quat<T> &q)
{
    return p.power(q);
}
*/

template <typename T>
inline Quat<T> Quat<T>::power(const Quat<T> &q) const
{
    Quat<T> ans = *this * q.log();
    return ans.exp();
}

template <typename T>
inline T Quat<T>::dot(Quat<T> q1) const
{
    return w * q1.w + x * q1.x + y * q1.y + z * q1.z;
}

/*
template <typename T>
inline Quat<T> cv::crossProduct(const Quat<T> &p, const Quat<T> &q)
{
    return p.crossProduct(q);
}
*/

template <typename T>
inline Quat<T> Quat<T>::crossProduct(const Quat<T> &q) const
{
    return (*this * q - q * *this) / 2;
}

template <typename T>
inline Quat<T> Quat<T>::normalize() const
{
    return Quat<T>(w / norm(), x / norm(), y / norm(), z / norm()) ;
}
/*
template <typename T>
inline Quat<T> cv::inv(const Quat<T> &q)
{
    return q.inv();
}
*/

template <typename T>
inline Quat<T> Quat<T>::inv() const
{
    return conjugate() / pow(norm(), 2);
}
/*
template <typename T>
inline Quat<T> cv::sinh(const Quat<T> &q)
{
    return q.sinh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::sinh() const
{
    cv::Vec<T, 3> v{x, y ,z};
    T vNorm = std::sqrt(v.dot(v));
    T k = std::cosh(w) * std::sin(vNorm) / vNorm;
    return Quat<T>(std::sinh(w) * std::cos(vNorm), v[0] * k, v[1] * k, v[2] * k);
}

/*
template <typename T>
inline Quat<T> cv::cosh(const Quat<T> &q)
{
    return q.cosh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::cosh() const
{
    cv::Vec<T, 3> v{x, y ,z};
    T vNorm = std::sqrt(v.dot(v));
    T k = std::sinh(w) * std::sin(vNorm) / vNorm;
    return Quat<T>(std::cosh(w) * std::cos(vNorm), v[0] * k, v[1] * k, v[2] * k);
}

/*

template <typename T>
inline Quat<T> cv::tanh(const Quat<T> &q)
{
    return q.tanh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::tanh() const
{
    return sinh() * cosh().inv();
}
/*

template <typename T>
inline Quat<T> cv::sin(const Quat<T> &q)
{
    return q.sin();
}
*/

template <typename T>
inline Quat<T> Quat<T>::sin() const
{
    cv::Vec<T, 3> v{x, y ,z};
    T vNorm = std::sqrt(v.dot(v));
    T k = std::cos(w) * std::sinh(vNorm) / vNorm;
    return Quat<T>(std::sin(w) * std::cosh(vNorm), v[0] * k, v[1] * k, v[2] * k);
}
/*

template <typename T>
inline Quat<T> cv::cos(const Quat<T> &q)
{
    return q.cos();
}
*/

template <typename T>
inline Quat<T> Quat<T>::cos() const
{
    cv::Vec<T, 3> v{x, y ,z};
    T vNorm = std::sqrt(v.dot(v));
    T k = std::sin(w) * std::sinh(vNorm) / vNorm;
    return Quat<T>(std::cos(w) * std::cosh(vNorm), -v[0] * k, -v[1] * k, -v[2] * k);
}
/*

template <typename T>
inline Quat<T> cv::tan(const Quat<T> &q)
{
    return q.tan();
}
*/

template <typename T>
inline Quat<T> Quat<T>::tan() const
{
    return sin() * cos().inv();
}
/*

template <typename T>
inline Quat<T> cv::asinh(Quat<T> &q)
{
    return q.asinh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::asinh() const
{
    Quat<T> c1 = *this * *this + Quat<T>(1,0,0,0);
    Quat<T> c2 = c1.power(0.5) + *this;
    return c2.log();
    // return log(*this + power(*this * *this + Quat<T>(1,0,0,0), 0.5));
}
/*

template <typename T>
inline Quat<T> cv::acosh(const Quat<T> &q)
{
    return q.acosh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::acosh() const
{
    Quat<T> c1 = *this * *this - Quat<T>(1,0,0,0);
    Quat<T> c2 = c1.power(0.5) + *this;
    return c2.log();
    //return cv::log(*this + cv::power(*this * *this - Quat<T>(1,0,0,0), 0.5));
}
/*

template <typename T>
inline Quat<T> cv::atanh(const Quat<T> &q)
{
    return q.atanh();
}
*/

template <typename T>
inline Quat<T> Quat<T>::atanh() const
{
    Quat<T> ident(1, 0, 0, 0);
    Quat<T> c1 = (ident + *this).log();
    Quat<T> c2 = (ident - *this).log();
    return 1 / 2 * (c1 - c2);
    //return 1/2 * (cv::log(ident + *this) - cv::log(ident - *this));
}
/*

template <typename T>
inline Quat<T> cv::asin(const Quat<T> &q)
{
    return q.asin();
}
*/

template <typename T>
inline Quat<T> Quat<T>::asin() const
{
    Quat<T> v(0, x, y, z);
    T vNorm = v.norm();
    return -v / vNorm * (*this * v / vNorm).asinh();
}
/*

template <typename T>
inline Quat<T> cv::acos(const Quat<T> &q)
{
    return q.acos();
}
*/

template <typename T>
inline Quat<T> Quat<T>::acos() const
{
    Quat<T> v(0, x, y, z);
    T vNorm = v.norm();
    return -v / vNorm * acosh();
}
/*

template <typename T>
inline Quat<T> cv::atan(const Quat<T> &q)
{
    return q.atan();
}
*/
template <typename T>
inline Quat<T> Quat<T>::atan() const
{
    Quat<T> v(0, x, y, z);
    T vNorm = v.norm();
    return -v / vNorm * (*this * v / vNorm).atanh();
}

template <typename T>
inline T Quat<T>::getAngle() const
{
    return 2 * std::acos(w / norm());
}

template <typename T>
inline cv::Vec<T, 3> Quat<T>::getAxis() const
{
    T angle = getAngle();
    if (abs(std::sin(angle / 2)) < EPS)
        return cv::Vec<T, 3> {x, y, z}; // TBD
    return cv::Vec<T, 3> {x, y, z} / (norm() * std::sin(angle / 2));
}

template <typename T>
cv::Mat Quat<T>::toRotMat4x4() const
{
    T dotVal = dot(*this);
    cv::Matx<T, 4, 4> R{
         dotVal, 0                           , 0                           , 0,
         0     , dotVal - 2 * (y * y + z * z),  2 * (x * y + w * z)        , 2 * (x * z - w * y),
         0     , 2 * (x * y - w * z)         , dotVal - 2 * (x * x + z * z), 2 * (y * z + w * x),
         0     , 2 * (x * z + w * y)         , 2 * (y * z - w * x)         , dotVal - 2 * (x * x + y * y)};
    return cv::Mat(R).t();
}

template <typename T>
cv::Mat Quat<T>::toRotMat3x3() const
{
    assertNormal();
    cv::Matx<T, 3, 3> R{
          1 - 2 * (y * y + z * z), 2 * (x * y + w * z)    , 2 * (x * z - w * y),
          2 * (x * y - w * z)    , 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
          2 * (x * z + w * y)    , 2 * (y * z - w * x)    , 1 - 2 * (x * x + y * y)};
    return cv::Mat(R).t();
}

template <typename T>
cv::Vec<T, 4> Quat<T>::toVec() const
{
    return cv::Vec<T, 4>{w, x, y, z};
}

template <typename T>
Quat<T> Quat<T>::lerp(const Quat<T> &q0, const Quat<T> &q1, const T t)
{
    return (1 - t) * q0 + t * q1;
}

template <typename T>
Quat<T> Quat<T>::slerp(Quat<T> &q0, Quat<T> &q1, const T t, bool assumeUnit)
{
    if (!assumeUnit)
    {
        q0 = q0.normalize();
        q1 = q1.normalize();
        // add warning:
    }
    T cosTheta = q0.dot(q1);
    T DOT_THRESHOLD = 0.995;
    /*if (cosTheta < 0)
    {
        q1 = q1;
    }
    */
    if (cosTheta > DOT_THRESHOLD)
        return nlerp(q0, q1, t);
    T sinTheta = std::sqrt(1 - cosTheta * cosTheta);
    T angle = atan2(sinTheta, cosTheta);
    return (std::sin((1 - t) * angle) / (sinTheta) * q0 + std::sin(t * angle) / (sinTheta) * q1).normalize();
}

template <typename T>
inline Quat<T> Quat<T>::nlerp(const Quat<T> &q0, const Quat<T> &q1, const T t)
{
    return ((1 - t) * q0 + t * q1).normalize();
}


template <typename T>
inline bool Quat<T>::isNormal() const
{

    double normVar = norm();
    if ((normVar > 1 - EPS) && (normVar < 1 + EPS))
        return true;
    return false;
}

template <typename T>
inline void Quat<T>::assertNormal() const
{
    if (!isNormal())
        throw ("Quaternions should be normalized");
}

template <typename T>
inline Quat<T> Quat<T>::squad(Quat<T> &q0, Quat<T> &q1,
							  Quat<T> &q2, Quat<T> &q3, const T t, bool assumeUnit)
{
    if (!assumeUnit)
    {
        q0 = q0.normalize();
        q1 = q1.normalize();
        q2 = q2.normalize();
        q3 = q3.normalize();
        // add warning in inter
    }

    Quat<T> c0 = slerp(q0, q3, t, assumeUnit);
    Quat<T> c1 = slerp(q1, q2, t, assumeUnit);
    return slerp(c0, c1, 2 * t * (1 - t), assumeUnit);
}

template <typename T>
Quat<T> Quat<T>::interPoint(Quat<T> &q0, Quat<T> &q1,
							Quat<T> &q2, bool assumeUnit)
{
    if (!assumeUnit)
    {
        q0 = q0.normalize();
        q1 = q1.normalize();
        q2 = q2.normalize();
        // add warning in inter
    }
    
    Quat<T> c1 = q1.conjugate() * q0;
    Quat<T> c2 = q1.conjugate() * q2;
    Quat<T> log1 = c1.log();
    Quat<T> log22 = c2.log();
    return q1 * ((-log1 - log22) / 4).exp();
    //return q1 * cv::exp(-(cv::log(q1.conjugate() * q0 + cv::log(q1.conjugate() * q2))) / 4);
}

template <typename T>
Quat<T> Quat<T>::spline(Quat<T> &q0, Quat<T> &q1, Quat<T> &q2, Quat<T> &q3, const T t, bool assumeUnit)
{
    if (!assumeUnit)
    {
        q0 = q0.normalize();
        q1 = q1.normalize();
        q2 = q2.normalize();
        q3 = q3.normalize();
        // add warning
    }
    T cosTheta;
    std::vector<Quat<T>> vec{q0, q1, q2};
    for (auto &i: vec)
    {
        cosTheta = q3.dot(i);
        if (cosTheta < 0)
        {
            i = -i;
        }
    }
    Quat<T> s1 = interPoint(q0, q1, q2, assumeUnit);
    Quat<T> s2 = interPoint(q1, q2, q3, assumeUnit);
    
    return squad(q1, s1, s2, q2, t, assumeUnit);
}



