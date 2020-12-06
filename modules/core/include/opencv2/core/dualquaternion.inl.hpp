// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2020, Huawei Technologies Co., Ltd. All rights reserved.
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
// Author: Liangqian Kong <chargerKong@126.com>
//         Longbu Wang <riskiest@gmail.com>

#ifndef OPENCV_CORE_DUALQUATERNION_INL_HPP
#define OPENCV_CORE_DUALQUATERNION_INL_HPP

#ifndef OPENCV_CORE_DUALQUATERNION_HPP
#error This is not a standalone header. Include dualquaternion.hpp instead.
#endif

///////////////////////////////////////////////////////////////////////////////////////
//Implementation
namespace cv {

template <typename T>
DualQuat<T>::DualQuat(const T vw, const T vx, const T vy, const T vz, const T _w, const T _x, const T _y, const T _z):
                      w(vw), x(vx), y(vy), z(vz), w_(_w), x_(_x), y_(_y), z_(_z){};

template <typename T>
DualQuat<T>::DualQuat(const Vec<T, 8> &q):w(q[0]), x(q[1]), y(q[2]), z(q[3]),
                                          w_(q[4]), x_(q[5]), y_(q[6]), z_(q[7]){};

template <typename T>
DualQuat<T> DualQuat<T>::createFromQuat(const Quat<T> &realPart, const Quat<T> &dualPart)
{
    T w = realPart.w;
    T x = realPart.x;
    T y = realPart.y;
    T z = realPart.z;
    T w_ = dualPart.w;
    T x_ = dualPart.x;
    T y_ = dualPart.y;
    T z_ = dualPart.z;
    return DualQuat<T>(w, x, y, z, w_, x_, y_, z_);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromAngleAxisTrans(const T angle, const Vec<T, 3> &axis, const Quat<T> &trans)
{
    Quat<T> r = Quat<T>::createFromAngleAxis(angle, axis);
    return createFromQuat(r, trans * r / 2);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromMat(InputArray _R)
{
    CV_CheckTypeEQ(_R.type(), cv::traits::Type<T>::value, "");

    Mat R = _R.getMat();
    Quat<T> r = Quat<T>::createFromRotMat(R.colRange(0, 3).rowRange(0, 3));
    Quat<T> trans(0, R.at<T>(0, 3), R.at<T>(1, 3), R.at<T>(2, 3));
    return createFromQuat(r, trans * r / 2);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromPitch(const T angle, const T d, const Quat<T> &axis, const Quat<T> &moment)
{
    T half_angle = angle / 2, half_d = d / 2;
    Quat<T> qaxis = axis.normalize();
    Quat<T> dual = Quat<T>(-half_d * std::sin(half_angle), 0.0, 0.0, 0.0) + std::sin(half_angle) * moment +
        half_d * std::cos(half_angle) * qaxis;
    return createFromQuat(Quat<T>::createFromAngleAxis(angle, Vec<T, 3>{qaxis[1], qaxis[2], qaxis[3]}), dual);
}

template <typename T>
inline bool DualQuat<T>::operator==(const DualQuat<T> &q) const
{
    return (abs(w - q.w) < CV_DUAL_QUAT_EPS && abs(x - q.x) < CV_DUAL_QUAT_EPS &&
            abs(y - q.y) < CV_DUAL_QUAT_EPS && abs(z - q.z) < CV_DUAL_QUAT_EPS &&
            abs(w_ - q.w_) < CV_DUAL_QUAT_EPS && abs(x_ - q.x_) < CV_DUAL_QUAT_EPS &&
            abs(y_ - q.y_) < CV_DUAL_QUAT_EPS && abs(z_ - q.z_) < CV_DUAL_QUAT_EPS);
}

template <typename T>
inline Quat<T> DualQuat<T>::getRealQuat() const
{
    return Quat<T>(w, x, y, z);
}

template <typename T>
inline Quat<T> DualQuat<T>::getDualQuat() const
{
    return Quat<T>(w_, x_, y_, z_);
}

template<typename T>
inline DualQuat<T> DualQuat<T>::conjugate() const
{
    return DualQuat<T>(w, -x, -y, -z, w_, -x_, -y_, -z_);
}

template <typename T>
DualQuat<T> DualQuat<T>::norm(QuatAssumeType assumeUnit) const
{
    Quat<T> real = getRealQuat();
    T realNorm = real.norm();
    if (assumeUnit)
    {
        return DualQuat<T>(realNorm, 0, 0, 0, 0, 0, 0, 0);
    }
    Quat<T> dual = getDualQuat();
    if (realNorm < CV_DUAL_QUAT_EPS){
        return DualQuat<T>(0, 0, 0, 0, 0, 0, 0, 0);
    }
    return DualQuat<T>(realNorm, 0, 0, 0, real.dot(dual) / realNorm, 0, 0, 0);
}

template <typename T>
inline Quat<T> DualQuat<T>::getRotation(QuatAssumeType assumeUnit) const
{
    if (assumeUnit)
    {
        return getRealQuat();
    }
    return getRealQuat().normalize();
}

template <typename T>
Quat<T> DualQuat<T>::getTranslation(QuatAssumeType assumeUnit) const
{
    if (assumeUnit)
    {
        return 2.0 * getDualQuat() * getRealQuat().conjugate();
    }
    DualQuat<T> q = normalize();
    return 2.0 * q.getDualQuat() * q.getRealQuat().conjugate();
}

template <typename T>
inline DualQuat<T> DualQuat<T>::normalize() const
{
    DualQuat<T> qNorm = norm();
    if (qNorm.w < CV_DUAL_QUAT_EPS)
    {
        CV_Error(Error::StsBadArg, "Cannot normalize this dual quaternion: the norm is too small.");
    }
    return *this / qNorm;
}

template <typename T>
inline T DualQuat<T>::dot(DualQuat<T> q) const
{
    return q.w * w + q.x * x + q.y * y + q.z * z + q.w_ * w_ + q.x_ * x_ + q.y_ * y_ + q.z_ * z_;
}

template <typename T>
inline DualQuat<T> DualQuat<T>::inv(QuatAssumeType assumeUnit) const
{
    if (assumeUnit){
        return conjugate();
    }
    DualQuat<T> qNorm = norm();
    if (qNorm.w < CV_DUAL_QUAT_EPS)
    {
        CV_Error(Error::StsBadArg, "This dual quaternion do not have inverse dual quaternion");
    }
    DualQuat<T> qNorm2 = qNorm * qNorm;
    DualQuat<T> conj = conjugate();
    Quat<T> p = conj.getRealQuat();
    return createFromQuat(p / qNorm2.w, -p * qNorm2.w_ / (qNorm2.w * qNorm2.w ) + conj.getDualQuat() / qNorm2.w);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator-(const DualQuat<T> &q) const
{
    return DualQuat<T>(w - q.w, x - q.x, y - q.y, z - q.z, w_ - q.w_, x_ - q.x_, y_ - q.y_, z_ - q.z_);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator-() const
{
    return DualQuat<T>(-w, -x, -y, -z, -w_, -x_, -y_, -z_);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator+(const DualQuat<T> &q) const
{
    return DualQuat<T>(w + q.w, x + q.x, y + q.y, z + q.z, w_ + q.w_, x_ + q.x_, y_ + q.y_, z_ + q.z_);
}

template <typename T>
inline DualQuat<T>& DualQuat<T>::operator+=(const DualQuat<T> &q)
{
    *this = *this + q;
    return *this;
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator*(const DualQuat<T> &q) const
{
    Quat<T> A = getRealQuat();
    Quat<T> B = getDualQuat();
    Quat<T> C = q.getRealQuat();
    Quat<T> D = q.getDualQuat();
    return DualQuat<T>::createFromQuat(A * C, A * D + B * C);
}

template <typename T>
inline DualQuat<T>& DualQuat<T>::operator*=(const DualQuat<T> &q)
{
    *this = *this * q;
    return *this;
}

template <typename T>
inline DualQuat<T> operator+(const T a, const DualQuat<T> &q)
{
    return DualQuat<T>(a + q.w, q.x, q.y, q.z, q.w_, q.x_, q.y_, q.z_);
}

template <typename T>
inline DualQuat<T> operator+(const DualQuat<T> &q, const T a)
{
    return DualQuat<T>(a + q.w, q.x, q.y, q.z, q.w_, q.x_, q.y_, q.z_);
}

template <typename T>
inline DualQuat<T> operator-(const DualQuat<T> &q, const T a)
{
    return DualQuat<T>(q.w - a, q.x, q.y, q.z, q.w_, q.x_, q.y_, q.z_);
}

template <typename T>
inline DualQuat<T>& DualQuat<T>::operator-=(const DualQuat<T> &q)
{
    *this = *this - q;
    return *this;
}

template <typename T>
inline DualQuat<T> operator-(const T a, const DualQuat<T> &q)
{
    return DualQuat<T>(a - q.w, -q.x, -q.y, -q.z, -q.w_, -q.x_, -q.y_, -q.z_);
}

template <typename T>
inline DualQuat<T> operator*(const T a, const DualQuat<T> &q)
{
    return DualQuat<T>(q.w * a, q.x * a, q.y * a, q.z * a, q.w_ * a, q.x_ * a, q.y_ * a, q.z_ * a);
}

template <typename T>
inline DualQuat<T> operator*(const DualQuat<T> &q, const T a)
{
    return DualQuat<T>(q.w * a, q.x * a, q.y * a, q.z * a, q.w_ * a, q.x_ * a, q.y_ * a, q.z_ * a);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator/(const T a) const
{
    return DualQuat<T>(w / a, x / a, y / a, z / a, w_ / a, x_ / a, y_ / a, z_ / a);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::operator/(const DualQuat<T> &q) const
{
    return *this * q.inv();
}

template <typename T>
inline DualQuat<T>& DualQuat<T>::operator/=(const DualQuat<T> &q)
{
    *this = *this / q;
    return *this;
}

template <typename T>
std::ostream & operator<<(std::ostream &os, const DualQuat<T> &q)
{
    os << "DualQuat " << Vec<T, 8>{q.w, q.x, q.y, q.z, q.w_, q.x_, q.y_, q.z_};
    return os;
}

template <typename T>
DualQuat<T> DualQuat<T>::exp() const
{
    DualQuat<T> v(0, x, y, z, 0, x_, y_, z_);
    DualQuat<T> normV = v.norm();
    DualQuat<T> sin_normV(std::sin(normV.w), 0, 0, 0, normV.w_ * std::cos(normV.w), 0, 0, 0);
    DualQuat<T> cos_normV(std::cos(normV.w), 0, 0, 0, -normV.w_ * std::sin(normV.w), 0, 0, 0);
    DualQuat<T> exp_w(std::exp(w), 0, 0, 0, w_ * std::exp(w), 0, 0, 0);
    if (normV.w < CV_DUAL_QUAT_EPS)
    {
        return exp_w * (cos_normV + v);
    }
    DualQuat<T> k = sin_normV / normV;
    return exp_w * (cos_normV + v * k);
}

template <typename T>
DualQuat<T> DualQuat<T>::log(QuatAssumeType assumeUnit) const
{
    DualQuat<T> v(0, x, y, z, 0, x_, y_, z_);
    DualQuat<T> normV = v.norm();
    if (assumeUnit)
    {
        if (normV.w < CV_DUAL_QUAT_EPS)
        {
            return v;
        }
        DualQuat<T> k = DualQuat<T>(std::acos(w), 0, 0, 0, -w_ * 1.0 / std::sqrt(1 - w * w), 0, 0, 0) / normV;
        return v * k;
    }
    DualQuat<T> qNorm = norm();
    if (qNorm.w < CV_DUAL_QUAT_EPS)
    {
        CV_Error(Error::StsBadArg, "Cannot apply this quaternion to log function: undefined");
    }
    DualQuat<T> log_qNorm(std::log(qNorm.w), 0, 0, 0, qNorm.w_ / qNorm.w, 0, 0, 0);
    if (normV.w < CV_DUAL_QUAT_EPS)
    {
        return log_qNorm + v;
    }
    DualQuat<T> coeff = DualQuat<T>(w, 0, 0, 0, w_, 0, 0, 0) / qNorm;
    DualQuat<T> k = DualQuat<T>{std::acos(coeff.w), 0, 0, 0, -coeff.w_ / std::sqrt(1 - coeff.w * coeff.w), 0, 0, 0} / normV;
    return log_qNorm + v * k;
}

template <typename T>
DualQuat<T> DualQuat<T>::power(const T t, QuatAssumeType assumeUnit) const
{
    return (t * log(assumeUnit)).exp();
}

template <typename T>
DualQuat<T> DualQuat<T>::power(const DualQuat<T> &q, QuatAssumeType assumeUnit) const
{
    return (q * log(assumeUnit)).exp();
}

template <typename T>
Vec<T, 8> DualQuat<T>::toVec() const
{
   return Vec<T, 8>(w, x, y, z, w_, x_, y_, z_);
}

template <typename T>
Matx<T, 4, 4> DualQuat<T>::toMat() const
{
    Matx<T, 4, 4> rot44 = getRotation().toRotMat4x4();
    Quat<T> translation = getTranslation();
    rot44(0, 3) = translation[1];
    rot44(1, 3) = translation[2];
    rot44(2, 3) = translation[3];
    return rot44;
}

template <typename T>
DualQuat<T> DualQuat<T>::sclerp(const DualQuat<T> &q0, const DualQuat<T> &q1, const T t, bool directChange, QuatAssumeType assumeUnit)
{
    DualQuat<T> v0(q0), v1(q1);
    if (!assumeUnit)
    {
        v0 = v0.normalize();
        v1 = v1.normalize();
    }
    Quat<T> v0Real = v0.getRealQuat();
    Quat<T> v1Real = v1.getRealQuat();
    if (directChange && v1Real.dot(v0Real) < 0)
    {
        v0 = -v0;
    }
    DualQuat<T> v0inv1 = v0.inv() * v1;
    return v0 * v0inv1.power(t, QUAT_ASSUME_UNIT);
}

} //namespace

#endif /*OPENCV_CORE_DUALQUATERNION_INL_HPP*/
