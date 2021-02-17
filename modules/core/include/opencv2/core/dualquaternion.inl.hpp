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
// Author: Liangqian Kong <kongliangqian@huawei.com>
//         Longbu Wang <wanglongbu@huawei.com>

#ifndef OPENCV_CORE_DUALQUATERNION_INL_HPP
#define OPENCV_CORE_DUALQUATERNION_INL_HPP

#ifndef OPENCV_CORE_DUALQUATERNION_HPP
#error This is not a standalone header. Include dualquaternion.hpp instead.
#endif

///////////////////////////////////////////////////////////////////////////////////////
//Implementation
namespace cv {

template <typename T>
DualQuat<T>::DualQuat():w(0), x(0), y(0), z(0), w_(0), x_(0), y_(0), z_(0){};

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
DualQuat<T> DualQuat<T>::createFromAngleAxisTrans(const T angle, const Vec<T, 3> &axis, const Vec<T, 3> &trans)
{
    Quat<T> r = Quat<T>::createFromAngleAxis(angle, axis);
    Quat<T> t{0, trans[0], trans[1], trans[2]};
    return createFromQuat(r, t * r / 2);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromMat(InputArray _R)
{
    CV_CheckTypeEQ(_R.type(), cv::traits::Type<T>::value, "");
    if (_R.size() != Size(4, 4))
    {
        CV_Error(Error::StsBadArg, "The input matrix must have 4 columns and 4 rows");
    }
    Mat R = _R.getMat();
    Quat<T> r = Quat<T>::createFromRotMat(R.colRange(0, 3).rowRange(0, 3));
    Quat<T> trans(0, R.at<T>(0, 3), R.at<T>(1, 3), R.at<T>(2, 3));
    return createFromQuat(r, trans * r / 2);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromAffine3(const Affine3<T> &R)
{
    return createFromMat(R.matrix);
}

template <typename T>
DualQuat<T> DualQuat<T>::createFromPitch(const T angle, const T d, const Vec<T, 3> &axis, const Vec<T, 3> &moment)
{
    T half_angle = angle / 2, half_d = d / 2;
    Quat<T> qaxis = Quat<T>(0, axis[0], axis[1], axis[2]).normalize();
    Quat<T> qmoment = Quat<T>(0, moment[0], moment[1], moment[2]);
    qmoment -= qaxis * axis.dot(moment);
    Quat<T> dual = -half_d * std::sin(half_angle) + std::sin(half_angle) * qmoment +
        half_d * std::cos(half_angle) * qaxis;
    return createFromQuat(Quat<T>::createFromAngleAxis(angle, axis), dual);
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
inline Quat<T> DualQuat<T>::getRealPart() const
{
    return Quat<T>(w, x, y, z);
}

template <typename T>
inline Quat<T> DualQuat<T>::getDualPart() const
{
    return Quat<T>(w_, x_, y_, z_);
}

template <typename T>
inline DualQuat<T> conjugate(const DualQuat<T> &dq)
{
    return dq.conjugate();
}

template <typename T>
inline DualQuat<T> DualQuat<T>::conjugate() const
{
    return DualQuat<T>(w, -x, -y, -z, w_, -x_, -y_, -z_);
}

template <typename T>
DualQuat<T> DualQuat<T>::norm() const
{
    Quat<T> real = getRealPart();
    T realNorm = real.norm();
    Quat<T> dual = getDualPart();
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
        return getRealPart();
    }
    return getRealPart().normalize();
}

template <typename T>
inline Vec<T, 3> DualQuat<T>::getTranslation(QuatAssumeType assumeUnit) const
{
    Quat<T> trans = 2.0 * (getDualPart() * getRealPart().inv(assumeUnit));
    return Vec<T, 3>{trans[1], trans[2], trans[3]};
}

template <typename T>
DualQuat<T> DualQuat<T>::normalize() const
{
    Quat<T> p = getRealPart();
    Quat<T> q = getDualPart();
    T p_norm = p.norm();
    if (p_norm < CV_DUAL_QUAT_EPS)
    {
        CV_Error(Error::StsBadArg, "Cannot normalize this dual quaternion: the norm is too small.");
    }
    Quat<T> p_nr = p / p_norm;
    Quat<T> q_nr = q / p_norm;
    return createFromQuat(p_nr, q_nr - p_nr * p_nr.dot(q_nr));
}

template <typename T>
inline T DualQuat<T>::dot(DualQuat<T> q) const
{
    return q.w * w + q.x * x + q.y * y + q.z * z + q.w_ * w_ + q.x_ * x_ + q.y_ * y_ + q.z_ * z_;
}

template <typename T>
inline DualQuat<T> inv(const DualQuat<T> &dq, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT)
{
    return dq.inv(assumeUnit);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::inv(QuatAssumeType assumeUnit) const
{
    Quat<T> real = getRealPart();
    Quat<T> dual = getDualPart();
    return createFromQuat(real.inv(assumeUnit), -real.inv(assumeUnit) * dual * real.inv(assumeUnit));
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
    Quat<T> A = getRealPart();
    Quat<T> B = getDualPart();
    Quat<T> C = q.getRealPart();
    Quat<T> D = q.getDualPart();
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
inline DualQuat<T> exp(const DualQuat<T> &dq)
{
    return dq.exp();
}

namespace detail {

template <typename _Tp>
Matx<_Tp, 4, 4> jacob_exp(const Quat<_Tp> &q)
{
    _Tp nv = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    _Tp sinc_nv = abs(nv) < cv::DualQuat<_Tp>::CV_DUAL_QUAT_EPS ? 1 - nv * nv / 6 : std::sin(nv) / nv;
    _Tp csiii_nv = abs(nv) < cv::DualQuat<_Tp>::CV_DUAL_QUAT_EPS ? -(_Tp)1.0 / 3 : (std::cos(nv) - sinc_nv) / nv / nv;
    Matx<_Tp, 4, 4> J_exp_quat {
        std::cos(nv), -sinc_nv * q.x,  -sinc_nv * q.y,  -sinc_nv * q.z,
        sinc_nv * q.x, csiii_nv * q.x * q.x + sinc_nv, csiii_nv * q.x * q.y, csiii_nv * q.x * q.z,
        sinc_nv * q.y, csiii_nv * q.y * q.x, csiii_nv * q.y * q.y + sinc_nv, csiii_nv * q.y * q.z,
        sinc_nv * q.z, csiii_nv * q.z * q.x, csiii_nv * q.z * q.y, csiii_nv * q.z * q.z + sinc_nv
    };
    return std::exp(q.w) * J_exp_quat;
}

} // namespace detail

template <typename T>
DualQuat<T> DualQuat<T>::exp() const
{
    Quat<T> real = getRealPart();
    return createFromQuat(real.exp(), Quat<T>(detail::jacob_exp(real) * getDualPart().toVec()));
}

template <typename T>
DualQuat<T> log(const DualQuat<T> &dq, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT)
{
    return dq.log(assumeUnit);
}

template <typename T>
DualQuat<T> DualQuat<T>::log(QuatAssumeType assumeUnit) const
{
    Quat<T> plog = getRealPart().log(assumeUnit);
    Matx<T, 4, 4> jacob = detail::jacob_exp(plog);
    return createFromQuat(plog, Quat<T>(jacob.inv() * getDualPart().toVec()));
}

template <typename T>
inline DualQuat<T> power(const DualQuat<T> &dq, const T t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT)
{
    return dq.power(t, assumeUnit);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::power(const T t, QuatAssumeType assumeUnit) const
{
    return (t * log(assumeUnit)).exp();
}

template <typename T>
inline DualQuat<T> power(const DualQuat<T> &p, const DualQuat<T> &q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT)
{
    return p.power(q, assumeUnit);
}

template <typename T>
inline DualQuat<T> DualQuat<T>::power(const DualQuat<T> &q, QuatAssumeType assumeUnit) const
{
    return (q * log(assumeUnit)).exp();
}

template <typename T>
inline Vec<T, 8> DualQuat<T>::toVec() const
{
   return Vec<T, 8>(w, x, y, z, w_, x_, y_, z_);
}

template <typename T>
Affine3<T> DualQuat<T>::toAffine3(QuatAssumeType assumeUnit) const
{
    return Affine3<T>(toMat(assumeUnit));
}

template <typename T>
Matx<T, 4, 4> DualQuat<T>::toMat(QuatAssumeType assumeUnit) const
{
    Matx<T, 4, 4> rot44 = getRotation(assumeUnit).toRotMat4x4();
    Vec<T, 3> translation = getTranslation(assumeUnit);
    rot44(0, 3) = translation[0];
    rot44(1, 3) = translation[1];
    rot44(2, 3) = translation[2];
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
    Quat<T> v0Real = v0.getRealPart();
    Quat<T> v1Real = v1.getRealPart();
    if (directChange && v1Real.dot(v0Real) < 0)
    {
        v0 = -v0;
    }
    DualQuat<T> v0inv1 = v0.inv() * v1;
    return v0 * v0inv1.power(t, QUAT_ASSUME_UNIT);
}

template <typename T>
DualQuat<T> DualQuat<T>::dqblend(const DualQuat<T> &q1, const DualQuat<T> &q2, const T t, QuatAssumeType assumeUnit)
{
    DualQuat<T> v1(q1), v2(q2);
    if (!assumeUnit)
    {
        v1 = v1.normalize();
        v2 = v2.normalize();
    }
    if (v1.getRotation(assumeUnit).dot(v2.getRotation(assumeUnit)) < 0)
    {
        return ((1 - t) * v1 - t * v2).normalize();
    }
    return ((1 - t) * v1 + t * v2).normalize();
}

template <typename T>
DualQuat<T> DualQuat<T>::gdqblend(InputArray _dualquat, InputArray _weight, QuatAssumeType assumeUnit)
{
    CV_CheckTypeEQ(_weight.type(), cv::traits::Type<T>::value, "");
    CV_CheckTypeEQ(_dualquat.type(), CV_MAKETYPE(CV_MAT_DEPTH(cv::traits::Type<T>::value), 8), "");
    Size dq_s = _dualquat.size();
    if (dq_s != _weight.size() || (dq_s.height != 1 && dq_s.width != 1))
    {
        CV_Error(Error::StsBadArg, "The size of weight must be the same as dualquat, both of them should be (1, n) or (n, 1)");
    }
    Mat dualquat = _dualquat.getMat(), weight = _weight.getMat();
    const int cn = std::max(dq_s.width, dq_s.height);
    if (!assumeUnit)
    {
        for (int i = 0; i < cn; ++i)
        {
            dualquat.at<Vec<T, 8>>(i) = DualQuat<T>{dualquat.at<Vec<T, 8>>(i)}.normalize().toVec();
        }
    }
    Vec<T, 8> dq_blend = dualquat.at<Vec<T, 8>>(0) * weight.at<T>(0);
    Quat<T> q0 = DualQuat<T> {dualquat.at<Vec<T, 8>>(0)}.getRotation(assumeUnit);
    for (int i = 1; i < cn; ++i)
    {
        T k = q0.dot(DualQuat<T>{dualquat.at<Vec<T, 8>>(i)}.getRotation(assumeUnit)) < 0 ? -1: 1;
        dq_blend = dq_blend + dualquat.at<Vec<T, 8>>(i) * k * weight.at<T>(i);
    }
    return DualQuat<T>{dq_blend}.normalize();
}

template <typename T>
template <int cn>
DualQuat<T> DualQuat<T>::gdqblend(const Vec<DualQuat<T>, cn> &_dualquat, InputArray _weight, QuatAssumeType assumeUnit)
{
    Vec<DualQuat<T>, cn> dualquat(_dualquat);
    if (cn == 0)
    {
        return DualQuat<T>(1, 0, 0, 0, 0, 0, 0, 0);
    }
    Mat dualquat_mat(cn, 1, CV_64FC(8));
    for (int i = 0; i < cn ; ++i)
    {
        dualquat_mat.at<Vec<T, 8>>(i) = dualquat[i].toVec();
    }
    return gdqblend(dualquat_mat, _weight, assumeUnit);
}

} //namespace cv

#endif /*OPENCV_CORE_DUALQUATERNION_INL_HPP*/
