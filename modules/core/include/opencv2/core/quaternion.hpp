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
#ifndef OPENCV_CORE_QUATERNION_HPP
#define OPENCV_CORE_QUATERNION_HPP

#include <opencv2/core.hpp>
#include <iostream>
namespace cv
{
//! @addtogroup core
//! @{

//! Unit quaternion flag
enum QuatAssumeType
{
    /**
     * This flag is specified by default.
     * If this flag is specified, the input quaternions are assumed to be not unit quaternions.
     * It can guarantee the correctness of the calculations,
     * although the calculation speed will be slower than the flag QUAT_ASSUME_UNIT.
     */
    QUAT_ASSUME_NOT_UNIT,
    /**
     * If this flag is specified, the input quaternions are assumed to be unit quaternions which
     * will save some computations. However, if this flag is specified without unit quaternion,
     * the program correctness of the result will not be guaranteed.
     */
    QUAT_ASSUME_UNIT
};

template <typename _Tp> class Quat;
template <typename _Tp> std::ostream& operator<<(std::ostream&, const Quat<_Tp>&);

/**
 * Quaternion is a number system that extends the complex numbers. It can be expressed as a
 * rotation in three-dimensional space.
 * A quaternion is generally represented in the form:
 *      \f[q = w + x\boldsymbol{i} + y\boldsymbol{j} + z\boldsymbol{k}\f]
 *      \f[q = [w, x, y, z]\f]
 *      \f[q = [w, \boldsymbol{v}] \f]
 *      \f[q = ||q||[\cos\psi, u_x\sin\psi,u_y\sin\psi,  u_z\sin\psi].\f]
 *      \f[q = ||q||[\cos\psi, \boldsymbol{u}\sin\psi]\f]
 * where \f$\psi = \frac{\theta}{2}\f$, \f$\theta\f$ represents rotation angle,
 * \f$\boldsymbol{u} = [u_x, u_y, u_z]\f$ represents normalized rotation axis,
 * and \f$||q||\f$ represents the norm of \f$q\f$.
 *
 * A unit quaternion is usually represents rotation, which has the form:
 *      \f[q = [\cos\psi, u_x\sin\psi,u_y\sin\psi,  u_z\sin\psi].\f]
 *
 * To create a quaternion representing the rotation around the axis \f$\boldsymbol{u}\f$
 * with angle \f$\theta\f$, you can use
 * ```
 * using namespace cv;
 * double angle = CV_PI;
 * Vec3d axis = {0, 0, 1};
 * Quatd q = Quatd::createFromAngleAxis(angle, axis);
 * ```
 *
 * You can simply use four same type number to create a quaternion
 * ```
 * Quatd q(1, 2, 3, 4);
 * ```
 * Or use a Vec4d or Vec4f vector.
 * ```
 * Vec4d vec{1, 2, 3, 4};
 * Quatd q(vec);
 * ```
 *
 * ```
 * Vec4f vec{1, 2, 3, 4};
 * Quatf q(vec);
 * ```
 *
 * If you already have a 3x3 rotation matrix R, then you can use
 * ```
 * Quatd q = Quatd::createFromRotMat(R);
 * ```
 *
 * If you already have a rotation vector rvec which has the form of `angle * axis`, then you can use
 * ```
 * Quatd q = Quatd::createFromRvec(rvec);
 * ```
 *
 * To extract the rotation matrix from quaternion, see toRotMat3x3()
 *
 * To extract the Vec4d or Vec4f, see toVec()
 *
 * To extract the rotation vector, see toRotVec()
 *
 * If there are two quaternions \f$q_0, q_1\f$ are needed to interpolate, you can use nlerp(), slerp() or spline()
 * ```
 * Quatd::nlerp(q0, q1, t)
 *
 * Quatd::slerp(q0, q1, t)
 *
 * Quatd::spline(q0, q0, q1, q1, t)
 * ```
 * spline can smoothly connect rotations of  multiple quaternions
 *
 * Three ways to get an element in Quaternion
 * ```
 * Quatf q(1,2,3,4);
 * std::cout << q.w << std::endl; // w=1, x=2, y=3, z=4
 * std::cout << q[0] << std::endl; // q[0]=1, q[1]=2, q[2]=3, q[3]=4
 * std::cout << q.at(0) << std::endl;
 * ```
 */
template <typename _Tp>
class Quat
{
    static_assert(std::is_floating_point<_Tp>::value, "Quaternion only make sense with type of float or double");
    using value_type = _Tp;

public:
    static constexpr _Tp CV_QUAT_EPS = (_Tp)1.e-6;

    Quat();

    /**
     * @brief From Vec4d or Vec4f.
     */
    explicit Quat(const Vec<_Tp, 4> &coeff);

    /**
     * @brief from four numbers.
     */
    Quat(_Tp w, _Tp x, _Tp y, _Tp z);

    /**
     * @brief from an angle, axis. Axis will be normalized in this function. And
     * it generates
     * \f[q = [\cos\psi, u_x\sin\psi,u_y\sin\psi,  u_z\sin\psi].\f]
     * where \f$\psi = \frac{\theta}{2}\f$, \f$\theta\f$ is the rotation angle.
     */
    static Quat<_Tp> createFromAngleAxis(const _Tp angle, const Vec<_Tp, 3> &axis);

    /**
     * @brief from a 3x3 rotation matrix.
     */
    static Quat<_Tp> createFromRotMat(InputArray R);

    /**
     * @brief from a rotation vector
     * \f$r\f$ has the form \f$\theta \cdot \boldsymbol{u}\f$, where \f$\theta\f$
     * represents rotation angle and \f$\boldsymbol{u}\f$ represents normalized rotation axis.
     *
     * Angle and axis could be easily derived as:
     * \f[
     * \begin{equation}
     * \begin{split}
     * \psi &= ||r||\\
     * \boldsymbol{u} &= \frac{r}{\theta}
     * \end{split}
     * \end{equation}
     * \f]
     * Then a quaternion can be calculated by
     *  \f[q = [\cos\psi, \boldsymbol{u}\sin\psi]\f]
     *  where \f$\psi = \theta / 2 \f$
     */
    static Quat<_Tp> createFromRvec(InputArray rvec);

    /**
     * @brief a way to get element.
     * @param index over a range [0, 3].
     *
     * A quaternion q
     *
     * q.at(0) is equivalent to q.w,
     *
     * q.at(1) is equivalent to q.x,
     *
     * q.at(2) is equivalent to q.y,
     *
     * q.at(3) is equivalent to q.z.
     */
    _Tp at(size_t index) const;

    /**
     * @brief return the conjugate of this quaternion.
     * \f[q.conjugate() = (w, -x, -y, -z).\f]
     */
    Quat<_Tp> conjugate() const;

    /**
     *
     * @brief return the value of exponential value.
     * \f[\exp(q) = e^w (\cos||\boldsymbol{v}||+ \frac{v}{||\boldsymbol{v}||})\sin||\boldsymbol{v}||\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example:
     * ```
     * Quatd q{1,2,3,4};
     * cout << exp(q) << endl;
     * ```
     */
    template <typename T>
    friend Quat<T> exp(const Quat<T> &q);

    /**
     * @brief return the value of exponential value.
     * \f[\exp(q) = e^w (\cos||\boldsymbol{v}||+ \frac{v}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q{1,2,3,4};
     * cout << q.exp() << endl;
     * ```
     */
    Quat<_Tp> exp() const;

    /**
     * @brief return the value of logarithm function.
     * \f[\ln(q) = \ln||q|| + \frac{\boldsymbol{v}}{||\boldsymbol{v}||}\arccos\frac{w}{||q||}.\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     * @param assumeUnit if QUAT_ASSUME_UNIT, q assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q1{1,2,3,4};
     * cout << log(q1) << endl;
     * ```
     */
    template <typename T>
    friend Quat<T> log(const Quat<T> &q, QuatAssumeType assumeUnit);

    /**
     * @brief return the value of logarithm function.
     *  \f[\ln(q) = \ln||q|| + \frac{\boldsymbol{v}}{||\boldsymbol{v}||}\arccos\frac{w}{||q||}\f].
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.log();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * Quatd q1(1,2,3,4);
     * q1.normalize().log(assumeUnit);
     * ```
     */
    Quat<_Tp> log(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the value of power function with index \f$x\f$.
     * \f[q^x = ||q||(cos(x\theta) + \boldsymbol{u}sin(x\theta))).\f]
     * @param q a quaternion.
     * @param x index of exponentiation.
     * @param assumeUnit if QUAT_ASSUME_UNIT, quaternion q assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * power(q, 2.0);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * double angle = CV_PI;
     * Vec3d axis{0, 0, 1};
     * Quatd q1 = Quatd::createFromAngleAxis(angle, axis); //generate a unit quat by axis and angle
     * power(q1, 2.0, assumeUnit);//This assumeUnit means q1 is a unit quaternion.
     * ```
     * @note the type of the index should be the same as the quaternion.
     */
    template <typename T>
    friend Quat<T> power(const Quat<T> &q, const T x, QuatAssumeType assumeUnit);

    /**
     * @brief return the value of power function with index \f$x\f$.
     * \f[q^x = ||q||(\cos(x\theta) + \boldsymbol{u}\sin(x\theta))).\f]
     * @param x index of exponentiation.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.power(2.0);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * double angle = CV_PI;
     * Vec3d axis{0, 0, 1};
     * Quatd q1 = Quatd::createFromAngleAxis(angle, axis); //generate a unit quat by axis and angle
     * q1.power(2.0, assumeUnit); //This assumeUnt means q1 is a unit quaternion
     * ```
     */
    Quat<_Tp> power(const _Tp x, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return \f$\sqrt{q}\f$.
     * @param q a quaternion.
     * @param assumeUnit if QUAT_ASSUME_UNIT, quaternion q assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatf q(1,2,3,4);
     * sqrt(q);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q = {1,0,0,0};
     * sqrt(q, assumeUnit); //This assumeUnit means q is a unit quaternion.
     * ```
     */
    template <typename T>
    friend Quat<T> sqrt(const Quat<T> &q, QuatAssumeType assumeUnit);

    /**
     * @brief return \f$\sqrt{q}\f$.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatf q(1,2,3,4);
     * q.sqrt();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q = {1,0,0,0};
     * q.sqrt(assumeUnit); //This assumeUnit means q is a unit quaternion
     * ```
     */
    Quat<_Tp> sqrt(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the value of power function with quaternion \f$q\f$.
     * \f[p^q = e^{q\ln(p)}.\f]
     * @param p base quaternion of power function.
     * @param q index quaternion of power function.
     * @param assumeUnit if QUAT_ASSUME_UNIT, quaternion \f$p\f$ assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd p(1,2,3,4);
     * Quatd q(5,6,7,8);
     * power(p, q);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * p = p.normalize();
     * power(p, q, assumeUnit); //This assumeUnit means p is a unit quaternion
     * ```
     */
    template <typename T>
    friend Quat<T> power(const Quat<T> &p, const Quat<T> &q, QuatAssumeType assumeUnit);

    /**
     * @brief return the value of power function with quaternion \f$q\f$.
     * \f[p^q = e^{q\ln(p)}.\f]
     * @param q index quaternion of power function.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd p(1,2,3,4);
     * Quatd q(5,6,7,8);
     * p.power(q);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * p = p.normalize();
     * p.power(q, assumeUnit); //This assumeUnit means p is a unit quaternion
     * ```
     */
    Quat<_Tp> power(const Quat<_Tp> &q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the crossProduct between \f$p = (a, b, c, d) = (a, \boldsymbol{u})\f$ and \f$q = (w, x, y, z) = (w, \boldsymbol{v})\f$.
     * \f[p \times q = \frac{pq- qp}{2}\f]
     * \f[p \times q = \boldsymbol{u} \times \boldsymbol{v}\f]
     * \f[p \times q = (cz-dy)i + (dx-bz)j + (by-xc)k \f]
     *
     * For example
     * ```
     * Quatd q{1,2,3,4};
     * Quatd p{5,6,7,8};
     * crossProduct(p, q);
     * ```
     */
    template <typename T>
    friend Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

    /**
     * @brief return the crossProduct between \f$p = (a, b, c, d) = (a, \boldsymbol{u})\f$ and \f$q = (w, x, y, z) = (w, \boldsymbol{v})\f$.
     * \f[p \times q = \frac{pq- qp}{2}.\f]
     * \f[p \times q = \boldsymbol{u} \times \boldsymbol{v}.\f]
     * \f[p \times q = (cz-dy)i + (dx-bz)j + (by-xc)k. \f]
     *
     * For example
     * ```
     * Quatd q{1,2,3,4};
     * Quatd p{5,6,7,8};
     * p.crossProduct(q)
     * ```
     */
    Quat<_Tp> crossProduct(const Quat<_Tp> &q) const;

    /**
     * @brief return the norm of quaternion.
     * \f[||q|| = \sqrt{w^2 + x^2 + y^2 + z^2}.\f]
     */
    _Tp norm() const;

    /**
     * @brief return a normalized \f$p\f$.
     * \f[p = \frac{q}{||q||}\f]
     * where \f$p\f$ satisfies \f$(p.x)^2 + (p.y)^2 + (p.z)^2 + (p.w)^2 = 1.\f$
     */
    Quat<_Tp> normalize() const;

    /**
     * @brief return \f$q^{-1}\f$ which is an inverse of \f$q\f$
     * which satisfies \f$q * q^{-1} = 1\f$.
     * @param q a quaternion.
     * @param assumeUnit if QUAT_ASSUME_UNIT, quaternion q assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * inv(q);
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q = q.normalize();
     * inv(q, assumeUnit);//This assumeUnit means p is a unit quaternion
     * ```
     */
    template <typename T>
    friend Quat<T> inv(const Quat<T> &q, QuatAssumeType assumeUnit);

    /**
     * @brief return \f$q^{-1}\f$ which is an inverse of \f$q\f$
     * satisfying \f$q * q^{-1} = 1\f$.
     * @param assumeUnit if QUAT_ASSUME_UNIT, quaternion q assume to be a unit quaternion and this function will save some computations.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.inv();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q = q.normalize();
     * q.inv(assumeUnit);  //assumeUnit means p is a unit quaternion
     * ```
     */
    Quat<_Tp> inv(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return sinh value of quaternion q, sinh could be calculated as:
     * \f[\sinh(p) = \sin(w)\cos(||\boldsymbol{v}||) + \cosh(w)\frac{v}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * sinh(q);
     * ```
     */
    template <typename T>
    friend Quat<T> sinh(const Quat<T> &q);

    /**
     * @brief return sinh value of this quaternion, sinh could be calculated as:
     * \f$\sinh(p) = \sin(w)\cos(||\boldsymbol{v}||) + \cosh(w)\frac{v}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||\f$
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.sinh();
     * ```
     */
    Quat<_Tp> sinh() const;

    /**
     * @brief return cosh value of quaternion q, cosh could be calculated as:
     * \f[\cosh(p) = \cosh(w) * \cos(||\boldsymbol{v}||) + \sinh(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sin(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * cosh(q);
     * ```
     */
    template <typename T>
    friend Quat<T> cosh(const Quat<T> &q);

    /**
     * @brief return cosh value of this quaternion, cosh could be calculated as:
     * \f[\cosh(p) = \cosh(w) * \cos(||\boldsymbol{v}||) + \sinh(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}sin(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.cosh();
     * ```
     */
    Quat<_Tp> cosh() const;

    /**
     * @brief return tanh value of quaternion q, tanh could be calculated as:
     * \f[ \tanh(q) = \frac{\sinh(q)}{\cosh(q)}.\f]
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * tanh(q);
     * ```
     * @sa sinh, cosh
     */
    template <typename T>
    friend Quat<T> tanh(const Quat<T> &q);

    /**
     * @brief return tanh value of this quaternion, tanh could be calculated as:
     * \f[ \tanh(q) = \frac{\sinh(q)}{\cosh(q)}.\f]
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.tanh();
     * ```
     * @sa sinh, cosh
     */
    Quat<_Tp> tanh() const;

    /**
     * @brief return tanh value of quaternion q, sin could be calculated as:
     * \f[\sin(p) = \sin(w) * \cosh(||\boldsymbol{v}||) + \cos(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sinh(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * sin(q);
     * ```
     */
    template <typename T>
    friend Quat<T> sin(const Quat<T> &q);

    /**
     * @brief return sin value of this quaternion, sin could be calculated as:
     * \f[\sin(p) = \sin(w) * \cosh(||\boldsymbol{v}||) + \cos(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sinh(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.sin();
     * ```
     */
    Quat<_Tp> sin() const;

    /**
     * @brief return sin value of quaternion q, cos could be calculated as:
     * \f[\cos(p) = \cos(w) * \cosh(||\boldsymbol{v}||) - \sin(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sinh(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * cos(q);
     * ```
     */
    template <typename T>
    friend Quat<T> cos(const Quat<T> &q);

    /**
     * @brief return cos value of this quaternion, cos could be calculated as:
     * \f[\cos(p) = \cos(w) * \cosh(||\boldsymbol{v}||) - \sin(w)\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sinh(||\boldsymbol{v}||)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.cos();
     * ```
     */
    Quat<_Tp> cos() const;

    /**
     * @brief return tan value of quaternion q, tan could be calculated as:
     * \f[\tan(q) = \frac{\sin(q)}{\cos(q)}.\f]
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * tan(q);
     * ```
     */
    template <typename T>
    friend Quat<T> tan(const Quat<T> &q);

    /**
     * @brief return tan value of this quaternion, tan could be calculated as:
     * \f[\tan(q) = \frac{\sin(q)}{\cos(q)}.\f]
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.tan();
     * ```
     */
    Quat<_Tp> tan() const;

    /**
     * @brief return arcsin value of quaternion q, arcsin could be calculated as:
     * \f[\arcsin(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arcsinh(q\frac{\boldsymbol{v}}{||\boldsymbol{v}||})\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * asin(q);
     * ```
     */
    template <typename T>
    friend Quat<T> asin(const Quat<T> &q);

    /**
     * @brief return arcsin value of this quaternion, arcsin could be calculated as:
     * \f[\arcsin(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arcsinh(q\frac{\boldsymbol{v}}{||\boldsymbol{v}||})\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.asin();
     * ```
     */
    Quat<_Tp> asin() const;

    /**
     * @brief return arccos value of quaternion q, arccos could be calculated as:
     * \f[\arccos(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arccosh(q)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * acos(q);
     * ```
     */
    template <typename T>
    friend Quat<T> acos(const Quat<T> &q);

    /**
     * @brief return arccos value of this quaternion, arccos could be calculated as:
     * \f[\arccos(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arccosh(q)\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.acos();
     * ```
     */
    Quat<_Tp> acos() const;

    /**
     * @brief return arctan value of quaternion q, arctan could be calculated as:
     * \f[\arctan(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arctanh(q\frac{\boldsymbol{v}}{||\boldsymbol{v}||})\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * atan(q);
     * ```
     */
    template <typename T>
    friend Quat<T> atan(const Quat<T> &q);

    /**
     * @brief return arctan value of this quaternion, arctan could be calculated as:
     * \f[\arctan(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arctanh(q\frac{\boldsymbol{v}}{||\boldsymbol{v}||})\f]
     * where \f$\boldsymbol{v} = [x, y, z].\f$
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.atan();
     * ```
     */
    Quat<_Tp> atan() const;

    /**
     * @brief return arcsinh value of quaternion q, arcsinh could be calculated as:
     * \f[arcsinh(q) = \ln(q + \sqrt{q^2 + 1})\f].
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * asinh(q);
     * ```
     */
    template <typename T>
    friend Quat<T> asinh(const Quat<T> &q);

    /**
     * @brief return arcsinh value of this quaternion, arcsinh could be calculated as:
     * \f[arcsinh(q) = \ln(q + \sqrt{q^2 + 1})\f].
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.asinh();
     * ```
     */
    Quat<_Tp> asinh() const;

    /**
     * @brief return arccosh value of quaternion q, arccosh could be calculated as:
     * \f[arccosh(q) = \ln(q + \sqrt{q^2 - 1})\f].
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * acosh(q);
     * ```
     */
    template <typename T>
    friend Quat<T> acosh(const Quat<T> &q);

    /**
     * @brief return arccosh value of this quaternion, arccosh could be calculated as:
     * \f[arcosh(q) = \ln(q + \sqrt{q^2 - 1})\f].
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.acosh();
     * ```
     */
    Quat<_Tp> acosh() const;

    /**
     * @brief return arctanh value of quaternion q, arctanh could be calculated as:
     * \f[arctanh(q) = \frac{\ln(q + 1) - \ln(1 - q)}{2}\f].
     * @param q a quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * atanh(q);
     * ```
     */
    template <typename T>
    friend Quat<T> atanh(const Quat<T> &q);

    /**
     * @brief return arctanh value of this quaternion, arctanh could be calculated as:
     * \f[arcsinh(q) = \frac{\ln(q + 1) - \ln(1 - q)}{2}\f].
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.atanh();
     * ```
     */
    Quat<_Tp> atanh() const;

    /**
     * @brief return true if this quaternion is a unit quaternion.
     * @param eps tolerance scope of normalization. The eps could be defined as
     *
     * \f[eps = |1 - dotValue|\f] where \f[dotValue = (this.w^2 + this.x^2 + this,y^2 + this.z^2).\f]
     * And this function will consider it is normalized when the dotValue over a range \f$[1-eps, 1+eps]\f$.
     */
    bool isNormal(_Tp eps=CV_QUAT_EPS) const;

    /**
     * @brief to throw an error if this quaternion is not a unit quaternion.
     * @param eps tolerance scope of normalization.
     * @sa isNormal
     */
    void assertNormal(_Tp eps=CV_QUAT_EPS) const;

    /**
     * @brief transform a quaternion to a 3x3 rotation matrix.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and
     * this function will save some computations. Otherwise, this function will normalized this
     * quaternion at first then to do the transformation.
     *
     * @note Matrix A which is to be rotated should have the form
     * \f[\begin{bmatrix}
     * x_0& x_1& x_2&...&x_n\\
     * y_0& y_1& y_2&...&y_n\\
     * z_0& z_1& z_2&...&z_n
     * \end{bmatrix}\f]
     * where the same subscript represents a point. The shape of A assume to be [3, n]
     * The points matrix A can be rotated by toRotMat3x3() * A.
     * The result has 3 rows and n columns too.

     * For example
     * ```
     * double angle = CV_PI;
     * Vec3d axis{0,0,1};
     * Quatd q_unit = Quatd::createFromAngleAxis(angle, axis); //quaternion could also be get by interpolation by two or more quaternions.
     *
     * //assume there is two points (1,0,0) and (1,0,1) to be rotated
     * Mat pointsA = (Mat_<double>(2, 3) << 1,0,0,1,0,1);
     * //change the shape
     * pointsA = pointsA.t();
     * // rotate 180 degrees around the z axis
     * Mat new_point = q_unit.toRotMat3x3() * pointsA;
     * // print two points
     * cout << new_point << endl;
     * ```
     */
    Matx<_Tp, 3, 3> toRotMat3x3(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief transform a quaternion to a 4x4 rotation matrix.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and
     * this function will save some computations. Otherwise, this function will normalized this
     * quaternion at first then to do the transformation.
     *
     * The operations is similar as toRotMat3x3
     * except that the points matrix should have the form
     * \f[\begin{bmatrix}
     * x_0& x_1& x_2&...&x_n\\
     * y_0& y_1& y_2&...&y_n\\
     * z_0& z_1& z_2&...&z_n\\
     * 0&0&0&...&0
     * \end{bmatrix}\f]
     *
     * @sa toRotMat3x3
     */

    Matx<_Tp, 4, 4> toRotMat4x4(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief transform the this quaternion to a Vec<T, 4>.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.toVec();
     * ```
     */
    Vec<_Tp, 4> toVec() const;

    /**
     * @brief transform this quaternion to a Rotation vector.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and
     * this function will save some computations.
     * Rotation vector rVec is defined as:
     * \f[ rVec = [\theta v_x, \theta v_y, \theta v_z]\f]
     * where \f$\theta\f$ represents rotation angle, and \f$\boldsymbol{v}\f$ represents the normalized rotation axis.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.toRotVec();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q.normalize().toRotVec(assumeUnit); //answer is same as q.toRotVec().
     * ```
     */
    Vec<_Tp, 3> toRotVec(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief get the angle of quaternion, it returns the rotation angle.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and
     * this function will save some computations.
     * \f[\psi = 2 *arccos(\frac{w}{||q||})\f]
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.getAngle();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q.normalize().getAngle(assumeUnit);//same as q.getAngle().
     * ```
     * @note It always return the value between \f$[0, 2\pi]\f$.
     */
    _Tp getAngle(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief get the axis of quaternion, it returns a vector of length 3.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this quaternion assume to be a unit quaternion and
     * this function will save some computations.
     *
     * the unit axis \f$\boldsymbol{u}\f$ is defined by
     * \f[\begin{equation}
     *    \begin{split}
     *      \boldsymbol{v}
     *      &= \boldsymbol{u} ||\boldsymbol{v}||\\
     *      &= \boldsymbol{u}||q||sin(\frac{\theta}{2})
     *    \end{split}
     *    \end{equation}\f]
     *  where \f$v=[x, y ,z]\f$ and \f$\theta\f$ represents rotation angle.
     *
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * q.getAxis();
     *
     * QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;
     * q.normalize().getAxis(assumeUnit);//same as q.getAxis()
     * ```
     */
    Vec<_Tp, 3> getAxis(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the dot between quaternion \f$q\f$ and this quaternion.
     *
     * dot(p, q) is a good metric of how close the quaternions are.
     * Indeed, consider the unit quaternion difference \f$p^{-1} * q\f$, its real part is dot(p, q).
     * At the same time its real part is equal to \f$\cos(\beta/2)\f$ where \f$\beta\f$ is
     * an angle of rotation between p and q, i.e.,
     * Therefore, the closer dot(p, q) to 1,
     * the smaller rotation between them.
     * \f[p \cdot q = p.w \cdot q.w + p.x \cdot q.x + p.y \cdot q.y + p.z \cdot q.z\f]
     * @param q the other quaternion.
     *
     * For example
     * ```
     * Quatd q(1,2,3,4);
     * Quatd p(5,6,7,8);
     * p.dot(q);
     * ```
     */
    _Tp dot(Quat<_Tp> q) const;

    /**
     * @brief To calculate the interpolation from \f$q_0\f$ to \f$q_1\f$ by Linear Interpolation(Nlerp)
     * For two quaternions, this interpolation curve can be displayed as:
     * \f[Lerp(q_0, q_1, t) = (1 - t)q_0 + tq_1.\f]
     * Obviously, the lerp will interpolate along a straight line if we think of \f$q_0\f$ and \f$q_1\f$ as a vector
     * in a two-dimensional space. When \f$t = 0\f$, it returns \f$q_0\f$ and when \f$t= 1\f$, it returns \f$q_1\f$.
     * \f$t\f$ should to be ranged in \f$[0, 1]\f$ normally.
     * @param q0 a quaternion used in linear interpolation.
     * @param q1 a quaternion used in linear interpolation.
     * @param t percent of vector \f$\overrightarrow{q_0q_1}\f$ over a range [0, 1].
     * @note it returns a non-unit quaternion.
     */
    static Quat<_Tp> lerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t);

    /**
     * @brief To calculate the interpolation from \f$q_0\f$ to \f$q_1\f$ by Normalized Linear Interpolation(Nlerp).
     * it returns a normalized quaternion of Linear Interpolation(Lerp).
     * \f[ Nlerp(q_0, q_1, t) = \frac{(1 - t)q_0 + tq_1}{||(1 - t)q_0 + tq_1||}.\f]
     * The interpolation will always choose the shortest path but the constant speed is not guaranteed.
     * @param q0 a quaternion used in normalized linear interpolation.
     * @param q1 a quaternion used in normalized linear interpolation.
     * @param t percent of vector \f$\overrightarrow{q_0q_1}\f$ over a range [0, 1].
     * @param assumeUnit if QUAT_ASSUME_UNIT, all input quaternions assume to be unit quaternion. Otherwise, all inputs
     quaternion will be normalized inside the function.
     * @sa lerp
     */
    static Quat<_Tp> nlerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

    /**
     @brief To calculate the interpolation between \f$q_0\f$ and \f$q_1\f$ by Spherical Linear
     Interpolation(Slerp), which can be defined as:
    \f[ Slerp(q_0, q_1, t) = \frac{\sin((1-t)\theta)}{\sin(\theta)}q_0 + \frac{\sin(t\theta)}{\sin(\theta)}q_1\f]
    where \f$\theta\f$ can be calculated as:
    \f[\theta=cos^{-1}(q_0\cdot q_1)\f]
    resulting from the both of their norm is unit.
    @param q0 a quaternion used in Slerp.
    @param q1 a quaternion used in Slerp.
    @param t percent of angle between \f$q_0\f$ and \f$q_1\f$ over a range [0, 1].
    @param assumeUnit if QUAT_ASSUME_UNIT, all input quaternions assume to be unit quaternions. Otherwise, all input
    quaternions will be normalized inside the function.
    @param directChange if QUAT_ASSUME_UNIT, the interpolation will choose the nearest path.
    @note If the interpolation angle is small, the error between Nlerp and Slerp is not so large. To improve efficiency and
    avoid zero division error, we use Nlerp instead of Slerp.
    */
    static Quat<_Tp> slerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT, bool directChange=true);

    /**
     * @brief To calculate the interpolation between \f$q_0\f$,\f$q_1\f$,\f$q_2\f$,\f$q_3\f$  by Spherical and quadrangle(Squad). This could be defined as:
     * \f[Squad(q_i, s_i, s_{i+1}, q_{i+1}, t) = Slerp(Slerp(q_i, q_{i+1}, t), Slerp(s_i, s_{i+1}, t), 2t(1-t))\f]
     * where
     * \f[s_i = q_i\exp(-\frac{\log(q^*_iq_{i+1}) + \log(q^*_iq_{i-1})}{4})\f]
     *
     * The Squad expression is analogous to the \f$B\acute{e}zier\f$ curve, but involves spherical linear
     * interpolation instead of simple linear interpolation. Each \f$s_i\f$ needs to be calculated by three
     * quaternions.
     *
     * @param q0 the first quaternion.
     * @param s0 the second quaternion.
     * @param s1 the third quaternion.
     * @param q1 thr fourth quaternion.
     * @param t interpolation parameter of quadratic and linear interpolation over a range \f$[0, 1]\f$.
     * @param assumeUnit if QUAT_ASSUME_UNIT, all input quaternions assume to be unit quaternion. Otherwise, all input
     * quaternions will be normalized inside the function.
     * @param directChange if QUAT_ASSUME_UNIT, squad will find the nearest path to interpolate.
     * @sa interPoint, spline
     */
    static Quat<_Tp> squad(const Quat<_Tp> &q0, const Quat<_Tp> &s0,
                            const Quat<_Tp> &s1, const Quat<_Tp> &q1,
                            const _Tp t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT,
                            bool directChange=true);

    /**
     * @brief This is the part calculation of squad.
     * To calculate the intermedia quaternion \f$s_i\f$ between each three quaternion
     * \f[s_i = q_i\exp(-\frac{\log(q^*_iq_{i+1}) + \log(q^*_iq_{i-1})}{4}).\f]
     * @param q0 the first quaternion.
     * @param q1 the second quaternion.
     * @param q2 the third quaternion.
     * @param assumeUnit if QUAT_ASSUME_UNIT, all input quaternions assume to be unit quaternion. Otherwise, all input
     * quaternions will be normalized inside the function.
     * @sa squad
     */
    static Quat<_Tp> interPoint(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                                 const Quat<_Tp> &q2, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

    /**
     * @brief to calculate a quaternion which is the result of a \f$C^1\f$ continuous
     * spline curve constructed by squad at the ratio t. Here, the interpolation values are
     * between \f$q_1\f$ and \f$q_2\f$. \f$q_0\f$ and \f$q_2\f$ are used to ensure the \f$C^1\f$
     * continuity. if t = 0, it returns \f$q_1\f$, if t = 1, it returns \f$q_2\f$.
     * @param q0 the first input quaternion to ensure \f$C^1\f$ continuity.
     * @param q1 the second input quaternion.
     * @param q2 the third input quaternion.
     * @param q3 the fourth input quaternion the same use of \f$q1\f$.
     * @param t ratio over a range [0, 1].
     * @param assumeUnit if QUAT_ASSUME_UNIT, \f$q_0, q_1, q_2, q_3\f$ assume to be unit quaternion. Otherwise, all input
     * quaternions will be normalized inside the function.
     *
     * For example:
     *
     * If there are three double quaternions \f$v_0, v_1, v_2\f$ waiting to be interpolated.
     *
     * Interpolation between \f$v_0\f$ and \f$v_1\f$ with a ratio \f$t_0\f$ could be calculated as
     * ```
     * Quatd::spline(v0, v0, v1, v2, t0);
     * ```
     * Interpolation between \f$v_1\f$ and \f$v_2\f$ with a ratio \f$t_0\f$ could be calculated as
     * ```
     * Quatd::spline(v0, v1, v2, v2, t0);
     * ```
     * @sa squad, slerp
     */
    static Quat<_Tp> spline(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                            const Quat<_Tp> &q2, const Quat<_Tp> &q3,
                            const _Tp t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

    /**
     * @brief Return opposite quaternion \f$-p\f$
     * which satisfies \f$p + (-p) = 0.\f$
     *
     * For example
     * ```
     * Quatd q{1, 2, 3, 4};
     * std::cout << -q << std::endl; // [-1, -2, -3, -4]
     * ```
     */
    Quat<_Tp> operator-() const;

    /**
     * @brief return true if two quaternions p and q are nearly equal, i.e. when the absolute
     * value of each \f$p_i\f$ and \f$q_i\f$ is less than CV_QUAT_EPS.
     */
    bool operator==(const Quat<_Tp>&) const;

    /**
     * @brief Addition operator of two quaternions p and q.
     * It returns a new quaternion that each value is the sum of \f$p_i\f$ and \f$q_i\f$.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * std::cout << p + q << std::endl; //[6, 8, 10, 12]
     * ```
     */
    Quat<_Tp> operator+(const Quat<_Tp>&) const;

    /**
     * @brief Addition assignment operator of two quaternions p and q.
     * It adds right operand to the left operand and assign the result to left operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * p += q; // equivalent to p = p + q
     * std::cout << p << std::endl; //[6, 8, 10, 12]
     *
     * ```
     */
    Quat<_Tp>& operator+=(const Quat<_Tp>&);

    /**
     * @brief Subtraction operator of two quaternions p and q.
     * It returns a new quaternion that each value is the sum of \f$p_i\f$ and \f$-q_i\f$.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * std::cout << p - q << std::endl; //[-4, -4, -4, -4]
     * ```
     */
    Quat<_Tp> operator-(const Quat<_Tp>&) const;

    /**
     * @brief Subtraction assignment operator of two quaternions p and q.
     * It subtracts right operand from the left operand and assign the result to left operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * p -= q; // equivalent to p = p - q
     * std::cout << p << std::endl; //[-4, -4, -4, -4]
     *
     * ```
     */
    Quat<_Tp>& operator-=(const Quat<_Tp>&);

    /**
     * @brief Multiplication assignment operator of two quaternions q and p.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of quaternion multiplication:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * q &= [p_0, \boldsymbol{u}]*[q_0, \boldsymbol{v}]\\
     * &=[p_0q_0 - \boldsymbol{u}\cdot \boldsymbol{v}, p_0\boldsymbol{v} + q_0\boldsymbol{u}+ \boldsymbol{u}\times \boldsymbol{v}].
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$\cdot\f$ means dot product and \f$\times \f$ means cross product.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * p *= q; // equivalent to p = p * q
     * std::cout << p << std::endl; //[-60, 12, 30, 24]
     * ```
     */
    Quat<_Tp>& operator*=(const Quat<_Tp>&);

    /**
     * @brief Multiplication assignment operator of a quaternions and a scalar.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z] * s\\
     * &=[w * s, x * s, y * s, z * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double s = 2.0;
     * p *= s; // equivalent to p = p * s
     * std::cout << p << std::endl; //[2.0, 4.0, 6.0, 8.0]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    Quat<_Tp>& operator*=(const _Tp s);

    /**
     * @brief Multiplication operator of two quaternions q and p.
     * Multiplies values on either side of the operator.
     *
     * Rule of quaternion multiplication:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * q &= [p_0, \boldsymbol{u}]*[q_0, \boldsymbol{v}]\\
     * &=[p_0q_0 - \boldsymbol{u}\cdot \boldsymbol{v}, p_0\boldsymbol{v} + q_0\boldsymbol{u}+ \boldsymbol{u}\times \boldsymbol{v}].
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$\cdot\f$ means dot product and \f$\times \f$ means cross product.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * std::cout << p * q << std::endl; //[-60, 12, 30, 24]
     * ```
     */
    Quat<_Tp> operator*(const Quat<_Tp>&) const;

    /**
     * @brief Division operator of a quaternions and a scalar.
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of quaternion division with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / s &= [w, x, y, z] / s\\
     * &=[w/s, x/s, y/s, z/s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double s = 2.0;
     * p /= s; // equivalent to p = p / s
     * std::cout << p << std::endl; //[0.5, 1, 1.5, 2]
     * ```
     * @note the type of scalar should be equal to this quaternion.
     */
    Quat<_Tp> operator/(const _Tp s) const;

    /**
     * @brief Division operator of two quaternions p and q.
     * Divides left hand operand by right hand operand.
     *
     * Rule of quaternion division with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / q &= p * q.inv()\\
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * std::cout << p / q << std::endl; // equivalent to p * q.inv()
     * ```
     */
    Quat<_Tp> operator/(const Quat<_Tp>&) const;

    /**
     * @brief Division assignment operator of a quaternions and a scalar.
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of quaternion division with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / s &= [w, x, y, z] / s\\
     * &=[w / s, x / s, y / s, z / s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double s = 2.0;;
     * p /= s; // equivalent to p = p / s
     * std::cout << p << std::endl; //[0.5, 1.0, 1.5, 2.0]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    Quat<_Tp>& operator/=(const _Tp s);

    /**
     * @brief Division assignment operator of two quaternions p and q;
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of quaternion division with a quaternion:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / q&= p * q.inv()\\
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * Quatd q{5, 6, 7, 8};
     * p /= q; // equivalent to p = p * q.inv()
     * std::cout << p << std::endl;
     * ```
     */
    Quat<_Tp>& operator/=(const Quat<_Tp>&);

    _Tp& operator[](std::size_t n);

    const _Tp& operator[](std::size_t n) const;

    /**
     * @brief Subtraction operator of a scalar and a quaternions.
     * Subtracts right hand operand from left hand operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double scalar = 2.0;
     * std::cout << scalar - p << std::endl; //[1.0, -2, -3, -4]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator-(const T s, const Quat<T>&);

    /**
     * @brief Subtraction operator of a quaternions and a scalar.
     * Subtracts right hand operand from left hand operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double scalar = 2.0;
     * std::cout << p - scalar << std::endl; //[-1.0, 2, 3, 4]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator-(const Quat<T>&, const T s);

    /**
     * @brief Addition operator of a quaternions and a scalar.
     * Adds right hand operand from left hand operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double scalar = 2.0;
     * std::cout << scalar + p << std::endl; //[3.0, 2, 3, 4]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator+(const T s, const Quat<T>&);

    /**
     * @brief Addition operator of a quaternions and a scalar.
     * Adds right hand operand from left hand operand.
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double scalar = 2.0;
     * std::cout << p + scalar << std::endl; //[3.0, 2, 3, 4]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator+(const Quat<T>&, const T s);

    /**
     * @brief Multiplication operator of a scalar and a quaternions.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z] * s\\
     * &=[w * s, x * s, y * s, z * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double s = 2.0;
     * std::cout << s * p << std::endl; //[2.0, 4.0, 6.0, 8.0]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator*(const T s, const Quat<T>&);

    /**
     * @brief Multiplication operator of a quaternion and a scalar.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z] * s\\
     * &=[w * s, x * s, y * s, z * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * Quatd p{1, 2, 3, 4};
     * double s = 2.0;
     * std::cout << p * s << std::endl; //[2.0, 4.0, 6.0, 8.0]
     * ```
     * @note the type of scalar should be equal to the quaternion.
     */
    template <typename T>
    friend Quat<T> cv::operator*(const Quat<T>&, const T s);

    template <typename S>
    friend std::ostream& cv::operator<<(std::ostream&, const Quat<S>&);

    _Tp w, x, y, z;

};

template <typename T>
Quat<T> inv(const Quat<T> &q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

template <typename T>
Quat<T> sinh(const Quat<T> &q);

template <typename T>
Quat<T> cosh(const Quat<T> &q);

template <typename T>
Quat<T> tanh(const Quat<T> &q);

template <typename T>
Quat<T> sin(const Quat<T> &q);

template <typename T>
Quat<T> cos(const Quat<T> &q);

template <typename T>
Quat<T> tan(const Quat<T> &q);

template <typename T>
Quat<T> asinh(const Quat<T> &q);

template <typename T>
Quat<T> acosh(const Quat<T> &q);

template <typename T>
Quat<T> atanh(const Quat<T> &q);

template <typename T>
Quat<T> asin(const Quat<T> &q);

template <typename T>
Quat<T> acos(const Quat<T> &q);

template <typename T>
Quat<T> atan(const Quat<T> &q);

template <typename T>
Quat<T> power(const Quat<T> &q, const Quat<T> &p, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

template <typename T>
Quat<T> exp(const Quat<T> &q);

template <typename T>
Quat<T> log(const Quat<T> &q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

template <typename T>
Quat<T> power(const Quat<T>& q, const T x, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

template <typename T>
Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

template <typename S>
Quat<S> sqrt(const Quat<S> &q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

template <typename T>
Quat<T> operator*(const T, const Quat<T>&);

template <typename T>
Quat<T> operator*(const Quat<T>&, const T);

template <typename S>
std::ostream& operator<<(std::ostream&, const Quat<S>&);

using Quatd = Quat<double>;
using Quatf = Quat<float>;

//! @} core
}

#include "opencv2/core/quaternion.inl.hpp"

#endif /* OPENCV_CORE_QUATERNION_HPP */
