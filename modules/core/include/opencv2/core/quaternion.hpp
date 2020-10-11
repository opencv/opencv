// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2020, Huawei Technologies Co., Ltd. All rights reserved.
// Third party copyrights are property of their repsective owners.
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
    static constexpr double CV_QUAT_EPS = 1.e-6;
    /**
     * Quaternion is a number system that extends the complex numbers. It can be exprssed as a
     * rotation in three-dimensional space.
     * A quaternion is generally represented in the form:
     *      \f[q = w + x\boldsymbol{i} + y\boldsymbol{j} + z\boldsymbol{k}\f]
     *      \f[q = [w, x, y, z]\f]
     *      \f[q = [w, \boldsymbol{v}] \f]
     *      \f[q = ||q||[\cos\theta, u_x\sin\theta,u_y\sin\theta,  u_z\sin\theta].\f]
     *      \f[q = ||q||[\cos\theta, \boldsymbol{u}\sin\theta]\f]
     * where \f$\theta = \frac{\psi}{2}\f$, \f$\psi\f$ represents rotate angle,
     * \f$\boldsymbol{u} = [u_x, u_y, u_z]\f$ represents normalized rotate axis,
     * and \f$||q||\f$ represents the norm of \f$q\f$
     *
     * A unit quaternion is usually represents rotation, which has the form:
     *      \f[q = [\cos\theta, u_x\sin\theta,u_y\sin\theta,  u_z\sin\theta].\f]
     *
     * To create a quaternion representing the rotation around the axis \f$\boldsymbol{v}\f$
     * with angle \f$\psi\f$, you can use
     * ```
     * double angle = CV_PI;
     * cv::Vec3d{0, 0, 1};
     * cv::Quatd q(angle, axis);
     * ```
     *
     * You can simply use four same type number to create a quaternion
     * ```
     * cv::Quatd q(1, 2, 3, 4);
     * ```
     * Or a Vec4d or Vec4f vec, then you can use
     * ```
     * cv::Quatd q(vec);
     * ```
     *
     * If you already have a 3x3 rotate matrix R, then you can use
     * ```
     * cv::Quatd q(R);
     * ```
     *
     * If you already have a Rodrigues vector rvec, then you can use
     * ```
     * cv::Quatd q(rvec);
     * ```
     *
     * To extrace the rotate matrix from quaternion, see toRotMat3x3()
     *
     * To extrace the Vec4d or Vec4f, see toVec()
     *
     * To extrace the Rodrigues vector, see toRodrigues()
     *
     * If there are two quaternions \f$q_0, q_1\f$ interpolation is needed, you can use nlerp(), slerp() or spline()
     * ```
     * cv::nlerp(q0, q1, t)
     *
     * cv::slerp(q0, q1, t)
     *
     * cv::spline(q0, q0, q1, q1, t)
     * ```
     * spline can show the rotation smoothness between multiple quaternions
     *
     *
     * Three ways to get a element in Quaternion
     * ```
     * Quatf q(1,2,3,4);
     * std::cout << q.w << std::endl; // w=1,x=2,y=3,z=4
     * std::cout << q[0] << std::endl; // q[0]=1, q[1]=2,q[2]=3,q[3]=4
     * std::cout << q.at(0) << std::endl;
     * ```
     */
    template <typename _Tp> class Quat;
    template <typename S>
    std::ostream& operator<<(std::ostream&, const Quat<S>&);

    template <typename _Tp>
    class Quat
    {
    using value_type = typename std::enable_if<
        std::is_same<float, _Tp>::value||
        std::is_same<double, _Tp>::value,
        _Tp
        >::type;
    public:
        Quat() = default;

        /**
         * @brief From Vec4d or Vec4f
         */
        explicit Quat(const cv::Vec<_Tp, 4> &coeff);

        /**
         * @brief from four numbers
         */
        Quat(_Tp w, _Tp x, _Tp y, _Tp z);

        /**
         * @brief from a angle, axis and qNorm with default 1, axis will be normalized in this constrctor. And
         * it generates
         * \f[q = [\cos\theta, u_x\sin\theta,u_y\sin\theta,  u_z\sin\theta].\f]
         */
        Quat(const _Tp angle, const cv::Vec<_Tp, 3> &axis);

        /**
         * @brief from a 3x3 rotate matrix
         */
        explicit Quat(const cv::Mat &R);

        /**
         * @brief from Rodrigues vector
         */
        explicit Quat(const Vec<_Tp, 3> &rvec);

        /**
         * @brief a way to get element
         * @param index over a range [0, 3].
         *
         * Assume a quaternion q,
         *
         * q.at(0) is equicalent to q.w,
         *
         * q.at(1) is equicalent to q.x,
         *
         * q.at(2) is equicalent to q.y,
         *
         * q.at(3) is equicalent to q.z,
         */
        _Tp at(size_t index) const;

        /**
         * @brief return the conjugate of this quaternion
         * \f[q.conjugate() = (w, -x, -y, -z)\f]
         */
        Quat<_Tp> conjugate() const;

        /**
         * @brief return the value of exponential value
         * \f[\exp(q) = e^w (\cos||\boldsymbol{v}||+ \frac{v}{||\boldsymbol{v}||})\sin||\boldsymbol{v}||\f]
         * @param q a quaternion
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
         * @brief return the value of exponential value
         * \f[\exp(q) = e^w (\cos||\boldsymbol{v}||+ \frac{v}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||)\f]
         *
         * For example
         * ```
         * Quatd q{1,2,3,4};
         * cout << q.exp() << endl;
         * ```
         */
        Quat<_Tp> exp() const;

        /**
         * @brief return the value of logarithm function
         * \f[\ln(q) = \ln||q|| + \frac{\boldsymbol{v}}{||\boldsymbol{v}||}\arccos\frac{a}{||q||}\f]
         * @param q a quaternion
         * @param assumeUnit if true, q should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * Quatd q1{1,2,3,4};
         * cout << log(q1) << endl;
         * ```
         */
        template <typename T>
        friend Quat<T> log(const Quat<T> &q, bool assumeUnit);

        /**
         * @brief return the value of logarithm function
         *  \f[\ln(q) = \ln||q|| + \frac{\boldsymbol{v}}{||\boldsymbol{v}||}\arccos\frac{w}{||q||}\f]
         * @param assumeUnit if true, this quaternion should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * q.log();
         *
         * Quatd q1(1,2,3,4);
         * q1.normalize().log(true);
         * ```
         */
        Quat<_Tp> log(bool assumeUnit=false) const;

        /**
         * @brief return the value of power function with index \f$x\f$
         * \f[q^x = ||q||(cos(x\theta) + \boldsymbol{u}sin(x\theta)))\f]
         * @param q a quaternion
         * @param x index of exponentiation
         * @param assumeUnit if true, quaternion q should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * quatd q(1,2,3,4);
         * power(q, 2);
         *
         * double angle = CV_PI;
         * Vec3d axis{0, 0, 1};
         * Quatd q1(angle, axis); //generate a unit quat by axis and angle
         * power(q1, 2, true);
         * ```
         */
        template <typename T, typename _T>
        friend Quat<T> power(const Quat<T> &q, _T x, bool assumeUnit);

        /**
         * @brief return the value of power function with index \f$x\f$
         * \f[q^x = ||q||(\cos(x\theta) + \boldsymbol{u}\sin(x\theta)))\f]
         * @param x index of exponentiation
         * @param assumeUnit if true, this quaternion should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * quatd q(1,2,3,4);
         * q.power(2);
         *
         * double angle = CV_PI;
         * Vec3d axis{0, 0, 1};
         * Quatd q1(angle, axis); //generate a unit quat by axis and angle
         * q1.power(2, true); //true means q is a unit quaternion
         * ```
         */
        template <typename _T>
        Quat<_Tp> power(_T x, bool assumeUnit=false) const;

        /**
         * @brief return \f$\sqrt{q}\f$
         * @param q a quaternion
         * @param assumeUnit if true, quaternion q should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * Quatf q(1,2,3,4);
         * sqrt(q);
         *
         * q = {1,0,0,0};
         * sqrt(q, true); //true means q is a unit quaternion
         * ```
         */
        template <typename T>
        friend Quat<T> sqrt(const Quat<T> &q, bool assumeUnit);

        /**
         * @brief return \f$\sqrt{q}\f$
         * @param assumeUnit if true, this quaternion should be a unit quaternion and this function will save some computations.
         *
         * For example
         * ```
         * Quatf q(1,2,3,4);
         * q.sqrt();
         *
         * q = {1,0,0,0};
         * q.sqrt(true); //true means q is a unit quaternion
         * ```
         */
        Quat<_Tp> sqrt(bool assumeUnit=false) const;

        /**
         * @brief return the value of power function with quaternion \f$q\f$
         * \f[p^q = e^{q\ln(p)}\f]
         * @param p base quaternion of power function
         * @param q index quaternion of power function
         * @param assumeUnit if true, quaternon \f$p\f$ should be a unit quaternion and this function will save some computations
         *
         * For example
         * ```
         * Quatd p(1,2,3,4);
         * Quatd q(5,6,7,8);
         * power(p, q);
         *
         * p = p.normalize();
         * power(p, q, true); //true means p is a unit quaternion
         * ```
         */
        template <typename T>
        friend Quat<T> power(const Quat<T> &p, const Quat<T> &q, bool assumeUnit);

        /**
         * @brief return the value of power function with quaternion \f$q\f$
         * \f[p^q = e^{q\ln(p)}\f]
         * @param q index quaternion of power function
         * @param assumeUnit if true, this quaternon should be a unit quaternion and this function will save some computations
         *
         * For example
         * ```
         * Quatd p(1,2,3,4);
         * Quatd q(5,6,7,8);
         * p.power(q);
         *
         * p = p.normalize();
         * p.power(q, true); //true means p is a unit quaternion
         * ```
         */
        Quat<_Tp> power(const Quat<_Tp> &q, bool assumeUnit=false) const;

        /**
         * @brief return the crossProduct between \f$p = (a, b, c, d) = (a, \boldsymbol{u})\f$ and \f$q = (w, x, y, z) = (w, \boldsymbol{v})\f$
         * \f[p \times q = \frac{pq- qp}{2}\f]
         * \f[p \times q = \boldsymbol{u} \times \boldsymbol{v}\f]
         * \f[p \times q = (cz-dy)i + (dx-bz)j + (by-xc)k \f]
         *
         * For example
         * ```
         * Quatd q{1,2,3,4};
         * Quatd p{5,6,7,8};
         * crossPruduct(p, q)
         * ```
         */
        template <typename T>
        friend Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

        /**
         * @brief return the crossProduct between \f$p = (a, b, c, d) = (a, \boldsymbol{u})\f$ and \f$q = (w, x, y, z) = (w, \boldsymbol{v})\f$
         * \f[p \times q = \frac{pq- qp}{2}\f]
         * \f[p \times q = \boldsymbol{u} \times \boldsymbol{v}\f]
         * \f[p \times q = (cz-dy)i + (dx-bz)j + (by-xc)k \f]
         *
         * For example
         * ```
         * Quatd q{1,2,3,4};
         * Quatd p{5,6,7,8};
         * p.crossPruduct(q)
         * ```
         */
        Quat<_Tp> crossProduct(const Quat<_Tp> &q) const;

        /**
         * @brief return the norm of quaternion
         * \f[||q|| = \sqrt{w^2 + x^2 + y^2 + z^2}\f]
         */
        _Tp norm() const;

        /**
         * @brief return a normalzed \f$p\f$
         * \f[p = \frac{q}{||q||}\f]
         * where \f$p\f$ satisfies \f$(p.x)^2 + (p.y)^2 + (p.z)^2 + (p.w)^2 = 1.\f$
         */
        Quat<_Tp> normalize() const;

        /**
         * @brief return \f$q^{-1}\f$ which is an inverse of \f$q\f$
         * which satisfies \f$q * q^{-1} = 1\f$
         * @param q a quaternion
         * @param assumeUnit if true, quaternon q should be a unit quaternion and this function will save some computations
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * inv(q);
         *
         * q = q.normalize();
         * inv(q, true);
         * ```
         */
        template <typename T>
        friend Quat<T> inv(const Quat<T> &q, bool assumeUnit);

        /**
         * @brief return \f$q^{-1}\f$ which is an inverse of \f$q\f$
         * satisfying \f$q * q^{-1} = 1\f$
         * @param assumeUnit if true, quaternon q should be a unit quaternion and this function will save some computations
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * q.inv();
         *
         * q = q.normalize();
         * q.inv(true);
         * ```
         */
        Quat<_Tp> inv(bool assumeUnit=false) const;

        /**
         * @brief return sinh value of quaternion q, sinh could be calculated as:
         * @param q a quaternion
         * \f[\sinh(p) = \sin(w)\cos(||\boldsymbol{v}||) + \cosh(w)\frac{v}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||\f]
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
         * @param q a quaternion
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
         * \f[ \tanh(q) = \frac{\sinh(q)}{\cosh(q)}\f]
         * @param q a quaternion
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
         * \f[ \tanh(q) = \frac{\sinh(q)}{\cosh(q)}\f]
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
         * @param q a quaternion
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
         * @param q a quaternion
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
         * \f[\tan(q) = \frac{\sin(q)}{\cos(q)}\f]
         * @param q a quaternion
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
         * \f[\tan(q) = \frac{\sin(q)}{\cos(q)}\f]
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
         * @param q a quaternion
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
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * q.asin();
         * ```
         */
        Quat<_Tp> asin() const;

        /**
         * @brief return arcsin value of quaternion q, arccos could be calculated as:
         * \f[\arccos(q) = -\frac{\boldsymbol{v}}{||\boldsymbol{v}||}arccosh(q)\f]
         * @param q a quaternion
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
         * @param q a quaternion
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
         * \f[arcsinh(q) = \ln(q + \sqrt{q^2 + 1})\f]
         * @param q a quaternion
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
         * \f[arcsinh(q) = \ln(q + \sqrt{q^2 + 1})\f]
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
         * \f[arccosh(q) = \ln(q + \sqrt{q^2 - 1})\f]
         * @param q a quaternion
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
         * \f[arcsinh(q) = \ln(q + \sqrt{q^2 - 1})\f]
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
         * \f[arctanh(q) = \frac{\ln(q + 1) - \ln(1 - q)}{2}\f]
         * @param q a quaternion
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
         * \f[arcsinh(q) = \frac{\ln(q + 1) - \ln(1 - q)}{2}\f]
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * q.atanh();
         * ```
         */
        Quat<_Tp> atanh() const;

        /**
         * @brief return true if this quaternion is a unit quaternion
         * @param eps tolerance scope of normalization. The eps could be defined as
         *
         * \f[eps = |1 - dotValue|\f] where \f[dotValue = (this.w^2 + this.x^2 + this,y^2 + this.z^2)\f]
         * And this function will consider it is normalized when the dotValude over a range \f$[1-eps, 1+eps]\f$.
         */
        bool isNormal(_Tp eps=CV_QUAT_EPS) const;

        /**
         * @brief to throw an error if this quaternion is not a unit quaternion
         * @param eps tolerance scope of normalization
         * @sa isNormal
         */
        void assertNormal(_Tp eps=CV_QUAT_EPS) const;

        /**
         * @brief transform a quaternion to a 3x3 rotate matrix.
         * @param assumeUnit if true, this quaternon should be a unit quaternion and
         * this function will save some computations. Otherwise, this function will normalize this
         * quaternion at first then to do the transformation.
         *
         * @note Matrix A which is to be rotated has the form
         * \f[\begin{bmatrix}
         * x_0& x_1& x_2&...&x_n\\
         * y_0& y_1& y_2&...&y_n\\
         * z_0& z_1& z_2&...&z_n
         * \end{bmatrix}\f]
         * where the same subscript represents a point. The shape of A equals [3, n]
         * The points matrix A can be rotated by toRotMat3x3() * A.
         * The result has 3 rows and n columns too

         * For example
         * ```
         * double angle = CV_PI;
         * Vec3d axis{0,0,1};
         * Quatd q_unit{angle, axis}; //quaternion coule also be get by interpolation by two or more quaternions.
         *
         * //assume there is two points (1,0,0) and (1,0,1) to be rotated
         * Mat pointsA = (Mat_<double>(2, 3) << 1,0,0,1,0,1);
         * //change the shape
         * pointsA = pointsA.t()
         * // rotate 180 degrees around the z axis
         * Mat new_point = q_unit.toRotMat3x3() * pointsA;
         * // print two points
         * cout << new_point << endl;
         * ```
         */
        cv::Mat toRotMat3x3(bool assumeUnit=false) const;

        /**
         * @brief transform a quaternion to a 4x4 rotate matrix.
         * @param assumeUnit if true, this quaternon should be a unit quaternion and
         * this function will save some computations. Otherwise, this function will normalize this
         * quaternion at first then to do the transformation.
         *
         * The operations is similar as toRotMat3x3
         * except that the points matrix should have the form
         * \f[\begin{bmatrix}
         * 0&0&0&...&0\\
         * x_0& x_1& x_2&...&x_n\\
         * y_0& y_1& y_2&...&y_n\\
         * z_0& z_1& z_2&...&z_n
         * \end{bmatrix}\f]
         *
         * @sa toRotMat3x3
         */
        cv::Mat toRotMat4x4(bool assumeUnit=false) const;

        /**
         * @brief transoform the this quaternion to a Vec<T, 4>
         *
         * For example
         * ```
         * Quatd q(1,2,3,4);
         * q.toVec();
         * ```
         */
        cv::Vec<_Tp, 4> toVec() const;

        /**
         * @brief transoform this quaternion to a Rodrigues vector
         * @param assumeUnit if true, this quaternon should be a unit quaternion and
         * this function will save some computations.
         * Rodtigues vector rVec is defined as:
         * \f[ rVec = [\tan{\frac{\theta}{2}}v_x, \tan{\frac{\theta}{2}}v_y, \tan{\frac{\theta}{2}}v_z]\f]
         * where \f$\theta\f$ represents rotate angle, and \f$v\f$ represents the normalized axis.
         * @note Due to the tangent, the quaternion cannot convert to rotation vector when the rotation angle equals \f$pi + 2kpi\f$ radians or \f$180 + 360k\f$ degree, \f$k = ... -2, -1, 1, 2 ...\f$.
         */
        cv::Vec<_Tp, 3> toRodrigues(bool assumeUnit=false)  const;

        /**
        * @brief get the angle of quaternion, it returns the rotate angle.
        * @param assumeUnit if true, this quaternon should be a unit quaternion and
        * this function will save some computations.
        * \f[\psi = 2 *arccos(\frac{w}{||q||})\f]
        *
        * For example
        * ```
        * Quatd q(1,2,3,4);
        * q.getAngle();
        *
        * q.normalize().getAngle(true);//same as q.getAngle()
        * ```
        */
        _Tp getAngle(bool assumeUnit=false) const;

        /**
        * @brief get the axis of quaternion, it returns a vector of length 3
        * @param assumeUnit if true, this quaternon should be a unit quaternion and
        * this function will save some computations.
        * the unit axis \f$\boldsymbol{n}\f$ is defined by
        * \f[\begin{array}
        *      \boldsymbol{v} &= \boldsymbol{u} ||\boldsymbol{v}||\\
        *             &= \boldsymbol{u}||q||sin(\theta)
        *      \end{array}\f]
        *
        * For example
        * ```
        * Quatd q(1,2,3,4);
        * q.getAxis();
        *
        * q.normalize().getAxis(true);//same as q.getAxis()
        * ```
        */
        cv::Vec<_Tp, 3> getAxis(bool assumeUnit=false) const;

        /**
         * @brief return the dot between quaternion \f$q\f$ and this quaternion, i.e.,
         * \f[p \cdot q = this.w \cdot q.w + this.x \cdot x + this.y \cdot y + this.z \cdot z\f]
         * @param q the other quaternion
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
         * \f[Lerp(q_0, q_1, t) = (1 - t)q_0 + tq_1 \f]
         * Obviously, the lerp will interpolate along a straight line if we think of \f$q_0\f$ and \f$q_1\f$ as a vector
         * in a two-demensional space. When \f$t = 0\f$, it returns \f$q_0\f$ and when \f$t= 1\f$, it returns \f$q_1\f$.
         * \f$t\f$ should be ranged in \f$[0, 1]\f$ normally
         * @param q0 a quaternion used in linear interpolation
         * @param q1 a quaternion used in linear interpolation
         * @param t percent of vector \f$\overrightarrow{q_0q_1}\f$ over a range [0, 1]
         * @note it returns a non-unit quaternion.
         */
        static Quat<_Tp> lerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t);

        /**
         * @brief To calculate the interpolation from \f$q_0\f$ to \f$q_1\f$ by Normalized Linear Interpolation(Nlerp).
         * it returns a normalized quaternion of Linear Interpolation(Lerp).
         * \f[ Nlerp(q_0, q_1, t) = \frac{(1 - t)q_0 + tq_1}{||(1 - t)q_0 + tq_1||}.\f]
         * The interplation will always choose the nearest path.
         * @param q0 a quaternion used in normalized linear interpolation
         * @param q1 a quaternion used in normalized linear interpolation
         * @param t percent of vector \f$\overrightarrow{q_0q_1}\f$ over a range [0, 1]
         * @param assumeUnit if true, all input quaternions should be unit quaternion. Otherwise, all inputs
         quaternion will be normalized inside the function
         * @sa lerp
         */
        static Quat<_Tp> nlerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, bool assumeUnit=false);

        /**
         @brief To calculate the interpolation between \f$q_0\f$ and \f$q_1\f$ by Spherical Linear
         Interpolation(Slerp), which can be defined as:
         \f[ Slerp(q_0, q_1, t) = \frac{\sin((1-t)\theta)}{\sin(\theta)}q_0 + \frac{\sin(t\theta)}{\sin(\theta)}q_1\f]
         where \f$\theta\f$ can be calculated as:
         \f[\theta=cos^{-1}(q_0\cdot q_1)\f]
         resulting from the both of their norm is unit
         @param q0 a quaternion used in Slerp
         @param q1 a quaternion used in Slerp
         @param t percent of angle between \f$q_0\f$ and \f$q_1\f$ over a range [0, 1]
         @param assumeUnit if true, all input quaternions should be unit quaternions. Otherwise, all input
         quaternions will be normalized inside the function
         @param directChange if true, the interpolation will choose the nearest path.
         @note If the interpolation angle is small, the error between Nlerp and Slerp is not so large. To improve efficiency and
         avoid zero division error, we use Nlerp instead of Slerp.
        */
        static Quat<_Tp> slerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, bool assumeUnit=false, bool directChange=true);

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
         * @param q0 the first quaternion
         * @param s0 the second quaternion
         * @param s1 the third quaternion
         * @param q1 thr fourth quaternion
         * @param t interpolation parameter of quadratic and linear interpolation over a range \f$[0, 1]\f$
         * @param assumeUnit if true, all input quaternions should be unit quaternion. Otherwise, all input
         * quaternions will be normalized inside the function
         * @param directChange if true, squad will find the nearest path to interpolate
         * @sa interPoint, spline
         */
        static Quat<_Tp> squad(const Quat<_Tp> &q0, const Quat<_Tp> &s0,
                               const Quat<_Tp> &s1, const Quat<_Tp> &q1,
                               const _Tp t, bool assumeUnit=false,
                               bool directChange=true);

        /**
         * @brief This is the part calculation of squad.
         * To calculate the intermedia quaternion \f$s_i\f$ between each three quaternion
         * \f[s_i = q_i\exp(-\frac{\log(q^*_iq_{i+1}) + \log(q^*_iq_{i-1})}{4})\f]
         * @param q0 the first quaternion
         * @param q1 the second quaternion
         * @param q2 the third quaternion
         * @param assumeUnit if true, all input quaternions should be unit quaternion. Otherwise, all input
         * quaternions will be normalized inside the function
         * @sa squad
         */
        static Quat<_Tp> interPoint(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                                    const Quat<_Tp> &q2, bool assumeUnit=false);

        /**
         * @brief to calculate a quaterion which is the result of a \f$C^1\f$ continuous
         * spline curve constructed by squad at the ratio t. Here, the interpolation values are
         * between \f$q_1\f$ and \f$q_2\f$. \f$q_0\f$ and \f$q_2\f$ are used to ensure the \f$C^1\f$
         * continuity. if t = 0, it returns \f$q_1\f$, if t = 1, it returns \f$q_2\f$.
         * @param q0 the first input quaternion to ensure \f$C^1\f$ continuity
         * @param q1 the second input quaternion
         * @param q2 the third input quaternion
         * @param q3 the fourth input quaternion the same use of \f$q1\f$
         * @param t ratio over a range [0, 1].
         * @param assumeUnit if true, \f$q_0, q_1, q_2, q_3\f$ should be normalized. Otherwise, all input
         * quaternions will be normalized inside the function.
         *
         * For example:
         *
         * If there are three double quaternions \f$v_0, v_1, v_2\f$ waiting to be interpolated.
         *
         * Interpolation between \f$v_0\f$ and \f$v_1\f$ with a ratio \f$t_0\f$ could be calculated as
         * ```
         * Quatd::spline(v0, v0, v1. v2, t0);
         * ```
         * Interpolation between \f$v_1\f$ and \f$v_2\f$ with a ratio \f$t_0\f$ could be calculated as
         * ```
         * Quatd::spline(v0, v1, v2. v2, t0);
         * ```
         * @sa squad, slerp
         */
        static Quat<_Tp> spline(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                                const Quat<_Tp> &q2, const Quat<_Tp> &q3,
                                const _Tp t, bool assumeUnit=false);


        Quat<_Tp> operator-() const;

        bool operator==(const Quat<_Tp>&) const;

        Quat<_Tp> operator+(const Quat<_Tp>&) const;

        Quat<_Tp>& operator+=(const Quat<_Tp>&);

        Quat<_Tp> operator-(const Quat<_Tp>&) const;

        Quat<_Tp>& operator-=(const Quat<_Tp>&);

        Quat<_Tp>& operator*=(const Quat<_Tp>&);

        Quat<_Tp>& operator*=(const _Tp&);

        Quat<_Tp> operator*(const Quat<_Tp>&) const;

        Quat<_Tp> operator/(const _Tp&) const;

        Quat<_Tp> operator/(const Quat<_Tp>&) const;

        Quat<_Tp>& operator/=(const _Tp&);

        Quat<_Tp>& operator/=(const Quat<_Tp>&);

        _Tp& operator[](std::size_t n);

        const _Tp& operator[](std::size_t n) const;

        template <typename S, typename T>
        friend Quat<S> cv::operator*(const T, const Quat<S>&);

        template <typename S, typename T>
        friend Quat<S> cv::operator*(const Quat<S>&, const T);

        template <typename S>
        friend std::ostream& cv::operator<<(std::ostream&, const Quat<S>&);

        _Tp w, x, y, z;

    };
    template <typename T>
    Quat<T> rvec2Quat(Vec<T, 3> &rvec);

    template <typename T>
    Quat<T> inv(const Quat<T> &q, bool assumeUnit=false);

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
    Quat<T> power(const Quat<T> &q, const Quat<T> &p, bool assumeUnit=false);

    template <typename T>
    Quat<T> exp(const Quat<T> &q);

    template <typename T>
    Quat<T> log(const Quat<T> &q, bool assumeUnit=false);

    template <typename T, typename _T>
    Quat<T> power(const Quat<T>& q, _T x, bool assumeUnit=false);

    template <typename T>
    Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

    template <typename S>
    Quat<S> sqrt(const Quat<S> &q, bool assumeUnit=false);

    template <typename S, typename T>
    Quat<S> operator*(const T, const Quat<S>&);

    template <typename S, typename T>
    Quat<S> operator*(const Quat<S>&, const T);

    template <typename S>
    std::ostream& operator<<(std::ostream&, const Quat<S>&);

    using Quatd = Quat<double>;
    using Quatf = Quat<float>;
    using std::sin;
    using std::cos;
    using std::tan;
    using std::exp;
    using std::log;
    using std::sinh;
    using std::cosh;
    using std::tanh;
    using std::sqrt;
    using std::asinh;
    using std::acosh;
    using std::atanh;
    using std::asin;
    using std::acos;
    using std::atan;
//! @} core
}

//! @cond IGNORE
///////////////////////////////////////////////////////////////////////////////////////
//Implementation
namespace cv{
    template <typename T>
    Quat<T>::Quat(const Vec<T, 4> &coeff):w(coeff[0]), x(coeff[1]), y(coeff[2]), z(coeff[3]){}

    template <typename T>
    Quat<T>::Quat(const T qw, const T qx, const T qy, const T qz):w(qw), x(qx), y(qy), z(qz){}

    template <typename T>
    Quat<T>::Quat(const T angle, const cv::Vec<T, 3> &axis)
    {
        T vNorm = std::sqrt(axis.dot(axis));
        if (vNorm < CV_QUAT_EPS)
        {
            CV_Error(Error::StsBadArg, "this quaternion does not represent a rotation");
        }
        w = std::cos(angle / 2);
        x = std::sin(angle / 2) * (axis[0] / vNorm);
        y = std::sin(angle / 2) * (axis[1] / vNorm);
        z = std::sin(angle / 2) * (axis[2] / vNorm);
    }

    template <typename T>
    Quat<T>::Quat(const Mat &R)
    {
        if (R.rows != 3 || R.cols != 3)
        {
            CV_Error(Error::StsBadArg, "Cannot matrix to quaternion: rotation matrix should be a 3x3 matrix");
        }
        T S;
        T trace = R.at<T>(0, 0) + R.at<T>(1, 1) + R.at<T>(2, 2);
        if (trace > 0)
        {
            S = std::sqrt(trace + 1) * 2;
            x = (R.at<T>(1, 2) - R.at<T>(2, 1)) / S;
            y = (R.at<T>(2, 0) - R.at<T>(0, 2)) / S;
            z = (R.at<T>(0, 1) - R.at<T>(1, 0)) / S;
            w = -0.25 * S;
        }
        else if (R.at<T>(0, 0) > R.at<T>(1, 1) && R.at<T>(0, 0) > R.at<T>(2, 2))
        {

            S = std::sqrt(1.0 + R.at<T>(0, 0) - R.at<T>(1, 1) - R.at<T>(2, 2)) * 2;
            x = -0.25 * S;
            y = -(R.at<T>(1, 0) + R.at<T>(0, 1)) / S;
            z = -(R.at<T>(0, 2) + R.at<T>(2, 0)) / S;
            w = (R.at<T>(1, 2) - R.at<T>(2, 1)) / S;
        }
        else if (R.at<T>(1, 1) > R.at<T>(2, 2))
        {
            S = std::sqrt(1.0 - R.at<T>(0, 0) + R.at<T>(1, 1) - R.at<T>(2, 2)) * 2;
            x = (R.at<T>(0, 1) + R.at<T>(1, 0)) / S;
            y = 0.25 * S;
            z = (R.at<T>(1, 2) + R.at<T>(2, 1)) / S;
            w = (R.at<T>(0, 2) - R.at<T>(2, 0)) / S;
        }
        else
        {
            S = std::sqrt(1.0 - R.at<T>(0, 0) - R.at<T>(1, 1) + R.at<T>(2, 2)) * 2;
            x = (R.at<T>(0, 2) + R.at<T>(2, 0)) / S;
            y = (R.at<T>(1, 2) + R.at<T>(2, 1)) / S;
            z = 0.25 * S;
            w = -(R.at<T>(0, 1) - R.at<T>(1, 0)) / S;
        }
    }

    template <typename T>
    Quat<T>::Quat(const Vec<T, 3> &rvec)
    {
        T tanVal = std::sqrt(rvec.dot(rvec));
        if (tanVal < CV_QUAT_EPS)
        {
            w = 0;
            x = 0;
            y = 0;
            z = 0;
        }
        else
        {
            T angle = std::atan(tanVal);
            w = std::cos(angle);
            x = rvec[0] / tanVal * std::sin(angle);
            y = rvec[1] / tanVal * std::sin(angle);
            z = rvec[2] / tanVal * std::sin(angle);
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
        return (abs(w - q.w) < CV_QUAT_EPS && abs(x - q.x) < CV_QUAT_EPS && abs(y - q.y) < CV_QUAT_EPS && abs(z - q.z) < CV_QUAT_EPS);
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


    template <typename T, typename S>
    Quat<T> operator*(const Quat<T> &q1, const S a)
    {
        return Quat<T>(a * q1.w, a * q1.x, a * q1.y, a * q1.z);
    }

    template <typename T, typename S>
    Quat<T> operator*(const S a, const Quat<T> &q1)
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
    inline Quat<T>& Quat<T>::operator/=(const Quat<T> &q1)
    {
        Quat<T> q(*this * q1.inv());
        w = q.w;
        x = q.x;
        y = q.y;
        z = q.z;
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
                CV_Error(Error::StsOutOfRange, "subscript exceeds the index range");
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
                CV_Error(Error::StsOutOfRange, "subscript exceeds the index range");
        }
    }

    template <typename T>
    std::ostream & operator<<(std::ostream &os, const Quat<T> &q)
    {
        os << "Quat " << cv::Vec<T, 4>{q.w, q.x, q.y, q.z};
        return os;
    }

    template <typename T>
    inline T Quat<T>::at(size_t index) const
    {
        return (*this)[index];
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

    template <typename T>
    Quat<T> exp(const Quat<T> &q)
    {
        return q.exp();
    }

    template <typename T>
    Quat<T> Quat<T>::exp() const
    {
        cv::Vec<T, 3> v{x, y, z};
        T normV = std::sqrt(v.dot(v));
        T k = normV < CV_QUAT_EPS ? 1 : std::sin(normV) / normV;
        return std::exp(w) * Quat<T>(std::cos(normV), v[0] * k, v[1] * k, v[2] * k);
    }

    template <typename T>
    Quat<T> log(const Quat<T> &q, bool assumeUnit)
    {
        return q.log(assumeUnit);
    }

    template <typename T>
    Quat<T> Quat<T>::log(bool assumeUnit) const
    {
        cv::Vec<T, 3> v{x, y, z};
        T vNorm = std::sqrt(v.dot(v));
        if (assumeUnit)
        {
            T k = vNorm < CV_QUAT_EPS ? 1 : std::acos(w) / vNorm;
            return Quat<T>(0, v[0] * k, v[1] * k, v[2] * k);
        }
        T qNorm = norm();
        if (qNorm < CV_QUAT_EPS)
        {
            CV_Error(Error::StsBadArg, "Cannot apply this quaternion to log function: undefined");
        }
        T k = vNorm < CV_QUAT_EPS ? 1 : std::acos(w / qNorm) / vNorm;
        return Quat<T>(std::log(qNorm), v[0] * k, v[1] * k, v[2] *k);
    }

    template <typename T, typename _T>
    inline Quat<T> power(const Quat<T> &q1, _T alpha, bool assumeUnit)
    {
        return q1.power(alpha, assumeUnit);
    }

    template <typename T>
    template <typename _T>
    inline Quat<T> Quat<T>::power(_T alpha, bool assumeUnit) const
    {
        if (x * x + y * y + z * z > CV_QUAT_EPS)
        {
            T angle = getAngle(assumeUnit);
            cv::Vec<T, 3> axis = getAxis(assumeUnit);
            if (assumeUnit)
            {
                return Quat<T>(alpha * angle, axis);
            }
            return std::pow(norm(), alpha) * Quat<T>(alpha * angle, axis);
        }
        else
        {
            return std::pow(norm(), alpha) * Quat<T>(w, x, y, z);
        }
    }


    template <typename T>
    inline Quat<T> sqrt(const Quat<T> &q, bool assumeUnit)
    {
        return q.sqrt(assumeUnit);
    }

    template <typename T>
    inline Quat<T> Quat<T>::sqrt(bool assumeUnit) const
    {
        return power(0.5, assumeUnit);
    }


    template <typename T>
    inline Quat<T> power(const Quat<T> &p, const Quat<T> &q, bool assumeUnit)
    {
        return p.power(q, assumeUnit);
    }


    template <typename T>
    inline Quat<T> Quat<T>::power(const Quat<T> &q, bool assumeUnit) const
    {
        return cv::exp(q * log(assumeUnit));
    }

    template <typename T>
    inline T Quat<T>::dot(Quat<T> q1) const
    {
        return w * q1.w + x * q1.x + y * q1.y + z * q1.z;
    }


    template <typename T>
    inline Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q)
    {
        return p.crossProduct(q);
    }


    template <typename T>
    inline Quat<T> Quat<T>::crossProduct(const Quat<T> &q) const
    {
        return Quat<T> (0, y * q.z - z * q.y, z * q.x - x * q.z, x * q.y - q.x * y);
    }

    template <typename T>
    inline Quat<T> Quat<T>::normalize() const
    {
        T normVal = norm();
        if (normVal < CV_QUAT_EPS)
        {
            CV_Error(Error::StsBadArg, "Cannot normalize this quaternion: the norm is too small.");
        }
        return Quat<T>(w / normVal, x / normVal, y / normVal, z / normVal) ;
    }

    template <typename T>
    inline Quat<T> inv(const Quat<T> &q, bool assumeUnit)
    {
        return q.inv(assumeUnit);
    }


    template <typename T>
    inline Quat<T> Quat<T>::inv(bool assumeUnit) const
    {
        if (assumeUnit)
        {
            return conjugate();
        }
        T norm2 = dot(*this);
        if (norm2 < CV_QUAT_EPS)
        {
            CV_Error(Error::StsBadArg, "This quaternion do not have inverse quaternion");
        }
        return conjugate() / norm2;
    }

    template <typename T>
    inline Quat<T> sinh(const Quat<T> &q)
    {
        return q.sinh();
    }


    template <typename T>
    inline Quat<T> Quat<T>::sinh() const
    {
        cv::Vec<T, 3> v{x, y ,z};
        T vNorm = std::sqrt(v.dot(v));
        T k = vNorm < CV_QUAT_EPS ? 1 : std::cosh(w) * std::sin(vNorm) / vNorm;
        return Quat<T>(std::sinh(w) * std::cos(vNorm), v[0] * k, v[1] * k, v[2] * k);
    }


    template <typename T>
    inline Quat<T> cosh(const Quat<T> &q)
    {
        return q.cosh();
    }


    template <typename T>
    inline Quat<T> Quat<T>::cosh() const
    {
        cv::Vec<T, 3> v{x, y ,z};
        T vNorm = std::sqrt(v.dot(v));
        T k = vNorm < CV_QUAT_EPS ? 1 : std::sinh(w) * std::sin(vNorm) / vNorm;
        return Quat<T>(std::cosh(w) * std::cos(vNorm), v[0] * k, v[1] * k, v[2] * k);
    }

    template <typename T>
    inline Quat<T> tanh(const Quat<T> &q)
    {
        return q.tanh();
    }

    template <typename T>
    inline Quat<T> Quat<T>::tanh() const
    {
        return sinh() * cosh().inv();
    }


    template <typename T>
    inline Quat<T> sin(const Quat<T> &q)
    {
        return q.sin();
    }


    template <typename T>
    inline Quat<T> Quat<T>::sin() const
    {
        cv::Vec<T, 3> v{x, y ,z};
        T vNorm = std::sqrt(v.dot(v));
        T k = vNorm < CV_QUAT_EPS ? 1 : std::cos(w) * std::sinh(vNorm) / vNorm;
        return Quat<T>(std::sin(w) * std::cosh(vNorm), v[0] * k, v[1] * k, v[2] * k);
    }

    template <typename T>
    inline Quat<T> cos(const Quat<T> &q)
    {
        return q.cos();
    }

    template <typename T>
    inline Quat<T> Quat<T>::cos() const
    {
        cv::Vec<T, 3> v{x, y ,z};
        T vNorm = std::sqrt(v.dot(v));
        T k = vNorm < CV_QUAT_EPS ? 1 : std::sin(w) * std::sinh(vNorm) / vNorm;
        return Quat<T>(std::cos(w) * std::cosh(vNorm), -v[0] * k, -v[1] * k, -v[2] * k);
    }

    template <typename T>
    inline Quat<T> tan(const Quat<T> &q)
    {
        return q.tan();
    }

    template <typename T>
    inline Quat<T> Quat<T>::tan() const
    {
        return sin() * cos().inv();
    }

    template <typename T>
    inline Quat<T> asinh(const Quat<T> &q)
    {
        return q.asinh();
    }

    template <typename T>
    inline Quat<T> Quat<T>::asinh() const
    {
        return cv::log(*this + cv::power(*this * *this + Quat<T>(1, 0, 0, 0), 0.5));
    }

    template <typename T>
    inline Quat<T> acosh(const Quat<T> &q)
    {
        return q.acosh();
    }

    template <typename T>
    inline Quat<T> Quat<T>::acosh() const
    {
        return cv::log(*this + cv::power(*this * *this - Quat<T>(1,0,0,0), 0.5));
    }

    template <typename T>
    inline Quat<T> atanh(const Quat<T> &q)
    {
        return q.atanh();
    }

    template <typename T>
    inline Quat<T> Quat<T>::atanh() const
    {
        Quat<T> ident(1, 0, 0, 0);
        Quat<T> c1 = (ident + *this).log();
        Quat<T> c2 = (ident - *this).log();
        return 0.5 * (c1 - c2);
    }

    template <typename T>
    inline Quat<T> asin(const Quat<T> &q)
    {
        return q.asin();
    }

    template <typename T>
    inline Quat<T> Quat<T>::asin() const
    {
        Quat<T> v(0, x, y, z);
        T vNorm = v.norm();
        T k = vNorm < CV_QUAT_EPS ? 1 : vNorm;
        return -v / k * (*this * v / k).asinh();
    }

    template <typename T>
    inline Quat<T> acos(const Quat<T> &q)
    {
        return q.acos();
    }

    template <typename T>
    inline Quat<T> Quat<T>::acos() const
    {
        Quat<T> v(0, x, y, z);
        T vNorm = v.norm();
        T k = vNorm < CV_QUAT_EPS ? 1 : vNorm;
        return -v / k * acosh();
    }

    template <typename T>
    inline Quat<T> atan(const Quat<T> &q)
    {
        return q.atan();
    }

    template <typename T>
    inline Quat<T> Quat<T>::atan() const
    {
        Quat<T> v(0, x, y, z);
        T vNorm = v.norm();
        T k = vNorm < CV_QUAT_EPS ? 1 : vNorm;
        return -v / k * (*this * v / k).atanh();
    }

    template <typename T>
    inline T Quat<T>::getAngle(bool assumeUnit) const
    {
        if (assumeUnit)
        {
            return 2 * std::acos(w);
        }
        if (x * x + y * y + z * z < CV_QUAT_EPS || norm() < CV_QUAT_EPS )
        {
            CV_Error(Error::StsBadArg, "This quaternion does not represent a rotation");
        }
        return 2 * std::acos(w / norm());
    }

    template <typename T>
    inline Vec<T, 3> Quat<T>::getAxis(bool assumeUnit) const
    {
        T angle = getAngle(assumeUnit);
        if (assumeUnit)
        {
            return Vec<T, 3>{x, y, z} / std::sin(angle / 2);
        }
        return Vec<T, 3> {x, y, z} / (norm() * std::sin(angle / 2));
    }

    template <typename T>
    cv::Mat Quat<T>::toRotMat4x4(bool assumeUnit) const
    {
        T a = w, b = x, c = y, d = z;
        if (!assumeUnit)
        {
            Quat<T> qTemp = normalize();
            a = qTemp.w;
            b = qTemp.x;
            c = qTemp.y;
            d = qTemp.z;
        }
        cv::Matx<T, 4, 4> R{
            1,                      0,                       0,                   0,
            0,1 - 2 * (c * c + d * d), 2 * (b * c + a * d)    , 2 * (b * d - a * c),
            0,2 * (b * c - a * d)    , 1 - 2 * (b * b + d * d), 2 * (c * d + a * b),
            0,2 * (b * d + a * c)    , 2 * (c * d - a * b)    , 1 - 2 * (b * b + c * c)};
        return cv::Mat(R).t();
    }

    template <typename T>
    cv::Mat Quat<T>::toRotMat3x3(bool assumeUnit) const
    {
        T a = w, b = x, c = y, d = z;
        if (!assumeUnit)
        {
            Quat<T> qTemp = normalize();
            a = qTemp.w;
            b = qTemp.x;
            c = qTemp.y;
            d = qTemp.z;
        }
        cv::Matx<T, 3, 3> R{
            1 - 2 * (c * c + d * d), 2 * (b * c + a * d)    , 2 * (b * d - a * c),
            2 * (b * c - a * d)    , 1 - 2 * (b * b + d * d), 2 * (c * d + a * b),
            2 * (b * d + a * c)    , 2 * (c * d - a * b)    , 1 - 2 * (b * b + c * c)};
        return cv::Mat(R).t();
    }

    template <typename T>
    Vec<T, 3> Quat<T>::toRodrigues(bool assumeUnit) const
    {
        if (abs(w) < CV_QUAT_EPS)
        {
            CV_Error(Error::StsBadArg, "Cannot convert quaternion to rotation vector: rotation vector is indeterminte");
        }
        if (assumeUnit)
        {
            return Vec<T, 3>{x / w, y / w, z / w};
        }
        Quat<T> q = normalize();
        return Vec<T, 3>{q.x / q.w, q.y / q.w, q.z / q.w};

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
    Quat<T> Quat<T>::slerp(const Quat<T> &q0, const Quat<T> &q1, const T t, bool assumeUnit, bool directChange)
    {
        Quatd v0(q0);
        Quatd v1(q1);

        if (!assumeUnit)
        {
            v0 = v0.normalize();
            v1 = v1.normalize();
            // add warning:
        }
        T cosTheta = v0.dot(v1);
        T DOT_THRESHOLD = 0.995;
        if (cosTheta > DOT_THRESHOLD)
        {
            return nlerp(v0, v1, t, true);
        }

        if (directChange && cosTheta < 0)
        {
            v0 = -v0;
            cosTheta = -cosTheta;
        }
        T sinTheta = std::sqrt(1 - cosTheta * cosTheta);
        T angle = atan2(sinTheta, cosTheta);
        return (std::sin((1 - t) * angle) / (sinTheta) * v0 + std::sin(t * angle) / (sinTheta) * v1).normalize();
    }


    template <typename T>
    inline Quat<T> Quat<T>::nlerp(const Quat<T> &q0, const Quat<T> &q1, const T t, bool assumeUnit)
    {
        Quat<T> v0(q0), v1(q1);
        if (v1.dot(v0) < 0)
        {
            v0 = -v0;
        }
        if (assumeUnit)
        {
            return ((1 - t) * v0 + t * v1).normalize();
        }
        // add warning
        v0 = v0.normalize();
        v1 = v1.normalize();
        return ((1 - t) * v0 + t * v1).normalize();
    }


    template <typename T>
    inline bool Quat<T>::isNormal(T eps) const
    {

        double normVar = norm();
        if ((normVar > 1 - eps) && (normVar < 1 + eps))
            return true;
        return false;
    }

    template <typename T>
    inline void Quat<T>::assertNormal(T eps) const
    {
        if (!isNormal(eps))
            CV_Error(Error::StsBadArg, "Quaternion should be normalized");
    }


    template <typename T>
    inline Quat<T> Quat<T>::squad(const Quat<T> &q0, const Quat<T> &q1,
                                const Quat<T> &q2, const Quat<T> &q3,
                                const T t, bool assumeUnit,
                                bool directChange)
    {
        Quat<T> v0(q0), v1(q1), v2(q2), v3(q3);
        if (!assumeUnit)
        {
            v0 = v0.normalize();
            v1 = v1.normalize();
            v2 = v2.normalize();
            v3 = v3.normalize();
        }

        Quat<T> c0 = slerp(v0, v3, t, assumeUnit, directChange);
        Quat<T> c1 = slerp(v1, v2, t, assumeUnit, directChange);
        return slerp(c0, c1, 2 * t * (1 - t), assumeUnit, directChange);
    }

    template <typename T>
    Quat<T> Quat<T>::interPoint(const Quat<T> &q0, const Quat<T> &q1,
                                const Quat<T> &q2, bool assumeUnit)
    {
        Quat<T> v0(q0), v1(q1), v2(q2);
        if (!assumeUnit)
        {
            v0 = v0.normalize();
            v1 = v1.normalize();
            v2 = v2.normalize();
            // add warning in inter
        }
        return v1 * cv::exp(-(cv::log(v1.conjugate() * v0, assumeUnit) + (cv::log(v1.conjugate() * v2, assumeUnit))) / 4);
    }

    template <typename T>
    Quat<T> Quat<T>::spline(const Quat<T> &q0, const Quat<T> &q1, const Quat<T> &q2, const Quat<T> &q3, const T t, bool assumeUnit)
    {
        Quatd v0, v1, v2, v3;
        v0 = q0;
        v1 = q1;
        v2 = q2;
        v3 = q3;
        if (!assumeUnit)
        {
            v0 = v0.normalize();
            v1 = v1.normalize();
            v2 = v2.normalize();
            v3 = v3.normalize();
        }
        T cosTheta;
        std::vector<Quat<T>> vec{v0, v1, v2, v3};
        for (size_t i = 0; i < 3; ++i)
        {
            cosTheta = vec[i].dot(vec[i + 1]);
            if (cosTheta < 0)
            {
                vec[i + 1] = -vec[i + 1];
            }
        }
        Quat<T> s1 = interPoint(vec[0], vec[1], vec[2], true);
        Quat<T> s2 = interPoint(vec[1], vec[2], vec[3], true);
        return squad(vec[1], s1, s2, vec[2], t, assumeUnit, false);
    }

    template <typename T>
    Quat<T> rvec2Quat(Vec<T, 3> &rvec)
    {
        return Quat<T>(rvec);
    }

}//namepsace

//! @endcond

#endif /* OPENCV_CORE_QUATERNION_HPP */