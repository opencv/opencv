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
// Author: Liangqian Kong <chargerKong@126.com>
//         Longbu Wang <riskiest@gmail.com>
#include <opencv2/core.hpp>
#include <iostream>
namespace cv 
{
    #define CV_QUAT_EPS 1.e-6
    /**
     * Quaternion are a number system that extends the complex numbers.
     * A quaternion is generally represented in the form:
     * 		\f[\begin{array}{rcl}
     * 		q &= w + x\boldsymbol{i} + y\boldsymbol{j} + z\boldsymbol{k}\\
     *		  &= [w, x, y, z]\\
     *		  &= [w, \boldsymbol{v}]
     *		  \end{array}.\f]
     * A unit quaternion is usually represents rotation, which has the form:
     * 		\f[q = [cos\theta, sin\theta\cdot u_x, sin\theta\cdot u_y, sin\theta\cdot u_z].\f]
     *
     * In fact. A general quaternion could also be represented in the form
     * 		\f[q = ||q||[cos\theta, sin\theta\cdot u_x, sin\theta\cdot u_y, sin\theta\cdot u_z].\f]
     * where \f$||q||\f$ represents the norm of \f$q\f$
     * all functions will be degenerate to unit cases when \f$||q|| = 1\f$
     */
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
        explicit Quat(const cv::Vec<_Tp, 4> &coeff);
        Quat(_Tp w, _Tp x, _Tp y, _Tp z);
        Quat(const _Tp angle, const cv::Vec<_Tp, 3> &axis, const _Tp qNorm=1);
        explicit Quat(const cv::Mat &R);
        explicit Quat(const Vec<_Tp, 3> &rodrigurs);

        /**
         * @brief a way to get element
         */
        _Tp at(size_t index) const;

        /**
         * @brief get the conjugate of quaternion
         * \f[q^* = (w, -x, -y, -z)\f]
         */
        Quat<_Tp> conjugate() const;

        /**
         * @brief return the value of exponential function
         * \f[e^q = e^w (cos||v||+ \frac{v}{||v||sin||v||})\f]
         */
        template <typename T>
        friend Quat<T> exp(const Quat<T> &q);

        Quat<_Tp> exp() const;

        /**
         * @brief return the value of logarithm function
         * \f[ln(q) = ln||q|| + \frac{v}{||v||}arccos\frac{a}{||q||}\f]
         */
        template <typename T>
        friend Quat<T> log(const Quat<T> &q, bool assumeUnit);

        Quat<_Tp> log(bool assumeUnit=false) const;

        /**
         * @brief return the value of power function with constant \f$x\f$
         * \f[q^x = ||q||(cos(x\theta) + \boldsymbol{v}sin(x\theta)))\f]
         */
        template <typename T, typename _T>
        friend Quat<T> power(const Quat<T> &q, _T x, bool assumeUnit);
        
        template <typename _T>
        Quat<_Tp> power(_T x, bool assumeUnit=false) const;
        /**
         * @brief return \f$\sqrt{q}\f$
         */
        template <typename T>
        friend Quat<T> sqrt(const Quat<T> &q, bool assumeUnit);

        Quat<_Tp> sqrt(bool assumeUnit=false) const;
        /**
         * @brief return the value of power function with quaternion \f$p\f$
         * \f[q^p = e^{pln(q)}\f]
         * @param assumeUnit represents \f$p\f$ is unit quaternion or not
         */
        template <typename T>
        friend Quat<T> power(const Quat<T> &p, const Quat<T> &q, bool assumeUnit);

        Quat<_Tp> power(const Quat<_Tp> &p, bool assumeUnit=false) const;
        /**
         * @ brief return the crossProduct between \f$q\f$ and \f$q1\f$
         */
        template <typename T>
        friend Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

        Quat<_Tp> crossProduct(const Quat<_Tp> &q) const;

        /**
         * @brief return the norm of quaternion
         * \f[||q|| = \sqrt{w^2 + x^2 + y^2 + z^2}\f]
         */
        _Tp norm() const;

        /**
         * @brief return a normalzed \f$q\f$
         * \f[q_n = \frac{q}{||q||}\f]
         * where \f$q_n.i\f$ satisfy \f$q_n.x^2 + q_n.y^2 + q_n.z^2 + q_n.w^2 = 1.\f$
         */
        Quat<_Tp> normalize() const;

        /**
         * @brief return an inverse \f$q q^-1\f$
         * which satisfies \f$q * q^-1 = 1\f$
         */
        template <typename T>
        friend Quat<T> inv(const Quat<T> &q1, bool assumeUnit);

        Quat<_Tp> inv(bool assumeUnit=false) const;

        /**
         * @brief \f$sinh(p) = sin(w)cos(||v||) + cosh(w)\frac{v}{||v||}sin||v||\f$
         */

        template <typename T>
        friend Quat<T> sinh(const Quat<T> &q1);

        Quat<_Tp> sinh() const;

        template <typename T>
        friend Quat<T> cosh(const Quat<T> &q1);

        Quat<_Tp> cosh() const;

        template <typename T>
        friend Quat<T> tanh(const Quat<T> &q1);

        Quat<_Tp> tanh() const;

        template <typename T>
        friend Quat<T> sin(const Quat<T> &q1);

        Quat<_Tp> sin() const;

        template <typename T>
        friend Quat<T> cos(const Quat<T> &q1);

        Quat<_Tp> cos() const;

        template <typename T>
        friend Quat<T> tan(const Quat<T> &q1);

        Quat<_Tp> tan() const;

        template <typename T>
        friend Quat<T> asin(const Quat<T> &q1);

        Quat<_Tp> asin() const;

        template <typename T>
        friend Quat<T> acos(const Quat<T> &q1);

        Quat<_Tp> acos() const;

        template <typename T>
        friend Quat<T> atan(const Quat<T> &q1);

        Quat<_Tp> atan() const;

        template <typename T>
        friend Quat<T> asinh(const Quat<T> &q1);

        Quat<_Tp> asinh() const;

        template <typename T>
        friend Quat<T> acosh(const Quat<T> &q1);

        Quat<_Tp> acosh() const;

        template <typename T>
        friend Quat<T> atanh(const Quat<T> &q1);

        Quat<_Tp> atanh() const;

        /**
         * @brief to dermined whether a quaternion is normalized or not
         */
        bool isNormal(_Tp esp=CV_QUAT_EPS) const;

        /**
         * @brief to throw an un-normalized error if its not an unit-quaternion
         */
        void assertNormal(_Tp esp=CV_QUAT_EPS) const;

        /**
         * @brief transform the quaternion q to a \f$3x3\f$ rotate matrix. The quaternion
         * has to be normalized before tranformation
         * matrix A which consists of n points
         * \f[\begin{bmatrix}
         * 	x0& x1& x2&...&xn\\
         * 	y0& y1& y2&...&yn\\
         * 	z0& z1& z2&...&zn
         * \end{bmatrix}\f]
         * where  A has \f$3\f$ rows and \f$n\f$ columns.
         * The points A can be rotated by
         * 			\f[toRotMat33() * A\f]
         * it returns a matrix which has
         * 3 rows and n coluns too
         */
        cv::Mat toRotMat3x3(bool assumeUnit=false) const;

        /**
         * @brief transform the quaternion \f$q\f$ to a \f$4x4\f$ rotate matrix
         *  n points matrix \f$A\f$ can be rotated by
         * 			\f[toRotMat4x4() * A\f]
         * where \f$A\f$ has 4 rows and n columns,it returns a matrix has
         * 4 rows and n coluns too
         * A has the form
         * \f[\begin{bmatrix}
         *  0&  0&  0&... &0 \\
         * 	x0& x1& x2&...&xn\\
         * 	y0& y1& y2&...&yn\\
         * 	z0& z1& z2&...&zn
         * \end{bmatrix}\f]
         */
        cv::Mat toRotMat4x4(bool assumeUnit=false) const;

        /**
         * @brief transoform the quaternion \f$q\f$ to a \f$Vec<T, 4>\f$
         */
        cv::Vec<_Tp, 4> toVec() const;

        /**
         * @brief transoform the queaternion to a rodrigues vector
         */
        cv::Vec<_Tp, 3> toRodrigues()  const;

        /**
        * @brief get the angle of quaternion
        * \f[\psi = 2 *arccos(\frac{w}{||q||})\f]
        */
        _Tp getAngle(bool assumeUnit=false) const;

        /**
        * @brief get the axis of quaternion
        * the unit axis \f$\boldsymbol{n}\f$ is defined by
        * \f[\begin{array}
        *      \boldsymbol{v} &= \boldsymbol{n} ||\boldsymbol{v}||\\
        *             &= \boldsymbol{n}||q||sin(\theta)
        *      \end{array}\f]
        */
        cv::Vec<_Tp, 3> getAxis(bool assumeUnit=false) const;

        /**
         * @brief return the dot between \f$q\f$ and \f$q1\f$
         * @param q a quaternion to be dot
         * \f[p \cdot q1 = w^2 + x^2 + y^2 + z^2\f]
         */
        _Tp dot(Quat<_Tp> q) const;

        /**
         * @brief To calculate the interpolation from \f$q_0\f$ and \f$q_1\f$ by Linear Interpolation(Lerp)
         * when \f$t = 0\f$, it returns \f$q_0\f$ and when \f$t= 1\f$, it returns \f$q_1\f$
         * \f$t\f$ should be ranged in \f$[0, 1]\f$ normally
         * @param q0 a quaternion used in linear interpolation
         * @param q1 a quaternion used in linear interpolation
         * @param t percent of vector \f$\overrightarrow{q_0q_1}\f$ between \f$q_0\f$ and \f$q_1\f$
         */
        static Quat<_Tp> lerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t);

        /**
         * @brief To calculate the interpolation from \f$q_0\f$ to \f$q_1\f$ by Normalized Linear Interpolation(Nlerp)
         * if assumeUnit=true, nlerp will not normalize the input.
         * it returns a normalized quaternion of Linear Interpolation(Lerp)
         * @param q0 a quaternion used in normalized linear interpolation
         * @param q1 a quaternion used in normalized linear interpolation
         * @param t pt percent of vector \f$\overrightarrow{q_0q_1}\f$
         * @param assumeUnit
         */
        static Quat<_Tp> nlerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, bool assumeUnit=false);

        /**
         @brief To calculate the interpolation between \f$q_0\f$ and \f$q_1\f$ by Spherical Linear Interpolation(Slerp).
         if \f$assumeUnit=true\f$, slerp will not normalize the input.
         it returns a normlized quaternion whether assumeUnit is true of false
         
         @param q0 a quaternion used in Slerp
         @param q1 a quaternion used in Slerp
         @param t percent of angle between \f$q_0\f$ and \f$q_1\f$
         @param assumeUnit true when \f$q_0\f$ and \f$q_1\f$ is unit quaternion, which represents the standard process of Slerp
         @param directChange
        */
        static Quat<_Tp> slerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t, bool assumeUnit=false, bool directChange=true);

        /**
         * @brief To calculate the interpolation between \f$q_0\f$,\f$q_1\f$,\f$q_2\f$,\f$q_3\f$  by Spherical and quadrangle(Squad).
         * Slerp uses a layer of quadratic interpolation nested with a layer of linear interpolation
         * @param q0 the first quaternion used in Squad
         * @param q1 the second quaternion used in Squad
         * @param q2 the third quaternion used in Squad
         * @param q3 thr fourth quaternion used in Squad
         * @param t in \f$[0, 1]\f$ interpolation parameter of quadratic and linear interpolation
         * @param assumeUnit true if all quaternions are unit
         * @param directChange
         */
        static Quat<_Tp> squad(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                               const Quat<_Tp> &q2, const Quat<_Tp> &q3,
                               const _Tp t, bool assumeUnit=false,
                               bool directChange=true);

        /**
         * @brief to calculate the intermedia quaternion between each three quaternion
         * @param q0 the first quaternion used in interPoint
         * @param q1 the second quaternion used in interPoint
         * @param q2 the third quaternion used in interPoint
         * @param assumeUnit true if all quaternions are unit
         */
        static Quat<_Tp> interPoint(const Quat<_Tp> &q0, const Quat<_Tp> &q1,
                                    const Quat<_Tp> &q2, bool assumeUnit=false);
        
        /**
         * @brief the spline curve is constructed by squad. The \f$C^1\f$ continuous
         * is composed of two quaternion \f$s_1\f$ and \f$s_2\f$, which can be calculated by
         * each three points by interPoint function. The \f$q_1\f$ and \f$q_2\f$ are curve
         * segment to be interpolated
         * @param q0
         * @param q1
         * @param q2
         * @param q3
         * @param t
         * @param assumeUnit
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
        template <typename S>
        friend Quat<S> cv::operator*(const S, const Quat<S>&);
        template <typename S>
        friend Quat<S> cv::operator*(const Quat<S>&, const S);
        template <typename S>
        friend std::ostream& cv::operator<<(std::ostream&, const Quat<S>&);

        _Tp w, x, y, z;
    };
    
    template <typename T>
    Quat<T> inv(const Quat<T> &q1, bool assumeUnit=false);

    template <typename T>
    Quat<T> sinh(const Quat<T> &q1);

    template <typename T>
    Quat<T> cosh(const Quat<T> &q1);

    template <typename T>
    Quat<T> tanh(const Quat<T> &q1);

    template <typename T>
    Quat<T> sin(const Quat<T> &q1);

    template <typename T>
    Quat<T> cos(const Quat<T> &q1);
    
    template <typename T>
    Quat<T> tan(const Quat<T> &q1);
    
    template <typename T>
    Quat<T> asinh(const Quat<T> &q1);

    template <typename T>
    Quat<T> acosh(const Quat<T> &q1);

    template <typename T>
    Quat<T> atanh(const Quat<T> &q1);

    template <typename T>
    Quat<T> asin(const Quat<T> &q1);

    template <typename T>
    Quat<T> acos(const Quat<T> &q1);

    template <typename T>
    Quat<T> atan(const Quat<T> &q1);
    /**
     * @brief
     * @param q
     * @param p
     * @param assumeUnit
     */
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

    template <typename S>
    Quat<S> operator*(const S, const Quat<S>&);

    template <typename S>
    Quat<S> operator*(const Quat<S>&, const S);

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
}//namespace
