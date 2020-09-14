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
//         Liangqian Kong <chargerKong@126.com>#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <iostream>
namespace cv 
{
    /**
     * Quaternion are a number system that extends the complex numbers.
     * A quaternion is generally represented in the form:
     * 		\begin{array}{rcl}
     * 		q &= w + x\boldsymbol{i} + y\boldsymbol{j} + z\boldsymbol{k}\\
     *		  &= [w, x, y, z]\\
     *		  &= [w, \boldsymbol{v}]
     *		  \end{array}
     * A unit quaternion is usually represents rotation, which has the form:
     * 		q = [cos\theta, sin\theta\cdot u_x, sin\theta\cdot u_y, sin\theta\cdot u_z]
     *
     * In fact. A general quaternion could also be represented in the form
     * 		q = ||q||[cos\theta, sin\theta\cdot u_x, sin\theta\cdot u_y, sin\theta\cdot u_z]
     * where ||q|| represents the norm of q
     * all functions will be degenerate to unit cases when ||q|| = 1
     */

    template <typename _Tp>
    class Quat
    {
    public:
        Quat() = default;
        explicit Quat(const cv::Vec<_Tp, 4> &coeff);
        Quat(_Tp qw, _Tp qx, _Tp qy, _Tp qz);
        Quat(const _Tp angle, const cv::Vec<_Tp, 3> &axis);
        explicit Quat(const cv::Mat &R);


        /**
         * @brief get the conjugate of quaternion
         * q^* = (w, -x, -y, -z)
         */
        Quat<_Tp> conjugate() const;

        /**
         * @brief return the value of exponential function
         * e^q = e^w (cos||v||+ \frac{v}{||v||sin||v||})
         */
        template <typename T>
        friend Quat<T> exp(const Quat<T> &q);

        Quat<_Tp> exp() const;

        /**
         * @brief return the value of logarithm function
         * ln(q) = ln||q|| + \frac{v}{||v||}arccos\frac{a}{||q||}
         */
        template <typename T>
        friend Quat<T> log(const Quat<T>& /*q*/);

        Quat<_Tp> log() const;

        /**
         * @brief return the value of power function with constant x
         * q^x = ||q||(cos(x\theta) + \boldsymbol{v}sin(x\theta)))
         */
        template <typename T>
        friend Quat<T> power(const Quat<T>& q, T x);

        Quat<_Tp> power(_Tp x) const;
        /**
         * @brief return thr \sqrt{q}
         */
        template <typename T>
        friend Quat<T> sqrt(const Quat<T>& q);

        Quat<_Tp> sqrt() const;
        /**
         * @brief return the value of power function with quaternion p
         * q^p = e^{pln(q)}
         */
        template <typename T>
        friend Quat<T> power(const Quat<T> &q, const Quat<T> &p);

        Quat<_Tp> power(const Quat<_Tp> &p) const;
        /**
         * @ brief return the crossProduct between q and q1
         */
        template <typename T>
        friend Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);

        Quat<_Tp> crossProduct(const Quat<_Tp> &q) const;

        /**
         * @brief return the norm of quaternion
         * ||q|| = \sqrt{w^2 + x^2 + y^2 + z^2}
         */
        _Tp norm() const;

        /**
         * @brief return a normalzed q
         * q_n = \frac{q}{||q||}
         * where q_n.i satisfy q_n.x^2 + q_n.y^2 + q_n.z^2 + q_n.w^2 = 1
         */
        Quat<_Tp> normalize() const;

        /**
         * @brief return an inverse q q^-1
         * which satisfies q * q^-1 = 1
         */
        template <typename T>
        friend Quat<T> inv(const Quat<T> &q1);
        Quat<_Tp> inv() const;

        /**
         * @brief sinh(p) = sin(w)cos(||v||) + cosh(w)\frac{v}{||v||}sin||v||
         */
        /*
        template <typename T>
        friend Quat<T> sinh(const Quat<T> &q1);
        */
        
	Quat<_Tp> sinh() const;

/*
        template <typename T>
        friend Quat<T> cosh(const Quat<T> &q1);
*/

        Quat<_Tp> cosh() const;

/*
        template <typename T>
        friend Quat<T> tanh(const Quat<T> &q1);
*/

        Quat<_Tp> tanh() const;

/*
        template <typename T>
        friend Quat<T> sin(const Quat<T> &q1);
*/

        Quat<_Tp> sin() const;

/*
        template <typename T>
        friend Quat<T> cos(const Quat<T> &q1);
*/

        Quat<_Tp> cos() const;

/*
        template <typename T>
        friend Quat<T> tan(const Quat<T> &q1);
*/

        Quat<_Tp> tan() const;

/*
        template <typename T>
        friend Quat<T> asin(const Quat<T> &q1);
*/

        Quat<_Tp> asin() const;

/*
        template <typename T>
        friend Quat<T> acos(const Quat<T> &q1);
*/

        Quat<_Tp> acos() const;

/*
        template <typename T>
        friend Quat<T> atan(const Quat<T> &q1);
*/

        Quat<_Tp> atan() const;
/*
        template <typename T>
        friend Quat<T> asinh(const Quat<T> &q1);
*/
        Quat<_Tp> asinh() const;
/*
        template <typename T>
        friend Quat<T> acosh(const Quat<T> &q1);
*/
        Quat<_Tp> acosh() const;

        //template <typename T>
        //friend Quat<T> atanh(const Quat<T> &q1);

        Quat<_Tp> atanh() const;

        /**
         * @brirf to dermined whether a quaternion is normalized or not
         */
        //bool isNormal(_Tp esp=defaultVa) const;
        bool isNormal() const;

        /**
         * @brief to throw an un-normalized error if its not an unit-quaternion
         */
        void assertNormal() const;

        /**
         * @brief transform the quaternion q to a 3x3 rotate matrix. The quaternion
         * has to be normalized before tranformation
         * n points matrix A
         * \begin{bmatrix}
         * 	x0& x1& x2&...&xn\\
         * 	y0& y1& y2&...&yn\\
         * 	z0& z1& z2&...&zn
         * \end{bmatrix}
         * where  A has 3 rows and n columns.
         * The points A can be rotated by
         * 			toRotMat33() * A
         * it returns a matrix has
         * 3 rows and n coluns too
         */
        cv::Mat toRotMat3x3() const;

        /**
         * @brief transform the quaternion q to a 4x4 rotate matrix
         *  n points matrix A can be rotated by
         * 			toRotMat4x4() * A
         * where A has 4 rows and n columns,it returns a matrix has
         * 4 rows and n coluns too
         * A has the form
         * \begin{bmatrix}
         *  0&  0&  0&... &0 \\
         * 	x0& x1& x2&...&xn\\
         * 	y0& y1& y2&...&yn\\
         * 	z0& z1& z2&...&zn
         * \end{bmatrix}

         */
        cv::Mat toRotMat4x4() const;

        /**
         * @brief transofrm the quaternion q to a Vec<T, 4>
         */
        cv::Vec<_Tp, 4> toVec() const;

        /**
        * @brief get the angle of quaternion
        * \psi = 2 *arccos(\frac{w}{||q||})
        */
        //_Tp getAngle(bool assumeUnit=false) const;
        _Tp getAngle() const;

        /**
        * @brief get the axis of quaternion
        * the unit axis \boldsymbol{n} is defined by
        * \begin{array}
        *      \boldsymbol{v} &= \boldsymbol{n} ||\boldsymbol{v}||\\
        *             &= \boldsymbol{n}||q||sin(\theta)
        *      \end{array}
        */
        cv::Vec<_Tp, 3> getAxis() const;

        /**
         * @brief return the dot between q and q1
         * @param q1 a quaternion to be dot
         * p \cdot q1 = w^2 + x^2 + y^2 + z^2
         */
        _Tp dot(Quat<_Tp> q1) const;

        /**
         * @brief To calculate the interpolation between q_0 and q_1 by Linear Interpolation(Lerp)
         * when t = 0, it returns q_0 and when t= 1, it returns q_1
         * t should be ranged in [0, 1] normally
         * @param q_0 a quaternion used in linear interpolation
         * @param q_1 a quaternion used in linear interpolation
         * @param t percent of vector \overrightarrow{q_0q_1} between q0 and q1
         */
        static Quat<_Tp> lerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t);

        /**
         * @brief To calculate the interpolation between q_0 and q_1 by Normalized Linear Interpolation(Nlerp)
         * it returns a normalized quaternion of Linear Interpolation(Lerp)
         * @param q_0 a quaternion used in normalized linear interpolation
         * @param q_1 a quaternion used in normalized linear interpolation
         * @param t percent of vector \overrightarrow{q_0q_1}
         */
        static Quat<_Tp> nlerp(const Quat<_Tp> &q0, const Quat &q1, const _Tp t);

        /**
         * @brief To calculate the interpolation between q_0 and q_1 by Spherical Linear Interpolation(Slerp)
         * it returns a normlized quaternion whether assumeUnit is true of false
         * @param q_0 a quaternion used in Slerp
         * @param q_1 a quaternion used in Slerp
         * @param t percent of angle between q_0 and q_1
         * @param assumeUnit true when q_0 and q_1 is unit quaternion, which represents the standard process of Slerp
        */
        static Quat<_Tp> slerp(Quat<_Tp> &q0, Quat &q1, const _Tp t, bool assumeUnit=false);

        /**
         * @brief To calculate the interpolation between q_0, q_1, q_2, q_3 by Spherical and quadrangle(Squad).
         * Slerp uses a layer of quadratic interpolation nested with a layer of linear interpolation
         * @param q_0 the first quaternion used in Squad
         * @param q_1 the second quaternion used in Squad
         * @param q_2 the third quaternion used in Squad
         * @param q_3 thr fourth quaternion used in Squad
         * @param t t in [0, 1] interpolation parameter of quadratic and linear interpolation
         * @param assumeUnit true if all quaternions are unit
         */
        static Quat<_Tp> squad(Quat<_Tp> &q0, Quat<_Tp> &q1,
                               Quat<_Tp> &q2, Quat<_Tp> &q3,
                               const _Tp t, bool assumeUnit=false);

        /**
         * @brief to calculate the intermedia quaternion between each three quaternion
         * @param q_0 the first quaternion used in interPoint
         * @param q_1 the second quaternion used in interPoint
         * @param q_2 the third quaternion used in interPoint
         * @param assumeUnit true if all quaternions are unit
         */
        static Quat<_Tp> interPoint(Quat<_Tp> &q0, Quat<_Tp> &q1,
                                    Quat<_Tp> &q2, bool assumeUnit=false);
        /**
         * @brief the spline curve is constructed by squad. The C^1 continuous
         * is composed of two quaternion s1 and s2, which can be calculated by
         * each three points by interPoint function. The q1 and q2 are curve
         * segment to be interpolated
         */
        static Quat<_Tp> spline(Quat<_Tp> &q0, Quat<_Tp> &q1,
                                Quat<_Tp> &q2, Quat<_Tp> &q3,
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
    Quat<T> inv(const Quat<T> &q1);
/*
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
    Quat<T> asinh(const Quat<T> &q1);

    template <typename T>
    Quat<T> tan(const Quat<T> &q1);

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

    template <typename T>
    Quat<T> power(const Quat<T> &q, const Quat<T> &p);

    template <typename T>
    Quat<T> exp(const Quat<T> &q);

    template <typename T>
    Quat<T> log(const Quat<T> &q);

    template <typename T>
    Quat<T> power(const Quat<T>& q, T x);

    template <typename T>
    Quat<T> crossProduct(const Quat<T> &p, const Quat<T> &q);
*/
    template <typename S>
    Quat<S> operator*(const S, const Quat<S>&);

    template <typename S>
    Quat<S> operator*(const Quat<S>&, const S);

    template <typename S>
    std::ostream& operator<<(std::ostream&, const Quat<S>&);

}//namespace
