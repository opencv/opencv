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
#ifndef OPENCV_CORE_DUALQUATERNION_HPP
#define OPENCV_CORE_DUALQUATERNION_HPP

#include <opencv2/core/quaternion.hpp>

namespace cv{
//! @addtogroup core
//! @{

template <typename _Tp> class DualQuat;
template <typename _Tp> std::ostream& operator<<(std::ostream&, const DualQuat<_Tp>&);

/**
 * Dual quaternion was introduced to describe translation and rotation while the quaternion can only
 * describe rotation. A unit dual quaternion can be classically represented as:
 * \f[
 * \begin{equation}
 * \begin{split}
 * \sigma &= \left(r+\frac{\epsilon}{2}tr\right)\\
 * &= [w, x, y, z, w\_, x\_, y\_, z\_]
 * \end{split}
 * \end{equation}
 * \f]
 * where \f$r, t\f$ represents the rotation (ordinary unit quaternion) and translation (pure ordinary quaternion) respectively.
 *
 * A more general dual quaternions who consist of two quaternions are usually represented in the form:
 * \f[
 * \sigma = p + \epsilon q
 * \f]
 * where the introduced dual unit \f$\epsilon\f$ satisfies \f$\epsilon^2 = \epsilon^3 =...=0\f$, and \f$p, q\f$ are all quaternions.
 *
 * Alternatively, dual quaternions can also be interpreted as four components which are all dual numbers:
 * \f[
 * \sigma = \hat{q}_w + \hat{q}_xi + \hat{q}_yj + \hat{q}_zk
 * \f]
 * If we set \f$\hat{q}_x, \hat{q}_y\f$ and \f$\hat{q}_z\f$ equal to 0, a dual quaternion is transformed to a dual number. see normalize().
 *
 * If you want to create a dual quaternion, you can use:
 *
 * ```
 * using namespace cv;
 * double angle = CV_PI;
 *
 * // create from eight number
 * DualQuatd dq1(1, 2, 3, 4, 5, 6, 7, 8); //p = [1,2,3,4]. q=[5,6,7,8]
 *
 * // create from Vec
 * Vec<double, 8> v{1,2,3,4,5,6,7,8};
 * DualQuatd dq_v{v};
 *
 * // create from two quaternion
 * Quatd p(1, 2, 3, 4);
 * Quatd q(5, 6, 7, 8);
 * DualQuatd dq2 = DualQuatd::createFromQuat(p, q);
 *
 * // create from an angle, an axis and a translation
 * Vec3d axis{0, 0, 1};
 * Quatd trans{0, 3, 4, 5};
 * DualQuatd dq3 = DualQuatd::createFromAngleAxisTrans(angle, axis, trans);
 *
 * // create from a linear transformation matrix R.
 * // see createFromMat() in detail for the form of R
 * Matx44d R = dq3.toMat();
 * DualQuatd dq4 = DualQuatd::createFromMat(R);
 *
 * // create from screw parameter.
 * Quatd qaxis{0, 0, 0, 1}; // that is [0, axis], qaxis should be normalized.
 * Quatd t{0, 3, 4 ,5};
 * Quatd moment = 1.0 / 2 * (t.crossProduct(qaxis) + qaxis.crossProduct(t.crossProduct(qaxis)) * std::cos(angle / 2) / std::sin(angle / 2));
 * double d = t.dot(qaxis);
 * DualQuatd dq5 = DualQuatd::createFromPitch(angle, d, qaxis, moment);
 * ```
 *
 * A point \f$v=(x, y, z)\f$ in form of dual quaternion is \f$[1+\epsilon v]=[1,0,0,0,0,x,y,z]\f$.
 * The transformation of a point \f$v_1\f$ to another point \f$v_1\f$ under the dual quaternion \f$\sigma\f$ is
 * \f[
 * 1 + \epsilon v_2 = \sigma * (1 + \epsilon v_1) * \sigma^{\star}
 * \f]
 * where \f$\sigma^{\star}=p^*-\epsilon q^*.\f$
 *
 * A line in the \f$Pl\ddot{u}cker\f$ \f$(\hat{l}, m)\f$ by the dual quaternion \f$l=\hat{l}+\epsilon m\f$.
 * To transform a line, \f[l_2 = \sigma * l_1 * \sigma^*,\f] where \f$\sigma=r+\frac{\epsilon}{2}rt\f$ and
 * \f$\sigma^*=p^*+\epsilon q^*\f$.
 *
 * To extract the Vec<double, 8> or Vec<float, 8>, see toVec();
 *
 * To extract the transformation matrix, see toMat();
 *
 * If there are two quaternions \f$q_0, q_1\f$ are needed to interpolate, you can use sclerp():
 * ```
 * Quatd::sclerp(q0, q1, t)
 * ```
 */
template <typename _Tp>
class DualQuat{
    static_assert(std::is_floating_point<_Tp>::value, "Dual quaternion only make sense with type of float or double");
    using value_type = _Tp;

public:
    static constexpr _Tp CV_DUAL_QUAT_EPS = (_Tp)1.e-6;

    DualQuat();

    /**
     * @brief create from eight same type number
     */
    DualQuat(const _Tp w, const _Tp x, const _Tp y, const _Tp z, const _Tp w_, const _Tp x_, const _Tp y_, const _Tp z_);

    /**
     * @brief create from a vector.
     */
    DualQuat(const Vec<_Tp, 8> &q);

    _Tp w, x, y, z, w_, x_, y_, z_;

    /**
     * @brief create Dual Quaternion from two same type quaternions p and q.
     * A Dual Quaternion \f$\sigma\f$ has the form:
     * \f[\sigma = p + \epsilon q\f]
     * where p and q are defined as:
     * \f[\begin{equation}
     *    \begin{split}
     *    p &= w + x\boldsymbol{i} + y\boldsymbol{j} + z\boldsymbol{k}\\
     *    q &= w\_ + x\_\boldsymbol{i} + y\_\boldsymbol{j} + z\_\boldsymbol{k}.
     *    \end{split}
     *   \end{equation}
     * \f]
     * The p and q are the real part and dual part respectively.
     * @param realPart a quaternion, real part of dual quaternion.
     * @param dualPart a quaternion, dual part of dual quaternion.
     * @sa Quat
    */
    static DualQuat<_Tp> createFromQuat(const Quat<_Tp> &realPart, const Quat<_Tp> &dualPart);

    /**
     * @brief create a dual quaternion from a rotation angle \f$\theta\f$, a rotation axis
     * \f$\boldsymbol{u}\f$ and a translation \f$\boldsymbol{t}\f$.
     * it generates a dual quaternion \f$\sigma\f$ in the form of
     * \f[\begin{equation}
     *    \begin{split}
     *    \sigma &= r + \frac{\epsilon}{2}\boldsymbol{t}r \\
     *           &= [\cos(\frac{\theta}{2}), \boldsymbol{u}\sin(\frac{\theta}{2})]
     *           + \frac{\epsilon}{2}[0, \boldsymbol{t}][[\cos(\frac{\theta}{2}),
     *           \boldsymbol{u}\sin(\frac{\theta}{2})]]\\
     *           &= \cos(\frac{\theta}{2}) + \boldsymbol{u}\sin(\frac{\theta}{2})
     *           + \frac{\epsilon}{2}(-(\boldsymbol{t} \cdot \boldsymbol{u})\sin(\frac{\theta}{2})
     *           + \boldsymbol{t}\cos(\frac{\theta}{2}) + \boldsymbol{u} \times \boldsymbol{t} \sin(\frac{\theta}{2})).
     *    \end{split}
     *    \end{equation}\f]
     * @param angle rotation angle.
     * @param axis a normalized rotation axis.
     * @param translation a pure quaternion.
     * @note axis will be normalized in this function. And translation is applied
     * after the rotation. Use createFromQuat(r, r * t / 2) to create a dual quaternion
     * which translation is applied before rotation.
     * @sa Quat
     */
    static DualQuat<_Tp> createFromAngleAxisTrans(const _Tp angle, const Vec<_Tp, 3> &axis, const Quat<_Tp> &translation);

    /**
     * @brief Transform this dual quaternion to a linear transformation matrix \f$M\f$.
     * Dual quaternion consists a rotation \f$r=[a,b,c,d]\f$ and a translation \f$t=[0,\Delta x,\Delta y,\Delta z]\f$. The transformation matrix \f$M\f$ has the form
     * \f[
     * \begin{bmatrix}
     * 1-2(e_2^2 +e_3^2) &2(e_1e_2-e_0e_3) &2(e_0e_2+e_1e_3) &\Delta x\\
     * 2(e_0e_3+e_1e_2)  &1-2(e_1^2+e_3^2) &2(e_2e_3-e_0e_1) &\Delta y\\
     * 2(e_1e_3-e_0e_2)  &2(e_0e_1+e_2e_3) &1-2(e_1^2-e_2^2) &\Delta z\\
     * 0&0&0&1
     * \end{bmatrix}
     * \f]
     *  if A is a matrix consisting of  n points to be transformed, this could be achieved by
     * \f[
     *  new\_A = M * A
     * \f]
     * where A has the form
     * \f[
     * \begin{bmatrix}
     * x_0& x_1& x_2&...&x_n\\
     * y_0& y_1& y_2&...&y_n\\
     * z_0& z_1& z_2&...&z_n\\
     * 1&1&1&...&1
     * \end{bmatrix}
     * \f]
     * where the same subscript represent the same point. The size of A should be \f$[4,n]\f$.
     * and the same size for matrix new_A.
     * @param _R 4x4 matrix that represents rotations and translation
     * @note Translation is applied after the rotation. Use createFromQuat(r, r * t / 2) to create
     * a dual quaternion which translation is applied before rotation.
     */
    static DualQuat<_Tp> createFromMat(InputArray _R);

    /**
     * @brief A dual quaternion is a vector in form of
     * \f[
     * \begin{equation}
     * \begin{split}
     * \sigma &=\boldsymbol{p} + \epsilon \boldsymbol{q}\\
     * &= \cos\hat{\frac{\theta}{2}}+\overline{\hat{l}}\sin\frac{\hat{\theta}}{2}
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$\hat{\theta}\f$ is dual angle and \f$\overline{\hat{l}}\f$ is dual axis:
     * \f[
     * \hat{\theta}=\theta + \epsilon d,\\
     * \overline{\hat{l}}= \hat{l} +\epsilon m.
     * \f]
     * In this representation, \f$\theta\f$ is rotation angle and \f$(\hat{l},m)\f$ is the screw axis, d is the translation distance along the axis.
     *
     * @param angle rotation angle.
     * @param d translation along the rotation axis.
     * @param axis rotation axis represented by quaternion with w = 0.
     * @param moment the moment of line.
     * @note Translation is applied after the rotation. Use createFromQuat(r, r * t / 2) to create
     * a dual quaternion which translation is applied before rotation.
     */
    static DualQuat<_Tp> createFromPitch(const _Tp angle, const _Tp d, const Quat<_Tp> &axis, const Quat<_Tp> &moment);

    /**
     * @brief return a quaternion which represent the real part of dual quaternion.
     * The definition of real part is in createFromQuat().
     * @sa createFromQuat, getDualQuat
     */
    Quat<_Tp> getRealQuat() const;

    /**
     * @brief return a quaternion which represent the dual part of dual quaternion.
     * The definition of dual part is in createFromQuat().
     * @sa createFromQuat, getRealQuat
     */
    Quat<_Tp> getDualQuat() const;

    /**
     * @brief return the conjugate of a dual quaternion.
     * \f[
     * \begin{equation}
     * \begin{split}
     * \sigma^* &= (p + \epsilon q)^*
     *          &= (p^* + \epsilon q^*)
     * \end{split}
     * \end{equation}
     * \f]
     */
    DualQuat<_Tp> conjugate() const;

    /**
     * @brief return the rotation in quaternion form.
     */
    Quat<_Tp> getRotation(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the translation in pure quaternion form.
     * The rotation \f$r\f$ in this dual quaternion \f$\sigma\f$ is applied before translation \f$t\f$,
     * which has the form
     * \f[\begin{equation}
     * \begin{split}
     * \sigma &= p + \epsilon q \\
     *        &= r + \frac{\epsilon}{2}\boldsymbol{t}r.
     * \end{split}
     * \end{equation}\f]
     * Thus, the translation can be obtained as:
     * \f[\boldsymbol{v} = 2qp^*.\f]
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion
     * and this function will save some computations.
     * @note this dual quaternion's translation is applied after the rotation.
     */
    Quat<_Tp> getTranslation(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the norm \f$||A||\f$ of dual quaternion \f$A = p + \epsilon q\f$.
     * \f[
     *  \begin{equation}
     *  \begin{split}
     *  ||A|| &= \sqrt{A * A^*} \\
     *        &= ||p|| + \epsilon \frac{p \cdot q}{||p||}.
     *  \end{split}
     *  \end{equation}
     *  \f]
     * Generally speaking, the norm of a not unit dual
     * quaternion is a dual number. For convenience, we return it in the form of a dual quaternion
     * , i.e.
     * \f[ ||A|| = [||p||, 0, 0, 0, \frac{p \cdot q}{||p||}, 0, 0, 0].\f]
     *
     * @note The data type of dual number is dual quaternion.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion and this f, this dual quaternion
     */
    DualQuat<_Tp> norm(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return a normalized dual quaternion.
     * A dual quaternion can be expressed as
     * \f[
     * \begin{equation}
     * \begin{split}
     * \sigma &= p + \epsilon q\\
     * &=||\sigma||\left(r+\frac{1}{2}tr\right)
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$r, t\f$ represents the rotation (ordinary quaternion) and translation (pure ordinary quaternion) respectively,
     * and \f$||\sigma||\f$ is the norm of dual quaternion(a dual number).
     * A dual quaternion is unit if and only if
     * \f[
     * ||p||=1, p \cdot q=0
     * \f]
     * where \f$\cdot\f$ means dot product.
     * The process of normalization is
     * \f[
     * \sigma_{u}=\frac{\sigma}{||\sigma||}
     * \f]
     * Next, we simply proof \f$\sigma_u\f$ is a unit dual quaternion:
     * \f[
     * \renewcommand{\Im}{\operatorname{Im}}
     * \begin{equation}
     * \begin{split}
     * \sigma_{u}=\frac{\sigma}{||\sigma||}&=\frac{p + \epsilon q}{||p||+\epsilon\frac{p\cdot q}{||p||}}\\
     * &=\frac{p}{||p||}+\epsilon\left(\frac{q}{||p||}-p\frac{p\cdot q}{||p||^3}\right)\\
     * &=\frac{p}{||p||}+\epsilon\frac{1}{||p||^2}\left(qp^{*}-p\cdot q\right)\frac{p}{||p||}\\
     * &=\frac{p}{||p||}+\epsilon\frac{1}{||p||^2}\Im(qp^*)\frac{p}{||p||}.\\
     * \end{split}
     * \end{equation}
     * \f]
     * As expected, the real part is a rotation and dual part is a pure quaternion.
     */
    DualQuat<_Tp> normalize() const;

    /**
     * @brief if \f$\sigma = p + \epsilon q\f$ is a dual quaternion, p is not zero,
     * then the inverse dual quaternion is
     * \f[\sigma^{-1} = \frac{\sigma^*}{||\sigma||^2}, \f]
     * or equivalentlly,
     * \f[\sigma^{-1} = p^{-1} - \epsilon p^{-1}qp^{-1}.\f]
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion
     * and this function will save some computations.
     */
    DualQuat<_Tp> inv(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief return the dot product of two dual quaternion
     * @param p other dual quaternion.
     */
    _Tp dot(DualQuat<_Tp> p) const;

    /**
    *
     * A convenience form of power function of dual quaternion can be expressed as:
     * \f[
     * \sigma^t = \cos\hat{\frac{t\theta}{2}}+\overline{\hat{l}}\sin\frac{\hat{t\theta}}{2}
     * \f]
     * Obviously,  this operation keeps the same screw axis and scales with both rotation angle and translation distance.
     *
     * @param t index of power function.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion
     * and this function will save some computations.
     */
    DualQuat<_Tp> power(const _Tp t, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief It returns the value of \f$p^q\f$ where p and q are dual quaternion.
     * This could be calculated as:
     * \f[
     * p^q = \exp(q\ln p)
     * \f]
     *
     * @param q a dual quaternion
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion
     * and this function will save some computations.
     */
    DualQuat<_Tp> power(const DualQuat<_Tp>& q, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief A dual quaternion is a vector in form of
     * \f[
     * \begin{equation}
     * \begin{split}
     * \sigma &=\boldsymbol{p} + \epsilon \boldsymbol{q}\\
     * &= [\hat{q}_0,\hat{q}_1,\hat{q}_2,\hat{q}_3]\\
     * &= [\hat{q}_0,\boldsymbol{v}]
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$\hat{q}_i = p_i+\epsilon q_i\f$. \f$p_i, q_i\f$ is the element of \f$\boldsymbol{p},\boldsymbol{q}\f$ respectively.
     *
     * Thus, the exponential function of a dual quaternion can be calculated as the method of a quaternion:
     * \f[
     * \exp(\sigma)=e^{\hat{q_0}}\left(\cos||\boldsymbol{v}||+\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\sin||\boldsymbol{v}||\right)
     * \f]
     * To calculate \f$e^{\hat{q_0}}\f$, we expand \f$e^{\hat{q_0}}\f$ by Taylor series:
     * \f[
     * \begin{equation}
     * \begin{split}
     * e^{\hat{q_0}} &= e^{p_0+\epsilon q_0}\\
     * &=e^{p_0}+\epsilon q_0e^{p_0}
     * \end{split}
     * \end{equation}
     * \f]
     * and the same operations for \f$\cos\f$ and \f$\sin.\f$
     */
    DualQuat<_Tp> exp() const;

    /**
     * @brief
     * A dual quaternion is a vector in form of
     * \f[
     * \begin{equation}
     * \begin{split}
     * \sigma &=\boldsymbol{p} + \epsilon \boldsymbol{q}\\
     * &= [\hat{q}_0,\hat{q}_1,\hat{q}_2,\hat{q}_3]\\
     * &= [\hat{q}_0,\boldsymbol{v}]
     * \end{split}
     * \end{equation}
     * \f]
     * where \f$\hat{q}_i = p_i+\epsilon q_i\f$. \f$p_i, q_i\f$ is the element
     * of \f$\boldsymbol{p},\boldsymbol{q}\f$ respectively.
     *
     * Thus, the logarithm function of a dual quaternion can be calculated as the method of a quaternion:
     * \f[
     * \ln(\sigma)=\ln||\sigma||+\frac{\boldsymbol{v}}{||\boldsymbol{v}||}\arccos\frac{\hat{q}_0}{\boldsymbol{||v||}}
     * \f]
     * To calculate each function, we expand them by Taylor series, see exp for example.
     *
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion assume to be a unit quaternion
     * and this function will save some computations.
     */
    DualQuat<_Tp> log(QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT) const;

    /**
     * @brief Transform dual quaternion to a vector.
     */
    Vec<_Tp, 8> toVec() const;

    /**
     * @brief Transform dual quaternion to a linear transformation matrix
     * the form of matrix, see createFromMat().
     */
    Matx<_Tp, 4, 4> toMat() const; //name may not proper

    /**
     * @brief The screw linear interpolation(ScLERP) is an extension of spherical linear interpolation of quaternion.
     * If \f$\sigma_1\f$ and \f$\sigma_2\f$ are two dual quaternion representing the initial and final pose.
     * The interpolation of ScLERP function can be defined as:
     * \f[
     * ScLERP(t;\sigma_1,\sigma_2) = \sigma_1 * (\sigma_1^{-1} * \sigma_2)^t, t\in[0,1]
     * \f]
     *
     * @param q1 a dual quaternion represents a initial pose.
     * @param q2 a dual quaternion represents a final pose.
     * @param t interpolation parameter
     * @param directChange if true, it will always return the shortest path.
     * @param assumeUnit if QUAT_ASSUME_UNIT, this dual quaternion  assume to be a unit quaternion
     * and this function will save some computations.
     *
     * For example
     * ```
     * double angle1 = CV_PI / 2;
     * Vec3d axis{0, 0, 1};
     * Quatd t(0, 0, 0, 3);
     * DualQuatd initial = DualQuatd::createFromAngleAxisTrans(angle1, axis, t);
     * double angle2 = CV_PI;
     * DualQuatd final = DualQuatd::createFromAngleAxisTrans(angle2, axis, t);
     * DualQuatd inter = DualQuatd::sclerp(initial, final, 0.5);
     * ```
     */
    static DualQuat<_Tp> sclerp(const DualQuat<_Tp> &q1, const DualQuat<_Tp> &q2, const _Tp t,
                                bool directChange=true, QuatAssumeType assumeUnit=QUAT_ASSUME_NOT_UNIT);

    /**
     * @brief Return opposite dual quaternion \f$-p\f$
     * which satisfies \f$p + (-p) = 0.\f$
     *
     * For example
     * ```
     * DualQuatd q{1, 2, 3, 4, 5, 6, 7, 8};
     * std::cout << -q << std::endl; // [-1, -2, -3, -4, -5, -6, -7, -8]
     * ```
     */
    DualQuat<_Tp> operator-() const;

    /**
     * @brief return true if two dual quaternions p and q are nearly equal, i.e. when the absolute
     * value of each \f$p_i\f$ and \f$q_i\f$ is less than CV_DUAL_QUAT_EPS.
     */
    bool operator==(const DualQuat<_Tp>&) const;

     /**
     * @brief Subtraction operator of two dual quaternions p and q.
     * It returns a new dual quaternion that each value is the sum of \f$p_i\f$ and \f$-q_i\f$.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * std::cout << p - q << std::endl; //[-4, -4, -4, -4, 4, -4, -4, -4]
     * ```
     */
    DualQuat<_Tp> operator-(const DualQuat<_Tp>&) const;

    /**
     * @brief Subtraction assignment operator of two dual quaternions p and q.
     * It subtracts right operand from the left operand and assign the result to left operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * p -= q; // equivalent to p = p - q
     * std::cout << p << std::endl; //[-4, -4, -4, -4, 4, -4, -4, -4]
     *
     * ```
     */
    DualQuat<_Tp>& operator-=(const DualQuat<_Tp>&);

    /**
     * @brief Addition operator of two dual quaternions p and q.
     * It returns a new dual quaternion that each value is the sum of \f$p_i\f$ and \f$q_i\f$.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * std::cout << p + q << std::endl; //[6, 8, 10, 12, 14, 16, 18, 20]
     * ```
     */
    DualQuat<_Tp> operator+(const DualQuat<_Tp>&) const;
    /**
     * @brief Addition assignment operator of two dual quaternions p and q.
     * It adds right operand to the left operand and assign the result to left operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * p += q; // equivalent to p = p + q
     * std::cout << p << std::endl; //[6, 8, 10, 12, 14, 16, 18, 20]
     *
     * ```
     */
    DualQuat<_Tp>& operator+=(const DualQuat<_Tp>&);

    /**
     * @brief Multiplication assignment operator of two quaternions.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of dual quaternion multiplication:
     * The dual quaternion can be written as an ordered pair of quaternions [A, B]. Thus
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * q &= [A, B][C, D]]\\
     * &=[AC, AD + BC]
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * p *= q;
     * std::cout << p << std::endl; //[-60, 12, 30, 24, -216, 80, 124, 120]
     * ```
     */
    DualQuat<_Tp>& operator*=(const DualQuat<_Tp>&);

    /**
     * @brief Multiplication assignment operator of a quaternions and a scalar.
     * It multiplies right operand with the left operand and assign the result to left operand.     *
     *
     * Rule of dual quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z, w\_, x\_, y\_, z\_] * s\\
     * &=[w * s, x * s, y * s, z * s, w\_ * s, x\_ * s, y\_ * s, z\_ * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double s = 2.0;
     * p *= s;
     * std::cout << p << std::endl; //[2, 4, 6, 8, 10, 12, 14, 16]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    DualQuat<_Tp> operator*=(const _Tp s);


    /**
     * @brief Multiplication operator of two dual quaternions q and p.
     * Multiplies values on either side of the operator.
     *
     * Rule of dual quaternion multiplication:
     * The dual quaternion can be written as an ordered pair of quaternions [A, B]. Thus
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * q &= [A, B][C, D]]\\
     * &=[AC, AD + BC]
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * std::cout << p * q << std::endl; //[-60, 12, 30, 24, -216, 80, 124, 120]
     * ```
     */
    DualQuat<_Tp> operator*(const DualQuat<_Tp>&) const;

    /**
     * @brief Division operator of a dual quaternions and a scalar.
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of dual quaternion division with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / s &= [w, x, y, z, w\_, x\_, y\_, z\_] / s\\
     * &=[w/s, x/s, y/s, z/s, w\_/s, x\_/s, y\_/s, z\_/s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double s = 2.0;
     * p /= s; // equivalent to p = p / s
     * std::cout << p << std::endl; //[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
     * ```
     * @note the type of scalar should be equal to this dual quaternion.
     */
    DualQuat<_Tp> operator/(const _Tp s) const;

    /**
     * @brief Division operator of two dual quaternions p and q.
     * Divides left hand operand by right hand operand.
     *
     * Rule of dual quaternion division with a dual quaternion:
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
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * std::cout << p / q << std::endl; // equivalent to p * q.inv()
     * ```
     */
    DualQuat<_Tp> operator/(const DualQuat<_Tp>&) const;

    /**
     * @brief Division assignment operator of two dual quaternions p and q;
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of dual quaternion division with a quaternion:
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
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * DualQuatd q{5, 6, 7, 8, 9, 10, 11, 12};
     * p /= q; // equivalent to p = p * q.inv()
     * std::cout << p << std::endl;
     * ```
     */
    DualQuat<_Tp>& operator/=(const DualQuat<_Tp>&);

    /**
     * @brief Division assignment operator of a dual quaternions and a scalar.
     * It divides left operand with the right operand and assign the result to left operand.
     *
     * Rule of dual quaternion division with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p / s &= [w, x, y, z, w\_, x\_, y\_ ,z\_] / s\\
     * &=[w / s, x / s, y / s, z / s, w\_ / s, x\_ / s, y\_ / s, z\_ / s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double s = 2.0;;
     * p /= s; // equivalent to p = p / s
     * std::cout << p << std::endl; //[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    Quat<_Tp>& operator/=(const _Tp s);

    /**
     * @brief Addition operator of a scalar and a dual quaternions.
     * Adds right hand operand from left hand operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double scalar = 2.0;
     * std::cout << scalar + p << std::endl; //[3.0, 2, 3, 4, 5, 6, 7, 8]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator+(const T s, const DualQuat<T>&);

    /**
     * @brief Addition operator of a dual quaternions and a scalar.
     * Adds right hand operand from left hand operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double scalar = 2.0;
     * std::cout << p + scalar << std::endl; //[3.0, 2, 3, 4, 5, 6, 7, 8]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator+(const DualQuat<T>&, const T s);

    /**
     * @brief Multiplication operator of a scalar and a dual quaternions.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of dual quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z, w\_, x\_, y\_, z\_] * s\\
     * &=[w * s, x * s, y * s, z * s, w\_ * s, x\_ * s, y\_ * s, z\_ * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double s = 2.0;
     * std::cout << s * p << std::endl; //[2, 4, 6, 8, 10, 12, 14, 16]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator*(const T s, const DualQuat<T>&);

    /**
     * @brief Subtraction operator of a dual quaternion and a scalar.
     * Subtracts right hand operand from left hand operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double scalar = 2.0;
     * std::cout << p - scalar << std::endl; //[-1, 2, 3, 4, 5, 6, 7, 8]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator-(const DualQuat<T>&, const T s);

    /**
     * @brief Subtraction operator of a scalar and a dual quaternions.
     * Subtracts right hand operand from left hand operand.
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double scalar = 2.0;
     * std::cout << scalar - p << std::endl; //[1.0, -2, -3, -4, -5, -6, -7, -8]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator-(const T s, const DualQuat<T>&);

    /**
     * @brief Multiplication operator of a dual quaternions and a scalar.
     * It multiplies right operand with the left operand and assign the result to left operand.
     *
     * Rule of dual quaternion multiplication with a scalar:
     * \f[
     * \begin{equation}
     * \begin{split}
     * p * s &= [w, x, y, z, w\_, x\_, y\_, z\_] * s\\
     * &=[w * s, x * s, y * s, z * s, w\_ * s, x\_ * s, y\_ * s, z\_ * s].
     * \end{split}
     * \end{equation}
     * \f]
     *
     * For example
     * ```
     * DualQuatd p{1, 2, 3, 4, 5, 6, 7, 8};
     * double s = 2.0;
     * std::cout << p * s << std::endl; //[2, 4, 6, 8, 10, 12, 14, 16]
     * ```
     * @note the type of scalar should be equal to the dual quaternion.
     */
    template <typename T>
    friend DualQuat<T> cv::operator*(const DualQuat<T>&, const T s);

    template <typename S>
    friend std::ostream& cv::operator<<(std::ostream&, const DualQuat<S>&);

};

using DualQuatd = DualQuat<double>;
using DualQuatf = DualQuat<float>;

//! @} core
}//namespace

#include "dualquaternion.inl.hpp"

#endif /* OPENCV_CORE_QUATERNION_HPP */
