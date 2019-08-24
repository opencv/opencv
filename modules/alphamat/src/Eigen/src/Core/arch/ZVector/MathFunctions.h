// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin, cos, exp, and log functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_ALTIVEC_H
#define EIGEN_MATH_FUNCTIONS_ALTIVEC_H

namespace Eigen {

namespace internal {

static _EIGEN_DECLARE_CONST_Packet2d(1 , 1.0);
static _EIGEN_DECLARE_CONST_Packet2d(2 , 2.0);
static _EIGEN_DECLARE_CONST_Packet2d(half, 0.5);

static _EIGEN_DECLARE_CONST_Packet2d(exp_hi,  709.437);
static _EIGEN_DECLARE_CONST_Packet2d(exp_lo, -709.436139303);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_LOG2EF, 1.4426950408889634073599);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p0, 1.26177193074810590878e-4);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p1, 3.02994407707441961300e-2);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p2, 9.99999999999999999910e-1);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q0, 3.00198505138664455042e-6);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q1, 2.52448340349684104192e-3);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q2, 2.27265548208155028766e-1);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q3, 2.00000000000000000009e0);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C1, 0.693145751953125);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C2, 1.42860682030941723212e-6);

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d& _x)
{
  Packet2d x = _x;

  Packet2d tmp, fx;
  Packet2l emm0;

  // clamp x
  x = pmax(pmin(x, p2d_exp_hi), p2d_exp_lo);
  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(p2d_cephes_LOG2EF, x, p2d_half);

  fx = vec_floor(fx);

  tmp = pmul(fx, p2d_cephes_exp_C1);
  Packet2d z = pmul(fx, p2d_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  Packet2d x2 = pmul(x,x);

  Packet2d px = p2d_cephes_exp_p0;
  px = pmadd(px, x2, p2d_cephes_exp_p1);
  px = pmadd(px, x2, p2d_cephes_exp_p2);
  px = pmul (px, x);

  Packet2d qx = p2d_cephes_exp_q0;
  qx = pmadd(qx, x2, p2d_cephes_exp_q1);
  qx = pmadd(qx, x2, p2d_cephes_exp_q2);
  qx = pmadd(qx, x2, p2d_cephes_exp_q3);

  x = pdiv(px,psub(qx,px));
  x = pmadd(p2d_2,x,p2d_1);

  // build 2^n
  emm0 = vec_ctsl(fx, 0);

  static const Packet2l p2l_1023 = { 1023, 1023 };
  static const Packet2ul p2ul_52 = { 52, 52 };

  emm0 = emm0 + p2l_1023;
  emm0 = emm0 << reinterpret_cast<Packet2l>(p2ul_52);

  // Altivec's max & min operators just drop silent NaNs. Check NaNs in 
  // inputs and return them unmodified.
  Packet2ul isnumber_mask = reinterpret_cast<Packet2ul>(vec_cmpeq(_x, _x));
  return vec_sel(_x, pmax(pmul(x, reinterpret_cast<Packet2d>(emm0)), _x),
                 isnumber_mask);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f& x)
{
  Packet4f res;
  res.v4f[0] = pexp<Packet2d>(x.v4f[0]);
  res.v4f[1] = pexp<Packet2d>(x.v4f[1]);
  return res;
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d psqrt<Packet2d>(const Packet2d& x)
{
  return  __builtin_s390_vfsqdb(x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psqrt<Packet4f>(const Packet4f& x)
{
  Packet4f res;
  res.v4f[0] = psqrt<Packet2d>(x.v4f[0]);
  res.v4f[1] = psqrt<Packet2d>(x.v4f[1]);
  return res;
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d prsqrt<Packet2d>(const Packet2d& x) {
  // Unfortunately we can't use the much faster mm_rqsrt_pd since it only provides an approximation.
  return pset1<Packet2d>(1.0) / psqrt<Packet2d>(x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f prsqrt<Packet4f>(const Packet4f& x) {
  Packet4f res;
  res.v4f[0] = prsqrt<Packet2d>(x.v4f[0]);
  res.v4f[1] = prsqrt<Packet2d>(x.v4f[1]);
  return res;
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_ALTIVEC_H
