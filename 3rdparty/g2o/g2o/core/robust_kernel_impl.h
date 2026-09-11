// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_ROBUST_KERNEL_IMPL_H
#define G2O_ROBUST_KERNEL_IMPL_H

#include "robust_kernel.h"
#include "g2o_core_api.h"

namespace g2o {

  /**
   * \brief scale a robust kernel to another delta (window size)
   *
   * Scales a robust kernel to another window size. Useful, in case if
   * one implements a kernel which only is designed for a fixed window
   * size.
   */
  class G2O_CORE_API RobustKernelScaleDelta : public RobustKernel
  {
    public:
      /**
       * construct the scaled kernel ontop of another kernel which might be shared accross
       * several scaled kernels
       */
      explicit RobustKernelScaleDelta(const RobustKernelPtr& kernel, number_t delta = 1.);
      explicit RobustKernelScaleDelta(number_t delta = 1.);

      //! return the underlying kernel
      const RobustKernelPtr kernel() const { return _kernel;}
      //! use another kernel for the underlying operation
      void setKernel(const RobustKernelPtr& ptr);

      void robustify(number_t error, Vector3& rho) const;

    protected:
      RobustKernelPtr _kernel;
  };

  /**
   * \brief Huber Cost Function
   *
   * Loss function as described by Huber
   * See http://en.wikipedia.org/wiki/Huber_loss_function
   *
   * If e^(1/2) < d
   * rho(e) = e
   *
   * else
   *
   *               1/2    2
   * rho(e) = 2 d e    - d
   */
  class G2O_CORE_API RobustKernelHuber : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Pseudo Huber Cost Function
   *
   * The smooth pseudo huber cost function:
   * See http://en.wikipedia.org/wiki/Huber_loss_function
   *
   *    2       e
   * 2 d  (sqrt(-- + 1) - 1)
   *             2
   *            d
   */
  class G2O_CORE_API RobustKernelPseudoHuber : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Cauchy cost function
   *
   *  2     e
   * d  log(-- + 1)
   *         2
   *        d
   */
  class G2O_CORE_API RobustKernelCauchy : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Geman-McClure cost function
   *
   * See http://research.microsoft.com/en-us/um/people/zhang/Papers/ZhangIVC-97-01.pdf
   * and http://www2.informatik.uni-freiburg.de/~agarwal/resources/agarwal-thesis.pdf
   *    e2
   *  -----
   *  e2 + 1
   */
  class G2O_CORE_API RobustKernelGemanMcClure : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Welsch cost function
   *
   * See http://research.microsoft.com/en-us/um/people/zhang/Papers/ZhangIVC-97-01.pdf
   *
   * d^2 [1 - exp(- e2/d^2)]
   *
   */
  class G2O_CORE_API RobustKernelWelsch : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Fair cost function
   *
   * See http://research.microsoft.com/en-us/um/people/zhang/Papers/ZhangIVC-97-01.pdf
   *
   * 2 * d^2 [e2 / d - log (1 + e2 / d)]
   *
   */
  class G2O_CORE_API RobustKernelFair : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Tukey Cost Function
   *
   * See http://research.microsoft.com/en-us/um/people/zhang/Papers/ZhangIVC-97-01.pdf
   *
   * If e2^(1/2) <= d
   * rho(e) = d^2 * (1 - ( 1 - e2 / d^2)^3) / 3
   *
   * else
   *
   * rho(e) = d^2 / 3
   */
  class G2O_CORE_API RobustKernelTukey : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Saturated cost function.
   *
   * The error is at most delta^2
   */
  class G2O_CORE_API RobustKernelSaturated : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };

  /**
   * \brief Dynamic covariance scaling - DCS
   *
   * See paper Robust Map Optimization from Agarwal et al.  ICRA 2013
   *
   * delta is used as $phi$
   */
  class G2O_CORE_API RobustKernelDCS : public RobustKernel
  {
    public:
      virtual void robustify(number_t e2, Vector3& rho) const;
  };
} // end namespace g2o

#endif
