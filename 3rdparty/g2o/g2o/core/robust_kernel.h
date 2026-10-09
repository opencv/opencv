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

#ifndef G2O_ROBUST_KERNEL_H
#define G2O_ROBUST_KERNEL_H

#include <memory>

#include "g2o_core_api.h"

namespace g2o {

  /**
   * \brief base for all robust cost functions
   *
   * Note in all the implementations for the other cost functions the e in the
   * funtions corresponds to the sqaured errors, i.e., the robustification
   * functions gets passed the squared error.
   *
   * e.g. the robustified least squares function is
   *
   * chi^2 = sum_{e} rho( e^T Omega e )
   */
  class G2O_CORE_API RobustKernel
  {
    public:
      RobustKernel();
      explicit RobustKernel(number_t delta);
      virtual ~RobustKernel() {}
      /**
       * compute the scaling factor for a error:
       * The error is e^T Omega e
       * The output rho is
       * rho[0]: The actual scaled error value
       * rho[1]: First derivative of the scaling function
       * rho[2]: Second derivative of the scaling function
       */
      virtual void robustify(number_t squaredError, Vector3& rho) const = 0;

      /**
       * set the window size of the error. A squared error above delta^2 is considered
       * as outlier in the data.
       */
      virtual void setDelta(number_t delta);
      number_t delta() const { return _delta;}

    protected:
      number_t _delta;
  };
  typedef std::shared_ptr<RobustKernel> RobustKernelPtr;

} // end namespace g2o

#endif
