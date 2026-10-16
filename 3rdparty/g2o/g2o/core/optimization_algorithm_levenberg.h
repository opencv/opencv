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

#ifndef G2O_SOLVER_LEVENBERG_H
#define G2O_SOLVER_LEVENBERG_H

#include "optimization_algorithm_with_hessian.h"
#include "g2o_core_api.h"

#include <memory>

namespace g2o {

  /**
   * \brief Implementation of the Levenberg Algorithm
   */
  class G2O_CORE_API OptimizationAlgorithmLevenberg : public OptimizationAlgorithmWithHessian
  {
    public:
      /**
       * construct the Levenberg algorithm, which will use the given Solver for solving the
       * linearized system.
       */
      explicit OptimizationAlgorithmLevenberg(std::unique_ptr<Solver> solver);
      virtual ~OptimizationAlgorithmLevenberg();

      virtual SolverResult solve(int iteration, bool online = false);

      virtual void printVerbose(std::ostream& os) const;

      //! return the currently used damping factor
      number_t currentLambda() const { return _currentLambda;}

      //! the number of internal iteration if an update step increases chi^2 within Levenberg-Marquardt
      void setMaxTrialsAfterFailure(int max_trials);

      //! get the number of inner iterations for Levenberg-Marquardt
      int maxTrialsAfterFailure() const { return _maxTrialsAfterFailure->value();}

      //! return the lambda set by the user, if < 0 the SparseOptimizer will compute the initial lambda
      number_t userLambdaInit() {return _userLambdaInit->value();}
      //! specify the initial lambda used for the first iteraion, if not given the SparseOptimizer tries to compute a suitable value
      void setUserLambdaInit(number_t lambda);

      //! return the number of levenberg iterations performed in the last round
      int levenbergIteration() { return _levenbergIterations;}

    protected:
      // Levenberg parameters
      Property<int>* _maxTrialsAfterFailure;
      Property<number_t>* _userLambdaInit;
      number_t _currentLambda;
      number_t _tau;
      number_t _goodStepLowerScale; ///< lower bound for lambda decrease if a good LM step
      number_t _goodStepUpperScale; ///< upper bound for lambda decrease if a good LM step
      number_t _ni;
      int _levenbergIterations;   ///< the numer of levenberg iterations performed to accept the last step

      /**
       * helper for Levenberg, this function computes the initial damping factor, if the user did not
       * specify an own value, see setUserLambdaInit()
       */
      number_t computeLambdaInit() const;
      number_t computeScale() const;

  private:
      std::unique_ptr<Solver> m_solver;
  };

} // end namespace

#endif
