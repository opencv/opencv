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

#ifndef G2O_OPTIMIZATION_ALGORITHM_H
#define G2O_OPTIMIZATION_ALGORITHM_H

#include <vector>
#include <utility>
#include <iosfwd>

#include "g2o/stuff/property.h"

#include "hyper_graph.h"
#include "sparse_block_matrix.h"
#include "g2o_core_api.h"

namespace g2o {

  class SparseOptimizer;

  /**
   * \brief Generic interface for a non-linear solver operating on a graph
   */
  class G2O_CORE_API OptimizationAlgorithm
  {
    public:
      enum G2O_CORE_API SolverResult {Terminate=2, OK=1, Fail=-1};
      OptimizationAlgorithm();
      virtual ~OptimizationAlgorithm();

      /**
       * initialize the solver, called once before the first call to solve()
       */
      virtual bool init(bool online = false) = 0;

      /**
       * Solve one iteration. The SparseOptimizer running on-top will call this
       * for the given number of iterations.
       * @param iteration indicates the current iteration
       */
      virtual SolverResult solve(int iteration, bool online = false) = 0;

      /**
       * computes the block diagonal elements of the pattern specified in the input
       * and stores them in given SparseBlockMatrix.
       * If your solver does not support computing the marginals, return false.
       */
      virtual bool computeMarginals(SparseBlockMatrix<MatrixX>& spinv, const std::vector<std::pair<int, int> >& blockIndices) = 0;

      /**
       * update the structures for online processing
       */
      virtual bool updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges) = 0;

      /**
       * called by the optimizer if verbose. re-implement, if you want to print something
       */
      virtual void printVerbose(std::ostream& os) const {(void) os;};

    public:
      //! return the optimizer operating on
      const SparseOptimizer* optimizer() const { return _optimizer;}
      SparseOptimizer* optimizer() { return _optimizer;}

      /**
       * specify on which optimizer the solver should work on
       */
      void setOptimizer(SparseOptimizer* optimizer);

      //! return the properties of the solver
      const PropertyMap& properties() const { return _properties;}

      /**
       * update the properties from a string, see PropertyMap::updateMapFromString()
       */
      bool updatePropertiesFromString(const std::string& propString);
      
      /**
       * print the properties to a stream in a human readable fashion
       */
      void printProperties(std::ostream& os) const;

    protected:
      SparseOptimizer* _optimizer;   ///< the optimizer the solver is working on
      PropertyMap _properties;       ///< the properties of your solver, use this to store the parameters of your solver

    private:
      // Disable the copy constructor and assignment operator
      OptimizationAlgorithm(const OptimizationAlgorithm&) { }
      OptimizationAlgorithm& operator= (const OptimizationAlgorithm&) { return *this; }
  };

} // end namespace

#endif
