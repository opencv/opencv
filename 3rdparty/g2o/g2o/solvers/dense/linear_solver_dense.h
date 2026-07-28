// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// Copyright (C) 2012 R. Kümmerle
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

#ifndef G2O_LINEAR_SOLVER_DENSE_H
#define G2O_LINEAR_SOLVER_DENSE_H

#include "g2o/core/linear_solver.h"
#include "g2o/core/batch_stats.h"

#include <vector>
#include <utility>
#include<Eigen/Core>
#include<Eigen/Cholesky>


namespace g2o {

  /**
   * \brief linear solver using dense cholesky decomposition
   */
  template <typename MatrixType>
  class LinearSolverDense : public LinearSolver<MatrixType>
  {
    public:
      LinearSolverDense() :
        LinearSolver<MatrixType>(),
        _reset(true)
      {
      }

      virtual ~LinearSolverDense()
      {
      }

      virtual bool init()
      {
        _reset = true;
        return true;
      }

      bool solve(const SparseBlockMatrix<MatrixType>& A, number_t* x, number_t* b)
      {
        int n = A.cols();
        int m = A.cols();

        MatrixX& H = _H;
        if (H.cols() != n) {
          H.resize(n, m);
          _reset = true;
        }
        if (_reset) {
          _reset = false;
          H.setZero();
        }

        // copy the sparse block matrix into a dense matrix
        int c_idx = 0;
        for (size_t i = 0; i < A.blockCols().size(); ++i) {
          int c_size = A.colsOfBlock(i);
          assert(c_idx == A.colBaseOfBlock(i) && "mismatch in block indices");

          const typename SparseBlockMatrix<MatrixType>::IntBlockMap& col = A.blockCols()[i];
          if (col.size() > 0) {
            typename SparseBlockMatrix<MatrixType>::IntBlockMap::const_iterator it;
            for (it = col.begin(); it != col.end(); ++it) {
              int r_idx = A.rowBaseOfBlock(it->first);
              // only the upper triangular block is processed
              if (it->first <= (int)i) {
                int r_size = A.rowsOfBlock(it->first);
                H.block(r_idx, c_idx, r_size, c_size) = *(it->second);
                if (r_idx != c_idx) // write the lower triangular block
                  H.block(c_idx, r_idx, c_size, r_size) = it->second->transpose();
              }
            }
          }

          c_idx += c_size;
        }

        // solving via Cholesky decomposition
        VectorX::MapType xvec(x, m);
        VectorX::ConstMapType bvec(b, n);
        _cholesky.compute(H);
        if (_cholesky.isPositive()) {
          xvec = _cholesky.solve(bvec);
          return true;
        }
        return false;
      }

    protected:
      bool _reset;
      MatrixX _H;
      Eigen::LDLT<MatrixX> _cholesky;

  };


}// end namespace

#endif
