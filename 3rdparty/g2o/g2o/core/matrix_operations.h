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

#ifndef G2O_CORE_MATRIX_OPERATIONS_H
#define G2O_CORE_MATRIX_OPERATIONS_H

#include <Eigen/Core>

namespace g2o {
  namespace internal {

    template<typename MatrixType>
    inline void axpy(const MatrixType& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment<MatrixType::RowsAtCompileTime>(yoff) += A * x.segment<MatrixType::ColsAtCompileTime>(xoff);
    }

    template<int t>
    inline void axpy(const Eigen::Matrix<number_t, Eigen::Dynamic, t>& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment(yoff, A.rows()) += A * x.segment<Eigen::Matrix<number_t, Eigen::Dynamic, t>::ColsAtCompileTime>(xoff);
    }

    template<>
    inline void axpy<MatrixX>(const MatrixX& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment(yoff, A.rows()) += A * x.segment(xoff, A.cols());
    }

    template<typename MatrixType>
    inline void atxpy(const MatrixType& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment<MatrixType::ColsAtCompileTime>(yoff) += A.transpose() * x.segment<MatrixType::RowsAtCompileTime>(xoff);
    }

    template<int t>
    inline void atxpy(const Eigen::Matrix<number_t, Eigen::Dynamic, t>& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment<Eigen::Matrix<number_t, Eigen::Dynamic, t>::ColsAtCompileTime>(yoff) += A.transpose() * x.segment(xoff, A.rows());
    }

    template<>
    inline void atxpy<MatrixX>(const MatrixX& A, const Eigen::Map<const VectorX>& x, int xoff, Eigen::Map<VectorX>& y, int yoff)
    {
      y.segment(yoff, A.cols()) += A.transpose() * x.segment(xoff, A.rows());
    }

  } // end namespace internal
} // end namespace g2o

#endif
