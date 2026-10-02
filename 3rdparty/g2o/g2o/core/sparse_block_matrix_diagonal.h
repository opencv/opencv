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

#ifndef G2O_SPARSE_BLOCK_MATRIX_DIAGONAL_H
#define G2O_SPARSE_BLOCK_MATRIX_DIAGONAL_H

#include <vector>
#include <Eigen/Core>

#include "g2o/config.h"
#include "matrix_operations.h"

namespace g2o {

  /**
   * \brief Sparse matrix which uses blocks on the diagonal
   *
   * This class is used as a const view on a SparseBlockMatrix
   * which allows a faster iteration over the elements of the
   * matrix.
   */
  template <class MatrixType>
  class SparseBlockMatrixDiagonal
  {
    public:
      //! this is the type of the elementary block, it is an Eigen::Matrix.
      typedef MatrixType SparseMatrixBlock;

      //! columns of the matrix
      int cols() const {return _blockIndices.size() ? _blockIndices.back() : 0;}
      //! rows of the matrix
      int rows() const {return _blockIndices.size() ? _blockIndices.back() : 0;}

      typedef std::vector<MatrixType, Eigen::aligned_allocator<MatrixType> >      DiagonalVector;

      SparseBlockMatrixDiagonal(const std::vector<int>& blockIndices) :
        _blockIndices(blockIndices)
      {}

      //! how many rows/cols does the block at block-row / block-column r has?
      inline int dimOfBlock(int r) const { return r ? _blockIndices[r] - _blockIndices[r-1] : _blockIndices[0] ; }

      //! where does the row /col at block-row / block-column r starts?
      inline int baseOfBlock(int r) const { return r ? _blockIndices[r-1] : 0 ; }

      //! the block matrices per block-column
      const DiagonalVector& diagonal() const { return _diagonal;}
      DiagonalVector& diagonal() { return _diagonal;}

      //! indices of the row blocks
      const std::vector<int>& blockIndices() const { return _blockIndices;}

      void multiply(number_t*& dest, const number_t* src) const
      {
        int destSize=cols();
        if (! dest) {
          dest=new number_t[destSize];
          memset(dest,0, destSize*sizeof(number_t));
        }

        // map the memory by Eigen
        Eigen::Map<VectorX> destVec(dest, destSize);
        Eigen::Map<const VectorX> srcVec(src, rows());

#      ifdef G2O_OPENMP
#      pragma omp parallel for default (shared) schedule(dynamic, 10)
#      endif
        for (int i=0; i < static_cast<int>(_diagonal.size()); ++i){
          int destOffset = baseOfBlock(i);
          int srcOffset = destOffset;
          const SparseMatrixBlock& A = _diagonal[i];
          // destVec += *A.transpose() * srcVec (according to the sub-vector parts)
          internal::template axpy<SparseMatrixBlock>(A, srcVec, srcOffset, destVec, destOffset);
        }
      }

    protected:
      const std::vector<int>& _blockIndices; ///< vector of the indices of the blocks along the diagonal
      DiagonalVector _diagonal;
  };

} //end namespace

#endif
