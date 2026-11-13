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

#ifndef G2O_MATRIX_STRUCTURE_H
#define G2O_MATRIX_STRUCTURE_H

#include "g2o_core_api.h"

namespace g2o {

/**
 * \brief representing the structure of a matrix in column compressed structure (only the upper triangular part of the matrix)
 */
class G2O_CORE_API MatrixStructure
{
  public:
    MatrixStructure();
    ~MatrixStructure();
    /**
     * allocate space for the Matrix Structure. You may call this on an already allocated struct, it will
     * then reallocate the memory + additional space (number_t the required space).
     */
    void alloc(int n_, int nz);

    void free();
    
    /**
     * Write the matrix pattern to a file. File is also loadable by octave, e.g., then use spy(matrix)
     */
    bool write(const char* filename) const;

    int n;    ///< A is m-by-n.  n must be >= 0.
    int m;    ///< A is m-by-n.  m must be >= 0.
    int* Ap;  ///< column pointers for A, of size n+1
    int* Aii; ///< row indices of A, of size nz = Ap [n]

    //! max number of non-zeros blocks
    int nzMax() const { return maxNz;}

  protected:
    int maxN;     ///< size of the allocated memory
    int maxNz;    ///< size of the allocated memory
};

} // end namespace

#endif
