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

#ifndef G2O_SPARSE_BLOCK_MATRIX_
#define G2O_SPARSE_BLOCK_MATRIX_

#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <Eigen/Core>
#include <memory>

#include "sparse_block_matrix_ccs.h"
#include "matrix_structure.h"
#include "matrix_operations.h"
#include "g2o/config.h"
#include "g2o/stuff/misc.h"

namespace g2o {
/**
 * \brief Sparse matrix which uses blocks
 *
 * Template class that specifies a sparse block matrix.  A block
 * matrix is a sparse matrix made of dense blocks.  These blocks
 * cannot have a random pattern, but follow a (variable) grid
 * structure. This structure is specified by a partition of the rows
 * and the columns of the matrix.  The blocks are represented by the
 * Eigen::Matrix structure, thus they can be statically or dynamically
 * allocated. For efficiency reasons it is convenient to allocate them
 * statically, when possible. A static block matrix has all blocks of
 * the same size, and the size of the block is specified by the
 * template argument.  If this is not the case, and you have different
 * block sizes than you have to use a dynamic-block matrix (default
 * template argument).  
 */
template <class MatrixType = MatrixX >
class SparseBlockMatrix {

  public:
    //! this is the type of the elementary block, it is an Eigen::Matrix.
    typedef MatrixType SparseMatrixBlock;

    //! columns of the matrix
    inline int cols() const {return _colBlockIndices.size() ? _colBlockIndices.back() : 0;}
    //! rows of the matrix
    inline int rows() const {return _rowBlockIndices.size() ? _rowBlockIndices.back() : 0;}

    typedef std::map<int, SparseMatrixBlock*> IntBlockMap;

    /**
     * constructs a sparse block matrix having a specific layout
     * @param rbi: array of int containing the row layout of the blocks. 
     * the component i of the array should contain the index of the first row of the block i+1.
     * @param cbi: array of int containing the column layout of the blocks. 
     *  the component i of the array should contain the index of the first col of the block i+1.
     * @param rb: number of row blocks
     * @param cb: number of col blocks
     * @param hasStorage: set it to true if the matrix "owns" the blocks, thus it deletes it on destruction.
     * if false the matrix is only a "view" over an existing structure.
     */
    SparseBlockMatrix( const int * rbi, const int* cbi, int rb, int cb, bool hasStorage=true);

    SparseBlockMatrix();

    ~SparseBlockMatrix();

    
    //! this zeroes all the blocks. If dealloc=true the blocks are removed from memory
    void clear(bool dealloc=false) ;

    //! returns the block at location r,c. if alloc=true he block is created if it does not exist
    SparseMatrixBlock* block(int r, int c, bool alloc=false);
    //! returns the block at location r,c
    const SparseMatrixBlock* block(int r, int c) const;

    //! how many rows does the block at block-row r has?
    inline int rowsOfBlock(int r) const { return r ? _rowBlockIndices[r] - _rowBlockIndices[r-1] : _rowBlockIndices[0] ; }

    //! how many cols does the block at block-col c has?
    inline int colsOfBlock(int c) const { return c ? _colBlockIndices[c] - _colBlockIndices[c-1] : _colBlockIndices[0]; }

    //! where does the row at block-row r starts?
    inline int rowBaseOfBlock(int r) const { return r ? _rowBlockIndices[r-1] : 0 ; }

    //! where does the col at block-col r starts?
    inline int colBaseOfBlock(int c) const { return c ? _colBlockIndices[c-1] : 0 ; }

    //! number of non-zero elements
    size_t nonZeros() const; 
    //! number of allocated blocks
    size_t nonZeroBlocks() const; 

    //! deep copy of a sparse-block-matrix;
    SparseBlockMatrix* clone() const ;

    /**
     * returns a view or a copy of the block matrix
     * @param rmin: starting block row
     * @param rmax: ending  block row
     * @param cmin: starting block col
     * @param cmax: ending  block col
     * @param alloc: if true it makes a deep copy, if false it creates a view.
     */
    SparseBlockMatrix*  slice(int rmin, int rmax, int cmin, int cmax, bool alloc=true) const;

    //! transposes a block matrix, The transposed type should match the argument false on failure
    template <class MatrixTransposedType>
    bool transpose(SparseBlockMatrix<MatrixTransposedType>& dest) const;

    template <class MatrixTransposedType>
    std::unique_ptr<SparseBlockMatrix<MatrixTransposedType>> transposed() const;

    //! adds the current matrix to the destination
    bool add(SparseBlockMatrix<MatrixType>& dest) const;
    std::unique_ptr<SparseBlockMatrix<MatrixType>> added() const;

    //! dest = (*this) *  M
    template <class MatrixResultType, class MatrixFactorType>
    bool multiply(SparseBlockMatrix<MatrixResultType> *& dest, const SparseBlockMatrix<MatrixFactorType>* M) const;

    //! dest = (*this) *  src
    void multiply(number_t*& dest, const number_t* src) const;

    /**
     * compute dest = (*this) *  src
     * However, assuming that this is a symmetric matrix where only the upper triangle is stored
     */
    void multiplySymmetricUpperTriangle(number_t*& dest, const number_t* src) const;

    //! dest = M * (*this)
    void rightMultiply(number_t*& dest, const number_t* src) const;

    //! *this *= a
    void scale( number_t a);

    /**
     * writes in dest a block permutaton specified by pinv.
     * @param pinv: array such that new_block[i] = old_block[pinv[i]]
     */
    bool symmPermutation(SparseBlockMatrix<MatrixType>*& dest, const int* pinv, bool onlyUpper=false) const;

    /**
     * fill the CCS arrays of a matrix, arrays have to be allocated beforehand
     */
    int fillCCS(int* Cp, int* Ci, number_t* Cx, bool upperTriangle = false) const;

    /**
     * fill the CCS arrays of a matrix, arrays have to be allocated beforehand. This function only writes
     * the values and assumes that column and row structures have already been written.
     */
    int fillCCS(number_t* Cx, bool upperTriangle = false) const;

    //! exports the non zero blocks in the structure matrix ms
    void fillBlockStructure(MatrixStructure& ms) const;

    //! the block matrices per block-column
    const std::vector<IntBlockMap>& blockCols() const { return _blockCols;}
    std::vector<IntBlockMap>& blockCols() { return _blockCols;}

    //! indices of the row blocks
    const std::vector<int>& rowBlockIndices() const { return _rowBlockIndices;}
    std::vector<int>& rowBlockIndices() { return _rowBlockIndices;}

    //! indices of the column blocks
    const std::vector<int>& colBlockIndices() const { return _colBlockIndices;}
    std::vector<int>& colBlockIndices() { return _colBlockIndices;}

    /**
     * write the content of this matrix to a stream loadable by Octave
     * @param upperTriangle does this matrix store only the upper triangular blocks
     */
    bool writeOctave(const char* filename, bool upperTriangle = true) const;

    /**
     * copy into CCS structure
     * @return number of processed blocks, -1 on error
     */
    int fillSparseBlockMatrixCCS(SparseBlockMatrixCCS<MatrixType>& blockCCS) const;

    /**
     * copy as transposed into a CCS structure
     * @return number of processed blocks, -1 on error
     */
    int fillSparseBlockMatrixCCSTransposed(SparseBlockMatrixCCS<MatrixType>& blockCCS) const;

    /**
     * take over the memory and matrix pattern from a hash matrix.
     * The structure of the hash matrix will be cleared.
     */
    void takePatternFromHash(SparseBlockMatrixHashMap<MatrixType>& hashMatrix);

  protected:
    std::vector<int> _rowBlockIndices; ///< vector of the indices of the blocks along the rows.
    std::vector<int> _colBlockIndices; ///< vector of the indices of the blocks along the cols
    //! array of maps of blocks. The index of the array represent a block column of the matrix
    //! and the block column is stored as a map row_block -> matrix_block_ptr.
    std::vector <IntBlockMap> _blockCols;
    bool _hasStorage;

  private:
    template <class MatrixTransposedType>
    void transpose_internal(SparseBlockMatrix<MatrixTransposedType>& dest) const;

    void add_internal(SparseBlockMatrix<MatrixType>& dest) const;
};

template < class  MatrixType >
std::ostream& operator << (std::ostream&, const SparseBlockMatrix<MatrixType>& m);

  typedef SparseBlockMatrix<MatrixX> SparseBlockMatrixX;   

} //end namespace

#include "sparse_block_matrix.hpp"

#endif
