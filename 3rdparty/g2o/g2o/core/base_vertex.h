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

#ifndef G2O_BASE_VERTEX_H
#define G2O_BASE_VERTEX_H

#include "optimizable_graph.h"
#include "creators.h"
#include "g2o/stuff/macros.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <stack>

namespace g2o {

/**
 * \brief Templatized BaseVertex
 *
 * Templatized BaseVertex
 * D  : minimal dimension of the vertex, e.g., 3 for rotation in 3D
 * T  : internal type to represent the estimate, e.g., Quaternion for rotation in 3D
 */
  template <int D, typename T>
  class BaseVertex : public OptimizableGraph::Vertex {
    public:
    typedef T EstimateType;
    typedef std::stack<EstimateType, 
                       std::vector<EstimateType,  Eigen::aligned_allocator<EstimateType> > >
    BackupStackType;

    static const int Dimension = D;           ///< dimension of the estimate (minimal) in the manifold space

    typedef Eigen::Map<Eigen::Matrix<number_t, D, D, Eigen::ColMajor>, Eigen::Matrix<number_t, D, D, Eigen::ColMajor>::Flags & Eigen::PacketAccessBit ? Eigen::Aligned : Eigen::Unaligned >  HessianBlockType;

  public:
    BaseVertex();

    virtual const number_t& hessian(int i, int j) const { assert(i<D && j<D); return _hessian(i,j);}
    virtual number_t& hessian(int i, int j)  { assert(i<D && j<D); return _hessian(i,j);}
    virtual number_t hessianDeterminant() const {return _hessian.determinant();}
    virtual number_t* hessianData() { return const_cast<number_t*>(_hessian.data());}

    inline virtual void mapHessianMemory(number_t* d);

    virtual int copyB(number_t* b_) const {
      memcpy(b_, _b.data(), Dimension * sizeof(number_t));
      return Dimension; 
    }

    virtual const number_t& b(int i) const { assert(i < D); return _b(i);}
    virtual number_t& b(int i) { assert(i < D); return _b(i);}
    virtual number_t* bData() { return _b.data();}

    inline virtual void clearQuadraticForm();

    //! updates the current vertex with the direct solution x += H_ii\b_ii
    //! @returns the determinant of the inverted hessian
    inline virtual number_t solveDirect(number_t lambda=0);

    //! return right hand side b of the constructed linear system
    Eigen::Matrix<number_t, D, 1, Eigen::ColMajor>& b() { return _b;}
    const Eigen::Matrix<number_t, D, 1, Eigen::ColMajor>& b() const { return _b;}
    //! return the hessian block associated with the vertex
    HessianBlockType& A() { return _hessian;}
    const HessianBlockType& A() const { return _hessian;}

    virtual void push() { _backup.push(_estimate);}
    virtual void pop() { assert(!_backup.empty()); _estimate = _backup.top(); _backup.pop(); updateCache();}
    virtual void discardTop() { assert(!_backup.empty()); _backup.pop();}
    virtual int stackSize() const {return _backup.size();}

    //! return the current estimate of the vertex
    const EstimateType& estimate() const { return _estimate;}
    //! set the estimate for the vertex also calls updateCache()
    void setEstimate(const EstimateType& et) { _estimate = et; updateCache();}

  protected:
    HessianBlockType _hessian;
    Eigen::Matrix<number_t, D, 1, Eigen::ColMajor> _b;
    EstimateType _estimate;
    BackupStackType _backup;
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#include "base_vertex.hpp"

} // end namespace g2o


#endif
