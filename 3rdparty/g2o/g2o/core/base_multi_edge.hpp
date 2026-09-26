// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, H. Strasdat, W. Burgard
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

namespace internal {
  inline int computeUpperTriangleIndex(int i, int j)
  {
    int elemsUpToCol = ((j-1) * j) / 2;
    return elemsUpToCol + i;
  }
}

template <int D, typename E>
void BaseMultiEdge<D, E>::constructQuadraticForm()
{
  if (this->robustKernel()) {
    number_t error = this->chi2();
    Vector3 rho;
    this->robustKernel()->robustify(error, rho);
    Eigen::Matrix<number_t, D, 1, Eigen::ColMajor> omega_r = - _information * _error;
    omega_r *= rho[1];
    computeQuadraticForm(this->robustInformation(rho), omega_r);
  } else {
    computeQuadraticForm(_information, - _information * _error);
  }
}


template <int D, typename E>
void BaseMultiEdge<D, E>::linearizeOplus(JacobianWorkspace& jacobianWorkspace)
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);
    assert(v->dimension() >= 0);
    new (&_jacobianOplus[i]) JacobianType(jacobianWorkspace.workspaceForVertex(i), D < 0 ? _dimension : D, v->dimension());
  }
  linearizeOplus();
}

template <int D, typename E>
void BaseMultiEdge<D, E>::linearizeOplus()
{
  const number_t delta = cst(1e-9);
  const number_t scalar = 1 / (2*delta);
  ErrorVector errorBak;
  ErrorVector errorBeforeNumeric = _error;

  dynamic_aligned_buffer<number_t> buffer{ 12 };

  for (size_t i = 0; i < _vertices.size(); ++i) {
    //Xi - estimate the jacobian numerically
    OptimizableGraph::Vertex* vi = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);

    if (vi->fixed()) {
      continue;
    } else {
      internal::QuadraticFormLock lck(*vi);
      const int vi_dim = vi->dimension();
      assert(vi_dim >= 0);

      number_t* add_vi = buffer.request(vi_dim);

      std::fill(add_vi, add_vi + vi_dim, cst(0.0));
      assert(_dimension >= 0);
      assert(_jacobianOplus[i].rows() == _dimension && _jacobianOplus[i].cols() == vi_dim && "jacobian cache dimension does not match");
        _jacobianOplus[i].resize(_dimension, vi_dim);
      // add small step along the unit vector in each dimension
      for (int d = 0; d < vi_dim; ++d) {
        vi->push();
        add_vi[d] = delta;
        vi->oplus(add_vi);
        computeError();
        errorBak = _error;
        vi->pop();
        vi->push();
        add_vi[d] = -delta;
        vi->oplus(add_vi);
        computeError();
        errorBak -= _error;
        vi->pop();
        add_vi[d] = 0.0;

        _jacobianOplus[i].col(d) = scalar * errorBak;
      } // end dimension
    }
  }
  _error = errorBeforeNumeric;
}

template <int D, typename E>
void BaseMultiEdge<D, E>::mapHessianMemory(number_t* d, int i, int j, bool rowMajor)
{
  int idx = internal::computeUpperTriangleIndex(i, j);
  assert(idx < (int)_hessian.size());
  OptimizableGraph::Vertex* vi = static_cast<OptimizableGraph::Vertex*>(HyperGraph::Edge::vertex(i));
  OptimizableGraph::Vertex* vj = static_cast<OptimizableGraph::Vertex*>(HyperGraph::Edge::vertex(j));
  assert(vi->dimension() >= 0);
  assert(vj->dimension() >= 0);
  HessianHelper& h = _hessian[idx];
  if (rowMajor) {
    if (h.matrix.data() != d || h.transposed != rowMajor)
      new (&h.matrix) HessianBlockType(d, vj->dimension(), vi->dimension());
  } else {
    if (h.matrix.data() != d || h.transposed != rowMajor)
      new (&h.matrix) HessianBlockType(d, vi->dimension(), vj->dimension());
  }
  h.transposed = rowMajor;
}

template <int D, typename E>
void BaseMultiEdge<D, E>::resize(size_t size)
{
  BaseEdge<D,E>::resize(size);
  int n = (int)_vertices.size();
  int maxIdx = (n * (n-1))/2;
  assert(maxIdx >= 0);
  _hessian.resize(maxIdx);
  _jacobianOplus.resize(size, JacobianType(0,0,0));
}

template <int D, typename E>
bool BaseMultiEdge<D, E>::allVerticesFixed() const
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    if (!static_cast<const OptimizableGraph::Vertex*> (_vertices[i])->fixed()) {
      return false;
    }
  }
  return true;
}

template <int D, typename E>
void BaseMultiEdge<D, E>::computeQuadraticForm(const InformationType& omega, const ErrorVector& weightedError)
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    OptimizableGraph::Vertex* from = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);
    bool istatus = !(from->fixed());

    if (istatus) {
      const JacobianType& A = _jacobianOplus[i];

      MatrixX AtO = A.transpose() * omega;
      int fromDim = from->dimension();
      assert(fromDim >= 0);
      Eigen::Map<MatrixX> fromMap(from->hessianData(), fromDim, fromDim);
      Eigen::Map<VectorX> fromB(from->bData(), fromDim);

      // ii block in the hessian
      {
        internal::QuadraticFormLock lck(*from);
        fromMap.noalias() += AtO * A;
        fromB.noalias() += A.transpose() * weightedError;
      }

      // compute the off-diagonal blocks ij for all j
      for (size_t j = i+1; j < _vertices.size(); ++j) {
        OptimizableGraph::Vertex* to = static_cast<OptimizableGraph::Vertex*>(_vertices[j]);

        bool jstatus = !(to->fixed());
        if (jstatus) {
          internal::QuadraticFormLock lck(*to);
          const JacobianType& B = _jacobianOplus[j];
          int idx = internal::computeUpperTriangleIndex(i, j);
          assert(idx < (int)_hessian.size());
          HessianHelper& hhelper = _hessian[idx];
          if (hhelper.transposed) { // we have to write to the block as transposed
            hhelper.matrix.noalias() += B.transpose() * AtO.transpose();
          } else {
            hhelper.matrix.noalias() += AtO * B;
          }
        }
      }
    }
  }
}


// PARTIAL TEMPLATE SPECIALIZATION

template <typename E>
void BaseMultiEdge<-1, E>::constructQuadraticForm()
{
  if (this->robustKernel()) {
    number_t error = this->chi2();
    Vector3 rho;
    this->robustKernel()->robustify(error, rho);
    Eigen::Matrix<number_t, Eigen::Dynamic, 1, Eigen::ColMajor> omega_r = - _information * _error;
    omega_r *= rho[1];
    computeQuadraticForm(this->robustInformation(rho), omega_r);
  } else {
    computeQuadraticForm(_information, - _information * _error);
  }
}


template <typename E>
void BaseMultiEdge<-1, E>::linearizeOplus(JacobianWorkspace& jacobianWorkspace)
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);
    assert(v->dimension() >= 0);
    new (&_jacobianOplus[i]) JacobianType(jacobianWorkspace.workspaceForVertex(i), _dimension, v->dimension());
  }
  linearizeOplus();
}

template <typename E>
void BaseMultiEdge<-1, E>::linearizeOplus()
{
  const number_t delta = cst(1e-9);
  const number_t scalar = 1 / (2*delta);
  ErrorVector errorBak;
  ErrorVector errorBeforeNumeric = _error;

  dynamic_aligned_buffer<number_t> buffer{ 12 };

  for (size_t i = 0; i < _vertices.size(); ++i) {
    //Xi - estimate the jacobian numerically
    OptimizableGraph::Vertex* vi = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);

    if (vi->fixed()) {
      continue;
    } else {
      internal::QuadraticFormLock lck(*vi);
      const int vi_dim = vi->dimension();
      assert(vi_dim >= 0);

      number_t* add_vi = buffer.request(vi_dim);

      std::fill(add_vi, add_vi + vi_dim, cst(0.0));
      assert(_dimension >= 0);
      assert(_jacobianOplus[i].rows() == _dimension && _jacobianOplus[i].cols() == vi_dim && "jacobian cache dimension does not match");
      _jacobianOplus[i].resize(_dimension, vi_dim);
      // add small step along the unit vector in each dimension
      for (int d = 0; d < vi_dim; ++d) {
        vi->push();
        add_vi[d] = delta;
        vi->oplus(add_vi);
        computeError();
        errorBak = _error;
        vi->pop();
        vi->push();
        add_vi[d] = -delta;
        vi->oplus(add_vi);
        computeError();
        errorBak -= _error;
        vi->pop();
        add_vi[d] = 0.0;

        _jacobianOplus[i].col(d) = scalar * errorBak;
      } // end dimension
    }
  }
  _error = errorBeforeNumeric;
}

template <typename E>
void BaseMultiEdge<-1, E>::mapHessianMemory(number_t* d, int i, int j, bool rowMajor)
{
  int idx = internal::computeUpperTriangleIndex(i, j);
  assert(idx < (int)_hessian.size());
  OptimizableGraph::Vertex* vi = static_cast<OptimizableGraph::Vertex*>(HyperGraph::Edge::vertex(i));
  OptimizableGraph::Vertex* vj = static_cast<OptimizableGraph::Vertex*>(HyperGraph::Edge::vertex(j));
  assert(vi->dimension() >= 0);
  assert(vj->dimension() >= 0);
  HessianHelper& h = _hessian[idx];
  if (rowMajor) {
    if (h.matrix.data() != d || h.transposed != rowMajor)
      new (&h.matrix) HessianBlockType(d, vj->dimension(), vi->dimension());
  } else {
    if (h.matrix.data() != d || h.transposed != rowMajor)
      new (&h.matrix) HessianBlockType(d, vi->dimension(), vj->dimension());
  }
  h.transposed = rowMajor;
}

template <typename E>
void BaseMultiEdge<-1, E>::resize(size_t size)
{
  BaseEdge<-1,E>::resize(size);
  int n = (int)_vertices.size();
  int maxIdx = (n * (n-1))/2;
  assert(maxIdx >= 0);
  _hessian.resize(maxIdx);
  _jacobianOplus.resize(size, JacobianType(0,0,0));
}

template <typename E>
bool BaseMultiEdge<-1, E>::allVerticesFixed() const
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    if (!static_cast<const OptimizableGraph::Vertex*> (_vertices[i])->fixed()) {
      return false;
    }
  }
  return true;
}

template <typename E>
void BaseMultiEdge<-1, E>::computeQuadraticForm(const InformationType& omega, const ErrorVector& weightedError)
{
  for (size_t i = 0; i < _vertices.size(); ++i) {
    OptimizableGraph::Vertex* from = static_cast<OptimizableGraph::Vertex*>(_vertices[i]);
    bool istatus = !(from->fixed());

    if (istatus) {
      const JacobianType& A = _jacobianOplus[i];

      MatrixX AtO = A.transpose() * omega;
      int fromDim = from->dimension();
      assert(fromDim >= 0);
      Eigen::Map<MatrixX> fromMap(from->hessianData(), fromDim, fromDim);
      Eigen::Map<VectorX> fromB(from->bData(), fromDim);

      // ii block in the hessian
      {
        internal::QuadraticFormLock lck(*from);
        fromMap.noalias() += AtO * A;
        fromB.noalias() += A.transpose() * weightedError;
      }

      // compute the off-diagonal blocks ij for all j
      for (size_t j = i+1; j < _vertices.size(); ++j) {
        OptimizableGraph::Vertex* to = static_cast<OptimizableGraph::Vertex*>(_vertices[j]);
        bool jstatus = !(to->fixed());
        if (jstatus) {
          internal::QuadraticFormLock lck(*to);
          const JacobianType& B = _jacobianOplus[j];
          int idx = internal::computeUpperTriangleIndex(i, j);
          assert(idx < (int)_hessian.size());
          HessianHelper& hhelper = _hessian[idx];
          if (hhelper.transposed) { // we have to write to the block as transposed
            hhelper.matrix.noalias() += B.transpose() * AtO.transpose();
          } else {
            hhelper.matrix.noalias() += AtO * B;
          }
        }
      }
    }
  }
}
