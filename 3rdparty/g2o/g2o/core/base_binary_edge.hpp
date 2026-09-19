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

template <int D, typename E, typename VertexXiType, typename VertexXjType>
OptimizableGraph::Vertex* BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::createFrom(){
  return createVertex(0);
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
OptimizableGraph::Vertex* BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::createTo(){
  return createVertex(1);
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
OptimizableGraph::Vertex* BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::createVertex(int i){
  switch(i) {
  case 0: return new VertexXiType();
  case 1: return new VertexXjType();
  default: return nullptr;
  }
}


template <int D, typename E, typename VertexXiType, typename VertexXjType>
void BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::resize(size_t size)
{
  assert(size == 2 && "attempting to resize a binary edge");
  BaseEdge<D, E>::resize(size);
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
bool BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::allVerticesFixed() const
{
  return (static_cast<const VertexXiType*> (_vertices[0])->fixed() &&
          static_cast<const VertexXjType*> (_vertices[1])->fixed());
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
void BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::constructQuadraticForm()
{
  VertexXiType* from = static_cast<VertexXiType*>(_vertices[0]);
  VertexXjType* to   = static_cast<VertexXjType*>(_vertices[1]);

  // get the Jacobian of the nodes in the manifold domain
  const JacobianXiOplusType& A = jacobianOplusXi();
  const JacobianXjOplusType& B = jacobianOplusXj();


  bool fromNotFixed = !(from->fixed());
  bool toNotFixed = !(to->fixed());

  if (fromNotFixed || toNotFixed) {
    const InformationType& omega = _information;
    Eigen::Matrix<number_t, D, 1, Eigen::ColMajor> omega_r = - omega * _error;
    if (this->robustKernel() == 0) {
      if (fromNotFixed) {
        Eigen::Matrix<number_t, VertexXiType::Dimension, D, Eigen::ColMajor> AtO = A.transpose() * omega;

        {
          internal::QuadraticFormLock lck(*from);

          from->b().noalias() += A.transpose() * omega_r;
          from->A().noalias() += AtO*A;
        }

        if (toNotFixed ) {
          if (_hessianRowMajor) // we have to write to the block as transposed
            _hessianTransposed.noalias() += B.transpose() * AtO.transpose();
          else
            _hessian.noalias() += AtO * B;
        }
      }
      if (toNotFixed) {
        internal::QuadraticFormLock lck(*to);

        to->b().noalias() += B.transpose() * omega_r;
        to->A().noalias() += B.transpose() * omega * B;
      }
    } else { // robust (weighted) error according to some kernel
      number_t error = this->chi2();
      Vector3 rho;
      this->robustKernel()->robustify(error, rho);
      InformationType weightedOmega = this->robustInformation(rho);
      //std::cout << PVAR(rho.transpose()) << std::endl;
      //std::cout << PVAR(weightedOmega) << std::endl;

      omega_r *= rho[1];
      if (fromNotFixed) {
        {
          internal::QuadraticFormLock lck(*from);

          from->b().noalias() += A.transpose() * omega_r;
          from->A().noalias() += A.transpose() * weightedOmega * A;
        }

        if (toNotFixed ) {
          if (_hessianRowMajor) // we have to write to the block as transposed
            _hessianTransposed.noalias() += B.transpose() * weightedOmega * A;
          else
            _hessian.noalias() += A.transpose() * weightedOmega * B;
        }
      }
      if (toNotFixed) {
        internal::QuadraticFormLock lck(*to);

        to->b().noalias() += B.transpose() * omega_r;
        to->A().noalias() += B.transpose() * weightedOmega * B;
      }
    }
  }
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
void BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::linearizeOplus(JacobianWorkspace& jacobianWorkspace)
{
  new (&_jacobianOplusXi) JacobianXiOplusType(jacobianWorkspace.workspaceForVertex(0), D < 0 ? _dimension : D, Di);
  new (&_jacobianOplusXj) JacobianXjOplusType(jacobianWorkspace.workspaceForVertex(1), D < 0 ? _dimension : D, Dj);
  linearizeOplus();
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
void BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::linearizeOplus()
{
  VertexXiType* vi = static_cast<VertexXiType*>(_vertices[0]);
  VertexXjType* vj = static_cast<VertexXjType*>(_vertices[1]);

  bool iNotFixed = !(vi->fixed());
  bool jNotFixed = !(vj->fixed());

  if (!iNotFixed && !jNotFixed)
    return;

  const number_t delta = cst(1e-9);
  const number_t scalar = 1 / (2*delta);
  ErrorVector errorBak;
  ErrorVector errorBeforeNumeric = _error;

  if (iNotFixed) {
    internal::QuadraticFormLock lck(*vi);
    //Xi - estimate the jacobian numerically
    number_t add_vi[VertexXiType::Dimension] = {};

    // add small step along the unit vector in each dimension
    for (int d = 0; d < VertexXiType::Dimension; ++d) {
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

      _jacobianOplusXi.col(d) = scalar * errorBak;
    } // end dimension
  }

  if (jNotFixed) {
    internal::QuadraticFormLock lck(*vj);
    //Xj - estimate the jacobian numerically
    number_t add_vj[VertexXjType::Dimension] = {};

    // add small step along the unit vector in each dimension
    for (int d = 0; d < VertexXjType::Dimension; ++d) {
      vj->push();
      add_vj[d] = delta;
      vj->oplus(add_vj);
      computeError();
      errorBak = _error;
      vj->pop();
      vj->push();
      add_vj[d] = -delta;
      vj->oplus(add_vj);
      computeError();
      errorBak -= _error;
      vj->pop();
      add_vj[d] = 0.0;

      _jacobianOplusXj.col(d) = scalar * errorBak;
    }
  } // end dimension

  _error = errorBeforeNumeric;
}

template <int D, typename E, typename VertexXiType, typename VertexXjType>
void BaseBinaryEdge<D, E, VertexXiType, VertexXjType>::mapHessianMemory(number_t* d, int i, int j, bool rowMajor)
{
  (void) i; (void) j;
  //assert(i == 0 && j == 1);
  if (rowMajor) {
    new (&_hessianTransposed) HessianBlockTransposedType(d, VertexXjType::Dimension, VertexXiType::Dimension);
  } else {
    new (&_hessian) HessianBlockType(d, VertexXiType::Dimension, VertexXjType::Dimension);
  }
  _hessianRowMajor = rowMajor;
}
