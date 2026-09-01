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

template <int D, typename E, typename VertexXiType>
void BaseUnaryEdge<D, E, VertexXiType>::resize(size_t size)
{
  assert(size == 1 && "error resizing unary edge where size != 1");
  BaseEdge<D, E>::resize(size);
}

template <int D, typename E, typename VertexXiType>
bool BaseUnaryEdge<D, E, VertexXiType>::allVerticesFixed() const
{
  return static_cast<const VertexXiType*> (_vertices[0])->fixed();
}

template <int D, typename E, typename VertexXiType>
OptimizableGraph::Vertex* BaseUnaryEdge<D, E, VertexXiType>::createVertex(int i)
{
  if (i!=0)
    return nullptr;
  return new VertexXiType();
}

template <int D, typename E, typename VertexXiType>
void BaseUnaryEdge<D, E, VertexXiType>::constructQuadraticForm()
{
  VertexXiType* from=static_cast<VertexXiType*>(_vertices[0]);

  // chain rule to get the Jacobian of the nodes in the manifold domain
  const JacobianXiOplusType& A = jacobianOplusXi();
  const InformationType& omega = _information;

  bool istatus = !from->fixed();
  if (istatus) {
#ifdef G2O_OPENMP
    from->lockQuadraticForm();
#endif
    if (this->robustKernel()) {
      number_t error = this->chi2();
      Vector3 rho;
      this->robustKernel()->robustify(error, rho);
      InformationType weightedOmega = this->robustInformation(rho);

      from->b().noalias() -= rho[1] * A.transpose() * omega * _error;
      from->A().noalias() += A.transpose() * weightedOmega * A;
    } else {
      from->b().noalias() -= A.transpose() * omega * _error;
      from->A().noalias() += A.transpose() * omega * A;
    }
#ifdef G2O_OPENMP
    from->unlockQuadraticForm();
#endif
  }
}

template <int D, typename E, typename VertexXiType>
void BaseUnaryEdge<D, E, VertexXiType>::linearizeOplus(JacobianWorkspace& jacobianWorkspace)
{
  new (&_jacobianOplusXi) JacobianXiOplusType(jacobianWorkspace.workspaceForVertex(0), D < 0 ? _dimension : D, VertexXiType::Dimension);
  linearizeOplus();
}

template <int D, typename E, typename VertexXiType>
void BaseUnaryEdge<D, E, VertexXiType>::linearizeOplus()
{
  //Xi - estimate the jacobian numerically
  VertexXiType* vi = static_cast<VertexXiType*>(_vertices[0]);

  if (vi->fixed())
    return;

#ifdef G2O_OPENMP
  vi->lockQuadraticForm();
#endif

  const number_t delta = cst(1e-9);
  const number_t scalar = 1 / (2*delta);
  ErrorVector error1;
  ErrorVector errorBeforeNumeric = _error;

  number_t add_vi[VertexXiType::Dimension] = {};

  // add small step along the unit vector in each dimension
  for (int d = 0; d < VertexXiType::Dimension; ++d) {
    vi->push();
    add_vi[d] = delta;
    vi->oplus(add_vi);
    computeError();
    error1 = _error;
    vi->pop();
    vi->push();
    add_vi[d] = -delta;
    vi->oplus(add_vi);
    computeError();
    vi->pop();
    add_vi[d] = 0.0;

    _jacobianOplusXi.col(d) = scalar * (error1 - _error);
  } // end dimension

  _error = errorBeforeNumeric;
#ifdef G2O_OPENMP
  vi->unlockQuadraticForm();
#endif
}

template <int D, typename E, typename VertexXiType>
void BaseUnaryEdge<D, E, VertexXiType>::initialEstimate(const OptimizableGraph::VertexSet&, OptimizableGraph::Vertex*)
{
  std::cerr << __PRETTY_FUNCTION__ << " is not implemented, please give implementation in your derived class" << std::endl;
}
