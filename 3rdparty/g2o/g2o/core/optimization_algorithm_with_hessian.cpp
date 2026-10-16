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

#include "optimization_algorithm_with_hessian.h"

#include "solver.h"
#include "optimizable_graph.h"
#include "sparse_optimizer.h"

#include <iostream>
using namespace std;

namespace g2o {

  OptimizationAlgorithmWithHessian::OptimizationAlgorithmWithHessian(Solver& solver) :
    OptimizationAlgorithm(),
    _solver(solver)
  {
    _writeDebug = _properties.makeProperty<Property<bool> >("writeDebug", true);
  }

  OptimizationAlgorithmWithHessian::~OptimizationAlgorithmWithHessian()
  {}

  bool OptimizationAlgorithmWithHessian::init(bool online)
  {
    assert(_optimizer && "_optimizer not set");
    _solver.setWriteDebug(_writeDebug->value());
    bool useSchur=false;
    for (OptimizableGraph::VertexContainer::const_iterator it=_optimizer->activeVertices().begin(); it!=_optimizer->activeVertices().end(); ++it) {
      OptimizableGraph::Vertex* v= *it;
      if (v->marginalized()){
        useSchur=true;
        break;
      }
    }
    if (useSchur)
    {
      if  (_solver.supportsSchur())
        _solver.setSchur(true);
    }
    else
    {
      if  (_solver.supportsSchur())
        _solver.setSchur(false);
    }

    bool initState = _solver.init(_optimizer, online);
    return initState;
  }

  bool OptimizationAlgorithmWithHessian::computeMarginals(SparseBlockMatrix<MatrixX>& spinv, const std::vector<std::pair<int, int> >& blockIndices)
  {
    return _solver.computeMarginals(spinv, blockIndices);
  }

  bool OptimizationAlgorithmWithHessian::buildLinearStructure()
  {
    return _solver.buildStructure();
  }

  void OptimizationAlgorithmWithHessian::updateLinearSystem()
  {
    _solver.buildSystem();
  }

  bool OptimizationAlgorithmWithHessian::updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges)
  {
    return _solver.updateStructure(vset, edges);
  }

  void OptimizationAlgorithmWithHessian::setWriteDebug(bool writeDebug)
  {
    _writeDebug->setValue(writeDebug);
  }

} // end namespace
