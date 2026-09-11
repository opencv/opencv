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

#include "jacobian_workspace.h"

#include <cmath>

#include "optimizable_graph.h"

using namespace std;

namespace g2o {

JacobianWorkspace::JacobianWorkspace() :
  _maxNumVertices(-1), _maxDimension(-1)
{
}

JacobianWorkspace::~JacobianWorkspace()
{
}

bool JacobianWorkspace::allocate()
{
  //cerr << __PRETTY_FUNCTION__ << " " << PVAR(this) << " " << PVAR(_maxNumVertices) << " " << PVAR(_maxDimension) << endl;
  if (_maxNumVertices <=0 || _maxDimension <= 0)
    return false;
  _workspace.resize(_maxNumVertices);
  for (WorkspaceVector::iterator it = _workspace.begin(); it != _workspace.end(); ++it) {
    it->resize(_maxDimension);
    it->setZero();
  }
  return true;
}

void JacobianWorkspace::updateSize(const HyperGraph::Edge* e_)
{
  const OptimizableGraph::Edge* e = static_cast<const OptimizableGraph::Edge*>(e_);
  int errorDimension = e->dimension();
  int numVertices = e->vertices().size();
  int maxDimensionForEdge = -1;
  for (int i = 0; i < numVertices; ++i) {
    const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(e->vertex(i));
    assert(v && "Edge has no vertex assigned");
    maxDimensionForEdge = max(v->dimension() * errorDimension, maxDimensionForEdge);
  }
  _maxNumVertices = max(numVertices, _maxNumVertices);
  _maxDimension = max(maxDimensionForEdge, _maxDimension);
  //cerr << __PRETTY_FUNCTION__ << " " << PVAR(this) << " " << PVAR(_maxNumVertices) << " " << PVAR(_maxDimension) << endl;
}

void JacobianWorkspace::updateSize(const OptimizableGraph& graph)
{
  for (OptimizableGraph::EdgeSet::const_iterator it = graph.edges().begin(); it != graph.edges().end(); ++it) {
    const OptimizableGraph::Edge* e = static_cast<const OptimizableGraph::Edge*>(*it);
    updateSize(e);
  }
}

void JacobianWorkspace::updateSize(int numVertices, int dimension)
{
  _maxNumVertices = max(numVertices, _maxNumVertices);
  _maxDimension = max(dimension, _maxDimension);
}

} // end namespace
