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

#ifndef JACOBIAN_WORKSPACE_H
#define JACOBIAN_WORKSPACE_H

#include <Eigen/Core>

#include <vector>
#include <cassert>

#include "g2o/config.h"
#include "g2o_core_api.h"
#include "hyper_graph.h"

namespace g2o {

  struct OptimizableGraph;

  /**
   * \brief provide memory workspace for computing the Jacobians
   *
   * The workspace is used by an OptimizableGraph to provide temporary memory
   * for computing the Jacobian of the error functions.
   * Before calling linearizeOplus on an edge, the workspace needs to be allocated
   * by calling allocate().
   */
  class G2O_CORE_API JacobianWorkspace
  {
    public:
      typedef std::vector<VectorX, Eigen::aligned_allocator<VectorX> >      WorkspaceVector;

    public:
      JacobianWorkspace();
      ~JacobianWorkspace();

      /**
       * allocate the workspace
       */
      bool allocate();

      /**
       * update the maximum required workspace needed by taking into account this edge
       */
      void updateSize(const HyperGraph::Edge* e);

      /**
       * update the required workspace by looking at a full graph
       */
      void updateSize(const OptimizableGraph& graph);

      /**
       * manually update with the given parameters
       */
      void updateSize(int numVertices, int dimension);

      /**
       * return the workspace for a vertex in an edge
       */
      number_t* workspaceForVertex(int vertexIndex)
      {
        assert(vertexIndex >= 0 && (size_t)vertexIndex < _workspace.size() && "Index out of bounds");
        return _workspace[vertexIndex].data();
      }

    protected:
      WorkspaceVector _workspace;   ///< the memory pre-allocated for computing the Jacobians
      int _maxNumVertices;          ///< the maximum number of vertices connected by a hyper-edge
      int _maxDimension;            ///< the maximum dimension (number of elements) for a Jacobian
  };

} // end namespace

#endif
