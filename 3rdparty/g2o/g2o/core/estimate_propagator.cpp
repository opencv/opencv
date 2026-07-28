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

#include "estimate_propagator.h"

#include <queue>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <fstream>

//#define DEBUG_ESTIMATE_PROPAGATOR

using namespace std;

namespace g2o {

# ifdef DEBUG_ESTIMATE_PROPAGATOR
  struct FrontierLevelCmp {
    bool operator()(EstimatePropagator::AdjacencyMapEntry* e1, EstimatePropagator::AdjacencyMapEntry* e2) const
    {
      return e1->frontierLevel() < e2->frontierLevel();
    }
  };
# endif

  EstimatePropagator::AdjacencyMapEntry::AdjacencyMapEntry()
  {
    reset();
  }

  void EstimatePropagator::AdjacencyMapEntry::reset()
  {
    _child = 0;
    _parent.clear();
    _edge = 0;
    _distance = numeric_limits<number_t>::max();
    _frontierLevel = -1;
    inQueue = false;
  }

  EstimatePropagator::EstimatePropagator(OptimizableGraph* g): _graph(g)
  {
    for (OptimizableGraph::VertexIDMap::const_iterator it=_graph->vertices().begin(); it!=_graph->vertices().end(); ++it){
      AdjacencyMapEntry entry;
      entry._child = static_cast<OptimizableGraph::Vertex*>(it->second);
      _adjacencyMap.insert(make_pair(entry.child(), entry));
    }
  }

  void EstimatePropagator::reset()
  {
    for (OptimizableGraph::VertexSet::iterator it=_visited.begin(); it!=_visited.end(); ++it){
      OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(*it);
      AdjacencyMap::iterator at = _adjacencyMap.find(v);
      assert(at != _adjacencyMap.end());
      at->second.reset();
    }
    _visited.clear();
  }

  void EstimatePropagator::propagate(OptimizableGraph::Vertex* v, 
      const EstimatePropagator::PropagateCost& cost, 
       const EstimatePropagator::PropagateAction& action,
       number_t maxDistance, 
       number_t maxEdgeCost)
  {
    OptimizableGraph::VertexSet vset;
    vset.insert(v);
    propagate(vset, cost, action, maxDistance, maxEdgeCost);
  }

  void EstimatePropagator::propagate(OptimizableGraph::VertexSet& vset, 
      const EstimatePropagator::PropagateCost& cost, 
       const EstimatePropagator::PropagateAction& action,
       number_t maxDistance, 
       number_t maxEdgeCost)
  {
    reset();

    PriorityQueue frontier;
    for (OptimizableGraph::VertexSet::iterator vit=vset.begin(); vit!=vset.end(); ++vit){
      OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(*vit);
      AdjacencyMap::iterator it = _adjacencyMap.find(v);
      assert(it != _adjacencyMap.end());
      it->second._distance = 0.;
      it->second._parent.clear();
      it->second._frontierLevel = 0;
      frontier.push(&it->second);
    }

    while(! frontier.empty()){
      AdjacencyMapEntry* entry = frontier.pop();
      OptimizableGraph::Vertex* u = entry->child();
      number_t uDistance = entry->distance();
      //cerr << "uDistance " << uDistance << endl;

      // initialize the vertex
      if (entry->_frontierLevel > 0) {
        action(entry->edge(), entry->parent(), u);
      }

      /* std::pair< OptimizableGraph::VertexSet::iterator, bool> insertResult = */ _visited.insert(u);
      OptimizableGraph::EdgeSet::iterator et = u->edges().begin();
      while (et != u->edges().end()){
        OptimizableGraph::Edge* edge = static_cast<OptimizableGraph::Edge*>(*et);
        ++et;

        int maxFrontier = -1;
        OptimizableGraph::VertexSet initializedVertices;
        for (size_t i = 0; i < edge->vertices().size(); ++i) {
          OptimizableGraph::Vertex* z = static_cast<OptimizableGraph::Vertex*>(edge->vertex(i));
	  if (! z)
	    continue;
          AdjacencyMap::iterator ot = _adjacencyMap.find(z);
          if (ot->second._distance != numeric_limits<number_t>::max()) {
            initializedVertices.insert(z);
            maxFrontier = (max)(maxFrontier, ot->second._frontierLevel);
          }
        }
        assert(maxFrontier >= 0);

        for (size_t i = 0; i < edge->vertices().size(); ++i) {
          OptimizableGraph::Vertex* z = static_cast<OptimizableGraph::Vertex*>(edge->vertex(i));
	  if (! z)
	    continue;
          if (z == u)
            continue;
          size_t wasInitialized = initializedVertices.erase(z);

          number_t edgeDistance = cost(edge, initializedVertices, z);
          if (edgeDistance > 0. && edgeDistance != std::numeric_limits<number_t>::max() && edgeDistance < maxEdgeCost) {
            number_t zDistance = uDistance + edgeDistance;
            //cerr << z->id() << " " << zDistance << endl;

            AdjacencyMap::iterator ot = _adjacencyMap.find(z);
            assert(ot!=_adjacencyMap.end());

            if (zDistance < ot->second.distance() && zDistance < maxDistance){
              //if (ot->second.inQueue)
                //cerr << "Updating" << endl;
              ot->second._distance = zDistance;
              ot->second._parent = initializedVertices;
              ot->second._edge = edge;
              ot->second._frontierLevel = maxFrontier + 1;
              frontier.push(&ot->second);
            }
          }

          if (wasInitialized > 0)
            initializedVertices.insert(z);

        }
      }
    }

    // writing debug information like cost for reaching each vertex and the parent used to initialize
#ifdef DEBUG_ESTIMATE_PROPAGATOR
    cerr << "Writing cost.dat" << endl;
    ofstream costStream("cost.dat");
    for (AdjacencyMap::const_iterator it = _adjacencyMap.begin(); it != _adjacencyMap.end(); ++it) {
      HyperGraph::Vertex* u = it->second.child();
      costStream << "vertex " << u->id() << "  cost " << it->second._distance << endl;
    }
    cerr << "Writing init.dat" << endl;
    ofstream initStream("init.dat");
    vector<AdjacencyMapEntry*> frontierLevels;
    for (AdjacencyMap::iterator it = _adjacencyMap.begin(); it != _adjacencyMap.end(); ++it) {
      if (it->second._frontierLevel > 0)
        frontierLevels.push_back(&it->second);
    }
    sort(frontierLevels.begin(), frontierLevels.end(), FrontierLevelCmp());
    for (vector<AdjacencyMapEntry*>::const_iterator it = frontierLevels.begin(); it != frontierLevels.end(); ++it) {
      AdjacencyMapEntry* entry       = *it;
      OptimizableGraph::Vertex* to   = entry->child();

      initStream << "calling init level = " << entry->_frontierLevel << "\t (";
      for (OptimizableGraph::VertexSet::iterator pit = entry->parent().begin(); pit != entry->parent().end(); ++pit) {
        initStream << " " << (*pit)->id();
      }
      initStream << " ) -> " << to->id() << endl;
    }
#endif

  }

  void EstimatePropagator::PriorityQueue::push(AdjacencyMapEntry* entry)
  {
    assert(entry != NULL);
    if (entry->inQueue) {
      assert(entry->queueIt->second == entry);
      erase(entry->queueIt);
    }

    entry->queueIt = insert(std::make_pair(entry->distance(), entry));
    assert(entry->queueIt != end());
    entry->inQueue = true;
  }

  EstimatePropagator::AdjacencyMapEntry* EstimatePropagator::PriorityQueue::pop()
  {
    assert(!empty());
    iterator it = begin();
    AdjacencyMapEntry* entry = it->second;
    erase(it);

    assert(entry != NULL);
    entry->queueIt = end();
    entry->inQueue = false;
    return entry;
  }

  EstimatePropagatorCost::EstimatePropagatorCost (SparseOptimizer* graph) :
    _graph(graph)
  {
  }

  number_t EstimatePropagatorCost::operator()(OptimizableGraph::Edge* edge, const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to_) const
  {
    OptimizableGraph::Edge* e = dynamic_cast<OptimizableGraph::Edge*>(edge);
    OptimizableGraph::Vertex* to = dynamic_cast<OptimizableGraph::Vertex*>(to_);
    SparseOptimizer::EdgeContainer::const_iterator it = _graph->findActiveEdge(e);
    if (it == _graph->activeEdges().end()) // it has to be an active edge
      return std::numeric_limits<number_t>::max();
    return e->initialEstimatePossible(from, to);
  }

  EstimatePropagatorCostOdometry::EstimatePropagatorCostOdometry(SparseOptimizer* graph) :
    EstimatePropagatorCost(graph)
  {
  }

  number_t EstimatePropagatorCostOdometry::operator()(OptimizableGraph::Edge* edge, const OptimizableGraph::VertexSet& from_, OptimizableGraph::Vertex* to_) const
  {
    OptimizableGraph::Edge* e = dynamic_cast<OptimizableGraph::Edge*>(edge);
    OptimizableGraph::Vertex* from = dynamic_cast<OptimizableGraph::Vertex*>(*from_.begin());
    OptimizableGraph::Vertex* to = dynamic_cast<OptimizableGraph::Vertex*>(to_);
    if (std::abs(from->id() - to->id()) != 1) // simple method to identify odometry edges in a pose graph
      return std::numeric_limits<number_t>::max();
    SparseOptimizer::EdgeContainer::const_iterator it = _graph->findActiveEdge(e);
    if (it == _graph->activeEdges().end()) // it has to be an active edge
      return std::numeric_limits<number_t>::max();
    return e->initialEstimatePossible(from_, to);
  }

} // end namespace
