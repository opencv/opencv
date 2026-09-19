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

#include "hyper_graph.h"

#include "ownership.h"

#include <assert.h>
#include <queue>
#include <unordered_set>
#include <iterator>

namespace g2o {

  HyperGraph::Data::Data() {
    _next = 0;
    _dataContainer = 0;
  }

  HyperGraph::Data::~Data() {
    delete _next;
  }

  HyperGraph::Vertex::Vertex(int id) : _id(id)
  {
  }

  HyperGraph::Vertex::~Vertex()
  {
  }

  HyperGraph::Edge::Edge(int id) : _id(id)
  {
  }

  HyperGraph::Edge::~Edge()
  {
  }

  int HyperGraph::Edge::numUndefinedVertices() const
  {
    return std::count_if(_vertices.begin(), _vertices.end(), [](const Vertex* ptr) { return ptr == nullptr; });
  }

  void HyperGraph::Edge::resize(size_t size)
  {
    _vertices.resize(size, 0);
  }

  void HyperGraph::Edge::setId(int id)
  {
    _id = id;
  }

  HyperGraph::Vertex* HyperGraph::vertex(int id)
  {
    VertexIDMap::iterator it=_vertices.find(id);
    if (it==_vertices.end())
      return nullptr;
    return it->second;
  }

  const HyperGraph::Vertex* HyperGraph::vertex(int id) const
  {
    VertexIDMap::const_iterator it=_vertices.find(id);
    if (it==_vertices.end())
      return nullptr;
    return it->second;
  }

  bool HyperGraph::addVertex(Vertex* v)
  {
    auto result = _vertices.insert(std::make_pair(v->id(), v));
    return result.second;
  }

  /**
   * changes the id of a vertex already in the graph, and updates the bookkeeping
   @ returns false if the vertex is not in the graph;
  */
  bool HyperGraph::changeId(Vertex* v, int newId){
    Vertex* v2 = vertex(v->id());
    if (v != v2)
      return false;
    _vertices.erase(v->id());
    v->setId(newId);
    _vertices.insert(std::make_pair(v->id(), v));
    return true;
  }

  bool HyperGraph::addEdge(Edge* e)
  { 
    for (Vertex* v : e->vertices())
    { // be sure that all vertices are set
      if (!v)
        return false;
    }

    // check for duplicates in the vertices and do not add this edge
    if (e->vertices().size() == 2) {
      if (e->vertices()[0] == e->vertices()[1])
        return false;
    } else if (e->vertices().size() == 3) {
      if (e->vertices()[0] == e->vertices()[1]
       || e->vertices()[0] == e->vertices()[2]
       || e->vertices()[1] == e->vertices()[2])
        return false;
    } else if (e->vertices().size() > 3) {
      std::unordered_set<Vertex*> vertexPointer;
      std::copy(e->vertices().begin(), e->vertices().end(), std::inserter(vertexPointer, vertexPointer.begin()));
      if (vertexPointer.size() != e->vertices().size())
        return false;
    }

    std::pair<EdgeSet::iterator, bool> result = _edges.insert(e);
    if (!result.second)
      return false;

    for (Vertex* v : e->vertices())
    { // connect the vertices to this edge
      v->edges().insert(e);
    }

    return true;
  }

  bool HyperGraph::setEdgeVertex(HyperGraph::Edge* e, int pos, HyperGraph::Vertex* v)
  {
    Vertex* vOld = e->vertex(pos);
    if (vOld)
      vOld->edges().erase(e);
    e->setVertex(pos, v);
    if (v)
      v->edges().insert(e);
    return true;
  }

  bool HyperGraph::mergeVertices(Vertex* vBig, Vertex* vSmall, bool erase)
  {
    VertexIDMap::iterator it=_vertices.find(vBig->id());
    if (it==_vertices.end())
      return false;

    it=_vertices.find(vSmall->id());
    if (it==_vertices.end())
      return false;

    EdgeSet tmp(vSmall->edges());
    bool ok = true;
    for(EdgeSet::iterator it=tmp.begin(); it!=tmp.end(); ++it){
      HyperGraph::Edge* e = *it;
      for (size_t i=0; i<e->vertices().size(); i++){
        Vertex* v=e->vertex(i);
        if (v==vSmall)
          ok &= setEdgeVertex(e,i,vBig);
      }
    }
    if (erase)
      removeVertex(vSmall);
    return ok;
  }

  bool HyperGraph::detachVertex(Vertex* v){
    VertexIDMap::iterator it=_vertices.find(v->id());
    if (it==_vertices.end())
      return false;
    assert(it->second==v);
    EdgeSet tmp(v->edges());
    for (EdgeSet::iterator it=tmp.begin(); it!=tmp.end(); ++it){
      HyperGraph::Edge* e = *it;
      for (size_t i = 0 ; i<e->vertices().size(); i++){
        if (v == e->vertex(i))
          setEdgeVertex(e,i,0);
      }
    }
    return true;
  }

  bool HyperGraph::removeVertex(Vertex* v, bool detach)
  {
    if (detach){
      bool result = detachVertex(v);
      if (! result) {
        assert (0 && "inconsistency in detaching vertex, ");
      }
    }
    VertexIDMap::iterator it=_vertices.find(v->id());
    if (it==_vertices.end())
      return false;
    assert(it->second==v);
    //remove all edges which are entering or leaving v;
    EdgeSet tmp(v->edges());
    for (EdgeSet::iterator it=tmp.begin(); it!=tmp.end(); ++it){
      if (!removeEdge(*it)){
        assert(0 && "error in erasing vertex");
      }
    }
    _vertices.erase(it);
    release(v);
    return true;
  }

  bool HyperGraph::removeEdge(Edge* e)
  {
    EdgeSet::iterator it = _edges.find(e);
    if (it == _edges.end())
      return false;
    _edges.erase(it);
    for (std::vector<Vertex*>::iterator vit = e->vertices().begin(); vit != e->vertices().end(); ++vit) {
      Vertex* v = *vit;
      if (!v)
        continue;
      it = v->edges().find(e);
      assert(it!=v->edges().end());
      v->edges().erase(it);
    }
    release(e);
    return true;
  }

  HyperGraph::HyperGraph()
  {
  }

  void HyperGraph::clear()
  {
#if G2O_DELETE_IMPLICITLY_OWNED_OBJECTS
    for (VertexIDMap::iterator it=_vertices.begin(); it!=_vertices.end(); ++it)
      delete (it->second);
    for (EdgeSet::iterator it=_edges.begin(); it!=_edges.end(); ++it)
      delete (*it);
#endif

    _vertices.clear();
    _edges.clear();
  }

  HyperGraph::~HyperGraph()
  {
    HyperGraph::clear();
  }

} // end namespace
