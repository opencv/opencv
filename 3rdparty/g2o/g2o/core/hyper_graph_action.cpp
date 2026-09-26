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

#include "hyper_graph_action.h"
#include "optimizable_graph.h"
#include "cache.h"
#include "g2o/stuff/macros.h"


#include <iostream>
#include <list>

namespace g2o {
  using namespace std;

  HyperGraphActionLibrary* HyperGraphActionLibrary::actionLibInstance = 0;

  HyperGraphAction::Parameters::~Parameters()
  {
  }

  HyperGraphAction::ParametersIteration::ParametersIteration(int iter) :
    HyperGraphAction::Parameters(),
    iteration(iter)
  {
  }

  HyperGraphAction::~HyperGraphAction()
  {
  }

  HyperGraphAction* HyperGraphAction::operator()(const HyperGraph*, Parameters*)
  {
    return nullptr;
  }

  HyperGraphElementAction::Parameters::~Parameters()
  {
  }

  HyperGraphElementAction::HyperGraphElementAction(const std::string& typeName_)
  {
    _typeName = typeName_;
  }

  void HyperGraphElementAction::setTypeName(const std::string& typeName_)
  {
    _typeName = typeName_;
  }


  HyperGraphElementAction* HyperGraphElementAction::operator()(HyperGraph::HyperGraphElement* , HyperGraphElementAction::Parameters* )
  {
    return nullptr;
  }
  
  HyperGraphElementAction* HyperGraphElementAction::operator()(const HyperGraph::HyperGraphElement* , HyperGraphElementAction::Parameters* )
  {
    return nullptr;
  }
  
  HyperGraphElementAction::~HyperGraphElementAction()
  {
  }

  HyperGraphElementActionCollection::HyperGraphElementActionCollection(const std::string& name_)
  {
    _name = name_;
  }

  HyperGraphElementActionCollection::~HyperGraphElementActionCollection()
  {
    for (ActionMap::iterator it = _actionMap.begin(); it != _actionMap.end(); ++it) {
      delete it->second;
    }
  }

  HyperGraphElementAction* HyperGraphElementActionCollection::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params)
  {
    ActionMap::iterator it=_actionMap.find(typeid(*element).name());
    //cerr << typeid(*element).name() << endl;
    if (it==_actionMap.end())
      return nullptr;
    HyperGraphElementAction* action=it->second;
    return (*action)(element, params);
  }

  HyperGraphElementAction* HyperGraphElementActionCollection::operator()(const HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params)
  {
    ActionMap::iterator it=_actionMap.find(typeid(*element).name());
    if (it==_actionMap.end())
      return nullptr;
    HyperGraphElementAction* action=it->second;
    return (*action)(element, params);
  }

  bool HyperGraphElementActionCollection::registerAction(HyperGraphElementAction* action)
  {
#  ifdef G2O_DEBUG_ACTIONLIB
    cerr << __PRETTY_FUNCTION__ << " " << action->name() << " " << action->typeName() << endl;
#  endif
    if (action->name()!=name()){
      cerr << __PRETTY_FUNCTION__  << ": invalid attempt to register an action in a collection with a different name " <<  name() << " " << action->name() << endl;
    }
    _actionMap.insert(make_pair ( action->typeName(), action) );
    return true;
  }

  bool HyperGraphElementActionCollection::unregisterAction(HyperGraphElementAction* action)
  {
    for (HyperGraphElementAction::ActionMap::iterator it=_actionMap.begin(); it != _actionMap.end(); ++it) {
      if (it->second == action){
        _actionMap.erase(it);
        return true;
      }
    }
    return false;
  }
  
  HyperGraphActionLibrary::HyperGraphActionLibrary()
  {
  }

  HyperGraphActionLibrary* HyperGraphActionLibrary::instance()
  {
    if (actionLibInstance == 0) {
      actionLibInstance = new HyperGraphActionLibrary;
    }
    return actionLibInstance;
  }

  void HyperGraphActionLibrary::destroy()
  {
    delete actionLibInstance;
    actionLibInstance = 0;
  }

  HyperGraphActionLibrary::~HyperGraphActionLibrary()
  {
    // memory is freed by Proxy
    //for (HyperGraphElementAction::ActionMap::iterator it = _actionMap.begin(); it != _actionMap.end(); ++it) {
      //delete it->second;
    //}
  }
  
  HyperGraphElementAction* HyperGraphActionLibrary::actionByName(const std::string& name)
  {
    HyperGraphElementAction::ActionMap::iterator it=_actionMap.find(name);
    if (it!=_actionMap.end())
      return it->second;
    return nullptr;
  }

  bool HyperGraphActionLibrary::registerAction(HyperGraphElementAction* action)
  {
    HyperGraphElementAction* oldAction = actionByName(action->name());
    HyperGraphElementActionCollection* collection = 0;
    if (oldAction) {
      collection = dynamic_cast<HyperGraphElementActionCollection*>(oldAction);
      if (! collection) {
        cerr << __PRETTY_FUNCTION__ << ": fatal error, a collection is not at the first level in the library" << endl;
        return false;
      }
    }
    if (! collection) {
#ifdef G2O_DEBUG_ACTIONLIB
      cerr << __PRETTY_FUNCTION__ << ": creating collection for \"" << action->name() << "\"" << endl;
#endif
      collection = new HyperGraphElementActionCollection(action->name());
      _actionMap.insert(make_pair(action->name(), collection));
    }
    return collection->registerAction(action);
  }
  
  bool HyperGraphActionLibrary::unregisterAction(HyperGraphElementAction* action)
  {
    list<HyperGraphElementActionCollection*> collectionDeleteList;

    // Search all the collections and delete the registered actions; if a collection becomes empty, schedule it for deletion; note that we can't delete the collections as we go because this will screw up the state of the iterators
    for (HyperGraphElementAction::ActionMap::iterator it=_actionMap.begin(); it != _actionMap.end(); ++it) {
      HyperGraphElementActionCollection* collection = dynamic_cast<HyperGraphElementActionCollection*> (it->second);
      if (collection != 0) {
        collection->unregisterAction(action);
        if (collection->actionMap().size() == 0) {
          collectionDeleteList.push_back(collection);
        }
      }
    }

    // Delete any empty action collections
    for (list<HyperGraphElementActionCollection*>::iterator itc = collectionDeleteList.begin(); itc != collectionDeleteList.end(); ++itc) {
      //cout << "Deleting collection " << (*itc)->name() << endl;
      _actionMap.erase((*itc)->name());
    }

    return true;
  }


  WriteGnuplotAction::WriteGnuplotAction(const std::string& typeName_)
    : HyperGraphElementAction(typeName_)
  {
    _name="writeGnuplot";
  }

  DrawAction::Parameters::Parameters(){
  }

  DrawAction::DrawAction(const std::string& typeName_) 
    : HyperGraphElementAction(typeName_)
  {
    _name="draw";
    _previousParams = (Parameters*)0x42;
    refreshPropertyPtrs(0);
    _cacheDrawActions = 0;
  }

  bool DrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (_previousParams == params_)
      return false;
    DrawAction::Parameters* p=dynamic_cast<DrawAction::Parameters*>(params_);
    if (! p){
      _previousParams = 0;
      _show = 0;
      _showId = 0;
    } else {
      _previousParams = p;
      _show = p->makeProperty<BoolProperty>(_typeName+"::SHOW", true);
      _showId = p->makeProperty<BoolProperty>(_typeName+"::SHOW_ID", false);
    }
    return true;
  }

  void DrawAction::initializeDrawActionsCache() {
    if (! _cacheDrawActions){
      _cacheDrawActions = HyperGraphActionLibrary::instance()->actionByName("draw");
    }
  }

  void DrawAction::drawCache(CacheContainer* caches, HyperGraphElementAction::Parameters* params_) {
    if (caches){
      for (CacheContainer::iterator it=caches->begin(); it!=caches->end(); it++){
        Cache* c = it->second;
        (*_cacheDrawActions)(c, params_);
      }
    }
  }

  void DrawAction::drawUserData(HyperGraph::Data* data, HyperGraphElementAction::Parameters* params_){
    while (data && _cacheDrawActions ){
      (*_cacheDrawActions)(data, params_);
      data=data->next();
    }
  }

  void applyAction(HyperGraph* graph, HyperGraphElementAction* action, HyperGraphElementAction::Parameters* params, const std::string& typeName)
  {
    for (HyperGraph::VertexIDMap::iterator it=graph->vertices().begin(); 
        it!=graph->vertices().end(); ++it){
      if ( typeName.empty() || typeid(*it->second).name()==typeName){
        (*action)(it->second, params);
      }
    }
    for (HyperGraph::EdgeSet::iterator it=graph->edges().begin(); 
        it!=graph->edges().end(); ++it){
      if ( typeName.empty() || typeid(**it).name()==typeName)
        (*action)(*it, params);
    }
  }

} // end namespace
