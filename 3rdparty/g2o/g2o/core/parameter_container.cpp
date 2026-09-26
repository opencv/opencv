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

#include "parameter_container.h"

#include <iostream>

#include "factory.h"
#include "parameter.h"

#include "g2o/stuff/macros.h"
#include "g2o/stuff/color_macros.h"
#include "g2o/stuff/string_tools.h"

namespace g2o {

  using namespace std;

  ParameterContainer::ParameterContainer(bool isMainStorage_) :
    _isMainStorage(isMainStorage_)
  {
  }

  void ParameterContainer::clear() {
    if (!_isMainStorage)
      return;
    for (iterator it = begin(); it!=end(); it++){
      delete it->second;
    }
    BaseClass::clear();
  }

  ParameterContainer::~ParameterContainer(){
    clear();
  }

  bool ParameterContainer::addParameter(Parameter* p){
    if (p->id()<0)
      return false;
    iterator it=find(p->id());
    if (it!=end())
      return false;
    insert(make_pair(p->id(), p));
    return true;
  }

  Parameter* ParameterContainer::getParameter(int id) {
    iterator it=find(id);
    if (it==end())
      return nullptr;
    return it->second;
  }

  const Parameter* ParameterContainer::getParameter(int id) const {
    const_iterator it=find(id);
    if (it==end())
      return nullptr;
    return it->second;
  }

  Parameter* ParameterContainer::detachParameter(int id){
    iterator it=find(id);
    if (it==end())
      return nullptr;
    Parameter* p=it->second;
    erase(it);
    return p;
  }
  
  bool ParameterContainer::write(std::ostream& os) const{
    Factory* factory = Factory::instance();
    for (const_iterator it=begin(); it!=end(); it++){
      os << factory->tag(it->second) << " ";
      os << it->second->id() << " ";
      it->second->write(os);
      os << endl;
    }
    return true;
  }

  bool ParameterContainer::read(std::istream& is, const std::map<std::string, std::string>* _renamedTypesLookup){
    stringstream currentLine;
    string token;

    Factory* factory = Factory::instance();
    HyperGraph::GraphElemBitset elemBitset;
    elemBitset[HyperGraph::HGET_PARAMETER] = 1;
    
    while (1) {
      int bytesRead = readLine(is, currentLine);
      if (bytesRead == -1)
        break;
      currentLine >> token;
      if (bytesRead == 0 || token.size() == 0 || token[0] == '#')
        continue;
      if (_renamedTypesLookup && _renamedTypesLookup->size()>0){
	map<string, string>::const_iterator foundIt = _renamedTypesLookup->find(token);
	if (foundIt != _renamedTypesLookup->end()) {
	  token = foundIt->second;
	}
      }

      HyperGraph::HyperGraphElement* element = factory->construct(token, elemBitset);
      if (! element) // not a parameter or otherwise unknown tag
        continue;
      assert(element->elementType() == HyperGraph::HGET_PARAMETER && "Should be a param");

      Parameter* p = static_cast<Parameter*>(element);
      int pid;
      currentLine >> pid;
      p->setId(pid);
      bool r = p->read(currentLine);
      if (! r) {
        cerr << __PRETTY_FUNCTION__ << ": Error reading data " << token << " for parameter " << pid << endl;
        delete p;
      } else {
        if (! addParameter(p) ){
          cerr << __PRETTY_FUNCTION__ << ": Parameter of type:" << token << " id:" << pid << " already defined" << endl;
        }
      }
    } // while read line
    
    return true;
  }
  
} // end namespace
