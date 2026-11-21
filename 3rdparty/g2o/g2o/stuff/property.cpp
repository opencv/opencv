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

#include "property.h"

#include <vector>
#include <iostream>

#include "macros.h"

#include "string_tools.h"
using namespace std;

namespace g2o {

  BaseProperty::BaseProperty(const std::string name_) :_name(name_){
  }

  BaseProperty::~BaseProperty(){}

  bool PropertyMap::addProperty(BaseProperty* p) {
    std::pair<PropertyMapIterator,bool> result = insert(make_pair(p->name(), p));
    return result.second;
  }

  bool PropertyMap::eraseProperty(const std::string& name) {
    PropertyMapIterator it=find(name);
    if (it==end())
      return false;
    delete it->second;
    erase(it);
    return true;
  }

  PropertyMap::~PropertyMap() {
    for (PropertyMapIterator it=begin(); it!=end(); it++){
      if (it->second)
        delete it->second;
    }
  }

  bool PropertyMap::updatePropertyFromString(const std::string& name, const std::string& value)
  {
    PropertyMapIterator it = find(name);
    if (it == end())
      return false;
    it->second->fromString(value);
    return true;
  }

  void PropertyMap::writeToCSV(std::ostream& os) const
  {
    for (PropertyMapConstIterator it=begin(); it!=end(); it++){
      BaseProperty* p =it->second;
      os << p->name() << ", ";
    }
    os << std::endl;
    for (PropertyMapConstIterator it=begin(); it!=end(); it++){
      BaseProperty* p =it->second;
      os << p->toString() << ", ";
    }
    os << std::endl;
  }

  bool PropertyMap::updateMapFromString(const std::string& values)
  {
    bool status = true;
    vector<string> valuesMap = strSplit(values, ",");
    for (size_t i = 0; i < valuesMap.size(); ++i) {
      vector<string> m = strSplit(valuesMap[i], "=");
      if (m.size() != 2) {
        cerr << __PRETTY_FUNCTION__ << ": unable to extract name=value pair from " << valuesMap[i] << endl;
        continue;
      }
      string name = trim(m[0]);
      string value = trim(m[1]);
      status = status && updatePropertyFromString(name, value);
    }
    return status;
  }

} // end namespace
