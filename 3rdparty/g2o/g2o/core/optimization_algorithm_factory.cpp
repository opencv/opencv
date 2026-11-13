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

#include "optimization_algorithm_factory.h"

#include <iostream>
#include <typeinfo>
#include <cassert>

using namespace std;

namespace g2o {

  AbstractOptimizationAlgorithmCreator::AbstractOptimizationAlgorithmCreator(const OptimizationAlgorithmProperty& p) :
    _property(p)
  {
  }

  OptimizationAlgorithmFactory* OptimizationAlgorithmFactory::factoryInstance = 0;

  OptimizationAlgorithmFactory::OptimizationAlgorithmFactory()
  {
  }

  OptimizationAlgorithmFactory::~OptimizationAlgorithmFactory()
  {
    for (CreatorList::iterator it = _creator.begin(); it != _creator.end(); ++it)
      delete *it;
  }

  OptimizationAlgorithmFactory* OptimizationAlgorithmFactory::instance()
  {
    if (factoryInstance == 0) {
      factoryInstance = new OptimizationAlgorithmFactory;
    }
    return factoryInstance;
  }

  void OptimizationAlgorithmFactory::registerSolver(AbstractOptimizationAlgorithmCreator* c)
  {
    const string& name = c->property().name;
    CreatorList::iterator foundIt = findSolver(name);
    if (foundIt != _creator.end()) {
      _creator.erase(foundIt);
      cerr << "SOLVER FACTORY WARNING: Overwriting Solver creator " << name << endl;
      assert(0);
    }
    _creator.push_back(c);
  }

  void OptimizationAlgorithmFactory::unregisterSolver(AbstractOptimizationAlgorithmCreator* c)
  {
    const string& name = c->property().name;
    CreatorList::iterator foundIt = findSolver(name);
    if (foundIt != _creator.end()) {
      delete *foundIt;
      _creator.erase(foundIt);
    }
  }

  OptimizationAlgorithm* OptimizationAlgorithmFactory::construct(const std::string& name, OptimizationAlgorithmProperty& solverProperty) const
  {
    CreatorList::const_iterator foundIt = findSolver(name);
    if (foundIt != _creator.end()) {
      solverProperty = (*foundIt)->property();
      return (*foundIt)->construct();
    }
    cerr << "SOLVER FACTORY WARNING: Unable to create solver " << name << endl;
    return nullptr;
  }

  void OptimizationAlgorithmFactory::destroy()
  {
    delete factoryInstance;
    factoryInstance = nullptr;
  }

  void OptimizationAlgorithmFactory::listSolvers(std::ostream& os) const
  {
    size_t solverNameColumnLength = 0;
    for (CreatorList::const_iterator it = _creator.begin(); it != _creator.end(); ++it)
      solverNameColumnLength = std::max(solverNameColumnLength, (*it)->property().name.size());
    solverNameColumnLength += 4;

    for (CreatorList::const_iterator it = _creator.begin(); it != _creator.end(); ++it) {
      const OptimizationAlgorithmProperty& sp = (*it)->property();
      os << sp.name;
      for (size_t i = sp.name.size(); i < solverNameColumnLength; ++i)
        os << ' ';
      os << sp.desc << endl;
    }
  }

  OptimizationAlgorithmFactory::CreatorList::const_iterator OptimizationAlgorithmFactory::findSolver(const std::string& name) const
  {
    for (CreatorList::const_iterator it = _creator.begin(); it != _creator.end(); ++it) {
      const OptimizationAlgorithmProperty& sp = (*it)->property();
      if (sp.name == name)
        return it;
    }
    return _creator.end();
  }

  OptimizationAlgorithmFactory::CreatorList::iterator OptimizationAlgorithmFactory::findSolver(const std::string& name)
  {
    for (CreatorList::iterator it = _creator.begin(); it != _creator.end(); ++it) {
      const OptimizationAlgorithmProperty& sp = (*it)->property();
      if (sp.name == name)
        return it;
    }
    return _creator.end();
  }

} // end namespace
