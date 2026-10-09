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

#ifndef G2O_CREATORS_H
#define G2O_CREATORS_H

#include "hyper_graph.h"

#include <string>
#include <typeinfo>

namespace g2o
{

  /**
   * \brief Abstract interface for allocating HyperGraphElement
   */
  class G2O_CORE_API AbstractHyperGraphElementCreator
  {
    public:
      /**
       * create a hyper graph element. Has to implemented in derived class.
       */
      virtual HyperGraph::HyperGraphElement* construct() = 0;
      /**
       * name of the class to be created. Has to implemented in derived class.
       */
      virtual const std::string& name() const = 0;

      virtual ~AbstractHyperGraphElementCreator() { }
  };

  /**
   * \brief templatized creator class which creates graph elements
   */
  template <typename T>
  class HyperGraphElementCreator : public AbstractHyperGraphElementCreator
  {
    public:
      HyperGraphElementCreator() : _name(typeid(T).name()) {}
#if defined (WINDOWS) && defined(__GNUC__) // force stack alignment on Windows with GCC
      __attribute__((force_align_arg_pointer))
#endif
      HyperGraph::HyperGraphElement* construct() { return new T;}
      virtual const std::string& name() const { return _name;}
    protected:
      std::string _name;
  };

} // end namespace

#endif
