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

#ifndef G2O_OPTMIZATION_ALGORITHM_PROPERTY_H
#define G2O_OPTMIZATION_ALGORITHM_PROPERTY_H

#include "g2o/config.h"
#include "optimization_algorithm_property.h"

#include <list>
#include <iostream>

#include "g2o_core_api.h"

// define to get some verbose output
//#define G2O_DEBUG_OPTIMIZATION_ALGORITHM_FACTORY

namespace g2o {

  // forward decl
  class G2O_CORE_API OptimizationAlgorithm;

  /**
   * \brief base for allocating an optimization algorithm
   *
   * Allocating a solver for a given optimizer. The method construct() has to be
   * implemented in your derived class to allocate the desired solver.
   */
  class G2O_CORE_API AbstractOptimizationAlgorithmCreator
  {
    public:
      AbstractOptimizationAlgorithmCreator(const OptimizationAlgorithmProperty& p);
      virtual ~AbstractOptimizationAlgorithmCreator() { }
      //! allocate a solver operating on optimizer, re-implement for your creator
      virtual OptimizationAlgorithm* construct() = 0;
      //! return the properties of the solver
      const OptimizationAlgorithmProperty& property() const { return _property;}
    protected:
      OptimizationAlgorithmProperty _property;
  };
  
  /**
   * \brief create solvers based on their short name
   *
   * Factory to allocate solvers based on their short name.
   * The Factory is implemented as a sigleton and the single
   * instance can be accessed via the instance() function.
   */
  class G2O_CORE_API OptimizationAlgorithmFactory
  {
    public:
      typedef std::list<AbstractOptimizationAlgorithmCreator*>      CreatorList;

      //! return the instance
      static OptimizationAlgorithmFactory* instance();

      //! free the instance
      static void destroy();

      /**
       * register a specific creator for allocating a solver
       */
      void registerSolver(AbstractOptimizationAlgorithmCreator* c);

      /**
       * unregister a specific creator for allocating a solver
       */
      void unregisterSolver(AbstractOptimizationAlgorithmCreator* c);
      
      /**
       * construct a solver based on its name, e.g., var, fix3_2_cholmod
       */
      OptimizationAlgorithm* construct(const std::string& tag, OptimizationAlgorithmProperty& solverProperty) const;

      //! list the known solvers into a stream
      void listSolvers(std::ostream& os) const;

      //! return the underlying list of creators
      const CreatorList& creatorList() const { return _creator;}

    protected:
      OptimizationAlgorithmFactory();
      ~OptimizationAlgorithmFactory();

      CreatorList _creator;

      CreatorList::const_iterator findSolver(const std::string& name) const;
      CreatorList::iterator findSolver(const std::string& name);

    private:
      static OptimizationAlgorithmFactory* factoryInstance;
  };

  class RegisterOptimizationAlgorithmProxy
  {
    public:
      RegisterOptimizationAlgorithmProxy(AbstractOptimizationAlgorithmCreator* c)
      {
        _creator = c;
#ifdef G2O_DEBUG_OPTIMIZATION_ALGORITHM_FACTORY
        std::cout << __FUNCTION__ << ": Registering " << _creator->property().name << " of type " << typeid(*_creator).name() << std::endl;
#endif
        OptimizationAlgorithmFactory::instance()->registerSolver(c);
      }

      ~RegisterOptimizationAlgorithmProxy()
      {
#ifdef G2O_DEBUG_OPTIMIZATION_ALGORITHM_FACTORY
        std::cout << __FUNCTION__ << ": Unregistering " << _creator->property().name << std::endl;
#endif
        OptimizationAlgorithmFactory::instance()->unregisterSolver(_creator);
      }
    private:
      AbstractOptimizationAlgorithmCreator* _creator;
  };

}

#if defined _MSC_VER && defined G2O_SHARED_LIBS
#  define G2O_OAF_EXPORT __declspec(dllexport)
#  define G2O_OAF_IMPORT __declspec(dllimport)
#else
#  define G2O_OAF_EXPORT
#  define G2O_OAF_IMPORT
#endif

/**
 * Use the following macro to register a whole library of
 * algorithms to the factory, e.g.,
 * G2O_REGISTER_OPTIMIZATION_LIBRARY(csparse)
 * In another programm that links to your library or loads
 * it via dlopen() (or similar functions) you should indicate
 * the desired usage of the solver by using the
 * G2O_REGISTER_OPTIMIZATION_LIBRARY() macro. Here, you
 * should then provide the same name as to the macro before.
 */
#define G2O_REGISTER_OPTIMIZATION_LIBRARY(libraryname) \
    extern "C" void G2O_OAF_EXPORT g2o_optimization_library_##libraryname(void) {}

/**
 * see the documentation to G2O_OAF_EXPORT() above.
 * You should but this into your code, if you expect the factory
 * to be able to allocate a solver but it fails. It enforces linking
 * to the library containing the solver. Hence, the usage of the macro
 * should enforce that the library is actually linked with the binary.
 */
#define G2O_USE_OPTIMIZATION_LIBRARY(libraryname) \
    extern "C" void G2O_OAF_IMPORT g2o_optimization_library_##libraryname(void); \
    static g2o::ForceLinker g2o_force_optimization_algorithm_library_##libraryname(g2o_optimization_library_##libraryname);

/**
 * Similarly to G2O_OAF_IMPORT this macro allows to register a singla
 * more specific algorithm to the library, i.e., gn_var, where gn_var
 * corresponds to a specific instance of csparse based solver for example
 */
#define G2O_REGISTER_OPTIMIZATION_ALGORITHM(optimizername, instance) \
    extern "C" void G2O_OAF_EXPORT g2o_optimization_algorithm_##optimizername(void) {} \
    static g2o::RegisterOptimizationAlgorithmProxy g_optimization_algorithm_proxy_##optimizername(instance);

/**
 * see the documentation of the macros above.
 * It allows to enforce linking to library that contains a specific
 * solver instance and guarantees its usage with the factory
 */
#define G2O_USE_OPTIMIZATION_ALGORITHM(optimizername) \
    extern "C" void G2O_OAF_IMPORT g2o_optimization_algorithm_##optimizername(void); \
    static g2o::ForceLinker g2o_force_optimization_algorithm_link_##optimizername(g2o_optimization_algorithm_##optimizername);

#endif
