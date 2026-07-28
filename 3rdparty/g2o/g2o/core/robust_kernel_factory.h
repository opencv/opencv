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

#ifndef G2O_ROBUST_KERNEL_FACTORY_H
#define G2O_ROBUST_KERNEL_FACTORY_H

#include "g2o_core_api.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>

namespace g2o {

  class RobustKernel;

    /**
   * \brief Abstract interface for allocating a robust kernel
   */
  class G2O_CORE_API AbstractRobustKernelCreator
  {
    public:
      /**
       * create a hyper graph element. Has to implemented in derived class.
       */
      virtual RobustKernel* construct() = 0;
      virtual ~AbstractRobustKernelCreator() { }
  };

  /**
   * \brief templatized creator class which creates graph elements
   */
  template <typename T>
  class RobustKernelCreator : public AbstractRobustKernelCreator
  {
    public:
      RobustKernel* construct() { return new T;}
  };

  /**
   * \brief create robust kernels based on their human readable name
   */
  class G2O_CORE_API RobustKernelFactory
  {
    public:

      //! return the instance
      static RobustKernelFactory* instance();

      //! free the instance
      static void destroy();

      /**
       * register a tag for a specific creator
       */
      void registerRobustKernel(const std::string& tag, AbstractRobustKernelCreator* c);

      /**
       * unregister a tag for a specific creator
       */
      void unregisterType(const std::string& tag);

      /**
       * construct a robust kernel based on its tag
       */
      RobustKernel* construct(const std::string& tag) const;

      /**
       * return the creator for a specific tag
       */
      AbstractRobustKernelCreator* creator(const std::string& tag) const;

      /**
       * get a list of all known robust kernels
       */
      void fillKnownKernels(std::vector<std::string>& types) const;

    protected:

      typedef std::map<std::string, AbstractRobustKernelCreator*>              CreatorMap;
      RobustKernelFactory();
      ~RobustKernelFactory();

      CreatorMap _creator;     ///< look-up map for the existing creators

    private:
      static RobustKernelFactory* factoryInstance;
  };

  template<typename T>
  class RegisterRobustKernelProxy
  {
    public:
      RegisterRobustKernelProxy(const std::string& name) : _name(name)
      {
        RobustKernelFactory::instance()->registerRobustKernel(_name, new RobustKernelCreator<T>());
      }

      ~RegisterRobustKernelProxy()
      {
        RobustKernelFactory::instance()->unregisterType(_name);
      }

    private:
      std::string _name;
  };

#if defined _MSC_VER && defined G2O_SHARED_LIBS
#  define G2O_ROBUST_KERNEL_FACTORY_EXPORT __declspec(dllexport)
#  define G2O_ROBUST_KERNEL_FACTORY_IMPORT __declspec(dllimport)
#else
#  define G2O_ROBUST_KERNEL_FACTORY_EXPORT
#  define G2O_ROBUST_KERNEL_FACTORY_IMPORT
#endif

  // These macros are used to automate registering of robust kernels and forcing linkage
#define G2O_REGISTER_ROBUST_KERNEL(name, classname) \
    extern "C" void G2O_ROBUST_KERNEL_FACTORY_EXPORT g2o_robust_kernel_##classname(void) {} \
    static g2o::RegisterRobustKernelProxy<classname> g_robust_kernel_proxy_##classname(#name);

#define G2O_USE_ROBUST_KERNEL(classname) \
    extern "C" void G2O_ROBUST_KERNEL_FACTORY_IMPORT g2o_robust_kernel_##classname(void); \
    static g2o::ForceLinker g2o_force_robust_kernel_link_##classname(g2o_robust_kernel_##classname);

} // end namespace g2o

#endif
