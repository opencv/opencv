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

#ifndef G2O_HYPER_GRAPH_ACTION_H
#define G2O_HYPER_GRAPH_ACTION_H

#include "hyper_graph.h"
#include "g2o/stuff/property.h"

#include <typeinfo>
#include <iosfwd>
#include <string>
#include <iostream>

#include "g2o_core_api.h"

// define to get verbose output
//#define G2O_DEBUG_ACTIONLIB

namespace g2o {

  class CacheContainer;
  /**
   * \brief Abstract action that operates on an entire graph
   */
  class G2O_CORE_API HyperGraphAction {
    public:
      class G2O_CORE_API Parameters {
        public:
          virtual ~Parameters();
      };

      class G2O_CORE_API ParametersIteration : public Parameters {
        public:
          explicit ParametersIteration(int iter);
          int iteration;
      };

      virtual ~HyperGraphAction();

      /**
       * re-implement to carry out an action given the graph
       */
      virtual HyperGraphAction* operator()(const HyperGraph* graph, Parameters* parameters = 0);
  };

  /**
   * \brief Abstract action that operates on a graph entity
   */
  class G2O_CORE_API HyperGraphElementAction{
    public:
      struct G2O_CORE_API Parameters{
        virtual ~Parameters();
      };
      typedef std::map<std::string, HyperGraphElementAction*> ActionMap;
      //! an action should be instantiated with the typeid.name of the graph element 
      //! on which it operates
      HyperGraphElementAction(const std::string& typeName_="");

      //! redefine this to do the action stuff. If successful, the action returns a pointer to itself
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, Parameters* parameters);

      //! redefine this to do the action stuff. If successful, the action returns a pointer to itself
      virtual HyperGraphElementAction* operator()(const HyperGraph::HyperGraphElement* element, Parameters* parameters);

      //! destroyed actions release the memory
      virtual ~HyperGraphElementAction();

      //! returns the typeid name of the action
      const std::string& typeName() const { return _typeName;}

      //! returns the name of an action, e.g "draw"
      const std::string& name() const{ return _name;}

      //! sets the type on which an action has to operate
      void setTypeName(const std::string& typeName_);

    protected:
      std::string _typeName;
      std::string _name;
  };

  /**
   * \brief collection of actions
   *
   * collection of actions calls contains homogeneous actions operating on different types
   * all collected actions have the same name and should have the same functionality
   */
  class G2O_CORE_API HyperGraphElementActionCollection: public HyperGraphElementAction{
    public:
      //! constructor. name_ is the name of the action e.g.draw).
      HyperGraphElementActionCollection(const std::string& name_);
      //! destructor: it deletes all actions in the pool.
      virtual ~HyperGraphElementActionCollection();
      //! calling functions, they return a pointer to the instance of action in actionMap
      //! that was active on element
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, Parameters* parameters);
      virtual HyperGraphElementAction* operator()(const HyperGraph::HyperGraphElement* element, Parameters* parameters);
      ActionMap& actionMap() {return _actionMap;}
      //! inserts an action in the pool. The action should have the same name of the container.
      //! returns false on failure (the container has a different name than the action);
      bool registerAction(HyperGraphElementAction* action);
      bool unregisterAction(HyperGraphElementAction* action);
    protected:
      ActionMap _actionMap;
  };

  /**
   * \brief library of actions, indexed by the action name;
   *
   * library of actions, indexed by the action name;
   * one can use ti to register a collection of actions
   */
  class G2O_CORE_API HyperGraphActionLibrary{
    public:
      //! return the single instance of the HyperGraphActionLibrary
      static HyperGraphActionLibrary* instance();
      //! free the instance
      static void destroy();

      // returns a pointer to a collection indexed by name
      HyperGraphElementAction* actionByName(const std::string& name);
      // registers a basic action in the pool. If necessary a container is created
      bool registerAction(HyperGraphElementAction* action);
      bool unregisterAction(HyperGraphElementAction* action);
      
      inline HyperGraphElementAction::ActionMap& actionMap() {return _actionMap;}
    protected:
      HyperGraphActionLibrary();
      ~HyperGraphActionLibrary();
      HyperGraphElementAction::ActionMap _actionMap;
    private:
      static HyperGraphActionLibrary* actionLibInstance;
  };

  /**
   * apply an action to all the elements of the graph.
   */
  void G2O_CORE_API applyAction(HyperGraph* graph, HyperGraphElementAction* action, HyperGraphElementAction::Parameters* parameters=0, const std::string& typeName="");

  /**
   * brief write into gnuplot
   */
  class G2O_CORE_API WriteGnuplotAction: public HyperGraphElementAction{
    public:
      struct G2O_CORE_API Parameters: public HyperGraphElementAction::Parameters{
        std::ostream* os;
      };
      WriteGnuplotAction(const std::string& typeName_);
  };

  /**
   * \brief draw actions
   */

  class G2O_CORE_API DrawAction : public HyperGraphElementAction{
  public:
    class G2O_CORE_API Parameters: public HyperGraphElementAction::Parameters,  public PropertyMap{
    public:
      Parameters();
    };
    DrawAction(const std::string& typeName_);
  protected:
    virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
    void initializeDrawActionsCache() ;
    void drawCache(CacheContainer* caches, HyperGraphElementAction::Parameters* params_);
    void drawUserData(HyperGraph::Data* data, HyperGraphElementAction::Parameters* params_);
    Parameters* _previousParams;
    BoolProperty* _show;
    BoolProperty* _showId;
    HyperGraphElementAction* _cacheDrawActions;
  };

  template<typename T> class RegisterActionProxy
    {
      public:
      RegisterActionProxy()
          {
#ifdef G2O_DEBUG_ACTIONLIB
            std::cout << __FUNCTION__ << ": Registering action of type " << typeid(T).name() << std::endl;
#endif
            _action = new T();
            HyperGraphActionLibrary::instance()->registerAction(_action);
          }
      
        ~RegisterActionProxy()
          {
#ifdef G2O_DEBUG_ACTIONLIB
            std::cout << __FUNCTION__ << ": Unregistering action of type " << typeid(T).name() << std::endl;
#endif
            HyperGraphActionLibrary::instance()->unregisterAction(_action);
            delete _action;
          }

    private:
        HyperGraphElementAction* _action;
  };

#define G2O_REGISTER_ACTION(classname) \
    extern "C" void g2o_action_##classname(void) {} \
    static g2o::RegisterActionProxy<classname> g_action_proxy_##classname;
};

#endif
