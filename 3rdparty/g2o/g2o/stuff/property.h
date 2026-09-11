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

#ifndef G2O_PROPERTY_H_
#define G2O_PROPERTY_H_

#include <string>
#include <map>
#include <sstream>

#include "string_tools.h"
#include "g2o_stuff_api.h"

namespace g2o {

  class G2O_STUFF_API BaseProperty {
    public:
      BaseProperty(const std::string name_);
      virtual ~BaseProperty();
      const std::string& name() {return _name;}
      virtual std::string toString() const = 0;
      virtual bool fromString(const std::string& s) = 0;
    protected:
      std::string _name;
  };

  template <typename T>
  class Property: public BaseProperty {
    public:
      typedef T ValueType;
      Property(const std::string& name_): BaseProperty(name_){}
      Property(const std::string& name_, const T& v): BaseProperty(name_), _value(v){}
      void setValue(const T& v) {_value = v; }
      const T& value() const {return _value;}
      virtual std::string toString() const
      {
        std::stringstream sstr;
        sstr << _value;
        return sstr.str();
      }
      virtual bool fromString(const std::string& s)
      {
        bool status = convertString(s, _value);
        return status;
      }
    protected:
      T _value;
  };

  /**
   * \brief a collection of properties mapping from name to the property itself
   */
  class G2O_STUFF_API PropertyMap : protected std::map<std::string, BaseProperty*>
  {
    public:
      typedef std::map<std::string, BaseProperty*>        BaseClass;
      typedef BaseClass::iterator                         PropertyMapIterator;
      typedef BaseClass::const_iterator                   PropertyMapConstIterator;

      ~PropertyMap();

      /**
       * add a property to the map
       */
      bool addProperty(BaseProperty* p);

      /**
       * remove a property from the map
       */
      bool eraseProperty(const std::string& name_);

      /**
       * return a property by its name
       */
      template <typename P> 
      P* getProperty(const std::string& name_)
      {
        PropertyMapIterator it=find(name_);
        if (it==end())
          return nullptr;
        return dynamic_cast<P*>(it->second);
      }
      template <typename P> 
      const P* getProperty(const std::string& name_) const
      {
        PropertyMapConstIterator it=find(name_);
        if (it==end())
          return nullptr;
        return dynamic_cast<P*>(it->second);
      }

      /**
       * create a property and insert it
       */
      template <typename P> 
      P* makeProperty(const std::string& name_, const typename P::ValueType& v)
      {
        PropertyMapIterator it=find(name_);
        if (it==end()){
          P* p=new P(name_, v);
          addProperty(p);
          return p;
        } else 
          return dynamic_cast<P*>(it->second);
      }

      /**
       * update a specfic property with a new value
       * @return true if the params is stored and update was carried out
       */
      bool updatePropertyFromString(const std::string& name, const std::string& value);

      /**
       * update the map based on a name=value string, e.g., name1=val1,name2=val2
       * @return true, if it was possible to update all parameters
       */
      bool updateMapFromString(const std::string& values);

      void writeToCSV(std::ostream& os) const;

      using BaseClass::size;
      using BaseClass::begin;
      using BaseClass::end;
      using BaseClass::iterator;
      using BaseClass::const_iterator;

  };

  typedef Property<int> IntProperty;
  typedef Property<bool> BoolProperty;
  typedef Property<float> FloatProperty;
  typedef Property<double> DoubleProperty;
  typedef Property<std::string> StringProperty;

} // end namespace
#endif
