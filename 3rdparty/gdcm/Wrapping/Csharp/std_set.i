/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/* -----------------------------------------------------------------------------
 * std_set.i
 *
 * SWIG typemaps for std::set< V >
 *
 * The C# wrapper is made to look and feel like a C# System.Collections.Generic.ISet<>.
 *
 * Using this wrapper is fairly simple. For example, to create a set of integers use:
 *
 *   %include <std_set.i>
 *   %template(SetInt) std::set<int>
 *
 * Notes:
 * 1) IEnumerable<> is implemented in the proxy class which is useful for using LINQ with
 *    C++ std::set wrappers.
 *
 * Warning: heavy macro usage in this file. Use swig -E to get a sane view on the real file contents!
 * ----------------------------------------------------------------------------- */

%{
#include <set>
#include <algorithm>
#include <stdexcept>
%}

/* V is the C++ value type */
%define SWIG_STD_SET_INTERNAL(V)

%typemap(csinterfaces) std::set< V > "IDisposable \n#if SWIG_DOTNET_3\n    , System.Collections.Generic.ISet<$typemap(cstype, V)>\n#endif\n";
%typemap(cscode) std::set<K, T > %{

  public $typemap(cstype, V) this[$typemap(cstype, V) key] {
    get {
      return getitem(key);
    }

    set {
      setitem(value);
    }
  }

//  public bool TryGetValue($typemap(cstype, V) value) {
//    if (this.ContainsKey(key)) {
//      value = this[key];
//      return true;
//    }
//    value = default($typemap(cstype, T));
//    return false;
//  }

  public int Count {
    get {
      return (int)size();
    }
  }

  public bool IsReadOnly {
    get {
      return false;
    }
  }

%}

  public:
    set();
    set(const set< V > &other);

    typedef V value_type;
    typedef size_t size_type;
    size_type size() const;
    bool empty() const;
    %rename(Clear) clear;
    void clear();
    %extend {
      const value_type& getitem(const value_type& val) throw (std::out_of_range) {
        std::set< V >::iterator iter = $self->find(val);
        if (iter != $self->end())
          return *iter;
        else
          throw std::out_of_range("key not found");
      }

      void setitem(const value_type& x) {
        (*$self).insert(x);
      }

      bool Contains(const value_type& val) {
        std::set< V >::iterator iter = $self->find(val);
        return iter != $self->end();
      }

      void Add(const value_type& val) throw (std::out_of_range) {
        std::set< V >::iterator iter = $self->find(val);
        if (iter != $self->end())
          throw std::out_of_range("key already exists");
        $self->insert(val);
      }

      bool Remove(const value_type& val) {
        std::set< V >::iterator iter = $self->find(val);
        if (iter != $self->end()) {
          $self->erase(iter);
          return true;
        }
        return false;
      }

      // create_iterator_begin(), get_next_key() and destroy_iterator work together to provide a collection of keys to C#
      %apply void *VOID_INT_PTR { std::set< V >::iterator *create_iterator_begin }
      %apply void *VOID_INT_PTR { std::set< V >::iterator *swigiterator }

      std::set< V >::iterator *create_iterator_begin() {
        return new std::set< V >::iterator($self->begin());
      }

      const value_type& get_next_key(std::set< V >::iterator *swigiterator) {
        std::set< V >::iterator iter = *swigiterator;
        swigiterator++;
        return *iter;
      }

      void destroy_iterator(std::set< V >::iterator *swigiterator) {
        delete swigiterator;
      }
    }


%enddef

%csmethodmodifiers std::set::size "private"
%csmethodmodifiers std::set::getitem "private"
%csmethodmodifiers std::set::setitem "private"
%csmethodmodifiers std::set::create_iterator_begin "private"
%csmethodmodifiers std::set::get_next_key "private"
%csmethodmodifiers std::set::destroy_iterator "private"

// Default implementation
namespace std {
  template<class V> class set {
    SWIG_STD_SET_INTERNAL(V)
  };
}
