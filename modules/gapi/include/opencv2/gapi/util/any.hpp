// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_ANY_HPP
#define OPENCV_GAPI_UTIL_ANY_HPP

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "opencv2/gapi/util/throw.hpp"

#if defined(_MSC_VER)
   // disable MSVC warning on "multiple copy constructors specified"
#  pragma warning(disable: 4521)
#endif

namespace cv
{

namespace internal
{
    template <class T, class Source>
    T down_cast(Source operand)
    {
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
       return dynamic_cast<T>(operand);
#else
    #warning used static cast instead of dynamic because RTTI is disabled
       return static_cast<T>(operand);
#endif
    }
}

namespace util
{
   class bad_any_cast : public std::bad_cast
   {
   public:
       virtual const char* what() const noexcept override
       {
           return "Bad any cast";
       }
   };

   //modeled against C++17 std::any

   class any
   {
   private:
      struct holder;
      using holder_ptr = std::unique_ptr<holder>;
      struct holder
      {
         virtual holder_ptr clone() = 0;
         virtual ~holder() = default;
      };

      template <typename value_t>
      struct holder_impl : holder
      {
         value_t v;
         template<typename arg_t>
         holder_impl(arg_t&& a) : v(std::forward<arg_t>(a)) {}
         holder_ptr clone() override { return holder_ptr(new holder_impl (v));}
      };

      holder_ptr hldr;
   public:
      template<class value_t>
      any(value_t&& arg) :  hldr(new holder_impl<typename std::decay<value_t>::type>( std::forward<value_t>(arg))) {}

      any(any const& src) : hldr( src.hldr ? src.hldr->clone() : nullptr) {}
      //simple hack in order not to write enable_if<not any> for the template constructor
      any(any & src) : any (const_cast<any const&>(src)) {}

      any()       = default;
      any(any&& ) = default;

      any& operator=(any&&) = default;

      any& operator=(any const& src)
      {
         any copy(src);
         swap(*this, copy);
         return *this;
      }

      template<class value_t>
      friend value_t* any_cast(any* operand);

      template<class value_t>
      friend const value_t* any_cast(const any* operand);

      template<class value_t>
      friend value_t& unsafe_any_cast(any& operand);

      template<class value_t>
      friend const value_t& unsafe_any_cast(const any& operand);

      friend void swap(any & lhs, any& rhs)
      {
         swap(lhs.hldr, rhs.hldr);
      }

   };

   template<class value_t>
   value_t* any_cast(any* operand)
   {
      auto casted = internal::down_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand->hldr.get());
      if (casted){
         return & (casted->v);
      }
      return nullptr;
   }

   template<class value_t>
   const value_t* any_cast(const any* operand)
   {
      auto casted = internal::down_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand->hldr.get());
      if (casted){
         return & (casted->v);
      }
      return nullptr;
   }

   template<class value_t>
   value_t& any_cast(any& operand)
   {
      auto ptr = any_cast<value_t>(&operand);
      if (ptr)
      {
         return *ptr;
      }

      throw_error(bad_any_cast());
   }


   template<class value_t>
   const value_t& any_cast(const any& operand)
   {
      auto ptr = any_cast<value_t>(&operand);
      if (ptr)
      {
         return *ptr;
      }

      throw_error(bad_any_cast());
   }

   template<class value_t>
   inline value_t& unsafe_any_cast(any& operand)
   {
#ifdef DEBUG
      return any_cast<value_t>(operand);
#else
      return static_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand.hldr.get())->v;
#endif
   }

   template<class value_t>
   inline const value_t& unsafe_any_cast(const any& operand)
   {
#ifdef DEBUG
      return any_cast<value_t>(operand);
#else
      return static_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand.hldr.get())->v;
#endif
   }

} // namespace util
} // namespace cv

#if defined(_MSC_VER)
   // Enable "multiple copy constructors specified" back
#  pragma warning(default: 4521)
#endif

#endif // OPENCV_GAPI_UTIL_ANY_HPP
