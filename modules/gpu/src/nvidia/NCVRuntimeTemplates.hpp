////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design 
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author or Addison-Welsey Longman make no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
// http://loki-lib.sourceforge.net/index.php?n=Main.License
////////////////////////////////////////////////////////////////////////////////

#ifndef _ncvruntimetemplates_hpp_
#define _ncvruntimetemplates_hpp_

#include <stdarg.h>
#include <vector>


namespace Loki
{
    //==============================================================================
    // class NullType
    // Used as a placeholder for "no type here"
    // Useful as an end marker in typelists 
    //==============================================================================

    class NullType {};

    //==============================================================================
    // class template Typelist
    // The building block of typelists of any length
    // Use it through the LOKI_TYPELIST_NN macros
    // Defines nested types:
    //     Head (first element, a non-typelist type by convention)
    //     Tail (second element, can be another typelist)
    //==============================================================================

    template <class T, class U>
    struct Typelist
    {
        typedef T Head;
        typedef U Tail;
    };

    //==============================================================================
    // class template Int2Type
    // Converts each integral constant into a unique type
    // Invocation: Int2Type<v> where v is a compile-time constant integral
    // Defines 'value', an enum that evaluates to v
    //==============================================================================

    template <int v>
    struct Int2Type
    {
        enum { value = v };
    };

    namespace TL
    {
        //==============================================================================
        // class template TypeAt
        // Finds the type at a given index in a typelist
        // Invocation (TList is a typelist and index is a compile-time integral 
        //     constant):
        // TypeAt<TList, index>::Result
        // returns the type in position 'index' in TList
        // If you pass an out-of-bounds index, the result is a compile-time error
        //==============================================================================

        template <class TList, unsigned int index> struct TypeAt;

        template <class Head, class Tail>
        struct TypeAt<Typelist<Head, Tail>, 0>
        {
            typedef Head Result;
        };

        template <class Head, class Tail, unsigned int i>
        struct TypeAt<Typelist<Head, Tail>, i>
        {
            typedef typename TypeAt<Tail, i - 1>::Result Result;
        };
    }
}


////////////////////////////////////////////////////////////////////////////////
// Runtime boolean template instance dispatcher
// Cyril Crassin <cyril.crassin@icare3d.org>
// NVIDIA, 2010
////////////////////////////////////////////////////////////////////////////////

namespace NCVRuntimeTemplateBool
{
    //This struct is used to transform a list of parameters into template arguments
    //The idea is to build a typelist containing the arguments
    //and to pass this typelist to a user defined functor
    template<typename TList, int NumArguments, class Func>
    struct KernelCaller
    {
        //Convenience function used by the user
        //Takes a variable argument list, transforms it into a list
        static void call(Func &functor, int dummy, ...)
        {
            //Vector used to collect arguments
            std::vector<int> templateParamList;

            //Variable argument list manipulation
            va_list listPointer;
            va_start(listPointer, dummy);
            //Collect parameters into the list
            for(int i=0; i<NumArguments; i++)
            {
                int val = va_arg(listPointer, int);
                templateParamList.push_back(val);
            }
            va_end(listPointer);

            //Call the actual typelist building function
            call(functor, templateParamList);
        }

        //Actual function called recursively to build a typelist based
        //on a list of values
        static void call( Func &functor, std::vector<int> &templateParamList)
        {
            //Get current parameter value in the list
            int val = templateParamList[templateParamList.size() - 1];
            templateParamList.pop_back();

            //Select the compile time value to add into the typelist
            //depending on the runtime variable and make recursive call. 
            //Both versions are really instantiated
            if(val)
            {
                KernelCaller<
                    Loki::Typelist<typename Loki::Int2Type<true>, TList >,
                    NumArguments-1, Func >
                    ::call(functor, templateParamList);
            }
            else
            {
                KernelCaller< 
                    Loki::Typelist<typename Loki::Int2Type<false>, TList >,
                    NumArguments-1, Func >
                    ::call(functor, templateParamList);
            }
        }
    };

    //Specialization for 0 value left in the list
    //-> actual kernel functor call
    template<class TList, class Func>
    struct KernelCaller<TList, 0, Func>
    {
        static void call(Func &functor)
        {
            //Call to the functor's kernel call method
            functor.call(TList()); //TList instantiated to get the method template parameter resolved
        }

        static void call(Func &functor, std::vector<int> &templateParams)
        {
            functor.call(TList());
        }
    };
}

#endif //_ncvruntimetemplates_hpp_
