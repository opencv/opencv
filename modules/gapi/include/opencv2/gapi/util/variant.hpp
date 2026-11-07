// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_VARIANT_HPP
#define OPENCV_GAPI_UTIL_VARIANT_HPP

#include <array>
#include <type_traits>

#include <opencv2/gapi/util/compiler_hints.hpp>
#include <opencv2/gapi/util/throw.hpp>
#include <opencv2/gapi/util/util.hpp> // max_of_t
#include <opencv2/gapi/util/type_traits.hpp>

// A poor man's `variant` implementation, incompletely modeled against C++17 spec.
namespace cv
{
namespace util
{
    namespace detail
    {
        template<std::size_t I, typename Target, typename First, typename... Remaining>
        struct type_list_index_helper
        {
            static const constexpr bool is_same = std::is_same<Target, First>::value;
            static const constexpr std::size_t value =
                std::conditional<is_same, std::integral_constant<std::size_t, I>, type_list_index_helper<I + 1, Target, Remaining...>>::type::value;
        };

        template<std::size_t I, typename Target, typename First>
        struct type_list_index_helper<I, Target, First>
        {
            static_assert(std::is_same<Target, First>::value, "Type not found");
            static const constexpr std::size_t value = I;
        };
    }

    template<typename Target, typename... Types>
    struct type_list_index
    {
        static const constexpr std::size_t value = detail::type_list_index_helper<0, Target, Types...>::value;
    };

    template<std::size_t Index, class... Types >
    struct type_list_element
    {
        using type = typename std::tuple_element<Index, std::tuple<Types...> >::type;
    };

    class bad_variant_access: public std::exception
    {
    public:
        virtual const char *what() const noexcept override
        {
            return "Bad variant access";
        }
    };

    // Interface ///////////////////////////////////////////////////////////////
    struct monostate {};
    inline bool operator==(const util::monostate&, const util::monostate&)
    {
        return true;
    }

    template<typename... Ts> // FIXME: no references, arrays, and void
    class variant
    {
        // FIXME: Replace with std::aligned_union after gcc4.8 support is dropped
        static constexpr const std::size_t S = cv::detail::max_of_t<sizeof(Ts)...>::value;
        static constexpr const std::size_t A = cv::detail::max_of_t<alignof(Ts)...>::value;
        using Memory = typename std::aligned_storage<S, A>::type[1];

        template<typename T> struct cctr_h {
            static void help(Memory memory, const Memory from) {
                new (memory) T(*reinterpret_cast<const T*>(from));
            }
        };

        template<typename T> struct mctr_h {
            static void help(Memory memory, void *pval) {
                new (memory) T(std::move(*reinterpret_cast<T*>(pval)));
            }
        };

        //FIXME: unify with cctr_h and mctr_h
        template<typename T> struct cnvrt_ctor_h {
            static void help(Memory memory, void* from) {
                using util::decay_t;
                new (memory) decay_t<T>(std::forward<T>(*reinterpret_cast<decay_t<T>*>(from)));
            }
        };

        template<typename T> struct copy_h {
            static void help(Memory to, const Memory from) {
                *reinterpret_cast<T*>(to) = *reinterpret_cast<const T*>(from);
            }
        };

        template<typename T> struct move_h {
            static void help(Memory to, Memory from) {
                *reinterpret_cast<T*>(to) = std::move(*reinterpret_cast<T*>(from));
            }
        };

        //FIXME: unify with copy_h and move_h
        template<typename T> struct cnvrt_assign_h {
            static void help(Memory to, void* from) {
                using util::decay_t;
                *reinterpret_cast<decay_t<T>*>(to) = std::forward<T>(*reinterpret_cast<decay_t<T>*>(from));
            }
        };

        template<typename T> struct swap_h {
            static void help(Memory to, Memory from) {
                std::swap(*reinterpret_cast<T*>(to), *reinterpret_cast<T*>(from));
            }
        };

        template<typename T> struct dtor_h {
            static void help(Memory memory) {
                (void) memory; // MSCV warning
                reinterpret_cast<T*>(memory)->~T();
            }
        };

        template<typename T> struct equal_h {
            static bool help(const Memory lhs, const Memory rhs) {
                const T& t_lhs = *reinterpret_cast<const T*>(lhs);
                const T& t_rhs = *reinterpret_cast<const T*>(rhs);
                return t_lhs == t_rhs;
            }
        };

        typedef void (*CCtr) (Memory, const Memory);  // Copy c-tor (variant)
        typedef void (*MCtr) (Memory, void*);         // Generic move c-tor
        typedef void (*Copy) (Memory, const Memory);  // Copy assignment
        typedef void (*Move) (Memory, Memory);        // Move assignment

        typedef void (*Swap) (Memory, Memory);        // Swap
        typedef void (*Dtor) (Memory);                // Destructor

        using  cnvrt_assgn_t   = void (*) (Memory, void*);  // Converting assignment (via std::forward)
        using  cnvrt_ctor_t    = void (*) (Memory, void*);  // Converting constructor (via std::forward)

        typedef bool (*Equal)(const Memory, const Memory); // Equality test (external)

        static constexpr std::array<CCtr, sizeof...(Ts)> cctrs(){ return {{(&cctr_h<Ts>::help)...}};}
        static constexpr std::array<MCtr, sizeof...(Ts)> mctrs(){ return {{(&mctr_h<Ts>::help)...}};}
        static constexpr std::array<Copy, sizeof...(Ts)> cpyrs(){ return {{(&copy_h<Ts>::help)...}};}
        static constexpr std::array<Move, sizeof...(Ts)> mvers(){ return {{(&move_h<Ts>::help)...}};}
        static constexpr std::array<Swap, sizeof...(Ts)> swprs(){ return {{(&swap_h<Ts>::help)...}};}
        static constexpr std::array<Dtor, sizeof...(Ts)> dtors(){ return {{(&dtor_h<Ts>::help)...}};}

        template<bool cond, typename T>
        struct conditional_ref : std::conditional<cond, typename std::remove_reference<T>::type&, typename std::remove_reference<T>::type > {};

        template<bool cond, typename T>
        using conditional_ref_t = typename conditional_ref<cond, T>::type;


        template<bool is_lvalue_arg>
        static constexpr std::array<cnvrt_assgn_t, sizeof...(Ts)> cnvrt_assgnrs(){
            return {{(&cnvrt_assign_h<conditional_ref_t<is_lvalue_arg,Ts>>::help)...}};
        }

        template<bool is_lvalue_arg>
        static constexpr std::array<cnvrt_ctor_t, sizeof...(Ts)> cnvrt_ctors(){
            return {{(&cnvrt_ctor_h<conditional_ref_t<is_lvalue_arg,Ts>>::help)...}};
        }

        std::size_t m_index = 0;

    protected:
        template<typename T, typename... Us> friend T& get(variant<Us...> &v);
        template<typename T, typename... Us> friend const T& get(const variant<Us...> &v);
        template<typename T, typename... Us> friend T* get_if(variant<Us...> *v) noexcept;
        template<typename T, typename... Us> friend const T* get_if(const variant<Us...> *v) noexcept;

        template<typename... Us> friend bool operator==(const variant<Us...> &lhs,
                                                        const variant<Us...> &rhs);
        Memory memory;

    public:
        // Constructors
        variant() noexcept;
        variant(const variant& other);
        variant(variant&& other) noexcept;
        // are_different_t is a SFINAE trick to avoid variant(T &&t) with T=variant
        // for some reason, this version is called instead of variant(variant&& o) when
        // variant is used in STL containers (examples: vector assignment).
        template<
            typename T,
            typename = util::are_different_t<variant, T>
        >
        explicit variant(T&& t);
        // template<class T, class... Args> explicit variant(Args&&... args);
        // FIXME: other constructors

        // Destructor
        ~variant();

        // Assignment
        variant& operator=(const variant& rhs);
        variant& operator=(variant &&rhs) noexcept;

        // SFINAE trick to avoid operator=(T&&) with T=variant<>, see comment above
        template<
            typename T,
            typename = util::are_different_t<variant, T>
        >
        variant& operator=(T&& t) noexcept;

        // Observers
        std::size_t index() const noexcept;
        // FIXME: valueless_by_exception()

        // Modifiers
        // FIXME: emplace()
        void swap(variant &rhs) noexcept;

        // Non-C++17x!
        template<typename T> static constexpr std::size_t index_of();
    };

    // FIMXE: visit
    template<typename T, typename... Types>
    T* get_if(util::variant<Types...>* v) noexcept;

    template<typename T, typename... Types>
    const T* get_if(const util::variant<Types...>* v) noexcept;

    template<typename T, typename... Types>
    T& get(util::variant<Types...> &v);

    template<typename T, typename... Types>
    const T& get(const util::variant<Types...> &v);

    template<std::size_t Index, typename... Types>
    typename util::type_list_element<Index, Types...>::type& get(util::variant<Types...> &v);

    template<std::size_t Index, typename... Types>
    const typename util::type_list_element<Index, Types...>::type& get(const util::variant<Types...> &v);

    template<typename T, typename... Types>
    bool holds_alternative(const util::variant<Types...> &v) noexcept;


    // Visitor
    namespace detail
    {
        struct visitor_interface {};

        // Class `visitor_return_type_deduction_helper`
        // introduces solution for deduction `return_type` in `visit` function in common way
        // for both Lambda and class Visitor and keep one interface invocation point: `visit` only
        // his helper class is required to unify return_type deduction mechanism because
        // for Lambda it is possible to take type of `decltype(visitor(get<0>(var)))`
        // but for class Visitor there is no operator() in base case,
        // because it provides `operator() (std::size_t index, ...)`
        // So `visitor_return_type_deduction_helper` expose `operator()`
        // uses only for class Visitor only for deduction `return type` in visit()
        template<typename R>
        struct visitor_return_type_deduction_helper
        {
            using return_type = R;

            // to be used in Lambda return type deduction context only
            template<typename T>
            return_type operator() (T&&);
        };
    }

    // Special purpose `static_visitor` can receive additional arguments
    template<typename R, typename Impl>
    struct static_visitor : public detail::visitor_interface,
                            public detail::visitor_return_type_deduction_helper<R> {

        // assign responsibility for return type deduction to helper class
        using return_type = typename detail::visitor_return_type_deduction_helper<R>::return_type;
        using detail::visitor_return_type_deduction_helper<R>::operator();
        friend Impl;

        template<typename VariantValue, typename ...Args>
        return_type operator() (std::size_t index, VariantValue&& value, Args&& ...args)
        {
            suppress_unused_warning(index);
            return static_cast<Impl*>(this)-> visit(
                                                std::forward<VariantValue>(value),
                                                std::forward<Args>(args)...);
        }
    };

    // Special purpose `static_indexed_visitor` can receive additional arguments
    // And make forwarding current variant index as runtime function argument to its `Impl`
    template<typename R, typename Impl>
    struct static_indexed_visitor : public detail::visitor_interface,
                                    public detail::visitor_return_type_deduction_helper<R> {

        // assign responsibility for return type deduction to helper class
        using return_type = typename detail::visitor_return_type_deduction_helper<R>::return_type;
        using detail::visitor_return_type_deduction_helper<R>::operator();
        friend Impl;

        template<typename VariantValue, typename ...Args>
        return_type operator() (std::size_t Index, VariantValue&& value, Args&& ...args)
        {
            return static_cast<Impl*>(this)-> visit(Index,
                                                std::forward<VariantValue>(value),
                                                std::forward<Args>(args)...);
        }
    };

    template <class T>
    struct variant_size;

    template <class... Types>
    struct variant_size<util::variant<Types...>>
        : std::integral_constant<std::size_t, sizeof...(Types)> { };
    // FIXME: T&&, const TT&& versions.

    // Implementation //////////////////////////////////////////////////////////
    template<typename... Ts>
    variant<Ts...>::variant() noexcept
    {
        typedef typename std::tuple_element<0, std::tuple<Ts...> >::type TFirst;
        new (memory) TFirst();
    }

    template<typename... Ts>
    variant<Ts...>::variant(const variant &other)
        : m_index(other.m_index)
    {
        (cctrs()[m_index])(memory, other.memory);
    }

    template<typename... Ts>
    variant<Ts...>::variant(variant &&other) noexcept
        : m_index(other.m_index)
    {
        (mctrs()[m_index])(memory, other.memory);
    }

    template<typename... Ts>
    template<class T, typename>
    variant<Ts...>::variant(T&& t)
        : m_index(util::type_list_index<util::decay_t<T>, Ts...>::value)
    {
        const constexpr bool is_lvalue_arg =  std::is_lvalue_reference<T>::value;
        (cnvrt_ctors<is_lvalue_arg>()[m_index])(memory, const_cast<util::decay_t<T> *>(&t));
    }

    template<typename... Ts>
    variant<Ts...>::~variant()
    {
        (dtors()[m_index])(memory);
    }

    template<typename... Ts>
    variant<Ts...>& variant<Ts...>::operator=(const variant<Ts...> &rhs)
    {
        if (m_index != rhs.m_index)
        {
            (dtors()[    m_index])(memory);
            (cctrs()[rhs.m_index])(memory, rhs.memory);
            m_index = rhs.m_index;
        }
        else
        {
            (cpyrs()[rhs.m_index])(memory, rhs.memory);
        }
        return *this;
    }

    template<typename... Ts>
    variant<Ts...>& variant<Ts...>::operator=(variant<Ts...> &&rhs) noexcept
    {
        if (m_index != rhs.m_index)
        {
            (dtors()[    m_index])(memory);
            (mctrs()[rhs.m_index])(memory, rhs.memory);
            m_index = rhs.m_index;
        }
        else
        {
            (mvers()[rhs.m_index])(memory, rhs.memory);
        }
        return *this;
    }

    template<typename... Ts>
    template<typename T, typename>
    variant<Ts...>& variant<Ts...>::operator=(T&& t) noexcept
    {
        using decayed_t = util::decay_t<T>;
        // FIXME: No version with implicit type conversion available!
        const constexpr std::size_t t_index =
            util::type_list_index<decayed_t, Ts...>::value;

        const constexpr bool is_lvalue_arg =  std::is_lvalue_reference<T>::value;

        if (t_index != m_index)
        {
            (dtors()[m_index])(memory);
            (cnvrt_ctors<is_lvalue_arg>()[t_index])(memory, &t);
            m_index = t_index;
        }
        else
        {
            (cnvrt_assgnrs<is_lvalue_arg>()[m_index])(memory, &t);
        }
        return *this;

    }

    template<typename... Ts>
    std::size_t util::variant<Ts...>::index() const noexcept
    {
        return m_index;
    }

    template<typename... Ts>
    void variant<Ts...>::swap(variant<Ts...> &rhs) noexcept
    {
        if (m_index == rhs.index())
        {
            (swprs()[m_index](memory, rhs.memory));
        }
        else
        {
            variant<Ts...> tmp(std::move(*this));
            *this = std::move(rhs);
            rhs   = std::move(tmp);
        }
    }

    template<typename... Ts>
    template<typename T>
    constexpr std::size_t variant<Ts...>::index_of()
    {
        return util::type_list_index<T, Ts...>::value; // FIXME: tests!
    }

    template<typename T, typename... Types>
    T* get_if(util::variant<Types...>* v) noexcept
    {
        const constexpr std::size_t t_index =
            util::type_list_index<T, Types...>::value;

        if (v && v->index() == t_index)
            return (T*)(&v->memory);  // workaround for ICC 2019
            // original code: return reinterpret_cast<T&>(v.memory);
        return nullptr;
    }

    template<typename T, typename... Types>
    const T* get_if(const util::variant<Types...>* v) noexcept
    {
        const constexpr std::size_t t_index =
            util::type_list_index<T, Types...>::value;

        if (v && v->index() == t_index)
            return (const T*)(&v->memory);  // workaround for ICC 2019
            // original code: return reinterpret_cast<const T&>(v.memory);
        return nullptr;
    }

    template<typename T, typename... Types>
    T& get(util::variant<Types...> &v)
    {
        if (auto* p = get_if<T>(&v))
            return *p;
        else
            throw_error(bad_variant_access());
    }

    template<typename T, typename... Types>
    const T& get(const util::variant<Types...> &v)
    {
        if (auto* p = get_if<T>(&v))
            return *p;
        else
            throw_error(bad_variant_access());
    }

    template<std::size_t Index, typename... Types>
    typename util::type_list_element<Index, Types...>::type& get(util::variant<Types...> &v)
    {
        using ReturnType = typename util::type_list_element<Index, Types...>::type;
        return const_cast<ReturnType&>(get<Index, Types...>(static_cast<const util::variant<Types...> &>(v)));
    }

    template<std::size_t Index, typename... Types>
    const typename util::type_list_element<Index, Types...>::type& get(const util::variant<Types...> &v)
    {
        static_assert(Index < sizeof...(Types),
                      "`Index` it out of bound of `util::variant` type list");
        using ReturnType = typename util::type_list_element<Index, Types...>::type;
        return get<ReturnType>(v);
    }

    template<typename T, typename... Types>
    bool holds_alternative(const util::variant<Types...> &v) noexcept
    {
        return v.index() == util::variant<Types...>::template index_of<T>();
    }

#if defined(__GNUC__) && (__GNUC__ == 11 || __GNUC__ == 12)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    template<typename... Us> bool operator==(const variant<Us...> &lhs,
                                             const variant<Us...> &rhs)
    {
        using V = variant<Us...>;

        // Instantiate table only here since it requires operator== for <Us...>
        // <Us...> should have operator== only if this one is used, not in general
        static const std::array<typename V::Equal, sizeof...(Us)> eqs = {
            {(&V::template equal_h<Us>::help)...}
        };
        if (lhs.index() != rhs.index())
            return false;
        return (eqs[lhs.index()])(lhs.memory, rhs.memory);
    }

#if defined(__GNUC__) && (__GNUC__ == 11 || __GNUC__ == 12)
#pragma GCC diagnostic pop
#endif

    template<typename... Us> bool operator!=(const variant<Us...> &lhs,
                                             const variant<Us...> &rhs)
    {
        return !(lhs == rhs);
    }

namespace detail
{
    // terminate recursion implementation for `non-void` ReturnType
    template<typename ReturnType, std::size_t CurIndex, std::size_t ElemCount,
             typename Visitor, typename Variant, typename... VisitorArgs>
    ReturnType apply_visitor_impl(Visitor&&, Variant&,
                                  std::true_type, std::false_type,
                                  VisitorArgs&& ...)
    {
        return {};
    }

    // terminate recursion implementation for `void` ReturnType
    template<typename ReturnType, std::size_t CurIndex, std::size_t ElemCount,
             typename Visitor, typename Variant, typename... VisitorArgs>
    void apply_visitor_impl(Visitor&&, Variant&,
                            std::true_type, std::true_type,
                            VisitorArgs&& ...)
    {
    }

    // Intermediate resursion processor for Lambda Visitors
    template<typename ReturnType, std::size_t CurIndex, std::size_t ElemCount,
             typename Visitor, typename Variant, bool no_return_value, typename... VisitorArgs>
    typename std::enable_if<!std::is_base_of<visitor_interface, typename std::decay<Visitor>::type>::value, ReturnType>::type
         apply_visitor_impl(Visitor&& visitor, Variant&& v, std::false_type not_processed,
                                               std::integral_constant<bool, no_return_value> should_no_return,
                                               VisitorArgs&& ...args)
    {
        static_assert(std::is_same<ReturnType, decltype(visitor(get<CurIndex>(v)))>::value,
                      "Different `ReturnType`s detected! All `Visitor::visit` or `overload_lamba_set`"
                      " must return the same type");
        suppress_unused_warning(not_processed);
        if (v.index() == CurIndex)
        {
            return visitor.operator()(get<CurIndex>(v), std::forward<VisitorArgs>(args)... );
        }

        using is_variant_processed_t = std::integral_constant<bool, CurIndex + 1 >= ElemCount>;
        return apply_visitor_impl<ReturnType, CurIndex +1, ElemCount>(
                                  std::forward<Visitor>(visitor),
                                  std::forward<Variant>(v),
                                  is_variant_processed_t{},
                                  should_no_return,
                                  std::forward<VisitorArgs>(args)...);
    }

    //Visual Studio 2014 compilation fix: cast visitor to base class before invoke operator()
    template<std::size_t CurIndex, typename ReturnType, typename Visitor, class Value, typename... VisitorArgs>
    typename std::enable_if<std::is_base_of<static_visitor<ReturnType, typename std::decay<Visitor>::type>,
                                            typename std::decay<Visitor>::type>::value, ReturnType>::type
    invoke_class_visitor(Visitor& visitor, Value&& v,  VisitorArgs&&...args)
    {
        return static_cast<static_visitor<ReturnType, typename std::decay<Visitor>::type>&>(visitor).operator() (CurIndex, std::forward<Value>(v), std::forward<VisitorArgs>(args)... );
    }

    //Visual Studio 2014 compilation fix: cast visitor to base class before invoke operator()
    template<std::size_t CurIndex, typename ReturnType, typename Visitor, class Value, typename... VisitorArgs>
    typename std::enable_if<std::is_base_of<static_indexed_visitor<ReturnType, typename std::decay<Visitor>::type>,
                                            typename std::decay<Visitor>::type>::value, ReturnType>::type
    invoke_class_visitor(Visitor& visitor, Value&& v,  VisitorArgs&&...args)
    {
        return static_cast<static_indexed_visitor<ReturnType, typename std::decay<Visitor>::type>&>(visitor).operator() (CurIndex, std::forward<Value>(v), std::forward<VisitorArgs>(args)... );
    }

    // Intermediate recursion processor for special case `visitor_interface` derived Visitors
    template<typename ReturnType, std::size_t CurIndex, std::size_t ElemCount,
             typename Visitor, typename Variant, bool no_return_value, typename... VisitorArgs>
    typename std::enable_if<std::is_base_of<visitor_interface, typename std::decay<Visitor>::type>::value, ReturnType>::type
         apply_visitor_impl(Visitor&& visitor, Variant&& v, std::false_type not_processed,
                                               std::integral_constant<bool, no_return_value> should_no_return,
                                               VisitorArgs&& ...args)
    {
        static_assert(std::is_same<ReturnType, decltype(visitor(get<CurIndex>(v)))>::value,
                      "Different `ReturnType`s detected! All `Visitor::visit` or `overload_lamba_set`"
                      " must return the same type");
        suppress_unused_warning(not_processed);
        if (v.index() == CurIndex)
        {
            return invoke_class_visitor<CurIndex, ReturnType>(visitor, get<CurIndex>(v), std::forward<VisitorArgs>(args)... );
        }

        using is_variant_processed_t = std::integral_constant<bool, CurIndex + 1 >= ElemCount>;
        return apply_visitor_impl<ReturnType, CurIndex +1, ElemCount>(
                                  std::forward<Visitor>(visitor),
                                  std::forward<Variant>(v),
                                  is_variant_processed_t{},
                                  should_no_return,
                                  std::forward<VisitorArgs>(args)...);
    }
} // namespace detail

    template<typename Visitor, typename Variant, typename... VisitorArg>
    auto visit(Visitor &visitor, const Variant& var, VisitorArg &&...args) -> decltype(visitor(get<0>(var)))
    {
        constexpr std::size_t varsize = util::variant_size<Variant>::value;
        static_assert(varsize != 0, "utils::variant must contains one type at least ");
        using is_variant_processed_t = std::false_type;

        using ReturnType = decltype(visitor(get<0>(var)));
        using return_t = std::is_same<ReturnType, void>;
        return detail::apply_visitor_impl<ReturnType, 0, varsize, Visitor>(
                                    std::forward<Visitor>(visitor),
                                    var, is_variant_processed_t{},
                                    return_t{},
                                    std::forward<VisitorArg>(args)...);
    }

    template<typename Visitor, typename Variant>
    auto visit(Visitor&& visitor, const Variant& var) -> decltype(visitor(get<0>(var)))
    {
        constexpr std::size_t varsize = util::variant_size<Variant>::value;
        static_assert(varsize != 0, "utils::variant must contains one type at least ");
        using is_variant_processed_t = std::false_type;

        using ReturnType = decltype(visitor(get<0>(var)));
        using return_t = std::is_same<ReturnType, void>;
        return detail::apply_visitor_impl<ReturnType, 0, varsize, Visitor>(
                                    std::forward<Visitor>(visitor),
                                    var, is_variant_processed_t{},
                                    return_t{});
    }
} // namespace util
} // namespace cv

#endif // OPENCV_GAPI_UTIL_VARIANT_HPP
