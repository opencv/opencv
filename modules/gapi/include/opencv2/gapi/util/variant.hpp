// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_VARIANT_HPP
#define OPENCV_GAPI_UTIL_VARIANT_HPP

#include <array>
#include <type_traits>

#include "opencv2/gapi/util/throw.hpp"
#include "opencv2/gapi/util/util.hpp" // max_of_t

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


        template<class T, class U, class V> using are_different =
            std::enable_if<!std::is_same<typename std::decay<T>::type,
                                         typename std::decay<U>::type>::value,
                           V>;
    }

    template<typename Target, typename... Types>
    struct type_list_index
    {
        static const constexpr std::size_t value = detail::type_list_index_helper<0, Target, Types...>::value;
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

        template<typename T> struct vctr_h {
            static void help(Memory memory, const void* pval) {
                new (memory) T(*reinterpret_cast<const T*>(pval));
            }
        };

        template<typename T> struct mctr_h {
            static void help(Memory memory, void *pval) {
                new (memory) T(std::move(*reinterpret_cast<T*>(pval)));
            }
        };

        template<typename T> struct copy_h {
            static void help(Memory to, const Memory from) {
                *reinterpret_cast<T*>(to) = *reinterpret_cast<const T*>(from);
            }
        };

        template<typename T> struct move_h {
            static void help(Memory to, const Memory from) {
                *reinterpret_cast<T*>(to) = std::move(*reinterpret_cast<const T*>(from));
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
        typedef void (*VCtr) (Memory, const void*);   // Copy c-tor (value)
        typedef void (*MCtr) (Memory, void*);         // Generic move c-tor
        typedef void (*Copy) (Memory, const Memory);  // Copy assignment
        typedef void (*Move) (Memory, const Memory);  // Move assignment
        typedef void (*Swap) (Memory, Memory);        // Swap
        typedef void (*Dtor) (Memory);                // Destructor

        typedef bool (*Equal)(const Memory, const Memory); // Equality test (external)

        static constexpr std::array<CCtr, sizeof...(Ts)> cctrs(){ return {{(&cctr_h<Ts>::help)...}};}
        static constexpr std::array<VCtr, sizeof...(Ts)> vctrs(){ return {{(&vctr_h<Ts>::help)...}};}
        static constexpr std::array<MCtr, sizeof...(Ts)> mctrs(){ return {{(&mctr_h<Ts>::help)...}};}
        static constexpr std::array<Copy, sizeof...(Ts)> cpyrs(){ return {{(&copy_h<Ts>::help)...}};}
        static constexpr std::array<Move, sizeof...(Ts)> mvers(){ return {{(&move_h<Ts>::help)...}};}
        static constexpr std::array<Swap, sizeof...(Ts)> swprs(){ return {{(&swap_h<Ts>::help)...}};}
        static constexpr std::array<Dtor, sizeof...(Ts)> dtors(){ return {{(&dtor_h<Ts>::help)...}};}

        std::size_t m_index = 0;

    protected:
        template<typename T, typename... Us> friend T& get(variant<Us...> &v);
        template<typename T, typename... Us> friend const T& get(const variant<Us...> &v);
        template<typename... Us> friend bool operator==(const variant<Us...> &lhs,
                                                        const variant<Us...> &rhs);
        Memory memory;

    public:
        // Constructors
        variant() noexcept;
        variant(const variant& other);
        variant(variant&& other) noexcept;
        template<typename T> explicit variant(const T& t);
        // are_different is a SFINAE trick to avoid variant(T &&t) with T=variant
        // for some reason, this version is called instead of variant(variant&& o) when
        // variant is used in STL containers (examples: vector assignment)
        template<typename T> explicit variant(T&& t, typename detail::are_different<variant, T, int>::type = 0);
        // template<class T, class... Args> explicit variant(Args&&... args);
        // FIXME: other constructors

        // Destructor
        ~variant();

        // Assignment
        variant& operator=(const variant& rhs);
        variant& operator=(variant &&rhs) noexcept;

        // SFINAE trick to avoid operator=(T&&) with T=variant<>, see comment above
        template<class T>
        typename detail::are_different<variant, T, variant&>
        ::type operator=(T&& t) noexcept;

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
    T& get(util::variant<Types...> &v);

    template<typename T, typename... Types>
    const T& get(const util::variant<Types...> &v);

    template<typename T, typename... Types>
    bool holds_alternative(const util::variant<Types...> &v) noexcept;

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
    template<class T>
    variant<Ts...>::variant(const T& t)
        : m_index(util::type_list_index<T, Ts...>::value)
    {
        (vctrs()[m_index])(memory, &t);
    }

    template<typename... Ts>
    template<class T>
    variant<Ts...>::variant(T&& t, typename detail::are_different<variant, T, int>::type)
        : m_index(util::type_list_index<typename std::remove_reference<T>::type, Ts...>::value)
    {
        (mctrs()[m_index])(memory, &t);
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
    template<class T> typename detail::are_different<variant<Ts...>, T, variant<Ts...>&>
    ::type variant<Ts...>::operator=(T&& t) noexcept
    {
        // FIXME: No version with implicit type conversion available!
        static const constexpr std::size_t t_index =
            util::type_list_index<T, Ts...>::value;

        if (t_index == m_index)
        {
            util::get<T>(*this) = std::move(t);
            return *this;
        }
        else return (*this = variant(std::move(t)));
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
    T& get(util::variant<Types...> &v)
    {
        const constexpr std::size_t t_index =
            util::type_list_index<T, Types...>::value;

        if (v.index() == t_index)
            return reinterpret_cast<T&>(v.memory);
        else
            throw_error(bad_variant_access());
    }

    template<typename T, typename... Types>
    const T& get(const util::variant<Types...> &v)
    {
        const constexpr std::size_t t_index =
            util::type_list_index<T, Types...>::value;

        if (v.index() == t_index)
            return reinterpret_cast<const T&>(v.memory);
        else
            throw_error(bad_variant_access());
    }

    template<typename T, typename... Types>
    bool holds_alternative(const util::variant<Types...> &v) noexcept
    {
        return v.index() == util::variant<Types...>::template index_of<T>();
    }

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

    template<typename... Us> bool operator!=(const variant<Us...> &lhs,
                                             const variant<Us...> &rhs)
    {
        return !(lhs == rhs);
    }
} // namespace cv
} // namespace util

#endif // OPENCV_GAPI_UTIL_VARIANT_HPP
